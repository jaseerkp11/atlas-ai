"""
M1 price-action alternating batch engine.

Flow:
  1) Wait flat
  2) M1 price action gives direction (first batch) OR forced OPPOSITE of last batch
  3) Burst-open exactly 5 trades
  4) Each closes at +1.5 points profit (or SL)
  5) When flat again → flip side → repeat

Math (XAUUSD, point=0.01, 1.0 lot):
  +1.5 points ≈ +$1.50 realized per trade (broker tick value dependent)
  5 winners ≈ +$7.50 gross before spread/commission
"""

from __future__ import annotations

import signal
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from atlas.tick_scalper.config import TickScalperConfig, load_tick_config
from atlas.tick_scalper.execution import TickExecutor
from atlas.tick_scalper.logger import TickLogger
from atlas.tick_scalper.mt5_feed import MT5TickFeed, Tick
from atlas.tick_scalper.risk import TickRiskManager
from atlas.tick_scalper.strategy import OpenState, Side, Signal, TickStrategy


@dataclass
class EngineStats:
    ticks: int = 0
    signals: int = 0
    entries: int = 0
    exits: int = 0
    blocked: int = 0
    batches: int = 0
    start_ts: float = 0.0


class TickEngine:
    def __init__(self, cfg: TickScalperConfig | None = None) -> None:
        self.cfg = cfg or load_tick_config()
        self.log = TickLogger()
        self.feed = MT5TickFeed(self.cfg)
        self.strategy = TickStrategy(self.cfg)
        self.risk = TickRiskManager(self.feed, self.cfg)
        self.exec = TickExecutor(self.feed, self.log, self.cfg)
        self.stats = EngineStats()
        self._running = False
        self._positions: list[OpenState] = []
        self._tick_counter = 0
        self._last_batch_side: Optional[Side] = None
        self._batch_active = False
        self._batch_side: Optional[Side] = None
        self._batch_opened = 0

    def start(self, max_ticks: int | None = None) -> EngineStats:
        pt = self.cfg.instant_profit_points
        # Rough $ math for banner (1.0 lot XAU ≈ $1 per point when point=0.01)
        approx_per = pt * self.cfg.fixed_lots
        approx_batch = approx_per * self.cfg.max_open_positions
        print("=" * 64)
        print("  ATLAS XAUUSD M1 PA — ALT BATCH ×5 @ +1.5pts")
        print(f"  Mode     : {self.cfg.mode}")
        print(f"  Symbol   : {self.cfg.symbol}")
        print(f"  Batch    : {self.cfg.max_open_positions} trades / side")
        print(f"  Target   : +{pt} points each (~${approx_per:.2f}/trade, ~${approx_batch:.2f}/batch)")
        print(f"  Alternate: BUY batch → SELL batch → BUY …")
        print(f"  Lots     : FIXED {self.cfg.fixed_lots}")
        print(f"  SL       : {self.cfg.stop_loss_points} pts")
        print("=" * 64)
        print("Pure M1 price action. Flat → open 5 → +1.5pt closes → opposite 5.")
        print("Ctrl+C to stop.\n")

        if not self.feed.connect():
            raise RuntimeError("MT5 connection failed — open MetaTrader 5 and check .env")

        self.strategy.set_mt5_live(self.feed._use_mt5)
        equity = self.feed.account_equity()
        self.risk.reset_day_if_needed(equity)
        if self.risk.state.starting_equity <= 0:
            self.risk.state.starting_equity = equity

        point = self.feed.symbol_point()
        self.stats.start_ts = time.perf_counter()
        self._running = True
        self._positions = self.exec.current_positions()

        def _stop(*_args) -> None:
            self._running = False

        prev_sig = signal.signal(signal.SIGINT, _stop)
        try:
            try:
                signal.signal(signal.SIGTERM, _stop)
            except (ValueError, OSError):
                pass
            for tick in self.feed.stream():
                if not self._running:
                    break
                self._on_tick(tick, point)
                if max_ticks is not None and self.stats.ticks >= max_ticks:
                    break
        finally:
            signal.signal(signal.SIGINT, prev_sig)
            self._shutdown()
        return self.stats

    def _required_side(self) -> Optional[Side]:
        """Next batch side: opposite of last completed batch (if alternating)."""
        if not self.cfg.alternate_batch_side or self._last_batch_side is None:
            return None  # free — follow M1 PA
        return Side.SELL if self._last_batch_side == Side.BUY else Side.BUY

    def _on_tick(self, tick: Tick, point: float) -> None:
        self.stats.ticks += 1
        self._tick_counter += 1
        self.strategy.update(tick, point)

        if self.cfg.log_every_tick or (self._tick_counter % self.cfg.tick_summary_every_n == 0):
            self.log.tick(
                bid=tick.bid,
                ask=tick.ask,
                spread_points=tick.spread_points(point),
                last=tick.last,
                volume=tick.volume,
            )

        # Manage exits every tick
        self._positions = self.exec.current_positions()
        self._manage_all(tick, point)
        self._positions = self.exec.current_positions()

        # Batch finished when we were active and now flat
        if self._batch_active and not self._positions:
            self._last_batch_side = self._batch_side
            self._batch_active = False
            nxt = self._required_side()
            print(
                f"[{_ts()}] BATCH DONE ({self._batch_side.value if self._batch_side else '?'}) "
                f"→ next must be {nxt.value if nxt else 'M1 PA'} | "
                f"opened={self._batch_opened}"
            )
            self._batch_side = None
            self._batch_opened = 0
            self.log.event(
                f"BATCH_DONE last={self._last_batch_side.value if self._last_batch_side else None} "
                f"next={nxt.value if nxt else 'PA'}"
            )

        if self.risk.state.halted:
            return

        # Only start a new batch when flat (all 5 closed)
        if self.cfg.require_flat_before_next_batch and self._positions:
            return
        if self._batch_active:
            # Still filling burst if somehow incomplete
            if len(self._positions) < self.cfg.max_open_positions and self._batch_side:
                self._fill_batch(tick, point, self._batch_side)
            return

        # Flat — decide next side
        signal = self.strategy.evaluate_entry(tick, point)
        if signal is None:
            return

        required = self._required_side()
        if required is not None and signal.side != required:
            # Wait for price action to agree with the forced alternate side
            return

        ok, why = self.risk.allows_new_trade(tick.time_msc)
        if not ok:
            self.stats.blocked += 1
            self.log.event(f"RISK_BLOCK {why}")
            return

        # Start batch
        self._batch_active = True
        self._batch_side = signal.side
        self._batch_opened = 0
        self.stats.batches += 1
        self.stats.signals += 1
        print(
            f"[{_ts()}] BATCH#{self.stats.batches} START {signal.side.value} ×"
            f"{self.cfg.max_open_positions} @ +{self.cfg.instant_profit_points}pts | {signal.reason}"
        )
        self.log.event(f"BATCH_START {signal.side.value} | {signal.reason}")
        self._fill_batch(tick, point, signal.side)

    def _fill_batch(self, tick: Tick, point: float, side: Side) -> None:
        """Open until 5 positions on this batch side."""
        while True:
            self._positions = self.exec.current_positions()
            if len(self._positions) >= self.cfg.max_open_positions:
                break
            # Build signal for current side
            sig = Signal(
                side=side,
                reason=f"batch {side.value} fill",
                bid=tick.bid,
                ask=tick.ask,
            )
            if not self._open_one(sig, tick, point, len(self._positions)):
                break
            self._batch_opened += 1
            # Refresh tick between orders
            fresh = self.feed.read_tick()
            if fresh is not None:
                tick = fresh

        self._positions = self.exec.current_positions()
        print(
            f"[{_ts()}] BATCH OPENED {len(self._positions)}/{self.cfg.max_open_positions} "
            f"{side.value}"
        )

    def _open_one(self, signal: Signal, tick: Tick, point: float, slot: int) -> bool:
        volume, size_msg = self.risk.position_size(self.cfg.stop_loss_points)
        if volume <= 0:
            self.log.event(f"SIZE_BLOCK {size_msg}")
            return False

        target = self.cfg.instant_profit_points  # always 1.5
        spread_pts = tick.spread_points(point)
        entry = signal.ask if signal.side == Side.BUY else signal.bid
        if self.feed._use_mt5:
            fresh = self.feed.read_tick()
            if fresh is not None:
                tick = fresh
                signal.bid, signal.ask = tick.bid, tick.ask
                entry = tick.ask if signal.side == Side.BUY else tick.bid
                spread_pts = tick.spread_points(point)

        sl, tp = self.strategy.levels_for(signal.side, entry, point, spread_points=spread_pts)
        # Hard TP slightly beyond 1.5 so broker TP backs us up
        tp_dist = max(self.cfg.take_profit_points, target + 1.0) * point
        if signal.side == Side.BUY:
            tp = entry + tp_dist
        else:
            tp = entry - tp_dist

        result = self.exec.open_market(
            signal, volume, sl, tp, profit_target_points=target
        )
        if not result.ok:
            self.log.event(f"ENTRY_FAIL slot={slot} {result.comment}")
            return False

        self.stats.entries += 1
        print(
            f"[{_ts()}] ENTRY#{self.stats.entries} {signal.side.value} "
            f"@{result.price:.3f} target=+{target}pts"
        )
        return True

    def _manage_all(self, tick: Tick, point: float) -> None:
        if not self._positions:
            return
        for pos in list(self._positions):
            reason = self.strategy.evaluate_exit(tick, pos, point)
            if reason is None:
                continue
            result = self.exec.close_market(pos, tick, reason)
            if not result.ok:
                self.log.event(f"EXIT_FAIL ticket={pos.ticket} {result.comment}")
                continue
            price = result.price
            if pos.side == Side.BUY:
                pnl_pts = (price - pos.entry) / point
            else:
                pnl_pts = (pos.entry - price) / point
            pnl_money = pnl_pts * pos.volume
            self.risk.record_pnl(pnl_money, self.feed.account_equity())
            self.risk.mark_close(tick.time_msc)
            self.stats.exits += 1
            print(
                f"[{_ts()}] EXIT {reason} @{price:.3f} "
                f"{pnl_pts:+.1f}pts ≈${pnl_money:.2f} "
                f"left={len(self.exec.current_positions())}"
            )

    def _shutdown(self) -> None:
        elapsed = max(time.perf_counter() - self.stats.start_ts, 1e-9)
        print("\n--- Stopped ---")
        print(
            f"ticks={self.stats.ticks} batches={self.stats.batches} "
            f"entries={self.stats.entries} exits={self.stats.exits} "
            f"~{self.stats.ticks / elapsed:.0f} ticks/s"
        )
        try:
            self.log.close()
        except Exception:
            pass
        self.feed.disconnect()


def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%H:%M:%S")


def run_tick_scalper(max_ticks: int | None = None) -> EngineStats:
    return TickEngine(load_tick_config(reload=True)).start(max_ticks=max_ticks)
