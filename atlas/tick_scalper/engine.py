"""
M1 price-action burst engine.

Watches 1-minute structure every second. When direction is clear:
  - BURST open up to max_open_positions (5) immediately
  - Each position closes independently on profit / SL (every tick)
  - When slots free and M1 still agrees → fill again
"""

from __future__ import annotations

import signal
import time
from dataclasses import dataclass
from datetime import datetime, timezone

from atlas.tick_scalper.config import TickScalperConfig, load_tick_config
from atlas.tick_scalper.execution import TickExecutor
from atlas.tick_scalper.logger import TickLogger
from atlas.tick_scalper.mt5_feed import MT5TickFeed, Tick
from atlas.tick_scalper.risk import TickRiskManager
from atlas.tick_scalper.strategy import OpenState, Side, TickStrategy


@dataclass
class EngineStats:
    ticks: int = 0
    signals: int = 0
    entries: int = 0
    exits: int = 0
    blocked: int = 0
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

    def start(self, max_ticks: int | None = None) -> EngineStats:
        print("=" * 64)
        print("  ATLAS XAUUSD M1 PRICE-ACTION ×5 BURST")
        print(f"  Mode     : {self.cfg.mode}")
        print(f"  Symbol   : {self.cfg.symbol}")
        print(f"  Chart    : M1 structure only (no indicators)")
        print(f"  Burst    : open {self.cfg.max_open_positions} trades on candle direction")
        print(f"  Ladder   : {list(self.cfg.profit_ladder_points)} pts")
        print(f"  Base SL  : {self.cfg.stop_loss_points} pts")
        print(f"  Lots     : FIXED {self.cfg.fixed_lots} × {self.cfg.max_open_positions}")
        print("=" * 64)
        print("Watch M1 → OPEN 5 → each closes on profit → REPEAT.")
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

        # Exits first — free slots for next burst
        self._positions = self.exec.current_positions()
        self._manage_all(tick, point)
        self._positions = self.exec.current_positions()

        if self.risk.state.halted:
            return

        open_n = len(self._positions)
        if open_n >= self.cfg.max_open_positions:
            return

        signal = self.strategy.evaluate_entry(tick, point)
        if signal is None:
            return

        # Don't re-burst the same M1 bar if already filled once and still full cycle
        # Allow refill when slots free while same bar bias holds
        ok, why = self.risk.allows_new_trade(tick.time_msc)
        if not ok:
            self.stats.blocked += 1
            self.log.event(f"RISK_BLOCK {why}")
            return

        # Opposite side open? Wait until flat
        for p in self._positions:
            if p.side != signal.side:
                self.log.event("ENTRY_BLOCK opposite side still open")
                return

        self.stats.signals += 1
        to_open = self.cfg.max_open_positions - open_n
        if not self.cfg.burst_fill:
            to_open = min(1, to_open)

        self.log.event(
            f"M1 SIGNAL {signal.side.value} → burst {to_open} "
            f"(open={open_n}/{self.cfg.max_open_positions}) | {signal.reason}"
        )

        opened = 0
        for i in range(to_open):
            slot = open_n + i
            if not self._open_one(signal, tick, point, slot):
                break
            opened += 1

        if opened:
            self.strategy.mark_burst_bar(signal.m1_bar_time)
            print(
                f"[{_ts()}] BURST +{opened} {signal.side.value} "
                f"open={len(self.exec.current_positions())}/{self.cfg.max_open_positions}"
            )

    def _open_one(self, signal, tick: Tick, point: float, slot: int) -> bool:
        volume, size_msg = self.risk.position_size(self.cfg.stop_loss_points)
        if volume <= 0:
            self.log.event(f"SIZE_BLOCK {size_msg}")
            return False

        # Re-check margin/count each fill
        self._positions = self.exec.current_positions()
        if len(self._positions) >= self.cfg.max_open_positions:
            return False

        target = self.cfg.profit_target_for_slot(slot)
        spread_pts = tick.spread_points(point)
        entry = signal.ask if signal.side == Side.BUY else signal.bid
        # Fresh quote for each order
        if self.feed._use_mt5:
            fresh = self.feed.read_tick()
            if fresh is not None:
                tick = fresh
                entry = tick.ask if signal.side == Side.BUY else tick.bid
                signal.bid, signal.ask = tick.bid, tick.ask
                spread_pts = tick.spread_points(point)

        sl, tp = self.strategy.levels_for(signal.side, entry, point, spread_points=spread_pts)
        tp_dist = max(self.cfg.take_profit_points, target + 5.0) * point
        if signal.side == Side.BUY:
            tp = entry + tp_dist
        else:
            tp = entry - tp_dist

        result = self.exec.open_market(
            signal, volume, sl, tp, profit_target_points=target
        )
        if not result.ok:
            self.log.event(f"ENTRY_FAIL slot={slot} vol={volume} {result.comment}")
            return False

        self.stats.entries += 1
        print(
            f"[{_ts()}] ENTRY#{self.stats.entries} {signal.side.value} "
            f"@{result.price:.3f} vol={result.volume} target=+{target}pts"
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
                f"[{_ts()}] EXIT {reason} ticket={pos.ticket} "
                f"@{price:.3f} pnl≈${pnl_money:.2f} "
                f"left={len(self.exec.current_positions())}"
            )

    def _shutdown(self) -> None:
        elapsed = max(time.perf_counter() - self.stats.start_ts, 1e-9)
        tps = self.stats.ticks / elapsed
        print("\n--- Tick Scalper stopped ---")
        print(
            f"ticks={self.stats.ticks} signals={self.stats.signals} "
            f"entries={self.stats.entries} exits={self.stats.exits} "
            f"blocked={self.stats.blocked} ~{tps:.0f} ticks/s"
        )
        try:
            self.log.close()
        except Exception:
            pass
        self.feed.disconnect()


def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%H:%M:%S")


def run_tick_scalper(max_ticks: int | None = None) -> EngineStats:
    cfg = load_tick_config(reload=True)
    return TickEngine(cfg).start(max_ticks=max_ticks)
