"""
Low-latency multi-entry tick scalper for XAUUSD.

- Multiple concurrent positions (max_open_positions)
- Each position closes independently on its profit-ladder target
- Pyramid winners only — never add while any open trade is red
- No martingale / grid / averaging into losers
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
        self._last_entry_mid: float | None = None

    def start(self, max_ticks: int | None = None) -> EngineStats:
        print("=" * 64)
        print("  ATLAS XAUUSD M1 + 5s MICRO SCALPER")
        print(f"  Mode     : {self.cfg.mode}")
        print(f"  Symbol   : {self.cfg.symbol}")
        print(f"  Micro TF : {self.cfg.micro_tf_seconds}s synthetic candles")
        print(f"  M1 filter: {self.cfg.use_m1_filter} (lookback={self.cfg.m1_lookback})")
        print(f"  Max open : {self.cfg.max_open_positions} (burst parallel)")
        print(f"  Ladder   : {list(self.cfg.profit_ladder_points)} pts")
        print(f"  Max sprd : {self.cfg.max_spread_points} pts")
        print(f"  Base SL  : {self.cfg.stop_loss_points} pts (spread-aware)")
        print(f"  Lots     : FIXED {self.cfg.fixed_lots} each")
        print("=" * 64)
        print("Entry on M1 + micro-bar agreement. Exit every tick on profit/SL.")
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

    def _unrealized(self, pos: OpenState, tick: Tick, point: float) -> float:
        if pos.side == Side.BUY:
            return (tick.bid - pos.entry) / point
        return (pos.entry - tick.ask) / point

    def _pyramid_ok(self, signal_side: Side, tick: Tick, point: float) -> tuple[bool, str]:
        if not self._positions:
            return True, "flat"
        # Never hedge both directions at once
        for p in self._positions:
            if p.side != signal_side:
                return False, "opposite side open — wait for flat/same side"
        if self.cfg.pyramid_winners_only:
            for p in self._positions:
                if self._unrealized(p, tick, point) < 0:
                    return False, "open loser — no averaging"
        if (
            self._last_entry_mid is not None
            and self.cfg.min_points_between_entries > 0
        ):
            moved = abs(tick.mid - self._last_entry_mid) / point
            if moved < self.cfg.min_points_between_entries:
                return False, f"need {self.cfg.min_points_between_entries}pts between entries"
        return True, "burst OK"

    def _on_tick(self, tick: Tick, point: float) -> None:
        t0 = time.perf_counter()
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

        # 1) Manage exits first — free slots → immediate next entry below
        self._positions = self.exec.current_positions()
        self._manage_all(tick, point)
        self._positions = self.exec.current_positions()

        if self.risk.state.halted:
            return
        if len(self._positions) >= self.cfg.max_open_positions:
            return

        signal = self.strategy.evaluate_entry(tick, point)
        if signal is None:
            return

        self.stats.signals += 1
        ok, why = self.risk.allows_new_trade(tick.time_msc)
        if not ok:
            self.stats.blocked += 1
            self.log.event(f"RISK_BLOCK {why}")
            return

        pyr_ok, pyr_why = self._pyramid_ok(signal.side, tick, point)
        if not pyr_ok:
            self.stats.blocked += 1
            self.log.event(f"ENTRY_BLOCK {pyr_why}")
            return

        volume, size_msg = self.risk.position_size(self.cfg.stop_loss_points)
        if volume <= 0:
            self.stats.blocked += 1
            self.log.event(f"SIZE_BLOCK {size_msg}")
            return

        slot = len(self._positions)
        target = self.cfg.profit_target_for_slot(slot)
        spread_pts = tick.spread_points(point)
        entry = signal.ask if signal.side == Side.BUY else signal.bid
        sl, tp = self.strategy.levels_for(signal.side, entry, point, spread_points=spread_pts)
        tp_dist = max(self.cfg.take_profit_points, target + 5.0) * point
        if signal.side == Side.BUY:
            tp = entry + tp_dist
        else:
            tp = entry - tp_dist

        self.log.event(
            f"SIGNAL {signal.side.value} slot={slot + 1}/{self.cfg.max_open_positions} "
            f"target=+{target}pts SL_dist≈{abs(entry - sl) / point:.0f}pts | {signal.reason}"
        )
        result = self.exec.open_market(
            signal, volume, sl, tp, profit_target_points=target
        )
        if not result.ok:
            self.log.event(f"ENTRY_FAIL vol={volume} {result.comment}")
            return

        self._positions = self.exec.current_positions()
        self._last_entry_mid = tick.mid
        self.stats.entries += 1
        print(
            f"[{_ts()}] ENTRY#{self.stats.entries} {signal.side.value} @{result.price:.3f} "
            f"vol={result.volume} target=+{target}pts open={len(self._positions)}/{self.cfg.max_open_positions}"
        )
        if (time.perf_counter() - t0) * 1000.0 > 50.0:
            self.log.event(f"SLOW_TICK process={(time.perf_counter() - t0) * 1000.0:.2f}ms")

    def _manage_all(self, tick: Tick, point: float) -> None:
        """Close every position that hits its own profit target / SL — independently."""
        if not self._positions:
            return
        # Copy list — we mutate as we close
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
            equity = self.feed.account_equity()
            self.risk.record_pnl(pnl_money, equity)
            self.risk.mark_close(tick.time_msc)
            self.stats.exits += 1
            print(
                f"[{_ts()}] EXIT {reason} ticket={pos.ticket} "
                f"@{price:.3f} pnl≈${pnl_money:.2f} "
                f"left={max(0, len(self.exec.current_positions()))}"
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
        left = self.exec.current_positions()
        if left and self.cfg.is_paper:
            print(f"NOTE: {len(left)} PAPER position(s) still open")
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
