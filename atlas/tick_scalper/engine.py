"""
Low-latency tick processing loop for XAUUSD.

Design goals:
  - Minimal work per tick (no heavy DataFrame rebuilds)
  - Immediate market entry when edge filters all pass
  - Immediate exit when TP / SL / protect rule hits
  - Resume scanning after flat — never force trades on a timer
  - No Martingale / Grid / averaging
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
from atlas.tick_scalper.strategy import OpenState, TickStrategy


@dataclass
class EngineStats:
    ticks: int = 0
    signals: int = 0
    entries: int = 0
    exits: int = 0
    blocked: int = 0
    start_ts: float = 0.0


class TickEngine:
    """High-speed every-tick scalper orchestrator."""

    def __init__(self, cfg: TickScalperConfig | None = None) -> None:
        self.cfg = cfg or load_tick_config()
        self.log = TickLogger()
        self.feed = MT5TickFeed(self.cfg)
        self.strategy = TickStrategy(self.cfg)
        self.risk = TickRiskManager(self.feed, self.cfg)
        self.exec = TickExecutor(self.feed, self.log, self.cfg)
        self.stats = EngineStats()
        self._running = False
        self._position: Optional[OpenState] = None
        self._tick_counter = 0

    def start(self, max_ticks: int | None = None) -> EngineStats:
        """
        Connect to MT5 and run the tick loop until Ctrl+C or max_ticks.
        """
        print("=" * 64)
        print("  ATLAS XAUUSD TICK SCALPER  (every tick)")
        print(f"  Mode     : {self.cfg.mode}")
        print(f"  Symbol   : {self.cfg.symbol}")
        print(f"  Max sprd : {self.cfg.max_spread_points} pts")
        print(f"  TP / SL  : {self.cfg.take_profit_points} / {self.cfg.stop_loss_points} pts")
        if self.cfg.use_fixed_lots:
            print(f"  Lots     : FIXED {self.cfg.fixed_lots} (cap {self.cfg.max_lots})")
        else:
            print(f"  Risk     : {self.cfg.risk_percent}% / trade (cap {self.cfg.max_lots} lots)")
        print(f"  Daily DD : {self.cfg.daily_loss_limit_percent}%")
        print(f"  Max open : {self.cfg.max_open_positions}")
        print("=" * 64)
        print("No Martingale / Grid / averaging. Edge-only entries.")
        print("Ctrl+C to stop.\n")

        if not self.feed.connect():
            raise RuntimeError("MT5 connection failed — open MetaTrader 5 and check .env")

        equity = self.feed.account_equity()
        self.risk.reset_day_if_needed(equity)
        if self.risk.state.starting_equity <= 0:
            self.risk.state.starting_equity = equity

        point = self.feed.symbol_point()
        self.stats.start_ts = time.perf_counter()
        self._running = True
        self._position = self.exec.current_position()

        def _stop(*_args) -> None:
            self._running = False

        prev_sig = signal.signal(signal.SIGINT, _stop)
        try:
            try:
                signal.signal(signal.SIGTERM, _stop)
            except (ValueError, OSError):
                pass

            sleep_s = max(0.0, self.cfg.poll_sleep_ms / 1000.0)
            for tick in self.feed.stream():
                if not self._running:
                    break
                self._on_tick(tick, point)
                if max_ticks is not None and self.stats.ticks >= max_ticks:
                    break
                if sleep_s > 0:
                    # stream() already sleeps; keep CPU light when quiet
                    pass
        finally:
            signal.signal(signal.SIGINT, prev_sig)
            self._shutdown()

        return self.stats

    def _on_tick(self, tick: Tick, point: float) -> None:
        t0 = time.perf_counter()
        self.stats.ticks += 1
        self._tick_counter += 1

        # Strategy state update (rolling mids + VWAP) — O(1)
        self.strategy.update(tick, point)

        # Optional tick logging (sampled by default for disk/CPU)
        if self.cfg.log_every_tick or (self._tick_counter % self.cfg.tick_summary_every_n == 0):
            self.log.tick(
                bid=tick.bid,
                ask=tick.ask,
                spread_points=tick.spread_points(point),
                last=tick.last,
                volume=tick.volume,
            )

        # Sync open position (LIVE may be closed by broker SL/TP)
        self._position = self.exec.current_position()

        # --- Priority path: manage open trade every tick ---
        if self._position is not None:
            self._manage_open(tick, point)
            return

        # --- Flat: evaluate entry (never force on time) ---
        if self.risk.state.halted:
            return

        signal = self.strategy.evaluate_entry(tick, point)
        if signal is None:
            return

        self.stats.signals += 1
        self.log.event(
            f"SIGNAL {signal.side.value} {signal.reason} "
            f"bid={signal.bid:.3f} ask={signal.ask:.3f}"
        )

        ok, why = self.risk.allows_new_trade(tick.time_msc)
        if not ok:
            self.stats.blocked += 1
            self.log.event(f"RISK_BLOCK {why}")
            return

        volume, size_msg = self.risk.position_size(self.cfg.stop_loss_points)
        if volume <= 0:
            self.stats.blocked += 1
            self.log.event(f"SIZE_BLOCK {size_msg}")
            return

        entry = signal.ask if signal.side.value == "BUY" else signal.bid
        sl, tp = self.strategy.levels_for(signal.side, entry, point)
        result = self.exec.open_market(signal, volume, sl, tp)
        if not result.ok:
            self.log.event(f"ENTRY_FAIL vol={volume} {result.comment}")
            return

        self._position = self.exec.current_position()
        self.stats.entries += 1
        latency_ms = (time.perf_counter() - t0) * 1000.0
        print(
            f"[{_ts()}] ENTRY {signal.side.value} @{result.price:.3f} "
            f"vol={volume} SL={sl:.3f} TP={tp:.3f} | {size_msg}"
        )
        if latency_ms > 8.0:
            self.log.event(f"SLOW_TICK process={latency_ms:.2f}ms")

    def _manage_open(self, tick: Tick, point: float) -> None:
        pos = self._position
        if pos is None:
            return

        reason = self.strategy.evaluate_exit(tick, pos, point)
        if reason is None:
            return

        result = self.exec.close_market(pos, tick, reason)
        if not result.ok:
            self.log.event(f"EXIT_FAIL ticket={pos.ticket} {result.comment}")
            return

        # Approximate PnL for daily risk tracker
        price = result.price
        if pos.side.value == "BUY":
            pnl_pts = (price - pos.entry) / point
        else:
            pnl_pts = (pos.entry - price) / point
        # ~$1 per point per lot when point=0.01 on XAUUSD (broker-dependent)
        pnl_money = pnl_pts * pos.volume
        equity = self.feed.account_equity()
        self.risk.record_pnl(pnl_money, equity)
        self.risk.mark_close(tick.time_msc)
        self.stats.exits += 1
        self._position = None
        print(
            f"[{_ts()}] EXIT {reason} ticket={pos.ticket} "
            f"@{price:.3f} approx_pnl=${pnl_money:.2f} → scanning…"
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
        if self._position is not None and self.cfg.is_paper:
            print(f"NOTE: PAPER position still open ticket={self._position.ticket}")
        try:
            self.log.close()
        except Exception:
            pass
        self.feed.disconnect()


def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%H:%M:%S")


def run_tick_scalper(max_ticks: int | None = None) -> EngineStats:
    """CLI entry: load config, connect MT5, start tick loop."""
    cfg = load_tick_config(reload=True)
    engine = TickEngine(cfg)
    return engine.start(max_ticks=max_ticks)
