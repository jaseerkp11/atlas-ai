"""
Live watch loop — continuous MT5 feed, signals ONLY on a newly closed M5.

Rules (strict):
1. On start: DO NOT analyze/print trade cards.
2. Wait until the next wall-clock M5 close (:00/:05/:10…).
3. Re-seed closed-bar ids from live MT5.
4. After that, emit a card only when last CLOSED M5 time changes (iloc[-2]).
5. Structure from H1+M15 closed bars; M5 closed bars for trigger.
"""

from __future__ import annotations

import csv
import logging
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

import pandas as pd

from atlas.analysis.levels import build_sr_map
from atlas.analysis.setups import detect_setup
from atlas.config import ROOT, load_settings
from atlas.execution.engine import ExecutionEngine
from atlas.execution.mt5_client import MT5Client
from atlas.journal.trade_journal import TradeJournal
from atlas.models import TradeDecision
from atlas.risk.position import ConcurrentTradeGuard, DailyRiskState
from atlas.scoring.engine import evaluate_setup

logger = logging.getLogger(__name__)

SIGNAL_FIELDS = [
    "time_utc",
    "symbol",
    "decision",
    "direction",
    "score",
    "rr",
    "entry",
    "stop",
    "target",
    "nearest_support",
    "nearest_resistance",
    "bid",
    "ask",
    "reasoning",
    "watch_note",
]


def _bar_key(ts) -> str:
    return str(pd.Timestamp(ts))


def _next_m5_boundary_utc(now: datetime | None = None) -> datetime:
    """Next M5 close time in UTC (exclusive of 'now' if exactly on a boundary)."""
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    # If exactly on a 5-min mark with 0 seconds, treat as already closed → next one
    floored = now.replace(second=0, microsecond=0)
    minute = (floored.minute // 5) * 5
    boundary = floored.replace(minute=minute)
    if now <= boundary:
        return boundary
    return boundary + timedelta(minutes=5)


def drop_forming_candle(df: pd.DataFrame | None) -> pd.DataFrame | None:
    """Remove the currently forming (incomplete) candle."""
    if df is None or len(df) < 3:
        return df
    return df.iloc[:-1].reset_index(drop=True)


def last_closed_m5_key(m5: pd.DataFrame) -> str | None:
    """
    Key of the last fully closed M5 bar.
    MT5 copy_rates[-1] is the forming candle; [-2] is the last closed one.
    """
    if m5 is None or len(m5) < 3:
        return None
    return _bar_key(m5["time"].iloc[-2])


class SignalJournal:
    def __init__(self, path=None) -> None:
        settings = load_settings()
        journal_dir = ROOT / settings.paths["journal_dir"]
        journal_dir.mkdir(parents=True, exist_ok=True)
        self.path = Path(path) if path else journal_dir / "signals.csv"
        if not self.path.exists() or self.path.stat().st_size == 0:
            with open(self.path, "w", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=SIGNAL_FIELDS).writeheader()

    def log(self, row: dict) -> None:
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=SIGNAL_FIELDS).writerow(row)


def format_watch_card(
    decision: TradeDecision,
    sr,
    bid: float,
    ask: float,
    execute_armed: bool,
    closed_m5: str,
) -> str:
    f = decision.features
    action = "WATCH — H1/M15 zones + M5 trigger (closed bars only)"
    if decision.allowed:
        action = "SETUP QUALIFIED"
        if execute_armed:
            action = "EXECUTE (gates passed)"

    lines = [
        "",
        "╔" + "═" * 62 + "╗",
        f"║  LIVE  {decision.symbol:8}  {decision.direction.value:6}  "
        f"score={decision.score.total}/100  RR=1:{f.reward_risk:.2f}",
        f"║  {action}",
        f"║  Closed M5    : {closed_m5}",
        "╠" + "═" * 62 + "╣",
        f"║  Live bid/ask : {bid:.5f} / {ask:.5f}",
        f"║  Entry        : {f.entry:.5f}",
        f"║  Stop         : {f.stop:.5f}",
        f"║  Target       : {f.target:.5f}",
    ]
    if sr.nearest_support:
        lines.append(
            f"║  Support      : {sr.nearest_support.price:.5f}  ({sr.nearest_support.strength})"
        )
    if sr.nearest_resistance:
        lines.append(
            f"║  Resistance   : {sr.nearest_resistance.price:.5f}  ({sr.nearest_resistance.strength})"
        )
    lines.append("║  Near S/R:")
    shown = sr.resistances[:3] + sr.supports[:3]
    if not shown:
        lines.append("║    (none near price)")
    for lvl in shown:
        tag = "RES" if lvl.kind == "resistance" else "SUP"
        lines.append(
            f"║    {tag} {lvl.price:.5f}  [{'+'.join(lvl.timeframes)}] {lvl.strength}"
        )
    lines.append("║  Why:")
    for factor in decision.score.factors:
        mark = "✓" if factor.passed else "✗"
        lines.append(f"║    {mark} {factor.name}: {factor.reason}")
    if decision.gate_failures:
        lines.append("║  Gates:")
        for g in decision.gate_failures:
            lines.append(f"║    • {g}")
    lines.append("╚" + "═" * 62 + "╝")
    return "\n".join(lines)


def should_emit_on_closed_m5(prev_key: str | None, new_key: str | None) -> bool:
    """
    Emit only when we already know a previous closed bar AND it changed.
    Never emit on first sight (prev is None) — that was the startup flood bug.
    """
    if new_key is None:
        return False
    if prev_key is None:
        return False
    return new_key != prev_key


class WatchLoop:
    def __init__(
        self,
        broker: MT5Client | None = None,
        execute: bool = False,
        on_log: Callable[[str], None] | None = None,
        skip_startup_wait: bool = False,
    ) -> None:
        self.broker = broker or MT5Client()
        self.execute = execute
        self.on_log = on_log or (lambda m: print(m, flush=True))
        self.signals = SignalJournal()
        self._last_closed_m5: dict[str, str] = {}
        self._armed = False  # False until next M5 boundary after start
        self._running = False
        self.skip_startup_wait = skip_startup_wait  # tests only
        self.engine: ExecutionEngine | None = None
        if execute:
            self.engine = ExecutionEngine(
                broker=self.broker,
                journal=TradeJournal(),
                daily_risk=DailyRiskState(),
                concurrent=ConcurrentTradeGuard(),
                on_log=self.on_log,
            )

    def start(self, max_cycles: int | None = None) -> None:
        settings = load_settings(reload=True)
        if not self.broker.connect():
            self.on_log("FATAL: broker connect failed")
            return

        interval = int(settings.analysis.get("watch_interval_seconds", 10))
        self.on_log("=" * 64)
        self.on_log("ATLAS LIVE WATCH v4")
        self.on_log("  NO analysis on startup — waits for the NEXT real M5 close")
        self.on_log("  Structure: H1 + M15 closed candles (FVG, OB, BOS, S/R)")
        self.on_log("  Trigger  : only when a new M5 candle has CLOSED")
        self.on_log(
            f"  Gates    : score>={settings.gates.min_score}  "
            f"RR=1:{settings.gates.min_reward_risk:g}  execute={self.execute}"
        )
        self.on_log(
            f"  Feed     : {'MT5_LIVE_MARKET' if self.broker.using_live_market_data else 'SYNTHETIC'}"
        )
        self.on_log("=" * 64)

        self._running = True
        self._armed = False

        # 1) Seed current closed-bar ids — never emit
        self._seed_all_symbols()
        self.on_log("Startup seed done. No signal cards printed.")

        # 2) Wait until next M5 wall-clock close so mid-candle start cannot fire
        if not self.skip_startup_wait:
            self._wait_for_next_m5_boundary(interval)
            # Re-seed after boundary so the just-closed bar is baseline, not a signal
            self._seed_all_symbols()
            self.on_log(
                "Armed. Next signal cards will appear on the FOLLOWING M5 close only."
            )
        else:
            self.on_log("Test mode: startup wait skipped.")

        self._armed = True
        cycles = 0
        try:
            while self._running:
                self._poll_and_maybe_emit()
                cycles += 1
                if max_cycles is not None and cycles >= max_cycles:
                    break
                time.sleep(interval)
        except KeyboardInterrupt:
            self.on_log("Watch stopped by user")
        finally:
            self._running = False
            self.broker.disconnect()
            self.on_log(f"Signals log: {self.signals.path}")

    def stop(self) -> None:
        self._running = False

    def _live_heartbeat(self) -> None:
        settings = load_settings()
        now = datetime.now(timezone.utc)
        nxt = _next_m5_boundary_utc(now)
        bits = []
        for sym in ("XAUUSD", "EURUSD", "GBPUSD"):
            if sym not in settings.symbols:
                continue
            snap = self.broker.live_tick_snapshot(sym)
            if snap:
                bits.append(f"{sym} {snap['bid']:.5f}/{snap['ask']:.5f}")
            else:
                bid, ask = self.broker.current_price(sym)
                bits.append(f"{sym} {bid:.5f}/{ask:.5f}")
        state = "ARMED" if self._armed else "WAITING_FIRST_CLOSE"
        self.on_log(
            f"[{now.strftime('%H:%M:%S')} UTC] {state} LIVE {' | '.join(bits)} "
            f"| next M5 ~{nxt.strftime('%H:%M:%S')} UTC"
        )

    def _wait_for_next_m5_boundary(self, interval: int) -> None:
        target = _next_m5_boundary_utc()
        self.on_log(
            f"Waiting until next M5 close at {target.strftime('%H:%M:%S')} UTC "
            f"before any analysis (you started mid-candle — this is correct)…"
        )
        while self._running:
            now = datetime.now(timezone.utc)
            if now >= target:
                self.on_log(f"M5 boundary reached at {now.strftime('%H:%M:%S')} UTC.")
                # Small buffer so MT5 has formed the new bar
                time.sleep(2)
                return
            self._live_heartbeat()
            remain = (target - now).total_seconds()
            time.sleep(min(interval, max(1, remain)))

    def _seed_all_symbols(self) -> None:
        settings = load_settings()
        for symbol in settings.symbols:
            try:
                m5 = self.broker.copy_rates(symbol, "M5", 100)
                key = last_closed_m5_key(m5)
                if key:
                    self._last_closed_m5[symbol] = key
            except Exception as exc:
                logger.exception(symbol)
                self.on_log(f"Seed error {symbol}: {exc}")
        self.on_log(
            f"Seeded closed-M5 ids for {len(self._last_closed_m5)} symbols "
            f"(baseline only — not a signal)."
        )

    def _poll_and_maybe_emit(self) -> None:
        settings = load_settings()
        self._live_heartbeat()
        if not self._armed:
            self.on_log("  → not armed yet (still in startup wait)")
            return

        any_new = False
        for symbol in settings.symbols:
            try:
                if self._process_symbol(symbol):
                    any_new = True
            except Exception as exc:
                logger.exception(symbol)
                self.on_log(f"Error {symbol}: {exc}")
        if not any_new:
            self.on_log("  → no NEW closed M5 since last poll (normal between closes)")

    def _fetch_frames(self, symbol: str) -> dict[str, pd.DataFrame]:
        raw = {
            "M5": self.broker.copy_rates(symbol, "M5", 400),
            "M15": self.broker.copy_rates(symbol, "M15", 300),
            "H1": self.broker.copy_rates(symbol, "H1", 200),
            "H4": self.broker.copy_rates(symbol, "H4", 150),
        }
        # Pass full frames to detect_setup (it drops forming bars itself),
        # but S/R map should also ignore forming candles.
        return raw

    def _process_symbol(self, symbol: str) -> bool:
        frames = self._fetch_frames(symbol)
        m5 = frames.get("M5")
        key = last_closed_m5_key(m5)
        prev = self._last_closed_m5.get(symbol)

        # Hard rule: never emit on first observation
        if not should_emit_on_closed_m5(prev, key):
            if key is not None and prev is None:
                self._last_closed_m5[symbol] = key
            return False

        assert key is not None and prev is not None
        self._last_closed_m5[symbol] = key
        self.on_log(f"\n>>> {symbol}: NEW closed M5 @ {key}")

        bid, ask = self.broker.current_price(symbol)
        mid = (bid + ask) / 2.0
        # Build S/R from closed candles only
        closed_frames = {
            tf: drop_forming_candle(df) if df is not None else None
            for tf, df in frames.items()
        }
        # Keep enough rows
        closed_frames = {
            tf: df for tf, df in closed_frames.items() if df is not None and len(df) > 20
        }
        sr = build_sr_map(symbol, closed_frames, mid=mid)

        features = detect_setup(symbol, frames)
        if features is None:
            self.on_log(f"{symbol}: insufficient data")
            return True

        decision = evaluate_setup(features)
        self.on_log(
            format_watch_card(decision, sr, bid, ask, bool(self.execute), closed_m5=key)
        )

        self.signals.log(
            {
                "time_utc": datetime.now(timezone.utc).isoformat(),
                "symbol": symbol,
                "decision": "TRADE" if decision.allowed else "NO_TRADE",
                "direction": decision.direction.value,
                "score": decision.score.total,
                "rr": round(features.reward_risk, 3),
                "entry": features.entry,
                "stop": features.stop,
                "target": features.target,
                "nearest_support": sr.nearest_support.price if sr.nearest_support else "",
                "nearest_resistance": sr.nearest_resistance.price if sr.nearest_resistance else "",
                "bid": bid,
                "ask": ask,
                "reasoning": " | ".join(decision.score.plain_language),
                "watch_note": f"closed_m5={key}",
            }
        )

        if self.execute and decision.allowed and self.engine is not None:
            self.engine.try_execute(decision)
        return True
