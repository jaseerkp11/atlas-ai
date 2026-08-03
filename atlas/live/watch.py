"""
Watch mode — updates on each NEW M5 candle close (scalp / challenge).

"watch poll" lines = heartbeat only (waiting for next M5 close).
Full signal cards print only when a new M5 bar appears in MT5.
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
    """Normalize bar time so equality checks are reliable across dtypes."""
    return str(pd.Timestamp(ts))


def _next_m5_close_utc(now: datetime | None = None) -> datetime:
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    minute = (now.minute // 5) * 5
    base = now.replace(minute=minute, second=0, microsecond=0)
    nxt = base + timedelta(minutes=5)
    return nxt


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
) -> str:
    f = decision.features
    action = "SCALP WATCH — mark near S/R on chart"
    if decision.allowed:
        action = "SCALP SETUP QUALIFIED"
        if execute_armed:
            action = "SCALP EXECUTE (gates passed)"

    lines = [
        "",
        "╔" + "═" * 62 + "╗",
        f"║  M5 SCALP  {decision.symbol:8}  {decision.direction.value:6}  "
        f"score={decision.score.total}/100  RR=1:{f.reward_risk:.2f}",
        f"║  {action}",
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
    lines.append("║  Near S/R (draw these — stale far levels hidden):")
    shown = (sr.resistances[:3] + sr.supports[:3])
    if not shown:
        lines.append("║    (none within proximity filter)")
    for lvl in shown:
        tag = "RES" if lvl.kind == "resistance" else "SUP"
        lines.append(
            f"║    {tag} {lvl.price:.5f}  [{'+'.join(lvl.timeframes)}] {lvl.strength}"
        )
    lines.append("║  Why (fresh structure only — old filled FVG/OB ignored):")
    for factor in decision.score.factors:
        mark = "✓" if factor.passed else "✗"
        lines.append(f"║    {mark} {factor.name}: {factor.reason}")
    if decision.gate_failures:
        lines.append("║  Gates:")
        for g in decision.gate_failures:
            lines.append(f"║    • {g}")
    lines.append("╚" + "═" * 62 + "╝")
    return "\n".join(lines)


class WatchLoop:
    """Re-scan on each new M5 candle close (~every 5 minutes)."""

    def __init__(
        self,
        broker: MT5Client | None = None,
        execute: bool = False,
        on_log: Callable[[str], None] | None = None,
    ) -> None:
        self.broker = broker or MT5Client()
        self.execute = execute
        self.on_log = on_log or (lambda m: print(m, flush=True))
        self.signals = SignalJournal()
        self._last_bar: dict[str, str] = {}
        self._running = False
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

        signal_tf = "M5"  # forced for scalp watch — do not use M15
        interval = int(settings.analysis.get("watch_interval_seconds", 10))
        self.on_log("=" * 60)
        self.on_log(
            f"M5 SCALP WATCH v2 | cards on each NEW M5 close | poll={interval}s"
        )
        self.on_log(
            f"min_score={settings.gates.min_score} | min_rr=1:{settings.gates.min_reward_risk:g} | "
            f"execute={self.execute} | risk={settings.risk.risk_percent}% | "
            f"data={'LIVE_MT5' if self.broker.using_live_market_data else 'SYNTHETIC'}"
        )
        self.on_log(
            "Heartbeat lines = waiting. Full cards appear only after a new 5m candle closes."
        )
        self.on_log(
            "FVG/OB older than ~2h or far from price are ignored (fixes stale hours-old zones)."
        )
        if self.execute and settings.is_live:
            self.on_log("EXECUTE + LIVE armed.")
        self.on_log("=" * 60)

        self._running = True
        cycles = 0
        try:
            while self._running:
                self._cycle(signal_tf)
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

    def _cycle(self, signal_tf: str) -> None:
        settings = load_settings()
        now = datetime.now(timezone.utc)
        nxt = _next_m5_close_utc(now)
        wait_s = max(0, int((nxt - now).total_seconds()))
        self.on_log(
            f"[{now.strftime('%H:%M:%S')} UTC] heartbeat — next M5 boundary ~{nxt.strftime('%H:%M')} UTC "
            f"(in ~{wait_s}s). Waiting for NEW M5 bar from MT5…"
        )

        any_new = False
        for symbol in settings.symbols:
            try:
                if self._process_symbol(symbol, signal_tf):
                    any_new = True
            except Exception as exc:
                logger.exception(symbol)
                self.on_log(f"Error {symbol}: {exc}")
        if not any_new:
            self.on_log("  (no new M5 closes yet — this is normal between candles)")

    def _process_symbol(self, symbol: str, signal_tf: str) -> bool:
        frames = {
            "M5": self.broker.copy_rates(symbol, "M5", 400),
            "M15": self.broker.copy_rates(symbol, "M15", 300),
            "H1": self.broker.copy_rates(symbol, "H1", 200),
            "H4": self.broker.copy_rates(symbol, "H4", 150),
        }
        bar_df = frames.get(signal_tf)
        if bar_df is None:
            bar_df = frames.get("M5")
        if bar_df is None or len(bar_df) < 40:
            return False

        last_t = bar_df["time"].iloc[-1]
        key = _bar_key(last_t)
        prev = self._last_bar.get(symbol)
        if prev is not None and key == prev:
            return False
        self._last_bar[symbol] = key
        self.on_log(f"\n>>> {symbol}: NEW M5 close @ {key}")

        bid, ask = self.broker.current_price(symbol)
        mid = (bid + ask) / 2.0
        sr = build_sr_map(symbol, frames, mid=mid)

        features = detect_setup(symbol, frames)
        if features is None:
            self.on_log(f"{symbol}: insufficient data for setup")
            return True

        decision = evaluate_setup(features)
        card = format_watch_card(decision, sr, bid, ask, execute_armed=bool(self.execute))
        self.on_log(card)

        watch_note = (
            f"Draw SUP {sr.nearest_support.price:.5f} / "
            f"RES {sr.nearest_resistance.price:.5f}"
            if sr.nearest_support and sr.nearest_resistance
            else "Mark near S/R"
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
                "watch_note": watch_note,
            }
        )

        if self.execute and decision.allowed and self.engine is not None:
            self.engine.try_execute(decision)
        elif decision.allowed:
            self.on_log(f"→ {symbol}: qualifies (score {decision.score.total}).")
        return True
