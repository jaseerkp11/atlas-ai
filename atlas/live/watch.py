"""
Watch / alert mode — scan every 15 minutes, print chart-ready signals.

Does NOT invent a win rate. High score + S/R confluence filters for quality;
you review on the chart. Optional auto-execute only when ATLAS_MODE=LIVE and
--execute is passed.
"""

from __future__ import annotations

import csv
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from atlas.analysis.levels import SRMap, build_sr_map
from atlas.analysis.setups import detect_setup
from atlas.config import ROOT, load_settings
from atlas.execution.engine import ExecutionEngine
from atlas.execution.mt5_client import MT5Client
from atlas.journal.trade_journal import TradeJournal
from atlas.models import Direction, TradeDecision
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


class SignalJournal:
    def __init__(self, path: Path | str | None = None) -> None:
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
    sr: SRMap,
    bid: float,
    ask: float,
    execute_armed: bool,
) -> str:
    f = decision.features
    action = "WATCH ON CHART"
    if decision.allowed:
        action = "HIGH-QUALITY SETUP — review on chart"
        if execute_armed:
            action = "EXECUTE CANDIDATE (gates passed)"

    lines = [
        "",
        "╔" + "═" * 62 + "╗",
        f"║  {decision.symbol:8}  {decision.direction.value:6}  score={decision.score.total}/100  RR={f.reward_risk:.2f}",
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
    lines.append("║  Draw on TradingView / MT5:")
    for lvl in (sr.resistances[:3] + sr.supports[:3]):
        tag = "RES" if lvl.kind == "resistance" else "SUP"
        lines.append(f"║    {tag} {lvl.price:.5f}  [{'+'.join(lvl.timeframes)}] {lvl.strength}")
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


class WatchLoop:
    """
    Re-scan on each new M15 candle close (≈ every 15 minutes).
    Default: alerts only for you to watch on the chart.
    """

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
        self._last_m15: dict[str, object] = {}
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
        settings = load_settings()
        if not self.broker.connect():
            self.on_log("FATAL: broker connect failed")
            return

        interval = int(settings.analysis.get("watch_interval_seconds", 60))
        self.on_log(
            f"WATCH mode started | scan on each NEW M15 close | poll every {interval}s | "
            f"min_score={settings.gates.min_score} | min_rr=1:{settings.gates.min_reward_risk:g} | "
            f"execute={self.execute} | data={'LIVE_MT5' if self.broker.using_live_market_data else 'SYNTHETIC'}"
        )
        self.on_log(
            "No win-rate guarantee. High score + S/R confluence = candidates for YOUR chart review."
        )
        if self.execute and settings.is_live:
            self.on_log("⚠ EXECUTE armed + LIVE mode — qualifying setups will send demo/real orders.")
        elif self.execute:
            self.on_log("Execute armed but mode is not LIVE — orders stay simulated.")
        else:
            self.on_log("Alerts only — no orders. Use: python main.py watch --execute  to arm orders.")

        self._running = True
        cycles = 0
        try:
            while self._running:
                self._cycle()
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

    def _cycle(self) -> None:
        settings = load_settings()
        now = datetime.now(timezone.utc)
        self.on_log(f"\n--- watch poll {now.isoformat()} ---")

        for symbol in settings.symbols:
            try:
                self._process_symbol(symbol)
            except Exception as exc:
                logger.exception(symbol)
                self.on_log(f"Error {symbol}: {exc}")

    def _process_symbol(self, symbol: str) -> None:
        settings = load_settings()
        frames = {
            "M5": self.broker.copy_rates(symbol, "M5", 400),
            "M15": self.broker.copy_rates(symbol, "M15", 300),
            "H1": self.broker.copy_rates(symbol, "H1", 200),
            "H4": self.broker.copy_rates(symbol, "H4", 150),
        }
        m15 = frames["M15"]
        if m15 is None or len(m15) < 40:
            return

        last_t = m15["time"].iloc[-1]
        prev = self._last_m15.get(symbol)
        # Only emit a full signal card on a NEW M15 candle close
        if prev is not None and last_t <= prev:
            return
        self._last_m15[symbol] = last_t

        bid, ask = self.broker.current_price(symbol)
        mid = (bid + ask) / 2.0
        sr = build_sr_map(symbol, frames, mid=mid)

        features = detect_setup(symbol, frames)
        if features is None:
            self.on_log(f"{symbol}: insufficient data for setup")
            for line in sr.chart_lines():
                self.on_log(line)
            return

        decision = evaluate_setup(features)
        card = format_watch_card(decision, sr, bid, ask, execute_armed=bool(self.execute))
        self.on_log(card)

        watch_note = (
            f"Draw SUP {sr.nearest_support.price:.5f} / "
            f"RES {sr.nearest_resistance.price:.5f} on chart and wait for reaction"
            if sr.nearest_support and sr.nearest_resistance
            else "Mark printed S/R on chart"
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

        # Optional execution path (same gates — no separate loose logic)
        if self.execute and decision.allowed and self.engine is not None:
            self.engine.try_execute(decision)
        elif decision.allowed:
            self.on_log(
                f"→ {symbol}: qualifies (score {decision.score.total}). "
                f"Open chart, mark S/R, confirm structure yourself before trading."
            )
