"""
Live watch loop — continuously pulls MT5 ticks/rates.

- Heartbeat every few seconds shows LIVE bid/ask (proves market feed is alive)
- Full signal cards fire when a NEW *closed* M5 candle appears
- Structure (FVG / OB / BOS / S/R) from H1 + M15; entry trigger on M5
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


def _next_m5_close_utc(now: datetime | None = None) -> datetime:
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    minute = (now.minute // 5) * 5
    base = now.replace(minute=minute, second=0, microsecond=0)
    return base + timedelta(minutes=5)


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
    action = "WATCH — H1/M15 zones + M5 trigger"
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
        f"║  Closed M5 bar : {closed_m5}",
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
    lines.append("║  Near S/R (H1/M15/M5 confluence):")
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


class WatchLoop:
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
        self._last_closed_m5: dict[str, str] = {}
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

        interval = int(settings.analysis.get("watch_interval_seconds", 10))
        self.on_log("=" * 64)
        self.on_log("ATLAS LIVE WATCH v3")
        self.on_log("  Data     : continuous MT5 live ticks + OHLC refresh every poll")
        self.on_log("  Structure: H1 + M15 (FVG, OB, BOS, S/R)")
        self.on_log("  Trigger  : each NEW closed M5 candle")
        self.on_log(
            f"  Gates    : score>={settings.gates.min_score}  RR=1:{settings.gates.min_reward_risk:g}  "
            f"execute={self.execute}"
        )
        self.on_log(
            f"  Feed     : {'MT5_LIVE_MARKET' if self.broker.using_live_market_data else 'SYNTHETIC'}"
        )
        self.on_log("=" * 64)
        if not self.broker.using_live_market_data:
            self.on_log("ERROR: not on live MT5 feed — fix connection before trusting signals.")

        self._running = True
        cycles = 0
        # First cycle: seed last-closed times WITHOUT trading (avoid flood), then wait for next close
        self._cycle(seed_only=True)
        try:
            while self._running:
                self._cycle(seed_only=False)
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

    def _cycle(self, seed_only: bool = False) -> None:
        settings = load_settings()
        now = datetime.now(timezone.utc)
        nxt = _next_m5_close_utc(now)

        # Prove live feed every poll
        live_bits = []
        for sym in ("XAUUSD", "EURUSD", "GBPUSD"):
            if sym not in settings.symbols:
                continue
            snap = self.broker.live_tick_snapshot(sym)
            if snap:
                live_bits.append(f"{sym} {snap['bid']:.5f}/{snap['ask']:.5f}")
            else:
                bid, ask = self.broker.current_price(sym)
                live_bits.append(f"{sym} {bid:.5f}/{ask:.5f}")
        self.on_log(
            f"[{now.strftime('%H:%M:%S')} UTC] LIVE {' | '.join(live_bits)} "
            f"| next M5 close ~{nxt.strftime('%H:%M')} UTC"
        )

        any_new = False
        for symbol in settings.symbols:
            try:
                if self._process_symbol(symbol, seed_only=seed_only):
                    any_new = True
            except Exception as exc:
                logger.exception(symbol)
                self.on_log(f"Error {symbol}: {exc}")

        if seed_only:
            self.on_log(
                "Seeded last closed M5 times from live MT5. "
                "Waiting for the NEXT M5 close to print full signal cards…"
            )
        elif not any_new:
            self.on_log("  → no new closed M5 yet (normal between :00/:05/:10/:15…)")

    def _process_symbol(self, symbol: str, seed_only: bool = False) -> bool:
        # Always refresh from MT5 (no stale cache path when live)
        frames = {
            "M5": self.broker.copy_rates(symbol, "M5", 400),
            "M15": self.broker.copy_rates(symbol, "M15", 300),
            "H1": self.broker.copy_rates(symbol, "H1", 200),
            "H4": self.broker.copy_rates(symbol, "H4", 150),
        }
        m5 = frames.get("M5")
        if m5 is None or len(m5) < 40:
            return False

        # CRITICAL: use last CLOSED candle (iloc[-2]), not the forming bar (iloc[-1])
        closed_ts = m5["time"].iloc[-2]
        key = _bar_key(closed_ts)
        prev = self._last_closed_m5.get(symbol)

        if seed_only:
            self._last_closed_m5[symbol] = key
            return False

        if prev is not None and key == prev:
            return False

        self._last_closed_m5[symbol] = key
        self.on_log(f"\n>>> {symbol}: NEW closed M5 @ {key} (matches MT5/TV 5m close)")

        bid, ask = self.broker.current_price(symbol)
        mid = (bid + ask) / 2.0
        sr = build_sr_map(symbol, frames, mid=mid)

        features = detect_setup(symbol, frames)
        if features is None:
            self.on_log(f"{symbol}: insufficient data")
            return True

        decision = evaluate_setup(features)
        self.on_log(
            format_watch_card(
                decision, sr, bid, ask, bool(self.execute), closed_m5=key
            )
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
