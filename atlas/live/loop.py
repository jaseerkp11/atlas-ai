"""
Real-time evaluation loop.

Re-evaluates on each new M5 candle close, logs full reasoning every time
(trade or no trade), and only auto-executes when every check passes.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Callable

import pandas as pd

from atlas.analysis.setups import detect_setup
from atlas.config import load_settings
from atlas.execution.engine import ExecutionEngine
from atlas.execution.mt5_client import MT5Client
from atlas.journal.trade_journal import TradeJournal
from atlas.risk.position import ConcurrentTradeGuard, DailyRiskState
from atlas.scoring.engine import evaluate_setup

logger = logging.getLogger(__name__)


class LiveLoop:
    def __init__(
        self,
        broker: MT5Client | None = None,
        poll_seconds: float = 15.0,
        on_log: Callable[[str], None] | None = None,
    ) -> None:
        self.broker = broker or MT5Client()
        self.poll_seconds = poll_seconds
        self.on_log = on_log or (lambda m: print(m, flush=True))
        self.journal = TradeJournal()
        self.daily_risk = DailyRiskState()
        self.concurrent = ConcurrentTradeGuard()
        self.engine = ExecutionEngine(
            broker=self.broker,
            journal=self.journal,
            daily_risk=self.daily_risk,
            concurrent=self.concurrent,
            on_log=self.on_log,
        )
        self._last_m5_time: dict[str, pd.Timestamp] = {}
        self._running = False

    def start(self, max_iterations: int | None = None) -> None:
        settings = load_settings()
        if not self.broker.connect():
            self.on_log("FATAL: could not connect broker — loop not started")
            return

        mode = settings.mode
        self.on_log(
            f"ATLAS live loop started | mode={mode} | "
            f"min_score={settings.gates.min_score} | "
            f"min_rr={settings.gates.min_reward_risk} | "
            f"symbols={settings.symbols}"
        )
        if mode == "LIVE":
            self.on_log(
                "⚠ LIVE MODE — real orders will be sent when all checks pass. "
                "No win-rate guarantee. Risk controls are enforced."
            )
        else:
            self.on_log("PAPER mode — no real orders. Pending fills simulated on price touch.")

        self._running = True
        iterations = 0
        try:
            while self._running:
                self._tick()
                iterations += 1
                if max_iterations is not None and iterations >= max_iterations:
                    break
                time.sleep(self.poll_seconds)
        except KeyboardInterrupt:
            self.on_log("Loop stopped by user")
        finally:
            self._running = False
            self.broker.disconnect()
            self.on_log(f"Journal: {self.journal.path_str()}")

    def stop(self) -> None:
        self._running = False

    def _tick(self) -> None:
        settings = load_settings()
        now = datetime.now(timezone.utc)
        self.on_log(f"\n--- tick {now.isoformat()} ---")

        for symbol in settings.symbols:
            try:
                self._process_symbol(symbol)
            except Exception as exc:
                logger.exception("Error processing %s", symbol)
                self.on_log(f"Error on {symbol}: {exc}")

    def _process_symbol(self, symbol: str) -> None:
        settings = load_settings()
        # Fetch multi-TF data
        frames = {
            "M5": self.broker.copy_rates(symbol, "M5", 400),
            "M15": self.broker.copy_rates(symbol, "M15", 300),
            "H1": self.broker.copy_rates(symbol, "H1", 200),
            "H4": self.broker.copy_rates(symbol, "H4", 150),
        }
        m5 = frames["M5"]
        if m5 is None or len(m5) < 50:
            self.on_log(f"{symbol}: insufficient M5 data")
            return

        last_time = pd.Timestamp(m5["time"].iloc[-1])
        prev = self._last_m5_time.get(symbol)
        # Manage open/pending every tick using latest bar
        bar = m5.iloc[-1]
        self.engine.update_market(
            symbol=symbol,
            bar_high=float(bar["high"]),
            bar_low=float(bar["low"]),
            bar_close=float(bar["close"]),
            bar_time=last_time.to_pydatetime(),
        )

        # Only re-score on NEW M5 candle close
        if prev is not None and last_time <= prev:
            return
        self._last_m5_time[symbol] = last_time
        self.on_log(f"{symbol}: new M5 close @ {last_time} close={float(bar['close']):.5f}")

        if any(t.symbol == symbol for t in self.engine.pending + self.engine.open_trades):
            self.on_log(f"{symbol}: skip — existing pending/open trade")
            return

        features = detect_setup(symbol, frames)
        if features is None:
            self.on_log(f"{symbol}: NO TRADE — could not build setup features (data)")
            return

        # Print timeframe summaries
        for tf_name in ("H4", "H1", "M15", "M5"):
            tf = getattr(features, tf_name.lower(), None)
            if tf:
                self.on_log(f"  {tf.summary}")

        decision = evaluate_setup(features)
        # try_execute logs reasoning and blocks if validation fails
        self.engine.try_execute(decision)
