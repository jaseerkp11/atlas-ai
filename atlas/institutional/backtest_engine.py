"""
Institutional backtest — shared decision logic on historical bars.

Walks M15 closes, rebuilds frames windows, calls InstitutionalAnalyzer.analyze.
Reports win rate, PF, drawdown, expectancy. No fabricated claims beyond sample stats.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import pandas as pd

from atlas.institutional.analyzer import InstitutionalAnalyzer
from atlas.institutional.config import load_institutional_config
from atlas.institutional.models import DecisionAction


@dataclass
class BTTrade:
    side: str
    entry: float
    stop: float
    tp: float
    exit: float
    pnl_r: float
    reason: str


@dataclass
class BTResult:
    trades: list[BTTrade] = field(default_factory=list)
    bars: int = 0
    signals: int = 0
    no_trade: int = 0

    @property
    def win_rate(self) -> float:
        if not self.trades:
            return 0.0
        return round(100.0 * sum(1 for t in self.trades if t.pnl_r > 0) / len(self.trades), 2)

    @property
    def profit_factor(self) -> float:
        gains = sum(t.pnl_r for t in self.trades if t.pnl_r > 0)
        losses = abs(sum(t.pnl_r for t in self.trades if t.pnl_r < 0))
        if losses <= 1e-9:
            return float("inf") if gains > 0 else 0.0
        return round(gains / losses, 3)

    @property
    def expectancy(self) -> float:
        if not self.trades:
            return 0.0
        return round(sum(t.pnl_r for t in self.trades) / len(self.trades), 3)

    @property
    def max_dd_r(self) -> float:
        equity = 0.0
        peak = 0.0
        dd = 0.0
        for t in self.trades:
            equity += t.pnl_r
            peak = max(peak, equity)
            dd = min(dd, equity - peak)
        return round(dd, 3)


def run_institutional_backtest(
    frames: dict[str, pd.DataFrame],
    step: int = 5,
    on_log: Callable[[str], None] | None = None,
) -> BTResult:
    """
    Simplified institutional backtest on provided multi-TF frames.
    Uses trailing window ending at each M15 step.
    """
    log = on_log or (lambda m: None)
    cfg = load_institutional_config(reload=True)
    analyzer = InstitutionalAnalyzer(cfg)
    # Don't need live MT5
    analyzer._connected = True

    m15 = frames.get("M15")
    if m15 is None or len(m15) < 100:
        log("insufficient M15 data")
        return BTResult()

    result = BTResult()
    pending: BTTrade | None = None

    for i in range(80, len(m15) - 2, step):
        result.bars += 1
        windowed: dict[str, pd.DataFrame] = {}
        for tf, df in frames.items():
            if df is None or len(df) < 30:
                continue
            # Align by taking proportional prefix
            frac = i / max(len(m15), 1)
            end = max(30, int(len(df) * frac))
            windowed[tf] = df.iloc[:end].reset_index(drop=True)
        if "M15" not in windowed:
            windowed["M15"] = m15.iloc[: i + 1].reset_index(drop=True)

        # Resolve pending on path
        if pending is not None:
            bar = m15.iloc[i]
            hi, lo = float(bar["high"]), float(bar["low"])
            hit = None
            if pending.side == "BUY":
                if lo <= pending.stop:
                    hit = pending.stop
                    pending.pnl_r = -1.0
                elif hi >= pending.tp:
                    hit = pending.tp
                    pending.pnl_r = pending.tp / pending.entry * 0 + (
                        (pending.tp - pending.entry) / (pending.entry - pending.stop)
                    )
            else:
                if hi >= pending.stop:
                    hit = pending.stop
                    pending.pnl_r = -1.0
                elif lo <= pending.tp:
                    hit = pending.tp
                    risk = pending.stop - pending.entry
                    pending.pnl_r = (pending.entry - pending.tp) / risk if risk else 0.0
            if hit is not None:
                pending.exit = hit
                result.trades.append(pending)
                pending = None

        if pending is not None:
            continue

        try:
            decision = analyzer.analyze(cfg.symbol, frames=windowed)
        except Exception as exc:
            log(f"analyze error @ {i}: {exc}")
            continue

        if not decision.is_executable():
            result.no_trade += 1
            continue

        result.signals += 1
        pending = BTTrade(
            side=decision.action.value,
            entry=decision.entry,
            stop=decision.stop,
            tp=decision.take_profit,
            exit=0.0,
            pnl_r=0.0,
            reason=decision.why[0] if decision.why else "",
        )

    log(
        f"BT done bars={result.bars} signals={result.signals} trades={len(result.trades)} "
        f"WR={result.win_rate}% PF={result.profit_factor} ExpR={result.expectancy} MaxDD_R={result.max_dd_r}"
    )
    return result
