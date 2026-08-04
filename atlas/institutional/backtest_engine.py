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
    probability: float = 0.0
    confluence: float = 0.0
    playbook: str = ""
    area_score: float = 0.0


@dataclass
class BTResult:
    trades: list[BTTrade] = field(default_factory=list)
    bars: int = 0
    signals: int = 0
    no_trade: int = 0
    wait: int = 0
    avg_prob_signal: float = 0.0
    avg_prob_standaside: float = 0.0
    avg_area_score: float = 0.0
    data_source: str = "unknown"

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

    @property
    def quality_score(self) -> float:
        """
        Composite 0–100 sample score (not a live guarantee):
          selectivity, expectancy, PF, drawdown, area quality.
        """
        if self.bars <= 0:
            return 0.0
        selectivity = 100.0 * (1.0 - min(1.0, self.signals / max(self.bars, 1)))
        # Prefer selective systems
        sel_pts = min(30.0, selectivity * 0.30)
        if not self.trades:
            # Standing aside often is fine for quality-first — partial credit
            return round(min(55.0, 25.0 + sel_pts * 0.5 + self.avg_area_score * 0.15), 1)

        exp_pts = max(0.0, min(25.0, (self.expectancy + 0.5) * 20.0))
        pf = self.profit_factor if self.profit_factor != float("inf") else 3.0
        pf_pts = max(0.0, min(20.0, (pf - 0.5) * 10.0))
        dd_pts = max(0.0, min(15.0, 15.0 + self.max_dd_r * 2.0))  # max_dd_r is negative
        wr_pts = max(0.0, min(15.0, self.win_rate * 0.15))
        area_pts = max(0.0, min(10.0, self.avg_area_score * 0.10))
        return round(min(100.0, sel_pts * 0.4 + exp_pts + pf_pts + dd_pts + wr_pts + area_pts), 1)

    def summary_lines(self) -> list[str]:
        pf = self.profit_factor if self.profit_factor != float("inf") else "inf"
        return [
            "=" * 64,
            "  INSTITUTIONAL BACKTEST SCORECARD (sample — not a live claim)",
            "=" * 64,
            f"  Data source     : {self.data_source}",
            f"  Bars scanned    : {self.bars}",
            f"  Signals (BUY/SELL): {self.signals}",
            f"  NO_TRADE        : {self.no_trade}",
            f"  WAIT            : {self.wait}",
            f"  Closed trades   : {len(self.trades)}",
            f"  Win rate        : {self.win_rate}%",
            f"  Profit factor   : {pf}",
            f"  Expectancy (R)  : {self.expectancy}",
            f"  Max DD (R)      : {self.max_dd_r}",
            f"  Avg prob signal : {self.avg_prob_signal:.1f}%",
            f"  Avg prob aside  : {self.avg_prob_standaside:.1f}%",
            f"  Avg area score  : {self.avg_area_score:.1f}/100",
            f"  QUALITY SCORE   : {self.quality_score}/100",
            "-" * 64,
            "  Read: high NO_TRADE + selective signals is intentional.",
            "  Validate on PAPER with live MT5 before any LIVE use.",
            "=" * 64,
        ]


def _area_top_score(decision) -> float:
    n = decision.narrative
    if not n:
        return 0.0
    ta = n.extras.get("trade_areas") or {}
    areas = ta.get("areas") or []
    if not areas:
        return 0.0
    return float(max(a.get("score", 0.0) for a in areas))


def _rr_multiple(side: str, entry: float, stop: float, tp: float) -> float:
    if side == "BUY":
        risk = entry - stop
        return (tp - entry) / risk if risk > 1e-9 else 0.0
    risk = stop - entry
    return (entry - tp) / risk if risk > 1e-9 else 0.0


def run_institutional_backtest(
    frames: dict[str, pd.DataFrame],
    step: int = 5,
    on_log: Callable[[str], None] | None = None,
    data_source: str = "historical",
) -> BTResult:
    """
    Simplified institutional backtest on provided multi-TF frames.
    Uses trailing window ending at each M15 step.
    """
    log = on_log or (lambda m: None)
    cfg = load_institutional_config(reload=True)
    analyzer = InstitutionalAnalyzer(cfg)
    analyzer._connected = True

    m15 = frames.get("M15")
    if m15 is None or len(m15) < 100:
        log("insufficient M15 data")
        return BTResult(data_source=data_source)

    result = BTResult(data_source=data_source)
    pending: BTTrade | None = None
    probs_sig: list[float] = []
    probs_aside: list[float] = []
    area_scores: list[float] = []

    for i in range(80, len(m15) - 2, step):
        result.bars += 1
        windowed: dict[str, pd.DataFrame] = {}
        for tf, df in frames.items():
            if df is None or len(df) < 30:
                continue
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
                    pending.pnl_r = _rr_multiple("BUY", pending.entry, pending.stop, pending.tp)
            else:
                if hi >= pending.stop:
                    hit = pending.stop
                    pending.pnl_r = -1.0
                elif lo <= pending.tp:
                    hit = pending.tp
                    pending.pnl_r = _rr_multiple("SELL", pending.entry, pending.stop, pending.tp)
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

        area_scores.append(_area_top_score(decision))

        if decision.action == DecisionAction.WAIT:
            result.wait += 1
            probs_aside.append(decision.probability)
            continue

        if not decision.is_executable():
            result.no_trade += 1
            probs_aside.append(decision.probability)
            continue

        result.signals += 1
        probs_sig.append(decision.probability)
        pb = ""
        if decision.narrative:
            pb = str(decision.narrative.extras.get("playbook", ""))
        pending = BTTrade(
            side=decision.action.value,
            entry=decision.entry,
            stop=decision.stop,
            tp=decision.take_profit,
            exit=0.0,
            pnl_r=0.0,
            reason=decision.why[0] if decision.why else "",
            probability=decision.probability,
            confluence=decision.confluence,
            playbook=pb,
            area_score=_area_top_score(decision),
        )

    if probs_sig:
        result.avg_prob_signal = sum(probs_sig) / len(probs_sig)
    if probs_aside:
        result.avg_prob_standaside = sum(probs_aside) / len(probs_aside)
    if area_scores:
        result.avg_area_score = sum(area_scores) / len(area_scores)

    log(
        f"BT done bars={result.bars} signals={result.signals} trades={len(result.trades)} "
        f"WR={result.win_rate}% PF={result.profit_factor} ExpR={result.expectancy} "
        f"MaxDD_R={result.max_dd_r} SCORE={result.quality_score}"
    )
    return result
