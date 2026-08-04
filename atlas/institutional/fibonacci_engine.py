"""Fibonacci AI — dynamic swing anchors, multi-TF confluence, never standalone."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from atlas.analysis.structure import find_swing_highs, find_swing_lows
from atlas.analysis.volatility import latest_atr
from atlas.institutional.models import Bias, FibLevel, ModuleScore

RATIOS = (0.236, 0.382, 0.5, 0.618, 0.705, 0.786)


def _classify(score: float) -> str:
    if score >= 85:
        return "Extremely Strong"
    if score >= 70:
        return "Strong"
    if score >= 50:
        return "Moderate"
    if score >= 30:
        return "Weak"
    return "Ignore"


@dataclass
class FibReport:
    levels: list[FibLevel]
    score: float
    bias: Bias
    summary: str
    module: ModuleScore
    swing_high: float
    swing_low: float
    anchor_tf: str


def _best_swing(df: pd.DataFrame, lookback: int, min_atr: float) -> tuple[float, float] | None:
    if df is None or len(df) < 30:
        return None
    atr = latest_atr(df, 14) or 1.0
    sh = find_swing_highs(df, lookback=lookback)
    sl = find_swing_lows(df, lookback=lookback)
    if not sh or not sl:
        return None
    hi = max(s.price for s in sh[-6:])
    lo = min(s.price for s in sl[-6:])
    if (hi - lo) < atr * min_atr:
        return None
    return hi, lo


def analyze_fibonacci(
    frames: dict[str, pd.DataFrame],
    mid: float,
    lookback: int = 3,
    min_swing_atr: float = 1.5,
    structure_bias: Bias = Bias.NEUTRAL,
) -> FibReport:
    # Choose highest-quality TF swing among H4→H1→M15→M5
    anchor_tf = ""
    pair = None
    for tf in ("H4", "H1", "M15", "M5"):
        p = _best_swing(frames.get(tf), lookback, min_swing_atr)  # type: ignore[arg-type]
        if p:
            pair = p
            anchor_tf = tf
            break
    if pair is None:
        mod = ModuleScore("fibonacci", 0, 8, Bias.NEUTRAL, "no valid swing", False)
        return FibReport([], 0, Bias.NEUTRAL, "no valid swing", mod, 0, 0, "")

    hi, lo = pair
    rng = hi - lo
    # Retracement direction depends on structure: bullish → buy discounts from high
    levels: list[FibLevel] = []
    for r in RATIOS:
        if structure_bias == Bias.BEARISH:
            price = lo + rng * r  # extension from low in bearish retrace up
        else:
            price = hi - rng * r  # classic bullish retracement from high
        dist = abs(price - mid) / max(rng, 1e-9)
        # Prefer golden pocket 0.618–0.705
        base = 40.0
        if 0.6 <= r <= 0.72:
            base = 75.0
        elif r in (0.5, 0.786):
            base = 60.0
        elif r == 0.382:
            base = 50.0
        score = max(0.0, base - dist * 80)
        levels.append(
            FibLevel(
                ratio=r,
                price=price,
                score=score,
                classification=_classify(score),
                timeframe=anchor_tf,
                confluence=score >= 70 and dist < 0.05,
            )
        )

    near = [lv for lv in levels if abs(lv.price - mid) / max(rng, 1e-9) < 0.08]
    score = max((lv.score for lv in near), default=30.0)
    bias = Bias.NEUTRAL
    if near:
        # In discount fib for bullish structure
        if structure_bias == Bias.BULLISH and mid < (hi + lo) / 2:
            bias = Bias.BULLISH
            score = min(100.0, score + 10)
        elif structure_bias == Bias.BEARISH and mid > (hi + lo) / 2:
            bias = Bias.BEARISH
            score = min(100.0, score + 10)
    summary = f"anchor={anchor_tf} HI={hi:.2f} LO={lo:.2f} near={len(near)}"
    mod = ModuleScore("fibonacci", score, 8, bias, summary, score >= 55 and bias != Bias.NEUTRAL)
    return FibReport(levels, score, bias, summary, mod, hi, lo, anchor_tf)
