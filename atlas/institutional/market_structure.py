"""
Market structure engine — HH/HL/LH/LL, BOS, CHoCH, premium/discount.

Wraps atlas.analysis.structure and adds institutional scoring.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from atlas.analysis.structure import (
    detect_bos,
    detect_choch,
    find_swing_highs,
    find_swing_lows,
    infer_trend_from_swings,
)
from atlas.analysis.volatility import latest_atr
from atlas.institutional.models import Bias, ModuleScore
from atlas.models import Direction, TrendStrength


@dataclass
class StructureReport:
    bias: Bias
    confidence: float
    strength: float
    direction_score: float
    last_bos: str | None
    last_choch: str | None
    premium: bool
    discount: bool
    equilibrium: float
    summary: str
    module: ModuleScore


def analyze_structure(df: pd.DataFrame, lookback: int = 3, atr_period: int = 14) -> StructureReport:
    if df is None or len(df) < 40:
        mod = ModuleScore("market_structure", 0, 14, Bias.NEUTRAL, "insufficient data", False)
        return StructureReport(Bias.NEUTRAL, 0, 0, 0, None, None, False, False, 0, "no data", mod)

    sh = find_swing_highs(df, lookback=lookback)
    sl = find_swing_lows(df, lookback=lookback)
    trend, strength, _conf = infer_trend_from_swings(sh, sl)
    bos_ok, bos_side = detect_bos(df, sh, sl)
    choch_ok, choch_side = detect_choch(df, sh, sl, prior_trend=trend)

    # Premium / discount from last major swing range
    eq = 0.0
    premium = discount = False
    if sh and sl:
        hi = max(s.price for s in sh[-5:])
        lo = min(s.price for s in sl[-5:])
        eq = (hi + lo) / 2.0
        mid = float(df["close"].iloc[-2] if len(df) > 2 else df["close"].iloc[-1])
        premium = mid > eq
        discount = mid < eq

    if trend == Direction.LONG:
        bias = Bias.BULLISH
    elif trend == Direction.SHORT:
        bias = Bias.BEARISH
    else:
        bias = Bias.NEUTRAL

    strength_map = {
        TrendStrength.STRONG: 85.0,
        TrendStrength.MODERATE: 65.0,
        TrendStrength.WEAK: 40.0,
        TrendStrength.NONE: 15.0,
    }
    base = strength_map.get(strength, 20.0)
    if bos_ok:
        base = min(100.0, base + 10.0)
    if choch_ok:
        base = min(100.0, base + 8.0)

    # Direction score: reward structure aligned with premium/discount
    direction_score = base
    if bias == Bias.BULLISH and discount:
        direction_score = min(100.0, direction_score + 8)
    if bias == Bias.BEARISH and premium:
        direction_score = min(100.0, direction_score + 8)

    conf = min(1.0, direction_score / 100.0)
    parts = [f"trend={trend.value}/{strength.value}"]
    if bos_side:
        parts.append(f"BOS={bos_side}")
    if choch_side:
        parts.append(f"CHoCH={choch_side}")
    parts.append("premium" if premium else ("discount" if discount else "mid"))
    summary = " | ".join(parts)

    mod = ModuleScore(
        name="market_structure",
        score=direction_score,
        weight=14,
        bias=bias,
        detail=summary,
        passed=direction_score >= 55 and bias != Bias.NEUTRAL,
    )
    return StructureReport(
        bias=bias,
        confidence=conf,
        strength=base,
        direction_score=direction_score,
        last_bos=bos_side,
        last_choch=choch_side,
        premium=premium,
        discount=discount,
        equilibrium=eq,
        summary=summary,
        module=mod,
    )
