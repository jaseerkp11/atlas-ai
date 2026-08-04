"""Fair Value Gap + Order Block institutional wrappers."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from atlas.analysis.fvg import active_fvgs_for_direction, detect_fair_value_gaps
from atlas.analysis.order_blocks import active_order_blocks, detect_order_blocks
from atlas.analysis.volatility import latest_atr
from atlas.institutional.models import Bias, ModuleScore, Zone, ZoneStrength
from atlas.models import Direction


@dataclass
class FVGReport:
    zones: list[Zone]
    score: float
    bias: Bias
    summary: str
    module: ModuleScore


@dataclass
class OBReport:
    zones: list[Zone]
    score: float
    bias: Bias
    summary: str
    module: ModuleScore


def analyze_fvg(df: pd.DataFrame, mid: float) -> FVGReport:
    if df is None or len(df) < 30:
        mod = ModuleScore("fair_value_gaps", 0, 8, Bias.NEUTRAL, "no data", False)
        return FVGReport([], 0, Bias.NEUTRAL, "no data", mod)
    atr = latest_atr(df, 14) or 1.0
    fvgs = detect_fair_value_gaps(df)
    bull = active_fvgs_for_direction(fvgs, Direction.LONG, price=mid, atr=atr)
    bear = active_fvgs_for_direction(fvgs, Direction.SHORT, price=mid, atr=atr)
    zones: list[Zone] = []
    for f in bull[:5]:
        fresh = abs(((f.top + f.bottom) / 2) - mid) / atr < 4
        score = 70.0 if not f.filled else 30.0
        if fresh:
            score += 15
        zones.append(
            Zone(
                kind="bullish_fvg",
                top=f.top,
                bottom=f.bottom,
                strength=ZoneStrength.STRONG if score >= 70 else ZoneStrength.NEUTRAL,
                score=score,
                timeframe="M15",
                fresh=fresh and not f.filled,
                label=f"BullFVG {f.bottom:.2f}-{f.top:.2f}",
            )
        )
    for f in bear[:5]:
        fresh = abs(((f.top + f.bottom) / 2) - mid) / atr < 4
        score = 70.0 if not f.filled else 30.0
        if fresh:
            score += 15
        zones.append(
            Zone(
                kind="bearish_fvg",
                top=f.top,
                bottom=f.bottom,
                strength=ZoneStrength.STRONG if score >= 70 else ZoneStrength.NEUTRAL,
                score=score,
                timeframe="M15",
                fresh=fresh and not f.filled,
                label=f"BearFVG {f.bottom:.2f}-{f.top:.2f}",
            )
        )
    near_bull = [z for z in zones if z.kind == "bullish_fvg" and z.bottom <= mid <= z.top + atr]
    near_bear = [z for z in zones if z.kind == "bearish_fvg" and z.bottom - atr <= mid <= z.top]
    bias = Bias.NEUTRAL
    score = 35.0
    if near_bull:
        bias = Bias.BULLISH
        score = max(z.score for z in near_bull)
    elif near_bear:
        bias = Bias.BEARISH
        score = max(z.score for z in near_bear)
    elif bull and not bear:
        bias = Bias.BULLISH
        score = 55
    elif bear and not bull:
        bias = Bias.BEARISH
        score = 55
    summary = f"active_bull={len(bull)} active_bear={len(bear)} near={len(near_bull)+len(near_bear)}"
    mod = ModuleScore("fair_value_gaps", score, 8, bias, summary, score >= 55 and bias != Bias.NEUTRAL)
    return FVGReport(zones, score, bias, summary, mod)


def analyze_order_blocks(df: pd.DataFrame, mid: float) -> OBReport:
    if df is None or len(df) < 40:
        mod = ModuleScore("order_blocks", 0, 8, Bias.NEUTRAL, "no data", False)
        return OBReport([], 0, Bias.NEUTRAL, "no data", mod)
    atr = latest_atr(df, 14) or 1.0
    obs = detect_order_blocks(df)
    bull = active_order_blocks(obs, Direction.LONG, price=mid, atr=atr)
    bear = active_order_blocks(obs, Direction.SHORT, price=mid, atr=atr)
    zones: list[Zone] = []
    for o in bull[:5]:
        score = 75.0 if not o.mitigated else 25.0
        zones.append(
            Zone(
                kind="bullish_ob",
                top=o.top,
                bottom=o.bottom,
                strength=ZoneStrength.STRONG if score >= 70 else ZoneStrength.WEAK,
                score=score,
                timeframe="M15",
                fresh=not o.mitigated,
                label=f"BullOB {o.bottom:.2f}-{o.top:.2f}",
            )
        )
    for o in bear[:5]:
        score = 75.0 if not o.mitigated else 25.0
        zones.append(
            Zone(
                kind="bearish_ob",
                top=o.top,
                bottom=o.bottom,
                strength=ZoneStrength.STRONG if score >= 70 else ZoneStrength.WEAK,
                score=score,
                timeframe="M15",
                fresh=not o.mitigated,
                label=f"BearOB {o.bottom:.2f}-{o.top:.2f}",
            )
        )
    near_bull = [z for z in zones if z.kind == "bullish_ob" and z.bottom - atr * 0.2 <= mid <= z.top + atr * 0.2]
    near_bear = [z for z in zones if z.kind == "bearish_ob" and z.bottom - atr * 0.2 <= mid <= z.top + atr * 0.2]
    bias = Bias.NEUTRAL
    score = 35.0
    if near_bull:
        bias = Bias.BULLISH
        score = max(z.score for z in near_bull)
    elif near_bear:
        bias = Bias.BEARISH
        score = max(z.score for z in near_bear)
    summary = f"bullOB={len(bull)} bearOB={len(bear)} near={len(near_bull)+len(near_bear)}"
    mod = ModuleScore("order_blocks", score, 8, bias, summary, score >= 55 and bias != Bias.NEUTRAL)
    return OBReport(zones, score, bias, summary, mod)
