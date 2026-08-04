"""
Multi-timeframe top-down analysis.

Each timeframe independently reports: trend direction, strength/confidence,
and key structural levels (recent swing highs/lows).
"""

from __future__ import annotations

import pandas as pd

from atlas.analysis.structure import (
    detect_bos,
    detect_choch,
    find_swing_highs,
    find_swing_lows,
    infer_trend_from_swings,
    swings_to_levels,
)
from atlas.analysis.volatility import latest_atr
from atlas.config import load_settings
from atlas.models import Direction, TimeframeAnalysis, TrendStrength


TF_MAP = {
    "H4": "H4",
    "H1": "H1",
    "M15": "M15",
    "M5": "M5",
}


def analyze_timeframe(
    df: pd.DataFrame,
    timeframe: str,
    atr_period: int | None = None,
    swing_lookback: int | None = None,
) -> TimeframeAnalysis:
    """Independent analysis for one timeframe dataframe."""
    settings = load_settings()
    swing_lookback = swing_lookback or int(settings.analysis["swing_lookback"])
    atr_period = atr_period or int(settings.risk.atr_period)

    if df is None or len(df) < swing_lookback * 2 + 5:
        return TimeframeAnalysis(
            timeframe=timeframe,
            direction=Direction.NEUTRAL,
            strength=TrendStrength.NONE,
            confidence=0.0,
            summary=f"{timeframe}: insufficient data",
        )

    sh = find_swing_highs(df, lookback=swing_lookback)
    sl = find_swing_lows(df, lookback=swing_lookback)
    direction, strength, confidence = infer_trend_from_swings(sh, sl)

    bos_ok, bos_side = detect_bos(df, sh, sl, direction_bias=direction)
    # Prior trend for CHoCH: opposite inference from older swings
    prior = Direction.NEUTRAL
    if len(sh) >= 3 and len(sl) >= 3:
        prior_dir, _, _ = infer_trend_from_swings(sh[:-1], sl[:-1])
        prior = prior_dir
    choch_ok, choch_side = detect_choch(df, sh, sl, prior_trend=prior)

    atr_val = latest_atr(df, period=atr_period)
    recent_highs = swings_to_levels(sh[-5:], "swing_high")
    recent_lows = swings_to_levels(sl[-5:], "swing_low")

    summary = (
        f"{timeframe}: {direction.value} ({strength.value}, "
        f"conf={confidence:.0%})"
    )
    if bos_ok:
        summary += f", BOS={bos_side}"
    if choch_ok:
        summary += f", CHoCH={choch_side}"

    return TimeframeAnalysis(
        timeframe=timeframe,
        direction=direction,
        strength=strength,
        confidence=confidence,
        swing_highs=recent_highs,
        swing_lows=recent_lows,
        last_bos=bos_side if bos_ok else None,
        last_choch=choch_side if choch_ok else None,
        atr=atr_val,
        summary=summary,
    )


def analyze_all_timeframes(
    frames: dict[str, pd.DataFrame],
) -> dict[str, TimeframeAnalysis]:
    """
    Top-down: H4 regime → H1 trend → M15 setup → M5 entry context.

    `frames` keys should include H4, H1, M15, M5 (as available).
    """
    settings = load_settings()
    result: dict[str, TimeframeAnalysis] = {}
    for role, tf_name in settings.timeframes.items():
        # role: regime/trend/setup/entry → tf_name: H4/H1/M15/M5
        key = tf_name.upper()
        df = frames.get(key)
        if df is None:
            df = frames.get(tf_name)
        if df is None:
            df = frames.get(role)
        if df is None or len(df) == 0:
            result[key] = TimeframeAnalysis(
                timeframe=key,
                direction=Direction.NEUTRAL,
                strength=TrendStrength.NONE,
                confidence=0.0,
                summary=f"{key}: no data",
            )
        else:
            result[key] = analyze_timeframe(df, key)
    return result


def aligned_bias(analyses: dict[str, TimeframeAnalysis]) -> Direction:
    """Require H4 and H1 agreement for a directional bias; else NEUTRAL."""
    h4 = analyses.get("H4")
    h1 = analyses.get("H1")
    if not h4 or not h1:
        return Direction.NEUTRAL
    if h4.direction == h1.direction and h4.direction != Direction.NEUTRAL:
        return h4.direction
    return Direction.NEUTRAL
