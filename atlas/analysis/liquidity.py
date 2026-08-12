"""
Liquidity sweep detection — reusable, not duplicated in scoring/setup code.
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from atlas.models import Direction, LiquiditySweep, SwingPoint


def detect_liquidity_sweep(
    df: pd.DataFrame,
    swing_highs: list[SwingPoint],
    swing_lows: list[SwingPoint],
    lookback: int = 30,
    wick_ratio: float = 0.4,
) -> LiquiditySweep | None:
    """
    A liquidity sweep takes liquidity beyond a swing then closes back inside.

    - Sweep of lows (bullish signal): wick below recent swing low, close back above it.
    - Sweep of highs (bearish signal): wick above recent swing high, close back below it.
    """
    if len(df) < 5:
        return None

    start = max(0, len(df) - lookback)
    # Prefer the most recent completed bar for sweep confirmation
    for offset in range(1, min(6, len(df))):
        i = len(df) - offset
        if i < 1:
            break
        row = df.iloc[i]
        high = float(row["high"])
        low = float(row["low"])
        close = float(row["close"])
        open_ = float(row["open"])
        body = abs(close - open_)
        full = high - low
        if full <= 0:
            continue

        # Candidate swing levels formed before this bar
        prior_lows = [s for s in swing_lows if s.index < i and s.index >= start]
        prior_highs = [s for s in swing_highs if s.index < i and s.index >= start]
        time = _bar_time(df, i)

        if prior_lows:
            level = prior_lows[-1].price
            lower_wick = min(open_, close) - low
            if low < level and close > level and lower_wick / full >= wick_ratio * 0.5:
                return LiquiditySweep(
                    direction=Direction.LONG,
                    level=level,
                    index=i,
                    time=time,
                    swept_kind="swing_low",
                )

        if prior_highs:
            level = prior_highs[-1].price
            upper_wick = high - max(open_, close)
            if high > level and close < level and upper_wick / full >= wick_ratio * 0.5:
                return LiquiditySweep(
                    direction=Direction.SHORT,
                    level=level,
                    index=i,
                    time=time,
                    swept_kind="swing_high",
                )
    return None


def find_equal_levels(
    swings: list[SwingPoint],
    tolerance: float,
) -> list[tuple[SwingPoint, SwingPoint]]:
    """Pairs of swings within `tolerance` price distance (equal highs/lows)."""
    pairs: list[tuple[SwingPoint, SwingPoint]] = []
    for i in range(len(swings)):
        for j in range(i + 1, len(swings)):
            if abs(swings[i].price - swings[j].price) <= tolerance:
                pairs.append((swings[i], swings[j]))
    return pairs


def _bar_time(df: pd.DataFrame, index: int) -> datetime | None:
    if "time" not in df.columns:
        return None
    val = df["time"].iloc[index]
    if pd.isna(val):
        return None
    return pd.Timestamp(val).to_pydatetime()
