"""
Fair Value Gap (imbalance) detection — three-candle displacement gaps.
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from atlas.models import Direction, FairValueGap


def detect_fair_value_gaps(
    df: pd.DataFrame,
    lookback: int = 30,
    min_gap_atr_frac: float = 0.05,
    atr_value: float | None = None,
) -> list[FairValueGap]:
    """
    Bullish FVG: candle[i-2].high < candle[i].low  (gap up left unfilled by middle wick fully covering).
    Bearish FVG: candle[i-2].low > candle[i].high.

    Returns most recent unfilled gaps within lookback, newest last.
    """
    gaps: list[FairValueGap] = []
    n = len(df)
    if n < 3:
        return gaps

    start = max(2, n - lookback)
    min_gap = (atr_value or 0.0) * min_gap_atr_frac

    for i in range(start, n):
        c0_high = float(df["high"].iloc[i - 2])
        c0_low = float(df["low"].iloc[i - 2])
        c2_high = float(df["high"].iloc[i])
        c2_low = float(df["low"].iloc[i])
        time = _bar_time(df, i)

        # Bullish imbalance
        if c2_low > c0_high:
            gap_size = c2_low - c0_high
            if gap_size >= min_gap:
                top, bottom = c2_low, c0_high
                filled = _is_filled(df, i + 1, bottom, top, bullish=True)
                gaps.append(
                    FairValueGap(
                        direction=Direction.LONG,
                        top=top,
                        bottom=bottom,
                        index=i,
                        time=time,
                        filled=filled,
                    )
                )

        # Bearish imbalance
        if c2_high < c0_low:
            gap_size = c0_low - c2_high
            if gap_size >= min_gap:
                top, bottom = c0_low, c2_high
                filled = _is_filled(df, i + 1, bottom, top, bullish=False)
                gaps.append(
                    FairValueGap(
                        direction=Direction.SHORT,
                        top=top,
                        bottom=bottom,
                        index=i,
                        time=time,
                        filled=filled,
                    )
                )

    return gaps


def active_fvgs_for_direction(
    gaps: list[FairValueGap],
    direction: Direction,
    *,
    price: float | None = None,
    atr: float | None = None,
    max_age_bars: int | None = None,
    current_index: int | None = None,
    max_distance_atr: float = 3.0,
) -> list[FairValueGap]:
    """Unfilled FVGs in direction; optionally require recent + near live price."""
    out: list[FairValueGap] = []
    for g in gaps:
        if g.direction != direction or g.filled:
            continue
        if max_age_bars is not None and current_index is not None:
            if current_index - g.index > max_age_bars:
                continue
        if price is not None and atr is not None and atr > 0:
            mid = (g.top + g.bottom) / 2.0
            if abs(mid - price) > max_distance_atr * atr:
                continue
        out.append(g)
    return out


def price_in_fvg(price: float, gap: FairValueGap) -> bool:
    return gap.bottom <= price <= gap.top


def _is_filled(
    df: pd.DataFrame,
    from_index: int,
    bottom: float,
    top: float,
    bullish: bool,
) -> bool:
    """
    Mark filled/invalid for scalp use when price fully trades through OR
    has already left the imbalance far behind (gap is stale context).
    """
    if from_index >= len(df):
        return False
    for j in range(from_index, len(df)):
        low = float(df["low"].iloc[j])
        high = float(df["high"].iloc[j])
        close = float(df["close"].iloc[j])
        # Classic full fill
        if bullish and low <= bottom:
            return True
        if not bullish and high >= top:
            return True
        # Price traded deep into the gap (majority fill)
        if bullish and low <= (bottom + top) / 2.0:
            return True
        if not bullish and high >= (bottom + top) / 2.0:
            return True
    return False


def _bar_time(df: pd.DataFrame, index: int) -> datetime | None:
    if "time" not in df.columns:
        return None
    val = df["time"].iloc[index]
    if pd.isna(val):
        return None
    return pd.Timestamp(val).to_pydatetime()
