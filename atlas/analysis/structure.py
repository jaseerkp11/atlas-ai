"""
Market structure: swing points, Break of Structure (BOS), Change of Character (CHoCH).

Reusable functions — call these from scoring/setup detection; do not reimplement.
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from atlas.models import Direction, StructureLevel, SwingPoint, TrendStrength


def find_swing_highs(df: pd.DataFrame, lookback: int = 5) -> list[SwingPoint]:
    """Local swing highs: high[i] > highs in [i-lookback, i+lookback] excluding i."""
    highs: list[SwingPoint] = []
    high = df["high"].values
    n = len(df)
    times = df["time"].values if "time" in df.columns else [None] * n
    for i in range(lookback, n - lookback):
        window = high[i - lookback : i + lookback + 1]
        if high[i] >= window.max() and (window == high[i]).sum() == 1:
            t = _to_dt(times[i])
            highs.append(SwingPoint(index=i, price=float(high[i]), time=t, kind="high"))
    return highs


def find_swing_lows(df: pd.DataFrame, lookback: int = 5) -> list[SwingPoint]:
    lows: list[SwingPoint] = []
    low = df["low"].values
    n = len(df)
    times = df["time"].values if "time" in df.columns else [None] * n
    for i in range(lookback, n - lookback):
        window = low[i - lookback : i + lookback + 1]
        if low[i] <= window.min() and (window == low[i]).sum() == 1:
            t = _to_dt(times[i])
            lows.append(SwingPoint(index=i, price=float(low[i]), time=t, kind="low"))
    return lows


def swings_to_levels(swings: list[SwingPoint], kind: str) -> list[StructureLevel]:
    return [
        StructureLevel(price=s.price, kind=kind, time=s.time, index=s.index)
        for s in swings
    ]


def detect_bos(
    df: pd.DataFrame,
    swing_highs: list[SwingPoint],
    swing_lows: list[SwingPoint],
    direction_bias: Direction | None = None,
) -> tuple[bool, str | None]:
    """
    Break of Structure: close beyond the most recent opposite swing in trend direction.

    Bullish BOS: close above most recent confirmed swing high.
    Bearish BOS: close below most recent confirmed swing low.

    Returns (detected, "bullish"|"bearish"|None).
    """
    if len(df) < 3:
        return False, None

    # Use last CLOSED bar so forming candle does not flip BOS mid-bar
    close = float(df["close"].iloc[-2] if len(df) >= 2 else df["close"].iloc[-1])
    # Use swings that are not on the last candle (need confirmation space)
    last_idx = len(df) - 1
    sh = [s for s in swing_highs if s.index < last_idx - 1]
    sl = [s for s in swing_lows if s.index < last_idx - 1]
    if not sh and not sl:
        return False, None

    bullish = False
    bearish = False
    if sh:
        level = sh[-1].price
        # Recent bars broke and closed above
        recent = df.iloc[max(0, last_idx - 3) :]
        if (recent["close"] > level).any() and close > level:
            bullish = True
    if sl:
        level = sl[-1].price
        recent = df.iloc[max(0, last_idx - 3) :]
        if (recent["close"] < level).any() and close < level:
            bearish = True

    if direction_bias == Direction.LONG and bullish:
        return True, "bullish"
    if direction_bias == Direction.SHORT and bearish:
        return True, "bearish"
    if direction_bias is None or direction_bias == Direction.NEUTRAL:
        if bullish and not bearish:
            return True, "bullish"
        if bearish and not bullish:
            return True, "bearish"
    return False, None


def detect_choch(
    df: pd.DataFrame,
    swing_highs: list[SwingPoint],
    swing_lows: list[SwingPoint],
    prior_trend: Direction,
) -> tuple[bool, str | None]:
    """
    Change of Character: first structural break against the prior trend.

    After a downtrend, bullish CHoCH = close above a recent swing high.
    After an uptrend, bearish CHoCH = close below a recent swing low.
    """
    if prior_trend == Direction.NEUTRAL or len(df) < 3:
        return False, None

    close = float(df["close"].iloc[-2] if len(df) >= 2 else df["close"].iloc[-1])
    last_idx = len(df) - 1
    sh = [s for s in swing_highs if s.index < last_idx - 1]
    sl = [s for s in swing_lows if s.index < last_idx - 1]

    if prior_trend == Direction.SHORT and sh:
        if close > sh[-1].price:
            return True, "bullish"
    if prior_trend == Direction.LONG and sl:
        if close < sl[-1].price:
            return True, "bearish"
    return False, None


def infer_trend_from_swings(
    swing_highs: list[SwingPoint],
    swing_lows: list[SwingPoint],
    min_points: int = 2,
) -> tuple[Direction, TrendStrength, float]:
    """
    Higher highs + higher lows → LONG; lower highs + lower lows → SHORT.
    Confidence based on how consistently the sequence aligns.
    """
    recent_h = swing_highs[-min_points:] if len(swing_highs) >= min_points else swing_highs
    recent_l = swing_lows[-min_points:] if len(swing_lows) >= min_points else swing_lows

    if len(recent_h) < 2 or len(recent_l) < 2:
        return Direction.NEUTRAL, TrendStrength.NONE, 0.0

    hh = all(recent_h[i].price > recent_h[i - 1].price for i in range(1, len(recent_h)))
    hl = all(recent_l[i].price > recent_l[i - 1].price for i in range(1, len(recent_l)))
    lh = all(recent_h[i].price < recent_h[i - 1].price for i in range(1, len(recent_h)))
    ll = all(recent_l[i].price < recent_l[i - 1].price for i in range(1, len(recent_l)))

    if hh and hl:
        strength = TrendStrength.STRONG if len(recent_h) >= 3 and len(recent_l) >= 3 else TrendStrength.MODERATE
        conf = 0.85 if strength == TrendStrength.STRONG else 0.65
        return Direction.LONG, strength, conf
    if lh and ll:
        strength = TrendStrength.STRONG if len(recent_h) >= 3 and len(recent_l) >= 3 else TrendStrength.MODERATE
        conf = 0.85 if strength == TrendStrength.STRONG else 0.65
        return Direction.SHORT, strength, conf
    if hh or hl:
        return Direction.LONG, TrendStrength.WEAK, 0.4
    if lh or ll:
        return Direction.SHORT, TrendStrength.WEAK, 0.4
    return Direction.NEUTRAL, TrendStrength.NONE, 0.1


def _to_dt(value) -> datetime | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, datetime):
        return value
    try:
        return pd.Timestamp(value).to_pydatetime()
    except Exception:
        return None
