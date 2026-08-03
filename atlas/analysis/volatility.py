"""ATR and volatility helpers — shared, not duplicated elsewhere."""

from __future__ import annotations

import numpy as np
import pandas as pd


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    ranges = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    )
    return ranges.max(axis=1)


def atr(
    df: pd.DataFrame,
    period: int = 14,
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
) -> pd.Series:
    """Wilder-smoothed Average True Range."""
    tr = true_range(df[high_col], df[low_col], df[close_col])
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def latest_atr(df: pd.DataFrame, period: int = 14) -> float:
    series = atr(df, period=period)
    val = series.iloc[-1]
    if pd.isna(val):
        return float(df["high"].iloc[-period:].max() - df["low"].iloc[-period:].min()) / period
    return float(val)


def is_volatility_expanding(
    df: pd.DataFrame,
    period: int = 14,
    lookback: int = 20,
    expansion_ratio: float = 1.1,
) -> bool:
    """True when current ATR is elevated vs recent average ATR."""
    series = atr(df, period=period).dropna()
    if len(series) < lookback + 1:
        return False
    current = float(series.iloc[-1])
    baseline = float(series.iloc[-(lookback + 1) : -1].mean())
    if baseline <= 0:
        return False
    return current >= baseline * expansion_ratio
