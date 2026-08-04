"""
VWAP — Volume Weighted Average Price.

Uses typical price (H+L+C)/3 and tick_volume (or volume) from MT5.
Session VWAP resets each UTC day by default (challenge-friendly intraday anchor).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from atlas.models import Direction


@dataclass
class VwapReading:
    value: float
    price: float
    side: str  # "above" | "below" | "at"
    distance: float
    distance_atr: float
    aligned: bool
    near_vwap: bool
    summary: str


def compute_vwap_series(
    df: pd.DataFrame,
    *,
    session_reset: bool = True,
) -> pd.Series:
    """
    Rolling session VWAP.
    Prefer 'volume' then 'tick_volume'. If missing, use ones (equal-weight typical price).
    """
    if df is None or len(df) == 0:
        return pd.Series(dtype=float)

    work = df.copy()
    if "volume" in work.columns:
        vol = work["volume"].astype(float).clip(lower=0)
    elif "tick_volume" in work.columns:
        vol = work["tick_volume"].astype(float).clip(lower=0)
    else:
        vol = pd.Series(np.ones(len(work)), index=work.index)

    # Avoid zero-volume collapse
    vol = vol.replace(0, np.nan).fillna(1.0)

    typical = (work["high"].astype(float) + work["low"].astype(float) + work["close"].astype(float)) / 3.0
    tpv = typical * vol

    if session_reset and "time" in work.columns:
        times = pd.to_datetime(work["time"], utc=True)
        day = times.dt.floor("D")
        cum_tpv = tpv.groupby(day).cumsum()
        cum_vol = vol.groupby(day).cumsum()
    else:
        cum_tpv = tpv.cumsum()
        cum_vol = vol.cumsum()

    vwap = cum_tpv / cum_vol.replace(0, np.nan)
    return vwap


def evaluate_vwap(
    df: pd.DataFrame,
    direction: Direction,
    *,
    atr: float = 0.0,
    near_atr_mult: float = 0.5,
    session_reset: bool = True,
) -> VwapReading:
    """
    Score-friendly VWAP reading for the latest closed bar context.
    LONG prefers price at/above VWAP (bullish auction).
    SHORT prefers price at/below VWAP.
    Near-VWAP pullbacks are marked for challenge scalp entries.
    """
    series = compute_vwap_series(df, session_reset=session_reset)
    if series.empty or pd.isna(series.iloc[-1]):
        price = float(df["close"].iloc[-1]) if df is not None and len(df) else 0.0
        return VwapReading(
            value=0.0,
            price=price,
            side="at",
            distance=0.0,
            distance_atr=0.0,
            aligned=False,
            near_vwap=False,
            summary="VWAP unavailable",
        )

    vwap = float(series.iloc[-1])
    price = float(df["close"].iloc[-1])
    dist = price - vwap
    dist_atr = (abs(dist) / atr) if atr and atr > 0 else 0.0
    near = bool(atr > 0 and abs(dist) <= near_atr_mult * atr)

    if abs(dist) <= max(atr * 0.05, 1e-8) if atr else abs(dist) < 1e-8:
        side = "at"
    elif dist > 0:
        side = "above"
    else:
        side = "below"

    aligned = False
    # LONG: above VWAP, or near VWAP (pullback/reclaim zone)
    # SHORT: below VWAP, or near VWAP (pullback/reclaim zone)
    if direction == Direction.LONG:
        aligned = side == "above" or (near and side in ("at", "below"))
    elif direction == Direction.SHORT:
        aligned = side == "below" or (near and side in ("at", "above"))

    summary = (
        f"VWAP={vwap:.5f} | price={price:.5f} ({side} VWAP) | "
        f"dist={dist:+.5f} ({dist_atr:.2f}×ATR) | "
        f"{'ALIGNED' if aligned else 'NOT ALIGNED'}"
        + (" | near-VWAP entry zone" if near else "")
    )
    return VwapReading(
        value=vwap,
        price=price,
        side=side,
        distance=dist,
        distance_atr=dist_atr,
        aligned=aligned,
        near_vwap=near,
        summary=summary,
    )
