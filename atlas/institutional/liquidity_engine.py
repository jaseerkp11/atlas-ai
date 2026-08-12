"""Liquidity analysis — equal highs/lows, session extremes, sweeps."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from atlas.analysis.liquidity import detect_liquidity_sweep, find_equal_levels
from atlas.analysis.structure import find_swing_highs, find_swing_lows
from atlas.analysis.volatility import latest_atr
from atlas.institutional.models import Bias, ModuleScore, Zone, ZoneStrength
from atlas.models import Direction


@dataclass
class LiquidityPool:
    kind: str
    price: float
    significance: float
    label: str


@dataclass
class LiquidityReport:
    pools: list[LiquidityPool]
    sweep_bullish: bool
    sweep_bearish: bool
    score: float
    bias: Bias
    summary: str
    module: ModuleScore


def _session_extremes(df: pd.DataFrame) -> list[LiquidityPool]:
    pools: list[LiquidityPool] = []
    if df is None or len(df) < 10 or "time" not in df.columns:
        return pools
    try:
        from atlas.institutional.session_levels import extract_session_levels

        report = extract_session_levels(df)
        kind_map = {
            "pdh": "daily_high",
            "pdl": "daily_low",
            "asian_high": "daily_high",
            "asian_low": "daily_low",
            "london_high": "daily_high",
            "london_low": "daily_low",
        }
        for lv in report.levels:
            pools.append(
                LiquidityPool(
                    kind_map.get(lv.kind, "swing_high_liq" if lv.side == "SELL" else "swing_low_liq"),
                    lv.price,
                    lv.score,
                    lv.label,
                )
            )
        # Keep weekly rolling extremes as secondary context
        closed = df.iloc[:-1] if len(df) >= 2 else df
        week = closed.tail(96 * 5)
        if len(week):
            pools.append(LiquidityPool("weekly_high", float(week["high"].max()), 90, "Weekly High"))
            pools.append(LiquidityPool("weekly_low", float(week["low"].min()), 90, "Weekly Low"))
    except Exception:
        try:
            closed = df.iloc[:-1] if len(df) >= 2 else df
            day = closed.tail(96)
            pools.append(LiquidityPool("daily_high", float(day["high"].max()), 80, "Daily High"))
            pools.append(LiquidityPool("daily_low", float(day["low"].min()), 80, "Daily Low"))
        except Exception:
            pass
    return pools



def analyze_liquidity(
    df: pd.DataFrame,
    lookback: int = 3,
    equal_tol_atr: float = 0.15,
) -> LiquidityReport:
    if df is None or len(df) < 40:
        mod = ModuleScore("liquidity", 0, 10, Bias.NEUTRAL, "no data", False)
        return LiquidityReport([], False, False, 0, Bias.NEUTRAL, "no data", mod)

    atr = latest_atr(df, 14) or 1.0
    sh = find_swing_highs(df, lookback=lookback)
    sl = find_swing_lows(df, lookback=lookback)
    tol = atr * float(equal_tol_atr)
    eq_h = find_equal_levels(sh, tolerance=tol)
    eq_l = find_equal_levels(sl, tolerance=tol)

    pools: list[LiquidityPool] = []
    for a, b in eq_h:
        p = (a.price + b.price) / 2.0
        pools.append(LiquidityPool("equal_highs", float(p), 75, f"EQH@{p:.2f}"))
    for a, b in eq_l:
        p = (a.price + b.price) / 2.0
        pools.append(LiquidityPool("equal_lows", float(p), 75, f"EQL@{p:.2f}"))
    if sh:
        pools.append(LiquidityPool("swing_high_liq", sh[-1].price, 60, "Swing High Liq"))
    if sl:
        pools.append(LiquidityPool("swing_low_liq", sl[-1].price, 60, "Swing Low Liq"))
    pools.extend(_session_extremes(df))

    sweep = detect_liquidity_sweep(df, sh, sl)
    sweep_bull = sweep is not None and sweep.direction == Direction.LONG
    sweep_bear = sweep is not None and sweep.direction == Direction.SHORT

    score = 40.0
    bias = Bias.NEUTRAL
    if sweep_bull:
        score += 30
        bias = Bias.BULLISH
    if sweep_bear:
        score += 30
        bias = Bias.BEARISH
    score += min(20.0, len(eq_h) * 5 + len(eq_l) * 5)
    score = min(100.0, score)

    summary = f"pools={len(pools)} eqH={len(eq_h)} eqL={len(eq_l)} sweep={'bull' if sweep_bull else ('bear' if sweep_bear else 'none')}"
    mod = ModuleScore("liquidity", score, 10, bias, summary, score >= 55 and bias != Bias.NEUTRAL)
    return LiquidityReport(pools, sweep_bull, sweep_bear, score, bias, summary, mod)
