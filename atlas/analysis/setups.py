"""
Setup detection orchestrator.

Identifies per symbol: BOS, CHoCH, liquidity sweeps, FVGs, order blocks,
and M5 trigger — composing the reusable analysis functions (no duplicated logic).
"""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from atlas.analysis.fvg import active_fvgs_for_direction, detect_fair_value_gaps
from atlas.analysis.liquidity import detect_liquidity_sweep
from atlas.analysis.order_blocks import active_order_blocks, detect_order_blocks
from atlas.analysis.structure import (
    detect_bos,
    detect_choch,
    find_swing_highs,
    find_swing_lows,
)
from atlas.analysis.timeframes import aligned_bias, analyze_all_timeframes
from atlas.analysis.volatility import is_volatility_expanding, latest_atr
from atlas.config import load_settings
from atlas.models import Direction, SetupFeatures


def detect_setup(
    symbol: str,
    frames: dict[str, pd.DataFrame],
) -> SetupFeatures | None:
    """
    Build a SetupFeatures object for `symbol` from multi-TF OHLC frames.

    Returns None only when there is no usable directional bias at all
    (caller still scores/rejects with reasoning when a setup object exists).
    For transparency we always return a features object when M15/M5 data exist,
    even if score will be low — so reasoning can be printed.
    """
    settings = load_settings()
    analyses = analyze_all_timeframes(frames)
    bias = aligned_bias(analyses)

    m15 = frames.get("M15")
    m5 = frames.get("M5")
    if m15 is None or m5 is None or len(m15) < 30 or len(m5) < 30:
        return None

    swing_lb = int(settings.analysis["swing_lookback"])
    atr_period = int(settings.risk.atr_period)

    # Structure on setup timeframe (M15)
    sh15 = find_swing_highs(m15, lookback=swing_lb)
    sl15 = find_swing_lows(m15, lookback=swing_lb)
    prior_trend = analyses["H1"].direction if "H1" in analyses else Direction.NEUTRAL

    bos_ok, bos_side = detect_bos(m15, sh15, sl15, direction_bias=bias if bias != Direction.NEUTRAL else None)
    choch_ok, choch_side = detect_choch(m15, sh15, sl15, prior_trend=prior_trend)

    # If no H4/H1 alignment, try to use structure-derived direction
    direction = bias
    if direction == Direction.NEUTRAL:
        if bos_side == "bullish" or choch_side == "bullish":
            direction = Direction.LONG
        elif bos_side == "bearish" or choch_side == "bearish":
            direction = Direction.SHORT

    atr_m15 = latest_atr(m15, period=atr_period)
    atr_m5 = latest_atr(m5, period=atr_period)
    atr_val = atr_m5 if atr_m5 > 0 else atr_m15

    sweep = detect_liquidity_sweep(
        m15,
        sh15,
        sl15,
        lookback=int(settings.analysis["liquidity_lookback"]),
    )

    fvgs = detect_fair_value_gaps(
        m15,
        lookback=int(settings.analysis["fvg_lookback"]),
        atr_value=atr_m15,
    )
    obs = detect_order_blocks(
        m15,
        lookback=int(settings.analysis["order_block_lookback"]),
        atr_value=atr_m15,
    )

    # Direction filter for features
    if direction == Direction.NEUTRAL and sweep is not None:
        direction = sweep.direction

    active_fvg = active_fvgs_for_direction(fvgs, direction) if direction != Direction.NEUTRAL else []
    active_ob = active_order_blocks(obs, direction) if direction != Direction.NEUTRAL else []

    m5_trigger = _detect_m5_trigger(m5, direction, analyses.get("M5"))
    vol_expanding = is_volatility_expanding(m15, period=atr_period)

    entry, stop, target, rr = _compute_levels(
        direction=direction,
        m5=m5,
        atr=atr_val,
        sweep=sweep,
        order_blocks=active_ob,
        fvgs=active_fvg,
        swing_highs=sh15,
        swing_lows=sl15,
    )

    # Confirm BOS/CHoCH align with chosen direction
    bos_aligned = bool(
        bos_ok
        and (
            (direction == Direction.LONG and bos_side == "bullish")
            or (direction == Direction.SHORT and bos_side == "bearish")
        )
    )
    choch_aligned = bool(
        choch_ok
        and (
            (direction == Direction.LONG and choch_side == "bullish")
            or (direction == Direction.SHORT and choch_side == "bearish")
        )
    )

    return SetupFeatures(
        symbol=symbol,
        direction=direction,
        h4=analyses.get("H4"),
        h1=analyses.get("H1"),
        m15=analyses.get("M15"),
        m5=analyses.get("M5"),
        bos=bos_aligned,
        choch=choch_aligned,
        liquidity_sweep=sweep if sweep and sweep.direction == direction else None,
        fair_value_gaps=active_fvg,
        order_blocks=active_ob,
        m5_trigger=m5_trigger,
        atr=atr_val,
        volatility_expanding=vol_expanding,
        entry=entry,
        stop=stop,
        target=target,
        reward_risk=rr,
    )


def _detect_m5_trigger(
    m5: pd.DataFrame,
    direction: Direction,
    m5_analysis,
) -> bool:
    """M5 trigger: micro BOS in trade direction + close in direction of bias."""
    if direction == Direction.NEUTRAL or len(m5) < 20:
        return False
    swing_lb = int(load_settings().analysis["swing_lookback"])
    sh = find_swing_highs(m5, lookback=max(3, swing_lb - 1))
    sl = find_swing_lows(m5, lookback=max(3, swing_lb - 1))
    bos_ok, bos_side = detect_bos(m5, sh, sl, direction_bias=direction)
    close = float(m5["close"].iloc[-1])
    open_ = float(m5["open"].iloc[-1])
    candle_ok = (close > open_) if direction == Direction.LONG else (close < open_)
    tf_ok = bool(m5_analysis and m5_analysis.direction == direction)
    return bool(bos_ok and candle_ok) or bool(tf_ok and candle_ok and bos_ok)


def _compute_levels(
    direction: Direction,
    m5: pd.DataFrame,
    atr: float,
    sweep,
    order_blocks,
    fvgs,
    swing_highs,
    swing_lows,
) -> tuple[float, float, float, float]:
    """
    Entry near structure (OB / FVG / sweep reclaim), stop beyond structure using ATR floor.
    Target set to satisfy structural projection with ATR-based distance.
    """
    settings = load_settings()
    stop_mult = settings.risk.atr_stop_multiplier
    tgt_mult = settings.risk.atr_target_multiplier
    price = float(m5["close"].iloc[-1])

    if direction == Direction.NEUTRAL or atr <= 0:
        return price, price, price, 0.0

    # Prefer entry at nearest unmitigated OB mid, else FVG mid, else slight pullback from close
    entry = price
    if order_blocks:
        ob = order_blocks[-1]
        entry = (ob.top + ob.bottom) / 2.0
    elif fvgs:
        g = fvgs[-1]
        entry = (g.top + g.bottom) / 2.0
    elif sweep is not None:
        entry = sweep.level

    # For limit-style entries away from market, keep entry on the correct side
    if direction == Direction.LONG:
        entry = min(entry, price)
        structural_stop = None
        if order_blocks:
            structural_stop = order_blocks[-1].bottom
        elif swing_lows:
            structural_stop = swing_lows[-1].price
        atr_stop = entry - atr * stop_mult
        stop = min(atr_stop, structural_stop) if structural_stop is not None else atr_stop
        # Ensure stop is below entry
        if stop >= entry:
            stop = entry - atr * stop_mult
        risk = entry - stop
        target = entry + max(risk * settings.gates.min_reward_risk, atr * tgt_mult)
    else:
        entry = max(entry, price)
        structural_stop = None
        if order_blocks:
            structural_stop = order_blocks[-1].top
        elif swing_highs:
            structural_stop = swing_highs[-1].price
        atr_stop = entry + atr * stop_mult
        stop = max(atr_stop, structural_stop) if structural_stop is not None else atr_stop
        if stop <= entry:
            stop = entry + atr * stop_mult
        risk = stop - entry
        target = entry - max(risk * settings.gates.min_reward_risk, atr * tgt_mult)

    risk = abs(entry - stop)
    reward = abs(target - entry)
    rr = (reward / risk) if risk > 0 else 0.0
    return float(entry), float(stop), float(target), float(rr)
