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
    Real strategy stack:
      H4/H1 = bias
      M15    = setup zones (BOS/CHoCH, sweep, FVG, order block, S/R context)
      M5     = entry trigger (uses last closed M5 bars, not the forming candle)
    """
    settings = load_settings()
    analyses = analyze_all_timeframes(frames)
    bias = aligned_bias(analyses)

    # Setup zones from M15 (and fall back to M5 only if missing)
    setup_tf = str(settings.timeframes.get("setup", "M15")).upper()
    setup_df = frames.get(setup_tf)
    if setup_df is None:
        setup_df = frames.get("M15")
    if setup_df is None:
        setup_df = frames.get("M5")

    m5_raw = frames.get("M5")
    m15 = frames.get("M15")
    h1 = frames.get("H1")

    if setup_df is None or m5_raw is None or len(setup_df) < 30 or len(m5_raw) < 30:
        return None

    # Drop currently forming M5 candle so trigger/levels use closed market structure
    m5 = m5_raw.iloc[:-1].reset_index(drop=True) if len(m5_raw) > 30 else m5_raw
    # Prefer last closed M15 bar set too when setup is M15
    if setup_tf == "M15" and len(setup_df) > 30:
        setup_df = setup_df.iloc[:-1].reset_index(drop=True)

    swing_lb = int(settings.analysis["swing_lookback"])
    atr_period = int(settings.risk.atr_period)

    sh = find_swing_highs(setup_df, lookback=swing_lb)
    sl = find_swing_lows(setup_df, lookback=swing_lb)
    prior_trend = analyses["H1"].direction if "H1" in analyses else Direction.NEUTRAL

    bos_ok, bos_side = detect_bos(
        setup_df, sh, sl, direction_bias=bias if bias != Direction.NEUTRAL else None
    )
    choch_ok, choch_side = detect_choch(setup_df, sh, sl, prior_trend=prior_trend)

    direction = bias
    if direction == Direction.NEUTRAL:
        if bos_side == "bullish" or choch_side == "bullish":
            direction = Direction.LONG
        elif bos_side == "bearish" or choch_side == "bearish":
            direction = Direction.SHORT

    atr_setup = latest_atr(setup_df, period=atr_period)
    atr_m5 = latest_atr(m5, period=atr_period)
    atr_val = atr_m5 if atr_m5 > 0 else atr_setup

    sweep = detect_liquidity_sweep(
        setup_df,
        sh,
        sl,
        lookback=int(settings.analysis["liquidity_lookback"]),
    )

    # Primary zones on M15; also accept fresh near-price H1 zones (confluence)
    fvgs_m15 = detect_fair_value_gaps(
        setup_df,
        lookback=int(settings.analysis["fvg_lookback"]),
        atr_value=atr_setup,
    )
    obs_m15 = detect_order_blocks(
        setup_df,
        lookback=int(settings.analysis["order_block_lookback"]),
        atr_value=atr_setup,
    )

    price_now = float(m5["close"].iloc[-1])
    max_age = int(settings.analysis.get("structure_max_age_bars", 40))
    prox = float(settings.analysis.get("structure_proximity_atr", 4.0))
    cur_idx = len(setup_df) - 1

    active_fvg: list = []
    active_ob: list = []
    if direction != Direction.NEUTRAL:
        active_fvg = active_fvgs_for_direction(
            fvgs_m15,
            direction,
            price=price_now,
            atr=atr_val,
            max_age_bars=max_age,
            current_index=cur_idx,
            max_distance_atr=prox,
        )
        active_ob = active_order_blocks(
            obs_m15,
            direction,
            price=price_now,
            atr=atr_val,
            max_age_bars=max_age,
            current_index=cur_idx,
            max_distance_atr=prox,
        )
        if h1 is not None and len(h1) >= 40:
            h1_use = h1.iloc[:-1].reset_index(drop=True) if len(h1) > 40 else h1
            atr_h1 = latest_atr(h1_use, period=atr_period)
            h1_age = min(20, max_age)
            active_fvg = active_fvg + active_fvgs_for_direction(
                detect_fair_value_gaps(
                    h1_use,
                    lookback=min(24, int(settings.analysis["fvg_lookback"])),
                    atr_value=atr_h1,
                ),
                direction,
                price=price_now,
                atr=atr_h1 if atr_h1 > 0 else atr_val,
                max_age_bars=h1_age,
                current_index=len(h1_use) - 1,
                max_distance_atr=prox,
            )
            active_ob = active_ob + active_order_blocks(
                detect_order_blocks(
                    h1_use,
                    lookback=min(24, int(settings.analysis["order_block_lookback"])),
                    atr_value=atr_h1,
                ),
                direction,
                price=price_now,
                atr=atr_h1 if atr_h1 > 0 else atr_val,
                max_age_bars=h1_age,
                current_index=len(h1_use) - 1,
                max_distance_atr=prox,
            )

    if direction == Direction.NEUTRAL and sweep is not None:
        direction = sweep.direction
        # re-filter if direction just became known
        if direction != Direction.NEUTRAL and not active_fvg and not active_ob:
            active_fvg = active_fvgs_for_direction(
                fvgs_m15,
                direction,
                price=price_now,
                atr=atr_val,
                max_age_bars=max_age,
                current_index=cur_idx,
                max_distance_atr=prox,
            )
            active_ob = active_order_blocks(
                obs_m15,
                direction,
                price=price_now,
                atr=atr_val,
                max_age_bars=max_age,
                current_index=cur_idx,
                max_distance_atr=prox,
            )

    m5_trigger = _detect_m5_trigger(m5, direction, analyses.get("M5"))
    vol_df = m15 if m15 is not None and len(m15) >= 30 else setup_df
    if vol_df is not None and len(vol_df) > 30:
        vol_df = vol_df.iloc[:-1]
    vol_expanding = is_volatility_expanding(vol_df, period=atr_period)

    entry, stop, target, rr = _compute_levels(
        direction=direction,
        m5=m5,
        atr=atr_val,
        sweep=sweep,
        order_blocks=active_ob,
        fvgs=active_fvg,
        swing_highs=sh,
        swing_lows=sl,
    )

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
    scalp_entry = bool(settings.analysis.get("scalp_market_entry", False))

    if direction == Direction.NEUTRAL or atr <= 0:
        return price, price, price, 0.0

    # Scalp: enter near live price. Swing/OB/FVG still define stop context.
    entry = price
    if not scalp_entry:
        if order_blocks:
            ob = order_blocks[-1]
            entry = (ob.top + ob.bottom) / 2.0
        elif fvgs:
            g = fvgs[-1]
            entry = (g.top + g.bottom) / 2.0
        elif sweep is not None:
            entry = sweep.level

    if direction == Direction.LONG:
        if not scalp_entry:
            entry = min(entry, price)
        structural_stop = None
        if order_blocks:
            structural_stop = order_blocks[-1].bottom
        elif swing_lows:
            structural_stop = swing_lows[-1].price
        atr_stop = entry - atr * stop_mult
        stop = min(atr_stop, structural_stop) if structural_stop is not None else atr_stop
        if stop >= entry:
            stop = entry - atr * stop_mult
        risk = entry - stop
        target = entry + max(risk * settings.gates.min_reward_risk, atr * tgt_mult)
    else:
        if not scalp_entry:
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
