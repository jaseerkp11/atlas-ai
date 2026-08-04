"""
High-probability institutional playbooks.

Drawn from widely used SMC / ICT-style filters that historically
improve setup quality (not guarantees of profit):

1. Liquidity Sweep + Reclaim (stop-hunt continuation)
2. OTE / Golden Pocket (0.618–0.786) in discount/premium
3. Order Block + FVG confluence
4. BOS / CHoCH continuation after pullback
5. Killzone session + decisive price action
6. Strong S/R + confirming PA
7. Premium/Discount array (buy discount / sell premium with HTF)
8. Displacement + FVG return (impulse → imbalance → entry)
9. Multi-TF structure stack (H1 + M15 agree)
10. Asian range raid / London expansion (Judas-style)

A playbook match BOOSTS probability; it never forces a trade alone.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from atlas.analysis.volatility import latest_atr
from atlas.institutional.models import Bias, MarketNarrative, ModuleScore, Zone


@dataclass
class PlaybookHit:
    name: str
    bias: Bias
    boost: float  # added to probability (0–25)
    reasons: list[str] = field(default_factory=list)


@dataclass
class PlaybookResult:
    hits: list[PlaybookHit]
    best: PlaybookHit | None
    total_boost: float
    summary: str
    unlock_hints: list[str]


def _mod(modules: list[ModuleScore], name: str) -> ModuleScore | None:
    for m in modules:
        if m.name == name:
            return m
    return None


def _detect_displacement(df: pd.DataFrame | None) -> Bias:
    """Large impulsive candle (≥1.4× ATR body) on last closed bar."""
    if df is None or len(df) < 20:
        return Bias.NEUTRAL
    atr = latest_atr(df, 14) or 1.0
    c = df.iloc[-2]
    body = abs(float(c["close"]) - float(c["open"]))
    if body < atr * 1.4:
        return Bias.NEUTRAL
    return Bias.BULLISH if float(c["close"]) > float(c["open"]) else Bias.BEARISH


def _asian_range(df: pd.DataFrame | None) -> tuple[float, float] | None:
    """Rough Asian session high/low (00:00–07:00 UTC) from recent bars."""
    if df is None or len(df) < 30 or "time" not in df.columns:
        return None
    try:
        tail = df.tail(120).copy()
        times = pd.to_datetime(tail["time"], utc=True)
        hours = times.dt.hour
        asian = tail[(hours >= 0) & (hours < 7)]
        if len(asian) < 4:
            return None
        return float(asian["high"].max()), float(asian["low"].min())
    except Exception:
        return None


def evaluate_playbooks(
    narrative: MarketNarrative,
    modules: list[ModuleScore],
    consensus: Bias,
    session_name: str,
    h1_bias: Bias,
    m15_bias: Bias = Bias.NEUTRAL,
    setup_df: pd.DataFrame | None = None,
) -> PlaybookResult:
    hits: list[PlaybookHit] = []
    unlock: list[str] = []
    atr = max(narrative.atr_m15, 1e-9)
    mid = narrative.mid

    liq = _mod(modules, "liquidity")
    struct = _mod(modules, "market_structure")
    pa = _mod(modules, "price_action")
    detail_s = (struct.detail or "") if struct else ""
    in_discount = "discount" in detail_s
    in_premium = "premium" in detail_s

    # --- 1) Liquidity Sweep + Reclaim ---
    if liq and "sweep=bull" in (liq.detail or ""):
        if h1_bias in (Bias.BULLISH, Bias.NEUTRAL) or (struct and struct.bias == Bias.BULLISH):
            hits.append(
                PlaybookHit(
                    "Sweep+Reclaim (Bullish)",
                    Bias.BULLISH,
                    18.0,
                    ["Buy-side liquidity swept & reclaimed", "Classic stop-hunt continuation"],
                )
            )
    if liq and "sweep=bear" in (liq.detail or ""):
        if h1_bias in (Bias.BEARISH, Bias.NEUTRAL) or (struct and struct.bias == Bias.BEARISH):
            hits.append(
                PlaybookHit(
                    "Sweep+Reclaim (Bearish)",
                    Bias.BEARISH,
                    18.0,
                    ["Sell-side liquidity swept & reclaimed", "Classic stop-hunt continuation"],
                )
            )
    if liq and "sweep=none" in (liq.detail or ""):
        unlock.append("Wait for liquidity sweep + reclaim on M15/H1")

    # --- 2) OTE / Golden Pocket ---
    ote = [
        lv
        for lv in narrative.fib_levels
        if 0.60 <= lv.ratio <= 0.79 and abs(lv.price - mid) / atr <= 1.2
    ]
    if ote and struct:
        near_ote = any(abs(lv.price - mid) / atr <= 0.85 for lv in ote)
        if struct.bias == Bias.BULLISH and (near_ote or in_discount):
            hits.append(
                PlaybookHit(
                    "OTE Discount Long",
                    Bias.BULLISH,
                    14.0,
                    [f"Fib OTE zone ({ote[0].ratio})", "Buy discount / golden pocket logic"],
                )
            )
        if struct.bias == Bias.BEARISH and (near_ote or in_premium):
            hits.append(
                PlaybookHit(
                    "OTE Premium Short",
                    Bias.BEARISH,
                    14.0,
                    [f"Fib OTE zone ({ote[0].ratio})", "Sell premium / golden pocket logic"],
                )
            )
    else:
        unlock.append("Need price into 61.8–78.6 OTE with structure alignment")

    # --- 3) OB + FVG overlap (must be near each other — not anywhere on chart) ---
    bull_fvg = [z for z in narrative.zones if z.kind == "bullish_fvg" and z.fresh]
    bear_fvg = [z for z in narrative.zones if z.kind == "bearish_fvg" and z.fresh]
    bull_ob = [z for z in narrative.zones if z.kind == "bullish_ob" and z.fresh]
    bear_ob = [z for z in narrative.zones if z.kind == "bearish_ob" and z.fresh]

    def _centers_near(a: Zone, b: Zone, max_atr: float = 1.25) -> bool:
        ca = (a.top + a.bottom) / 2.0
        cb = (b.top + b.bottom) / 2.0
        # Overlap OR centers within max_atr
        overlap = min(a.top, b.top) >= max(a.bottom, b.bottom)
        return overlap or abs(ca - cb) / atr <= max_atr

    if any(_centers_near(f, o) for f in bull_fvg for o in bull_ob):
        hits.append(
            PlaybookHit(
                "Bullish OB+FVG Confluence",
                Bias.BULLISH,
                16.0,
                ["Fresh bullish order block overlapping / near FVG"],
            )
        )
    if any(_centers_near(f, o) for f in bear_fvg for o in bear_ob):
        hits.append(
            PlaybookHit(
                "Bearish OB+FVG Confluence",
                Bias.BEARISH,
                16.0,
                ["Fresh bearish order block overlapping / near FVG"],
            )
        )
    if not (bull_fvg or bear_fvg or bull_ob or bear_ob):
        unlock.append("Need fresh FVG and/or Order Block near price")

    # --- 4) Structure BOS/CHoCH continuation ---
    if struct and ("BOS=bullish" in detail_s or "CHoCH=bullish" in detail_s):
        if h1_bias != Bias.BEARISH:
            hits.append(
                PlaybookHit(
                    "Bullish BOS/CHoCH Continuation",
                    Bias.BULLISH,
                    12.0,
                    [struct.detail],
                )
            )
    if struct and ("BOS=bearish" in detail_s or "CHoCH=bearish" in detail_s):
        if h1_bias != Bias.BULLISH:
            hits.append(
                PlaybookHit(
                    "Bearish BOS/CHoCH Continuation",
                    Bias.BEARISH,
                    12.0,
                    [struct.detail],
                )
            )
    if struct and "BOS=" not in detail_s and "CHoCH=" not in detail_s:
        unlock.append("Wait for clear BOS or CHoCH on M15/H1")

    # --- 5) Killzone + PA ---
    killzone = session_name in ("London", "NewYork", "London-NY Overlap")
    if killzone and pa and pa.passed and pa.bias != Bias.NEUTRAL:
        if h1_bias == Bias.NEUTRAL or h1_bias == pa.bias:
            hits.append(
                PlaybookHit(
                    f"Killzone PA ({session_name})",
                    pa.bias,
                    10.0,
                    [f"Session={session_name}", f"PA={pa.detail}"],
                )
            )
    if not killzone:
        unlock.append("Prefer London / NY / Overlap killzone for XAUUSD")

    # --- 6) Support bounce / resistance reject near strong S/R ---
    strong_s = [
        z
        for z in narrative.zones
        if z.kind == "support"
        and z.score >= 85
        and abs(((z.top + z.bottom) / 2) - mid) / atr <= 1.0
    ]
    strong_r = [
        z
        for z in narrative.zones
        if z.kind == "resistance"
        and z.score >= 85
        and abs(((z.top + z.bottom) / 2) - mid) / atr <= 1.0
    ]
    if strong_s and pa and pa.bias == Bias.BULLISH:
        hits.append(
            PlaybookHit(
                "Strong Support + Bullish PA",
                Bias.BULLISH,
                12.0,
                [strong_s[0].label, pa.detail],
            )
        )
    if strong_r and pa and pa.bias == Bias.BEARISH:
        hits.append(
            PlaybookHit(
                "Strong Resistance + Bearish PA",
                Bias.BEARISH,
                12.0,
                [strong_r[0].label, pa.detail],
            )
        )

    # --- 7) Premium / Discount array (world standard SMC filter) ---
    if h1_bias == Bias.BULLISH and in_discount and (pa is None or pa.bias != Bias.BEARISH):
        hits.append(
            PlaybookHit(
                "Discount Array Long",
                Bias.BULLISH,
                11.0,
                ["H1 bullish + price in discount — institutional long array"],
            )
        )
    elif h1_bias == Bias.BEARISH and in_premium and (pa is None or pa.bias != Bias.BULLISH):
        hits.append(
            PlaybookHit(
                "Premium Array Short",
                Bias.BEARISH,
                11.0,
                ["H1 bearish + price in premium — institutional short array"],
            )
        )
    elif h1_bias == Bias.BULLISH and in_premium:
        unlock.append("H1 bullish but price still in premium — wait for discount pullback")
    elif h1_bias == Bias.BEARISH and in_discount:
        unlock.append("H1 bearish but price still in discount — wait for premium rally")

    # --- 8) Displacement + FVG return ---
    disp = _detect_displacement(setup_df)
    if disp == Bias.BULLISH and bull_fvg:
        near = any(abs(((z.top + z.bottom) / 2) - mid) / atr <= 1.5 for z in bull_fvg)
        if near:
            hits.append(
                PlaybookHit(
                    "Displacement + Bull FVG",
                    Bias.BULLISH,
                    15.0,
                    ["Impulsive bullish displacement", "Price returning into bullish FVG"],
                )
            )
    if disp == Bias.BEARISH and bear_fvg:
        near = any(abs(((z.top + z.bottom) / 2) - mid) / atr <= 1.5 for z in bear_fvg)
        if near:
            hits.append(
                PlaybookHit(
                    "Displacement + Bear FVG",
                    Bias.BEARISH,
                    15.0,
                    ["Impulsive bearish displacement", "Price returning into bearish FVG"],
                )
            )
    if disp == Bias.NEUTRAL:
        unlock.append("Wait for displacement candle (≥1.4×ATR) then FVG return")

    # --- 9) Multi-TF structure stack ---
    if h1_bias != Bias.NEUTRAL and m15_bias == h1_bias:
        boost = 13.0 if (h1_bias == Bias.BULLISH and in_discount) or (
            h1_bias == Bias.BEARISH and in_premium
        ) else 9.0
        hits.append(
            PlaybookHit(
                "H1+M15 Structure Stack",
                h1_bias,
                boost,
                [f"H1={h1_bias.value} M15={m15_bias.value} aligned"],
            )
        )
    else:
        unlock.append("Need H1 and M15 structure to agree")

    # --- 10) Asian range raid / London expansion ---
    ar = _asian_range(setup_df)
    if ar is not None and setup_df is not None and len(setup_df) >= 24:
        a_hi, a_lo = ar
        width = max(a_hi - a_lo, atr * 0.3)
        recent_low = float(setup_df["low"].tail(24).min())
        recent_high = float(setup_df["high"].tail(24).max())
        # Raid below Asian low then reclaim → bullish Judas/AMD manipulation
        if recent_low <= a_lo + atr * 0.05 and mid > a_lo + atr * 0.1:
            if h1_bias != Bias.BEARISH and killzone:
                hits.append(
                    PlaybookHit(
                        "Asian Low Raid + Reclaim",
                        Bias.BULLISH,
                        14.0,
                        [f"Asian range {a_lo:.2f}-{a_hi:.2f}", "Judas/AMD-style liquidity grab"],
                    )
                )
        if recent_high >= a_hi - atr * 0.05 and mid < a_hi - atr * 0.1:
            if h1_bias != Bias.BULLISH and killzone:
                hits.append(
                    PlaybookHit(
                        "Asian High Raid + Reject",
                        Bias.BEARISH,
                        14.0,
                        [f"Asian range {a_lo:.2f}-{a_hi:.2f}", "Judas/AMD-style liquidity grab"],
                    )
                )
        # Clear break & hold of Asian range in HTF direction
        if h1_bias == Bias.BULLISH and mid > a_hi + width * 0.15 and killzone:
            hits.append(
                PlaybookHit(
                    "London Expansion Long",
                    Bias.BULLISH,
                    10.0,
                    ["Price holding above Asian high into London expansion"],
                )
            )
        if h1_bias == Bias.BEARISH and mid < a_lo - width * 0.15 and killzone:
            hits.append(
                PlaybookHit(
                    "London Expansion Short",
                    Bias.BEARISH,
                    10.0,
                    ["Price holding below Asian low into London expansion"],
                )
            )

    # Filter hits that fight hard H1 bias
    if h1_bias != Bias.NEUTRAL:
        hits = [h for h in hits if h.bias == h1_bias]

    # Cap total boost — quality stack, not score inflation
    hits.sort(key=lambda h: -h.boost)
    total = 0.0
    kept: list[PlaybookHit] = []
    for h in hits:
        if total >= 30:
            break
        if kept and kept[0].bias != h.bias:
            continue
        kept.append(h)
        total += h.boost
    total = min(30.0, total)

    best = kept[0] if kept else None
    summary = (
        f"playbooks={len(kept)} boost=+{total:.0f} best={best.name if best else 'none'}"
    )
    if not kept:
        unlock.append("No high-probability playbook active — stand aside")

    # Deduplicate unlock hints
    seen: set[str] = set()
    uniq: list[str] = []
    for u in unlock:
        if u not in seen:
            seen.add(u)
            uniq.append(u)

    return PlaybookResult(kept, best, total, summary, uniq[:6])
