"""
High-probability institutional playbooks.

Based on widely used SMC / ICT-style confluence patterns that historically
filter for higher-quality setups (not guarantees):

1. Liquidity Sweep + Reclaim (in HTF direction)
2. OTE / Golden Pocket (0.618–0.786) in discount/premium
3. Order Block + FVG overlap
4. CHoCH / BOS continuation after pullback
5. Killzone session + decisive price action

A playbook match BOOSTS probability; it never forces a trade alone.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from atlas.institutional.models import Bias, FibLevel, MarketNarrative, ModuleScore, Zone


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


def evaluate_playbooks(
    narrative: MarketNarrative,
    modules: list[ModuleScore],
    consensus: Bias,
    session_name: str,
    h1_bias: Bias,
) -> PlaybookResult:
    hits: list[PlaybookHit] = []
    unlock: list[str] = []

    liq = _mod(modules, "liquidity")
    struct = _mod(modules, "market_structure")
    fvg = _mod(modules, "fair_value_gaps")
    ob = _mod(modules, "order_blocks")
    fib = _mod(modules, "fibonacci")
    pa = _mod(modules, "price_action")
    trend = _mod(modules, "trend")
    sess = _mod(modules, "session")

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
        if 0.60 <= lv.ratio <= 0.79 and abs(lv.price - narrative.mid) / max(narrative.atr_m15, 1e-9) <= 1.2
    ]
    if ote and struct:
        near_ote = any(
            abs(lv.price - narrative.mid) / max(narrative.atr_m15, 1e-9) <= 0.85 for lv in ote
        )
        in_discount = "discount" in (struct.detail or "")
        in_premium = "premium" in (struct.detail or "")
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

    # --- 3) OB + FVG overlap ---
    bull_fvg = [z for z in narrative.zones if z.kind == "bullish_fvg" and z.fresh]
    bear_fvg = [z for z in narrative.zones if z.kind == "bearish_fvg" and z.fresh]
    bull_ob = [z for z in narrative.zones if z.kind == "bullish_ob" and z.fresh]
    bear_ob = [z for z in narrative.zones if z.kind == "bearish_ob" and z.fresh]
    if bull_fvg and bull_ob:
        hits.append(
            PlaybookHit(
                "Bullish OB+FVG Confluence",
                Bias.BULLISH,
                16.0,
                ["Fresh bullish order block overlapping FVG"],
            )
        )
    if bear_fvg and bear_ob:
        hits.append(
            PlaybookHit(
                "Bearish OB+FVG Confluence",
                Bias.BEARISH,
                16.0,
                ["Fresh bearish order block overlapping FVG"],
            )
        )
    if not (bull_fvg or bear_fvg or bull_ob or bear_ob):
        unlock.append("Need fresh FVG and/or Order Block near price")

    # --- 4) Structure BOS/CHoCH continuation ---
    if struct and ("BOS=bullish" in (struct.detail or "") or "CHoCH=bullish" in (struct.detail or "")):
        if h1_bias != Bias.BEARISH:
            hits.append(
                PlaybookHit(
                    "Bullish BOS/CHoCH Continuation",
                    Bias.BULLISH,
                    12.0,
                    [struct.detail],
                )
            )
    if struct and ("BOS=bearish" in (struct.detail or "") or "CHoCH=bearish" in (struct.detail or "")):
        if h1_bias != Bias.BULLISH:
            hits.append(
                PlaybookHit(
                    "Bearish BOS/CHoCH Continuation",
                    Bias.BEARISH,
                    12.0,
                    [struct.detail],
                )
            )
    if struct and "BOS=" not in (struct.detail or "") and "CHoCH=" not in (struct.detail or ""):
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
        and abs(((z.top + z.bottom) / 2) - narrative.mid) / max(narrative.atr_m15, 1e-9) <= 1.0
    ]
    strong_r = [
        z
        for z in narrative.zones
        if z.kind == "resistance"
        and z.score >= 85
        and abs(((z.top + z.bottom) / 2) - narrative.mid) / max(narrative.atr_m15, 1e-9) <= 1.0
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

    # Filter hits that fight hard H1 bias
    if h1_bias != Bias.NEUTRAL:
        hits = [h for h in hits if h.bias == h1_bias or h1_bias == Bias.NEUTRAL]

    # Cap total boost
    hits.sort(key=lambda h: -h.boost)
    total = 0.0
    kept: list[PlaybookHit] = []
    for h in hits:
        if total >= 28:
            break
        # Avoid stacking opposite playbooks
        if kept and kept[0].bias != h.bias:
            continue
        kept.append(h)
        total += h.boost
    total = min(28.0, total)

    best = kept[0] if kept else None
    summary = (
        f"playbooks={len(kept)} boost=+{total:.0f} best={best.name if best else 'none'}"
    )
    if not kept:
        unlock.append("No high-probability playbook active — stand aside")

    return PlaybookResult(kept, best, total, summary, unlock[:6])
