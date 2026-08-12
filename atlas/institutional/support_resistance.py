"""Support & Resistance AI — reaction quality, freshness, confluence scoring."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from atlas.analysis.levels import SRMap, build_sr_map
from atlas.analysis.volatility import latest_atr
from atlas.institutional.models import Bias, ModuleScore, Zone, ZoneStrength


def _classify(score: float) -> ZoneStrength:
    if score >= 85:
        return ZoneStrength.VERY_STRONG
    if score >= 70:
        return ZoneStrength.STRONG
    if score >= 45:
        return ZoneStrength.NEUTRAL
    if score >= 25:
        return ZoneStrength.WEAK
    return ZoneStrength.VERY_WEAK


@dataclass
class SRReport:
    zones: list[Zone]
    score: float
    bias: Bias
    summary: str
    module: ModuleScore


def analyze_support_resistance(
    symbol: str,
    frames: dict[str, pd.DataFrame],
    mid: float,
    atr: float,
) -> SRReport:
    sr: SRMap = build_sr_map(symbol, frames, mid=mid)
    zones: list[Zone] = []
    atr = atr or 1.0

    # Resistances above, supports below
    for lv in getattr(sr, "resistances", []) or []:
        price = float(getattr(lv, "price", lv) if not isinstance(lv, (int, float)) else lv)
        reactions = int(getattr(lv, "touches", getattr(lv, "count", 1)) or 1)
        dist = abs(price - mid) / atr
        score = min(100.0, 40 + reactions * 12 + (10 if dist < 3 else 0) - max(0, dist - 5) * 3)
        zones.append(
            Zone(
                kind="resistance",
                top=price + atr * 0.1,
                bottom=price - atr * 0.1,
                strength=_classify(score),
                score=score,
                timeframe="+".join(getattr(lv, "timeframes", []) or ["MULTI"]),
        reactions=reactions,
        fresh=dist > 0.3,
        label=f"R@{price:.2f}",
            )
        )
    for lv in getattr(sr, "supports", []) or []:
        price = float(getattr(lv, "price", lv) if not isinstance(lv, (int, float)) else lv)
        reactions = int(getattr(lv, "touches", getattr(lv, "count", 1)) or 1)
        dist = abs(price - mid) / atr
        score = min(100.0, 40 + reactions * 12 + (10 if dist < 3 else 0) - max(0, dist - 5) * 3)
        zones.append(
            Zone(
                kind="support",
                top=price + atr * 0.1,
                bottom=price - atr * 0.1,
                strength=_classify(score),
                score=score,
                timeframe="+".join(getattr(lv, "timeframes", []) or ["MULTI"]),
                reactions=reactions,
                fresh=dist > 0.3,
                label=f"S@{price:.2f}",
            )
        )

    # Also parse chart levels if SRMap exposes unified levels
    if not zones and hasattr(sr, "levels"):
        for lv in sr.levels:  # type: ignore[attr-defined]
            price = float(lv.price)
            kind = "resistance" if price >= mid else "support"
            score = 50.0
            zones.append(
                Zone(
                    kind=kind,
                    top=price + atr * 0.1,
                    bottom=price - atr * 0.1,
                    strength=_classify(score),
                    score=score,
                    timeframe="MULTI",
                    reactions=1,
                    label=f"{kind[0].upper()}@{price:.2f}",
                )
            )

    near = [z for z in zones if abs(((z.top + z.bottom) / 2) - mid) / atr <= 2.5]
    avg = sum(z.score for z in near) / len(near) if near else (sum(z.score for z in zones[:5]) / max(1, min(5, len(zones))))
    bias = Bias.NEUTRAL
    # Near support in discount → bullish lean; near resistance in premium → bearish
    supports_near = [z for z in near if z.kind == "support"]
    resists_near = [z for z in near if z.kind == "resistance"]
    if supports_near and not resists_near:
        bias = Bias.BULLISH
    elif resists_near and not supports_near:
        bias = Bias.BEARISH

    summary = f"zones={len(zones)} near={len(near)} avg={avg:.0f}"
    mod = ModuleScore("support_resistance", avg, 10, bias, summary, avg >= 55)
    return SRReport(zones=zones, score=avg, bias=bias, summary=summary, module=mod)
