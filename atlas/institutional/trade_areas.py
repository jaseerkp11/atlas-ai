"""
Best Trade Areas — scored chart levels for manual TradingView confirmation.

Ranks BUY-side / SELL-side areas from live-analysis modules:
  - Buy-side liquidity (equal lows / swing lows / daily-weekly lows)
  - Sell-side liquidity (equal highs / swing highs / daily-weekly highs)
  - Unfilled / fresh Fair Value Gaps
  - Unfilled / fresh Order Blocks
  - Strong S/R near price
  - Fibonacci OTE (0.618–0.786)

Scores use distance-to-mid, freshness, module strength, and HTF alignment.
These are analysis levels for personal chart checks — not auto-orders.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from atlas.institutional.liquidity_engine import LiquidityPool, LiquidityReport
from atlas.institutional.models import Bias, FibLevel, MarketNarrative, Zone


@dataclass
class TradeArea:
    side: str  # BUY | SELL
    kind: str  # buy_liquidity | sell_liquidity | bullish_fvg | bearish_fvg | bullish_ob | bearish_ob | support | resistance | fib_ote
    price_low: float
    price_high: float
    mid_price: float
    score: float  # 0–100
    distance_atr: float
    fresh_unfilled: bool
    reasons: list[str] = field(default_factory=list)
    label: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class TradeAreasReport:
    areas: list[TradeArea]
    buy_best: list[TradeArea]
    sell_best: list[TradeArea]
    summary: str

    def as_dict(self) -> dict:
        return {
            "areas": [a.as_dict() for a in self.areas],
            "buy_best": [a.as_dict() for a in self.buy_best],
            "sell_best": [a.as_dict() for a in self.sell_best],
            "summary": self.summary,
        }


def _dist_atr(price: float, mid: float, atr: float) -> float:
    return abs(price - mid) / max(atr, 1e-9)


def _proximity_boost(dist: float) -> float:
    """Closer to price = more actionable for chart watch."""
    if dist <= 0.35:
        return 18.0
    if dist <= 0.85:
        return 12.0
    if dist <= 1.5:
        return 6.0
    if dist <= 3.0:
        return 2.0
    return -8.0


def _htf_align_boost(side: str, overall: Bias, h1: Bias) -> float:
    bias = h1 if h1 != Bias.NEUTRAL else overall
    if bias == Bias.NEUTRAL:
        return 0.0
    if side == "BUY" and bias == Bias.BULLISH:
        return 10.0
    if side == "SELL" and bias == Bias.BEARISH:
        return 10.0
    if side == "BUY" and bias == Bias.BEARISH:
        return -12.0
    if side == "SELL" and bias == Bias.BULLISH:
        return -12.0
    return 0.0


def _from_zone(
    z: Zone,
    mid: float,
    atr: float,
    overall: Bias,
    h1: Bias,
) -> TradeArea | None:
    kind = z.kind
    if kind in ("bullish_fvg", "bullish_ob", "support"):
        side = "BUY"
    elif kind in ("bearish_fvg", "bearish_ob", "resistance"):
        side = "SELL"
    else:
        return None

    center = (z.top + z.bottom) / 2.0
    dist = _dist_atr(center, mid, atr)
    # Prefer unfilled / fresh for FVG & OB
    is_imbalance = kind.endswith("_fvg") or kind.endswith("_ob")
    fresh = bool(z.fresh) if is_imbalance else True
    if is_imbalance and not fresh:
        return None

    score = float(z.score)
    score += _proximity_boost(dist)
    score += _htf_align_boost(side, overall, h1)
    if is_imbalance and fresh:
        score += 8.0
    if z.reactions:
        score += min(8.0, z.reactions * 2.0)
    score = max(0.0, min(100.0, score))

    reasons: list[str] = []
    if kind == "bullish_fvg":
        reasons.append("Unfilled bullish FVG (buy-side imbalance)")
    elif kind == "bearish_fvg":
        reasons.append("Unfilled bearish FVG (sell-side imbalance)")
    elif kind == "bullish_ob":
        reasons.append("Fresh bullish Order Block (unmitigated)")
    elif kind == "bearish_ob":
        reasons.append("Fresh bearish Order Block (unmitigated)")
    elif kind == "support":
        reasons.append(f"Strong support zone score={z.score:.0f}")
    elif kind == "resistance":
        reasons.append(f"Strong resistance zone score={z.score:.0f}")
    reasons.append(f"Range {z.bottom:.2f}-{z.top:.2f} | {dist:.2f} ATR from mid")
    if side == "BUY" and h1 == Bias.BULLISH:
        reasons.append("Aligned with H1 bullish bias")
    if side == "SELL" and h1 == Bias.BEARISH:
        reasons.append("Aligned with H1 bearish bias")

    return TradeArea(
        side=side,
        kind=kind,
        price_low=min(float(z.bottom), float(z.top)),
        price_high=max(float(z.bottom), float(z.top)),
        mid_price=float(center),
        score=score,
        distance_atr=dist,
        fresh_unfilled=fresh,
        reasons=reasons,
        label=z.label or kind,
    )


def _from_liquidity(
    pool: LiquidityPool,
    mid: float,
    atr: float,
    overall: Bias,
    h1: Bias,
) -> TradeArea | None:
    kind = pool.kind
    # Buy-side liquidity sits below (equal lows / lows) — swept then reclaim often long
    # Sell-side liquidity sits above (equal highs / highs)
    if kind in ("equal_lows", "swing_low_liq", "daily_low", "weekly_low"):
        side = "BUY"
        area_kind = "buy_liquidity"
        reason0 = "Buy-side liquidity (stops below lows)"
    elif kind in ("equal_highs", "swing_high_liq", "daily_high", "weekly_high"):
        side = "SELL"
        area_kind = "sell_liquidity"
        reason0 = "Sell-side liquidity (stops above highs)"
    else:
        return None

    p = float(pool.price)
    half = atr * 0.12
    dist = _dist_atr(p, mid, atr)
    if dist > 4.5:
        return None

    score = float(pool.significance)
    score += _proximity_boost(dist)
    score += _htf_align_boost(side, overall, h1)
    if kind.startswith("equal_"):
        score += 6.0
    if kind.startswith("weekly_"):
        score += 5.0
    score = max(0.0, min(100.0, score))

    return TradeArea(
        side=side,
        kind=area_kind,
        price_low=p - half,
        price_high=p + half,
        mid_price=p,
        score=score,
        distance_atr=dist,
        fresh_unfilled=True,
        reasons=[
            reason0,
            pool.label,
            f"{dist:.2f} ATR from mid — watch for sweep then reclaim/reject",
        ],
        label=pool.label,
    )


def _from_fib(
    lv: FibLevel,
    mid: float,
    atr: float,
    overall: Bias,
    h1: Bias,
) -> TradeArea | None:
    if not (0.60 <= lv.ratio <= 0.79):
        return None
    # OTE direction follows H1 / overall
    bias = h1 if h1 != Bias.NEUTRAL else overall
    if bias == Bias.BEARISH:
        side = "SELL"
    elif bias == Bias.BULLISH:
        side = "BUY"
    else:
        return None

    p = float(lv.price)
    half = atr * 0.15
    dist = _dist_atr(p, mid, atr)
    if dist > 2.5:
        return None

    score = float(lv.score) + _proximity_boost(dist) + _htf_align_boost(side, overall, h1)
    if lv.confluence:
        score += 8.0
    score = max(0.0, min(100.0, score))

    return TradeArea(
        side=side,
        kind="fib_ote",
        price_low=p - half,
        price_high=p + half,
        mid_price=p,
        score=score,
        distance_atr=dist,
        fresh_unfilled=True,
        reasons=[
            f"Fibonacci OTE {lv.ratio} @ {p:.2f} ({lv.timeframe})",
            f"Class={lv.classification} score={lv.score:.0f}",
            "Golden-pocket pullback area for chart confirmation",
        ],
        label=f"OTE {lv.ratio}",
    )


def build_trade_areas(
    narrative: MarketNarrative,
    liquidity: LiquidityReport | None = None,
    h1_bias: Bias = Bias.NEUTRAL,
    max_per_side: int = 5,
) -> TradeAreasReport:
    mid = narrative.mid
    atr = max(narrative.atr_m15, 1e-9)
    overall = narrative.overall_bias
    areas: list[TradeArea] = []

    for z in narrative.zones:
        a = _from_zone(z, mid, atr, overall, h1_bias)
        if a is not None:
            areas.append(a)

    if liquidity is not None:
        for pool in liquidity.pools:
            a = _from_liquidity(pool, mid, atr, overall, h1_bias)
            if a is not None:
                areas.append(a)

    for lv in narrative.fib_levels:
        a = _from_fib(lv, mid, atr, overall, h1_bias)
        if a is not None:
            areas.append(a)

    # De-dupe near-identical prices (same side, within 0.2 ATR)
    areas.sort(key=lambda x: -x.score)
    kept: list[TradeArea] = []
    for a in areas:
        clash = False
        for k in kept:
            if k.side != a.side:
                continue
            if abs(k.mid_price - a.mid_price) / atr < 0.2:
                # Keep higher score; merge reason if different kind
                if a.kind not in k.kind and a.score >= k.score - 5:
                    k.reasons.append(f"+ overlap {a.kind} ({a.label})")
                    k.score = min(100.0, k.score + 3.0)
                clash = True
                break
        if not clash:
            kept.append(a)

    buy = [a for a in kept if a.side == "BUY"][:max_per_side]
    sell = [a for a in kept if a.side == "SELL"][:max_per_side]
    top = sorted(buy + sell, key=lambda x: -x.score)

    summary = (
        f"areas={len(top)} buy={len(buy)} sell={len(sell)} "
        f"best={' / '.join(f'{a.side}@{a.mid_price:.2f}({a.score:.0f})' for a in top[:3]) or 'none'}"
    )
    return TradeAreasReport(top, buy, sell, summary)


def render_trade_areas_block(report: TradeAreasReport | None, mid: float = 0.0) -> list[str]:
    """Last dashboard lines — chart-check summary for TradingView."""
    lines: list[str] = []
    lines.append("-" * 72)
    lines.append("  BEST TRADE AREAS  (mark on TradingView — verify live before deciding)")
    lines.append("-" * 72)
    if report is None or not report.areas:
        lines.append("  No high-quality trade areas near price right now.")
        lines.append("  Stand aside until liquidity / FVG / OB / OTE forms near mid.")
        return lines

    lines.append(f"  Mid reference: {mid:.3f}  |  {report.summary}")
    lines.append("")
    lines.append("  BUY-SIDE AREAS (long interest / buy liquidity / bull FVG-OB)")
    if report.buy_best:
        for i, a in enumerate(report.buy_best, 1):
            uf = "UNFILLED" if a.fresh_unfilled else "touched"
            lines.append(
                f"  {i}. BUY  {a.price_low:.2f}-{a.price_high:.2f}  "
                f"@ {a.mid_price:.2f}  score={a.score:.0f}/100  "
                f"{a.kind}  {uf}  dist={a.distance_atr:.2f}ATR"
            )
            for r in a.reasons[:3]:
                lines.append(f"      · {r}")
    else:
        lines.append("     (none ranked)")

    lines.append("")
    lines.append("  SELL-SIDE AREAS (short interest / sell liquidity / bear FVG-OB)")
    if report.sell_best:
        for i, a in enumerate(report.sell_best, 1):
            uf = "UNFILLED" if a.fresh_unfilled else "touched"
            lines.append(
                f"  {i}. SELL {a.price_low:.2f}-{a.price_high:.2f}  "
                f"@ {a.mid_price:.2f}  score={a.score:.0f}/100  "
                f"{a.kind}  {uf}  dist={a.distance_atr:.2f}ATR"
            )
            for r in a.reasons[:3]:
                lines.append(f"      · {r}")
    else:
        lines.append("     (none ranked)")

    lines.append("")
    lines.append("  How to use: plot these prices on TradingView, wait for reaction")
    lines.append("  (sweep/reclaim or FVG/OB fill), then decide yourself — bot does not force entry.")
    return lines
