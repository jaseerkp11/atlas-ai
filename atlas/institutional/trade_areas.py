"""
Best Trade Areas — scored chart levels for manual TradingView confirmation.

Ranks BUY-side / SELL-side areas from live-analysis modules:
  - Buy-side liquidity (equal lows / swing lows / daily-weekly lows)
  - Sell-side liquidity (equal highs / swing highs / daily-weekly highs)
  - Unfilled / fresh Fair Value Gaps
  - Unfilled / fresh Order Blocks
  - Strong S/R near price
  - Fibonacci OTE (0.618–0.786)
  - Explicit H4/H1 major highs & lows (range extremes + confirmed swings)

Scores use distance-to-mid, freshness, module strength, and HTF alignment.
These are analysis levels for personal chart checks — not auto-orders.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import pandas as pd

from atlas.analysis.structure import find_swing_highs, find_swing_lows
from atlas.institutional.liquidity_engine import LiquidityPool, LiquidityReport
from atlas.institutional.models import Bias, FibLevel, MarketNarrative, Zone


@dataclass
class TradeArea:
    side: str  # BUY | SELL
    kind: str  # buy_liquidity | sell_liquidity | bullish_fvg | bearish_fvg | bullish_ob | bearish_ob | support | resistance | fib_ote | htf_support | htf_resistance
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


def _htf_major_areas(
    frames: dict[str, Any] | None,
    mid: float,
    atr: float,
    overall: Bias,
    h1: Bias,
) -> list[TradeArea]:
    """
    Explicit previous H4/H1 highs & lows as major S/R magnets.

    Clustered multi-TF S/R already exists; this adds first-class levels for:
      - recent H4/H1 range high/low (catches impulse extremes like a 4180 high)
      - last confirmed H4/H1 swing high/low
      - previous completed H4/H1 candle high/low
    Kept even when somewhat distant so they remain TP / danger anchors.
    """
    if not frames:
        return []

    out: list[TradeArea] = []
    specs = (
        ("H4", 2, 20, 94.0, 14.0),  # tf, swing_lb, range_bars, base_score, max_dist_atr
        ("H1", 3, 24, 88.0, 10.0),
    )

    for tf, swing_lb, range_bars, base, max_dist in specs:
        df = frames.get(tf)
        if df is None or not isinstance(df, pd.DataFrame) or len(df) < max(30, swing_lb * 2 + 5):
            continue
        # Prefer closed bars so the forming candle does not invent a "major"
        closed = df.iloc[:-1] if len(df) >= 2 else df
        if len(closed) < 10:
            continue

        window = closed.tail(min(range_bars, len(closed)))
        range_high = float(window["high"].max())
        range_low = float(window["low"].min())
        last = closed.iloc[-1]
        prev_high = float(last["high"])
        prev_low = float(last["low"])

        sh = find_swing_highs(closed, lookback=swing_lb)
        sl = find_swing_lows(closed, lookback=swing_lb)
        swing_high = float(sh[-1].price) if sh else None
        swing_low = float(sl[-1].price) if sl else None

        candidates: list[tuple[str, float, str, float]] = [
            ("high", range_high, f"{tf} major high (recent range)", base + 3.0),
            ("low", range_low, f"{tf} major low (recent range)", base + 3.0),
            ("high", prev_high, f"Prev {tf} candle high", base - 2.0),
            ("low", prev_low, f"Prev {tf} candle low", base - 2.0),
        ]
        if swing_high is not None:
            candidates.append(("high", swing_high, f"{tf} swing high", base))
        if swing_low is not None:
            candidates.append(("low", swing_low, f"{tf} swing low", base))

        # De-dupe within TF (~0.25 ATR), keep highest score label
        kept_local: list[tuple[str, float, str, float]] = []
        for kind_hl, price, label, sc in sorted(candidates, key=lambda x: -x[3]):
            if any(abs(price - p) / atr < 0.25 and kind_hl == k for k, p, _, _ in kept_local):
                continue
            kept_local.append((kind_hl, price, label, sc))

        half = atr * (0.18 if tf == "H4" else 0.12)
        for kind_hl, price, label, sc in kept_local:
            dist = _dist_atr(price, mid, atr)
            if dist > max_dist:
                continue
            side = "SELL" if kind_hl == "high" else "BUY"
            area_kind = "htf_resistance" if kind_hl == "high" else "htf_support"
            score = sc + _proximity_boost(dist) + _htf_align_boost(side, overall, h1)
            # HTF majors stay useful as targets even against bias (don't crush score)
            if side == "SELL" and h1 == Bias.BULLISH:
                score += 8.0  # undo most of against-bias penalty — still a magnet
            if side == "BUY" and h1 == Bias.BEARISH:
                score += 8.0
            score = max(0.0, min(100.0, score))
            out.append(
                TradeArea(
                    side=side,
                    kind=area_kind,
                    price_low=price - half,
                    price_high=price + half,
                    mid_price=price,
                    score=score,
                    distance_atr=dist,
                    fresh_unfilled=True,
                    reasons=[
                        label,
                        f"Major {tf} {'resistance' if kind_hl == 'high' else 'support'} "
                        f"@ {price:.2f} ({dist:.2f} ATR from mid)",
                        "Use as TP / danger / invalidation anchor on TradingView",
                    ],
                    label=label,
                )
            )
    return out


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
    frames: dict[str, Any] | None = None,
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

    areas.extend(_htf_major_areas(frames, mid, atr, overall, h1_bias))

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
                    # Promote to HTF kind when a major overlaps a weaker zone
                    if a.kind.startswith("htf_") and not k.kind.startswith("htf_"):
                        k.kind = a.kind
                        k.label = a.label or k.label
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
    lines.append("  BUY-SIDE AREAS (long interest / buy liquidity / bull FVG-OB / H4-H1 lows)")
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
    lines.append("  SELL-SIDE AREAS (short interest / sell liquidity / bear FVG-OB / H4-H1 highs)")
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
