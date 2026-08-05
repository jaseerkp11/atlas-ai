"""
Manual high-probability scanner — analysis only (no auto trade).

Builds graded BUY/SELL setup cards for TradingView confirmation:
  - S/R, liquidity, unfilled FVG/OB, OTE
  - IF / THEN triggers (what must happen before YOU enter)
  - Invalidation + what to avoid
  - Grade A+ / A / B (quality filter — not a promised win rate)

Designed so the human decides and trades; the bot only maps the plan.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from atlas.institutional.models import Bias, MarketNarrative
from atlas.institutional.playbook_engine import PlaybookResult
from atlas.institutional.trade_areas import TradeArea, TradeAreasReport


@dataclass
class SetupCard:
    grade: str  # A+ | A | B
    side: str  # BUY | SELL
    zone_low: float
    zone_high: float
    focus_price: float
    kind: str
    score: float
    status: str  # READY_TO_WATCH | WAIT_FOR_TRIGGER | AVOID_NOW
    if_then: str
    confirm_on_tv: list[str]
    invalidation: str
    avoid: str
    reasons: list[str] = field(default_factory=list)
    # Manual risk map (from opposing high-prob areas) — not auto orders
    stop_loss: float = 0.0
    take_profit_1: float = 0.0
    take_profit_2: float = 0.0
    sl_label: str = ""
    tp1_label: str = ""
    tp2_label: str = ""
    reward_risk: float = 0.0

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class SideCompareBoard:
    """
    Raw structure probability — independent of H1 grade (A/B).

    Grade = bias filter (what to prefer). Board = which side has stronger
    support/resistance/FVG pressure right now (including correction risk).
    """

    buy_best: float = 0.0
    sell_best: float = 0.0
    buy_avg_top3: float = 0.0
    sell_avg_top3: float = 0.0
    buy_pressure: float = 0.0  # sum top-3 scores
    sell_pressure: float = 0.0
    support_best: float = 0.0
    resistance_best: float = 0.0
    bull_fvg_best: float = 0.0
    bear_fvg_best: float = 0.0
    bull_ob_best: float = 0.0
    bear_ob_best: float = 0.0
    buy_liq_best: float = 0.0
    sell_liq_best: float = 0.0
    htf_support_best: float = 0.0
    htf_resistance_best: float = 0.0
    lean: str = "BALANCED"  # BUY_LEAN | SELL_LEAN | BALANCED
    edge: float = 0.0  # buy_pressure - sell_pressure
    advice: str = ""
    rows: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class ManualScanReport:
    bias: str
    stance: str  # LONG_BIAS | SHORT_BIAS | NO_EDGE
    headline: str
    cards: list[SetupCard]
    buy_cards: list[SetupCard]
    sell_cards: list[SetupCard]
    watchlist: list[str]
    do_not: list[str]
    summary: str
    side_compare: SideCompareBoard | None = None

    def as_dict(self) -> dict:
        return {
            "bias": self.bias,
            "stance": self.stance,
            "headline": self.headline,
            "cards": [c.as_dict() for c in self.cards],
            "buy_cards": [c.as_dict() for c in self.buy_cards],
            "sell_cards": [c.as_dict() for c in self.sell_cards],
            "watchlist": self.watchlist,
            "do_not": self.do_not,
            "summary": self.summary,
            "side_compare": self.side_compare.as_dict() if self.side_compare else None,
        }


def _best_kind(cards: list[SetupCard], kinds: set[str]) -> float:
    vals = [c.score for c in cards if c.kind in kinds]
    return max(vals) if vals else 0.0


def _top_stats(cards: list[SetupCard], n: int = 3) -> tuple[float, float, float]:
    """Return (best, avg_top_n, pressure_sum_top_n)."""
    if not cards:
        return 0.0, 0.0, 0.0
    scores = sorted((c.score for c in cards), reverse=True)[:n]
    best = scores[0]
    avg = sum(scores) / len(scores)
    return best, round(avg, 1), round(sum(scores), 1)


def build_side_compare(
    buy_cards: list[SetupCard],
    sell_cards: list[SetupCard],
    stance: str,
) -> SideCompareBoard:
    """
    Compare buy vs sell structure scores (support/resistance/FVG/OB/liq/HTF).

    Intentionally ignores H1 grade so a B-grade sell can still outrank an A buy
    on raw zone pressure — that flags correction risk for the human trader.
    """
    buy_best, buy_avg, buy_pressure = _top_stats(buy_cards)
    sell_best, sell_avg, sell_pressure = _top_stats(sell_cards)
    edge = round(buy_pressure - sell_pressure, 1)

    if edge >= 12:
        lean = "BUY_LEAN"
    elif edge <= -12:
        lean = "SELL_LEAN"
    else:
        lean = "BALANCED"

    support_best = _best_kind(buy_cards, {"support", "htf_support"})
    resistance_best = _best_kind(sell_cards, {"resistance", "htf_resistance"})
    bull_fvg = _best_kind(buy_cards, {"bullish_fvg"})
    bear_fvg = _best_kind(sell_cards, {"bearish_fvg"})
    bull_ob = _best_kind(buy_cards, {"bullish_ob"})
    bear_ob = _best_kind(sell_cards, {"bearish_ob"})
    buy_liq = _best_kind(buy_cards, {"buy_liquidity"})
    sell_liq = _best_kind(sell_cards, {"sell_liquidity"})
    htf_sup = _best_kind(buy_cards, {"htf_support"})
    htf_res = _best_kind(sell_cards, {"htf_resistance"})

    def _pair(label: str, b: float, s: float) -> str:
        winner = "BUY" if b > s + 1 else ("SELL" if s > b + 1 else "TIE")
        return f"{label:<14} BUY {b:5.0f}  vs  SELL {s:5.0f}   → {winner}"

    rows = [
        _pair("Overall best", buy_best, sell_best),
        _pair("Top3 avg", buy_avg, sell_avg),
        _pair("Pressure Σ3", buy_pressure, sell_pressure),
        _pair("Support/Res", support_best, resistance_best),
        _pair("FVG", bull_fvg, bear_fvg),
        _pair("Order Block", bull_ob, bear_ob),
        _pair("Liquidity", buy_liq, sell_liq),
        _pair("HTF major", htf_sup, htf_res),
    ]

    if stance == "LONG_BIAS" and lean == "SELL_LEAN":
        advice = (
            "CORRECTION RISK: sell-side scores > buy — do not force A/A+ longs. "
            "Wait bounce confirm, or watch resistance rejection as a correction (smaller size)."
        )
    elif stance == "SHORT_BIAS" and lean == "BUY_LEAN":
        advice = (
            "BOUNCE RISK: buy-side scores > sell — do not force A/A+ shorts. "
            "Wait reject confirm, or watch support reclaim as a bounce (smaller size)."
        )
    elif stance == "LONG_BIAS" and lean == "BUY_LEAN":
        advice = "Aligned: buy structure leads — prefer graded BUY cards; sells stay targets/danger."
    elif stance == "SHORT_BIAS" and lean == "SELL_LEAN":
        advice = "Aligned: sell structure leads — prefer graded SELL cards; buys stay targets/danger."
    elif lean == "BALANCED":
        advice = (
            "Balanced pressure — wait for clearer zone reaction before choosing side; "
            "grade still follows H1 bias."
        )
    else:
        advice = (
            f"Structure lean={lean} while stance={stance} — use board to size/skip, "
            "not to ignore H1 grade filter on full-size entries."
        )

    return SideCompareBoard(
        buy_best=buy_best,
        sell_best=sell_best,
        buy_avg_top3=buy_avg,
        sell_avg_top3=sell_avg,
        buy_pressure=buy_pressure,
        sell_pressure=sell_pressure,
        support_best=support_best,
        resistance_best=resistance_best,
        bull_fvg_best=bull_fvg,
        bear_fvg_best=bear_fvg,
        bull_ob_best=bull_ob,
        bear_ob_best=bear_ob,
        buy_liq_best=buy_liq,
        sell_liq_best=sell_liq,
        htf_support_best=htf_sup,
        htf_resistance_best=htf_res,
        lean=lean,
        edge=edge,
        advice=advice,
        rows=rows,
    )


def _grade(
    area: TradeArea,
    h1: Bias,
    killzone: bool,
    agree_stack: bool,
    market_mid: float,
) -> tuple[str, str]:
    """Return (grade, status). Prefer pullback zones in bias direction."""
    with_trend = (area.side == "BUY" and h1 == Bias.BULLISH) or (
        area.side == "SELL" and h1 == Bias.BEARISH
    )
    against = (area.side == "BUY" and h1 == Bias.BEARISH) or (
        area.side == "SELL" and h1 == Bias.BULLISH
    )
    premium_kind = area.kind in (
        "buy_liquidity",
        "sell_liquidity",
        "bullish_fvg",
        "bearish_fvg",
        "bullish_ob",
        "bearish_ob",
        "fib_ote",
        "htf_support",
        "htf_resistance",
    )

    if against:
        return "B", "AVOID_NOW"

    # Path context vs live mid
    if area.side == "BUY":
        in_path = area.mid_price <= market_mid + 1e-9  # at/below = pullback path
        already_passed = area.price_high < market_mid  # fully below
        above_market = area.price_low > market_mid  # need rally/reclaim first
    else:
        in_path = area.mid_price >= market_mid - 1e-9
        already_passed = area.price_low > market_mid
        above_market = area.price_high < market_mid

    near = area.distance_atr <= 1.6
    strong = area.score >= 82

    if with_trend and above_market and near:
        # Zone not yet reached — reclaim/reach plan (BUY above mid, or SELL below mid)
        status = "WAIT_FOR_RECLAIM" if area.side == "BUY" else "WAIT_FOR_ZONE"
        return ("A" if strong else "B"), status

    elite = (
        with_trend
        and in_path
        and near
        and strong
        and premium_kind
        and killzone
        and agree_stack
        and area.score >= 88
    )
    if elite:
        return "A+", "WAIT_FOR_TRIGGER"
    if with_trend and in_path and near and area.score >= 75:
        return "A", "WAIT_FOR_TRIGGER" if premium_kind or area.score >= 85 else "READY_TO_WATCH"
    # Distant already-passed zones: watch only — do not inflate to A trigger
    if with_trend and already_passed and area.score >= 80:
        if near:
            return "A", "WAIT_FOR_TRIGGER"
        return "B", "READY_TO_WATCH"
    if with_trend and area.score >= 70:
        if in_path and near:
            return "A", "READY_TO_WATCH"
        if above_market:
            return "B", ("WAIT_FOR_RECLAIM" if area.side == "BUY" else "WAIT_FOR_ZONE")
        return "B", "READY_TO_WATCH"
    return "B", "READY_TO_WATCH"


def _if_then(area: TradeArea, market_mid: float) -> tuple[str, list[str], str, str]:
    """
    Plain-English plan that is hard to misunderstand.
    Format: WAIT | CONFIRM | THEN | NOW
    """
    lo, hi, focus = area.price_low, area.price_high, area.mid_price
    kind = area.kind

    def _pack(wait: str, confirm: str, then: str, now: str, cancel: str, avoid: str, tips: list[str]):
        ift = f"WAIT: {wait} | CONFIRM: {confirm} | THEN: {then} | NOW: {now}"
        return ift, tips, f"CANCEL if: {cancel}", f"AVOID: {avoid}"

    if area.side == "BUY":
        cancel = f"M5 closes below {lo - (hi - lo) * 0.35:.2f} and holds"
        avoid = "buying in the middle — only at the zone after confirmation"
        above = lo > market_mid
        if above and kind == "buy_liquidity":
            return _pack(
                f"price reclaims back above {focus:.2f}",
                f"M5 CLOSES above {focus:.2f} (wick alone is not enough)",
                "consider LONG",
                f"do NOTHING now (price {market_mid:.2f} is still below the zone)",
                cancel,
                avoid,
                ["RECLAIM plan — not a dip-buy"],
            )
        if above:
            return _pack(
                f"price reaches box {lo:.2f}-{hi:.2f}",
                "M5 holds/rejects upward inside the box",
                "consider LONG",
                f"do NOTHING now (zone is ABOVE market {market_mid:.2f})",
                cancel,
                avoid,
                ["Wait until price is inside the box"],
            )
        if kind == "buy_liquidity":
            return _pack(
                f"price sweeps UNDER {focus:.2f}",
                f"M5 CLOSES back ABOVE {focus:.2f}",
                "consider LONG",
                "do NOTHING until sweep + close-back-above happens",
                cancel,
                avoid,
                ["Order: 1) sweep  2) reclaim close  3) you decide"],
            )
        if kind in ("bullish_fvg", "bullish_ob"):
            return _pack(
                f"price taps {kind} box {lo:.2f}-{hi:.2f}",
                "M5 bullish pin or engulf from the box",
                "consider LONG",
                "do NOTHING until tap + bullish M5",
                cancel,
                avoid,
                [f"Draw box {lo:.2f}-{hi:.2f}"],
            )
        if kind == "fib_ote":
            return _pack(
                f"pullback reaches OTE {lo:.2f}-{hi:.2f}",
                "M5 bullish confirmation candle",
                "consider LONG",
                "do NOTHING until OTE is tagged",
                cancel,
                avoid,
                ["Fib 61.8-78.6"],
            )
        return _pack(
            f"price dips INTO support {lo:.2f}-{hi:.2f}",
            "M5 bullish pin or engulf from support",
            "consider LONG",
            f"do NOTHING unless price is inside {lo:.2f}-{hi:.2f}",
            cancel,
            avoid,
            [f"Mark support {lo:.2f}-{hi:.2f}"],
        )

    cancel = f"M5 closes above {hi + (hi - lo) * 0.35:.2f} and holds"
    avoid = "selling in the middle — only at the zone after confirmation"
    below = hi < market_mid
    if below and kind == "sell_liquidity":
        return _pack(
            f"price rejects back below {focus:.2f}",
            f"M5 CLOSES below {focus:.2f}",
            "consider SHORT",
            f"do NOTHING now (price {market_mid:.2f} is still above the zone)",
            cancel,
            avoid,
            ["REJECT plan — wait for close back below"],
        )
    if below:
        return _pack(
            f"price falls into box {lo:.2f}-{hi:.2f}",
            "M5 rejects downward inside the box",
            "consider SHORT",
            f"do NOTHING now (zone is BELOW market {market_mid:.2f})",
            cancel,
            avoid,
            ["Wait until price is inside the box"],
        )
    if kind == "sell_liquidity":
        return _pack(
            f"price sweeps ABOVE {focus:.2f}",
            f"M5 CLOSES back BELOW {focus:.2f}",
            "consider SHORT",
            "do NOTHING until sweep + close-back-below happens",
            cancel,
            avoid,
            ["Order: 1) sweep  2) reject close  3) you decide"],
        )
    if kind in ("bearish_fvg", "bearish_ob"):
        return _pack(
            f"price taps {kind} box {lo:.2f}-{hi:.2f}",
            "M5 bearish pin or engulf from the box",
            "consider SHORT",
            "do NOTHING until tap + bearish M5",
            cancel,
            avoid,
            [f"Draw box {lo:.2f}-{hi:.2f}"],
        )
    if kind == "fib_ote":
        return _pack(
            f"rally reaches OTE {lo:.2f}-{hi:.2f}",
            "M5 bearish confirmation candle",
            "consider SHORT",
            "do NOTHING until OTE is tagged",
            cancel,
            avoid,
            ["Fib 61.8-78.6"],
        )
    return _pack(
        f"price rallies INTO resistance {lo:.2f}-{hi:.2f}",
        "M5 bearish pin or engulf from resistance",
        "consider SHORT",
        f"do NOTHING unless price is inside {lo:.2f}-{hi:.2f}",
        cancel,
        avoid,
        [f"Mark resistance {lo:.2f}-{hi:.2f}"],
    )


def _plan_sl_tp(
    area: TradeArea,
    all_areas: list[TradeArea],
    atr: float,
) -> tuple[float, float, float, str, str, str, float]:
    """
    Map SL/TP from high-probability opposing areas.
    BUY: SL under entry zone; TP at resistance / sell-liquidity / bear FVG above.
    SELL: SL above entry zone; TP at support / buy-liquidity / bull FVG below.
    """
    atr = max(float(atr), 1e-9)
    lo, hi, focus = area.price_low, area.price_high, area.mid_price
    buffer = atr * 0.20

    def _rr(entry: float, sl: float, tp: float, side: str) -> float:
        if side == "BUY":
            risk = entry - sl
            reward = tp - entry
        else:
            risk = sl - entry
            reward = entry - tp
        if risk <= 1e-9:
            return 0.0
        return round(reward / risk, 2)

    if area.side == "BUY":
        sl = lo - buffer
        sl_label = f"below buy zone {lo:.2f} (−{buffer:.2f})"
        # Targets above: sell-side / resistance style areas
        above = sorted(
            [
                a
                for a in all_areas
                if a.side == "SELL" and a.mid_price > hi + atr * 0.15
            ],
            key=lambda a: a.mid_price,
        )
        # High-prob opposing areas within ~4 ATR; else any above
        near_band = [a for a in above if (a.mid_price - focus) / atr <= 4.0]
        pool = near_band if near_band else above
        # Prefer HTF major highs as magnets, then score, then nearer
        top = sorted(
            pool,
            key=lambda a: (
                0 if a.kind == "htf_resistance" else 1,
                -a.score,
                a.mid_price,
            ),
        )[:2]
        top = sorted(top, key=lambda a: a.mid_price)
        if top:
            tp1 = float(top[0].mid_price)
            tp1_label = f"{top[0].kind} @{tp1:.2f} (score={top[0].score:.0f})"
            if len(top) > 1:
                tp2 = float(top[1].mid_price)
                tp2_label = f"{top[1].kind} @{tp2:.2f} (score={top[1].score:.0f})"
            else:
                tp2 = tp1 + atr * 1.5
                tp2_label = f"extension +1.5 ATR @{tp2:.2f}"
        else:
            tp1 = focus + atr * 1.5
            tp2 = focus + atr * 2.5
            tp1_label = f"ATR target +1.5 @{tp1:.2f}"
            tp2_label = f"ATR target +2.5 @{tp2:.2f}"
        # Ensure TP above entry
        tp1 = max(tp1, hi + atr * 0.35)
        tp2 = max(tp2, tp1 + atr * 0.35)
        rr = _rr(focus, sl, tp1, "BUY")
        return sl, tp1, tp2, sl_label, tp1_label, tp2_label, rr

    # SELL
    sl = hi + buffer
    sl_label = f"above sell zone {hi:.2f} (+{buffer:.2f})"
    below = [
        a for a in all_areas if a.side == "BUY" and a.mid_price < lo - atr * 0.15
    ]
    near_band = [a for a in below if (focus - a.mid_price) / atr <= 4.0]
    pool = near_band if near_band else below
    # Prefer HTF major lows as magnets, then score, then nearer
    top = sorted(
        pool,
        key=lambda a: (
            0 if a.kind == "htf_support" else 1,
            -a.score,
            -a.mid_price,
        ),
    )[:2]
    top = sorted(top, key=lambda a: -a.mid_price)
    if top:
        tp1 = float(top[0].mid_price)
        tp1_label = f"{top[0].kind} @{tp1:.2f} (score={top[0].score:.0f})"
        if len(top) > 1:
            tp2 = float(top[1].mid_price)
            tp2_label = f"{top[1].kind} @{tp2:.2f} (score={top[1].score:.0f})"
        else:
            tp2 = tp1 - atr * 1.5
            tp2_label = f"extension −1.5 ATR @{tp2:.2f}"
    else:
        tp1 = focus - atr * 1.5
        tp2 = focus - atr * 2.5
        tp1_label = f"ATR target −1.5 @{tp1:.2f}"
        tp2_label = f"ATR target −2.5 @{tp2:.2f}"
    tp1 = min(tp1, lo - atr * 0.35)
    tp2 = min(tp2, tp1 - atr * 0.35)
    rr = _rr(focus, sl, tp1, "SELL")
    return sl, tp1, tp2, sl_label, tp1_label, tp2_label, rr


def build_manual_scan(
    narrative: MarketNarrative,
    trade_areas: TradeAreasReport | None,
    playbooks: PlaybookResult | None,
    h1_bias: Bias,
    h4_bias: Bias,
    m15_bias: Bias,
    session_name: str,
) -> ManualScanReport:
    killzone = session_name in ("London", "NewYork", "London-NY Overlap")
    agree_stack = h1_bias != Bias.NEUTRAL and h1_bias == m15_bias
    market_mid = float(narrative.mid)
    atr = max(float(narrative.atr_m15 or 0.0), 1e-9)

    if h1_bias == Bias.BULLISH:
        stance = "LONG_BIAS"
        headline = (
            "Prefer LONG ideas only at graded buy areas after IF/THEN confirmation. "
            "Treat sell zones as targets / danger, not shorts (unless H1 flips)."
        )
    elif h1_bias == Bias.BEARISH:
        stance = "SHORT_BIAS"
        headline = (
            "Prefer SHORT ideas only at graded sell areas after IF/THEN confirmation. "
            "Treat buy zones as targets / danger, not longs (unless H1 flips)."
        )
    else:
        stance = "NO_EDGE"
        headline = "No clear H1 bias — map levels, stand aside until structure clarifies."

    cards: list[SetupCard] = []
    areas = (trade_areas.areas if trade_areas else [])[:12]
    for a in areas:
        grade, status = _grade(a, h1_bias, killzone, agree_stack, market_mid)
        ift, conf, inv, avoid = _if_then(a, market_mid)
        reasons = list(a.reasons[:3])
        if playbooks and playbooks.best:
            if (a.side == "BUY" and playbooks.best.bias == Bias.BULLISH) or (
                a.side == "SELL" and playbooks.best.bias == Bias.BEARISH
            ):
                reasons.append(f"Supports playbook: {playbooks.best.name}")

        if status == "AVOID_NOW":
            ift = "SKIP ENTRY (against H1). " + ift

        lo = min(a.price_low, a.price_high)
        hi = max(a.price_low, a.price_high)
        # Use full area list for opposing TP targets (not only first 12 slice peer)
        pool = trade_areas.areas if trade_areas else [a]
        sl, tp1, tp2, sl_l, tp1_l, tp2_l, rr = _plan_sl_tp(a, pool, atr)
        # Soft-cap A+ when R:R to zone TP1 is unclear/weak (ATR fallback often ~1.x)
        if grade == "A+" and rr > 0 and rr < 1.5:
            grade = "A"
            reasons.append(f"Capped A+→A: R:R to TP1 only 1:{rr:.2f} (need ≥1.5)")
        cards.append(
            SetupCard(
                grade=grade,
                side=a.side,
                zone_low=lo,
                zone_high=hi,
                focus_price=a.mid_price,
                kind=a.kind,
                score=a.score,
                status=status,
                if_then=ift,
                confirm_on_tv=conf,
                invalidation=inv,
                avoid=avoid,
                reasons=reasons,
                stop_loss=sl,
                take_profit_1=tp1,
                take_profit_2=tp2,
                sl_label=sl_l,
                tp1_label=tp1_l,
                tp2_label=tp2_l,
                reward_risk=rr,
            )
        )

    # Sort: actionable pullback path first, then reclaim, avoid last
    rank = {"A+": 0, "A": 1, "B": 2}
    status_rank = {
        "WAIT_FOR_TRIGGER": 0,
        "READY_TO_WATCH": 1,
        "WAIT_FOR_RECLAIM": 2,
        "WAIT_FOR_ZONE": 2,
        "AVOID_NOW": 3,
    }

    def _path_key(c: SetupCard) -> tuple:
        if c.side == "BUY":
            below = 0 if c.focus_price <= market_mid else 1
        else:
            below = 0 if c.focus_price >= market_mid else 1
        return (
            rank.get(c.grade, 9),
            status_rank.get(c.status, 9),
            below,
            abs(c.focus_price - market_mid),
            -c.score,
        )

    cards.sort(key=_path_key)

    buy = [c for c in cards if c.side == "BUY"][:5]
    sell = [c for c in cards if c.side == "SELL"][:5]

    watchlist = [
        f"H1 bias = {h1_bias.value} (primary). H4={h4_bias.value} M15={m15_bias.value}",
        f"Session = {session_name}" + (" (killzone OK)" if killzone else " (prefer London/NY)"),
    ]
    if playbooks and playbooks.best:
        watchlist.append(f"Active playbook theme: {playbooks.best.name}")
    if playbooks and playbooks.unlock_hints:
        watchlist.extend(f"Wait: {u}" for u in playbooks.unlock_hints[:3])

    # One clear focus line for the trader
    focus_cards = [
        c
        for c in (buy if stance == "LONG_BIAS" else sell if stance == "SHORT_BIAS" else cards)
        if c.status in ("WAIT_FOR_TRIGGER", "READY_TO_WATCH") and c.grade in ("A+", "A")
    ]
    if focus_cards:
        f0 = focus_cards[0]
        watchlist.insert(
            0,
            f"FOCUS NOW: [{f0.grade}] {f0.side} {f0.zone_low:.2f}-{f0.zone_high:.2f} — {f0.status}",
        )

    do_not = [
        "Do NOT auto-trade from this scanner — confirm on TradingView yourself",
        "Do NOT enter mid-range without tagging a graded zone + IF/THEN trigger",
        "Do NOT fight H1 bias on full-size entries just because board leans the other way",
        "Do NOT force A/A+ longs when board shows SELL_LEAN near major resistance",
        "Do NOT treat WAIT_FOR_RECLAIM zones above/below mid as immediate dip entries",
    ]
    if not killzone:
        do_not.append("Outside killzone — reduce size or skip until London/NY")

    a_plus = sum(1 for c in cards if c.grade == "A+")
    a_cnt = sum(1 for c in cards if c.grade == "A")
    if buy and sell:
        summary = (
            f"stance={stance} A+={a_plus} A={a_cnt} | "
            f"best BUY @{buy[0].focus_price:.2f}[{buy[0].grade}/{buy[0].status}] | "
            f"best SELL @{sell[0].focus_price:.2f}[{sell[0].grade}/{sell[0].status}]"
        )
    elif buy:
        summary = (
            f"stance={stance} A+={a_plus} A={a_cnt} | "
            f"best BUY @{buy[0].focus_price:.2f}[{buy[0].grade}/{buy[0].status}]"
        )
    elif sell:
        summary = (
            f"stance={stance} A+={a_plus} A={a_cnt} | "
            f"best SELL @{sell[0].focus_price:.2f}[{sell[0].grade}/{sell[0].status}]"
        )
    else:
        summary = f"stance={stance} cards={len(cards)} A+={a_plus} A={a_cnt}"

    side_compare = build_side_compare(buy, sell, stance)
    summary = (
        f"{summary} | board={side_compare.lean} "
        f"BUYΣ{side_compare.buy_pressure:.0f}/SELLΣ{side_compare.sell_pressure:.0f}"
    )
    if side_compare.lean == "SELL_LEAN" and stance == "LONG_BIAS":
        watchlist.insert(
            0 if not watchlist or not str(watchlist[0]).startswith("FOCUS") else 1,
            "BOARD: sell pressure > buy — correction risk; skip forcing longs",
        )
    elif side_compare.lean == "BUY_LEAN" and stance == "SHORT_BIAS":
        watchlist.insert(
            0 if not watchlist or not str(watchlist[0]).startswith("FOCUS") else 1,
            "BOARD: buy pressure > sell — bounce risk; skip forcing shorts",
        )

    return ManualScanReport(
        bias=h1_bias.value,
        stance=stance,
        headline=headline,
        cards=cards,
        buy_cards=buy,
        sell_cards=sell,
        watchlist=watchlist,
        do_not=do_not,
        summary=summary,
        side_compare=side_compare,
    )


def render_manual_scan_block(
    report: ManualScanReport | None,
    mid: float = 0.0,
    brief: bool = False,
) -> list[str]:
    """Full setup cards by default; brief=True for compact card."""
    lines: list[str] = []
    lines.append("=" * 72)
    lines.append("  MANUAL SETUPS  (YOU decide · YOU trade · NO auto)")
    lines.append("=" * 72)
    if report is None:
        lines.append("  No setups available.")
        return lines

    lines.append(f"  Mid      : {mid:.3f}")
    lines.append(f"  Stance   : {report.stance}  (H1={report.bias})")
    lines.append(f"  Plan     : {report.headline}")
    lines.append(f"  Snapshot : {report.summary}")

    sc = report.side_compare
    if sc is not None:
        lines.append("-" * 72)
        lines.append("  BUY vs SELL PROBABILITY BOARD  (raw scores — not H1 grade)")
        lines.append(
            f"  Lean={sc.lean}  edge={sc.edge:+.0f}  "
            f"(BUY pressure Σ{sc.buy_pressure:.0f} vs SELL Σ{sc.sell_pressure:.0f})"
        )
        for row in sc.rows:
            lines.append(f"    {row}")
        lines.append(f"  → {sc.advice}")
        lines.append(
            "  Note: Grade A/B = bias filter. Board = structure strength. Use both."
        )

    lines.append("-" * 72)
    lines.append("  WATCHLIST")
    for w in report.watchlist[:6]:
        lines.append(f"    • {w}")

    lines.append("-" * 72)
    lines.append("  DO NOT")
    for d in report.do_not[:5]:
        lines.append(f"    • {d}")

    # Always show both sides so correction trades are visible even under HTF bias
    buy_n = 3 if brief else 5
    sell_n = 3 if brief else 5
    if brief and report.stance == "LONG_BIAS":
        lines.append("-" * 72)
        lines.append(
            "  Sells under LONG_BIAS: grade B / AVOID for full-size shorts — "
            "still shown for correction watch + board compare"
        )
    elif brief and report.stance == "SHORT_BIAS":
        lines.append("-" * 72)
        lines.append(
            "  Buys under SHORT_BIAS: grade B / AVOID for full-size longs — "
            "still shown for bounce watch + board compare"
        )

    def _emit(title: str, cards: list[SetupCard], limit: int) -> None:
        lines.append("-" * 72)
        lines.append(f"  {title}")
        if not cards or limit <= 0:
            lines.append("    (none)")
            return
        for i, c in enumerate(cards[:limit], 1):
            lo, hi = min(c.zone_low, c.zone_high), max(c.zone_low, c.zone_high)
            lines.append(
                f"  {i}. [{c.grade}] {c.side}  {lo:.2f}-{hi:.2f}  "
                f"@{c.focus_price:.2f}  {c.kind}  score={c.score:.0f}  {c.status}"
            )
            parts = [p.strip() for p in c.if_then.split("|")]
            for p in parts:
                lines.append(f"      {p}")
            lines.append(f"      {c.invalidation}")
            lines.append(f"      {c.avoid}")
            if c.stop_loss > 0 and c.take_profit_1 > 0:
                lines.append(
                    f"      SL  : {c.stop_loss:.2f}  ({c.sl_label})"
                )
                lines.append(
                    f"      TP1 : {c.take_profit_1:.2f}  ({c.tp1_label})"
                )
                if c.take_profit_2 > 0:
                    lines.append(
                        f"      TP2 : {c.take_profit_2:.2f}  ({c.tp2_label})"
                    )
                if c.reward_risk > 0:
                    lines.append(f"      R:R : 1:{c.reward_risk:.2f}  (to TP1, from zone mid)")
            if not brief:
                for t in c.confirm_on_tv[:2]:
                    lines.append(f"      TV · {t}")
                for r in c.reasons[:2]:
                    lines.append(f"      Why · {r}")
            if c.status == "AVOID_NOW":
                lines.append("      >>> DO NOT ENTER THIS SIDE (against H1 bias)")

    if buy_n:
        _emit(
            "BEST BUY AREAS (support / H4-H1 lows / buy liq / bull FVG-OB / OTE)",
            report.buy_cards,
            buy_n,
        )
    if sell_n:
        _emit(
            "BEST SELL AREAS (resistance / H4-H1 highs / sell liq / bear FVG-OB / OTE)",
            report.sell_cards,
            sell_n,
        )

    lines.append("-" * 72)
    lines.append("  RULES: WAIT first → then CONFIRM candle → only then THEN")
    lines.append("         If CANCEL prints, drop the idea. Bot never sends orders.")
    return lines


