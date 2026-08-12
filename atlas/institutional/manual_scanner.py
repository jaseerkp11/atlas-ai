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
    # Manual risk map — prop-style 1:2 / 1:3 from zone mid + structure SL
    stop_loss: float = 0.0
    take_profit_1: float = 0.0
    take_profit_2: float = 0.0
    sl_label: str = ""
    tp1_label: str = ""
    tp2_label: str = ""
    reward_risk: float = 0.0  # to TP1
    reward_risk_tp2: float = 0.0
    entry_price: float = 0.0
    risk_points: float = 0.0
    reward_points: float = 0.0
    magnet_note: str = ""
    distance_atr: float = 0.0
    # SMC sequence relative to this zone
    sweep_state: str = "WAITING_SWEEP"  # WAITING_SWEEP | SWEPT | RECLAIMED | IN_ZONE
    quality: float = 0.0  # 0–100 checklist-style quality
    challenge_ok: bool = True
    risk_usd: float = 0.0
    reward_usd_tp1: float = 0.0
    reward_usd_tp2: float = 0.0
    lot_size: float = 0.1

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class FocusChecklistItem:
    name: str
    passed: bool
    detail: str

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class FocusPlan:
    """Single best idea + 5-point checklist for the trader."""

    side: str = ""
    grade: str = ""
    kind: str = ""
    zone_low: float = 0.0
    zone_high: float = 0.0
    entry: float = 0.0
    stop_loss: float = 0.0
    take_profit_1: float = 0.0
    take_profit_2: float = 0.0
    reward_risk: float = 0.0
    sweep_state: str = ""
    quality: float = 0.0
    verdict: str = "SKIP"  # WATCH_READY | WAIT_SWEEP | SKIP
    checklist: list[FocusChecklistItem] = field(default_factory=list)
    why: str = ""

    def as_dict(self) -> dict:
        return {
            "side": self.side,
            "grade": self.grade,
            "kind": self.kind,
            "zone_low": self.zone_low,
            "zone_high": self.zone_high,
            "entry": self.entry,
            "stop_loss": self.stop_loss,
            "take_profit_1": self.take_profit_1,
            "take_profit_2": self.take_profit_2,
            "reward_risk": self.reward_risk,
            "sweep_state": self.sweep_state,
            "quality": self.quality,
            "verdict": self.verdict,
            "checklist": [c.as_dict() for c in self.checklist],
            "why": self.why,
        }


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
    buy_liq_count: int = 0
    sell_liq_count: int = 0
    pending_proxy: str = ""  # liquidity-pressure proxy — NOT a real order book
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
    focus: FocusPlan | None = None

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
            "focus": self.focus.as_dict() if self.focus else None,
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
    buy_liq_n = sum(1 for c in buy_cards if c.kind in ("buy_liquidity", "htf_support"))
    sell_liq_n = sum(1 for c in sell_cards if c.kind in ("sell_liquidity", "htf_resistance"))

    def _pair(label: str, b: float, s: float) -> str:
        winner = "BUY" if b > s + 1 else ("SELL" if s > b + 1 else "TIE")
        return f"{label:<14} BUY {b:5.0f}  vs  SELL {s:5.0f}   → {winner}"

    # Liquidity-pressure proxy (NOT a live order book / DOM)
    if sell_liq_n > buy_liq_n + 1 or (sell_liq > buy_liq + 8 and sell_liq_n >= buy_liq_n):
        pending_proxy = (
            f"More SELL-SIDE liquidity magnets ({sell_liq_n} vs {buy_liq_n}) — "
            "stops/resting interest likely ABOVE (proxy, not DOM)"
        )
    elif buy_liq_n > sell_liq_n + 1 or (buy_liq > sell_liq + 8 and buy_liq_n >= sell_liq_n):
        pending_proxy = (
            f"More BUY-SIDE liquidity magnets ({buy_liq_n} vs {sell_liq_n}) — "
            "stops/resting interest likely BELOW (proxy, not DOM)"
        )
    else:
        pending_proxy = (
            f"Liquidity magnets mixed (buy={buy_liq_n} sell={sell_liq_n}) — "
            "no clear pending-side proxy"
        )

    rows = [
        _pair("Overall best", buy_best, sell_best),
        _pair("Top3 avg", buy_avg, sell_avg),
        _pair("Pressure Σ3", buy_pressure, sell_pressure),
        _pair("Support/Res", support_best, resistance_best),
        _pair("FVG", bull_fvg, bear_fvg),
        _pair("Order Block", bull_ob, bear_ob),
        _pair("Liquidity", buy_liq, sell_liq),
        _pair("HTF major", htf_sup, htf_res),
        f"{'Liq magnets':<14} BUY {buy_liq_n:5d}  vs  SELL {sell_liq_n:5d}",
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
    advice = f"{advice} | {pending_proxy}"

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
        buy_liq_count=buy_liq_n,
        sell_liq_count=sell_liq_n,
        pending_proxy=pending_proxy,
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


def _refine_area_for_challenge(
    area: TradeArea,
    max_risk_points: float,
    atr: float,
) -> TradeArea:
    """
    If a zone is wider than the challenge stop, shrink to the reactive edge band.

    BUY  → upper portion of demand (defend after low sweep)
    SELL → lower portion of supply (reject after high sweep)
    Keeps original mid as fallback only when already tight.
    """
    lo = min(float(area.price_low), float(area.price_high))
    hi = max(float(area.price_low), float(area.price_high))
    width = hi - lo
    cap = max(float(max_risk_points), 1.5)
    # Leave room for buffer under the $50 risk
    target_width = min(cap * 0.70, max(atr * 0.35, 2.0))
    if width <= cap * 0.95:
        return area

    if area.side == "BUY":
        new_hi = hi
        new_lo = max(lo, hi - target_width)
    else:
        new_lo = lo
        new_hi = min(hi, lo + target_width)
    new_mid = (new_lo + new_hi) / 2.0
    reasons = list(area.reasons)
    reasons.insert(
        0,
        f"Refined wide {width:.2f}→{new_hi-new_lo:.2f} edge band for 0.1 lot / $50 stop",
    )
    return TradeArea(
        side=area.side,
        kind=area.kind,
        price_low=new_lo,
        price_high=new_hi,
        mid_price=new_mid,
        score=area.score,
        distance_atr=area.distance_atr,
        fresh_unfilled=area.fresh_unfilled,
        reasons=reasons,
        label=area.label,
    )


def _plan_sl_tp(
    area: TradeArea,
    all_areas: list[TradeArea],
    atr: float,
    min_rr_tp1: float = 2.0,
    min_rr_tp2: float = 3.0,
    max_risk_points: float | None = None,
    lot_size: float = 0.1,
    usd_per_price_unit_per_lot: float = 100.0,
) -> tuple[
    float, float, float, float, str, str, str, float, float, float, float, str, bool, float, float, float
]:
    """
    Challenge risk map: 0.1 lot / ~$50 max loss → fixed risk distance, TP 1:2 / 1:3.

    Returns:
      entry, sl, tp1, tp2, sl_label, tp1_label, tp2_label,
      rr1, rr2, risk_pts, reward_pts, magnet_note,
      challenge_ok, risk_usd, reward_usd_tp1, reward_usd_tp2
    """
    atr = max(float(atr), 1e-9)
    lo, hi, focus = float(area.price_low), float(area.price_high), float(area.mid_price)
    width = abs(hi - lo)
    # Tight buffer — challenge stops are small (~$5 on XAU 0.1 lot)
    buffer = min(max(atr * 0.12, width * 0.10), 1.0)
    px_value = max(float(usd_per_price_unit_per_lot) * max(float(lot_size), 1e-9), 1e-9)

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

    def _usd(pts: float) -> float:
        return round(abs(pts) * px_value, 2)

    # Challenge risk cap (default $5 price = $50 @ 0.1 lot)
    if max_risk_points is None or max_risk_points <= 0:
        max_risk_points = 5.0
    max_risk_points = float(max_risk_points)
    # Floor so gold noise does not make absurdly tiny stops
    min_risk = max(1.5, atr * 0.15)

    side = area.side
    entry = focus
    challenge_ok = True
    magnet_note = ""

    if side == "BUY":
        struct_sl = lo - buffer
        struct_risk = max(entry - struct_sl, min_risk)
        if struct_risk <= max_risk_points * 1.05:
            risk = max(struct_risk, min_risk)
            sl = entry - risk
            sl_label = f"below zone {lo:.2f} (struct, risk={risk:.2f}≈${_usd(risk):.0f})"
        else:
            # Zone too wide for $50 @ 0.1 lot — cap risk, mark unfit for FOCUS
            risk = max_risk_points
            sl = entry - risk
            challenge_ok = False
            sl_label = (
                f"challenge cap −{risk:.2f} (${_usd(risk):.0f} @ {lot_size} lot) "
                f"— zone wider than stop"
            )
            magnet_note = (
                f"Zone risk {struct_risk:.2f} > challenge max {max_risk_points:.2f} — "
                "prefer tighter area"
            )
        struct_tp1 = entry + min_rr_tp1 * risk
        struct_tp2 = entry + min_rr_tp2 * risk
        magnets = sorted(
            [a for a in all_areas if a.side == "SELL" and a.mid_price > entry + risk * 0.5],
            key=lambda a: a.mid_price,
        )
        tp1, tp1_label = struct_tp1, f"1:{min_rr_tp1:g} (+${_usd(min_rr_tp1 * risk):.0f}) @{struct_tp1:.2f}"
        tp2, tp2_label = struct_tp2, f"1:{min_rr_tp2:g} (+${_usd(min_rr_tp2 * risk):.0f}) @{struct_tp2:.2f}"
        for m in magnets:
            m_rr = _rr(entry, sl, float(m.mid_price), "BUY")
            if min_rr_tp1 <= m_rr <= min_rr_tp1 + 0.75:
                tp1 = float(m.mid_price)
                tp1_label = f"{m.kind} @{tp1:.2f} (~1:{m_rr:g}, +${_usd(tp1-entry):.0f})"
                break
        for m in magnets:
            m_rr = _rr(entry, sl, float(m.mid_price), "BUY")
            if m_rr + 1e-9 >= min_rr_tp2 and float(m.mid_price) > tp1:
                tp2 = float(m.mid_price)
                tp2_label = f"{m.kind} @{tp2:.2f} (≥1:{min_rr_tp2:g}, +${_usd(tp2-entry):.0f})"
                break
        early = [a for a in magnets if float(a.mid_price) < entry + min_rr_tp1 * risk * 0.95]
        if early and not magnet_note:
            e0 = early[0]
            magnet_note = (
                f"Partial before 2R: {e0.kind} @{e0.mid_price:.2f} (scale out only)"
            )
        if tp2 <= tp1:
            tp2 = max(entry + min_rr_tp2 * risk, tp1 + risk * 0.5)
            tp2_label = f"1:{min_rr_tp2:g} (+${_usd(tp2-entry):.0f}) @{tp2:.2f}"
        rr1 = _rr(entry, sl, tp1, "BUY")
        rr2 = _rr(entry, sl, tp2, "BUY")
        return (
            entry, sl, tp1, tp2, sl_label, tp1_label, tp2_label,
            rr1, rr2, round(risk, 2), round(tp1 - entry, 2), magnet_note,
            challenge_ok, _usd(risk), _usd(tp1 - entry), _usd(tp2 - entry),
        )

    # SELL
    struct_sl = hi + buffer
    struct_risk = max(struct_sl - entry, min_risk)
    if struct_risk <= max_risk_points * 1.05:
        risk = max(struct_risk, min_risk)
        sl = entry + risk
        sl_label = f"above zone {hi:.2f} (struct, risk={risk:.2f}≈${_usd(risk):.0f})"
    else:
        risk = max_risk_points
        sl = entry + risk
        challenge_ok = False
        sl_label = (
            f"challenge cap +{risk:.2f} (${_usd(risk):.0f} @ {lot_size} lot) "
            f"— zone wider than stop"
        )
        magnet_note = (
            f"Zone risk {struct_risk:.2f} > challenge max {max_risk_points:.2f} — "
            "prefer tighter area"
        )
    struct_tp1 = entry - min_rr_tp1 * risk
    struct_tp2 = entry - min_rr_tp2 * risk
    magnets = sorted(
        [a for a in all_areas if a.side == "BUY" and a.mid_price < entry - risk * 0.5],
        key=lambda a: -a.mid_price,
    )
    tp1, tp1_label = struct_tp1, f"1:{min_rr_tp1:g} (+${_usd(min_rr_tp1 * risk):.0f}) @{struct_tp1:.2f}"
    tp2, tp2_label = struct_tp2, f"1:{min_rr_tp2:g} (+${_usd(min_rr_tp2 * risk):.0f}) @{struct_tp2:.2f}"
    for m in magnets:
        m_rr = _rr(entry, sl, float(m.mid_price), "SELL")
        if min_rr_tp1 <= m_rr <= min_rr_tp1 + 0.75:
            tp1 = float(m.mid_price)
            tp1_label = f"{m.kind} @{tp1:.2f} (~1:{m_rr:g}, +${_usd(entry-tp1):.0f})"
            break
    for m in magnets:
        m_rr = _rr(entry, sl, float(m.mid_price), "SELL")
        if m_rr + 1e-9 >= min_rr_tp2 and float(m.mid_price) < tp1:
            tp2 = float(m.mid_price)
            tp2_label = f"{m.kind} @{tp2:.2f} (≥1:{min_rr_tp2:g}, +${_usd(entry-tp2):.0f})"
            break
    early = [a for a in magnets if float(a.mid_price) > entry - min_rr_tp1 * risk * 0.95]
    if early and not magnet_note:
        e0 = early[0]
        magnet_note = (
            f"Partial before 2R: {e0.kind} @{e0.mid_price:.2f} (scale out only)"
        )
    if tp2 >= tp1:
        tp2 = min(entry - min_rr_tp2 * risk, tp1 - risk * 0.5)
        tp2_label = f"1:{min_rr_tp2:g} (+${_usd(entry-tp2):.0f}) @{tp2:.2f}"
    rr1 = _rr(entry, sl, tp1, "SELL")
    rr2 = _rr(entry, sl, tp2, "SELL")
    return (
        entry, sl, tp1, tp2, sl_label, tp1_label, tp2_label,
        rr1, rr2, round(risk, 2), round(entry - tp1, 2), magnet_note,
        challenge_ok, _usd(risk), _usd(entry - tp1), _usd(entry - tp2),
    )



def _sweep_state_for_area(
    area: TradeArea,
    market_mid: float,
    sweep_bull: bool = False,
    sweep_bear: bool = False,
    last_low: float | None = None,
    last_high: float | None = None,
    atr: float = 1.0,
) -> str:
    """
    Zone-local SMC sequence:
      WAITING_SWEEP → SWEPT → RECLAIMED / IN_ZONE

    RECLAIMED requires a wick beyond THIS zone on the last closed bar
    (not a global unrelated swing sweep).
    """
    lo = min(area.price_low, area.price_high)
    hi = max(area.price_low, area.price_high)
    atr = max(float(atr), 1e-9)
    zone_mid = (lo + hi) / 2.0
    near = abs(market_mid - zone_mid) / atr <= 2.5
    _ = (sweep_bull, sweep_bear)  # reserved / dashboard context only

    if area.side == "BUY":
        wicked_below = last_low is not None and float(last_low) < lo
        if market_mid < lo:
            return "SWEPT"
        if lo <= market_mid <= hi:
            return "RECLAIMED" if wicked_below else "IN_ZONE"
        # Above the box: only RECLAIMED if this zone was wicked and price is back near
        if wicked_below and market_mid >= lo and near:
            return "RECLAIMED"
        return "WAITING_SWEEP"

    wicked_above = last_high is not None and float(last_high) > hi
    if market_mid > hi:
        return "SWEPT"
    if lo <= market_mid <= hi:
        return "RECLAIMED" if wicked_above else "IN_ZONE"
    if wicked_above and market_mid <= hi and near:
        return "RECLAIMED"
    return "WAITING_SWEEP"


def _card_quality(
    grade: str,
    killzone: bool,
    pd_ok: bool,
    board_ok: bool,
    rr: float,
    sweep_state: str,
    agree_stack: bool,
) -> float:
    q = 40.0
    if grade == "A+":
        q += 20
    elif grade == "A":
        q += 12
    elif grade == "B":
        q += 4
    if killzone:
        q += 10
    if pd_ok:
        q += 10
    if board_ok:
        q += 8
    if rr >= 3.0:
        q += 12
    elif rr >= 2.0:
        q += 8
    elif rr >= 1.5:
        q += 3
    if sweep_state in ("RECLAIMED", "IN_ZONE"):
        q += 10
    elif sweep_state == "SWEPT":
        q += 4
    if agree_stack:
        q += 6
    return max(0.0, min(100.0, q))


def build_focus_plan(
    cards: list[SetupCard],
    stance: str,
    h1_bias: Bias,
    killzone: bool,
    pd_zone: str,
    board_lean: str,
    session_name: str,
) -> FocusPlan | None:
    """Pick one best with-trend card and score a 5-point prop checklist."""
    pool = [
        c
        for c in cards
        if c.status != "AVOID_NOW"
        and c.challenge_ok
        and (
            (stance == "LONG_BIAS" and c.side == "BUY")
            or (stance == "SHORT_BIAS" and c.side == "SELL")
            or stance == "NO_EDGE"
        )
        and c.grade in ("A+", "A", "B")
    ]
    if not pool:
        # Fall back without challenge_ok so user still sees a map
        pool = [
            c
            for c in cards
            if c.status != "AVOID_NOW"
            and (
                (stance == "LONG_BIAS" and c.side == "BUY")
                or (stance == "SHORT_BIAS" and c.side == "SELL")
                or stance == "NO_EDGE"
            )
            and c.grade in ("A+", "A", "B")
        ]
    if not pool:
        return None
    pool.sort(
        key=lambda c: (
            0 if c.challenge_ok else 1,
            -c.quality,
            {"A+": 0, "A": 1, "B": 2}.get(c.grade, 9),
            c.distance_atr,
        )
    )
    c = pool[0]

    bias_ok = (c.side == "BUY" and h1_bias == Bias.BULLISH) or (
        c.side == "SELL" and h1_bias == Bias.BEARISH
    )
    if c.side == "BUY":
        pd_ok = pd_zone in ("discount", "equilibrium", "")
        pd_detail = f"H1 array={pd_zone or '?'} (want discount/eq for buys)"
    else:
        pd_ok = pd_zone in ("premium", "equilibrium", "")
        pd_detail = f"H1 array={pd_zone or '?'} (want premium/eq for sells)"

    if stance == "LONG_BIAS":
        board_ok = board_lean != "SELL_LEAN"
        board_detail = f"board={board_lean} (SELL_LEAN = correction risk)"
    elif stance == "SHORT_BIAS":
        board_ok = board_lean != "BUY_LEAN"
        board_detail = f"board={board_lean} (BUY_LEAN = bounce risk)"
    else:
        board_ok = True
        board_detail = f"board={board_lean}"

    rr_ok = c.reward_risk >= 2.0
    needs_liq_reclaim = c.kind in ("buy_liquidity", "sell_liquidity")
    # Liquidity ideas need true reclaim; tap zones can use IN_ZONE
    if needs_liq_reclaim:
        sweep_ok = c.sweep_state == "RECLAIMED"
    else:
        sweep_ok = c.sweep_state in ("RECLAIMED", "IN_ZONE")

    checklist = [
        FocusChecklistItem("H1 bias aligned", bias_ok, f"side={c.side} H1={h1_bias.value}"),
        FocusChecklistItem("Premium/Discount OK", pd_ok, pd_detail),
        FocusChecklistItem("Killzone session", killzone, f"session={session_name}"),
        FocusChecklistItem("Board not fighting", board_ok, board_detail),
        FocusChecklistItem("R:R ≥ 1:2 to TP1", rr_ok, f"R:R=1:{c.reward_risk:.2f}"),
        FocusChecklistItem(
            "Challenge risk fit",
            c.challenge_ok,
            f"0.1 lot risk≈${c.risk_usd:.0f} (max $50) | TP1≈${c.reward_usd_tp1:.0f} TP2≈${c.reward_usd_tp2:.0f}",
        ),
        FocusChecklistItem(
            "Sweep/reclaim state",
            sweep_ok,
            (
                f"state={c.sweep_state} "
                + (
                    "(liquidity: need RECLAIMED after sweep)"
                    if needs_liq_reclaim
                    else "(PASS if IN_ZONE or RECLAIMED)"
                )
            ),
        ),
    ]
    core = [checklist[0], checklist[1], checklist[3], checklist[4], checklist[5]]
    core_pass = all(i.passed for i in core)

    if not bias_ok or not rr_ok:
        verdict = "SKIP"
        why = "Fail core filter (bias or R:R) — map only, do not take."
    elif not c.challenge_ok:
        verdict = "SKIP"
        why = "Zone too wide for $50 / 0.1 lot plan — wait for a tighter area."
    elif not core_pass:
        verdict = "SKIP"
        why = "Checklist incomplete — wait for PD/board alignment."
    elif c.sweep_state == "WAITING_SWEEP":
        verdict = "WAIT_SWEEP"
        why = "Plan is valid — wait for price to tag/sweep the zone first."
    elif c.sweep_state == "SWEPT":
        verdict = "WAIT_SWEEP"
        why = "Liquidity taken — wait for reclaim close back through the zone."
    elif needs_liq_reclaim and c.sweep_state == "IN_ZONE":
        verdict = "WAIT_SWEEP"
        why = "Inside liquidity box but no sweep wick yet — wait sweep + reclaim."
    elif not sweep_ok:
        verdict = "WAIT_SWEEP"
        why = "Sweep/reclaim not complete — do not enter yet."
    elif not killzone:
        verdict = "WATCH_READY"
        why = (
            f"Ready (reduce size off-killzone). Lot 0.1 | risk≈${c.risk_usd:.0f} | "
            f"TP1≈${c.reward_usd_tp1:.0f} (1:2) | TP2≈${c.reward_usd_tp2:.0f} (1:3)."
        )
    else:
        verdict = "WATCH_READY"
        why = (
            f"FOCUS ready @ 0.1 lot. Risk≈${c.risk_usd:.0f} → "
            f"TP1≈${c.reward_usd_tp1:.0f} / TP2≈${c.reward_usd_tp2:.0f}. "
            "Confirm M5 on TradingView, then decide."
        )

    return FocusPlan(
        side=c.side,
        grade=c.grade,
        kind=c.kind,
        zone_low=min(c.zone_low, c.zone_high),
        zone_high=max(c.zone_low, c.zone_high),
        entry=c.entry_price or c.focus_price,
        stop_loss=c.stop_loss,
        take_profit_1=c.take_profit_1,
        take_profit_2=c.take_profit_2,
        reward_risk=c.reward_risk,
        sweep_state=c.sweep_state,
        quality=c.quality,
        verdict=verdict,
        checklist=checklist,
        why=why,
    )


def build_manual_scan(
    narrative: MarketNarrative,
    trade_areas: TradeAreasReport | None,
    playbooks: PlaybookResult | None,
    h1_bias: Bias,
    h4_bias: Bias,
    m15_bias: Bias,
    session_name: str,
) -> ManualScanReport:
    from atlas.institutional.config import load_institutional_config

    cfg = load_institutional_config()
    ch = cfg.challenge or {}
    ch_on = bool(ch.get("enabled", True))
    lot_size = float(ch.get("lot_size", 0.1))
    usd_per = float(ch.get("usd_per_price_unit_per_lot", 100.0))
    max_risk_pts = cfg.challenge_max_risk_points() if ch_on else None
    min_rr1 = float(ch.get("min_rr_tp1", 2.0))
    min_rr2 = float(ch.get("min_rr_tp2", 3.0))
    prefer_dist = float(ch.get("prefer_distance_atr", 1.2))
    _ = float(ch.get("max_zone_width_atr", 1.25))  # reserved for future soft filter

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
    raw_areas = list(trade_areas.areas if trade_areas else [])
    # Prefer tight, near, high-score areas that fit challenge stop distance
    def _area_rank(a: TradeArea) -> tuple:
        width = abs(a.price_high - a.price_low)
        risk_cap = float(max_risk_pts or 5.0)
        # Fit = zone width can sit under challenge stop (+ small slack)
        if width <= risk_cap * 0.85:
            fit = 0
        elif width <= risk_cap * 1.25:
            fit = 1
        else:
            fit = 2
        near_ok = a.distance_atr <= prefer_dist
        return (
            fit,
            0 if near_ok else 1,
            a.distance_atr,
            -a.score,
        )

    areas = sorted(raw_areas, key=_area_rank)[:12]
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

        # Challenge: shrink wide boxes to reactive edge before SL/TP map
        plan_area = a
        if ch_on and max_risk_pts is not None:
            plan_area = _refine_area_for_challenge(a, float(max_risk_pts), atr)
            if plan_area.mid_price != a.mid_price:
                reasons.append(plan_area.reasons[0])

        lo = min(plan_area.price_low, plan_area.price_high)
        hi = max(plan_area.price_low, plan_area.price_high)
        # Use full area list for opposing TP targets (not only first 12 slice peer)
        pool = trade_areas.areas if trade_areas else [a]
        (
            entry,
            sl,
            tp1,
            tp2,
            sl_l,
            tp1_l,
            tp2_l,
            rr,
            rr2,
            risk_pts,
            reward_pts,
            magnet_note,
            challenge_ok,
            risk_usd,
            reward_usd_tp1,
            reward_usd_tp2,
        ) = _plan_sl_tp(
            plan_area,
            pool,
            atr,
            min_rr_tp1=min_rr1,
            min_rr_tp2=min_rr2,
            max_risk_points=max_risk_pts,
            lot_size=lot_size,
            usd_per_price_unit_per_lot=usd_per,
        )
        # Prop gate: A+ needs planned ≥1:2 to TP1
        if grade == "A+" and rr > 0 and rr < 2.0:
            grade = "A"
            reasons.append(f"Capped A+→A: R:R to TP1 only 1:{rr:.2f} (need ≥1:2 for prop)")
        elif grade == "A" and rr > 0 and rr < 1.8:
            grade = "B"
            reasons.append(f"Capped A→B: R:R to TP1 only 1:{rr:.2f} (weak for challenge)")
        if ch_on and not challenge_ok:
            if grade == "A+":
                grade = "A"
            reasons.append(
                f"Wide zone for 0.1 lot/${ch.get('max_risk_usd', 50):.0f} plan — "
                "prefer tighter area"
            )
        if magnet_note:
            reasons.append(magnet_note)
        sweep_bull = bool((narrative.extras or {}).get("sweep_bullish"))
        sweep_bear = bool((narrative.extras or {}).get("sweep_bearish"))
        last_low = (narrative.extras or {}).get("last_closed_low")
        last_high = (narrative.extras or {}).get("last_closed_high")
        try:
            last_low_f = float(last_low) if last_low is not None else None
        except (TypeError, ValueError):
            last_low_f = None
        try:
            last_high_f = float(last_high) if last_high is not None else None
        except (TypeError, ValueError):
            last_high_f = None
        sweep_state = _sweep_state_for_area(
            plan_area,
            market_mid,
            sweep_bull,
            sweep_bear,
            last_low=last_low_f,
            last_high=last_high_f,
            atr=atr,
        )
        pd_zone_now = str((narrative.extras or {}).get("pd_zone", "") or "")
        if plan_area.side == "BUY":
            pd_ok = pd_zone_now in ("discount", "equilibrium", "")
        else:
            pd_ok = pd_zone_now in ("premium", "equilibrium", "")
        # Board lean unknown until after cards; approximate with H1 for quality seed
        board_ok_seed = status != "AVOID_NOW"
        quality = _card_quality(
            grade, killzone, pd_ok, board_ok_seed, rr, sweep_state, agree_stack
        )
        cards.append(
            SetupCard(
                grade=grade,
                side=plan_area.side,
                zone_low=lo,
                zone_high=hi,
                focus_price=plan_area.mid_price,
                kind=plan_area.kind,
                score=plan_area.score,
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
                reward_risk_tp2=rr2,
                entry_price=entry,
                risk_points=risk_pts,
                reward_points=reward_pts,
                magnet_note=magnet_note,
                distance_atr=float(plan_area.distance_atr),
                sweep_state=sweep_state,
                quality=quality,
                challenge_ok=challenge_ok,
                risk_usd=risk_usd,
                reward_usd_tp1=reward_usd_tp1,
                reward_usd_tp2=reward_usd_tp2,
                lot_size=lot_size,
            )
        )

    # Sort: challenge-fit + actionable pullback path first, then reclaim, avoid last
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
            0 if c.challenge_ok else 1,
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
    pd_zone = str((narrative.extras or {}).get("pd_zone", "") or "")
    if pd_zone:
        watchlist.append(
            f"H1 array = {pd_zone.upper()}"
            + (
                " — prefer buys in discount"
                if pd_zone == "discount" and h1_bias == Bias.BULLISH
                else (
                    " — prefer sells in premium"
                    if pd_zone == "premium" and h1_bias == Bias.BEARISH
                    else (
                        " — wait discount pullback"
                        if pd_zone == "premium" and h1_bias == Bias.BULLISH
                        else (
                            " — wait premium rally"
                            if pd_zone == "discount" and h1_bias == Bias.BEARISH
                            else ""
                        )
                    )
                )
            )
        )
    news_status = str(getattr(narrative, "news_status", "") or "")
    news_detail = str((narrative.extras or {}).get("news_detail", "") or "")
    if news_status and news_status != "Trade Today":
        watchlist.append(f"NEWS: {news_status}" + (f" — {news_detail}" if news_detail else ""))
    if playbooks and playbooks.best:
        watchlist.append(f"Active playbook theme: {playbooks.best.name}")
    if playbooks and playbooks.unlock_hints:
        watchlist.extend(f"Wait: {u}" for u in playbooks.unlock_hints[:3])

    # One clear focus line for the trader (include reclaim magnets, not only trigger-ready)
    focus_cards = [
        c
        for c in (buy if stance == "LONG_BIAS" else sell if stance == "SHORT_BIAS" else cards)
        if c.status
        in ("WAIT_FOR_TRIGGER", "READY_TO_WATCH", "WAIT_FOR_RECLAIM", "WAIT_FOR_ZONE")
        and c.grade in ("A+", "A")
        and c.challenge_ok
    ]
    if not focus_cards:
        focus_cards = [
            c
            for c in (buy if stance == "LONG_BIAS" else sell if stance == "SHORT_BIAS" else cards)
            if c.status
            in ("WAIT_FOR_TRIGGER", "READY_TO_WATCH", "WAIT_FOR_RECLAIM", "WAIT_FOR_ZONE")
            and c.grade in ("A+", "A")
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
        "Challenge plan: 0.1 lot only — skip [WIDE ZONE] cards; risk≈$50 → TP1≈$100 / TP2≈$150",
    ]
    if news_status == "Avoid Trading Today" or bool((narrative.extras or {}).get("news_block")):
        do_not.insert(0, f"Do NOT trade through news block — {news_detail or news_status}")
    elif news_status == "Trade With Caution":
        do_not.insert(0, f"Reduce size / skip new entries — {news_detail or news_status}")
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

    # Recompute quality with real board lean, then build FOCUS plan
    for c in cards:
        if c.side == "BUY":
            pd_ok = pd_zone in ("discount", "equilibrium", "")
            board_ok = side_compare.lean != "SELL_LEAN"
        else:
            pd_ok = pd_zone in ("premium", "equilibrium", "")
            board_ok = side_compare.lean != "BUY_LEAN"
        c.quality = _card_quality(
            c.grade,
            killzone,
            pd_ok,
            board_ok or c.status == "AVOID_NOW",
            c.reward_risk,
            c.sweep_state,
            agree_stack,
        )

    focus = build_focus_plan(
        cards,
        stance,
        h1_bias,
        killzone,
        pd_zone,
        side_compare.lean,
        session_name,
    )
    if focus:
        summary = (
            f"{summary} | FOCUS={focus.verdict} Q={focus.quality:.0f} "
            f"{focus.side}@{focus.entry:.2f}"
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
        focus=focus,
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

    fp = report.focus
    if fp is not None and fp.side:
        lines.append("-" * 72)
        lines.append("  FOCUS TRADE  (one idea — confirm on TradingView)")
        lines.append(
            f"  Verdict={fp.verdict}  Quality={fp.quality:.0f}/100  "
            f"[{fp.grade}] {fp.side} {fp.kind}  sweep={fp.sweep_state}"
        )
        lines.append(
            f"  Zone {fp.zone_low:.2f}-{fp.zone_high:.2f}  |  "
            f"ENTRY {fp.entry:.2f}  SL {fp.stop_loss:.2f}  "
            f"TP1 {fp.take_profit_1:.2f}  TP2 {fp.take_profit_2:.2f}  "
            f"R:R 1:{fp.reward_risk:.2f}"
        )
        # Pull $ plan from matching card if present
        for c in list(report.buy_cards) + list(report.sell_cards):
            if (
                c.side == fp.side
                and abs(c.entry_price - fp.entry) < 1e-6
                and c.lot_size > 0
            ):
                lines.append(
                    f"  Money  : {c.lot_size:g} lot | "
                    f"risk≈${c.risk_usd:.0f} | "
                    f"TP1≈${c.reward_usd_tp1:.0f} | TP2≈${c.reward_usd_tp2:.0f}"
                )
                break
        for item in fp.checklist:
            mark = "PASS" if item.passed else "FAIL"
            lines.append(f"    [{mark}] {item.name} — {item.detail}")
        lines.append(f"  → {fp.why}")

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
        if sc.pending_proxy:
            lines.append(f"  Liquidity proxy: {sc.pending_proxy}")
        lines.append(f"  → {sc.advice}")
        lines.append(
            "  Note: Grade A/B = bias filter. Board = structure strength. "
            "Liquidity proxy ≠ real pending orders / DOM."
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
                f"@{c.focus_price:.2f}  {c.kind}  score={c.score:.0f}  "
                f"Q={c.quality:.0f}  dist={c.distance_atr:.2f}ATR  "
                f"sweep={c.sweep_state}  {c.status}"
            )
            parts = [p.strip() for p in c.if_then.split("|")]
            for p in parts:
                lines.append(f"      {p}")
            lines.append(f"      {c.invalidation}")
            lines.append(f"      {c.avoid}")
            if c.stop_loss > 0 and c.take_profit_1 > 0:
                lines.append(
                    f"      LOT  : {c.lot_size:g}  |  "
                    f"RISK≈${c.risk_usd:.0f}  TP1≈${c.reward_usd_tp1:.0f}  "
                    f"TP2≈${c.reward_usd_tp2:.0f}"
                    + ("" if c.challenge_ok else "  [WIDE ZONE — skip for challenge]")
                )
                lines.append(
                    f"      ENTRY: {c.entry_price:.2f}  (zone mid — plan R:R from here)"
                )
                lines.append(
                    f"      SL   : {c.stop_loss:.2f}  ({c.sl_label})"
                )
                lines.append(
                    f"      TP1  : {c.take_profit_1:.2f}  ({c.tp1_label})"
                )
                if c.take_profit_2 > 0:
                    lines.append(
                        f"      TP2  : {c.take_profit_2:.2f}  ({c.tp2_label})"
                    )
                lines.append(
                    f"      RISK : {c.risk_points:.2f} pts  |  "
                    f"REWARD TP1: {c.reward_points:.2f} pts"
                )
                if c.reward_risk > 0:
                    rr2 = f"  TP2 1:{c.reward_risk_tp2:.2f}" if c.reward_risk_tp2 > 0 else ""
                    lines.append(
                        f"      R:R  : 1:{c.reward_risk:.2f} to TP1{rr2}  "
                        f"(challenge 1:2 / 1:3 @ 0.1 lot)"
                    )
                if c.magnet_note:
                    lines.append(f"      NOTE : {c.magnet_note}")
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


