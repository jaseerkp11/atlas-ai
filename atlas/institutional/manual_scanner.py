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
        }


def _grade(area: TradeArea, h1: Bias, killzone: bool, agree_stack: bool) -> tuple[str, str]:
    """Return (grade, status). With-trend near zones grade higher."""
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
    )

    if against:
        return "B", "AVOID_NOW"

    near = area.distance_atr <= 1.6
    strong = area.score >= 82
    elite = (
        with_trend
        and near
        and strong
        and premium_kind
        and killzone
        and agree_stack
        and area.score >= 88
    )
    if elite:
        return "A+", "WAIT_FOR_TRIGGER"
    if with_trend and near and area.score >= 75:
        return "A", "WAIT_FOR_TRIGGER" if premium_kind or area.score >= 85 else "READY_TO_WATCH"
    if with_trend and area.score >= 70:
        return "A", "READY_TO_WATCH"
    return "B", "READY_TO_WATCH"


def _if_then(area: TradeArea) -> tuple[str, list[str], str, str]:
    lo, hi, mid = area.price_low, area.price_high, area.mid_price
    kind = area.kind
    if area.side == "BUY":
        inv = f"Invalidate long idea if M5 closes below {lo - (hi - lo) * 0.35:.2f} and holds"
        avoid = f"Do not chase longs above resistance / mid if price never tags {lo:.2f}-{hi:.2f}"
        if kind == "buy_liquidity":
            ift = (
                f"IF price sweeps below {mid:.2f} (buy-side liquidity) THEN reclaim back above "
                f"{mid:.2f} on M5 close → look LONG"
            )
            conf = [
                "Mark equal lows / liquidity pool on TradingView",
                "Wait for wick below then CLOSE back above the level",
                "Prefer bullish pin / engulf on M5 after reclaim",
            ]
        elif kind in ("bullish_fvg", "bullish_ob"):
            ift = (
                f"IF price revisits unfilled {kind} {lo:.2f}-{hi:.2f} THEN holds / rejects higher "
                f"on M5 → look LONG"
            )
            conf = [
                f"Draw {kind} box {lo:.2f}-{hi:.2f}",
                "Wait for touch + bullish reaction candle",
                "No entry on first touch without reaction",
            ]
        elif kind == "fib_ote":
            ift = (
                f"IF pullback reaches OTE {lo:.2f}-{hi:.2f} THEN bullish M5 confirmation → look LONG"
            )
            conf = [
                "Plot Fib swing; focus 61.8–78.6",
                "Need displacement prior + reaction in pocket",
            ]
        else:  # support
            ift = (
                f"IF price dips into support {lo:.2f}-{hi:.2f} THEN bullish M5 pin/engulf → look LONG"
            )
            conf = [
                f"Horizontal support {lo:.2f}-{hi:.2f}",
                "Wait for rejection wick + close back up",
            ]
        return ift, conf, inv, avoid

    # SELL
    inv = f"Invalidate short idea if M5 closes above {hi + (hi - lo) * 0.35:.2f} and holds"
    avoid = f"Do not chase shorts into support if price never tags {lo:.2f}-{hi:.2f}"
    if kind == "sell_liquidity":
        ift = (
            f"IF price sweeps above {mid:.2f} (sell-side liquidity) THEN rejects back below "
            f"{mid:.2f} on M5 close → look SHORT"
        )
        conf = [
            "Mark equal highs / liquidity pool",
            "Wait for wick above then CLOSE back below",
            "Prefer bearish pin / engulf on M5 after reject",
        ]
    elif kind in ("bearish_fvg", "bearish_ob"):
        ift = (
            f"IF price revisits unfilled {kind} {lo:.2f}-{hi:.2f} THEN holds / rejects lower "
            f"on M5 → look SHORT"
        )
        conf = [
            f"Draw {kind} box {lo:.2f}-{hi:.2f}",
            "Wait for touch + bearish reaction candle",
        ]
    elif kind == "fib_ote":
        ift = (
            f"IF rally reaches OTE {lo:.2f}-{hi:.2f} THEN bearish M5 confirmation → look SHORT"
        )
        conf = ["Plot Fib; focus 61.8–78.6 premium", "Need reaction, not blind fade"]
    else:
        ift = (
            f"IF price rallies into resistance {lo:.2f}-{hi:.2f} THEN bearish M5 pin/engulf → look SHORT"
        )
        conf = [
            f"Horizontal resistance {lo:.2f}-{hi:.2f}",
            "Wait for rejection wick + close back down",
        ]
    return ift, conf, inv, avoid


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
        grade, status = _grade(a, h1_bias, killzone, agree_stack)
        ift, conf, inv, avoid = _if_then(a)
        reasons = list(a.reasons[:3])
        if playbooks and playbooks.best and playbooks.best.bias.value.startswith(a.side[:3]):
            # BUY vs BULLISH — simple link
            pass
        if playbooks and playbooks.best:
            if (a.side == "BUY" and playbooks.best.bias == Bias.BULLISH) or (
                a.side == "SELL" and playbooks.best.bias == Bias.BEARISH
            ):
                reasons.append(f"Supports playbook: {playbooks.best.name}")

        # Soften counter-trend cards
        if status == "AVOID_NOW":
            ift = "AGAINST H1 bias — skip unless H1 flips. " + ift

        cards.append(
            SetupCard(
                grade=grade,
                side=a.side,
                zone_low=a.price_low,
                zone_high=a.price_high,
                focus_price=a.mid_price,
                kind=a.kind,
                score=a.score,
                status=status,
                if_then=ift,
                confirm_on_tv=conf,
                invalidation=inv,
                avoid=avoid,
                reasons=reasons,
            )
        )

    # Sort: A+ first, then A, then with-trend, then score
    rank = {"A+": 0, "A": 1, "B": 2}
    cards.sort(key=lambda c: (rank.get(c.grade, 9), 0 if c.status != "AVOID_NOW" else 1, -c.score))

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

    do_not = [
        "Do NOT auto-trade from this scanner — confirm on TradingView yourself",
        "Do NOT enter mid-range without tagging a graded zone + IF/THEN trigger",
        "Do NOT fight H1 bias on B-grade / AVOID_NOW cards",
    ]
    if not killzone:
        do_not.append("Outside killzone — reduce size or skip until London/NY")

    a_plus = sum(1 for c in cards if c.grade == "A+")
    a_cnt = sum(1 for c in cards if c.grade == "A")
    if buy and sell:
        summary = (
            f"stance={stance} A+={a_plus} A={a_cnt} | "
            f"best BUY @{buy[0].focus_price:.2f}[{buy[0].grade}] | "
            f"best SELL @{sell[0].focus_price:.2f}[{sell[0].grade}]"
        )
    elif buy:
        summary = f"stance={stance} A+={a_plus} A={a_cnt} | best BUY @{buy[0].focus_price:.2f}[{buy[0].grade}]"
    elif sell:
        summary = f"stance={stance} A+={a_plus} A={a_cnt} | best SELL @{sell[0].focus_price:.2f}[{sell[0].grade}]"
    else:
        summary = f"stance={stance} cards={len(cards)} A+={a_plus} A={a_cnt}"

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
    )


def render_manual_scan_block(report: ManualScanReport | None, mid: float = 0.0) -> list[str]:
    lines: list[str] = []
    lines.append("=" * 72)
    lines.append("  MANUAL HIGH-PROBABILITY SCANNER  (YOU decide · YOU trade · NO auto)")
    lines.append("  Grades = setup quality for TradingView confirmation — not a win-rate promise")
    lines.append("=" * 72)
    if report is None:
        lines.append("  No scan available.")
        return lines

    lines.append(f"  Mid           : {mid:.3f}")
    lines.append(f"  Stance        : {report.stance}  (H1={report.bias})")
    lines.append(f"  Plan          : {report.headline}")
    lines.append(f"  Snapshot      : {report.summary}")
    lines.append("-" * 72)
    lines.append("  WATCHLIST")
    for w in report.watchlist:
        lines.append(f"    • {w}")
    lines.append("-" * 72)
    lines.append("  DO NOT")
    for d in report.do_not:
        lines.append(f"    • {d}")

    def _emit(title: str, cards: list[SetupCard]) -> None:
        lines.append("-" * 72)
        lines.append(f"  {title}")
        if not cards:
            lines.append("     (none)")
            return
        for i, c in enumerate(cards, 1):
            lines.append(
                f"  {i}. [{c.grade}] {c.side}  {c.zone_low:.2f}-{c.zone_high:.2f}  "
                f"@{c.focus_price:.2f}  {c.kind}  score={c.score:.0f}  {c.status}"
            )
            lines.append(f"      IF/THEN : {c.if_then}")
            lines.append(f"      Invalid : {c.invalidation}")
            lines.append(f"      Avoid   : {c.avoid}")
            for t in c.confirm_on_tv[:3]:
                lines.append(f"      TV      · {t}")
            for r in c.reasons[:2]:
                lines.append(f"      Why     · {r}")

    _emit("BEST BUY AREAS  (long interest / buy liquidity / bull FVG-OB / support / OTE)", report.buy_cards)
    _emit("BEST SELL AREAS (short interest / sell liquidity / bear FVG-OB / resistance / OTE)", report.sell_cards)

    lines.append("-" * 72)
    lines.append("  HOW TO USE ON TRADINGVIEW")
    lines.append("    1) Draw only A+ and A zones first")
    lines.append("    2) Wait for the IF condition (sweep/tap) — do nothing before that")
    lines.append("    3) Enter only after THEN confirmation candle on M5")
    lines.append("    4) Honor invalidation — if hit, cancel the idea")
    lines.append("    5) Trade yourself — this scanner never sends orders")
    return lines
