"""
Multi-factor high-probability setup scorer (manual chart plan — not a win guarantee).

Stacks independent chart conditions that historically improve selectivity:
  HTF bias, M15 agree, PD array, killzone, session levels, sweep/reclaim,
  zone type (liq/FVG/OB/OTE), freshness, fib/trendline overlap, playbook,
  challenge R:R fit, proximity, news, board lean, structure/displacement.

Score is 0–100. Use as a filter: prefer ≥70; A+ usually needs ≥78.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from atlas.institutional.models import Bias, MarketNarrative
from atlas.institutional.playbook_engine import PlaybookResult
from atlas.institutional.trade_areas import TradeArea


@dataclass
class ProbFactor:
    name: str
    points: float
    max_points: float
    passed: bool
    detail: str

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class SetupProbability:
    score: float  # 0–100
    tier: str  # ELITE | HIGH | MEDIUM | LOW | AVOID
    factors: list[ProbFactor] = field(default_factory=list)
    summary: str = ""

    def as_dict(self) -> dict:
        return {
            "score": self.score,
            "tier": self.tier,
            "factors": [f.as_dict() for f in self.factors],
            "summary": self.summary,
        }


def _near_price(price: float, mid: float, atr: float, max_atr: float = 0.45) -> bool:
    return abs(price - mid) / max(atr, 1e-9) <= max_atr


def _session_levels_near(
    extras: dict[str, Any],
    zone_lo: float,
    zone_hi: float,
    atr: float,
) -> list[str]:
    sess = extras.get("session_levels") or {}
    hits: list[str] = []
    for key, label in (
        ("pdh", "PDH"),
        ("pdl", "PDL"),
        ("asian_high", "Asian High"),
        ("asian_low", "Asian Low"),
        ("london_high", "London High"),
        ("london_low", "London Low"),
    ):
        raw = sess.get(key)
        if raw is None:
            continue
        try:
            px = float(raw)
        except (TypeError, ValueError):
            continue
        # Level inside or kissing the zone
        if zone_lo - atr * 0.25 <= px <= zone_hi + atr * 0.25:
            hits.append(label)
        elif _near_price(px, (zone_lo + zone_hi) / 2.0, atr, 0.35):
            hits.append(label)
    return hits


def _fib_near_zone(
    narrative: MarketNarrative,
    zone_lo: float,
    zone_hi: float,
    atr: float,
    side: str,
) -> bool:
    mid_z = (zone_lo + zone_hi) / 2.0
    for lv in narrative.fib_levels or []:
        if not (0.60 <= float(lv.ratio) <= 0.79):
            continue
        if abs(float(lv.price) - mid_z) / max(atr, 1e-9) > 0.40:
            continue
        # OTE should match side intent when classification known
        if side == "BUY" and "premium" in (lv.classification or "").lower():
            continue
        if side == "SELL" and "discount" in (lv.classification or "").lower():
            continue
        return True
    return False


def _trendline_near(
    extras: dict[str, Any],
    zone_lo: float,
    zone_hi: float,
    atr: float,
    side: str,
) -> bool:
    want = "support" if side == "BUY" else "resistance"
    for ln in extras.get("trendlines") or []:
        kind = str(ln.get("kind", "")).lower()
        quality = str(ln.get("quality", ""))
        if quality == "Broken":
            continue
        if kind != want:
            continue
        try:
            px = float(ln.get("price", 0))
        except (TypeError, ValueError):
            continue
        if zone_lo - atr * 0.3 <= px <= zone_hi + atr * 0.3:
            return True
    return False


def _module_bias(narrative: MarketNarrative, name: str) -> Bias:
    for m in narrative.module_scores or []:
        if m.name == name:
            return m.bias
    return Bias.NEUTRAL


def _module_score(narrative: MarketNarrative, name: str) -> float:
    for m in narrative.module_scores or []:
        if m.name == name:
            return float(m.score)
    return 0.0


def score_setup_probability(
    area: TradeArea,
    *,
    narrative: MarketNarrative,
    h1_bias: Bias,
    h4_bias: Bias,
    m15_bias: Bias,
    session_name: str,
    sweep_state: str,
    reward_risk: float,
    challenge_ok: bool,
    playbooks: PlaybookResult | None,
    board_lean: str = "BALANCED",
    against_htf: bool = False,
) -> SetupProbability:
    """
    Score one graded area using stacked chart factors.
    against_htf → hard AVOID path (still returns LOW score for map).
    """
    atr = max(float(narrative.atr_m15 or 0.0), 1e-9)
    mid = float(narrative.mid)
    extras = narrative.extras or {}
    lo = min(float(area.price_low), float(area.price_high))
    hi = max(float(area.price_low), float(area.price_high))
    side = area.side
    factors: list[ProbFactor] = []

    def add(name: str, pts: float, mx: float, ok: bool, detail: str) -> None:
        factors.append(ProbFactor(name, pts if ok else 0.0, mx, ok, detail))

    # --- Hard veto context ---
    if against_htf:
        add("HTF alignment", 0, 14, False, f"side={side} fights H1={h1_bias.value}")
        total = sum(f.points for f in factors)
        return SetupProbability(
            score=max(0.0, min(35.0, total)),
            tier="AVOID",
            factors=factors,
            summary="Against H1 — map only, do not take",
        )

    # 1) H1 bias (primary)
    h1_ok = (side == "BUY" and h1_bias == Bias.BULLISH) or (
        side == "SELL" and h1_bias == Bias.BEARISH
    )
    add("H1 bias aligned", 14.0 if h1_ok else 0.0, 14.0, h1_ok, f"H1={h1_bias.value}")

    # 2) H4 stack
    h4_ok = (side == "BUY" and h4_bias == Bias.BULLISH) or (
        side == "SELL" and h4_bias == Bias.BEARISH
    )
    h4_neutral = h4_bias == Bias.NEUTRAL
    add(
        "H4 agrees",
        8.0 if h4_ok else (3.0 if h4_neutral else 0.0),
        8.0,
        h4_ok or h4_neutral,
        f"H4={h4_bias.value}",
    )

    # 3) M15 structure agree
    m15_ok = (side == "BUY" and m15_bias == Bias.BULLISH) or (
        side == "SELL" and m15_bias == Bias.BEARISH
    )
    add("M15 agrees", 7.0 if m15_ok else 0.0, 7.0, m15_ok, f"M15={m15_bias.value}")

    # 4) Premium / Discount
    pd = str(extras.get("pd_zone", "") or "")
    if side == "BUY":
        pd_ok = pd in ("discount", "equilibrium", "")
        pd_detail = f"array={pd or '?'} (want discount/eq)"
    else:
        pd_ok = pd in ("premium", "equilibrium", "")
        pd_detail = f"array={pd or '?'} (want premium/eq)"
    pd_pts = 8.0 if pd in ("discount", "premium") and pd_ok else (5.0 if pd_ok else 0.0)
    add("Premium/Discount", pd_pts, 8.0, pd_ok, pd_detail)

    # 5) Killzone
    killzone = session_name in ("London", "NewYork", "London-NY Overlap")
    kz_pts = 8.0 if session_name == "London-NY Overlap" else (6.0 if killzone else 0.0)
    add("Killzone session", kz_pts, 8.0, killzone, f"session={session_name}")

    # 6) Session level magnet (PDH/PDL/Asian/London)
    sess_hits = _session_levels_near(extras, lo, hi, atr)
    sess_ok = bool(sess_hits)
    add(
        "Session level confluence",
        7.0 if sess_ok else 0.0,
        7.0,
        sess_ok,
        ", ".join(sess_hits) if sess_hits else "none near zone",
    )

    # 7) Sweep / reclaim SMC sequence
    if sweep_state == "RECLAIMED":
        sw_pts, sw_ok = 10.0, True
    elif sweep_state == "IN_ZONE" and area.kind not in ("buy_liquidity", "sell_liquidity"):
        sw_pts, sw_ok = 6.0, True
    elif sweep_state == "SWEPT":
        sw_pts, sw_ok = 3.0, False
    else:
        sw_pts, sw_ok = 0.0, False
    add("Sweep/reclaim state", sw_pts, 10.0, sw_ok, f"state={sweep_state}")

    # 8) Zone type quality
    elite_kinds = {
        "buy_liquidity",
        "sell_liquidity",
        "bullish_fvg",
        "bearish_fvg",
        "bullish_ob",
        "bearish_ob",
        "fib_ote",
    }
    good_kinds = elite_kinds | {"htf_support", "htf_resistance", "support", "resistance"}
    if area.kind in ("buy_liquidity", "sell_liquidity", "bullish_fvg", "bearish_fvg", "bullish_ob", "bearish_ob"):
        kind_pts = 8.0
    elif area.kind == "fib_ote":
        kind_pts = 7.0
    elif area.kind.startswith("htf_"):
        kind_pts = 5.0
    elif area.kind in ("support", "resistance"):
        kind_pts = 3.0
    else:
        kind_pts = 1.0
    add("Zone type", kind_pts, 8.0, area.kind in good_kinds, f"kind={area.kind}")

    # 9) Fresh / unfilled
    fresh_ok = bool(area.fresh_unfilled)
    add("Fresh/unfilled", 5.0 if fresh_ok else 0.0, 5.0, fresh_ok, f"fresh={fresh_ok}")

    # 10) Fib OTE overlap
    fib_ok = _fib_near_zone(narrative, lo, hi, atr, side)
    add("Fib OTE overlap", 5.0 if fib_ok else 0.0, 5.0, fib_ok, "OTE near zone" if fib_ok else "no OTE")

    # 11) Trendline confluence
    tl_ok = _trendline_near(extras, lo, hi, atr, side)
    add("Trendline confluence", 4.0 if tl_ok else 0.0, 4.0, tl_ok, "TL touch" if tl_ok else "no TL")

    # 12) Playbook match
    pb_ok = False
    pb_name = ""
    pb_pts = 0.0
    if playbooks and playbooks.hits:
        for h in playbooks.hits:
            if (side == "BUY" and h.bias == Bias.BULLISH) or (
                side == "SELL" and h.bias == Bias.BEARISH
            ):
                pb_ok = True
                pb_name = h.name
                pb_pts = min(8.0, 4.0 + float(h.boost) * 0.2)
                break
    add("Playbook match", pb_pts, 8.0, pb_ok, pb_name or "no matching playbook")

    # 13) Challenge R:R + stop fit
    rr_ok = reward_risk >= 2.0 and challenge_ok
    if challenge_ok and reward_risk >= 3.0:
        rr_pts = 8.0
    elif challenge_ok and reward_risk >= 2.0:
        rr_pts = 6.0
    elif challenge_ok:
        rr_pts = 2.0
    else:
        rr_pts = 0.0
    add(
        "Challenge R:R fit",
        rr_pts,
        8.0,
        rr_ok,
        f"R:R=1:{reward_risk:.2f} challenge_ok={challenge_ok}",
    )

    # 14) Proximity (actionable, not mid-chop far away)
    dist = float(area.distance_atr)
    if dist <= 0.50:
        dist_pts, dist_ok = 5.0, True
    elif dist <= 1.20:
        dist_pts, dist_ok = 4.0, True
    elif dist <= 2.00:
        dist_pts, dist_ok = 2.0, True
    else:
        dist_pts, dist_ok = 0.0, False
    add("Proximity to price", dist_pts, 5.0, dist_ok, f"dist={dist:.2f} ATR")

    # 15) News clear
    news_status = str(getattr(narrative, "news_status", "") or "")
    news_block = bool(extras.get("news_block"))
    news_ok = (not news_block) and news_status not in ("Avoid Trading Today",)
    news_pts = 4.0 if news_ok and news_status != "Trade With Caution" else (
        2.0 if news_ok else 0.0
    )
    add("News clear", news_pts, 4.0, news_ok, news_status or "ok")

    # 16) Board lean not fighting
    if side == "BUY":
        board_ok = board_lean != "SELL_LEAN"
    else:
        board_ok = board_lean != "BUY_LEAN"
    add("Board not fighting", 4.0 if board_ok else 0.0, 4.0, board_ok, f"board={board_lean}")

    # 17) Structure / liquidity / PA module support
    struct_bias = _module_bias(narrative, "market_structure")
    liq_sc = _module_score(narrative, "liquidity")
    pa_sc = _module_score(narrative, "price_action")
    struct_ok = (side == "BUY" and struct_bias == Bias.BULLISH) or (
        side == "SELL" and struct_bias == Bias.BEARISH
    )
    mod_pts = 0.0
    if struct_ok:
        mod_pts += 3.0
    if liq_sc >= 60:
        mod_pts += 2.0
    if pa_sc >= 60:
        mod_pts += 2.0
    add(
        "Structure/liq/PA modules",
        mod_pts,
        7.0,
        mod_pts >= 3.0,
        f"struct={struct_bias.value} liq={liq_sc:.0f} pa={pa_sc:.0f}",
    )

    # 18) Reason overlap / confluence tags on area
    overlap_n = sum(1 for r in area.reasons if "overlap" in r.lower() or "confluence" in r.lower())
    ov_ok = overlap_n > 0 or ("+ overlap" in " ".join(area.reasons))
    # Also count multi-reason density
    if len(area.reasons) >= 4:
        ov_ok = True
    add(
        "Multi-factor overlap",
        4.0 if ov_ok else 0.0,
        4.0,
        ov_ok,
        f"reasons={len(area.reasons)} overlaps={overlap_n}",
    )

    raw = sum(f.points for f in factors)
    # Max theoretical ~119; normalize softly into 0–100
    score = max(0.0, min(100.0, raw * (100.0 / 110.0)))
    passed_n = sum(1 for f in factors if f.passed)
    if score >= 78 and passed_n >= 10 and h1_ok and rr_ok:
        tier = "ELITE"
    elif score >= 70 and h1_ok:
        tier = "HIGH"
    elif score >= 55:
        tier = "MEDIUM"
    else:
        tier = "LOW"

    top = sorted([f for f in factors if f.passed], key=lambda x: -x.points)[:4]
    summary = (
        f"{tier} {score:.0f}/100 · {passed_n}/{len(factors)} factors · "
        + ", ".join(f.name for f in top)
    )
    return SetupProbability(score=round(score, 1), tier=tier, factors=factors, summary=summary)


def grade_from_probability(
    base_grade: str,
    status: str,
    prob: SetupProbability,
    challenge_ok: bool,
) -> str:
    """Upgrade/downgrade letter grade from multi-factor probability."""
    if status == "AVOID_NOW" or prob.tier == "AVOID":
        return "B"
    if not challenge_ok and base_grade == "A+":
        base_grade = "A"
    if prob.tier == "ELITE" and challenge_ok and status in (
        "WAIT_FOR_TRIGGER",
        "READY_TO_WATCH",
        "WAIT_FOR_RECLAIM",
        "WAIT_FOR_ZONE",
    ):
        return "A+"
    if prob.score >= 70 and base_grade == "B" and status != "AVOID_NOW":
        return "A"
    if prob.score < 55 and base_grade == "A+":
        return "A"
    if prob.score < 45 and base_grade in ("A+", "A"):
        return "B"
    if prob.tier == "HIGH" and base_grade == "B" and challenge_ok:
        return "A"
    return base_grade
