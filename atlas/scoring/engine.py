"""
Signal scoring & plain-language reasoning.

Every setup is scored out of 100 with clearly weighted factors.
Rejected and accepted setups both emit full reasoning.
Trade gates (min score, min RR) are enforced via atlas.config.get_gates() only.
"""

from __future__ import annotations

from datetime import datetime, timezone

from atlas.config import ScoringWeights, get_gates, load_settings
from atlas.models import (
    Direction,
    ScoreFactor,
    ScoreResult,
    SetupFeatures,
    TradeDecision,
    TrendStrength,
)


def score_setup(features: SetupFeatures) -> ScoreResult:
    """Score a setup using configured weights. Returns detailed factor breakdown."""
    weights = load_settings().weights
    factors: list[ScoreFactor] = []

    factors.append(_score_h4(features, weights))
    factors.append(_score_h1(features, weights))
    factors.append(_score_structure(features, weights))
    factors.append(_score_liquidity(features, weights))
    factors.append(_score_fvg(features, weights))
    factors.append(_score_order_block(features, weights))
    factors.append(_score_m5_trigger(features, weights))
    factors.append(_score_volatility(features, weights))

    total = sum(f.earned for f in factors)
    passed = [f.name for f in factors if f.passed]
    failed = [f.name for f in factors if not f.passed]
    plain = [f"{'PASS' if f.passed else 'FAIL'}: {f.name} — {f.reason}" for f in factors]

    return ScoreResult(
        total=total,
        max_total=100,
        factors=factors,
        passed_factors=passed,
        failed_factors=failed,
        plain_language=plain,
    )


def evaluate_setup(features: SetupFeatures) -> TradeDecision:
    """
    Score + enforce gates in one place.

    Always returns a TradeDecision with full reasoning, whether allowed or not.
    """
    score = score_setup(features)
    gates = get_gates()
    allowed_by_gates, gate_failures = gates.allows(score.total, features.reward_risk)

    rejection: list[str] = []
    if features.direction == Direction.NEUTRAL:
        rejection.append("No clear directional bias across timeframes")
        allowed_by_gates = False
        gate_failures = list(gate_failures) + rejection

    if features.entry <= 0 or features.stop <= 0 or features.target <= 0:
        msg = "Invalid entry/stop/target levels"
        rejection.append(msg)
        allowed_by_gates = False
        if msg not in gate_failures:
            gate_failures = list(gate_failures) + [msg]

    # Directional consistency of stop/target
    if features.direction == Direction.LONG and not (
        features.stop < features.entry < features.target
    ):
        msg = "LONG levels invalid (need stop < entry < target)"
        rejection.append(msg)
        allowed_by_gates = False
        gate_failures = list(gate_failures) + [msg]
    if features.direction == Direction.SHORT and not (
        features.target < features.entry < features.stop
    ):
        msg = "SHORT levels invalid (need target < entry < stop)"
        rejection.append(msg)
        allowed_by_gates = False
        gate_failures = list(gate_failures) + [msg]

    return TradeDecision(
        symbol=features.symbol,
        direction=features.direction,
        score=score,
        features=features,
        allowed=allowed_by_gates and features.direction != Direction.NEUTRAL,
        gate_failures=gate_failures,
        rejection_reasons=rejection,
        evaluated_at=datetime.now(timezone.utc),
    )


def _score_h4(features: SetupFeatures, w: ScoringWeights) -> ScoreFactor:
    name = "H4 regime alignment"
    weight = w.h4_regime_alignment
    h4 = features.h4
    if h4 is None:
        return ScoreFactor(name, weight, 0, False, "H4 data unavailable")
    if features.direction == Direction.NEUTRAL:
        return ScoreFactor(name, weight, 0, False, "No trade direction to align with")
    if h4.direction == features.direction:
        earned = weight
        if h4.strength == TrendStrength.WEAK:
            earned = max(1, weight // 2)
        return ScoreFactor(
            name,
            weight,
            earned,
            True,
            f"H4 {h4.direction.value} ({h4.strength.value}) aligns with setup"
            + (f" — partial credit {earned}/{weight}" if earned < weight else ""),
        )
    if h4.direction == Direction.NEUTRAL:
        return ScoreFactor(name, weight, 0, False, "H4 is neutral — no regime bias")
    return ScoreFactor(
        name,
        weight,
        0,
        False,
        f"H4 is {h4.direction.value}, conflicts with {features.direction.value}",
    )


def _score_h1(features: SetupFeatures, w: ScoringWeights) -> ScoreFactor:
    name = "H1 trend confirmation"
    weight = w.h1_trend_confirmation
    h1 = features.h1
    if h1 is None:
        return ScoreFactor(name, weight, 0, False, "H1 data unavailable")
    if features.direction == Direction.NEUTRAL:
        return ScoreFactor(name, weight, 0, False, "No trade direction to confirm")
    if h1.direction == features.direction:
        earned = weight
        if h1.strength == TrendStrength.WEAK:
            earned = max(1, weight // 2)
        return ScoreFactor(
            name,
            weight,
            earned,
            True,
            f"H1 {h1.direction.value} ({h1.strength.value}) confirms setup"
            + (f" — partial credit {earned}/{weight}" if earned < weight else ""),
        )
    if h1.direction == Direction.NEUTRAL:
        return ScoreFactor(name, weight, 0, False, "H1 is neutral — no trend confirmation")
    return ScoreFactor(
        name,
        weight,
        0,
        False,
        f"H1 is {h1.direction.value}, conflicts with {features.direction.value}",
    )


def _score_structure(features: SetupFeatures, w: ScoringWeights) -> ScoreFactor:
    name = "Structure break (BOS/CHoCH)"
    weight = w.structure_break
    if features.bos and features.choch:
        return ScoreFactor(
            name, weight, weight, True, "Both BOS and CHoCH confirmed in trade direction"
        )
    if features.bos:
        return ScoreFactor(name, weight, weight, True, "Break of structure confirmed")
    if features.choch:
        # CHoCH alone earns partial — early reversal, less confirmed
        earned = max(1, (weight * 2) // 3)
        return ScoreFactor(
            name,
            weight,
            earned,
            True,
            f"CHoCH present but no confirmed BOS yet — partial credit {earned}/{weight}",
        )
    return ScoreFactor(name, weight, 0, False, "No BOS or CHoCH in trade direction")


def _score_liquidity(features: SetupFeatures, w: ScoringWeights) -> ScoreFactor:
    name = "Liquidity sweep"
    weight = w.liquidity_sweep
    sweep = features.liquidity_sweep
    if sweep is None:
        return ScoreFactor(name, weight, 0, False, "No liquidity sweep detected")
    if sweep.direction != features.direction:
        return ScoreFactor(
            name,
            weight,
            0,
            False,
            f"Sweep direction {sweep.direction.value} does not match setup",
        )
    return ScoreFactor(
        name,
        weight,
        weight,
        True,
        f"Swept {sweep.swept_kind} at {sweep.level:.5f}, reclaim supports {sweep.direction.value}",
    )


def _score_fvg(features: SetupFeatures, w: ScoringWeights) -> ScoreFactor:
    name = "Fair value gap"
    weight = w.fair_value_gap
    if not features.fair_value_gaps:
        return ScoreFactor(name, weight, 0, False, "No unfilled FVG in trade direction")
    g = features.fair_value_gaps[-1]
    return ScoreFactor(
        name,
        weight,
        weight,
        True,
        f"Unfilled {g.direction.value} FVG zone {g.bottom:.5f}–{g.top:.5f}",
    )


def _score_order_block(features: SetupFeatures, w: ScoringWeights) -> ScoreFactor:
    name = "Order block"
    weight = w.order_block
    if not features.order_blocks:
        return ScoreFactor(name, weight, 0, False, "No active order block in trade direction")
    ob = features.order_blocks[-1]
    return ScoreFactor(
        name,
        weight,
        weight,
        True,
        f"Active {ob.direction.value} OB {ob.bottom:.5f}–{ob.top:.5f}",
    )


def _score_m5_trigger(features: SetupFeatures, w: ScoringWeights) -> ScoreFactor:
    name = "M5 trigger confirmation"
    weight = w.m5_trigger
    if features.m5_trigger:
        return ScoreFactor(name, weight, weight, True, "M5 trigger confirmed in trade direction")
    return ScoreFactor(name, weight, 0, False, "M5 trigger not confirmed")


def _score_volatility(features: SetupFeatures, w: ScoringWeights) -> ScoreFactor:
    name = "Volatility context"
    weight = w.volatility_context
    if features.atr <= 0:
        return ScoreFactor(name, weight, 0, False, "ATR unavailable")
    if features.volatility_expanding:
        return ScoreFactor(
            name, weight, weight, True, f"ATR expanding (ATR={features.atr:.5f})"
        )
    # Partial credit if ATR exists but not expanding — still usable for stops
    earned = max(1, weight // 2)
    return ScoreFactor(
        name,
        weight,
        earned,
        True,
        f"ATR available ({features.atr:.5f}) but not expanding — partial credit {earned}/{weight}",
    )
