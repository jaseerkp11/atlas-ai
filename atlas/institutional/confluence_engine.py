"""
Confluence engine — weighted multi-module institutional score.

Never forces a trade; low confluence → NO TRADE upstream.
"""

from __future__ import annotations

from dataclasses import dataclass

from atlas.institutional.models import Bias, ModuleScore


@dataclass
class ConfluenceResult:
    confluence: float
    probability: float
    confidence: float
    consensus_bias: Bias
    agreeing: int
    disagreeing: int
    modules: list[ModuleScore]
    summary: str


def compute_confluence(
    modules: list[ModuleScore],
    weights: dict[str, float],
) -> ConfluenceResult:
    # Apply configured weights when present
    scored: list[ModuleScore] = []
    for m in modules:
        w = float(weights.get(m.name, m.weight))
        scored.append(
            ModuleScore(m.name, m.score, w, m.bias, m.detail, m.passed)
        )

    total_w = sum(m.weight for m in scored) or 1.0
    confluence = sum(m.score * m.weight for m in scored) / total_w

    # Consensus bias by weighted vote
    bull = sum(m.weight for m in scored if m.bias == Bias.BULLISH)
    bear = sum(m.weight for m in scored if m.bias == Bias.BEARISH)
    if bull > bear * 1.15:
        consensus = Bias.BULLISH
    elif bear > bull * 1.15:
        consensus = Bias.BEARISH
    else:
        consensus = Bias.NEUTRAL

    agreeing = sum(1 for m in scored if m.bias == consensus and consensus != Bias.NEUTRAL)
    disagreeing = sum(
        1
        for m in scored
        if m.bias != Bias.NEUTRAL and consensus != Bias.NEUTRAL and m.bias != consensus
    )

    # Probability: confluence scaled by agreement ratio
    agree_ratio = agreeing / max(1, agreeing + disagreeing)
    probability = confluence * (0.55 + 0.45 * agree_ratio)
    if consensus == Bias.NEUTRAL:
        probability *= 0.5

    # Confidence: how tight module scores + agreement
    conf = min(100.0, probability * 0.7 + agree_ratio * 30 + (10 if agreeing >= 4 else 0))

    summary = (
        f"confluence={confluence:.1f} prob={probability:.1f} conf={conf:.1f} "
        f"bias={consensus.value} agree={agreeing} disagree={disagreeing}"
    )
    return ConfluenceResult(
        confluence=confluence,
        probability=probability,
        confidence=conf,
        consensus_bias=consensus,
        agreeing=agreeing,
        disagreeing=disagreeing,
        modules=scored,
        summary=summary,
    )
