"""
Confluence engine — weighted scores + institutional playbook probability boost.

Never forces a trade; low confluence / no playbook → NO TRADE upstream.
"""

from __future__ import annotations

from dataclasses import dataclass

from atlas.institutional.models import Bias, ModuleScore
from atlas.institutional.playbook_engine import PlaybookResult


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
    playbook_boost: float = 0.0
    playbook_name: str = ""


def compute_confluence(
    modules: list[ModuleScore],
    weights: dict[str, float],
    playbooks: PlaybookResult | None = None,
    h1_bias: Bias = Bias.NEUTRAL,
) -> ConfluenceResult:
    scored: list[ModuleScore] = []
    for m in modules:
        w = float(weights.get(m.name, m.weight))
        scored.append(ModuleScore(m.name, m.score, w, m.bias, m.detail, m.passed))

    total_w = sum(m.weight for m in scored) or 1.0
    confluence = sum(m.score * m.weight for m in scored) / total_w

    bull = sum(m.weight for m in scored if m.bias == Bias.BULLISH)
    bear = sum(m.weight for m in scored if m.bias == Bias.BEARISH)

    # H1 bias gets a soft tie-break (world standard: trade with H1 for gold intraday)
    if h1_bias == Bias.BULLISH:
        bull += 6
    elif h1_bias == Bias.BEARISH:
        bear += 6

    if playbooks and playbooks.best:
        if playbooks.best.bias == Bias.BULLISH:
            bull += playbooks.total_boost * 0.35
        elif playbooks.best.bias == Bias.BEARISH:
            bear += playbooks.total_boost * 0.35

    if bull > bear * 1.12:
        consensus = Bias.BULLISH
    elif bear > bull * 1.12:
        consensus = Bias.BEARISH
    else:
        consensus = Bias.NEUTRAL

    # If strong playbook, allow it to set consensus when modules are mixed
    if consensus == Bias.NEUTRAL and playbooks and playbooks.best and playbooks.total_boost >= 14:
        consensus = playbooks.best.bias

    agreeing = sum(1 for m in scored if m.bias == consensus and consensus != Bias.NEUTRAL)
    disagreeing = sum(
        1
        for m in scored
        if m.bias != Bias.NEUTRAL and consensus != Bias.NEUTRAL and m.bias != consensus
    )

    agree_ratio = agreeing / max(1, agreeing + disagreeing)
    probability = confluence * (0.50 + 0.50 * agree_ratio)

    boost = 0.0
    pb_name = ""
    if playbooks and playbooks.best and playbooks.best.bias == consensus:
        boost = playbooks.total_boost
        pb_name = playbooks.best.name
        probability = min(96.0, probability + boost)
    elif playbooks and playbooks.best and consensus == Bias.NEUTRAL:
        # Weak lift only — still need more agreement
        boost = playbooks.total_boost * 0.35
        pb_name = playbooks.best.name
        probability = min(70.0, probability + boost)

    if consensus == Bias.NEUTRAL:
        probability *= 0.55

    # Penalize fighting H1
    if h1_bias != Bias.NEUTRAL and consensus != Bias.NEUTRAL and h1_bias != consensus:
        probability *= 0.65

    conf = min(
        100.0,
        probability * 0.65
        + agree_ratio * 28
        + (12 if agreeing >= 4 else 0)
        + (8 if boost >= 14 else 0),
    )

    summary = (
        f"confluence={confluence:.1f} prob={probability:.1f} conf={conf:.1f} "
        f"bias={consensus.value} agree={agreeing}/{disagreeing} "
        f"playbook={pb_name or 'none'}(+{boost:.0f})"
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
        playbook_boost=boost,
        playbook_name=pb_name,
    )
