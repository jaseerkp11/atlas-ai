"""
High-probability scenario layer.

Builds a ranked institutional checklist and next-trigger plan so every
M5 close returns actionable analysis even when the decision is NO_TRADE.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from atlas.institutional.models import Bias, MarketNarrative
from atlas.institutional.playbook_engine import PlaybookResult


@dataclass
class ChecklistItem:
    name: str
    ok: bool
    detail: str


@dataclass
class ScenarioPlan:
    primary: str
    alternate: str
    invalidation: str
    next_triggers: list[str]
    checklist: list[ChecklistItem]
    edge_score: float  # 0–100 composite of gates+playbooks
    summary: str

    def as_dict(self) -> dict:
        return {
            "primary": self.primary,
            "alternate": self.alternate,
            "invalidation": self.invalidation,
            "next_triggers": self.next_triggers,
            "checklist": [{"name": c.name, "ok": c.ok, "detail": c.detail} for c in self.checklist],
            "edge_score": self.edge_score,
            "summary": self.summary,
        }


def build_scenario(
    narrative: MarketNarrative,
    playbooks: PlaybookResult | None,
    h1_bias: Bias,
    h4_bias: Bias,
    m15_bias: Bias,
    h1_aligned: bool,
    session_ok: bool,
    news_block: bool,
    probability: float,
    confluence: float,
    confidence: float,
    min_probability: float,
    min_confluence: float,
    min_confidence: float,
) -> ScenarioPlan:
    checklist = [
        ChecklistItem(
            "H1 primary bias",
            h1_bias != Bias.NEUTRAL,
            f"H1={h1_bias.value} (primary HTF for XAUUSD intraday)",
        ),
        ChecklistItem(
            "H1 ↔ consensus aligned",
            h1_aligned,
            "Trade only with H1 — H4 soft/transitional OK",
        ),
        ChecklistItem(
            "H1 + M15 stack",
            h1_bias != Bias.NEUTRAL and h1_bias == m15_bias,
            f"H1={h1_bias.value} M15={m15_bias.value}",
        ),
        ChecklistItem(
            "Killzone / session",
            session_ok,
            f"Best session={narrative.best_session}",
        ),
        ChecklistItem(
            "News window clear",
            not news_block,
            narrative.news_status,
        ),
        ChecklistItem(
            "Playbook active",
            bool(playbooks and playbooks.best and playbooks.total_boost >= 10),
            playbooks.summary if playbooks else "none",
        ),
        ChecklistItem(
            f"Probability ≥ {min_probability:.0f}%",
            probability >= min_probability,
            f"{probability:.1f}%",
        ),
        ChecklistItem(
            f"Confluence ≥ {min_confluence:.0f}",
            confluence >= min_confluence,
            f"{confluence:.1f}",
        ),
        ChecklistItem(
            f"Confidence ≥ {min_confidence:.0f}%",
            confidence >= min_confidence,
            f"{confidence:.1f}%",
        ),
    ]

    passed = sum(1 for c in checklist if c.ok)
    edge = (passed / max(1, len(checklist))) * 100.0
    if playbooks and playbooks.total_boost:
        edge = min(100.0, edge + min(12.0, playbooks.total_boost * 0.35))

    bias = narrative.overall_bias
    if bias == Bias.NEUTRAL and h1_bias != Bias.NEUTRAL:
        bias = h1_bias

    if bias == Bias.BULLISH:
        primary = (
            "LONG bias: wait for discount pullback / OTE / FVG fill with H1 bullish + playbook"
        )
        alternate = "If H1 flips bearish with premium reject → flip to short plan"
        invalidation = "Invalidation: H1 CHoCH bearish or break below active bullish OB/FVG"
    elif bias == Bias.BEARISH:
        primary = (
            "SHORT bias: wait for premium rally / OTE / FVG fill with H1 bearish + playbook"
        )
        alternate = "If H1 flips bullish with discount reclaim → flip to long plan"
        invalidation = "Invalidation: H1 CHoCH bullish or break above active bearish OB/FVG"
    else:
        primary = "NO directional edge — stand aside until H1 bias clarifies"
        alternate = "Watch for sweep+reclaim or BOS/CHoCH to define bias"
        invalidation = "No trade plan active — avoid mid-range chop"

    triggers: list[str] = []
    if playbooks and playbooks.unlock_hints:
        triggers.extend(playbooks.unlock_hints[:4])
    missing = [c for c in checklist if not c.ok]
    for c in missing[:3]:
        triggers.append(f"Clear gate: {c.name} ({c.detail})")
    if not triggers:
        triggers.append("All high-probability gates green — wait M1 timing candle")

    if playbooks and playbooks.best:
        summary = (
            f"edge={edge:.0f}/100 | best={playbooks.best.name} | "
            f"gates={passed}/{len(checklist)} | H4={h4_bias.value} H1={h1_bias.value}"
        )
    else:
        summary = (
            f"edge={edge:.0f}/100 | no playbook | "
            f"gates={passed}/{len(checklist)} | H4={h4_bias.value} H1={h1_bias.value}"
        )

    return ScenarioPlan(
        primary=primary,
        alternate=alternate,
        invalidation=invalidation,
        next_triggers=triggers[:6],
        checklist=checklist,
        edge_score=edge,
        summary=summary,
    )
