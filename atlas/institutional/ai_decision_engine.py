"""
AI Decision Engine — high-probability institutional analyst.

Defaults to NO TRADE / WAIT.
Uses H1 as primary HTF (H4 soft filter) + playbook unlock checklist.
"""

from __future__ import annotations

from atlas.analysis.volatility import latest_atr
from atlas.institutional.config import InstitutionalConfig
from atlas.institutional.confluence_engine import ConfluenceResult
from atlas.institutional.models import (
    Bias,
    DecisionAction,
    InstitutionalDecision,
    MarketNarrative,
    RiskLevel,
)
from atlas.institutional.playbook_engine import PlaybookResult
from atlas.institutional.price_action import PriceActionReport
import pandas as pd


def _risk_level(probability: float, news_block: bool, playbook: bool) -> RiskLevel:
    if news_block:
        return RiskLevel.EXTREME
    if probability >= 82 and playbook:
        return RiskLevel.LOW
    if probability >= 72:
        return RiskLevel.MEDIUM
    if probability >= 58:
        return RiskLevel.HIGH
    return RiskLevel.EXTREME


def decide(
    cfg: InstitutionalConfig,
    narrative: MarketNarrative,
    confluence: ConfluenceResult,
    m15: pd.DataFrame | None,
    m1: pd.DataFrame | None,
    pa: PriceActionReport,
    news_block: bool,
    session_ok: bool,
    htf_aligned: bool,
    h1_aligned: bool,
    playbooks: PlaybookResult | None = None,
) -> InstitutionalDecision:
    why: list[str] = []
    unlock = list(playbooks.unlock_hints) if playbooks else []

    def _blocked(action: DecisionAction, extra: list[str], risk: RiskLevel) -> InstitutionalDecision:
        hints = unlock[:4]
        if hints:
            extra = extra + ["Unlock path:"] + [f"  → {h}" for h in hints]
        if playbooks and playbooks.summary:
            extra.append(f"Playbooks: {playbooks.summary}")
        return InstitutionalDecision(
            action=action,
            why=extra,
            confidence=confluence.confidence,
            probability=confluence.probability,
            confluence=confluence.confluence,
            risk_level=risk,
            htf_confirm=htf_aligned or h1_aligned,
            ltf_confirm=False,
            narrative=narrative,
        )

    if news_block:
        return _blocked(
            DecisionAction.NO_TRADE,
            ["News filter: avoid trading window"],
            RiskLevel.EXTREME,
        )

    if confluence.consensus_bias == Bias.NEUTRAL:
        return _blocked(
            DecisionAction.WAIT,
            ["No institutional consensus — modules conflict", confluence.summary],
            RiskLevel.HIGH,
        )

    # HTF gate: H1 primary for XAUUSD (H4 soft — transitional H4 allowed)
    if cfg.require_htf_alignment and not h1_aligned:
        return _blocked(
            DecisionAction.NO_TRADE,
            [
                "H1 not aligned with trade bias — quality gate failed",
                f"H1 must agree with {confluence.consensus_bias.value} (primary HTF for XAUUSD)",
            ],
            RiskLevel.HIGH,
        )
    if cfg.require_htf_alignment and not htf_aligned:
        why.append("H4 transitional/conflict — allowed with H1 alignment (soft HTF)")

    if not session_ok:
        why.append("Session quality soft warning — prefer London/NY")

    # Prefer at least one playbook for EXECUTE (higher probability AI)
    require_pb = True
    if require_pb and (not playbooks or not playbooks.best or playbooks.total_boost < 10):
        return _blocked(
            DecisionAction.WAIT,
            [
                "No high-probability playbook active yet",
                confluence.summary,
                "Standing aside beats forcing a low-edge entry",
            ],
            RiskLevel.MEDIUM,
        )

    if confluence.confluence < cfg.min_confluence and confluence.playbook_boost < 14:
        return _blocked(
            DecisionAction.NO_TRADE,
            [f"Confluence {confluence.confluence:.1f} < min {cfg.min_confluence}"],
            _risk_level(confluence.probability, False, False),
        )

    # With strong playbook, allow slightly softer confluence
    eff_min_prob = cfg.min_probability
    if confluence.playbook_boost >= 16:
        eff_min_prob = max(68.0, cfg.min_probability - 4)

    if confluence.probability < eff_min_prob:
        return _blocked(
            DecisionAction.NO_TRADE,
            [f"Probability {confluence.probability:.1f}% < min {eff_min_prob}%"],
            _risk_level(confluence.probability, False, False),
        )

    if confluence.confidence < cfg.min_confidence:
        return _blocked(
            DecisionAction.WAIT,
            [f"Confidence {confluence.confidence:.1f}% < min {cfg.min_confidence}%"],
            RiskLevel.MEDIUM,
        )

    # M1 timing
    ltf_ok = True
    if cfg.require_m1_timing and m1 is not None and len(m1) >= 5:
        c = m1.iloc[-2]
        bull = float(c["close"]) > float(c["open"])
        if confluence.consensus_bias == Bias.BULLISH and not bull:
            ltf_ok = False
            why.append("M1 timing not bullish — WAIT for entry candle on next M1/M5")
        if confluence.consensus_bias == Bias.BEARISH and bull:
            ltf_ok = False
            why.append("M1 timing not bearish — WAIT for entry candle on next M1/M5")

    if not ltf_ok:
        return _blocked(
            DecisionAction.WAIT,
            why,
            RiskLevel.MEDIUM,
        )

    mid = narrative.mid
    atr = narrative.atr_m15 or (latest_atr(m15, 14) if m15 is not None else 1.0) or 1.0
    stop_dist = atr * float(cfg.risk.get("atr_stop_mult", 1.2))
    tp_dist = atr * float(cfg.risk.get("atr_tp_mult", 3.0))
    rr = tp_dist / stop_dist if stop_dist > 0 else 0.0

    if rr < cfg.min_reward_risk:
        return _blocked(
            DecisionAction.NO_TRADE,
            [f"R:R 1:{rr:.2f} below min 1:{cfg.min_reward_risk}"],
            RiskLevel.HIGH,
        )

    if confluence.consensus_bias == Bias.BULLISH:
        action = DecisionAction.BUY
        entry, stop, tp = mid, mid - stop_dist, mid + tp_dist
    else:
        action = DecisionAction.SELL
        entry, stop, tp = mid, mid + stop_dist, mid - tp_dist

    why.append(confluence.summary)
    if playbooks and playbooks.best:
        why.append(f"PLAYBOOK: {playbooks.best.name} (+{playbooks.total_boost:.0f})")
        why.extend(playbooks.best.reasons)
    why.append(f"Structure: {narrative.structure_summary}")
    why.append(f"Liquidity: {narrative.liquidity_summary}")
    why.append(f"PA: {pa.summary}")
    why.append(f"Session: {narrative.best_session} | News: {narrative.news_status}")
    why.append("High-probability confirmations aligned — quality gate PASSED")

    return InstitutionalDecision(
        action=action,
        why=why,
        confidence=confluence.confidence,
        probability=confluence.probability,
        confluence=confluence.confluence,
        risk_level=_risk_level(confluence.probability, False, True),
        entry=entry,
        stop=stop,
        take_profit=tp,
        reward_risk=rr,
        expected_hold="M5–H1 swing (playbook dependent)",
        htf_confirm=h1_aligned,
        ltf_confirm=True,
        invalidation=stop,
        narrative=narrative,
    )
