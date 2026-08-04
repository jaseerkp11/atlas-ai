"""
AI Decision Engine — institutional analyst, not a signal spammer.

Default action is NO TRADE / WAIT unless gates clear.
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
from atlas.institutional.price_action import PriceActionReport
import pandas as pd


def _risk_level(probability: float, news_block: bool, adx: float) -> RiskLevel:
    if news_block:
        return RiskLevel.EXTREME
    if probability >= 80 and adx >= 25:
        return RiskLevel.LOW
    if probability >= 70:
        return RiskLevel.MEDIUM
    if probability >= 55:
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
) -> InstitutionalDecision:
    why: list[str] = []
    action = DecisionAction.NO_TRADE

    if news_block:
        why.append("News filter: avoid trading window")
        return InstitutionalDecision(
            action=DecisionAction.NO_TRADE,
            why=why,
            confidence=confluence.confidence,
            probability=confluence.probability,
            confluence=confluence.confluence,
            risk_level=RiskLevel.EXTREME,
            narrative=narrative,
        )

    if confluence.consensus_bias == Bias.NEUTRAL:
        why.append("No institutional consensus bias across modules")
        return InstitutionalDecision(
            action=DecisionAction.WAIT,
            why=why + [confluence.summary],
            confidence=confluence.confidence,
            probability=confluence.probability,
            confluence=confluence.confluence,
            risk_level=RiskLevel.HIGH,
            narrative=narrative,
        )

    if cfg.require_htf_alignment and not htf_aligned:
        why.append("Higher timeframes (H4/H1) not aligned — quality gate failed")
        return InstitutionalDecision(
            action=DecisionAction.NO_TRADE,
            why=why,
            confidence=confluence.confidence,
            probability=confluence.probability,
            confluence=confluence.confluence,
            risk_level=RiskLevel.HIGH,
            narrative=narrative,
        )

    if not session_ok:
        why.append("Session quality below threshold — prefer better window")
        action = DecisionAction.WAIT

    if confluence.confluence < cfg.min_confluence:
        why.append(
            f"Confluence {confluence.confluence:.1f} < min {cfg.min_confluence}"
        )
        return InstitutionalDecision(
            action=DecisionAction.NO_TRADE,
            why=why,
            confidence=confluence.confidence,
            probability=confluence.probability,
            confluence=confluence.confluence,
            risk_level=_risk_level(confluence.probability, False, 0),
            narrative=narrative,
        )

    if confluence.probability < cfg.min_probability:
        why.append(
            f"Probability {confluence.probability:.1f}% < min {cfg.min_probability}%"
        )
        return InstitutionalDecision(
            action=DecisionAction.NO_TRADE,
            why=why,
            confidence=confluence.confidence,
            probability=confluence.probability,
            confluence=confluence.confluence,
            risk_level=_risk_level(confluence.probability, False, 0),
            narrative=narrative,
        )

    if confluence.confidence < cfg.min_confidence:
        why.append(
            f"Confidence {confluence.confidence:.1f}% < min {cfg.min_confidence}%"
        )
        return InstitutionalDecision(
            action=DecisionAction.WAIT,
            why=why,
            confidence=confluence.confidence,
            probability=confluence.probability,
            confluence=confluence.confluence,
            risk_level=RiskLevel.MEDIUM,
            narrative=narrative,
        )

    # M1 timing confirmation (entry only)
    ltf_ok = True
    if cfg.require_m1_timing and m1 is not None and len(m1) >= 5:
        c = m1.iloc[-2]
        bull = float(c["close"]) > float(c["open"])
        if confluence.consensus_bias == Bias.BULLISH and not bull:
            ltf_ok = False
            why.append("M1 timing not bullish — WAIT for entry candle")
        if confluence.consensus_bias == Bias.BEARISH and bull:
            ltf_ok = False
            why.append("M1 timing not bearish — WAIT for entry candle")

    if not ltf_ok:
        return InstitutionalDecision(
            action=DecisionAction.WAIT,
            why=why,
            confidence=confluence.confidence,
            probability=confluence.probability,
            confluence=confluence.confluence,
            risk_level=RiskLevel.MEDIUM,
            htf_confirm=htf_aligned,
            ltf_confirm=False,
            narrative=narrative,
        )

    # Build levels from ATR
    mid = narrative.mid
    atr = narrative.atr_m15 or (latest_atr(m15, 14) if m15 is not None else 1.0) or 1.0
    stop_dist = atr * float(cfg.risk.get("atr_stop_mult", 1.2))
    tp_dist = atr * float(cfg.risk.get("atr_tp_mult", 3.0))
    rr = tp_dist / stop_dist if stop_dist > 0 else 0.0

    if rr < cfg.min_reward_risk:
        why.append(f"R:R 1:{rr:.2f} below min 1:{cfg.min_reward_risk}")
        return InstitutionalDecision(
            action=DecisionAction.NO_TRADE,
            why=why,
            confidence=confluence.confidence,
            probability=confluence.probability,
            confluence=confluence.confluence,
            risk_level=RiskLevel.HIGH,
            reward_risk=rr,
            narrative=narrative,
        )

    if confluence.consensus_bias == Bias.BULLISH:
        action = DecisionAction.BUY
        entry = mid
        stop = mid - stop_dist
        tp = mid + tp_dist
        inval = stop
    else:
        action = DecisionAction.SELL
        entry = mid
        stop = mid + stop_dist
        tp = mid - tp_dist
        inval = stop

    why.append(confluence.summary)
    why.append(f"Structure: {narrative.structure_summary}")
    why.append(f"Liquidity: {narrative.liquidity_summary}")
    why.append(f"PA: {pa.summary}")
    why.append(f"Session: {narrative.best_session} | News: {narrative.news_status}")
    why.append("Multiple independent confirmations aligned — quality gate PASSED")

    return InstitutionalDecision(
        action=action,
        why=why,
        confidence=confluence.confidence,
        probability=confluence.probability,
        confluence=confluence.confluence,
        risk_level=_risk_level(confluence.probability, False, 25),
        entry=entry,
        stop=stop,
        take_profit=tp,
        reward_risk=rr,
        expected_hold="15m–4h (HTF dependent)",
        htf_confirm=htf_aligned,
        ltf_confirm=True,
        invalidation=inval,
        narrative=narrative,
    )
