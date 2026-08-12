"""Institutional analysis data models — narrative, scores, decisions."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class Bias(str, Enum):
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"


class DecisionAction(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    WAIT = "WAIT"
    NO_TRADE = "NO_TRADE"


class ZoneStrength(str, Enum):
    VERY_STRONG = "VERY_STRONG"
    STRONG = "STRONG"
    NEUTRAL = "NEUTRAL"
    WEAK = "WEAK"
    VERY_WEAK = "VERY_WEAK"


class RiskLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    EXTREME = "EXTREME"


@dataclass
class ModuleScore:
    name: str
    score: float  # 0–100
    weight: float
    bias: Bias = Bias.NEUTRAL
    detail: str = ""
    passed: bool = False


@dataclass
class Zone:
    kind: str
    top: float
    bottom: float
    strength: ZoneStrength
    score: float
    timeframe: str
    reactions: int = 0
    fresh: bool = True
    confluence: int = 0
    label: str = ""


@dataclass
class FibLevel:
    ratio: float
    price: float
    score: float
    classification: str
    timeframe: str
    confluence: bool = False


@dataclass
class MarketNarrative:
    """Complete institutional narrative before any trade decision."""

    symbol: str
    as_of: datetime
    overall_bias: Bias
    htf_bias: Bias
    mtf_bias: Bias
    structure_summary: str
    liquidity_summary: str
    institutional_confluence: str
    sr_summary: str
    trend_quality: str
    volatility_regime: str
    best_session: str
    news_status: str
    module_scores: list[ModuleScore] = field(default_factory=list)
    zones: list[Zone] = field(default_factory=list)
    fib_levels: list[FibLevel] = field(default_factory=list)
    mid: float = 0.0
    atr_m15: float = 0.0
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass
class InstitutionalDecision:
    action: DecisionAction
    why: list[str]
    confidence: float
    probability: float
    confluence: float
    risk_level: RiskLevel
    entry: float = 0.0
    stop: float = 0.0
    take_profit: float = 0.0
    reward_risk: float = 0.0
    expected_hold: str = ""
    htf_confirm: bool = False
    ltf_confirm: bool = False
    invalidation: float = 0.0
    narrative: MarketNarrative | None = None

    def is_executable(self) -> bool:
        return self.action in (DecisionAction.BUY, DecisionAction.SELL)

    def summary(self) -> str:
        lines = [
            f"DECISION: {self.action.value}",
            f"Probability: {self.probability:.1f}% | Confidence: {self.confidence:.1f}% | Confluence: {self.confluence:.1f}",
            f"Risk: {self.risk_level.value}",
        ]
        if self.is_executable():
            lines += [
                f"Entry={self.entry:.3f} SL={self.stop:.3f} TP={self.take_profit:.3f} R:R=1:{self.reward_risk:.2f}",
                f"Invalidation={self.invalidation:.3f} | Hold≈{self.expected_hold}",
                f"HTF confirm={self.htf_confirm} LTF(M1) confirm={self.ltf_confirm}",
            ]
        lines.append("Why:")
        for w in self.why:
            lines.append(f"  - {w}")
        return "\n".join(lines)


def utcnow() -> datetime:
    return datetime.now(timezone.utc)
