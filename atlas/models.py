"""Shared data models used by analysis, scoring, execution, backtest, and journal."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Any


class Direction(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    NEUTRAL = "NEUTRAL"


class TrendStrength(str, Enum):
    STRONG = "STRONG"
    MODERATE = "MODERATE"
    WEAK = "WEAK"
    NONE = "NONE"


class TradeStatus(str, Enum):
    PENDING = "PENDING"
    OPEN = "OPEN"
    CLOSED = "CLOSED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


class TradeMode(str, Enum):
    PAPER = "PAPER"
    LIVE = "LIVE"
    BACKTEST = "BACKTEST"


@dataclass
class SwingPoint:
    index: int
    price: float
    time: datetime | None
    kind: str  # "high" | "low"


@dataclass
class StructureLevel:
    price: float
    kind: str  # "swing_high" | "swing_low"
    time: datetime | None = None
    index: int = -1


@dataclass
class TimeframeAnalysis:
    timeframe: str
    direction: Direction
    strength: TrendStrength
    confidence: float  # 0–1
    swing_highs: list[StructureLevel] = field(default_factory=list)
    swing_lows: list[StructureLevel] = field(default_factory=list)
    last_bos: str | None = None  # "bullish" | "bearish" | None
    last_choch: str | None = None
    atr: float = 0.0
    summary: str = ""


@dataclass
class FairValueGap:
    direction: Direction  # LONG = bullish FVG, SHORT = bearish FVG
    top: float
    bottom: float
    index: int
    time: datetime | None = None
    filled: bool = False


@dataclass
class OrderBlock:
    direction: Direction  # LONG = bullish OB, SHORT = bearish OB
    top: float
    bottom: float
    index: int
    time: datetime | None = None
    mitigated: bool = False


@dataclass
class LiquiditySweep:
    direction: Direction  # LONG = swept lows (buy-side liquidity grab), SHORT = swept highs
    level: float
    index: int
    time: datetime | None = None
    swept_kind: str = ""  # "swing_low" | "swing_high"


@dataclass
class SetupFeatures:
    """All detected structural features for a symbol at evaluation time."""

    symbol: str
    direction: Direction
    h4: TimeframeAnalysis | None = None
    h1: TimeframeAnalysis | None = None
    m15: TimeframeAnalysis | None = None
    m5: TimeframeAnalysis | None = None
    bos: bool = False
    choch: bool = False
    liquidity_sweep: LiquiditySweep | None = None
    fair_value_gaps: list[FairValueGap] = field(default_factory=list)
    order_blocks: list[OrderBlock] = field(default_factory=list)
    m5_trigger: bool = False
    atr: float = 0.0
    volatility_expanding: bool = False
    entry: float = 0.0
    stop: float = 0.0
    target: float = 0.0
    reward_risk: float = 0.0


@dataclass
class ScoreFactor:
    name: str
    weight: int
    earned: int
    passed: bool
    reason: str


@dataclass
class ScoreResult:
    total: int
    max_total: int = 100
    factors: list[ScoreFactor] = field(default_factory=list)
    passed_factors: list[str] = field(default_factory=list)
    failed_factors: list[str] = field(default_factory=list)
    plain_language: list[str] = field(default_factory=list)

    def summary_lines(self) -> list[str]:
        lines = [f"Score: {self.total}/{self.max_total}"]
        lines.append("Passed:")
        if self.passed_factors:
            for f in self.factors:
                if f.passed:
                    lines.append(f"  ✓ [{f.earned}/{f.weight}] {f.name}: {f.reason}")
        else:
            lines.append("  (none)")
        lines.append("Failed:")
        if self.failed_factors:
            for f in self.factors:
                if not f.passed:
                    lines.append(f"  ✗ [{f.earned}/{f.weight}] {f.name}: {f.reason}")
        else:
            lines.append("  (none)")
        return lines


@dataclass
class TradeDecision:
    """Outcome of scoring + gate check for one potential setup."""

    symbol: str
    direction: Direction
    score: ScoreResult
    features: SetupFeatures
    allowed: bool
    gate_failures: list[str] = field(default_factory=list)
    rejection_reasons: list[str] = field(default_factory=list)
    evaluated_at: datetime | None = None

    def reasoning_text(self) -> str:
        lines = [
            f"{'=' * 60}",
            f"SETUP: {self.symbol} {self.direction.value}",
            f"Decision: {'TRADE ALLOWED' if self.allowed else 'NO TRADE'}",
            f"{'=' * 60}",
        ]
        lines.extend(self.score.summary_lines())
        rr = self.features.reward_risk
        lines.append(
            f"Levels: entry={self.features.entry:.5f}  "
            f"stop={self.features.stop:.5f}  "
            f"target={self.features.target:.5f}  "
            f"R:R={rr:.2f}"
        )
        if self.gate_failures:
            lines.append("Gate failures:")
            for g in self.gate_failures:
                lines.append(f"  • {g}")
        if self.rejection_reasons:
            lines.append("Additional rejections:")
            for r in self.rejection_reasons:
                lines.append(f"  • {r}")
        lines.append("")
        return "\n".join(lines)


@dataclass
class TradeRecord:
    trade_id: str
    mode: TradeMode
    symbol: str
    direction: Direction
    status: TradeStatus
    entry: float
    stop: float
    target: float
    exit_price: float | None = None
    volume: float = 0.0
    score: int = 0
    reasoning: str = ""
    reward_risk_planned: float = 0.0
    r_multiple: float | None = None
    result: str | None = None  # "win" | "loss" | "breakeven" | None
    opened_at: datetime | None = None
    closed_at: datetime | None = None
    filled_at: datetime | None = None
    pending_until: datetime | None = None
    mt5_ticket: int | None = None
    notes: str = ""

    def to_journal_row(self) -> dict[str, Any]:
        return {
            "trade_id": self.trade_id,
            "mode": self.mode.value,
            "symbol": self.symbol,
            "direction": self.direction.value,
            "status": self.status.value,
            "entry": self.entry,
            "stop": self.stop,
            "target": self.target,
            "exit": self.exit_price,
            "volume": self.volume,
            "score": self.score,
            "reasoning": self.reasoning,
            "planned_rr": self.reward_risk_planned,
            "r_multiple": self.r_multiple,
            "result": self.result,
            "opened_at": self.opened_at.isoformat() if self.opened_at else None,
            "filled_at": self.filled_at.isoformat() if self.filled_at else None,
            "closed_at": self.closed_at.isoformat() if self.closed_at else None,
            "mt5_ticket": self.mt5_ticket,
            "notes": self.notes,
        }


@dataclass
class ValidationResult:
    valid: bool
    checks: list[tuple[str, bool, str]] = field(default_factory=list)

    def failures(self) -> list[str]:
        return [msg for name, ok, msg in self.checks if not ok]

    def summary(self) -> str:
        lines = ["Pre-trade validation:"]
        for name, ok, msg in self.checks:
            mark = "✓" if ok else "✗"
            lines.append(f"  {mark} {name}: {msg}")
        return "\n".join(lines)


@dataclass
class BacktestStats:
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    breakevens: int = 0
    win_rate: float = 0.0
    average_r: float = 0.0
    max_drawdown_pct: float = 0.0
    net_r: float = 0.0
    starting_balance: float = 0.0
    ending_balance: float = 0.0
    limitations: list[str] = field(default_factory=list)
