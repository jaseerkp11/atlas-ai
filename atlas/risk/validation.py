"""
Pre-trade validation — must actually block execution when invalid.

Checks: broker connection, trading session, spread, margin, conflicting positions.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Protocol

from atlas.config import load_settings
from atlas.models import Direction, TradeDecision, ValidationResult
from atlas.risk.position import ConcurrentTradeGuard, DailyRiskState, PositionSizeResult


class BrokerInfo(Protocol):
    def is_connected(self) -> bool: ...
    def account_balance(self) -> float: ...
    def account_equity(self) -> float: ...
    def account_free_margin(self) -> float: ...
    def symbol_spread_pips(self, symbol: str) -> float: ...
    def has_open_position(self, symbol: str) -> bool: ...
    def open_position_count(self) -> int: ...
    def symbol_trade_mode_allowed(self, symbol: str) -> bool: ...


def is_session_open(now: datetime | None = None) -> tuple[bool, str]:
    settings = load_settings()
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    hour = now.hour
    start, end = settings.sessions["allowed_hours_utc"]
    # Weekend guard (forex): Saturday full close, Sunday before start closed
    weekday = now.weekday()  # Mon=0 … Sun=6
    if weekday == 5:
        return False, "Weekend — Saturday market closed"
    if weekday == 6 and hour < start:
        return False, "Weekend — Sunday session not yet open"
    if start <= hour < end:
        return True, f"Session open (UTC hour {hour} within [{start}, {end}))"
    return False, f"Outside allowed session hours (UTC {hour} not in [{start}, {end}))"


def validate_trade(
    decision: TradeDecision,
    broker: BrokerInfo,
    sizing: PositionSizeResult,
    daily_risk: DailyRiskState,
    concurrent: ConcurrentTradeGuard,
    now: datetime | None = None,
) -> ValidationResult:
    """
    Run all pre-trade checks. If any fail, result.valid is False and
    execution MUST NOT proceed.
    """
    checks: list[tuple[str, bool, str]] = []
    settings = load_settings()
    symbol = decision.symbol

    # 1. Decision already gated?
    checks.append(
        (
            "score_and_rr_gates",
            decision.allowed,
            "Passed score/RR gates" if decision.allowed else "; ".join(decision.gate_failures) or "Failed gates",
        )
    )

    # 2. Broker connection
    connected = broker.is_connected()
    checks.append(
        (
            "broker_connection",
            connected,
            "Connected" if connected else "Broker not connected",
        )
    )

    # 3. Trading session
    session_ok, session_msg = is_session_open(now)
    checks.append(("trading_session", session_ok, session_msg))

    # 4. Symbol tradable
    try:
        tradable = broker.symbol_trade_mode_allowed(symbol)
    except Exception as exc:
        tradable = False
        checks.append(("symbol_tradable", False, f"Symbol check failed: {exc}"))
    else:
        checks.append(
            (
                "symbol_tradable",
                tradable,
                "Symbol allowed for trading" if tradable else "Symbol not tradable",
            )
        )

    # 5. Spread
    try:
        spread = broker.symbol_spread_pips(symbol)
        max_spread = settings.max_spread_for(symbol)
        spread_ok = spread <= max_spread
        checks.append(
            (
                "spread",
                spread_ok,
                f"Spread {spread:.2f} pips (max {max_spread:.2f})"
                if spread_ok
                else f"Spread {spread:.2f} pips exceeds max {max_spread:.2f}",
            )
        )
    except Exception as exc:
        checks.append(("spread", False, f"Spread check failed: {exc}"))

    # 6. Position size valid (must use calculated size)
    checks.append(
        (
            "position_size",
            sizing.valid and sizing.volume > 0,
            sizing.reason if not sizing.valid else sizing.formula,
        )
    )

    # 7. Margin
    try:
        free_margin = broker.account_free_margin()
        # Rough estimate: require free margin > risk amount
        margin_ok = free_margin >= sizing.risk_amount
        checks.append(
            (
                "margin",
                margin_ok,
                f"Free margin {free_margin:.2f} vs risk {sizing.risk_amount:.2f}"
                if margin_ok
                else f"Insufficient margin ({free_margin:.2f} < risk {sizing.risk_amount:.2f})",
            )
        )
    except Exception as exc:
        checks.append(("margin", False, f"Margin check failed: {exc}"))

    # 8. No conflicting open position on same symbol
    try:
        has_pos = broker.has_open_position(symbol)
        checks.append(
            (
                "no_conflict",
                not has_pos,
                "No open position on symbol"
                if not has_pos
                else f"Conflicting open position on {symbol}",
            )
        )
    except Exception as exc:
        checks.append(("no_conflict", False, f"Position check failed: {exc}"))

    # 9. Concurrent trade limit
    try:
        open_count = broker.open_position_count()
    except Exception:
        open_count = concurrent.open_count
    conc_ok, conc_msg = concurrent.allows(open_count)
    checks.append(("max_concurrent", conc_ok, conc_msg))

    # 10. Daily loss limit
    try:
        equity = broker.account_equity()
    except Exception:
        equity = 0.0
    day_ok, day_msg = daily_risk.allows_new_trades(equity)
    checks.append(("daily_loss_limit", day_ok, day_msg))

    valid = all(ok for _, ok, _ in checks)
    return ValidationResult(valid=valid, checks=checks)
