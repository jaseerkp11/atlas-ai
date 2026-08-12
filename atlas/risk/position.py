"""
Risk management — position sizing, daily loss limit, concurrent trade cap.

Position size = account_balance × risk_percent ÷ stop_distance
The calculated size is what execution MUST use (not silently overridden).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timezone

from atlas.config import load_settings


@dataclass
class PositionSizeResult:
    volume: float
    risk_amount: float
    stop_distance: float
    balance: float
    risk_percent: float
    formula: str
    valid: bool
    reason: str = ""


def calculate_position_size(
    balance: float,
    entry: float,
    stop: float,
    symbol: str,
    *,
    tick_size: float = 0.00001,
    tick_value: float = 1.0,
    lot_step: float = 0.01,
    min_lot: float = 0.01,
    max_lot: float = 100.0,
    contract_size: float = 100000.0,
    risk_percent: float | None = None,
) -> PositionSizeResult:
    """
    Calculate volume from risk percent and stop distance.

    For forex: risk_amount / (stop_distance_in_price × contract_size / account_currency_factor)
    Simplified: volume = risk_amount / (stop_distance × contract_size)
    adjusted by tick_value when provided by broker.

    Gold (XAUUSD) typically uses contract_size=100.
    """
    settings = load_settings()
    risk_pct = risk_percent if risk_percent is not None else settings.risk.risk_percent
    stop_distance = abs(entry - stop)

    if balance <= 0:
        return PositionSizeResult(
            0, 0, stop_distance, balance, risk_pct,
            "invalid", False, "Account balance must be positive",
        )
    if stop_distance <= 0:
        return PositionSizeResult(
            0, 0, stop_distance, balance, risk_pct,
            "invalid", False, "Stop distance must be positive",
        )

    risk_amount = balance * (risk_pct / 100.0)

    # Value per 1.0 lot per 1.0 price unit
    if symbol.upper().startswith("XAU"):
        contract_size = 100.0
    elif symbol.upper().startswith("XAG"):
        contract_size = 5000.0

    # Money lost per lot if SL hit ≈ stop_distance × contract_size
    # (for USD-quoted pairs this is approximate; live path should pass tick_value)
    if tick_value > 0 and tick_size > 0:
        loss_per_lot = (stop_distance / tick_size) * tick_value
    else:
        loss_per_lot = stop_distance * contract_size

    if loss_per_lot <= 0:
        return PositionSizeResult(
            0, risk_amount, stop_distance, balance, risk_pct,
            "invalid", False, "Could not compute loss per lot",
        )

    raw_volume = risk_amount / loss_per_lot
    volume = _round_lot(raw_volume, lot_step, min_lot, max_lot)

    formula = (
        f"volume = ({balance:.2f} × {risk_pct}% = {risk_amount:.2f}) "
        f"/ (stop_dist {stop_distance:.5f} → loss/lot {loss_per_lot:.2f}) "
        f"= {raw_volume:.4f} → rounded {volume:.2f}"
    )

    if volume < min_lot:
        return PositionSizeResult(
            0, risk_amount, stop_distance, balance, risk_pct,
            formula, False,
            f"Computed volume {raw_volume:.4f} below broker minimum {min_lot}",
        )

    return PositionSizeResult(
        volume=volume,
        risk_amount=risk_amount,
        stop_distance=stop_distance,
        balance=balance,
        risk_percent=risk_pct,
        formula=formula,
        valid=True,
        reason="OK",
    )


def _round_lot(volume: float, step: float, min_lot: float, max_lot: float) -> float:
    if step <= 0:
        step = 0.01
    # Use decimal-safe flooring to avoid 0.2 → 0.19 float artifacts
    steps = int(round(volume / step + 1e-12))
    # Floor to not exceed risk
    if steps * step > volume + 1e-12:
        steps -= 1
    rounded = max(0.0, min(steps * step, max_lot))
    decimals = max(0, len(str(step).rstrip("0").split(".")[-1]) if "." in str(step) else 0)
    return round(rounded, decimals)


@dataclass
class DailyRiskState:
    """Tracks realized P&L for the current UTC day; halts NEW trades when limit hit."""

    day: date = field(default_factory=lambda: datetime.now(timezone.utc).date())
    starting_equity: float = 0.0
    realized_pnl: float = 0.0
    halted: bool = False

    def reset_if_new_day(self, equity: float) -> None:
        today = datetime.now(timezone.utc).date()
        if today != self.day:
            self.day = today
            self.starting_equity = equity
            self.realized_pnl = 0.0
            self.halted = False

    def record_closed_pnl(self, pnl: float, equity: float) -> None:
        self.reset_if_new_day(equity)
        self.realized_pnl += pnl
        limit_pct = load_settings().risk.daily_loss_limit_percent
        base = self.starting_equity if self.starting_equity > 0 else equity
        if base > 0 and self.realized_pnl < 0:
            loss_pct = abs(self.realized_pnl) / base * 100.0
            if loss_pct >= limit_pct:
                self.halted = True

    def allows_new_trades(self, equity: float) -> tuple[bool, str]:
        self.reset_if_new_day(equity)
        if self.starting_equity <= 0:
            self.starting_equity = equity
        if self.halted:
            limit = load_settings().risk.daily_loss_limit_percent
            return False, (
                f"Daily loss limit ({limit}%) hit — "
                f"realized PnL today {self.realized_pnl:.2f}; new trades halted"
            )
        return True, "Daily loss limit OK"


@dataclass
class ConcurrentTradeGuard:
    open_count: int = 0

    def allows(self, open_count: int | None = None) -> tuple[bool, str]:
        count = self.open_count if open_count is None else open_count
        max_n = load_settings().risk.max_concurrent_trades
        if count >= max_n:
            return False, f"Max concurrent trades ({max_n}) already open ({count})"
        return True, f"Open trades {count}/{max_n}"
