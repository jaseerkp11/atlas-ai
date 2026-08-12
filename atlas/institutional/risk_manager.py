"""Institutional risk manager — no martingale/grid/averaging."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timezone

from atlas.institutional.config import InstitutionalConfig


@dataclass
class RiskState:
    day: date = field(default_factory=lambda: datetime.now(timezone.utc).date())
    trades_today: int = 0
    consecutive_losses: int = 0
    realized_pnl_pct: float = 0.0
    starting_equity: float = 0.0
    halted: bool = False
    halt_reason: str = ""


class InstitutionalRiskManager:
    def __init__(self, cfg: InstitutionalConfig) -> None:
        self.cfg = cfg
        self.state = RiskState()

    def reset_day(self, equity: float) -> None:
        today = datetime.now(timezone.utc).date()
        if today != self.state.day:
            self.state = RiskState(day=today, starting_equity=equity)

    def allows_trade(self, equity: float, open_positions: int) -> tuple[bool, str]:
        self.reset_day(equity)
        if self.state.starting_equity <= 0:
            self.state.starting_equity = equity
        r = self.cfg.risk
        if self.state.halted:
            return False, self.state.halt_reason or "halted"
        if open_positions >= int(r.get("max_open_positions", 1)):
            return False, "max open positions"
        if self.state.trades_today >= int(r.get("max_daily_trades", 6)):
            return False, "max daily trades"
        if self.state.consecutive_losses >= int(r.get("max_consecutive_losses", 3)):
            self.state.halted = True
            self.state.halt_reason = "max consecutive losses"
            return False, self.state.halt_reason
        if self.state.realized_pnl_pct <= -float(r.get("daily_loss_limit_percent", 1.5)):
            self.state.halted = True
            self.state.halt_reason = "daily loss limit"
            return False, self.state.halt_reason
        if self.state.realized_pnl_pct >= float(r.get("daily_profit_target_percent", 2.0)):
            if r.get("pause_after_daily_limit", True):
                self.state.halted = True
                self.state.halt_reason = "daily profit target reached"
                return False, self.state.halt_reason
        return True, "ok"

    def register_trade(self) -> None:
        self.state.trades_today += 1

    def register_result(self, pnl_pct: float) -> None:
        self.state.realized_pnl_pct += pnl_pct
        if pnl_pct < 0:
            self.state.consecutive_losses += 1
        else:
            self.state.consecutive_losses = 0

    def position_size(self, equity: float, stop_distance: float, point_value: float = 1.0) -> float:
        if equity <= 0 or stop_distance <= 0:
            return 0.0
        risk_cash = equity * (float(self.cfg.risk.get("risk_percent", 0.35)) / 100.0)
        loss_per_lot = stop_distance * point_value
        if loss_per_lot <= 0:
            return 0.0
        raw = risk_cash / loss_per_lot
        # Cap for gold sanity
        return max(0.0, min(round(raw, 2), 1.0))
