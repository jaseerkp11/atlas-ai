"""
Strict risk management for the XAUUSD tick scalper.

No martingale, no grid, no averaging into losers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timezone

from atlas.tick_scalper.config import TickScalperConfig, load_tick_config
from atlas.tick_scalper.mt5_feed import MT5TickFeed


@dataclass
class RiskState:
    day: date = field(default_factory=lambda: datetime.now(timezone.utc).date())
    starting_equity: float = 0.0
    realized_pnl: float = 0.0
    halted: bool = False
    last_close_msc: int = 0


class TickRiskManager:
    def __init__(self, feed: MT5TickFeed, cfg: TickScalperConfig | None = None) -> None:
        self.feed = feed
        self.cfg = cfg or load_tick_config()
        self.state = RiskState()

    def reset_day_if_needed(self, equity: float) -> None:
        today = datetime.now(timezone.utc).date()
        if today != self.state.day:
            self.state = RiskState(day=today, starting_equity=equity)

    def record_pnl(self, pnl: float, equity: float) -> None:
        self.reset_day_if_needed(equity)
        if self.state.starting_equity <= 0:
            self.state.starting_equity = equity - pnl
        self.state.realized_pnl += pnl
        base = self.state.starting_equity if self.state.starting_equity > 0 else equity
        if base > 0 and self.state.realized_pnl < 0:
            loss_pct = abs(self.state.realized_pnl) / base * 100.0
            if loss_pct >= self.cfg.daily_loss_limit_percent:
                self.state.halted = True

    def mark_close(self, time_msc: int) -> None:
        self.state.last_close_msc = time_msc

    def in_cooldown(self, time_msc: int) -> bool:
        if self.state.last_close_msc <= 0:
            return False
        return (time_msc - self.state.last_close_msc) < self.cfg.cooldown_ms_after_close

    def in_session(self, now: datetime | None = None) -> tuple[bool, str]:
        now = now or datetime.now(timezone.utc)
        start, end = self.cfg.session_hours_utc
        if now.weekday() == 5:
            return False, "Saturday closed"
        if now.weekday() == 6 and now.hour < start:
            return False, "Sunday pre-open"
        if start <= now.hour < end:
            return True, "session open"
        return False, f"outside session UTC [{start},{end})"

    def allows_new_trade(self, time_msc: int) -> tuple[bool, str]:
        equity = self.feed.account_equity()
        self.reset_day_if_needed(equity)
        if self.state.starting_equity <= 0:
            self.state.starting_equity = equity

        if self.state.halted:
            return False, "daily loss limit hit — new entries halted"

        session_ok, session_msg = self.in_session()
        if not session_ok:
            return False, session_msg

        if self.in_cooldown(time_msc):
            return False, "post-close cooldown"

        open_n = self.feed.open_position_count()
        if open_n >= self.cfg.max_open_positions:
            return False, f"max open positions ({self.cfg.max_open_positions})"

        return True, "risk OK"

    def position_size(self, stop_points: float) -> tuple[float, str]:
        """
        volume from balance * risk% / (stop_points * tick_value_approx).
        For XAUUSD: 1.00 lot ≈ $1 per 0.01 move → $100 per 1.00 point? 
        Actually: contract 100 oz, $1 move = $100/lot. point=0.01 → $1 per point per lot.
        """
        balance = self.feed.account_balance()
        if balance <= 0 or stop_points <= 0:
            return 0.0, "invalid balance/stop"

        risk_cash = balance * (self.cfg.risk_percent / 100.0)
        # $ per point per 1.0 lot for XAU ≈ 1.0 when point=0.01 (broker dependent)
        point = self.feed.symbol_point()
        # money per point per lot ≈ contract_size * point; XAU contract 100 → $1 per 0.01
        money_per_point_per_lot = 1.0
        if self.feed._use_mt5:
            try:
                import MetaTrader5 as mt5

                info = mt5.symbol_info(self.cfg.symbol)
                if info and info.trade_tick_value and info.trade_tick_size:
                    money_per_point_per_lot = float(info.trade_tick_value) * (
                        point / float(info.trade_tick_size)
                    )
            except Exception:
                pass

        loss_per_lot = stop_points * money_per_point_per_lot
        if loss_per_lot <= 0:
            return 0.0, "loss_per_lot invalid"

        raw = risk_cash / loss_per_lot
        min_lot, step, max_lot = self.feed.symbol_volume_limits()
        steps = int(raw / step)
        vol = max(0.0, min(steps * step, max_lot))
        vol = round(vol, 2)
        if vol < min_lot:
            return 0.0, f"volume {raw:.4f} below min {min_lot}"
        return vol, (
            f"size={vol} from risk ${risk_cash:.2f} / ({stop_points:.1f} pts "
            f"× ${money_per_point_per_lot:.2f}/pt) "
        )
