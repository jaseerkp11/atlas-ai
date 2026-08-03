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
        if now.weekday() == 6 and now.hour < 22:
            return False, "Sunday pre-open (gold ~22:00 UTC)"
        if now.weekday() == 4 and now.hour >= 22:
            return False, "Friday late close / thin liquidity"
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

    def _normalize_volume(self, raw: float) -> float:
        min_lot, step, broker_max = self.feed.symbol_volume_limits()
        cap = min(broker_max, self.cfg.max_lots)
        if step <= 0:
            step = 0.01
        steps = int(raw / step + 1e-9)
        vol = max(0.0, min(steps * step, cap))
        # Keep volume digits consistent with step
        decimals = max(0, len(str(step).rstrip("0").split(".")[-1]) if "." in str(step) else 0)
        vol = round(vol, decimals or 2)
        if vol < min_lot:
            return 0.0
        return vol

    def _money_per_point_per_lot(self) -> float:
        """
        $ PnL per 1.0 point move per 1.0 lot.
        Brokers often mis-report tick_value for gold — sanity-clamp to ~1.0.
        """
        point = self.feed.symbol_point()
        money = 1.0
        if self.feed._use_mt5:
            try:
                import MetaTrader5 as mt5

                info = mt5.symbol_info(self.cfg.symbol)
                if info and info.trade_tick_value and info.trade_tick_size:
                    calc = float(info.trade_tick_value) * (
                        point / float(info.trade_tick_size)
                    )
                    # XAUUSD typical band ~0.5–10 $/point/lot; outside → ignore
                    if 0.5 <= calc <= 10.0:
                        money = calc
            except Exception:
                pass
        return money

    def position_size(self, stop_points: float) -> tuple[float, str]:
        """
        Prefer fixed micro lots for gold scalping (reliable margin).
        Optional risk% path is capped by max_lots.
        """
        min_lot, step, _ = self.feed.symbol_volume_limits()

        if self.cfg.use_fixed_lots:
            vol = self._normalize_volume(self.cfg.fixed_lots)
            if vol <= 0:
                return 0.0, f"fixed_lots {self.cfg.fixed_lots} below min {min_lot}"
            return vol, f"size={vol} FIXED (max={self.cfg.max_lots})"

        balance = self.feed.account_balance()
        if balance <= 0 or stop_points <= 0:
            return 0.0, "invalid balance/stop"

        risk_cash = balance * (self.cfg.risk_percent / 100.0)
        money_per_point_per_lot = self._money_per_point_per_lot()
        loss_per_lot = stop_points * money_per_point_per_lot
        if loss_per_lot <= 0:
            return 0.0, "loss_per_lot invalid"

        raw = risk_cash / loss_per_lot
        vol = self._normalize_volume(raw)
        if vol <= 0:
            # Fall back to minimum lot if risk% is tiny but account can trade
            vol = self._normalize_volume(min_lot)
            if vol <= 0:
                return 0.0, f"volume {raw:.4f} below min {min_lot}"
            return vol, f"size={vol} MIN_LOT fallback (risk raw={raw:.4f})"

        return vol, (
            f"size={vol} from risk ${risk_cash:.2f} / ({stop_points:.1f} pts "
            f"× ${money_per_point_per_lot:.2f}/pt) cap={self.cfg.max_lots}"
        )
