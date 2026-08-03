"""
Tick strategy — PURE M1 price action.

No micro-TF indicators, no VWAP, no tick-momentum gates.
Direction comes only from the 1-minute chart structure.
Exits still evaluated every tick for instant profit / SL.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from atlas.tick_scalper.config import TickScalperConfig, load_tick_config
from atlas.tick_scalper.m1_price_action import M1Bias, read_m1_price_action
from atlas.tick_scalper.mt5_feed import Tick


class Side(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


@dataclass
class Signal:
    side: Side
    reason: str
    bid: float
    ask: float
    m1_bar_time: int = 0


@dataclass
class OpenState:
    side: Side
    entry: float
    sl: float
    tp: float
    volume: float
    ticket: int
    vwap_at_entry: float = 0.0
    opened_msc: int = 0
    profit_target_points: float = 0.0


class TickStrategy:
    def __init__(self, cfg: TickScalperConfig | None = None) -> None:
        self.cfg = cfg or load_tick_config()
        self._use_mt5 = False
        self._bias: Optional[M1Bias] = None
        self._last_poll_s: float = 0.0
        self._last_burst_bar: int = 0

    def set_mt5_live(self, live: bool) -> None:
        self._use_mt5 = bool(live)

    def update(self, tick: Tick, point: float) -> None:
        """Refresh M1 bias about once per second (cheap)."""
        import time

        now = time.time()
        poll = max(0.5, float(getattr(self.cfg, "m1_poll_seconds", 1) or 1))
        if now - self._last_poll_s < poll and self._bias is not None:
            return
        self._last_poll_s = now
        self._bias = read_m1_price_action(
            self.cfg.symbol,
            self._use_mt5,
            structure_bars=int(getattr(self.cfg, "m1_structure_bars", 12) or 12),
        )

    @property
    def current_bias(self) -> Optional[M1Bias]:
        return self._bias

    def evaluate_entry(self, tick: Tick, point: float) -> Signal | None:
        spread_pts = tick.spread_points(point)
        if spread_pts > self.cfg.max_spread_points:
            return None

        bias = self._bias
        if bias is None:
            # Offline / synthetic fallback: tiny mid move only for plumbing tests
            if not self._use_mt5:
                return None
            return None

        side = Side.BUY if bias.side == "BUY" else Side.SELL
        return Signal(
            side=side,
            reason=f"M1 PA | {bias.reason} | spread={spread_pts:.1f}pts",
            bid=tick.bid,
            ask=tick.ask,
            m1_bar_time=bias.bar_time,
        )

    def mark_burst_bar(self, bar_time: int) -> None:
        self._last_burst_bar = bar_time

    def already_burst_this_bar(self, bar_time: int) -> bool:
        return bar_time != 0 and bar_time == self._last_burst_bar

    def _unrealized_points(self, tick: Tick, pos: OpenState, point: float) -> float:
        if pos.side == Side.BUY:
            return (tick.bid - pos.entry) / point
        return (pos.entry - tick.ask) / point

    def evaluate_exit(self, tick: Tick, pos: OpenState, point: float) -> str | None:
        pnl_pts = self._unrealized_points(tick, pos, point)
        target = pos.profit_target_points or self.cfg.instant_profit_points

        if pnl_pts >= target:
            return "instant_profit"

        if pos.side == Side.BUY:
            if tick.bid >= pos.tp:
                return "take_profit"
            if tick.bid <= pos.sl:
                return "stop_loss"
        else:
            if tick.ask <= pos.tp:
                return "take_profit"
            if tick.ask >= pos.sl:
                return "stop_loss"

        # If M1 structure flipped against us while green → bank
        bias = self._bias
        if bias is not None and pnl_pts > 0:
            if pos.side == Side.BUY and bias.side == "SELL":
                return "m1_flip_profit"
            if pos.side == Side.SELL and bias.side == "BUY":
                return "m1_flip_profit"

        return None

    def levels_for(
        self, side: Side, entry: float, point: float, spread_points: float = 0.0
    ) -> tuple[float, float]:
        sl_pts = max(self.cfg.stop_loss_points, spread_points * 1.2 + 20.0)
        tp_pts = max(self.cfg.take_profit_points, self.cfg.instant_profit_points + 5.0)
        sl_dist = sl_pts * point
        tp_dist = tp_pts * point
        if side == Side.BUY:
            return entry - sl_dist, entry + tp_dist
        return entry + sl_dist, entry - tp_dist
