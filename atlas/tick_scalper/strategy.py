"""
Tick strategy — RAPID CYCLE scalper.

Entry: short-tape momentum + spread filter (fast, not a blind 1-second timer).
Exit (every tick, priority):
  1. Instant profit — close as soon as unrealized >= instant_profit_points
  2. Hard take-profit / stop-loss
  3. Soft profit protect on fade (rapid mode)

Losers are NOT averaged and NOT held forever "hoping" for green — SL exits them.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum

from atlas.tick_scalper.config import TickScalperConfig, load_tick_config
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
    profit_target_points: float = 0.0  # per-slot ladder target


class TickStrategy:
    def __init__(self, cfg: TickScalperConfig | None = None) -> None:
        self.cfg = cfg or load_tick_config()
        self._mids: deque[float] = deque(maxlen=max(32, self.cfg.momentum_lookback * 4))
        self._vwap_num = 0.0
        self._vwap_den = 0.0
        self._vwap_day = -1

    def _reset_vwap_if_new_day(self, tick: Tick) -> None:
        day = int(tick.time_msc // 86_400_000)
        if day != self._vwap_day:
            self._vwap_day = day
            self._vwap_num = 0.0
            self._vwap_den = 0.0

    def update(self, tick: Tick, point: float) -> None:
        self._mids.append(tick.mid)
        if self.cfg.vwap_enabled:
            self._reset_vwap_if_new_day(tick)
            vol = max(1.0, float(tick.volume or 1))
            self._vwap_num += tick.mid * vol
            self._vwap_den += vol

    @property
    def vwap(self) -> float:
        if self._vwap_den <= 0:
            return 0.0
        return self._vwap_num / self._vwap_den

    def evaluate_entry(self, tick: Tick, point: float) -> Signal | None:
        spread_pts = tick.spread_points(point)
        if spread_pts > self.cfg.max_spread_points:
            return None

        need = self.cfg.momentum_lookback + 1
        if len(self._mids) < need:
            return None

        mids = list(self._mids)
        window = mids[-need:]
        rng = (max(window) - min(window)) / point
        if rng < self.cfg.min_tick_range_points:
            return None

        ups = sum(1 for i in range(1, len(window)) if window[i] > window[i - 1])
        downs = sum(1 for i in range(1, len(window)) if window[i] < window[i - 1])

        side: Side | None = None
        reason_parts: list[str] = []

        if ups >= self.cfg.min_tick_momentum and ups > downs:
            side = Side.BUY
            reason_parts.append(f"momentum up {ups}/{self.cfg.momentum_lookback}")
        elif downs >= self.cfg.min_tick_momentum and downs > ups:
            side = Side.SELL
            reason_parts.append(f"momentum down {downs}/{self.cfg.momentum_lookback}")
        else:
            return None

        # Rapid cycle: last tick must confirm direction (reduces instant flip noise)
        if self.cfg.rapid_cycle and len(window) >= 2:
            last_up = window[-1] > window[-2]
            if side == Side.BUY and not last_up:
                return None
            if side == Side.SELL and last_up:
                return None

        vwap = self.vwap
        if self.cfg.vwap_enabled and vwap > 0:
            dist_pts = abs(tick.mid - vwap) / point
            near = dist_pts <= self.cfg.vwap_near_points
            if side == Side.BUY and tick.mid < vwap and not near:
                return None
            if side == Side.SELL and tick.mid > vwap and not near:
                return None
            reason_parts.append(f"VWAP={vwap:.2f} dist={dist_pts:.1f}pts")

        reason_parts.append(f"spread={spread_pts:.1f}pts range={rng:.1f}pts")
        if self.cfg.rapid_cycle:
            reason_parts.append("RAPID")
        return Signal(side=side, reason=" | ".join(reason_parts), bid=tick.bid, ask=tick.ask)

    def _unrealized_points(self, tick: Tick, pos: OpenState, point: float) -> float:
        if pos.side == Side.BUY:
            return (tick.bid - pos.entry) / point
        return (pos.entry - tick.ask) / point

    def _momentum_fading(self, side: Side) -> bool:
        if len(self._mids) < self.cfg.momentum_lookback + 1:
            return False
        window = list(self._mids)[-(self.cfg.momentum_lookback + 1) :]
        ups = sum(1 for i in range(1, len(window)) if window[i] > window[i - 1])
        downs = sum(1 for i in range(1, len(window)) if window[i] < window[i - 1])
        if side == Side.BUY:
            return downs >= self.cfg.min_tick_momentum and downs > ups
        return ups >= self.cfg.min_tick_momentum and ups > downs

    def evaluate_exit(self, tick: Tick, pos: OpenState, point: float) -> str | None:
        """
        Every tick. Instant-profit first so winners close ASAP and cycle restarts.
        """
        pnl_pts = self._unrealized_points(tick, pos, point)
        target = pos.profit_target_points or self.cfg.instant_profit_points

        # 1) INSTANT / LADDER PROFIT — bank this slot ASAP
        if pnl_pts >= target:
            return "instant_profit"

        # 2) Hard broker levels
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

        # 3) Rapid: any green + fade → close (still profit-only early exit)
        if self.cfg.rapid_cycle and pnl_pts > 0 and self._momentum_fading(pos.side):
            return "rapid_fade_profit"

        # 4) Classic soft protect (non-rapid or backup)
        if not self.cfg.rapid_cycle:
            soft_tp = self.cfg.take_profit_points * 0.85
            fading = self._momentum_fading(pos.side)
            if pnl_pts >= soft_tp and fading:
                return "soft_tp_momentum_fade"
            if fading and pnl_pts >= self.cfg.take_profit_points * 0.40:
                return "momentum_fade_profit"

        return None

    def levels_for(self, side: Side, entry: float, point: float) -> tuple[float, float]:
        sl_dist = self.cfg.stop_loss_points * point
        tp_dist = self.cfg.take_profit_points * point
        if side == Side.BUY:
            return entry - sl_dist, entry + tp_dist
        return entry + sl_dist, entry - tp_dist
