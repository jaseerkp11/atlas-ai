"""
Tick strategy — RAPID CYCLE with M1 + synthetic 5s/1s micro-candle filters.

Entry requires:
  - tick momentum + spread
  - last closed micro-bar(s) agree (1s or 5s built from ticks)
  - optional M1 candle alignment (broker PERIOD_M1)

Exit still every tick for instant profit / SL (high-speed close).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from atlas.tick_scalper.config import TickScalperConfig, load_tick_config
from atlas.tick_scalper.micro_bars import MicroBar, MicroBarBuilder, fetch_m1_micro_trend
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
    profit_target_points: float = 0.0


class TickStrategy:
    def __init__(self, cfg: TickScalperConfig | None = None) -> None:
        self.cfg = cfg or load_tick_config()
        self._mids: deque[float] = deque(maxlen=max(32, self.cfg.momentum_lookback * 4))
        self._vwap_num = 0.0
        self._vwap_den = 0.0
        self._vwap_day = -1
        self.micro = MicroBarBuilder(period_seconds=self.cfg.micro_tf_seconds)
        self._last_closed_micro: Optional[MicroBar] = None
        self._m1_cache: Optional[str] = None
        self._m1_cache_bucket: int = -1
        self._use_mt5 = False

    def set_mt5_live(self, live: bool) -> None:
        self._use_mt5 = bool(live)

    def _reset_vwap_if_new_day(self, tick: Tick) -> None:
        day = int(tick.time_msc // 86_400_000)
        if day != self._vwap_day:
            self._vwap_day = day
            self._vwap_num = 0.0
            self._vwap_den = 0.0

    def update(self, tick: Tick, point: float) -> None:
        self._mids.append(tick.mid)
        closed = self.micro.on_tick(tick)
        if closed is not None:
            self._last_closed_micro = closed
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

    def _m1_direction(self, tick: Tick) -> Optional[str]:
        # Refresh at most once per M1 bucket
        bucket = int(tick.time_msc // 60_000)
        if bucket == self._m1_cache_bucket and self._m1_cache is not None:
            return self._m1_cache
        direction = fetch_m1_micro_trend(
            self.cfg.symbol, self._use_mt5, lookback=self.cfg.m1_lookback
        )
        self._m1_cache_bucket = bucket
        self._m1_cache = direction
        return direction

    def evaluate_entry(self, tick: Tick, point: float) -> Signal | None:
        spread_pts = tick.spread_points(point)
        if spread_pts > self.cfg.max_spread_points:
            return None

        # Prefer entries right as a micro-bar closes (cleaner edge)
        if self.cfg.require_micro_bar_close and self.micro._just_closed is None:
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
            reason_parts.append(f"tick-mom up {ups}/{self.cfg.momentum_lookback}")
        elif downs >= self.cfg.min_tick_momentum and downs > ups:
            side = Side.SELL
            reason_parts.append(f"tick-mom down {downs}/{self.cfg.momentum_lookback}")
        else:
            return None

        if self.cfg.rapid_cycle and len(window) >= 2:
            last_up = window[-1] > window[-2]
            if side == Side.BUY and not last_up:
                return None
            if side == Side.SELL and last_up:
                return None

        # --- Micro TF filter (1s or 5s synthetic candles) ---
        n_agree = max(1, self.cfg.micro_bars_agree)
        if len(self.micro.closed) < n_agree:
            return None
        last = self.micro.last_closed
        assert last is not None
        if side == Side.BUY and not last.bullish:
            return None
        if side == Side.SELL and not last.bearish:
            return None
        if not self.micro.last_n_agree(side == Side.BUY, n_agree):
            return None
        reason_parts.append(
            f"{self.cfg.micro_tf_seconds}s×{n_agree} "
            f"{'bull' if side == Side.BUY else 'bear'} "
            f"O={last.open:.2f} C={last.close:.2f}"
        )

        # --- M1 filter (broker candle) ---
        if self.cfg.use_m1_filter:
            m1 = self._m1_direction(tick)
            if m1 is None and self._use_mt5:
                return None  # wait until M1 data available
            if m1 is not None and m1 != side.value:
                return None
            if m1 is not None:
                reason_parts.append(f"M1={m1}")

        vwap = self.vwap
        if self.cfg.vwap_enabled and vwap > 0:
            dist_pts = abs(tick.mid - vwap) / point
            near = dist_pts <= self.cfg.vwap_near_points
            if side == Side.BUY and tick.mid < vwap and not near:
                return None
            if side == Side.SELL and tick.mid > vwap and not near:
                return None
            reason_parts.append(f"VWAP={vwap:.2f}")

        reason_parts.append(f"spread={spread_pts:.1f}pts")
        # Consume just-closed flag so we don't re-fire every tick in same bar
        if self.cfg.require_micro_bar_close:
            self.micro._just_closed = None

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

        # Micro-bar flip while green → bank profit
        just = self.micro._just_closed
        if just is not None and pnl_pts > 0:
            if pos.side == Side.BUY and just.bearish:
                return "micro_bar_fade_profit"
            if pos.side == Side.SELL and just.bullish:
                return "micro_bar_fade_profit"

        if self.cfg.rapid_cycle and pnl_pts > 0 and self._momentum_fading(pos.side):
            return "rapid_fade_profit"

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
