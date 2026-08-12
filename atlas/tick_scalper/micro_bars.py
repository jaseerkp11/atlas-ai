"""
Synthetic micro-candles built from live ticks.

MT5 has no native 1s/5s chart API — we aggregate ticks into N-second OHLC bars.
Also supports M1 direction from broker (PERIOD_M1).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional

from atlas.tick_scalper.mt5_feed import Tick

try:
    import MetaTrader5 as mt5

    MT5_OK = True
except ImportError:
    mt5 = None  # type: ignore
    MT5_OK = False


@dataclass(frozen=True)
class MicroBar:
    """One closed N-second (or M1) candle."""

    start_msc: int
    open: float
    high: float
    low: float
    close: float
    ticks: int

    @property
    def bullish(self) -> bool:
        return self.close > self.open

    @property
    def bearish(self) -> bool:
        return self.close < self.open

    @property
    def range(self) -> float:
        return self.high - self.low


class MicroBarBuilder:
    """
    Build closed bars every `period_seconds` from the tick stream.
    period_seconds=1 → 1-second candles; 5 → 5-second candles.
    """

    def __init__(self, period_seconds: int = 5, history: int = 64) -> None:
        self.period_ms = max(1, int(period_seconds)) * 1000
        self.closed: Deque[MicroBar] = deque(maxlen=history)
        self._bucket: int | None = None
        self._o = self._h = self._l = self._c = 0.0
        self._n = 0
        self._just_closed: MicroBar | None = None

    def on_tick(self, tick: Tick) -> Optional[MicroBar]:
        """
        Feed a tick. Returns the bar that just closed (if any), else None.
        """
        self._just_closed = None
        bucket = int(tick.time_msc // self.period_ms) * self.period_ms
        mid = tick.mid

        if self._bucket is None:
            self._bucket = bucket
            self._o = self._h = self._l = self._c = mid
            self._n = 1
            return None

        if bucket == self._bucket:
            self._h = max(self._h, mid)
            self._l = min(self._l, mid)
            self._c = mid
            self._n += 1
            return None

        # Bucket rolled → close previous bar
        bar = MicroBar(
            start_msc=self._bucket,
            open=self._o,
            high=self._h,
            low=self._l,
            close=self._c,
            ticks=self._n,
        )
        self.closed.append(bar)
        self._just_closed = bar

        # Start new bucket
        self._bucket = bucket
        self._o = self._h = self._l = self._c = mid
        self._n = 1
        return bar

    @property
    def last_closed(self) -> Optional[MicroBar]:
        return self.closed[-1] if self.closed else None

    def last_n_agree(self, side_buy: bool, n: int = 2) -> bool:
        if len(self.closed) < n:
            return False
        bars = list(self.closed)[-n:]
        if side_buy:
            return all(b.bullish for b in bars)
        return all(b.bearish for b in bars)


def fetch_m1_direction(symbol: str, use_mt5: bool) -> Optional[str]:
    """
    Return 'BUY' / 'SELL' from the last CLOSED M1 candle, or None.
    """
    if not use_mt5 or not MT5_OK or mt5 is None:
        return None
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 0, 5)
    if rates is None or len(rates) < 3:
        return None
    # rates[-1] is forming; rates[-2] last closed
    closed = rates[-2]
    o = float(closed["open"])
    c = float(closed["close"])
    if c > o:
        return "BUY"
    if c < o:
        return "SELL"
    return None


def fetch_m1_micro_trend(symbol: str, use_mt5: bool, lookback: int = 3) -> Optional[str]:
    """
    Majority direction of last N closed M1 candles.
    """
    if not use_mt5 or not MT5_OK or mt5 is None:
        return None
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 0, lookback + 2)
    if rates is None or len(rates) < lookback + 1:
        return None
    closed = rates[-(lookback + 1) : -1]
    ups = sum(1 for r in closed if float(r["close"]) > float(r["open"]))
    downs = sum(1 for r in closed if float(r["close"]) < float(r["open"]))
    if ups > downs:
        return "BUY"
    if downs > ups:
        return "SELL"
    return None
