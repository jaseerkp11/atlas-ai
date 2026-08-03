"""
MT5 tick feed — lowest practical latency for Python.

Reads symbol_info_tick in a tight loop and yields only when the quote changes
(or when require_price_change is False).
"""

from __future__ import annotations

import platform
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Deque, Iterator

from atlas.tick_scalper.config import TickScalperConfig, load_tick_config

try:
    import MetaTrader5 as mt5

    MT5_OK = True
except ImportError:
    mt5 = None  # type: ignore
    MT5_OK = False


@dataclass(frozen=True)
class Tick:
    time_msc: int
    bid: float
    ask: float
    last: float
    volume: int

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2.0

    @property
    def spread(self) -> float:
        return self.ask - self.bid

    def spread_points(self, point: float) -> float:
        return self.spread / point if point > 0 else 0.0


class MT5TickFeed:
    """
    Live tick source.

    PAPER on non-Windows / missing MT5 uses a synthetic jitter feed so the
    engine can be exercised offline (NOT for real trading decisions).
    """

    def __init__(self, cfg: TickScalperConfig | None = None) -> None:
        self.cfg = cfg or load_tick_config()
        self._connected = False
        self._use_mt5 = False
        self._last_key: tuple[float, float, int] | None = None
        self._history: Deque[Tick] = deque(maxlen=max(64, self.cfg.momentum_lookback * 4))
        self._synth_price = 4050.0

    @property
    def history(self) -> Deque[Tick]:
        return self._history

    def connect(self) -> bool:
        cfg = self.cfg
        if MT5_OK and platform.system() == "Windows":
            assert mt5 is not None
            # Prefer already-logged terminal; optional env credentials via atlas.config
            from atlas.config import load_settings

            settings = load_settings()
            kwargs = {}
            if settings.mt5_path:
                kwargs["path"] = settings.mt5_path
            if settings.mt5_login and settings.mt5_password and settings.mt5_server:
                kwargs.update(
                    login=int(settings.mt5_login),
                    password=settings.mt5_password,
                    server=settings.mt5_server,
                )
            ok = mt5.initialize(**kwargs) if kwargs else mt5.initialize()
            if not ok:
                self._connected = False
                self._use_mt5 = False
                return False
            mt5.symbol_select(cfg.symbol, True)
            info = mt5.account_info()
            if info is None:
                mt5.shutdown()
                return False
            self._connected = True
            self._use_mt5 = True
            return True

        # Synthetic feed (dev / non-Windows)
        self._connected = True
        self._use_mt5 = False
        return True

    def disconnect(self) -> None:
        if self._use_mt5 and mt5 is not None:
            mt5.shutdown()
        self._connected = False
        self._use_mt5 = False

    def account_balance(self) -> float:
        if self._use_mt5 and mt5 is not None:
            info = mt5.account_info()
            return float(info.balance) if info else 0.0
        return 10000.0

    def account_equity(self) -> float:
        if self._use_mt5 and mt5 is not None:
            info = mt5.account_info()
            return float(info.equity) if info else 0.0
        return self.account_balance()

    def symbol_point(self) -> float:
        if self._use_mt5 and mt5 is not None:
            info = mt5.symbol_info(self.cfg.symbol)
            if info and info.point:
                return float(info.point)
        return self.cfg.point_size

    def symbol_volume_limits(self) -> tuple[float, float, float]:
        """min_lot, lot_step, max_lot"""
        if self._use_mt5 and mt5 is not None:
            info = mt5.symbol_info(self.cfg.symbol)
            if info:
                return (
                    float(info.volume_min or 0.01),
                    float(info.volume_step or 0.01),
                    float(info.volume_max or 50.0),
                )
        return 0.01, 0.01, 50.0

    def open_position_count(self) -> int:
        if self._use_mt5 and mt5 is not None:
            positions = mt5.positions_get(symbol=self.cfg.symbol)
            if not positions:
                return 0
            return sum(1 for p in positions if int(p.magic) == self.cfg.magic)
        return 0

    def read_tick(self) -> Tick | None:
        """Single non-blocking tick read. Returns None if unchanged (optional)."""
        if self._use_mt5 and mt5 is not None:
            t = mt5.symbol_info_tick(self.cfg.symbol)
            if t is None:
                return None
            tick = Tick(
                time_msc=int(t.time_msc),
                bid=float(t.bid),
                ask=float(t.ask),
                last=float(getattr(t, "last", 0.0) or 0.0),
                volume=int(getattr(t, "volume", 0) or 0),
            )
        else:
            # Synthetic micro-moves for offline plumbing tests
            import math
            import random

            self._synth_price += random.uniform(-0.15, 0.15)
            spread = 0.25
            now_msc = int(time.time() * 1000)
            tick = Tick(
                time_msc=now_msc,
                bid=self._synth_price,
                ask=self._synth_price + spread,
                last=self._synth_price,
                volume=1,
            )

        key = (tick.bid, tick.ask, tick.time_msc)
        if self.cfg.require_price_change and key == self._last_key:
            return None
        self._last_key = key
        self._history.append(tick)
        return tick

    def stream(self) -> Iterator[Tick]:
        """
        Infinite tick stream. Sleeps poll_sleep_ms between polls when set,
        otherwise yields as fast as the terminal provides quotes.
        """
        sleep_s = max(0.0, self.cfg.poll_sleep_ms / 1000.0)
        while self._connected:
            tick = self.read_tick()
            if tick is not None:
                yield tick
            if sleep_s > 0:
                time.sleep(sleep_s)
            else:
                # Yield GIL briefly to keep CPU reasonable even in busy-wait
                time.sleep(0)
