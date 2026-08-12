"""
MT5 broker client with PAPER-safe fallback.

MetaTrader5 Python package is Windows-only. On non-Windows hosts (or when MT5
is unavailable), the client runs in simulated/paper mode with synthetic quotes
so analysis, scoring, and backtests still work.
"""

from __future__ import annotations

import logging
import platform
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from atlas.config import load_settings

logger = logging.getLogger(__name__)

try:
    import MetaTrader5 as mt5

    MT5_AVAILABLE = True
except ImportError:
    mt5 = None  # type: ignore
    MT5_AVAILABLE = False


TF_TO_MT5 = {
    "M1": 1,
    "M5": 5,
    "M15": 15,
    "H1": 16385,
    "H4": 16388,
}

# Human-readable aliases when mt5 module present
def _tf_constant(name: str) -> int:
    if MT5_AVAILABLE and mt5 is not None:
        mapping = {
            "M1": mt5.TIMEFRAME_M1,
            "M5": mt5.TIMEFRAME_M5,
            "M15": mt5.TIMEFRAME_M15,
            "H1": mt5.TIMEFRAME_H1,
            "H4": mt5.TIMEFRAME_H4,
        }
        if name not in mapping:
            raise KeyError(f"Unsupported timeframe {name!r}; supported: {sorted(mapping)}")
        return mapping[name]
    if name not in TF_TO_MT5:
        raise KeyError(f"Unsupported timeframe {name!r}; supported: {sorted(TF_TO_MT5)}")
    return TF_TO_MT5[name]


class MT5Client:
    """
    Broker interface.

    - PAPER + MT5 available: real live prices from the terminal, simulated orders only
    - LIVE + MT5 available: real prices AND real order_send
    - No MT5 / non-Windows: synthetic fallback (offline demos only — prices are NOT market)
    """

    def __init__(self) -> None:
        self._connected = False
        self._paper_balance = float(load_settings().backtest.get("initial_balance", 10000))
        self._paper_equity = self._paper_balance
        self._paper_positions: dict[str, dict[str, Any]] = {}
        self._use_real_mt5 = False
        self._data_source = "none"  # "mt5" | "synthetic"
        self._last_rates: dict[tuple[str, str], pd.DataFrame] = {}

    @property
    def using_live_market_data(self) -> bool:
        return self._use_real_mt5 and self._data_source == "mt5"

    # ── Connection ────────────────────────────────────────────────────────────

    def connect(self) -> bool:
        """
        Always prefer a real MT5 terminal connection for market data when possible.
        ATLAS_MODE only controls whether orders are real (LIVE) or simulated (PAPER).
        """
        settings = load_settings()

        if MT5_AVAILABLE and platform.system() == "Windows":
            if self._connect_mt5_terminal(settings):
                mode = "LIVE ORDERS" if settings.is_live else "PAPER ORDERS (no real sends)"
                logger.info(
                    "MT5 live market data OK | order mode=%s | login=%s balance=%s",
                    mode,
                    self.account_info_dict().get("login"),
                    self.account_balance(),
                )
                return True
            logger.error(
                "Could not connect to MetaTrader 5 terminal. "
                "Open MT5, log in, enable Algo Trading, check .env MT5_* values. "
                "Falling back to SYNTHETIC data (prices will NOT match the live market)."
            )

        # Synthetic fallback — not live market
        self._connected = True
        self._use_real_mt5 = False
        self._data_source = "synthetic"
        reasons = []
        if not MT5_AVAILABLE:
            reasons.append("MetaTrader5 package missing")
        if platform.system() != "Windows":
            reasons.append(f"OS={platform.system()}")
        reasons.append(f"order_mode={settings.mode}")
        logger.warning(
            "Using SYNTHETIC prices (%s). Gold/FX levels will look wrong vs the real market.",
            ", ".join(reasons),
        )
        return True

    def _connect_mt5_terminal(self, settings) -> bool:
        assert mt5 is not None
        kwargs: dict[str, Any] = {}
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
            logger.error("mt5.initialize failed: %s", mt5.last_error())
            self._connected = False
            self._use_real_mt5 = False
            self._data_source = "none"
            return False

        info = mt5.account_info()
        if info is None:
            logger.error("MT5 initialized but account_info() is None: %s", mt5.last_error())
            mt5.shutdown()
            return False

        # Ensure configured symbols are visible in Market Watch
        for symbol in settings.symbols:
            mt5.symbol_select(symbol, True)

        self._connected = True
        self._use_real_mt5 = True
        self._data_source = "mt5"
        # Sync paper accounting to the real account balance for realistic sizing
        self._paper_balance = float(info.balance)
        self._paper_equity = float(info.equity)
        return True

    def disconnect(self) -> None:
        if self._use_real_mt5 and mt5 is not None:
            mt5.shutdown()
        self._connected = False
        self._use_real_mt5 = False

    def is_connected(self) -> bool:
        if self._use_real_mt5 and mt5 is not None:
            return mt5.terminal_info() is not None
        return self._connected

    # ── Account ───────────────────────────────────────────────────────────────

    def account_balance(self) -> float:
        if self._use_real_mt5 and mt5 is not None:
            info = mt5.account_info()
            if info:
                return float(info.balance)
        return self._paper_balance

    def account_equity(self) -> float:
        if self._use_real_mt5 and mt5 is not None:
            info = mt5.account_info()
            if info:
                return float(info.equity)
        return self._paper_equity

    def account_free_margin(self) -> float:
        if self._use_real_mt5 and mt5 is not None:
            info = mt5.account_info()
            if info:
                return float(info.margin_free)
        return max(0.0, self._paper_equity * 0.9)

    def account_info_dict(self) -> dict[str, Any]:
        settings = load_settings()
        if self._use_real_mt5 and mt5 is not None:
            info = mt5.account_info()
            if info is None:
                return {}
            return {
                "login": info.login,
                "name": info.name,
                "server": info.server,
                "balance": info.balance,
                "equity": info.equity,
                "leverage": info.leverage,
                "data_source": "MT5_LIVE_MARKET",
                "order_mode": settings.mode,
            }
        return {
            "login": 0,
            "name": "SYNTHETIC",
            "server": "OFFLINE_SIM",
            "balance": self._paper_balance,
            "equity": self._paper_equity,
            "leverage": 100,
            "data_source": "SYNTHETIC_NOT_LIVE",
            "order_mode": settings.mode,
        }

    # ── Symbols / quotes ──────────────────────────────────────────────────────

    def symbol_spread_pips(self, symbol: str) -> float:
        if self._use_real_mt5 and mt5 is not None:
            tick = mt5.symbol_info_tick(symbol)
            info = mt5.symbol_info(symbol)
            if tick is None or info is None:
                return 999.0
            point = info.point or 0.00001
            # For JPY and gold, pip definition differs; use points/10 as pip approx for 5-digit
            digits = info.digits
            pip_size = point * 10 if digits in (3, 5) else point
            return abs(tick.ask - tick.bid) / pip_size
        # Simulated tight spread
        if symbol.upper().startswith("XAU"):
            return 18.0
        return 0.8

    def has_open_position(self, symbol: str) -> bool:
        if self._use_real_mt5 and mt5 is not None:
            positions = mt5.positions_get(symbol=symbol)
            return bool(positions)
        return symbol in self._paper_positions

    def open_position_count(self) -> int:
        if self._use_real_mt5 and mt5 is not None:
            positions = mt5.positions_get()
            return len(positions) if positions else 0
        return len(self._paper_positions)

    def symbol_trade_mode_allowed(self, symbol: str) -> bool:
        if self._use_real_mt5 and mt5 is not None:
            info = mt5.symbol_info(symbol)
            if info is None:
                return False
            # SYMBOL_TRADE_MODE_FULL = 4 typically
            return bool(info.visible) and info.trade_mode != 0
        return True

    def symbol_specs(self, symbol: str) -> dict[str, float]:
        if self._use_real_mt5 and mt5 is not None:
            info = mt5.symbol_info(symbol)
            if info:
                return {
                    "tick_size": float(info.trade_tick_size or info.point or 0.00001),
                    "tick_value": float(info.trade_tick_value or 1.0),
                    "lot_step": float(info.volume_step or 0.01),
                    "min_lot": float(info.volume_min or 0.01),
                    "max_lot": float(info.volume_max or 100.0),
                    "contract_size": float(info.trade_contract_size or 100000.0),
                }
        if symbol.upper().startswith("XAU"):
            return {
                "tick_size": 0.01,
                "tick_value": 1.0,
                "lot_step": 0.01,
                "min_lot": 0.01,
                "max_lot": 50.0,
                "contract_size": 100.0,
            }
        if "JPY" in symbol.upper():
            return {
                "tick_size": 0.001,
                "tick_value": 1.0,
                "lot_step": 0.01,
                "min_lot": 0.01,
                "max_lot": 100.0,
                "contract_size": 100000.0,
            }
        return {
            "tick_size": 0.00001,
            "tick_value": 1.0,
            "lot_step": 0.01,
            "min_lot": 0.01,
            "max_lot": 100.0,
            "contract_size": 100000.0,
        }

    def current_price(self, symbol: str) -> tuple[float, float]:
        """Return (bid, ask)."""
        if self._use_real_mt5 and mt5 is not None:
            mt5.symbol_select(symbol, True)
            tick = mt5.symbol_info_tick(symbol)
            if tick:
                return float(tick.bid), float(tick.ask)
        df = self._last_rates.get((symbol, "M5"))
        if df is not None and len(df):
            c = float(df["close"].iloc[-1])
            spread = 0.00010 if not symbol.startswith("XAU") else 0.20
            return c - spread / 2, c + spread / 2
        # Fallback synthetic mid
        mid = 1.1000 if not symbol.startswith("XAU") else 2400.0
        return mid, mid + 0.0001

    def live_tick_snapshot(self, symbol: str) -> dict[str, Any] | None:
        """Return live tick fields for price verification against the MT5 chart."""
        if not (self._use_real_mt5 and mt5 is not None):
            return None
        mt5.symbol_select(symbol, True)
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return None
        return {
            "symbol": symbol,
            "bid": float(tick.bid),
            "ask": float(tick.ask),
            "last": float(getattr(tick, "last", 0.0) or 0.0),
            "time": datetime.fromtimestamp(int(tick.time), tz=timezone.utc).isoformat(),
            "volume": int(getattr(tick, "volume", 0) or 0),
        }

    # ── Historical rates ──────────────────────────────────────────────────────

    def copy_rates(self, symbol: str, timeframe: str, bars: int = 500) -> pd.DataFrame:
        timeframe = timeframe.upper()
        if self._use_real_mt5 and mt5 is not None:
            mt5.symbol_select(symbol, True)
            rates = mt5.copy_rates_from_pos(symbol, _tf_constant(timeframe), 0, bars)
            if rates is None:
                logger.warning("No rates for %s %s: %s", symbol, timeframe, mt5.last_error())
                return pd.DataFrame()
            df = pd.DataFrame(rates)
            df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
            df = df.rename(columns={"tick_volume": "volume"})
            cols = ["time", "open", "high", "low", "close", "volume"]
            out = df[[c for c in cols if c in df.columns]].copy()
            self._last_rates[(symbol, timeframe)] = out
            return out

        # Simulated / cached — NOT live market
        cached = self._last_rates.get((symbol, timeframe))
        if cached is not None and len(cached) >= min(bars, 100):
            return cached.tail(bars).reset_index(drop=True)

        df = generate_synthetic_ohlc(symbol, timeframe, bars)
        self._last_rates[(symbol, timeframe)] = df
        return df

    def set_cached_rates(self, symbol: str, timeframe: str, df: pd.DataFrame) -> None:
        self._last_rates[(symbol, timeframe)] = df.copy()

    def update_paper_balance(self, balance: float, equity: float | None = None) -> None:
        self._paper_balance = balance
        self._paper_equity = equity if equity is not None else balance

    def register_paper_position(self, symbol: str, meta: dict[str, Any]) -> None:
        self._paper_positions[symbol] = meta

    def clear_paper_position(self, symbol: str) -> None:
        self._paper_positions.pop(symbol, None)

    # ── Orders (live only; paper goes through execution engine) ───────────────

    def place_market_order(
        self,
        symbol: str,
        direction: str,
        volume: float,
        stop: float,
        target: float,
        comment: str = "ATLAS",
    ) -> dict[str, Any]:
        settings = load_settings()
        # PAPER always simulates fills — even when MT5 live data is connected
        if settings.is_paper or not self._use_real_mt5 or mt5 is None:
            bid, ask = self.current_price(symbol)
            fill = ask if direction == "LONG" else bid
            ticket = int(datetime.now(timezone.utc).timestamp())
            self.register_paper_position(
                symbol,
                {"ticket": ticket, "direction": direction, "volume": volume, "entry": fill},
            )
            return {
                "ok": True,
                "ticket": ticket,
                "price": fill,
                "volume": volume,
                "retcode": 0,
                "comment": "PAPER_FILL",
            }

        bid, ask = self.current_price(symbol)
        order_type = mt5.ORDER_TYPE_BUY if direction == "LONG" else mt5.ORDER_TYPE_SELL
        price = ask if direction == "LONG" else bid
        filling = mt5.ORDER_FILLING_IOC

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": float(volume),
            "type": order_type,
            "price": price,
            "sl": float(stop),
            "tp": float(target),
            "deviation": 20,
            "magic": 20250803,
            "comment": comment[:31],
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": filling,
        }
        result = mt5.order_send(request)
        if result is None:
            return {"ok": False, "ticket": None, "retcode": -1, "comment": str(mt5.last_error())}
        ok = result.retcode == mt5.TRADE_RETCODE_DONE
        return {
            "ok": ok,
            "ticket": result.order,
            "price": result.price,
            "volume": result.volume,
            "retcode": result.retcode,
            "comment": result.comment,
        }


def generate_synthetic_ohlc(
    symbol: str,
    timeframe: str,
    bars: int,
    seed: int | None = None,
) -> pd.DataFrame:
    """
    Generate deterministic-ish OHLC for offline tests and paper mode without MT5.
    Not for performance claims — analysis plumbing only.
    """
    rng = np.random.default_rng(seed if seed is not None else abs(hash(symbol)) % (2**32))
    minutes = {"M1": 1, "M5": 5, "M15": 15, "H1": 60, "H4": 240}[timeframe.upper()]
    end = pd.Timestamp.now(tz="UTC").floor(f"{minutes}min")
    times = pd.date_range(end=end, periods=bars, freq=f"{minutes}min", tz="UTC")

    if symbol.upper().startswith("XAU"):
        price = 2350.0
        vol = 2.5
    elif "JPY" in symbol.upper():
        price = 150.0
        vol = 0.08
    else:
        price = 1.1000
        vol = 0.0008

    # Regime chunks: trending + ranging to create structure
    rets = rng.normal(0, vol, size=bars)
    for i in range(0, bars, max(bars // 8, 1)):
        chunk = slice(i, min(i + bars // 8, bars))
        drift = rng.choice([-1, 1]) * vol * 0.15
        rets[chunk] += drift

    closes = price + np.cumsum(rets)
    opens = np.roll(closes, 1)
    opens[0] = price
    spreads = np.abs(rng.normal(0, vol, size=bars))
    highs = np.maximum(opens, closes) + spreads
    lows = np.minimum(opens, closes) - spreads

    # Inject occasional sweep wicks
    for idx in rng.choice(bars, size=min(20, bars // 20), replace=False):
        if rng.random() > 0.5:
            lows[idx] -= vol * 3
        else:
            highs[idx] += vol * 3

    df = pd.DataFrame(
        {
            "time": times,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": rng.integers(100, 2000, size=bars),
        }
    )
    return df
