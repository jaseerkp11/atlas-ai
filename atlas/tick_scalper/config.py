"""
Tick-scalper configuration loader (separate from candle bot settings).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "config" / "tick_scalper.yaml"
load_dotenv(ROOT / ".env")


@dataclass(frozen=True)
class TickScalperConfig:
    mode: str
    symbol: str
    magic: int
    poll_sleep_ms: int
    require_price_change: bool
    max_spread_points: float
    max_slippage_points: float
    risk_percent: float
    stop_loss_points: float
    take_profit_points: float
    max_open_positions: int
    daily_loss_limit_percent: float
    cooldown_ms_after_close: int
    vwap_enabled: bool
    vwap_near_points: float
    min_tick_momentum: int
    momentum_lookback: int
    min_tick_range_points: float
    session_hours_utc: tuple[int, int]
    log_every_tick: bool
    tick_summary_every_n: int
    log_dir: str

    @property
    def is_paper(self) -> bool:
        return self.mode.upper() == "PAPER"

    @property
    def is_live(self) -> bool:
        return self.mode.upper() == "LIVE"

    @property
    def point_size(self) -> float:
        """XAUUSD typical point (0.01)."""
        return 0.01


_CFG: TickScalperConfig | None = None


def load_tick_config(reload: bool = False) -> TickScalperConfig:
    global _CFG
    if _CFG is not None and not reload:
        return _CFG

    with open(CONFIG_PATH, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    mode = os.getenv("ATLAS_MODE", raw.get("mode", "PAPER")).upper()
    if mode not in ("PAPER", "LIVE"):
        raise ValueError(f"Invalid mode {mode}")

    t = raw["tick_scalper"]
    hours = t["session_hours_utc"]
    _CFG = TickScalperConfig(
        mode=mode,
        symbol=str(t["symbol"]),
        magic=int(t["magic"]),
        poll_sleep_ms=int(t["poll_sleep_ms"]),
        require_price_change=bool(t["require_price_change"]),
        max_spread_points=float(t["max_spread_points"]),
        max_slippage_points=float(t["max_slippage_points"]),
        risk_percent=float(t["risk_percent"]),
        stop_loss_points=float(t["stop_loss_points"]),
        take_profit_points=float(t["take_profit_points"]),
        max_open_positions=int(t["max_open_positions"]),
        daily_loss_limit_percent=float(t["daily_loss_limit_percent"]),
        cooldown_ms_after_close=int(t["cooldown_ms_after_close"]),
        vwap_enabled=bool(t["vwap_enabled"]),
        vwap_near_points=float(t["vwap_near_points"]),
        min_tick_momentum=int(t["min_tick_momentum"]),
        momentum_lookback=int(t["momentum_lookback"]),
        min_tick_range_points=float(t["min_tick_range_points"]),
        session_hours_utc=(int(hours[0]), int(hours[1])),
        log_every_tick=bool(t["log_every_tick"]),
        tick_summary_every_n=int(t["tick_summary_every_n"]),
        log_dir=str(t["log_dir"]),
    )
    return _CFG
