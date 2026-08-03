"""
Tick-scalper configuration loader (separate from candle bot settings).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

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
    use_fixed_lots: bool
    fixed_lots: float
    max_lots: float
    risk_percent: float
    rapid_cycle: bool
    instant_profit_points: float
    profit_ladder_points: tuple[float, ...]
    stop_loss_points: float
    take_profit_points: float
    max_open_positions: int
    burst_fill: bool
    alternate_batch_side: bool
    require_flat_before_next_batch: bool
    pyramid_winners_only: bool
    min_points_between_entries: float
    daily_loss_limit_percent: float
    cooldown_ms_after_close: int
    use_m1_price_action: bool
    m1_structure_bars: int
    m1_poll_seconds: float
    micro_tf_seconds: int
    require_micro_bar_close: bool
    micro_bars_agree: int
    use_m1_filter: bool
    m1_lookback: int
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
        return 0.01

    def profit_target_for_slot(self, slot_index: int) -> float:
        if self.profit_ladder_points:
            return float(self.profit_ladder_points[slot_index % len(self.profit_ladder_points)])
        return self.instant_profit_points


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
    ladder = t.get("profit_ladder_points") or [t.get("instant_profit_points", 5.0)]
    _CFG = TickScalperConfig(
        mode=mode,
        symbol=str(t["symbol"]),
        magic=int(t["magic"]),
        poll_sleep_ms=int(t["poll_sleep_ms"]),
        require_price_change=bool(t["require_price_change"]),
        max_spread_points=float(t["max_spread_points"]),
        max_slippage_points=float(t["max_slippage_points"]),
        use_fixed_lots=bool(t.get("use_fixed_lots", True)),
        fixed_lots=float(t.get("fixed_lots", 1.0)),
        max_lots=float(t.get("max_lots", 1.0)),
        risk_percent=float(t["risk_percent"]),
        rapid_cycle=bool(t.get("rapid_cycle", True)),
        instant_profit_points=float(t.get("instant_profit_points", 5.0)),
        profit_ladder_points=tuple(float(x) for x in ladder),
        stop_loss_points=float(t["stop_loss_points"]),
        take_profit_points=float(t["take_profit_points"]),
        max_open_positions=int(t["max_open_positions"]),
        burst_fill=bool(t.get("burst_fill", True)),
        alternate_batch_side=bool(t.get("alternate_batch_side", True)),
        require_flat_before_next_batch=bool(t.get("require_flat_before_next_batch", True)),
        pyramid_winners_only=bool(t.get("pyramid_winners_only", False)),
        min_points_between_entries=float(t.get("min_points_between_entries", 0.0)),
        daily_loss_limit_percent=float(t["daily_loss_limit_percent"]),
        cooldown_ms_after_close=int(t["cooldown_ms_after_close"]),
        use_m1_price_action=bool(t.get("use_m1_price_action", True)),
        m1_structure_bars=int(t.get("m1_structure_bars", 12)),
        m1_poll_seconds=float(t.get("m1_poll_seconds", 1)),
        micro_tf_seconds=int(t.get("micro_tf_seconds", 5)),
        require_micro_bar_close=bool(t.get("require_micro_bar_close", False)),
        micro_bars_agree=int(t.get("micro_bars_agree", 1)),
        use_m1_filter=bool(t.get("use_m1_filter", False)),
        m1_lookback=int(t.get("m1_lookback", 3)),
        vwap_enabled=bool(t.get("vwap_enabled", False)),
        vwap_near_points=float(t.get("vwap_near_points", 80.0)),
        min_tick_momentum=int(t.get("min_tick_momentum", 1)),
        momentum_lookback=int(t.get("momentum_lookback", 2)),
        min_tick_range_points=float(t.get("min_tick_range_points", 0.0)),
        session_hours_utc=(int(hours[0]), int(hours[1])),
        log_every_tick=bool(t["log_every_tick"]),
        tick_summary_every_n=int(t["tick_summary_every_n"]),
        log_dir=str(t["log_dir"]),
    )
    return _CFG
