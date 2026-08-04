"""
Central configuration loader.

Trade gates (min_score, min_reward_risk) and risk limits live ONLY here.
Other modules import from this module — never hard-code thresholds elsewhere.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config" / "settings.yaml"

load_dotenv(ROOT / ".env")


def _load_yaml() -> dict[str, Any]:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


@dataclass(frozen=True)
class TradeGates:
    """Single source of truth for trade allowance thresholds."""

    min_score: int
    min_reward_risk: float

    def allows(self, score: float, reward_risk: float) -> tuple[bool, list[str]]:
        """Return (allowed, failure_reasons). Enforced in one place only."""
        failures: list[str] = []
        if score < self.min_score:
            failures.append(
                f"Score {score:.0f}/100 is below minimum {self.min_score}"
            )
        # Tiny epsilon so exact 3.00 is not rejected by float noise
        if reward_risk + 1e-9 < self.min_reward_risk:
            failures.append(
                f"R:R {reward_risk:.2f} is below minimum {self.min_reward_risk:.2f}"
            )
        return (len(failures) == 0, failures)


@dataclass(frozen=True)
class ScoringWeights:
    h4_regime_alignment: int
    h1_trend_confirmation: int
    structure_break: int
    liquidity_sweep: int
    fair_value_gap: int
    order_block: int
    m5_trigger: int
    volatility_context: int
    vwap_alignment: int

    def total(self) -> int:
        return (
            self.h4_regime_alignment
            + self.h1_trend_confirmation
            + self.structure_break
            + self.liquidity_sweep
            + self.fair_value_gap
            + self.order_block
            + self.m5_trigger
            + self.volatility_context
            + self.vwap_alignment
        )

    def validate(self) -> None:
        t = self.total()
        if t != 100:
            raise ValueError(f"Scoring weights must sum to 100, got {t}")


@dataclass(frozen=True)
class RiskConfig:
    risk_percent: float
    atr_period: int
    atr_stop_multiplier: float
    atr_target_multiplier: float
    max_concurrent_trades: int
    daily_loss_limit_percent: float
    max_spread_pips: dict[str, float]


@dataclass(frozen=True)
class Settings:
    mode: str
    symbols: list[str]
    timeframes: dict[str, str]
    gates: TradeGates
    weights: ScoringWeights
    risk: RiskConfig
    sessions: dict[str, Any]
    analysis: dict[str, Any]
    backtest: dict[str, Any]
    paths: dict[str, str]
    mt5_login: int | None = None
    mt5_password: str | None = None
    mt5_server: str | None = None
    mt5_path: str | None = None

    @property
    def is_paper(self) -> bool:
        return self.mode.upper() == "PAPER"

    @property
    def is_live(self) -> bool:
        return self.mode.upper() == "LIVE"

    def max_spread_for(self, symbol: str) -> float:
        return float(
            self.risk.max_spread_pips.get(
                symbol, self.risk.max_spread_pips.get("default", 2.0)
            )
        )


_SETTINGS: Settings | None = None


def load_settings(reload: bool = False) -> Settings:
    """Load settings once; call with reload=True in tests."""
    global _SETTINGS
    if _SETTINGS is not None and not reload:
        return _SETTINGS

    raw = _load_yaml()
    mode = os.getenv("ATLAS_MODE", raw.get("mode", "PAPER")).upper()
    if mode not in ("PAPER", "LIVE"):
        raise ValueError(f"Invalid mode '{mode}'; must be PAPER or LIVE")

    gates = TradeGates(
        min_score=int(raw["gates"]["min_score"]),
        min_reward_risk=float(raw["gates"]["min_reward_risk"]),
    )
    weights = ScoringWeights(**raw["scoring_weights"])
    weights.validate()

    risk_raw = raw["risk"]
    risk = RiskConfig(
        risk_percent=float(risk_raw["risk_percent"]),
        atr_period=int(risk_raw["atr_period"]),
        atr_stop_multiplier=float(risk_raw["atr_stop_multiplier"]),
        atr_target_multiplier=float(risk_raw["atr_target_multiplier"]),
        max_concurrent_trades=int(risk_raw["max_concurrent_trades"]),
        daily_loss_limit_percent=float(risk_raw["daily_loss_limit_percent"]),
        max_spread_pips={k: float(v) for k, v in risk_raw["max_spread_pips"].items()},
    )

    _SETTINGS = Settings(
        mode=mode,
        symbols=list(raw["symbols"]),
        timeframes=dict(raw["timeframes"]),
        gates=gates,
        weights=weights,
        risk=risk,
        sessions=dict(raw["sessions"]),
        analysis=dict(raw["analysis"]),
        backtest=dict(raw["backtest"]),
        paths=dict(raw["paths"]),
        mt5_login=_env_int("MT5_LOGIN"),
        mt5_password=os.getenv("MT5_PASSWORD") or None,
        mt5_server=os.getenv("MT5_SERVER") or None,
        mt5_path=os.getenv("MT5_PATH") or None,
    )
    return _SETTINGS


def get_gates() -> TradeGates:
    """Convenience accessor for the single trade-gate definition."""
    return load_settings().gates


def _env_int(key: str) -> int | None:
    val = os.getenv(key)
    if val is None or val.strip() == "":
        return None
    return int(val)
