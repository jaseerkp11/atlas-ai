"""Institutional engine configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "config" / "institutional.yaml"
load_dotenv(ROOT / ".env")


@dataclass(frozen=True)
class InstitutionalConfig:
    mode: str
    symbol: str
    magic: int
    timeframes: tuple[str, ...]
    min_probability: float
    min_confidence: float
    min_confluence: float
    min_reward_risk: float
    require_htf_alignment: bool
    require_m1_timing: bool
    require_playbook: bool
    weights: dict[str, float]
    risk: dict[str, Any]
    news: dict[str, Any]
    analysis: dict[str, Any]
    log_dir: str
    challenge: dict[str, Any]

    @property
    def is_live(self) -> bool:
        return self.mode.upper() == "LIVE"

    @property
    def is_paper(self) -> bool:
        return self.mode.upper() == "PAPER"

    def challenge_max_risk_points(self) -> float:
        """Price distance ($) for max_risk_usd at configured lot size."""
        ch = self.challenge or {}
        if not ch.get("enabled", True):
            return 1e9
        lot = max(float(ch.get("lot_size", 0.1)), 1e-9)
        max_usd = float(ch.get("max_risk_usd", 50.0))
        per_lot = max(float(ch.get("usd_per_price_unit_per_lot", 100.0)), 1e-9)
        return max_usd / (per_lot * lot)


_CFG: InstitutionalConfig | None = None


def load_institutional_config(reload: bool = False) -> InstitutionalConfig:
    global _CFG
    if _CFG is not None and not reload:
        return _CFG
    with open(CONFIG_PATH, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    mode = os.getenv("ATLAS_MODE", raw.get("mode", "PAPER")).upper()
    if mode not in ("PAPER", "LIVE"):
        raise ValueError(f"Invalid mode {mode}")
    t = raw["institutional"]
    _CFG = InstitutionalConfig(
        mode=mode,
        symbol=str(t["symbol"]),
        magic=int(t["magic"]),
        timeframes=tuple(str(x) for x in t["timeframes"]),
        min_probability=float(t["min_probability"]),
        min_confidence=float(t["min_confidence"]),
        min_confluence=float(t["min_confluence"]),
        min_reward_risk=float(t["min_reward_risk"]),
        require_htf_alignment=bool(t["require_htf_alignment"]),
        require_m1_timing=bool(t["require_m1_timing"]),
        require_playbook=bool(t.get("require_playbook", True)),
        weights={k: float(v) for k, v in t["weights"].items()},
        risk=dict(t["risk"]),
        news=dict(t["news"]),
        analysis=dict(t["analysis"]),
        log_dir=str(t["logging"]["dir"]),
        challenge=dict(t.get("challenge") or {}),
    )
    return _CFG
