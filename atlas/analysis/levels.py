"""
Support & resistance from live multi-timeframe market structure.

Levels are derived from the same live MT5 OHLC you already verified (same market
as TradingView for that symbol). We do not scrape TradingView (fragile / ToS);
instead we print levels you can draw on any chart (MT5 or TradingView).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from atlas.analysis.structure import find_swing_highs, find_swing_lows
from atlas.config import load_settings


@dataclass
class SRLevel:
    price: float
    kind: str  # "resistance" | "support"
    timeframes: list[str] = field(default_factory=list)
    touches: int = 1
    strength: str = "MODERATE"  # WEAK | MODERATE | STRONG
    distance_pips: float = 0.0

    def label(self) -> str:
        tfs = "+".join(self.timeframes)
        return f"{self.kind.upper():11} {self.price:.5f}  [{tfs}]  touches={self.touches}  {self.strength}"


@dataclass
class SRMap:
    symbol: str
    mid: float
    supports: list[SRLevel] = field(default_factory=list)
    resistances: list[SRLevel] = field(default_factory=list)
    nearest_support: SRLevel | None = None
    nearest_resistance: SRLevel | None = None

    def chart_lines(self) -> list[str]:
        """Human lines to draw on TradingView / MT5."""
        lines = [
            f"=== {self.symbol} S/R (live market structure) mid={self.mid:.5f} ===",
            "Draw these as horizontal lines on your chart:",
        ]
        for r in self.resistances[:5]:
            lines.append(f"  RES  {r.price:.5f}   ({'+'.join(r.timeframes)}, {r.strength})")
        for s in self.supports[:5]:
            lines.append(f"  SUP  {s.price:.5f}   ({'+'.join(s.timeframes)}, {s.strength})")
        if self.nearest_support:
            lines.append(
                f"Nearest support    : {self.nearest_support.price:.5f} "
                f"({self.nearest_support.distance_pips:.1f} pips below)"
            )
        if self.nearest_resistance:
            lines.append(
                f"Nearest resistance : {self.nearest_resistance.price:.5f} "
                f"({self.nearest_resistance.distance_pips:.1f} pips above)"
            )
        return lines


def _pip_size(symbol: str, price: float) -> float:
    if symbol.upper().startswith("XAU"):
        return 0.1
    if "JPY" in symbol.upper():
        return 0.01
    return 0.0001


def build_sr_map(
    symbol: str,
    frames: dict[str, pd.DataFrame],
    mid: float | None = None,
    cluster_atr_frac: float = 0.25,
) -> SRMap:
    """
    Cluster swing highs/lows across H4/H1/M15 into support & resistance zones.
    Multi-TF confluence → stronger level. Near-price levels preferred for scalp cards.
    """
    settings = load_settings()
    swing_lb = int(settings.analysis.get("swing_lookback", 5))
    raw: list[tuple[float, str, str]] = []

    for tf in ("H4", "H1", "M15", "M5"):
        df = frames.get(tf)
        if df is None or len(df) < swing_lb * 2 + 5:
            continue
        lb = swing_lb if tf != "M5" else max(3, swing_lb - 1)
        for s in find_swing_highs(df, lookback=lb)[-12:]:
            raw.append((s.price, "resistance", tf))
        for s in find_swing_lows(df, lookback=lb)[-12:]:
            raw.append((s.price, "support", tf))

    if mid is None:
        m15 = frames.get("M15")
        if m15 is None:
            m15 = frames.get("M5")
        mid = float(m15["close"].iloc[-1]) if m15 is not None and len(m15) else 0.0

    m15 = frames.get("M15")
    if m15 is None:
        m15 = frames.get("M5")
    if m15 is not None and len(m15) >= 20:
        atr_proxy = float((m15["high"] - m15["low"]).tail(20).mean())
    else:
        atr_proxy = mid * 0.001 if mid else 0.001
    tol = max(atr_proxy * cluster_atr_frac, _pip_size(symbol, mid) * 3)

    supports = _cluster(raw, "support", tol, mid, symbol)
    resistances = _cluster(raw, "resistance", tol, mid, symbol)

    # Keep only levels near live price for scalp watch cards
    prox = float(settings.analysis.get("sr_proximity_atr", 8.0))
    max_dist = atr_proxy * prox
    supports = [s for s in supports if abs(s.price - mid) <= max_dist]
    resistances = [r for r in resistances if abs(r.price - mid) <= max_dist]

    below = [s for s in supports if s.price <= mid]
    above = [r for r in resistances if r.price >= mid]
    nearest_s = max(below, key=lambda x: x.price) if below else None
    nearest_r = min(above, key=lambda x: x.price) if above else None

    pip = _pip_size(symbol, mid)
    if nearest_s:
        nearest_s.distance_pips = abs(mid - nearest_s.price) / pip
    if nearest_r:
        nearest_r.distance_pips = abs(nearest_r.price - mid) / pip

    resistances.sort(key=lambda x: x.price)
    supports.sort(key=lambda x: x.price, reverse=True)

    return SRMap(
        symbol=symbol,
        mid=mid,
        supports=supports,
        resistances=list(reversed(resistances)),
        nearest_support=nearest_s,
        nearest_resistance=nearest_r,
    )


def _cluster(
    raw: list[tuple[float, str, str]],
    kind: str,
    tol: float,
    mid: float,
    symbol: str,
) -> list[SRLevel]:
    points = sorted([p for p in raw if p[1] == kind], key=lambda x: x[0])
    if not points:
        return []

    clusters: list[list[tuple[float, str, str]]] = []
    current = [points[0]]
    for p in points[1:]:
        if abs(p[0] - current[-1][0]) <= tol:
            current.append(p)
        else:
            clusters.append(current)
            current = [p]
    clusters.append(current)

    levels: list[SRLevel] = []
    for cluster in clusters:
        prices = [c[0] for c in cluster]
        tfs = sorted({c[2] for c in cluster})
        price = sum(prices) / len(prices)
        touches = len(cluster)
        strength = "WEAK"
        if touches >= 3 or len(tfs) >= 2:
            strength = "MODERATE"
        if touches >= 4 and len(tfs) >= 2:
            strength = "STRONG"
        if len(tfs) >= 3:
            strength = "STRONG"
        pip = _pip_size(symbol, mid)
        levels.append(
            SRLevel(
                price=price,
                kind=kind,
                timeframes=tfs,
                touches=touches,
                strength=strength,
                distance_pips=abs(mid - price) / pip if pip else 0.0,
            )
        )
    return levels
