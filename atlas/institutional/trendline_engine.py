"""Trendline engine — swing-based lines with touch count & quality."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from atlas.analysis.structure import find_swing_highs, find_swing_lows
from atlas.institutional.models import Bias, ModuleScore


@dataclass
class Trendline:
    kind: str  # support | resistance
    touches: int
    quality: str  # Strong | Medium | Weak | Broken
    angle: float
    break_prob: float
    confluence: float
    y_at_end: float


@dataclass
class TrendlineReport:
    lines: list[Trendline]
    score: float
    bias: Bias
    summary: str
    module: ModuleScore


def _fit_line(points: list[tuple[int, float]]) -> tuple[float, float] | None:
    if len(points) < 2:
        return None
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    den = sum((x - mx) ** 2 for x in xs) or 1e-9
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den
    intercept = my - slope * mx
    return slope, intercept


def analyze_trendlines(df: pd.DataFrame, lookback: int = 3) -> TrendlineReport:
    if df is None or len(df) < 40:
        mod = ModuleScore("trendlines", 0, 5, Bias.NEUTRAL, "no data", False)
        return TrendlineReport([], 0, Bias.NEUTRAL, "no data", mod)

    sh = find_swing_highs(df, lookback=lookback)[-6:]
    sl = find_swing_lows(df, lookback=lookback)[-6:]
    mid = float(df["close"].iloc[-2])
    lines: list[Trendline] = []

    for kind, swings in (("resistance", sh), ("support", sl)):
        pts = [(s.index, s.price) for s in swings]
        fit = _fit_line(pts)
        if not fit:
            continue
        slope, intercept = fit
        end_i = len(df) - 2
        y = slope * end_i + intercept
        # Count near-touches
        touches = 0
        for i, p in pts:
            pred = slope * i + intercept
            if abs(pred - p) / max(abs(p), 1e-9) < 0.0015 or abs(pred - p) < 1.0:
                touches += 1
        broken = (kind == "support" and mid < y) or (kind == "resistance" and mid > y)
        if broken:
            quality = "Broken"
            break_prob = 0.85
        elif touches >= 3:
            quality = "Strong"
            break_prob = 0.25
        elif touches == 2:
            quality = "Medium"
            break_prob = 0.45
        else:
            quality = "Weak"
            break_prob = 0.6
        angle = abs(slope)
        conf = min(100.0, touches * 25 + (20 if quality == "Strong" else 0))
        lines.append(
            Trendline(kind, touches, quality, angle, break_prob, conf, y)
        )

    score = max((ln.confluence for ln in lines if ln.quality != "Broken"), default=25.0)
    bias = Bias.NEUTRAL
    for ln in lines:
        if ln.quality in ("Strong", "Medium") and ln.kind == "support" and mid >= ln.y_at_end:
            bias = Bias.BULLISH
        if ln.quality in ("Strong", "Medium") and ln.kind == "resistance" and mid <= ln.y_at_end:
            bias = Bias.BEARISH
    summary = f"lines={len(lines)} " + ", ".join(f"{l.kind}:{l.quality}" for l in lines)
    mod = ModuleScore("trendlines", score, 5, bias, summary, score >= 50)
    return TrendlineReport(lines, score, bias, summary, mod)
