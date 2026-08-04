"""Price action + trend filter (EMA/ADX/ATR/VWAP) + session + news."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import pandas as pd

from atlas.analysis.volatility import atr as atr_series, latest_atr
from atlas.analysis.vwap import evaluate_vwap
from atlas.institutional.models import Bias, ModuleScore


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _adx(df: pd.DataFrame, period: int = 14) -> float:
    if len(df) < period + 5:
        return 0.0
    high = df["high"]
    low = df["low"]
    close = df["close"]
    plus_dm = high.diff().clip(lower=0)
    minus_dm = (-low.diff()).clip(lower=0)
    tr = pd.concat(
        [
            (high - low),
            (high - close.shift()).abs(),
            (low - close.shift()).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.ewm(alpha=1 / period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr.replace(0, 1e-9))
    minus_di = 100 * (minus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr.replace(0, 1e-9))
    dx = (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, 1e-9) * 100
    return float(dx.ewm(alpha=1 / period, adjust=False).mean().iloc[-1])


@dataclass
class PriceActionReport:
    patterns: list[str]
    score: float
    bias: Bias
    summary: str
    module: ModuleScore


def analyze_price_action(df: pd.DataFrame) -> PriceActionReport:
    if df is None or len(df) < 5:
        mod = ModuleScore("price_action", 0, 8, Bias.NEUTRAL, "no data", False)
        return PriceActionReport([], 0, Bias.NEUTRAL, "no data", mod)
    # Use last closed candle
    c = df.iloc[-2]
    p = df.iloc[-3]
    o, h, l, cl = float(c["open"]), float(c["high"]), float(c["low"]), float(c["close"])
    body = abs(cl - o)
    rng = max(h - l, 1e-9)
    upper = h - max(o, cl)
    lower = min(o, cl) - l
    patterns: list[str] = []
    bias = Bias.NEUTRAL
    score = 40.0

    # Pin bar
    if lower > body * 2 and upper < body:
        patterns.append("bullish_pin")
        bias = Bias.BULLISH
        score = 70
    elif upper > body * 2 and lower < body:
        patterns.append("bearish_pin")
        bias = Bias.BEARISH
        score = 70

    # Engulfing
    po, pcl = float(p["open"]), float(p["close"])
    if cl > o and pcl < po and cl >= po and o <= pcl:
        patterns.append("bullish_engulfing")
        bias = Bias.BULLISH
        score = max(score, 75)
    if cl < o and pcl > po and cl <= po and o >= pcl:
        patterns.append("bearish_engulfing")
        bias = Bias.BEARISH
        score = max(score, 75)

    # Inside / outside
    if h < float(p["high"]) and l > float(p["low"]):
        patterns.append("inside_bar")
        score = max(score, 45)
    if h > float(p["high"]) and l < float(p["low"]):
        patterns.append("outside_bar")
        score = max(score, 55)

    # Momentum
    if body / rng >= 0.7:
        patterns.append("momentum_candle")
        if cl > o:
            bias = Bias.BULLISH if bias == Bias.NEUTRAL else bias
        else:
            bias = Bias.BEARISH if bias == Bias.NEUTRAL else bias
        score = max(score, 60)

    summary = ", ".join(patterns) if patterns else "no clear PA"
    mod = ModuleScore("price_action", score, 8, bias, summary, score >= 60 and bias != Bias.NEUTRAL)
    return PriceActionReport(patterns, score, bias, summary, mod)


@dataclass
class TrendReport:
    regime: str
    score: float
    bias: Bias
    adx: float
    ema50: float
    ema200: float
    vwap: float
    summary: str
    module: ModuleScore
    momentum_module: ModuleScore
    volatility_module: ModuleScore


def analyze_trend(df: pd.DataFrame) -> TrendReport:
    if df is None or len(df) < 210:
        # Still try with shorter if possible
        pass
    if df is None or len(df) < 50:
        empty = ModuleScore("trend", 0, 12, Bias.NEUTRAL, "no data", False)
        mom = ModuleScore("momentum", 0, 5, Bias.NEUTRAL, "no data", False)
        vol = ModuleScore("volatility", 0, 4, Bias.NEUTRAL, "no data", False)
        return TrendReport("unknown", 0, Bias.NEUTRAL, 0, 0, 0, 0, "no data", empty, mom, vol)

    close = df["close"]
    ema50 = float(_ema(close, 50).iloc[-2])
    ema200 = float(_ema(close, min(200, len(close) - 1)).iloc[-2])
    mid = float(close.iloc[-2])
    adx = _adx(df)
    atr_v = latest_atr(df, 14) or 1.0
    atr_s = atr_series(df, 14)
    expanding = False
    if atr_s is not None and len(atr_s) > 20:
        expanding = float(atr_s.iloc[-2]) > float(atr_s.iloc[-20]) * 1.15

    from atlas.models import Direction as Dir

    vwap_r = evaluate_vwap(df, Dir.NEUTRAL, atr=atr_v)
    vwap = float(getattr(vwap_r, "value", None) or mid)

    bias = Bias.NEUTRAL
    if mid > ema50 > ema200:
        bias = Bias.BULLISH
    elif mid < ema50 < ema200:
        bias = Bias.BEARISH

    if adx >= 25:
        regime = "Trending"
        if expanding:
            regime = "High Momentum"
    elif adx < 18:
        regime = "Ranging"
        if not expanding:
            regime = "Low Momentum"
    else:
        regime = "Transitional"

    score = 40.0
    if bias != Bias.NEUTRAL:
        score += 25
    if adx >= 25:
        score += 20
    if (bias == Bias.BULLISH and mid >= vwap) or (bias == Bias.BEARISH and mid <= vwap):
        score += 10
    score = min(100.0, score)

    summary = f"{regime} ADX={adx:.1f} EMA50={ema50:.2f} EMA200={ema200:.2f} VWAP={vwap:.2f}"
    mod = ModuleScore("trend", score, 12, bias, summary, score >= 55 and bias != Bias.NEUTRAL)
    mom_score = min(100.0, adx * 2.5)
    mom = ModuleScore(
        "momentum",
        mom_score,
        5,
        bias if adx >= 20 else Bias.NEUTRAL,
        f"ADX={adx:.1f}",
        adx >= 20,
    )
    vol_score = 70.0 if expanding else 45.0
    vol = ModuleScore(
        "volatility",
        vol_score,
        4,
        Bias.NEUTRAL,
        f"ATR={atr_v:.2f} expanding={expanding}",
        True,
    )
    return TrendReport(regime, score, bias, adx, ema50, ema200, vwap, summary, mod, mom, vol)


@dataclass
class SessionReport:
    best_session: str
    ranking: list[str]
    score: float
    summary: str
    module: ModuleScore
    trade_window_ok: bool


def analyze_session(now: datetime | None = None) -> SessionReport:
    now = now or datetime.now(timezone.utc)
    hour = now.hour
    # UTC approximations
    sessions = {
        "Asian": (0, 7),
        "London": (7, 12),
        "NewYork": (12, 17),
        "London-NY Overlap": (12, 16),
    }
    active = []
    for name, (a, b) in sessions.items():
        if a <= hour < b:
            active.append(name)
    # Quality ranking for gold
    quality_order = ["London-NY Overlap", "London", "NewYork", "Asian"]
    ranking = [s for s in quality_order if s in active] + [s for s in quality_order if s not in active]
    best = ranking[0]
    # Score by current window
    if "London-NY Overlap" in active:
        score = 90.0
    elif "London" in active:
        score = 80.0
    elif "NewYork" in active:
        score = 75.0
    elif "Asian" in active:
        score = 45.0
    else:
        score = 30.0
        best = "Off-hours"
    ok = score >= 50
    summary = f"UTC={hour:02d}:00 active={active or ['none']} best_today={ranking[0]}"
    mod = ModuleScore("session", score, 4, Bias.NEUTRAL, summary, ok)
    return SessionReport(best, ranking, score, summary, mod, ok)


@dataclass
class NewsReport:
    status: str  # Trade Today | Trade With Caution | Avoid Trading Today
    confidence: float
    score: float
    summary: str
    module: ModuleScore
    block_trading: bool


def analyze_news(enabled: bool = True) -> NewsReport:
    """
    News confidence. Attempts a lightweight public calendar; falls back to
    weekday heuristic so the engine never hard-crashes offline.
    """
    if not enabled:
        mod = ModuleScore("news", 70, 4, Bias.NEUTRAL, "news filter disabled", True)
        return NewsReport("Trade Today", 70, 70, "disabled", mod, False)

    # Heuristic: Fridays late / known high-impact weekday windows → caution
    now = datetime.now(timezone.utc)
    status = "Trade Today"
    score = 75.0
    block = False
    detail = "no high-impact block detected (heuristic)"

    # Try optional fetch (best-effort)
    try:
        import json
        import urllib.request

        # Public sample endpoint — if blocked, ignore
        url = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
        req = urllib.request.Request(url, headers={"User-Agent": "ATLAS-Institutional/1.0"})
        with urllib.request.urlopen(req, timeout=3) as resp:
            events = json.loads(resp.read().decode("utf-8"))
        high_near = 0
        for ev in events:
            if str(ev.get("impact", "")).lower() != "high":
                continue
            # Title may mention USD / gold-sensitive
            title = str(ev.get("title", "")).lower()
            country = str(ev.get("country", "")).upper()
            if country not in ("USD", "United States", "US") and "usd" not in title:
                continue
            high_near += 1
        if high_near >= 3:
            status = "Trade With Caution"
            score = 50.0
            detail = f"FF calendar: {high_near} USD high-impact events this week"
        elif high_near >= 5:
            status = "Avoid Trading Today"
            score = 25.0
            block = True
            detail = f"FF calendar: dense USD high-impact week ({high_near})"
    except Exception:
        # Weekend / off session soft caution
        if now.weekday() >= 5:
            status = "Avoid Trading Today"
            score = 20.0
            block = True
            detail = "weekend — market closed / thin"
        elif now.weekday() == 4 and now.hour >= 18:
            status = "Trade With Caution"
            score = 45.0
            detail = "Friday late liquidity caution"

    mod = ModuleScore("news", score, 4, Bias.NEUTRAL, detail, not block and score >= 40)
    return NewsReport(status, score, score, detail, mod, block)
