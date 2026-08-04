"""
M1 price-action / market-structure reader.

No indicators (no VWAP, RSI, micro-TF). Only OHLC structure:
  - last closed M1 candle direction
  - swing HH/HL vs LH/LL
  - optional break of prior swing high/low (BOS-lite)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

try:
    import MetaTrader5 as mt5

    MT5_OK = True
except ImportError:
    mt5 = None  # type: ignore
    MT5_OK = False


@dataclass(frozen=True)
class M1Bias:
    side: str  # BUY | SELL
    reason: str
    bar_time: int  # closed bar time (unix)


def _swings(highs: list[float], lows: list[float], left: int = 2) -> tuple[list[float], list[float]]:
    """Simple fractal swings."""
    sh: list[float] = []
    sl: list[float] = []
    n = len(highs)
    for i in range(left, n - left):
        if highs[i] == max(highs[i - left : i + left + 1]):
            sh.append(highs[i])
        if lows[i] == min(lows[i - left : i + left + 1]):
            sl.append(lows[i])
    return sh, sl


def read_m1_price_action(
    symbol: str,
    use_mt5: bool,
    structure_bars: int = 12,
) -> Optional[M1Bias]:
    """
    Return directional bias from the last CLOSED M1 candle + structure.
    """
    if not use_mt5 or not MT5_OK or mt5 is None:
        return None

    need = max(20, structure_bars + 5)
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 0, need)
    if rates is None or len(rates) < structure_bars + 2:
        return None

    # Drop forming candle; work on closed only
    closed = rates[:-1]
    window = closed[-structure_bars:]
    last = window[-1]
    o = float(last["open"])
    h = float(last["high"])
    l = float(last["low"])
    c = float(last["close"])
    bar_time = int(last["time"])

    candle_buy = c > o
    candle_sell = c < o
    if not candle_buy and not candle_sell:
        return None  # doji — no trade

    highs = [float(r["high"]) for r in window]
    lows = [float(r["low"]) for r in window]
    closes = [float(r["close"]) for r in window]
    sh, sl = _swings(highs, lows, left=2)

    # Structure: higher highs + higher lows = bullish; opposite = bearish
    struct_buy = False
    struct_sell = False
    if len(sh) >= 2 and len(sl) >= 2:
        struct_buy = sh[-1] > sh[-2] and sl[-1] > sl[-2]
        struct_sell = sh[-1] < sh[-2] and sl[-1] < sl[-2]
    else:
        # Fallback: rising/falling closes
        struct_buy = closes[-1] > closes[-3] and closes[-2] > closes[-4]
        struct_sell = closes[-1] < closes[-3] and closes[-2] < closes[-4]

    # BOS-lite: close breaks prior swing
    bos_buy = len(sh) >= 2 and c > sh[-2]
    bos_sell = len(sl) >= 2 and c < sl[-2]

    reasons: list[str] = []
    side: str | None = None

    if candle_buy and (struct_buy or bos_buy):
        side = "BUY"
        reasons.append("M1 bullish candle")
        if struct_buy:
            reasons.append("HH+HL structure")
        if bos_buy:
            reasons.append("BOS up")
    elif candle_sell and (struct_sell or bos_sell):
        side = "SELL"
        reasons.append("M1 bearish candle")
        if struct_sell:
            reasons.append("LH+LL structure")
        if bos_sell:
            reasons.append("BOS down")
    elif candle_buy and c > max(closes[:-1]):
        # Strong engulf / break of recent closes
        side = "BUY"
        reasons.append("M1 bullish + break recent highs")
    elif candle_sell and c < min(closes[:-1]):
        side = "SELL"
        reasons.append("M1 bearish + break recent lows")
    else:
        # Soft: last candle only if body is meaningful vs range
        body = abs(c - o)
        rng = max(h - l, 1e-9)
        if body / rng >= 0.55:
            side = "BUY" if candle_buy else "SELL"
            reasons.append(f"M1 strong body candle ({body / rng:.0%})")
        else:
            return None

    assert side is not None
    reasons.append(f"O={o:.2f} C={c:.2f}")
    return M1Bias(side=side, reason=" | ".join(reasons), bar_time=bar_time)
