"""
Order block detection — last opposing candle before a impulsive displacement.
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from atlas.models import Direction, OrderBlock


def detect_order_blocks(
    df: pd.DataFrame,
    lookback: int = 40,
    impulse_atr_mult: float = 1.2,
    atr_value: float | None = None,
) -> list[OrderBlock]:
    """
    Bullish OB: last down/bearish candle before a strong bullish impulse.
    Bearish OB: last up/bullish candle before a strong bearish impulse.

    Impulse measured as body (or range) exceeding ATR × multiplier.
    """
    blocks: list[OrderBlock] = []
    n = len(df)
    if n < 5:
        return blocks

    atr = atr_value or _fallback_atr(df)
    threshold = atr * impulse_atr_mult
    start = max(2, n - lookback)

    for i in range(start, n):
        open_i = float(df["open"].iloc[i])
        close_i = float(df["close"].iloc[i])
        high_i = float(df["high"].iloc[i])
        low_i = float(df["low"].iloc[i])
        body = abs(close_i - open_i)
        rng = high_i - low_i
        move = max(body, rng * 0.7)

        # Strong bullish impulse → look back for last bearish candle
        if close_i > open_i and move >= threshold:
            for j in range(i - 1, max(start - 5, i - 8), -1):
                o = float(df["open"].iloc[j])
                c = float(df["close"].iloc[j])
                if c < o:
                    # Prefer candle body as the order-block zone
                    top = float(max(o, c))
                    bottom = float(min(o, c))
                    mitigated = _mitigated(df, j + 1, bottom, top, Direction.LONG)
                    blocks.append(
                        OrderBlock(
                            direction=Direction.LONG,
                            top=top,
                            bottom=bottom,
                            index=j,
                            time=_bar_time(df, j),
                            mitigated=mitigated,
                        )
                    )
                    break

        # Strong bearish impulse → last bullish candle
        if close_i < open_i and move >= threshold:
            for j in range(i - 1, max(start - 5, i - 8), -1):
                o = float(df["open"].iloc[j])
                c = float(df["close"].iloc[j])
                if c > o:
                    top = float(df["close"].iloc[j])
                    bottom = float(df["open"].iloc[j])
                    if top < bottom:
                        top, bottom = bottom, top
                    mitigated = _mitigated(df, j + 1, bottom, top, Direction.SHORT)
                    blocks.append(
                        OrderBlock(
                            direction=Direction.SHORT,
                            top=top,
                            bottom=bottom,
                            index=j,
                            time=_bar_time(df, j),
                            mitigated=mitigated,
                        )
                    )
                    break

    return _dedupe_nearest(blocks)


def active_order_blocks(
    blocks: list[OrderBlock],
    direction: Direction,
) -> list[OrderBlock]:
    return [b for b in blocks if b.direction == direction and not b.mitigated]


def price_in_order_block(price: float, block: OrderBlock, buffer: float = 0.0) -> bool:
    return (block.bottom - buffer) <= price <= (block.top + buffer)


def _mitigated(
    df: pd.DataFrame,
    from_index: int,
    bottom: float,
    top: float,
    direction: Direction,
) -> bool:
    """OB mitigated when price closes through the far side of the zone."""
    for j in range(from_index, len(df)):
        close = float(df["close"].iloc[j])
        if direction == Direction.LONG and close < bottom:
            return True
        if direction == Direction.SHORT and close > top:
            return True
    return False


def _dedupe_nearest(blocks: list[OrderBlock]) -> list[OrderBlock]:
    """Keep latest unique blocks per direction (by index)."""
    seen: dict[tuple[str, int], OrderBlock] = {}
    for b in blocks:
        key = (b.direction.value, b.index)
        seen[key] = b
    return sorted(seen.values(), key=lambda x: x.index)


def _fallback_atr(df: pd.DataFrame, period: int = 14) -> float:
    window = df.tail(period)
    return float((window["high"] - window["low"]).mean())


def _bar_time(df: pd.DataFrame, index: int) -> datetime | None:
    if "time" not in df.columns:
        return None
    val = df["time"].iloc[index]
    if pd.isna(val):
        return None
    return pd.Timestamp(val).to_pydatetime()
