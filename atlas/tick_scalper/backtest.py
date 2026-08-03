"""
Tick-history backtester for the XAUUSD scalper.

Uses the SAME TickStrategy + TickRiskManager rules as live/paper.
Replays historical MT5 ticks (or synthetic ticks offline).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List

from atlas.tick_scalper.config import TickScalperConfig, load_tick_config
from atlas.tick_scalper.mt5_feed import Tick
from atlas.tick_scalper.strategy import OpenState, Side, TickStrategy


@dataclass
class BacktestTrade:
    side: str
    entry: float
    exit: float
    sl: float
    tp: float
    volume: float
    pnl_points: float
    reason: str


@dataclass
class BacktestResult:
    trades: List[BacktestTrade] = field(default_factory=list)
    ticks_processed: int = 0
    signals: int = 0

    @property
    def total_pnl_points(self) -> float:
        return sum(t.pnl_points for t in self.trades)

    @property
    def win_rate(self) -> float:
        if not self.trades:
            return 0.0
        wins = sum(1 for t in self.trades if t.pnl_points > 0)
        return round(100.0 * wins / len(self.trades), 2)


class TickBacktester:
    """
    Replay ticks through strategy. No martingale / grid / averaging.
    Position size fixed at 0.01 for apples-to-apples point PnL.
    """

    def __init__(self, cfg: TickScalperConfig | None = None, volume: float = 0.01) -> None:
        self.cfg = cfg or load_tick_config()
        self.volume = volume
        self.strategy = TickStrategy(self.cfg)
        self.point = self.cfg.point_size

    def run(self, ticks: Iterable[Tick]) -> BacktestResult:
        result = BacktestResult()
        pos: OpenState | None = None
        ticket = 1

        for tick in ticks:
            result.ticks_processed += 1
            self.strategy.update(tick, self.point)

            if pos is not None:
                reason = self.strategy.evaluate_exit(tick, pos, self.point)
                if reason is None:
                    continue
                price = tick.bid if pos.side == Side.BUY else tick.ask
                if pos.side == Side.BUY:
                    pnl = (price - pos.entry) / self.point
                else:
                    pnl = (pos.entry - price) / self.point
                result.trades.append(
                    BacktestTrade(
                        side=pos.side.value,
                        entry=pos.entry,
                        exit=price,
                        sl=pos.sl,
                        tp=pos.tp,
                        volume=pos.volume,
                        pnl_points=pnl,
                        reason=reason,
                    )
                )
                pos = None
                continue

            signal = self.strategy.evaluate_entry(tick, self.point)
            if signal is None:
                continue
            result.signals += 1
            entry = signal.ask if signal.side == Side.BUY else signal.bid
            sl, tp = self.strategy.levels_for(signal.side, entry, self.point)
            ticket += 1
            pos = OpenState(
                side=signal.side,
                entry=entry,
                sl=sl,
                tp=tp,
                volume=self.volume,
                ticket=ticket,
                vwap_at_entry=self.strategy.vwap,
            )

        return result


def generate_synthetic_ticks(
    n: int = 5000,
    start_price: float = 4050.0,
    seed: int = 42,
) -> List[Tick]:
    """Offline tape for plumbing / unit tests (NOT a real market)."""
    import math
    import random

    rng = random.Random(seed)
    ticks: List[Tick] = []
    price = start_price
    t0 = 1_700_000_000_000
    for i in range(n):
        # Mild trending + noise so momentum filters can fire
        drift = 0.02 * math.sin(i / 40.0) + rng.uniform(-0.08, 0.08)
        price += drift
        spread = 0.20 + rng.uniform(0.0, 0.15)
        ticks.append(
            Tick(
                time_msc=t0 + i * 200,
                bid=price,
                ask=price + spread,
                last=price,
                volume=1,
            )
        )
    return ticks
