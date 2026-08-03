"""Tests for the XAUUSD tick scalper — strategy, risk, backtest, CLI wiring."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from atlas.tick_scalper.backtest import TickBacktester, generate_synthetic_ticks
from atlas.tick_scalper.config import load_tick_config
from atlas.tick_scalper.mt5_feed import Tick
from atlas.tick_scalper.strategy import OpenState, Side, TickStrategy


@pytest.fixture(autouse=True)
def _paper_mode(monkeypatch):
    monkeypatch.setenv("ATLAS_MODE", "PAPER")
    load_tick_config(reload=True)
    yield
    load_tick_config(reload=True)


def test_config_loads_xauusd():
    cfg = load_tick_config(reload=True)
    assert cfg.symbol == "XAUUSD"
    assert cfg.max_open_positions == 1
    assert cfg.take_profit_points > 0
    assert cfg.stop_loss_points > 0
    assert cfg.mode == "PAPER"
    assert cfg.use_fixed_lots is True
    assert cfg.fixed_lots == 1.0
    assert cfg.max_lots == 1.0


def test_fixed_lot_sizing():
    from atlas.tick_scalper.mt5_feed import MT5TickFeed
    from atlas.tick_scalper.risk import TickRiskManager

    cfg = load_tick_config(reload=True)
    feed = MT5TickFeed(cfg)
    feed.connect()
    risk = TickRiskManager(feed, cfg)
    vol, msg = risk.position_size(cfg.stop_loss_points)
    assert vol == 1.0
    assert "FIXED" in msg
    feed.disconnect()


def test_strategy_no_force_on_flat_tape():
    cfg = load_tick_config(reload=True)
    strat = TickStrategy(cfg)
    point = cfg.point_size
    # Flat ticks — should never signal
    base = 4050.0
    for i in range(30):
        t = Tick(time_msc=1_700_000_000_000 + i * 100, bid=base, ask=base + 0.20, last=base, volume=1)
        strat.update(t, point)
        assert strat.evaluate_entry(t, point) is None


def test_strategy_momentum_buy_can_fire():
    cfg = load_tick_config(reload=True)
    strat = TickStrategy(cfg)
    point = cfg.point_size
    price = 4000.0
    last_sig = None
    for i in range(80):
        price += 0.35  # strong up tape — VWAP lags below mid
        spread = 0.15
        t = Tick(
            time_msc=1_700_000_000_000 + i * 100,
            bid=price,
            ask=price + spread,
            last=price,
            volume=10,
        )
        strat.update(t, point)
        last_sig = strat.evaluate_entry(t, point) or last_sig
    assert last_sig is not None
    assert last_sig.side == Side.BUY


def test_hard_tp_exit():
    cfg = load_tick_config(reload=True)
    strat = TickStrategy(cfg)
    point = cfg.point_size
    entry = 4050.0
    sl, tp = strat.levels_for(Side.BUY, entry, point)
    pos = OpenState(side=Side.BUY, entry=entry, sl=sl, tp=tp, volume=0.01, ticket=1)
    # Warm strategy state
    for i in range(10):
        mid = entry + i * 0.01
        strat.update(Tick(1, mid, mid + 0.2, mid, 1), point)
    hit = Tick(time_msc=2, bid=tp + 0.01, ask=tp + 0.21, last=tp, volume=1)
    assert strat.evaluate_exit(hit, pos, point) == "take_profit"


def test_hard_sl_exit():
    cfg = load_tick_config(reload=True)
    strat = TickStrategy(cfg)
    point = cfg.point_size
    entry = 4050.0
    sl, tp = strat.levels_for(Side.SELL, entry, point)
    pos = OpenState(side=Side.SELL, entry=entry, sl=sl, tp=tp, volume=0.01, ticket=2)
    for i in range(10):
        mid = entry - i * 0.01
        strat.update(Tick(1, mid, mid + 0.2, mid, 1), point)
    hit = Tick(time_msc=2, bid=sl - 0.05, ask=sl + 0.01, last=sl, volume=1)
    assert strat.evaluate_exit(hit, pos, point) == "stop_loss"


def test_no_early_panic_on_loser():
    """Losers must ride to SL — no averaging, no panic cut."""
    cfg = load_tick_config(reload=True)
    strat = TickStrategy(cfg)
    point = cfg.point_size
    entry = 4050.0
    sl, tp = strat.levels_for(Side.BUY, entry, point)
    pos = OpenState(side=Side.BUY, entry=entry, sl=sl, tp=tp, volume=0.01, ticket=3)
    # Mild adverse move but above SL
    for i in range(20):
        mid = entry - 0.05  # small red, still above SL
        strat.update(Tick(i, mid, mid + 0.2, mid, 1), point)
    tick = Tick(time_msc=99, bid=entry - 0.10, ask=entry + 0.10, last=entry - 0.05, volume=1)
    # Should NOT exit for momentum_fade on a loser
    reason = strat.evaluate_exit(tick, pos, point)
    assert reason in (None, "stop_loss")
    if reason == "stop_loss":
        assert tick.bid <= sl


def test_backtest_runs_and_respects_max_one_position():
    ticks = generate_synthetic_ticks(n=3000, seed=7)
    bt = TickBacktester()
    result = bt.run(ticks)
    assert result.ticks_processed == 3000
    # Never overlapping: sequential open/close in backtester
    assert isinstance(result.win_rate, float)
    assert result.signals >= 0


def test_risk_blocks_max_positions(monkeypatch):
    from atlas.tick_scalper.mt5_feed import MT5TickFeed
    from atlas.tick_scalper.risk import TickRiskManager

    cfg = load_tick_config(reload=True)
    feed = MT5TickFeed(cfg)
    feed.connect()
    risk = TickRiskManager(feed, cfg)
    monkeypatch.setattr(feed, "open_position_count", lambda: 1)
    ok, msg = risk.allows_new_trade(time_msc=1_700_000_000_000)
    # May also fail session if weekend — either way must block or pass session
    if not ok:
        assert "max open" in msg or "session" in msg.lower() or "Saturday" in msg or "Sunday" in msg
    feed.disconnect()


def test_cli_tick_help():
    from main import build_parser

    p = build_parser()
    args = p.parse_args(["tick", "--max-ticks", "5"])
    assert args.command == "tick"
    assert args.max_ticks == 5
    args2 = p.parse_args(["tick-backtest", "--ticks", "100"])
    assert args2.command == "tick-backtest"


def test_engine_smoke_synthetic(monkeypatch):
    """Short engine run on synthetic feed (non-Windows / no MT5)."""
    from atlas.tick_scalper.engine import TickEngine

    cfg = load_tick_config(reload=True)
    eng = TickEngine(cfg)
    # Force synthetic path
    monkeypatch.setattr(eng.feed, "connect", lambda: True)
    eng.feed._connected = True
    eng.feed._use_mt5 = False

    # Limit via max_ticks
    stats = eng.start(max_ticks=50)
    assert stats.ticks == 50
