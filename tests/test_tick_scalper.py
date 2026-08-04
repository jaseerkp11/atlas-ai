"""Tests for M1 price-action burst tick scalper."""

from __future__ import annotations

import pytest

from atlas.tick_scalper.config import load_tick_config
from atlas.tick_scalper.mt5_feed import Tick
from atlas.tick_scalper.strategy import OpenState, Side, TickStrategy


@pytest.fixture(autouse=True)
def _paper_mode(monkeypatch):
    monkeypatch.setenv("ATLAS_MODE", "PAPER")
    load_tick_config(reload=True)
    yield
    load_tick_config(reload=True)


def test_config_m1_burst():
    cfg = load_tick_config(reload=True)
    assert cfg.symbol == "XAUUSD"
    assert cfg.max_open_positions == 5
    assert cfg.burst_fill is True
    assert cfg.use_m1_price_action is True
    assert cfg.fixed_lots == 1.0
    assert cfg.instant_profit_points == 1.5
    assert cfg.alternate_batch_side is True
    assert cfg.require_flat_before_next_batch is True
    assert cfg.require_micro_bar_close is False
    assert cfg.vwap_enabled is False


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


def test_entry_none_without_m1_bias():
    cfg = load_tick_config(reload=True)
    strat = TickStrategy(cfg)
    strat.set_mt5_live(False)
    t = Tick(1, 4050.0, 4050.3, 4050.0, 1)
    strat.update(t, cfg.point_size)
    assert strat.evaluate_entry(t, cfg.point_size) is None


def test_entry_uses_m1_bias(monkeypatch):
    from atlas.tick_scalper import strategy as stmod
    from atlas.tick_scalper.m1_price_action import M1Bias

    cfg = load_tick_config(reload=True)
    strat = TickStrategy(cfg)
    strat.set_mt5_live(True)
    strat._bias = M1Bias(side="BUY", reason="test HH+HL", bar_time=123)
    t = Tick(1, 4050.0, 4050.2, 4050.0, 1)  # spread 20 pts
    sig = strat.evaluate_entry(t, cfg.point_size)
    assert sig is not None
    assert sig.side == Side.BUY
    assert "M1 PA" in sig.reason


def test_instant_profit_exit():
    cfg = load_tick_config(reload=True)
    strat = TickStrategy(cfg)
    point = cfg.point_size
    entry = 4050.0
    target = 1.5
    sl, tp = strat.levels_for(Side.BUY, entry, point, spread_points=20)
    pos = OpenState(
        side=Side.BUY, entry=entry, sl=sl, tp=tp, volume=1.0, ticket=1, profit_target_points=target
    )
    bid = entry + (target + 0.2) * point
    tick = Tick(2, bid, bid + 0.2, bid, 1)
    assert strat.evaluate_exit(tick, pos, point) == "instant_profit"


def test_alternate_side_logic():
    from atlas.tick_scalper.engine import TickEngine

    eng = TickEngine(load_tick_config(reload=True))
    assert eng._required_side() is None
    eng._last_batch_side = Side.BUY
    assert eng._required_side() == Side.SELL
    eng._last_batch_side = Side.SELL
    assert eng._required_side() == Side.BUY


def test_hard_sl_exit():
    cfg = load_tick_config(reload=True)
    strat = TickStrategy(cfg)
    point = cfg.point_size
    entry = 4050.0
    sl, tp = strat.levels_for(Side.SELL, entry, point)
    pos = OpenState(side=Side.SELL, entry=entry, sl=sl, tp=tp, volume=1.0, ticket=2)
    hit = Tick(3, sl - 0.05, sl + 0.01, sl, 1)
    assert strat.evaluate_exit(hit, pos, point) == "stop_loss"


def test_burst_opens_five_paper():
    from atlas.tick_scalper.execution import TickExecutor
    from atlas.tick_scalper.logger import TickLogger
    from atlas.tick_scalper.mt5_feed import MT5TickFeed
    from atlas.tick_scalper.strategy import Signal

    cfg = load_tick_config(reload=True)
    feed = MT5TickFeed(cfg)
    feed.connect()
    feed._use_mt5 = False
    ex = TickExecutor(feed, TickLogger(), cfg)
    sig = Signal(side=Side.BUY, reason="M1", bid=4050.0, ask=4050.2, m1_bar_time=1)
    for i in range(5):
        r = ex.open_market(sig, 1.0, 4040.0, 4060.0, profit_target_points=5.0)
        assert r.ok
    assert len(ex.current_positions()) == 5
    feed.disconnect()


def test_cli_tick_help():
    from main import build_parser

    p = build_parser()
    args = p.parse_args(["tick", "--max-ticks", "5"])
    assert args.command == "tick"


def test_m1_swings_helper():
    from atlas.tick_scalper.m1_price_action import _swings

    highs = [1, 2, 5, 3, 2, 6, 4, 3]
    lows = [0.5, 1, 2, 1.5, 1, 3, 2, 1]
    sh, sl = _swings(highs, lows, left=2)
    assert isinstance(sh, list)


def test_engine_smoke_synthetic(monkeypatch):
    from atlas.tick_scalper.engine import TickEngine

    cfg = load_tick_config(reload=True)
    eng = TickEngine(cfg)
    monkeypatch.setattr(eng.feed, "connect", lambda: True)
    eng.feed._connected = True
    eng.feed._use_mt5 = False
    stats = eng.start(max_ticks=30)
    assert stats.ticks == 30
