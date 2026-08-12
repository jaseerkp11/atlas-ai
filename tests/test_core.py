"""Unit tests for structure, scoring gates, sizing, and pending-fill backtest semantics."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from atlas.analysis.fvg import detect_fair_value_gaps
from atlas.analysis.liquidity import detect_liquidity_sweep
from atlas.analysis.order_blocks import detect_order_blocks
from atlas.analysis.structure import detect_bos, find_swing_highs, find_swing_lows, infer_trend_from_swings
from atlas.analysis.volatility import atr, latest_atr
from atlas.config import get_gates, load_settings
from atlas.execution.mt5_client import MT5Client, generate_synthetic_ohlc
from atlas.models import Direction, SetupFeatures, TrendStrength
from atlas.risk.position import calculate_position_size
from atlas.scoring.engine import evaluate_setup, score_setup


@pytest.fixture(autouse=True)
def _reload_settings():
    load_settings(reload=True)
    yield


def _trending_df(n: int = 120, upward: bool = True) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    times = pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC")
    drift = 0.0003 if upward else -0.0003
    noise = rng.normal(0, 0.0004, n)
    close = 1.10 + np.cumsum(np.full(n, drift) + noise)
    open_ = np.roll(close, 1)
    open_[0] = 1.10
    high = np.maximum(open_, close) + 0.0003
    low = np.minimum(open_, close) - 0.0003
    return pd.DataFrame(
        {"time": times, "open": open_, "high": high, "low": low, "close": close, "volume": 100}
    )


def test_weights_sum_to_100():
    s = load_settings()
    assert s.weights.total() == 100


def test_gates_single_source():
    gates = get_gates()
    ok, fails = gates.allows(gates.min_score, gates.min_reward_risk)
    assert ok and not fails
    ok2, fails2 = gates.allows(gates.min_score - 1, gates.min_reward_risk)
    assert not ok2 and fails2
    ok3, fails3 = gates.allows(gates.min_score, gates.min_reward_risk - 0.1)
    assert not ok3 and fails3


def test_swing_and_trend_detection():
    times = pd.date_range("2024-01-01", periods=40, freq="15min", tz="UTC")
    high = np.array(
        [
            1.0, 1.1, 1.2, 1.3, 1.5, 1.4, 1.3, 1.2, 1.1, 1.0,
            1.1, 1.2, 1.3, 1.4, 1.6, 1.5, 1.4, 1.3, 1.2, 1.1,
            1.2, 1.3, 1.4, 1.5, 1.7, 1.6, 1.5, 1.4, 1.3, 1.2,
            1.3, 1.4, 1.5, 1.6, 1.8, 1.7, 1.6, 1.5, 1.4, 1.3,
        ],
        dtype=float,
    )
    low = high - 0.15
    close = (high + low) / 2
    df = pd.DataFrame(
        {
            "time": times,
            "open": close,
            "high": high,
            "low": low,
            "close": close,
            "volume": 100,
        }
    )
    sh = find_swing_highs(df, lookback=2)
    sl = find_swing_lows(df, lookback=2)
    assert len(sh) >= 1 and len(sl) >= 1
    direction, strength, conf = infer_trend_from_swings(sh, sl)
    assert direction == Direction.LONG
    assert 0 <= conf <= 1
    assert strength != TrendStrength.NONE


def test_bos_on_break():
    df = _trending_df(upward=True)
    sh = find_swing_highs(df, lookback=3)
    sl = find_swing_lows(df, lookback=3)
    # Force a clear break above last swing high
    if sh:
        df = df.copy()
        df.loc[df.index[-1], "close"] = sh[-1].price + 0.01
        df.loc[df.index[-1], "high"] = sh[-1].price + 0.012
        ok, side = detect_bos(df, sh, sl, direction_bias=Direction.LONG)
        assert ok is True
        assert side == "bullish"


def test_fvg_and_ob_and_sweep_run():
    df = generate_synthetic_ohlc("EURUSD", "M15", 200, seed=7)
    atr_v = latest_atr(df)
    gaps = detect_fair_value_gaps(df, atr_value=atr_v)
    obs = detect_order_blocks(df, atr_value=atr_v)
    sh = find_swing_highs(df, 3)
    sl = find_swing_lows(df, 3)
    sweep = detect_liquidity_sweep(df, sh, sl)
    assert isinstance(gaps, list)
    assert isinstance(obs, list)
    # sweep may or may not exist — just ensure no crash
    assert sweep is None or sweep.direction in (Direction.LONG, Direction.SHORT)


def test_atr_positive():
    df = _trending_df()
    series = atr(df, 14)
    assert float(series.dropna().iloc[-1]) > 0


def test_position_size_uses_risk_formula():
    result = calculate_position_size(
        balance=10_000,
        entry=1.1000,
        stop=1.0950,
        symbol="EURUSD",
        tick_size=0.00001,
        tick_value=1.0,
        contract_size=100_000,
        risk_percent=1.0,
    )
    assert result.valid
    assert result.volume > 0
    # risk $100; stop 50 pips = 0.005; loss/lot = 500 → ~0.2 lots
    assert abs(result.volume - 0.2) < 0.011
    assert "100.00" in result.formula


def test_position_size_rejects_zero_stop():
    result = calculate_position_size(10_000, 1.1, 1.1, "EURUSD")
    assert not result.valid


def test_score_reasoning_always_present():
    features = SetupFeatures(
        symbol="EURUSD",
        direction=Direction.LONG,
        bos=True,
        choch=False,
        m5_trigger=True,
        atr=0.001,
        volatility_expanding=True,
        entry=1.1000,
        stop=1.0950,
        target=1.1150,
        reward_risk=3.0,
    )
    score = score_setup(features)
    assert score.total >= 0
    assert score.plain_language
    assert len(score.factors) == 9
    decision = evaluate_setup(features)
    text = decision.reasoning_text()
    assert "Score:" in text
    assert "Passed:" in text or "Failed:" in text


def test_rejected_setup_still_has_reasoning():
    features = SetupFeatures(
        symbol="EURUSD",
        direction=Direction.LONG,
        entry=1.1,
        stop=1.099,
        target=1.101,  # poor RR
        reward_risk=0.5,
    )
    decision = evaluate_setup(features)
    assert decision.allowed is False
    assert decision.gate_failures
    assert "NO TRADE" in decision.reasoning_text()


def test_pending_fill_not_instant():
    """Paper/backtest: pending must wait until price reaches entry."""
    from atlas.execution.engine import ExecutionEngine
    from atlas.journal.trade_journal import TradeJournal
    from atlas.models import ScoreResult, TradeDecision
    from atlas.scoring.engine import score_setup

    client = MT5Client()
    client.connect()
    client.update_paper_balance(10_000)
    journal = TradeJournal(path="/tmp/atlas_test_journal.csv")
    engine = ExecutionEngine(broker=client, journal=journal, on_log=lambda m: None)

    features = SetupFeatures(
        symbol="EURUSD",
        direction=Direction.LONG,
        h4=None,
        h1=None,
        bos=True,
        choch=True,
        m5_trigger=True,
        atr=0.0012,
        volatility_expanding=True,
        entry=1.1050,  # above market
        stop=1.1000,
        target=1.1200,
        reward_risk=3.0,
    )
    # Build a high-scoring decision by mocking allowed
    decision = evaluate_setup(features)
    # Force allowed path by lowering gates check — use try_execute only if allowed
    # Ensure features produce enough score or patch allowed
    decision.allowed = True
    decision.gate_failures = []

    # Bypass validation soft fails by using paper and ensuring session — may block on session
    # Call pending path directly
    from atlas.risk.position import calculate_position_size

    sizing = calculate_position_size(10_000, features.entry, features.stop, "EURUSD", risk_percent=0.5)
    rec = engine._execute_paper_pending(decision, sizing, sizing.volume)
    assert rec.status.value == "PENDING"
    assert len(engine.pending) == 1

    # Price not at entry yet
    engine.update_market("EURUSD", bar_high=1.1040, bar_low=1.1000, bar_close=1.1030)
    assert len(engine.pending) == 1
    assert len(engine.open_trades) == 0

    # Price reaches entry
    engine.update_market("EURUSD", bar_high=1.1060, bar_low=1.1040, bar_close=1.1055)
    assert len(engine.pending) == 0
    assert len(engine.open_trades) == 1
    assert engine.open_trades[0].status.value == "OPEN"


def test_backtest_runs_and_reports_limitations():
    from atlas.backtest.runner import BACKTEST_LIMITATIONS, BacktestRunner

    runner = BacktestRunner(on_log=lambda m: None)
    data = {
        "EURUSD": generate_synthetic_ohlc("EURUSD", "M5", 800, seed=1),
        "GBPUSD": generate_synthetic_ohlc("GBPUSD", "M5", 800, seed=2),
    }
    stats = runner.run(symbols=["EURUSD", "GBPUSD"], bars=800, data=data)
    assert stats.limitations == BACKTEST_LIMITATIONS
    assert stats.total_trades >= 0
    assert stats.starting_balance > 0


def test_sr_map_builds_from_frames():
    from atlas.analysis.levels import build_sr_map
    from atlas.backtest.runner import _resample_from_m5

    m5 = generate_synthetic_ohlc("EURUSD", "M5", 600, seed=3)
    frames = {
        "M5": m5,
        "M15": _resample_from_m5(m5, "15min"),
        "H1": _resample_from_m5(m5, "1h"),
        "H4": _resample_from_m5(m5, "4h"),
    }
    mid = float(m5["close"].iloc[-1])
    sr = build_sr_map("EURUSD", frames, mid=mid)
    assert sr.mid == mid
    lines = sr.chart_lines()
    assert any("SUP" in x or "RES" in x or "S/R" in x for x in lines)


def test_watch_never_emits_on_startup_prev_none():
    from atlas.live.watch import should_emit_on_closed_m5, last_closed_m5_key

    # The bug: first observation must NOT count as a new close
    assert should_emit_on_closed_m5(None, "2024-01-01 12:00") is False
    assert should_emit_on_closed_m5("2024-01-01 12:00", "2024-01-01 12:00") is False
    assert should_emit_on_closed_m5("2024-01-01 12:00", "2024-01-01 12:05") is True

    times = pd.date_range("2024-01-01", periods=10, freq="5min", tz="UTC")
    df = pd.DataFrame({"time": times, "open": 1.0, "high": 1.1, "low": 0.9, "close": 1.0})
    # last closed is second-to-last, not forming last
    assert last_closed_m5_key(df) == str(pd.Timestamp(times[-2]))


def test_vwap_computes_and_aligns():
    from atlas.analysis.vwap import compute_vwap_series, evaluate_vwap
    from atlas.models import Direction

    n = 80
    times = pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC")
    close = np.linspace(1.10, 1.12, n)
    df = pd.DataFrame(
        {
            "time": times,
            "open": close,
            "high": close + 0.0003,
            "low": close - 0.0003,
            "close": close,
            "volume": np.full(n, 1000),
        }
    )
    series = compute_vwap_series(df)
    assert len(series) == n
    assert float(series.iloc[-1]) > 0
    reading = evaluate_vwap(df, Direction.LONG, atr=0.001)
    assert reading.value > 0
    assert "VWAP=" in reading.summary
