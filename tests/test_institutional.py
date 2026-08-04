"""Tests for Institutional Market Analysis Engine."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from atlas.institutional.config import load_institutional_config
from atlas.institutional.confluence_engine import compute_confluence
from atlas.institutional.market_structure import analyze_structure
from atlas.institutional.models import Bias, DecisionAction, ModuleScore
from atlas.institutional.price_action import analyze_price_action, analyze_session


def _ohlc(n: int = 250, start: float = 4000.0, drift: float = 0.15) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    rows = []
    price = start
    t0 = np.datetime64("2024-01-01T00:00:00")
    for i in range(n):
        price += drift + float(rng.normal(0, 0.4))
        o = price
        h = price + abs(float(rng.normal(0.5, 0.2)))
        l = price - abs(float(rng.normal(0.5, 0.2)))
        c = price + float(rng.normal(0, 0.2))
        rows.append(
            {
                "time": t0 + np.timedelta64(i, "h"),
                "open": o,
                "high": max(o, h, c),
                "low": min(o, l, c),
                "close": c,
                "tick_volume": 100,
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture(autouse=True)
def _paper(monkeypatch):
    monkeypatch.setenv("ATLAS_MODE", "PAPER")
    load_institutional_config(reload=True)
    yield
    load_institutional_config(reload=True)


def test_config_gates_quality_first():
    cfg = load_institutional_config(reload=True)
    assert cfg.symbol == "XAUUSD"
    assert cfg.min_probability >= 70
    assert cfg.min_confluence >= 65
    assert cfg.require_htf_alignment is True
    assert cfg.require_playbook is True
    assert "H4" in cfg.timeframes and "M1" in cfg.timeframes


def test_structure_on_trending_series():
    df = _ohlc(200, drift=0.4)
    rep = analyze_structure(df, lookback=3)
    assert rep.module.name == "market_structure"
    assert 0 <= rep.direction_score <= 100


def test_price_action_detects_patterns():
    df = _ohlc(50)
    # Force bullish pin on last closed
    i = len(df) - 2
    df.loc[i, "open"] = 4050.0
    df.loc[i, "close"] = 4050.2
    df.loc[i, "high"] = 4050.3
    df.loc[i, "low"] = 4048.0
    rep = analyze_price_action(df)
    assert isinstance(rep.patterns, list)


def test_session_engine():
    rep = analyze_session()
    assert rep.best_session
    assert 0 <= rep.score <= 100


def test_confluence_prefers_agreement():
    mods = [
        ModuleScore("trend", 80, 12, Bias.BULLISH, "t", True),
        ModuleScore("market_structure", 75, 14, Bias.BULLISH, "s", True),
        ModuleScore("liquidity", 70, 10, Bias.BULLISH, "l", True),
        ModuleScore("news", 60, 4, Bias.NEUTRAL, "n", True),
    ]
    cfg = load_institutional_config(reload=True)
    c = compute_confluence(mods, cfg.weights)
    assert c.consensus_bias == Bias.BULLISH
    assert c.probability > 50


def test_playbook_boost_raises_probability():
    from atlas.institutional.models import MarketNarrative, utcnow
    from atlas.institutional.playbook_engine import PlaybookHit, PlaybookResult

    mods = [
        ModuleScore("trend", 70, 12, Bias.BULLISH, "t", True),
        ModuleScore("market_structure", 65, 14, Bias.BULLISH, "BOS=bullish discount", True),
        ModuleScore("liquidity", 55, 10, Bias.NEUTRAL, "sweep=none", False),
        ModuleScore("price_action", 70, 8, Bias.BULLISH, "bullish_pin", True),
        ModuleScore("session", 80, 4, Bias.NEUTRAL, "London", True),
    ]
    pb = PlaybookResult(
        hits=[PlaybookHit("Killzone PA (London)", Bias.BULLISH, 10.0, ["x"])],
        best=PlaybookHit("Killzone PA (London)", Bias.BULLISH, 10.0, ["x"]),
        total_boost=10.0,
        summary="playbooks=1",
        unlock_hints=[],
    )
    cfg = load_institutional_config(reload=True)
    base = compute_confluence(mods, cfg.weights, h1_bias=Bias.BULLISH)
    boosted = compute_confluence(mods, cfg.weights, playbooks=pb, h1_bias=Bias.BULLISH)
    assert boosted.probability >= base.probability
    assert boosted.playbook_name != ""


def test_analyzer_no_trade_without_data_quality():
    from atlas.institutional.analyzer import InstitutionalAnalyzer
    from atlas.institutional.ai_decision_engine import decide
    from atlas.institutional.confluence_engine import ConfluenceResult
    from atlas.institutional.models import MarketNarrative, utcnow
    from atlas.institutional.price_action import PriceActionReport

    cfg = load_institutional_config(reload=True)
    narrative = MarketNarrative(
        symbol="XAUUSD",
        as_of=utcnow(),
        overall_bias=Bias.NEUTRAL,
        htf_bias=Bias.NEUTRAL,
        mtf_bias=Bias.NEUTRAL,
        structure_summary="n/a",
        liquidity_summary="n/a",
        institutional_confluence="n/a",
        sr_summary="n/a",
        trend_quality="Ranging",
        volatility_regime="low",
        best_session="Asian",
        news_status="Trade Today",
        mid=4050.0,
        atr_m15=5.0,
    )
    conf = ConfluenceResult(40, 30, 30, Bias.NEUTRAL, 0, 0, [], "low")
    pa = PriceActionReport([], 40, Bias.NEUTRAL, "none", ModuleScore("price_action", 40, 8, Bias.NEUTRAL, "", False))
    d = decide(cfg, narrative, conf, None, None, pa, False, True, False, False, None)
    assert d.action in (DecisionAction.NO_TRADE, DecisionAction.WAIT)


def test_cli_analyze_registered():
    from main import build_parser

    p = build_parser()
    args = p.parse_args(["analyze", "--symbol", "XAUUSD"])
    assert args.command == "analyze"
    args2 = p.parse_args(["institutional", "--cycles", "1", "--poll", "1"])
    assert args2.command == "institutional"


def test_dashboard_renders():
    from atlas.institutional.dashboard import render_dashboard
    from atlas.institutional.models import InstitutionalDecision, MarketNarrative, RiskLevel, utcnow

    n = MarketNarrative(
        symbol="XAUUSD",
        as_of=utcnow(),
        overall_bias=Bias.BULLISH,
        htf_bias=Bias.NEUTRAL,
        mtf_bias=Bias.BULLISH,
        structure_summary="test",
        liquidity_summary="test",
        institutional_confluence="test",
        sr_summary="test",
        trend_quality="Transitional",
        volatility_regime="ATR=1",
        best_session="London",
        news_status="ok",
        mid=4050.0,
        atr_m15=5.0,
        extras={
            "h1_bias": "BULLISH",
            "h4_bias": "NEUTRAL",
            "m15_bias": "BULLISH",
            "playbook": "playbooks=0",
            "playbook_hits": [],
            "scenario": {
                "primary": "LONG bias wait",
                "alternate": "flip",
                "invalidation": "H1 flip",
                "edge_score": 44,
                "summary": "edge=44",
                "checklist": [{"name": "H1", "ok": True, "detail": "BULLISH"}],
                "next_triggers": ["Wait for OTE"],
            },
            "trade_areas": {
                "summary": "areas=1 buy=1 sell=0",
                "areas": [
                    {
                        "side": "BUY",
                        "kind": "bullish_fvg",
                        "price_low": 4048.0,
                        "price_high": 4049.5,
                        "mid_price": 4048.75,
                        "score": 82.0,
                        "distance_atr": 0.4,
                        "fresh_unfilled": True,
                        "reasons": ["Unfilled bullish FVG"],
                        "label": "BullFVG",
                    }
                ],
                "buy_best": [
                    {
                        "side": "BUY",
                        "kind": "bullish_fvg",
                        "price_low": 4048.0,
                        "price_high": 4049.5,
                        "mid_price": 4048.75,
                        "score": 82.0,
                        "distance_atr": 0.4,
                        "fresh_unfilled": True,
                        "reasons": ["Unfilled bullish FVG"],
                        "label": "BullFVG",
                    }
                ],
                "sell_best": [],
            },
        },
    )
    d = InstitutionalDecision(
        action=DecisionAction.NO_TRADE,
        why=["test"],
        confidence=40,
        probability=40,
        confluence=40,
        risk_level=RiskLevel.HIGH,
        narrative=n,
    )
    text = render_dashboard(d)
    assert "DECISION: NO_TRADE" in text
    assert "INSTITUTIONAL" in text
    assert "AI SCENARIO PLAN" in text
    assert "HIGH-PROBABILITY PLAYBOOKS" in text
    assert "BEST TRADE AREAS" in text
    assert "BUY-SIDE AREAS" in text
    assert text.index("DECISION: NO_TRADE") < text.index("BEST TRADE AREAS")


def test_discount_array_playbook():
    from atlas.institutional.models import MarketNarrative, utcnow
    from atlas.institutional.playbook_engine import evaluate_playbooks

    n = MarketNarrative(
        symbol="XAUUSD",
        as_of=utcnow(),
        overall_bias=Bias.BULLISH,
        htf_bias=Bias.BULLISH,
        mtf_bias=Bias.BULLISH,
        structure_summary="discount",
        liquidity_summary="x",
        institutional_confluence="x",
        sr_summary="x",
        trend_quality="Trending",
        volatility_regime="ok",
        best_session="London",
        news_status="ok",
        mid=4050.0,
        atr_m15=5.0,
    )
    mods = [
        ModuleScore("market_structure", 70, 14, Bias.BULLISH, "trend=LONG/MODERATE | discount", True),
        ModuleScore("liquidity", 40, 10, Bias.NEUTRAL, "sweep=none", False),
        ModuleScore("price_action", 50, 8, Bias.NEUTRAL, "none", False),
        ModuleScore("session", 80, 4, Bias.NEUTRAL, "London", True),
    ]
    pb = evaluate_playbooks(
        n, mods, Bias.BULLISH, "London", Bias.BULLISH, m15_bias=Bias.BULLISH
    )
    names = [h.name for h in pb.hits]
    assert any("Discount Array" in x or "Structure Stack" in x for x in names)
    assert pb.total_boost >= 9


def test_trade_areas_rank_fvg_and_liquidity():
    from atlas.institutional.liquidity_engine import LiquidityPool, LiquidityReport
    from atlas.institutional.models import (
        MarketNarrative,
        Zone,
        ZoneStrength,
        utcnow,
    )
    from atlas.institutional.trade_areas import build_trade_areas

    n = MarketNarrative(
        symbol="XAUUSD",
        as_of=utcnow(),
        overall_bias=Bias.BULLISH,
        htf_bias=Bias.BULLISH,
        mtf_bias=Bias.BULLISH,
        structure_summary="x",
        liquidity_summary="x",
        institutional_confluence="x",
        sr_summary="x",
        trend_quality="t",
        volatility_regime="v",
        best_session="London",
        news_status="ok",
        mid=4050.0,
        atr_m15=5.0,
        zones=[
            Zone(
                "bullish_fvg",
                4049.0,
                4047.5,
                ZoneStrength.STRONG,
                85,
                "M15",
                fresh=True,
                label="BullFVG",
            ),
            Zone(
                "bearish_ob",
                4058.0,
                4056.0,
                ZoneStrength.STRONG,
                80,
                "M15",
                fresh=True,
                label="BearOB",
            ),
        ],
    )
    liq = LiquidityReport(
        pools=[
            LiquidityPool("equal_lows", 4045.0, 75, "EQL@4045"),
            LiquidityPool("equal_highs", 4060.0, 75, "EQH@4060"),
        ],
        sweep_bullish=False,
        sweep_bearish=False,
        score=50,
        bias=Bias.NEUTRAL,
        summary="test",
        module=ModuleScore("liquidity", 50, 10, Bias.NEUTRAL, "x", False),
    )
    rep = build_trade_areas(n, liquidity=liq, h1_bias=Bias.BULLISH)
    assert rep.buy_best
    assert any(a.kind in ("bullish_fvg", "buy_liquidity") for a in rep.buy_best)
    assert "buy=" in rep.summary

