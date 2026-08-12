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
            "manual_stance": "LONG_BIAS",
            "scenario": {
                "primary": "LONG bias wait",
                "alternate": "flip",
                "invalidation": "H1 flip",
                "edge_score": 44,
                "summary": "edge=44",
                "checklist": [{"name": "H1", "ok": True, "detail": "BULLISH"}],
                "next_triggers": ["Wait for OTE"],
            },
            "manual_scan": {
                "bias": "BULLISH",
                "stance": "LONG_BIAS",
                "headline": "Prefer LONG ideas",
                "summary": "stance=LONG_BIAS",
                "watchlist": ["H1 bias = BULLISH"],
                "do_not": ["Do NOT auto-trade"],
                "cards": [],
                "buy_cards": [
                    {
                        "grade": "A",
                        "side": "BUY",
                        "zone_low": 4048.0,
                        "zone_high": 4049.5,
                        "focus_price": 4048.75,
                        "kind": "support",
                        "score": 90.0,
                        "status": "WAIT_FOR_TRIGGER",
                            "if_then": "WAIT: price dips INTO support | CONFIRM: M5 bullish pin | THEN: consider LONG | NOW: do NOTHING",
                        "confirm_on_tv": ["Mark support"],
                        "invalidation": "Below zone",
                        "avoid": "No chase",
                        "reasons": ["Strong support"],
                    }
                ],
                "sell_cards": [],
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
    assert "MANUAL HIGH-PROBABILITY MARKET SCANNER" in text or "MANUAL SETUPS" in text
    assert "WAIT:" in text
    assert "MODULE SCORES" in text
    assert "no auto" in text.lower() or "NO auto" in text


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


def test_manual_scanner_grades_and_if_then():
    from atlas.institutional.manual_scanner import build_manual_scan
    from atlas.institutional.models import MarketNarrative, utcnow
    from atlas.institutional.trade_areas import TradeArea, TradeAreasReport

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
    )
    areas = TradeAreasReport(
        areas=[],
        buy_best=[],
        sell_best=[],
        summary="x",
    )
    areas.areas = [
        TradeArea(
            "BUY",
            "buy_liquidity",
            4048,
            4050,
            4049,
            92,
            0.3,
            True,
            ["Buy-side liquidity"],
            "EQL",
        ),
        TradeArea(
            "SELL",
            "resistance",
            4055,
            4057,
            4056,
            90,
            1.0,
            True,
            ["Resistance"],
            "R",
        ),
    ]
    rep = build_manual_scan(
        n, areas, None, Bias.BULLISH, Bias.NEUTRAL, Bias.BULLISH, "London"
    )
    assert rep.stance == "LONG_BIAS"
    assert rep.buy_cards
    assert "WAIT:" in rep.buy_cards[0].if_then and "THEN:" in rep.buy_cards[0].if_then
    assert rep.buy_cards[0].grade in ("A+", "A", "B")
    assert rep.buy_cards[0].stop_loss > 0
    assert rep.buy_cards[0].stop_loss < rep.buy_cards[0].zone_low
    assert rep.buy_cards[0].take_profit_1 > rep.buy_cards[0].focus_price
    assert rep.buy_cards[0].take_profit_2 > rep.buy_cards[0].take_profit_1
    # Prop-style: TP1 ~1:2 and TP2 ~1:3 from planned entry
    assert rep.buy_cards[0].reward_risk >= 1.95
    assert rep.buy_cards[0].reward_risk_tp2 >= 2.95
    assert abs(rep.buy_cards[0].entry_price - rep.buy_cards[0].focus_price) < 1e-6
    assert any(c.side == "SELL" for c in rep.sell_cards)
    # Against-H1 sells stay B / AVOID when H1 bullish — intentional
    assert all(c.grade == "B" and c.status == "AVOID_NOW" for c in rep.sell_cards)
    sell = rep.sell_cards[0]
    assert sell.stop_loss > sell.zone_high
    assert sell.take_profit_1 < sell.focus_price
    assert sell.take_profit_2 <= sell.take_profit_1


def test_htf_major_h4_h1_highs_lows_marked():
    """Explicit H4/H1 range highs/lows become major S/R trade areas."""
    from atlas.institutional.models import MarketNarrative, utcnow
    from atlas.institutional.trade_areas import build_trade_areas

    # Synthetic impulse: grind up to ~4180 then pull back — mid at 4157
    def _tf(n: int, step_h: int, prices: list[float]) -> pd.DataFrame:
        rows = []
        t0 = np.datetime64("2024-06-01T00:00:00")
        for i, p in enumerate(prices):
            rows.append(
                {
                    "time": t0 + np.timedelta64(i * step_h, "h"),
                    "open": p - 0.5,
                    "high": p + 1.5,
                    "low": p - 1.5,
                    "close": p,
                    "tick_volume": 100,
                }
            )
        # Pad to enough bars if needed
        while len(rows) < n:
            i = len(rows)
            p = prices[-1]
            rows.append(
                {
                    "time": t0 + np.timedelta64(i * step_h, "h"),
                    "open": p - 0.3,
                    "high": p + 0.8,
                    "low": p - 0.8,
                    "close": p,
                    "tick_volume": 100,
                }
            )
        return pd.DataFrame(rows)

    # H4 path: start ~4070, spike 4180, settle ~4157
    h4_path = (
        [4070 + i * 4 for i in range(18)]
        + [4180, 4175, 4168, 4160, 4157]
    )
    h1_path = (
        [4070 + i * 1.5 for i in range(40)]
        + [4180, 4172, 4165, 4160, 4157]
    )
    frames = {
        "H4": _tf(40, 4, h4_path),
        "H1": _tf(60, 1, h1_path),
    }
    # Force exact major high onto last closed-ish window
    frames["H4"].loc[frames["H4"].index[-6], "high"] = 4180.0
    frames["H1"].loc[frames["H1"].index[-8], "high"] = 4180.0
    frames["H4"].loc[frames["H4"].index[2], "low"] = 4070.0

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
        mid=4157.0,
        atr_m15=8.0,
        zones=[],
    )
    rep = build_trade_areas(n, liquidity=None, h1_bias=Bias.BULLISH, frames=frames)
    htf = [a for a in rep.areas if a.kind.startswith("htf_")]
    assert htf, "expected explicit H4/H1 major areas"
    assert any(a.kind == "htf_resistance" and abs(a.mid_price - 4180.0) < 2.0 for a in htf)
    assert any(a.kind == "htf_support" for a in htf)


def test_a_plus_capped_when_rr_weak():
    """A+ soft-caps to A when mapped R:R to TP1 is below 1.5."""
    from atlas.institutional.manual_scanner import build_manual_scan
    from atlas.institutional.models import MarketNarrative, utcnow
    from atlas.institutional.trade_areas import TradeArea, TradeAreasReport

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
        mid=4157.0,
        atr_m15=8.0,
    )
    # Tight TP above entry → weak R:R; elite buy path otherwise
    buy = TradeArea(
        "BUY",
        "buy_liquidity",
        4154,
        4156,
        4155,
        92,
        0.25,
        True,
        ["Buy-side liquidity"],
        "EQL",
    )
    sell = TradeArea(
        "SELL",
        "htf_resistance",
        4157.2,
        4157.8,
        4157.5,
        95,
        0.06,
        True,
        ["H4 major high"],
        "H4 major high",
    )
    areas = TradeAreasReport(areas=[buy, sell], buy_best=[buy], sell_best=[sell], summary="x")
    rep = build_manual_scan(
        n, areas, None, Bias.BULLISH, Bias.BULLISH, Bias.BULLISH, "London"
    )
    assert rep.buy_cards
    card = rep.buy_cards[0]
    assert card.reward_risk > 0
    # Prop gate: weak R:R cannot keep A+
    if card.reward_risk < 2.0:
        assert card.grade != "A+"


def test_side_compare_board_flags_correction_risk():
    """Raw sell scores can lean SELL even when H1 grades sells as B/AVOID."""
    from atlas.institutional.manual_scanner import build_manual_scan, render_manual_scan_block
    from atlas.institutional.models import MarketNarrative, utcnow
    from atlas.institutional.trade_areas import TradeArea, TradeAreasReport

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
        mid=4175.0,
        atr_m15=8.0,
    )
    buy = TradeArea(
        "BUY", "support", 4150, 4152, 4151, 70, 3.0, True, ["Support"], "S"
    )
    sell_a = TradeArea(
        "SELL", "htf_resistance", 4178, 4182, 4180, 96, 0.6, True, ["H4 high"], "H4"
    )
    sell_b = TradeArea(
        "SELL", "bearish_fvg", 4176, 4179, 4177.5, 93, 0.3, True, ["Bear FVG"], "BFVG"
    )
    sell_c = TradeArea(
        "SELL", "resistance", 4174, 4176, 4175, 90, 0.1, True, ["R"], "R"
    )
    areas = TradeAreasReport(
        areas=[buy, sell_a, sell_b, sell_c],
        buy_best=[buy],
        sell_best=[sell_a, sell_b, sell_c],
        summary="x",
    )
    rep = build_manual_scan(
        n, areas, None, Bias.BULLISH, Bias.BULLISH, Bias.BULLISH, "London"
    )
    assert rep.side_compare is not None
    assert rep.side_compare.lean == "SELL_LEAN"
    assert rep.side_compare.sell_pressure > rep.side_compare.buy_pressure
    assert "CORRECTION RISK" in rep.side_compare.advice
    assert all(c.grade == "B" for c in rep.sell_cards)
    text = "\n".join(render_manual_scan_block(rep, mid=4175.0, brief=False))
    assert "BUY vs SELL PROBABILITY BOARD" in text
    assert "SELL_LEAN" in text


def test_plan_sl_tp_prop_rr_two_and_three():
    """TP1/TP2 target prop 1:2 / 1:3 from zone mid; near magnets are not forced TP1."""
    from atlas.institutional.manual_scanner import _plan_sl_tp
    from atlas.institutional.trade_areas import TradeArea

    buy = TradeArea("BUY", "support", 4148, 4152, 4150, 88, 0.5, True, ["S"], "S")
    # Near magnet (~$7 above) would crush R:R if used as TP1 — note-only
    near = TradeArea(
        "SELL", "htf_resistance", 4156.5, 4157.5, 4157.0, 95, 0.9, True, ["H4"], "H4"
    )
    far = TradeArea(
        "SELL", "resistance", 4168, 4170, 4169, 90, 2.0, True, ["R"], "R"
    )
    entry, sl, tp1, tp2, sl_l, tp1_l, tp2_l, rr1, rr2, risk, reward, note, ok, ru, r1, r2 = _plan_sl_tp(
        buy, [buy, near, far], atr=8.0, max_risk_points=5.0, lot_size=0.1
    )
    assert entry == 4150.0
    assert sl < entry
    assert tp2 > tp1 > entry
    assert rr1 >= 1.99
    assert rr2 >= 2.99
    # Near magnet before true 2R must not become a sub-2R TP1
    assert tp1 >= entry + risk * 1.95


def test_news_proximity_blocks_inside_window(monkeypatch):
    """Config block windows must gate high-impact USD events by time."""
    from datetime import datetime, timedelta, timezone

    import atlas.institutional.price_action as pa

    now = datetime(2026, 8, 5, 12, 0, tzinfo=timezone.utc)

    class _FakeDT:
        @staticmethod
        def now(tz=None):
            return now

    monkeypatch.setattr(pa, "datetime", _FakeDT)

    events = [
        {
            "title": "FOMC Statement",
            "country": "USD",
            "impact": "High",
            "date": (now + timedelta(minutes=10)).isoformat(),
        }
    ]

    class _Resp:
        def read(self):
            import json

            return json.dumps(events).encode()

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    class _URL:
        @staticmethod
        def urlopen(req, timeout=3):
            return _Resp()

    import urllib.request

    monkeypatch.setattr(urllib.request, "urlopen", _URL.urlopen)
    rep = pa.analyze_news(enabled=True, block_minutes_before=30, block_minutes_after=30)
    assert rep.block_trading is True
    assert rep.status == "Avoid Trading Today"


def test_next_m5_boundary_and_watch_helpers():
    from datetime import datetime, timezone

    from atlas.live.watch import _next_m5_boundary_utc
    from atlas.institutional.watch_m5 import _bar_open_time, _parse_bar_ts

    now = datetime(2026, 8, 5, 20, 15, 34, tzinfo=timezone.utc)
    nxt = _next_m5_boundary_utc(now)
    assert nxt.hour == 20 and nxt.minute == 20
    on_boundary = datetime(2026, 8, 5, 20, 15, 0, tzinfo=timezone.utc)
    # Exactly on boundary → next one (already closed)
    nxt2 = _next_m5_boundary_utc(on_boundary)
    assert nxt2.minute == 20

    df = pd.DataFrame(
        {
            "time": pd.to_datetime(
                ["2026-08-05 22:45:00+00:00", "2026-08-05 22:50:00+00:00", "2026-08-05 22:55:00+00:00"]
            )
        }
    )
    assert "22:50" in _bar_open_time(df, -2)
    assert "22:55" in _bar_open_time(df, -1)
    ts = _parse_bar_ts("2026-08-05 22:50:00+00:00")
    assert ts is not None and ts.minute == 50


def test_session_levels_pdh_pdl_asian():
    from datetime import datetime, timezone

    from atlas.institutional.session_levels import extract_session_levels

    rows = []
    t0 = datetime(2026, 8, 4, 0, 0, tzinfo=timezone.utc)
    # Prev day 4th: high 4180 low 4070
    for h in range(24):
        price = 4100 + h
        hi = 4180 if h == 12 else price + 2
        lo = 4070 if h == 3 else price - 2
        rows.append(
            {
                "time": t0.replace(hour=h),
                "open": price,
                "high": hi,
                "low": lo,
                "close": price,
                "tick_volume": 10,
            }
        )
    # Today 5th asian 00-06
    t1 = datetime(2026, 8, 5, 0, 0, tzinfo=timezone.utc)
    for h in range(0, 7):
        rows.append(
            {
                "time": t1.replace(hour=h),
                "open": 4150,
                "high": 4160 if h == 2 else 4152,
                "low": 4140 if h == 4 else 4148,
                "close": 4150,
                "tick_volume": 10,
            }
        )
    # Forming bar
    rows.append(
        {
            "time": datetime(2026, 8, 5, 12, 0, tzinfo=timezone.utc),
            "open": 4155,
            "high": 4199,
            "low": 4001,
            "close": 4155,
            "tick_volume": 10,
        }
    )
    df = pd.DataFrame(rows)
    now = datetime(2026, 8, 5, 12, 30, tzinfo=timezone.utc)
    rep = extract_session_levels(df, now=now)
    assert rep.pdh is not None and abs(rep.pdh - 4180) < 1e-6
    assert rep.pdl is not None and abs(rep.pdl - 4070) < 1e-6
    assert rep.asian_high is not None and abs(rep.asian_high - 4160) < 1e-6
    assert rep.asian_low is not None and abs(rep.asian_low - 4140) < 1e-6
    # Forming spike must not invent PDH
    assert rep.pdh < 4199


def test_focus_plan_and_sweep_state():
    from atlas.institutional.manual_scanner import build_manual_scan, render_manual_scan_block
    from atlas.institutional.models import MarketNarrative, utcnow
    from atlas.institutional.trade_areas import TradeArea, TradeAreasReport

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
        mid=4157.0,
        atr_m15=8.0,
        extras={
            "pd_zone": "discount",
            "sweep_bullish": True,
            "sweep_bearish": False,
            "last_closed_low": 4155.0,
            "last_closed_high": 4160.0,
        },
    )
    buy = TradeArea(
        "BUY", "buy_liquidity", 4150, 4154, 4152, 90, 0.6, True, ["EQL"], "EQL"
    )
    sell = TradeArea(
        "SELL", "htf_resistance", 4178, 4182, 4180, 88, 2.8, True, ["H4"], "H4"
    )
    areas = TradeAreasReport([buy, sell], [buy], [sell], "x")
    rep = build_manual_scan(
        n, areas, None, Bias.BULLISH, Bias.BULLISH, Bias.BULLISH, "London"
    )
    assert rep.focus is not None
    assert rep.focus.side == "BUY"
    # Mid above buy zone, last low did NOT sweep the box → WAITING_SWEEP
    assert rep.buy_cards[0].sweep_state == "WAITING_SWEEP"
    assert rep.focus.verdict == "WAIT_SWEEP"
    sweep_item = next(i for i in rep.focus.checklist if i.name.startswith("Sweep"))
    assert sweep_item.passed is False
    text = "\n".join(render_manual_scan_block(rep, mid=4157.0))
    assert "FOCUS TRADE" in text


def test_sweep_reclaim_zone_specific_not_global():
    from atlas.institutional.manual_scanner import _sweep_state_for_area
    from atlas.institutional.trade_areas import TradeArea

    buy = TradeArea("BUY", "buy_liquidity", 4150, 4154, 4152, 90, 0.5, True, ["EQL"], "EQL")
    # Global flag alone must not reclaim a distant magnet
    assert (
        _sweep_state_for_area(buy, 4180.0, True, False, last_low=4175.0, atr=8.0)
        == "WAITING_SWEEP"
    )
    # Wick below THIS zone + close back above near zone = RECLAIMED
    assert (
        _sweep_state_for_area(buy, 4156.0, False, False, last_low=4148.0, atr=8.0)
        == "RECLAIMED"
    )
    # Inside box without wick = IN_ZONE (liquidity still waits)
    assert (
        _sweep_state_for_area(buy, 4152.0, False, False, last_low=4151.0, atr=8.0)
        == "IN_ZONE"
    )
    # Below box = SWEPT
    assert (
        _sweep_state_for_area(buy, 4145.0, False, False, last_low=4144.0, atr=8.0)
        == "SWEPT"
    )


def test_focus_liquidity_in_zone_not_watch_ready():
    from atlas.institutional.manual_scanner import build_manual_scan
    from atlas.institutional.models import MarketNarrative, utcnow
    from atlas.institutional.trade_areas import TradeArea, TradeAreasReport

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
        mid=4152.0,
        atr_m15=8.0,
        extras={"pd_zone": "discount", "last_closed_low": 4151.0, "last_closed_high": 4153.0},
    )
    buy = TradeArea(
        "BUY", "buy_liquidity", 4150, 4154, 4152, 90, 0.1, True, ["EQL"], "EQL"
    )
    areas = TradeAreasReport([buy], [buy], [], "x")
    rep = build_manual_scan(
        n, areas, None, Bias.BULLISH, Bias.BULLISH, Bias.BULLISH, "London"
    )
    assert rep.buy_cards[0].sweep_state == "IN_ZONE"
    assert rep.focus is not None
    assert rep.focus.verdict == "WAIT_SWEEP"


def test_challenge_sl_tp_fifty_dollar_point_one_lot():
    """Broker: 0.1 lot = $10 per $1 → $50 risk = $5 stop; TP 1:2 / 1:3."""
    from atlas.institutional.config import load_institutional_config
    from atlas.institutional.manual_scanner import _plan_sl_tp, _refine_area_for_challenge
    from atlas.institutional.trade_areas import TradeArea

    cfg = load_institutional_config(reload=True)
    assert abs(cfg.challenge_px_value() - 10.0) < 1e-6
    assert abs(cfg.challenge_max_risk_points() - 5.0) < 1e-6
    buy = TradeArea("BUY", "support", 4148.5, 4151.5, 4150.0, 90, 0.3, True, ["S"], "S")
    entry, sl, tp1, tp2, *_rest, ok, risk_usd, rew1, rew2 = _plan_sl_tp(
        buy, [buy], atr=8.0, max_risk_points=5.0, lot_size=0.1, usd_per_price_unit_per_lot=100.0
    )
    risk = entry - sl
    assert risk <= 5.05
    assert ok is True
    assert risk_usd <= 50.5
    assert abs(rew1 / risk_usd - 2.0) < 0.15
    assert abs(rew2 / risk_usd - 3.0) < 0.2
    # Wide zone should fail challenge_ok without refine
    wide = TradeArea("BUY", "support", 4130.0, 4150.0, 4140.0, 85, 1.0, True, ["W"], "W")
    *_, ok2, risk_usd2, _, _ = _plan_sl_tp(
        wide, [wide], atr=8.0, max_risk_points=5.0, lot_size=0.1
    )
    assert ok2 is False
    assert risk_usd2 <= 50.5
    # Refine shrinks to upper edge → structural SL can fit challenge
    refined = _refine_area_for_challenge(wide, 5.0, atr=8.0)
    assert (refined.price_high - refined.price_low) < (wide.price_high - wide.price_low)
    assert refined.price_high == wide.price_high
    *_, ok3, risk_usd3, _, _ = _plan_sl_tp(
        refined, [refined], atr=8.0, max_risk_points=5.0, lot_size=0.1
    )
    assert ok3 is True
    assert risk_usd3 <= 50.5


def test_challenge_width_boost_prefers_tight_zones():
    from atlas.institutional.trade_areas import _challenge_width_boost

    assert _challenge_width_boost(2.5, atr=8.0, max_risk_pts=5.0) > _challenge_width_boost(
        20.0, atr=8.0, max_risk_pts=5.0
    )


def test_setup_probability_stacks_chart_factors():
    from datetime import datetime, timezone

    from atlas.institutional.models import Bias, MarketNarrative, ModuleScore
    from atlas.institutional.playbook_engine import PlaybookHit, PlaybookResult
    from atlas.institutional.setup_probability import grade_from_probability, score_setup_probability
    from atlas.institutional.trade_areas import TradeArea

    area = TradeArea(
        "BUY",
        "bullish_fvg",
        4148,
        4151,
        4149.5,
        92,
        0.4,
        True,
        ["Unfilled bullish FVG", "+ overlap htf_support"],
        "FVG",
    )
    narr = MarketNarrative(
        symbol="XAUUSD",
        as_of=datetime.now(timezone.utc),
        overall_bias=Bias.BULLISH,
        htf_bias=Bias.BULLISH,
        mtf_bias=Bias.BULLISH,
        structure_summary="BOS",
        liquidity_summary="ok",
        institutional_confluence="ok",
        sr_summary="ok",
        trend_quality="Trending",
        volatility_regime="normal",
        best_session="London",
        news_status="Trade Today",
        module_scores=[
            ModuleScore("market_structure", 75, 14, Bias.BULLISH, "BOS", True),
            ModuleScore("liquidity", 70, 10, Bias.BULLISH, "ok", True),
            ModuleScore("price_action", 65, 8, Bias.BULLISH, "pin", True),
        ],
        mid=4152.0,
        atr_m15=8.0,
        extras={
            "pd_zone": "discount",
            "session_levels": {"pdl": 4149.0, "asian_low": 4148.5},
            "trendlines": [{"kind": "support", "quality": "Strong", "price": 4149.2, "touches": 3}],
            "news_block": False,
        },
    )
    pb = PlaybookResult(
        hits=[PlaybookHit("Sweep+Reclaim (Bullish)", Bias.BULLISH, 18.0, ["x"])],
        best=PlaybookHit("Sweep+Reclaim (Bullish)", Bias.BULLISH, 18.0, ["x"]),
        total_boost=18.0,
        summary="hit",
        unlock_hints=[],
    )
    high = score_setup_probability(
        area,
        narrative=narr,
        h1_bias=Bias.BULLISH,
        h4_bias=Bias.BULLISH,
        m15_bias=Bias.BULLISH,
        session_name="London-NY Overlap",
        sweep_state="RECLAIMED",
        reward_risk=2.5,
        challenge_ok=True,
        playbooks=pb,
        board_lean="BUY_LEAN",
        against_htf=False,
    )
    assert high.score >= 70
    assert high.tier in ("HIGH", "ELITE")
    assert grade_from_probability("A", "WAIT_FOR_TRIGGER", high, True) in ("A", "A+")

    low = score_setup_probability(
        area,
        narrative=narr,
        h1_bias=Bias.BEARISH,
        h4_bias=Bias.BEARISH,
        m15_bias=Bias.BEARISH,
        session_name="Asian",
        sweep_state="WAITING_SWEEP",
        reward_risk=1.0,
        challenge_ok=False,
        playbooks=None,
        board_lean="SELL_LEAN",
        against_htf=True,
    )
    assert low.tier == "AVOID"
    assert low.score < 50
