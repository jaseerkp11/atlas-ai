"""
Institutional Market Analysis orchestrator.

Builds a full market narrative across H4/H1/M15/M5/M1, runs confluence,
and returns BUY / SELL / WAIT / NO_TRADE — preferring NO TRADE.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from atlas.analysis.timeframes import analyze_timeframe
from atlas.analysis.volatility import latest_atr
from atlas.execution.mt5_client import MT5Client
from atlas.institutional.ai_decision_engine import decide
from atlas.institutional.config import InstitutionalConfig, load_institutional_config
from atlas.institutional.confluence_engine import compute_confluence
from atlas.institutional.fair_value_gap import analyze_fvg, analyze_order_blocks
from atlas.institutional.fibonacci_engine import analyze_fibonacci
from atlas.institutional.liquidity_engine import analyze_liquidity
from atlas.institutional.market_structure import analyze_structure
from atlas.institutional.models import (
    Bias,
    InstitutionalDecision,
    MarketNarrative,
    utcnow,
)
from atlas.institutional.price_action import (
    analyze_news,
    analyze_price_action,
    analyze_session,
    analyze_trend,
)
from atlas.institutional.support_resistance import analyze_support_resistance
from atlas.institutional.trendline_engine import analyze_trendlines
from atlas.models import Direction


class InstitutionalAnalyzer:
    def __init__(self, cfg: InstitutionalConfig | None = None, client: MT5Client | None = None) -> None:
        self.cfg = cfg or load_institutional_config()
        self.client = client or MT5Client()
        self._connected = False

    def connect(self) -> bool:
        self._connected = self.client.connect()
        return self._connected

    def disconnect(self) -> None:
        self.client.disconnect()
        self._connected = False

    def load_frames(self, symbol: str | None = None) -> dict[str, pd.DataFrame]:
        symbol = symbol or self.cfg.symbol
        frames: dict[str, pd.DataFrame] = {}
        bars = {"H4": 200, "H1": 300, "M15": 400, "M5": 500, "M1": 300}
        for tf in self.cfg.timeframes:
            frames[tf] = self.client.copy_rates(symbol, tf, bars.get(tf, 300))
        return frames

    def analyze(self, symbol: str | None = None, frames: dict[str, pd.DataFrame] | None = None) -> InstitutionalDecision:
        symbol = symbol or self.cfg.symbol
        if frames is None:
            if not self._connected:
                self.connect()
            frames = self.load_frames(symbol)

        m15 = frames.get("M15")
        h4 = frames.get("H4")
        h1 = frames.get("H1")
        m5 = frames.get("M5")
        m1 = frames.get("M1")
        setup_df = m15 if m15 is not None and len(m15) else m5

        mid = 0.0
        if setup_df is not None and len(setup_df):
            mid = float(setup_df["close"].iloc[-2] if len(setup_df) > 2 else setup_df["close"].iloc[-1])
        atr = latest_atr(setup_df, 14) if setup_df is not None else 1.0

        lookback = int(self.cfg.analysis.get("swing_lookback", 3))

        # --- Module analyses ---
        struct = analyze_structure(setup_df, lookback=lookback)
        # HTF structure for bias
        h4_s = analyze_structure(h4, lookback=lookback) if h4 is not None else struct
        h1_s = analyze_structure(h1, lookback=lookback) if h1 is not None else struct

        liq = analyze_liquidity(setup_df, lookback=lookback)
        sr = analyze_support_resistance(symbol, frames, mid, atr or 1.0)
        fvg = analyze_fvg(setup_df, mid)
        ob = analyze_order_blocks(setup_df, mid)
        # Prefer H1 structure bias for fib anchors
        fib = analyze_fibonacci(
            frames,
            mid,
            lookback=lookback,
            min_swing_atr=float(self.cfg.analysis.get("fib_min_swing_atr", 1.5)),
            structure_bias=h1_s.bias,
        )
        tl = analyze_trendlines(h1 if h1 is not None else setup_df, lookback=lookback)
        pa = analyze_price_action(m5 if m5 is not None else setup_df)
        trend = analyze_trend(h1 if h1 is not None and len(h1) > 50 else setup_df)
        session = analyze_session()
        news = analyze_news(enabled=bool(self.cfg.news.get("enabled", True)))

        modules = [
            trend.module,
            h1_s.module,
            liq.module,
            sr.module,
            ob.module,
            fvg.module,
            fib.module,
            tl.module,
            pa.module,
            trend.momentum_module,
            trend.volatility_module,
            news.module,
            session.module,
        ]
        confluence = compute_confluence(modules, self.cfg.weights)

        htf_aligned = (
            h4_s.bias == h1_s.bias
            and h4_s.bias != Bias.NEUTRAL
            and h1_s.bias != Bias.NEUTRAL
        )
        overall = confluence.consensus_bias
        if htf_aligned:
            overall = h1_s.bias

        narrative = MarketNarrative(
            symbol=symbol,
            as_of=utcnow(),
            overall_bias=overall,
            htf_bias=h4_s.bias if h4_s.bias == h1_s.bias else Bias.NEUTRAL,
            mtf_bias=struct.bias,
            structure_summary=h1_s.summary,
            liquidity_summary=liq.summary,
            institutional_confluence=confluence.summary,
            sr_summary=sr.summary,
            trend_quality=trend.regime,
            volatility_regime=trend.volatility_module.detail,
            best_session=session.best_session,
            news_status=news.status,
            module_scores=confluence.modules,
            zones=sr.zones + fvg.zones + ob.zones,
            fib_levels=fib.levels,
            mid=mid,
            atr_m15=atr or 0.0,
            extras={
                "h4": h4_s.summary,
                "h1": h1_s.summary,
                "fib_anchor": fib.anchor_tf,
                "session_rank": session.ranking,
            },
        )

        return decide(
            cfg=self.cfg,
            narrative=narrative,
            confluence=confluence,
            m15=setup_df,
            m1=m1,
            pa=pa,
            news_block=news.block_trading,
            session_ok=session.trade_window_ok,
            htf_aligned=htf_aligned,
        )
