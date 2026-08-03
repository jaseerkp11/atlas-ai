# ATLAS Roadmap

## Phase 1 — Foundation
✅ Git / Python / project layout / central config

## Phase 2 — Broker & data
✅ MT5 client abstraction (LIVE on Windows, PAPER/sim elsewhere)
✅ Multi-timeframe OHLC fetch / synthetic fallback

## Phase 3 — Market structure
✅ Swing HH/HL/LH/LL
✅ BOS / CHoCH

## Phase 4 — Liquidity
✅ Sweeps of swing highs/lows
✅ Session window filter

## Phase 5 — Volatility
✅ ATR-based stops
✅ Expansion context in scoring

## Phase 6 — Opportunity & scoring
✅ Setup orchestrator (FVG, OB, sweep, trigger)
✅ Weighted score /100 + plain-language reasoning
✅ Hard min-score + min-RR gates (one place)

## Phase 7 — Risk & execution
✅ Position sizing from risk formula (enforced)
✅ Pre-trade validation (blocks on failure)
✅ Daily loss limit + max concurrent trades
✅ PAPER / LIVE execution engine

## Phase 8 — Backtest & journal
✅ Shared-logic backtester with pending fills
✅ CSV trade journal with score/reasoning

## Future
- Dashboard / replay UI
- Broker-accurate tick-value sizing for every symbol suffix
- Optional AI narrative layer on top of the deterministic score
