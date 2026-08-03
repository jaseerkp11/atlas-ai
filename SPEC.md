# ATLAS AI - System Specification

## Vision

ATLAS is a disciplined, risk-managed trading system for major/minor forex pairs and gold.

It scores setups with transparent reasoning, enforces hard risk gates, and can execute in
PAPER (default) or LIVE mode via MetaTrader 5. It does **not** claim a win rate or
guaranteed edge — consistency and capital preservation come first.

---

# Markets

Major / minor forex and gold (configurable in `config/settings.yaml`):

- EURUSD, GBPUSD, USDJPY, AUDUSD, USDCAD, USDCHF, NZDUSD
- EURGBP, EURJPY, GBPJPY, AUDNZD, CADJPY
- XAUUSD

---

# Timeframes

| Role | TF |
|------|-----|
| Market Regime | H4 |
| Trend | H1 |
| Setup Detection | M15 |
| Precision Entry | M5 |

Each timeframe independently reports: trend direction, strength/confidence, swing highs/lows.

---

# Core Principles

Never force trades. If gates fail: **NO TRADE** — with full reasoning printed either way.

Capital preservation is always the priority.

---

# Setup Detection (reusable functions)

- Break of Structure (BOS)
- Change of Character (CHoCH)
- Liquidity sweeps
- Fair value gaps
- Order blocks

---

# Confidence Score (weights sum to 100)

| Factor | Weight |
|--------|--------|
| H4 regime alignment | 15 |
| H1 trend confirmation | 15 |
| Structure break (BOS/CHoCH) | 15 |
| Liquidity sweep | 15 |
| Fair value gap | 10 |
| Order block | 10 |
| M5 trigger | 15 |
| Volatility context | 5 |

---

# Trade Gates (single source of truth: `config/settings.yaml`)

- Minimum score (default 70)
- Minimum reward:risk (default 3.0)

Both enforced in `atlas.config.TradeGates.allows()` — not duplicated elsewhere.

---

# Risk Management (enforced, not decorative)

- Position size = balance × risk% ÷ stop distance (calculated size is used)
- ATR-based stop distance
- Pre-trade validation blocks execution: connection, session, spread, margin, conflicts
- Daily loss limit halts **new** trades
- Max concurrent open trades

---

# Execution

- `ATLAS_MODE=PAPER` (default) or `LIVE`
- Real-time loop on each new M5 close
- Full reasoning logged every evaluation

---

# Backtesting

Same scoring/decision code as live. Pending fills until price reaches entry.
Reports: trades, win rate, average R, max drawdown, CSV log + explicit limitations.

---

# Trade Journal

Every paper / backtest / live trade: time, symbol, direction, entry, stop, target,
exit, result, R-multiple, score, reasoning.
