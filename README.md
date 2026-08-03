"""
ATLAS — Explainable, risk-managed MT5 auto-trading system

This system does **not** claim a win rate, edge, or guaranteed profitability.
Its value is consistency, transparent reasoning, and enforced risk control.

## What it does

1. **Multi-timeframe analysis** — H4 regime → H1 trend → M15 setups → M5 trigger
2. **Structure detection** — BOS, CHoCH, liquidity sweeps, fair value gaps, order blocks
3. **Scored decisions** — every setup scored /100 with plain-language pass/fail reasons
4. **Hard gates** — minimum score **and** minimum R:R, defined once in `config/settings.yaml`
5. **Enforced risk** — ATR stops, balance×risk%÷stop sizing (actually used), session/spread/margin checks, daily loss halt, max concurrent trades
6. **PAPER / LIVE** — defaults to PAPER; LIVE only when `ATLAS_MODE=LIVE`
7. **Backtest** — same scoring code as live; pending fills until price reaches entry
8. **Journal** — CSV log of every trade with score and reasoning

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt

# One-shot scan (synthetic data if MT5 unavailable)
python main.py scan

# Short paper loop (2 ticks)
python main.py run --iterations 2 --poll 1

# Backtest (same decision engine)
python main.py backtest --bars 1500 --max-symbols 2

# Connection check
python main.py connect
```

## Configuration

All trade gates live in **one place**:

```yaml
# config/settings.yaml
gates:
  min_score: 70
  min_reward_risk: 3.0

risk:
  risk_percent: 0.5
  daily_loss_limit_percent: 2.0
  max_concurrent_trades: 3
```

Mode switch (env overrides YAML):

```bash
export ATLAS_MODE=PAPER   # default
# export ATLAS_MODE=LIVE  # real orders — Windows + MT5 terminal required
```

## Scoring weights (sum = 100)

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

Every evaluation prints which factors passed and which failed — including rejected setups.

## Project layout

```
atlas/
  config.py           # loads gates/weights — single source of truth
  models.py
  analysis/           # structure, liquidity, FVG, OB, MTF, setups
  scoring/engine.py   # score + gate check (shared live/backtest)
  risk/               # sizing, validation, daily limits
  execution/          # MT5 client + PAPER/LIVE engine
  backtest/runner.py  # pending-fill simulation
  journal/            # CSV trade journal
  live/loop.py        # M5 close loop
config/settings.yaml
main.py
tests/
```

## MT5 note

The official `MetaTrader5` Python package requires **Windows** and a running MT5 terminal.
On Linux/macOS (and in PAPER mode), ATLAS uses a simulation client so analysis, scoring,
backtests, and paper trading still run. LIVE order routing needs Windows + MT5.

## Backtest limitations

- Spread/slippage not modeled
- Limited historical (or synthetic) window
- Intrabar path unknown (OHLC range fills; stop-first if SL and TP both hit)
- Past results do not guarantee future ones
- No fabricated edge — metrics describe process outcomes only

## Safety

- Default mode is PAPER
- Pre-trade validation **blocks** execution on failure
- Daily loss limit halts **new** trades only
- Position size from the risk formula is what gets sent — not overridden
