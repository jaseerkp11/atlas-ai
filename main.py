#!/usr/bin/env python3
"""
ATLAS CLI — paper/live loop, backtest, and connection check.

No win-rate claims. Gates (min score, min R:R) are enforced from config/settings.yaml only.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure project root on path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    log_dir = ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_dir / "atlas.log", encoding="utf-8"),
        ],
    )


def cmd_connect(_: argparse.Namespace) -> int:
    from atlas.config import load_settings
    from atlas.execution.mt5_client import MT5Client

    client = MT5Client()
    ok = client.connect()
    info = client.account_info_dict()
    settings = load_settings()
    print("=" * 50)
    print("ATLAS — Broker Connection")
    print("=" * 50)
    print(f"Connected   : {ok}")
    print(f"Data source : {info.get('data_source', '?')}")
    print(f"Order mode  : {info.get('order_mode', settings.mode)}")
    for k in ("login", "name", "server", "balance", "equity", "leverage"):
        if k in info:
            print(f"{k.capitalize():12}: {info[k]}")
    print(f"Symbols     : {len(settings.symbols)} configured")

    if client.using_live_market_data:
        for sym in ("XAUUSD", "EURUSD"):
            if sym in settings.symbols:
                bid, ask = client.current_price(sym)
                print(f"Live {sym:7}: bid={bid:.5f} ask={ask:.5f}")
        print()
        print("OK — using LIVE market data from MT5.")
        if settings.is_paper:
            print("Orders stay PAPER (simulated) until you set ATLAS_MODE=LIVE.")
    else:
        print()
        print("WARNING — SYNTHETIC data. Prices will NOT match the real market.")
        print("Fix: open MetaTrader 5, log in, then set MT5_* in .env and re-run.")

    client.disconnect()
    return 0 if ok else 1


def cmd_scan(args: argparse.Namespace) -> int:
    """One-shot scan: score every symbol and print reasoning (no execution)."""
    from atlas.analysis.setups import detect_setup
    from atlas.config import load_settings
    from atlas.execution.mt5_client import MT5Client
    from atlas.scoring.engine import evaluate_setup

    settings = load_settings()
    client = MT5Client()
    client.connect()
    symbols = args.symbols or settings.symbols
    src = "MT5_LIVE_MARKET" if client.using_live_market_data else "SYNTHETIC_NOT_LIVE"
    print(f"Mode={settings.mode} | data={src} | min_score={settings.gates.min_score} | min_rr={settings.gates.min_reward_risk}")
    if not client.using_live_market_data:
        print("WARNING: synthetic prices — connect MT5 for real market levels.\n")
    print()
    for symbol in symbols:
        frames = {
            "M5": client.copy_rates(symbol, "M5", 400),
            "M15": client.copy_rates(symbol, "M15", 300),
            "H1": client.copy_rates(symbol, "H1", 200),
            "H4": client.copy_rates(symbol, "H4", 150),
        }
        features = detect_setup(symbol, frames)
        if features is None:
            print(f"{symbol}: NO TRADE — insufficient data\n")
            continue
        decision = evaluate_setup(features)
        print(decision.reasoning_text())
    client.disconnect()
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    from atlas.live.loop import LiveLoop

    loop = LiveLoop(poll_seconds=args.poll)
    loop.start(max_iterations=args.iterations)
    return 0


def cmd_backtest(args: argparse.Namespace) -> int:
    from atlas.backtest.runner import BacktestRunner
    from atlas.config import load_settings

    settings = load_settings()
    symbols = args.symbols or settings.symbols[: args.max_symbols]
    runner = BacktestRunner(on_log=lambda m: logging.getLogger("backtest").info(m) if args.quiet else print(m))
    stats = runner.run(symbols=symbols, bars=args.bars)
    # Non-zero only on hard failure; low win rate is not a failure
    print(
        f"\nDone. trades={stats.total_trades} win_rate={stats.win_rate}% "
        f"avg_R={stats.average_r} max_DD={stats.max_drawdown_pct}%"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="ATLAS — explainable, risk-managed MT5 trading system (PAPER default)"
    )
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)

    c = sub.add_parser("connect", help="Test broker / paper connection")
    c.set_defaults(func=cmd_connect)

    s = sub.add_parser("scan", help="Score all symbols once and print reasoning")
    s.add_argument("--symbols", nargs="+", default=None)
    s.set_defaults(func=cmd_scan)

    r = sub.add_parser("run", help="Start real-time loop (PAPER unless ATLAS_MODE=LIVE)")
    r.add_argument("--poll", type=float, default=15.0, help="Seconds between polls")
    r.add_argument("--iterations", type=int, default=None, help="Stop after N ticks (tests)")
    r.set_defaults(func=cmd_run)

    b = sub.add_parser("backtest", help="Run backtest with the same scoring code as live")
    b.add_argument("--symbols", nargs="+", default=None)
    b.add_argument("--bars", type=int, default=None)
    b.add_argument("--max-symbols", type=int, default=3)
    b.add_argument("--quiet", action="store_true")
    b.set_defaults(func=cmd_backtest)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _setup_logging(args.verbose)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
