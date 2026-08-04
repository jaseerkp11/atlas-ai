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
    print(f"Min score   : {settings.gates.min_score}/100")
    print(f"Min R:R     : 1:{settings.gates.min_reward_risk:g}")
    for k in ("login", "name", "server", "balance", "equity", "leverage"):
        if k in info:
            print(f"{k.capitalize():12}: {info[k]}")
    print(f"Symbols     : {len(settings.symbols)} configured")

    if client.using_live_market_data:
        print()
        print("Live tick check (compare these to your MT5 Market Watch):")
        for sym in ("XAUUSD", "EURUSD", "GBPUSD"):
            if sym not in settings.symbols:
                continue
            snap = client.live_tick_snapshot(sym)
            if not snap:
                bid, ask = client.current_price(sym)
                print(f"  {sym:7} bid={bid:.5f} ask={ask:.5f}")
            else:
                print(
                    f"  {snap['symbol']:7} bid={snap['bid']:.5f} ask={snap['ask']:.5f} "
                    f"time_utc={snap['time']}"
                )
        print()
        print("OK — prices come from the live MT5 terminal tick feed.")
        if settings.is_live:
            print("LIVE MODE — real orders will be sent when all gates pass.")
            print("You are on whatever account MT5 is logged into (demo or real).")
        else:
            print("PAPER mode — live prices, simulated orders only.")
    else:
        print()
        print("WARNING — SYNTHETIC data. Prices will NOT match the real market.")
        print("Fix: open MetaTrader 5, log in, then set MT5_* in .env and re-run.")

    client.disconnect()
    return 0 if ok else 1


def cmd_scan(args: argparse.Namespace) -> int:
    """One-shot scan: score every symbol and print reasoning + S/R for chart watch."""
    from atlas.analysis.levels import build_sr_map
    from atlas.analysis.setups import detect_setup
    from atlas.config import load_settings
    from atlas.execution.mt5_client import MT5Client
    from atlas.live.watch import format_watch_card
    from atlas.scoring.engine import evaluate_setup

    settings = load_settings()
    client = MT5Client()
    client.connect()
    symbols = args.symbols or settings.symbols
    src = "MT5_LIVE_MARKET" if client.using_live_market_data else "SYNTHETIC_NOT_LIVE"
    print(
        f"Mode={settings.mode} | data={src} | "
        f"min_score={settings.gates.min_score} | min_rr=1:{settings.gates.min_reward_risk:g}"
    )
    if client.using_live_market_data:
        print("Live prices (verify vs MT5 Market Watch):")
        for sym in symbols[:5]:
            snap = client.live_tick_snapshot(sym)
            if snap:
                print(
                    f"  {snap['symbol']:7} bid={snap['bid']:.5f} ask={snap['ask']:.5f} "
                    f"utc={snap['time']}"
                )
    else:
        print("WARNING: synthetic prices — connect MT5 for real market levels.")
    print()
    for symbol in symbols:
        frames = {
            "M5": client.copy_rates(symbol, "M5", 400),
            "M15": client.copy_rates(symbol, "M15", 300),
            "H1": client.copy_rates(symbol, "H1", 200),
            "H4": client.copy_rates(symbol, "H4", 150),
        }
        bid, ask = client.current_price(symbol)
        mid = (bid + ask) / 2.0
        sr = build_sr_map(symbol, frames, mid=mid)
        features = detect_setup(symbol, frames)
        if features is None:
            print(f"{symbol}: NO TRADE — insufficient data\n")
            for line in sr.chart_lines():
                print(line)
            print()
            continue
        decision = evaluate_setup(features)
        print(format_watch_card(decision, sr, bid, ask, execute_armed=False))
        print(decision.reasoning_text())
    client.disconnect()
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    from atlas.live.loop import LiveLoop

    loop = LiveLoop(poll_seconds=args.poll)
    loop.start(max_iterations=args.iterations)
    return 0


def cmd_watch(args: argparse.Namespace) -> int:
    """15-minute M15 watch: S/R + signals for chart review; optional --execute."""
    from atlas.live.watch import WatchLoop

    loop = WatchLoop(execute=args.execute)
    loop.start(max_cycles=args.cycles)
    return 0


def cmd_backtest(args: argparse.Namespace) -> int:
    from atlas.backtest.runner import BacktestRunner
    from atlas.config import load_settings

    settings = load_settings()
    symbols = args.symbols or settings.symbols[: args.max_symbols]
    runner = BacktestRunner(on_log=lambda m: logging.getLogger("backtest").info(m) if args.quiet else print(m))
    stats = runner.run(symbols=symbols, bars=args.bars)
    print(
        f"\nDone. trades={stats.total_trades} win_rate={stats.win_rate}% "
        f"avg_R={stats.average_r} max_DD={stats.max_drawdown_pct}%"
    )
    return 0


def cmd_tick(args: argparse.Namespace) -> int:
    """
    High-speed XAUUSD tick scalper — reads every MT5 tick, not candle closes.
    Uses ATLAS_MODE from .env (PAPER | LIVE). Start with PAPER on demo.
    """
    from atlas.tick_scalper.engine import run_tick_scalper

    try:
        run_tick_scalper(max_ticks=args.max_ticks)
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        return 1
    return 0


def cmd_tick_backtest(args: argparse.Namespace) -> int:
    """Replay synthetic or provided tick count through the same tick strategy."""
    from atlas.tick_scalper.backtest import TickBacktester, generate_synthetic_ticks

    ticks = generate_synthetic_ticks(n=args.ticks, seed=args.seed)
    bt = TickBacktester()
    result = bt.run(ticks)
    print("=" * 50)
    print("ATLAS TICK SCALPER BACKTEST (synthetic tape)")
    print("=" * 50)
    print(f"ticks     : {result.ticks_processed}")
    print(f"signals   : {result.signals}")
    print(f"trades    : {len(result.trades)}")
    print(f"win_rate  : {result.win_rate}%  (sample only — not a live claim)")
    print(f"pnl_pts   : {result.total_pnl_points:.1f}")
    if result.trades and not args.quiet:
        print("\nLast trades:")
        for t in result.trades[-10:]:
            print(
                f"  {t.side:4} entry={t.entry:.3f} exit={t.exit:.3f} "
                f"pnl={t.pnl_points:+.1f}pts ({t.reason})"
            )
    print("\nNote: synthetic backtest ≠ live expectancy. Validate on PAPER first.")
    return 0


def cmd_analyze(args: argparse.Namespace) -> int:
    """Manual high-probability scanner (one-shot). No auto trade."""
    from atlas.institutional.analyzer import InstitutionalAnalyzer
    from atlas.institutional.config import load_institutional_config
    from atlas.institutional.dashboard import render_dashboard

    cfg = load_institutional_config(reload=True)
    symbol = args.symbol or cfg.symbol
    analyzer = InstitutionalAnalyzer(cfg)
    if not analyzer.connect():
        print("ERROR: MT5 connection failed")
        return 1
    try:
        decision = analyzer.analyze(symbol)
        print(render_dashboard(decision, full=bool(getattr(args, "full", False))))
    finally:
        analyzer.disconnect()
    return 0


def cmd_institutional(args: argparse.Namespace) -> int:
    """Manual scanner on each NEW M5 close — analysis only."""
    from atlas.institutional.config import load_institutional_config
    from atlas.institutional.watch_m5 import InstitutionalWatch

    cfg = load_institutional_config(reload=True)
    watch = InstitutionalWatch(cfg, execute=bool(getattr(args, "execute", False)))
    try:
        watch.start(max_cycles=args.cycles)
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        return 1
    return 0


def cmd_institutional_backtest(args: argparse.Namespace) -> int:
    from atlas.institutional.backtest_engine import (
        run_institutional_backtest,
        run_m5_institutional_backtest,
    )
    from atlas.institutional.config import load_institutional_config

    cfg = load_institutional_config(reload=True)
    m5_bars = getattr(args, "m5_bars", None)
    if m5_bars:
        m5 = None
        data_source = ""
        # Prefer YOUR broker MT5 XAUUSD history (real score). Yahoo only as fallback.
        from atlas.execution.mt5_client import MT5Client

        client = MT5Client()
        client.connect()
        if client.using_live_market_data:
            need = int(m5_bars) + int(getattr(args, "warmup", 250) or 250) + 50
            raw = client.copy_rates(cfg.symbol, "M5", need)
            if raw is not None and len(raw) >= 300:
                m5 = raw.rename(columns={"volume": "tick_volume"})
                if "tick_volume" not in m5.columns:
                    m5["tick_volume"] = 100
                data_source = f"MT5_{cfg.symbol}_M5_{m5_bars}"
                print(
                    f"Using MT5 live history: {cfg.symbol} M5 bars={len(m5)} "
                    f"(requested window {m5_bars})"
                )
            else:
                print("WARNING: MT5 M5 history too short — falling back to Yahoo GC=F")
        else:
            print("WARNING: MT5 live data not available — falling back to Yahoo GC=F proxy")
        client.disconnect()

        if m5 is None:
            try:
                import yfinance as yf
            except ImportError:
                print("ERROR: install yfinance OR run on Windows with MT5 for real history")
                return 1
            print(f"Fetching Yahoo GC=F 5m proxy for {m5_bars} bars…")
            raw = yf.Ticker("GC=F").history(period="60d", interval="5m", auto_adjust=True)
            if raw is None or len(raw) < 100:
                print("ERROR: could not fetch Yahoo GC=F 5m history")
                return 1
            m5 = raw.reset_index().rename(
                columns={
                    "Datetime": "time",
                    "Open": "open",
                    "High": "high",
                    "Low": "low",
                    "Close": "close",
                    "Volume": "tick_volume",
                }
            )
            data_source = f"YAHOO_GC=F_M5_{m5_bars}_PROXY"

        result = run_m5_institutional_backtest(
            m5,
            bars=int(m5_bars),
            warmup=int(getattr(args, "warmup", 250) or 250),
            step=int(args.step),
            on_log=print if not args.quiet else (lambda m: None),
            data_source=data_source,
        )
    else:
        from atlas.execution.mt5_client import MT5Client

        client = MT5Client()
        client.connect()
        src = "MT5_LIVE" if client.using_live_market_data else "SYNTHETIC_SAMPLE"
        frames = {
            "H4": client.copy_rates(cfg.symbol, "H4", 300),
            "H1": client.copy_rates(cfg.symbol, "H1", 500),
            "M15": client.copy_rates(cfg.symbol, "M15", 800),
            "M5": client.copy_rates(cfg.symbol, "M5", 1000),
            "M1": client.copy_rates(cfg.symbol, "M1", 500),
        }
        client.disconnect()
        result = run_institutional_backtest(
            frames,
            step=args.step,
            on_log=print if not args.quiet else (lambda m: None),
            data_source=src,
        )
    for line in result.summary_lines():
        print(line)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="ATLAS — explainable, risk-managed MT5 trading (watch / paper / live)"
    )
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)

    c = sub.add_parser("connect", help="Test MT5 connection + live tick check")
    c.set_defaults(func=cmd_connect)

    s = sub.add_parser("scan", help="One-shot score + S/R levels for chart watch")
    s.add_argument("--symbols", nargs="+", default=None)
    s.set_defaults(func=cmd_scan)

    w = sub.add_parser(
        "watch",
        help="Scalp watch: update on each NEW M5 close + optional --execute",
    )
    w.add_argument(
        "--execute",
        action="store_true",
        help="Also send orders when gates pass (uses ATLAS_MODE PAPER/LIVE)",
    )
    w.add_argument("--cycles", type=int, default=None, help="Stop after N poll cycles")
    w.set_defaults(func=cmd_watch)

    r = sub.add_parser("run", help="M5 close loop (auto path)")
    r.add_argument("--poll", type=float, default=15.0, help="Seconds between polls")
    r.add_argument("--iterations", type=int, default=None, help="Stop after N ticks (tests)")
    r.set_defaults(func=cmd_run)

    b = sub.add_parser("backtest", help="Backtest with the same scoring code as live")
    b.add_argument("--symbols", nargs="+", default=None)
    b.add_argument("--bars", type=int, default=None)
    b.add_argument("--max-symbols", type=int, default=3)
    b.add_argument("--quiet", action="store_true")
    b.set_defaults(func=cmd_backtest)

    t = sub.add_parser(
        "tick",
        help="XAUUSD high-speed tick scalper (every MT5 tick; PAPER/LIVE via ATLAS_MODE)",
    )
    t.add_argument(
        "--max-ticks",
        type=int,
        default=None,
        help="Stop after N processed ticks (tests / smoke)",
    )
    t.set_defaults(func=cmd_tick)

    tb = sub.add_parser(
        "tick-backtest",
        help="Replay synthetic ticks through the tick scalper strategy",
    )
    tb.add_argument("--ticks", type=int, default=8000, help="Number of synthetic ticks")
    tb.add_argument("--seed", type=int, default=42)
    tb.add_argument("--quiet", action="store_true")
    tb.set_defaults(func=cmd_tick_backtest)

    a = sub.add_parser(
        "analyze",
        help="Manual high-prob scanner (short brief). Add --full for details",
    )
    a.add_argument("--symbol", default=None, help="Override symbol (default XAUUSD)")
    a.add_argument(
        "--full",
        action="store_true",
        help="Show full module/fib/playbook dump (default is short brief)",
    )
    a.set_defaults(func=cmd_analyze)

    inst = sub.add_parser(
        "institutional",
        help="Manual high-prob M5 scanner (each new M5 close) — analysis only, no auto trade",
    )
    inst.add_argument(
        "--cycles",
        type=int,
        default=None,
        help="Stop after N closed-M5 scans",
    )
    inst.add_argument(
        "--execute",
        action="store_true",
        help=argparse.SUPPRESS,  # ignored — manual scanner only
    )
    inst.add_argument("--poll", type=float, default=60.0, help=argparse.SUPPRESS)
    inst.set_defaults(func=cmd_institutional)

    ib = sub.add_parser(
        "institutional-backtest",
        help="Backtest institutional decision engine on historical MT5/synthetic bars",
    )
    ib.add_argument("--step", type=int, default=1, help="Bar step (1 = every bar)")
    ib.add_argument(
        "--m5-bars",
        type=int,
        default=None,
        help="If set, walk this many Yahoo GC=F 5m bars with structure multi-TF",
    )
    ib.add_argument("--warmup", type=int, default=250, help="M5 warmup bars before decisions")
    ib.add_argument("--quiet", action="store_true")
    ib.set_defaults(func=cmd_institutional_backtest)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _setup_logging(args.verbose)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
