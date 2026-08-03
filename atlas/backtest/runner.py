"""
Backtest runner — uses the EXACT same scoring/decision code as live trading.

Orders are simulated as PENDING until price actually reaches the entry level
(no instant fills). Outputs total trades, win rate, average R-multiple,
max drawdown, and a full CSV trade log.

Limitations are stated explicitly — this is not a performance promise.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import pandas as pd

from atlas.analysis.setups import detect_setup
from atlas.config import ROOT, load_settings
from atlas.execution.engine import ExecutionEngine
from atlas.execution.mt5_client import MT5Client, generate_synthetic_ohlc
from atlas.journal.trade_journal import TradeJournal
from atlas.models import BacktestStats, Direction, TradeMode, TradeStatus
from atlas.scoring.engine import evaluate_setup

logger = logging.getLogger(__name__)

BACKTEST_LIMITATIONS = [
    "Spread and slippage are NOT modeled — real fills will be worse.",
    "Historical / synthetic data window is limited; regimes change.",
    "Pending fills use OHLC range (intrabar path unknown); stop-first if SL+TP same bar.",
    "Past results do not guarantee future performance.",
    "No guarantee of edge — this measures process consistency, not a promised win rate.",
]


def _resample_from_m5(m5: pd.DataFrame, rule: str) -> pd.DataFrame:
    df = m5.copy()
    df = df.set_index("time")
    ohlc = df.resample(rule).agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
    ).dropna()
    ohlc = ohlc.reset_index()
    return ohlc


def build_frames_at(
    m5: pd.DataFrame,
    end_idx: int,
    min_bars: int = 50,
) -> dict[str, pd.DataFrame] | None:
    """Slice M5 history up to end_idx and derive higher TFs — no look-ahead."""
    if end_idx < min_bars:
        return None
    slice_m5 = m5.iloc[: end_idx + 1].copy().reset_index(drop=True)
    frames = {
        "M5": slice_m5,
        "M15": _resample_from_m5(slice_m5, "15min"),
        "H1": _resample_from_m5(slice_m5, "1h"),
        "H4": _resample_from_m5(slice_m5, "4h"),
    }
    if len(frames["M15"]) < 30 or len(frames["H1"]) < 20 or len(frames["H4"]) < 10:
        return None
    return frames


class BacktestRunner:
    def __init__(
        self,
        broker: MT5Client | None = None,
        on_log: Callable[[str], None] | None = None,
    ) -> None:
        self.broker = broker or MT5Client()
        self.on_log = on_log or (lambda m: logger.info(m))

    def run(
        self,
        symbols: list[str] | None = None,
        bars: int | None = None,
        data: dict[str, pd.DataFrame] | None = None,
        journal_path: Path | str | None = None,
    ) -> BacktestStats:
        settings = load_settings()
        symbols = symbols or list(settings.symbols[:4])  # default subset for speed
        bars = bars or int(settings.backtest["history_bars_m5"])
        initial = float(settings.backtest["initial_balance"])
        pending_expiry = int(settings.backtest["pending_expiry_bars"])

        out_dir = ROOT / settings.paths.get("backtest_dir", "journals/backtest")
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        jpath = Path(journal_path) if journal_path else out_dir / f"backtest_{stamp}.csv"

        # Force paper-like broker state for backtest accounting
        self.broker.connect()
        self.broker.update_paper_balance(initial, initial)

        journal = TradeJournal(path=jpath)
        engine = ExecutionEngine(
            broker=self.broker,
            journal=journal,
            on_log=self.on_log,
        )

        # Load or synthesize M5 data per symbol
        m5_data: dict[str, pd.DataFrame] = {}
        for sym in symbols:
            if data and sym in data:
                m5_data[sym] = data[sym].copy()
            else:
                df = self.broker.copy_rates(sym, "M5", bars)
                if df is None or len(df) < 100:
                    df = generate_synthetic_ohlc(sym, "M5", bars, seed=hash(sym) % 10_000)
                m5_data[sym] = df.reset_index(drop=True)

        equity_curve = [initial]
        bars_waited: dict[str, int] = {}
        decisions_logged = 0

        # Align on the shortest series length
        max_len = min(len(df) for df in m5_data.values())
        warmup = max(200, max_len // 10)

        self.on_log(
            f"Backtest start: symbols={symbols} bars={max_len} warmup={warmup} "
            f"balance={initial} gates=score>={settings.gates.min_score} "
            f"RR>={settings.gates.min_reward_risk}"
        )

        for i in range(warmup, max_len):
            # Only evaluate on new M5 candle close (each bar is a close)
            for sym, m5 in m5_data.items():
                bar = m5.iloc[i]
                bar_time = pd.Timestamp(bar["time"]).to_pydatetime()
                if bar_time.tzinfo is None:
                    bar_time = bar_time.replace(tzinfo=timezone.utc)

                # Update pending/open management first
                engine.update_market(
                    symbol=sym,
                    bar_high=float(bar["high"]),
                    bar_low=float(bar["low"]),
                    bar_close=float(bar["close"]),
                    bar_time=bar_time,
                )

                # Age pendings
                for p in list(engine.pending):
                    if p.symbol == sym:
                        bars_waited[p.trade_id] = bars_waited.get(p.trade_id, 0) + 1
                engine.expire_pending(pending_expiry, bars_waited)

                # Skip new signals if already pending/open on symbol
                if any(t.symbol == sym for t in engine.pending + engine.open_trades):
                    continue

                frames = build_frames_at(m5, i)
                if frames is None:
                    continue

                features = detect_setup(sym, frames)
                if features is None:
                    continue

                decision = evaluate_setup(features)  # SAME code as live
                decisions_logged += 1
                # Log reasoning (accepted or rejected)
                self.on_log(decision.reasoning_text())

                if decision.allowed:
                    # Tag mode as BACKTEST on the record after
                    rec = engine.try_execute(decision)
                    if rec is not None:
                        rec.mode = TradeMode.BACKTEST
                        # Re-journal with corrected mode
                        journal.log_trade(rec)

            equity_curve.append(self.broker.account_equity())

        # Cancel remaining pendings; leave opens marked (unresolved at end)
        for p in list(engine.pending):
            p.status = TradeStatus.CANCELLED
            p.notes = "Backtest ended — pending unfilled"
            p.closed_at = datetime.now(timezone.utc)
            journal.log_trade(p)
        engine.pending.clear()

        stats = self._compute_stats(journal.path, initial, equity_curve)
        stats.limitations = list(BACKTEST_LIMITATIONS)
        self._print_report(stats, journal.path)
        return stats

    def _compute_stats(
        self,
        journal_path: Path,
        initial: float,
        equity_curve: list[float],
    ) -> BacktestStats:
        df = pd.read_csv(journal_path)
        closed = df[df["status"] == "CLOSED"].copy() if len(df) else df
        # Deduplicate by trade_id keeping last CLOSED row
        if len(closed):
            closed = closed.drop_duplicates(subset=["trade_id"], keep="last")

        total = len(closed)
        wins = int((closed["result"] == "win").sum()) if total else 0
        losses = int((closed["result"] == "loss").sum()) if total else 0
        be = int((closed["result"] == "breakeven").sum()) if total else 0
        win_rate = (wins / total * 100.0) if total else 0.0
        avg_r = float(closed["r_multiple"].astype(float).mean()) if total else 0.0
        net_r = float(closed["r_multiple"].astype(float).sum()) if total else 0.0

        # Max drawdown from equity curve
        peak = equity_curve[0]
        max_dd = 0.0
        for e in equity_curve:
            peak = max(peak, e)
            if peak > 0:
                dd = (peak - e) / peak * 100.0
                max_dd = max(max_dd, dd)

        return BacktestStats(
            total_trades=total,
            wins=wins,
            losses=losses,
            breakevens=be,
            win_rate=round(win_rate, 2),
            average_r=round(avg_r, 3),
            max_drawdown_pct=round(max_dd, 2),
            net_r=round(net_r, 3),
            starting_balance=initial,
            ending_balance=round(equity_curve[-1], 2) if equity_curve else initial,
        )

    def _print_report(self, stats: BacktestStats, journal_path: Path) -> None:
        lines = [
            "",
            "=" * 60,
            "BACKTEST RESULTS",
            "=" * 60,
            f"Total trades     : {stats.total_trades}",
            f"Wins / Losses / BE: {stats.wins} / {stats.losses} / {stats.breakevens}",
            f"Win rate         : {stats.win_rate:.1f}%",
            f"Average R        : {stats.average_r:.3f}",
            f"Net R            : {stats.net_r:.3f}",
            f"Max drawdown     : {stats.max_drawdown_pct:.2f}%",
            f"Start balance    : {stats.starting_balance:.2f}",
            f"End balance      : {stats.ending_balance:.2f}",
            f"Trade log CSV    : {journal_path}",
            "",
            "LIMITATIONS (read before drawing conclusions):",
        ]
        for lim in stats.limitations:
            lines.append(f"  • {lim}")
        lines.append("=" * 60)
        report = "\n".join(lines)
        self.on_log(report)
        print(report)
