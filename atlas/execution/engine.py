"""
Trade execution — PAPER (default) and LIVE.

PAPER: simulates pending → fill when price reaches entry → manage SL/TP.
LIVE: places real MT5 orders only after every validation check passes.
Calculated position size is always used — never silently overridden.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Callable

from atlas.config import load_settings
from atlas.execution.mt5_client import MT5Client
from atlas.journal.trade_journal import TradeJournal
from atlas.models import (
    Direction,
    TradeDecision,
    TradeMode,
    TradeRecord,
    TradeStatus,
)
from atlas.risk.position import (
    ConcurrentTradeGuard,
    DailyRiskState,
    PositionSizeResult,
    calculate_position_size,
)
from atlas.risk.validation import validate_trade

logger = logging.getLogger(__name__)


class ExecutionEngine:
    def __init__(
        self,
        broker: MT5Client,
        journal: TradeJournal | None = None,
        daily_risk: DailyRiskState | None = None,
        concurrent: ConcurrentTradeGuard | None = None,
        on_log: Callable[[str], None] | None = None,
    ) -> None:
        self.broker = broker
        self.journal = journal or TradeJournal()
        self.daily_risk = daily_risk or DailyRiskState()
        self.concurrent = concurrent or ConcurrentTradeGuard()
        self.on_log = on_log or (lambda msg: logger.info(msg))
        self.pending: list[TradeRecord] = []
        self.open_trades: list[TradeRecord] = []

    def _mode(self) -> TradeMode:
        settings = load_settings()
        return TradeMode.LIVE if settings.is_live else TradeMode.PAPER

    def try_execute(self, decision: TradeDecision) -> TradeRecord | None:
        """
        Attempt execution for a scored decision.
        Prints/logs full reasoning whether trade fires or not.
        Returns TradeRecord if an order was accepted (pending or filled), else None.
        """
        self.on_log(decision.reasoning_text())

        specs = self.broker.symbol_specs(decision.symbol)
        balance = self.broker.account_balance()
        sizing = calculate_position_size(
            balance=balance,
            entry=decision.features.entry,
            stop=decision.features.stop,
            symbol=decision.symbol,
            tick_size=specs["tick_size"],
            tick_value=specs["tick_value"],
            lot_step=specs["lot_step"],
            min_lot=specs["min_lot"],
            max_lot=specs["max_lot"],
            contract_size=specs["contract_size"],
        )
        self.on_log(f"Position sizing: {sizing.formula} [{sizing.reason}]")

        validation = validate_trade(
            decision=decision,
            broker=self.broker,
            sizing=sizing,
            daily_risk=self.daily_risk,
            concurrent=self.concurrent,
        )
        self.on_log(validation.summary())

        if not validation.valid:
            self.on_log(
                f"EXECUTION BLOCKED for {decision.symbol}: "
                + "; ".join(validation.failures())
            )
            rejected = self._build_record(
                decision, sizing, TradeStatus.REJECTED, filled=False
            )
            rejected.notes = "; ".join(validation.failures())
            self.journal.log_trade(rejected)
            return None

        # Use calculated size — do not override
        volume = sizing.volume
        mode = self._mode()

        if mode == TradeMode.LIVE:
            return self._execute_live(decision, sizing, volume)
        return self._execute_paper_pending(decision, sizing, volume)

    def _execute_live(
        self,
        decision: TradeDecision,
        sizing: PositionSizeResult,
        volume: float,
    ) -> TradeRecord | None:
        settings = load_settings()
        if not settings.is_live:
            self.on_log("Refusing live order — mode is not LIVE")
            return None

        result = self.broker.place_market_order(
            symbol=decision.symbol,
            direction=decision.direction.value,
            volume=volume,
            stop=decision.features.stop,
            target=decision.features.target,
            comment=f"ATLAS:{decision.score.total}",
        )
        if not result.get("ok"):
            self.on_log(f"LIVE order failed: {result}")
            rec = self._build_record(decision, sizing, TradeStatus.REJECTED, filled=False)
            rec.notes = str(result)
            self.journal.log_trade(rec)
            return None

        rec = self._build_record(decision, sizing, TradeStatus.OPEN, filled=True)
        rec.volume = volume  # enforce calculated size
        rec.entry = float(result.get("price") or decision.features.entry)
        rec.mt5_ticket = result.get("ticket")
        rec.filled_at = datetime.now(timezone.utc)
        self.open_trades.append(rec)
        self.concurrent.open_count = self.broker.open_position_count()
        self.journal.log_trade(rec)
        self.on_log(self._format_execution_log(rec))
        return rec

    def _execute_paper_pending(
        self,
        decision: TradeDecision,
        sizing: PositionSizeResult,
        volume: float,
    ) -> TradeRecord:
        """
        Paper trades start as PENDING and fill only when price reaches entry.
        Avoids instant/unrealistic fills.
        """
        rec = self._build_record(decision, sizing, TradeStatus.PENDING, filled=False)
        rec.volume = volume
        self.pending.append(rec)
        self.journal.log_trade(rec)
        self.on_log(
            f"PAPER PENDING {rec.symbol} {rec.direction.value} "
            f"@ {rec.entry:.5f} vol={rec.volume} (waiting for price to reach entry)"
        )
        self.on_log(self._format_execution_log(rec))
        return rec

    def update_market(
        self,
        symbol: str,
        bar_high: float,
        bar_low: float,
        bar_close: float,
        bar_time: datetime | None = None,
    ) -> list[TradeRecord]:
        """
        Process pending fills and open trade SL/TP for one symbol bar.
        Called from live loop and backtest with the same semantics.
        """
        closed: list[TradeRecord] = []
        bar_time = bar_time or datetime.now(timezone.utc)

        # Fill pendings if price trades through entry
        still_pending: list[TradeRecord] = []
        for trade in self.pending:
            if trade.symbol != symbol:
                still_pending.append(trade)
                continue
            filled = False
            if trade.direction == Direction.LONG and bar_low <= trade.entry <= bar_high:
                filled = True
            elif trade.direction == Direction.SHORT and bar_low <= trade.entry <= bar_high:
                filled = True
            if filled:
                trade.status = TradeStatus.OPEN
                trade.filled_at = bar_time
                self.open_trades.append(trade)
                self.broker.register_paper_position(
                    symbol,
                    {
                        "ticket": trade.trade_id,
                        "direction": trade.direction.value,
                        "volume": trade.volume,
                        "entry": trade.entry,
                    },
                )
                self.concurrent.open_count = len(self.open_trades)
                self.journal.log_trade(trade)
                self.on_log(
                    f"FILLED {trade.mode.value} {trade.symbol} {trade.direction.value} "
                    f"@ {trade.entry:.5f} (pending → open)"
                )
            else:
                still_pending.append(trade)
        self.pending = still_pending

        # Manage open trades — SL/TP (intrabar: conservative stop-first if both hit)
        still_open: list[TradeRecord] = []
        for trade in self.open_trades:
            if trade.symbol != symbol:
                still_open.append(trade)
                continue
            exit_price, result = self._check_exit(trade, bar_high, bar_low)
            if exit_price is not None:
                closed_rec = self._close_trade(trade, exit_price, result, bar_time)
                closed.append(closed_rec)
            else:
                still_open.append(trade)
        self.open_trades = still_open
        self.concurrent.open_count = len(self.open_trades)
        return closed

    def expire_pending(self, max_age_bars: int, bars_waited: dict[str, int]) -> None:
        kept: list[TradeRecord] = []
        for trade in self.pending:
            waited = bars_waited.get(trade.trade_id, 0)
            if waited >= max_age_bars:
                trade.status = TradeStatus.CANCELLED
                trade.notes = f"Pending expired after {waited} bars"
                trade.closed_at = datetime.now(timezone.utc)
                self.journal.log_trade(trade)
                self.on_log(f"CANCELLED pending {trade.symbol} {trade.trade_id}")
            else:
                kept.append(trade)
        self.pending = kept

    def _check_exit(
        self, trade: TradeRecord, high: float, low: float
    ) -> tuple[float | None, str | None]:
        if trade.direction == Direction.LONG:
            hit_sl = low <= trade.stop
            hit_tp = high >= trade.target
            if hit_sl and hit_tp:
                return trade.stop, "loss"  # conservative: assume stop first
            if hit_sl:
                return trade.stop, "loss"
            if hit_tp:
                return trade.target, "win"
        else:
            hit_sl = high >= trade.stop
            hit_tp = low <= trade.target
            if hit_sl and hit_tp:
                return trade.stop, "loss"
            if hit_sl:
                return trade.stop, "loss"
            if hit_tp:
                return trade.target, "win"
        return None, None

    def _close_trade(
        self,
        trade: TradeRecord,
        exit_price: float,
        result: str,
        when: datetime,
    ) -> TradeRecord:
        risk = abs(trade.entry - trade.stop)
        if trade.direction == Direction.LONG:
            pnl_price = exit_price - trade.entry
        else:
            pnl_price = trade.entry - exit_price
        r_mult = (pnl_price / risk) if risk > 0 else 0.0
        if abs(r_mult) < 0.05:
            result = "breakeven"

        trade.status = TradeStatus.CLOSED
        trade.exit_price = exit_price
        trade.result = result
        trade.r_multiple = round(r_mult, 3)
        trade.closed_at = when

        # Approximate cash PnL for daily loss tracking
        specs = self.broker.symbol_specs(trade.symbol)
        pnl_cash = (pnl_price / specs["tick_size"]) * specs["tick_value"] * trade.volume
        # Simpler: risk_amount * R
        risk_amount = self.broker.account_balance() * (
            load_settings().risk.risk_percent / 100.0
        )
        pnl_cash = risk_amount * r_mult

        bal = self.broker.account_balance() + pnl_cash
        eq = bal
        self.broker.update_paper_balance(bal, eq)
        self.broker.clear_paper_position(trade.symbol)
        self.daily_risk.record_closed_pnl(pnl_cash, eq)

        self.journal.log_trade(trade)
        self.on_log(
            f"CLOSED {trade.symbol} {trade.direction.value} "
            f"exit={exit_price:.5f} result={result} R={r_mult:.2f} "
            f"score={trade.score}"
        )
        return trade

    def _build_record(
        self,
        decision: TradeDecision,
        sizing: PositionSizeResult,
        status: TradeStatus,
        filled: bool,
    ) -> TradeRecord:
        now = datetime.now(timezone.utc)
        return TradeRecord(
            trade_id=str(uuid.uuid4())[:8],
            mode=self._mode(),
            symbol=decision.symbol,
            direction=decision.direction,
            status=status,
            entry=decision.features.entry,
            stop=decision.features.stop,
            target=decision.features.target,
            volume=sizing.volume if sizing.valid else 0.0,
            score=decision.score.total,
            reasoning=" | ".join(decision.score.plain_language),
            reward_risk_planned=decision.features.reward_risk,
            opened_at=now,
            filled_at=now if filled else None,
        )

    def _format_execution_log(self, trade: TradeRecord) -> str:
        return (
            f"\n{'─' * 60}\n"
            f"EXECUTED [{trade.mode.value}] {trade.status.value}\n"
            f"  Symbol    : {trade.symbol}\n"
            f"  Direction : {trade.direction.value}\n"
            f"  Volume    : {trade.volume}  (from risk formula — not overridden)\n"
            f"  Entry     : {trade.entry:.5f}\n"
            f"  Stop      : {trade.stop:.5f}\n"
            f"  Target    : {trade.target:.5f}\n"
            f"  Planned RR: {trade.reward_risk_planned:.2f}\n"
            f"  Score     : {trade.score}/100\n"
            f"  Reasoning : {trade.reasoning}\n"
            f"{'─' * 60}"
        )
