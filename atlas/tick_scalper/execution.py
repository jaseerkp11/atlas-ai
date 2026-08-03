"""
Trade execution — MARKET entries/exits only. No pending grids, no averaging.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from atlas.tick_scalper.config import TickScalperConfig, load_tick_config
from atlas.tick_scalper.logger import TickLogger
from atlas.tick_scalper.mt5_feed import MT5TickFeed, Tick
from atlas.tick_scalper.strategy import OpenState, Side, Signal

try:
    import MetaTrader5 as mt5

    MT5_OK = True
except ImportError:
    mt5 = None  # type: ignore
    MT5_OK = False


@dataclass
class ExecResult:
    ok: bool
    ticket: int
    price: float
    volume: float
    comment: str


class TickExecutor:
    def __init__(
        self,
        feed: MT5TickFeed,
        log: TickLogger,
        cfg: TickScalperConfig | None = None,
    ) -> None:
        self.feed = feed
        self.log = log
        self.cfg = cfg or load_tick_config()
        self.paper_position: OpenState | None = None
        self._paper_ticket = 900000

    def open_market(self, signal: Signal, volume: float, sl: float, tp: float) -> ExecResult:
        symbol = self.cfg.symbol
        if volume <= 0:
            return ExecResult(False, 0, 0.0, 0.0, "volume<=0")

        if self.cfg.is_paper or not self.feed._use_mt5 or not MT5_OK:
            price = signal.ask if signal.side == Side.BUY else signal.bid
            self._paper_ticket += 1
            ticket = self._paper_ticket
            self.paper_position = OpenState(
                side=signal.side,
                entry=price,
                sl=sl,
                tp=tp,
                volume=volume,
                ticket=ticket,
            )
            self.log.trade(
                event="ENTRY",
                symbol=symbol,
                direction=signal.side.value,
                volume=volume,
                price=price,
                sl=sl,
                tp=tp,
                ticket=ticket,
                reason=signal.reason,
                mode="PAPER",
            )
            return ExecResult(True, ticket, price, volume, "PAPER_FILL")

        assert mt5 is not None
        order_type = mt5.ORDER_TYPE_BUY if signal.side == Side.BUY else mt5.ORDER_TYPE_SELL
        price = signal.ask if signal.side == Side.BUY else signal.bid
        deviation = int(self.cfg.max_slippage_points)
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": float(volume),
            "type": order_type,
            "price": float(price),
            "sl": float(sl),
            "tp": float(tp),
            "deviation": deviation,
            "magic": self.cfg.magic,
            "comment": "ATLAS_TICK",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        result = mt5.order_send(request)
        if result is None:
            return ExecResult(False, 0, 0.0, 0.0, str(mt5.last_error()))
        ok = result.retcode == mt5.TRADE_RETCODE_DONE
        fill = float(result.price or price)
        ticket = int(result.order or result.deal or 0)
        if ok:
            self.log.trade(
                event="ENTRY",
                symbol=symbol,
                direction=signal.side.value,
                volume=volume,
                price=fill,
                sl=sl,
                tp=tp,
                ticket=ticket,
                reason=signal.reason,
                mode="LIVE",
            )
        return ExecResult(ok, ticket, fill, float(result.volume or volume), result.comment)

    def close_market(self, pos: OpenState, tick: Tick, reason: str) -> ExecResult:
        symbol = self.cfg.symbol
        # PAPER / offline only — never skip the LIVE broker close because of a local cache
        if self.cfg.is_paper or not self.feed._use_mt5 or not MT5_OK:
            price = tick.bid if pos.side == Side.BUY else tick.ask
            pnl_points = (
                (price - pos.entry) if pos.side == Side.BUY else (pos.entry - price)
            ) / self.feed.symbol_point()
            # rough $: 1 point ≈ $1/lot for XAU when point=0.01
            pnl = pnl_points * pos.volume
            self.log.trade(
                event="EXIT",
                symbol=symbol,
                direction=pos.side.value,
                volume=pos.volume,
                price=price,
                sl=pos.sl,
                tp=pos.tp,
                ticket=pos.ticket,
                pnl=round(pnl, 2),
                reason=reason,
                mode="PAPER",
            )
            self.paper_position = None
            return ExecResult(True, pos.ticket, price, pos.volume, reason)

        assert mt5 is not None
        # Close by opposite deal
        close_type = mt5.ORDER_TYPE_SELL if pos.side == Side.BUY else mt5.ORDER_TYPE_BUY
        price = tick.bid if pos.side == Side.BUY else tick.ask
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": float(pos.volume),
            "type": close_type,
            "position": int(pos.ticket),
            "price": float(price),
            "deviation": int(self.cfg.max_slippage_points),
            "magic": self.cfg.magic,
            "comment": f"ATLAS_X_{reason[:12]}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        # If ticket is order id not position id, find position
        positions = mt5.positions_get(symbol=symbol)
        pos_id = pos.ticket
        if positions:
            for p in positions:
                if int(p.magic) == self.cfg.magic:
                    pos_id = int(p.ticket)
                    break
        request["position"] = pos_id
        result = mt5.order_send(request)
        if result is None:
            return ExecResult(False, pos.ticket, 0.0, pos.volume, str(mt5.last_error()))
        ok = result.retcode == mt5.TRADE_RETCODE_DONE
        fill = float(result.price or price)
        pnl = 0.0
        if ok:
            self.log.trade(
                event="EXIT",
                symbol=symbol,
                direction=pos.side.value,
                volume=pos.volume,
                price=fill,
                sl=pos.sl,
                tp=pos.tp,
                ticket=pos_id,
                pnl=pnl,
                reason=reason,
                mode="LIVE",
            )
        return ExecResult(ok, pos_id, fill, pos.volume, reason)

    def current_position(self) -> OpenState | None:
        if self.cfg.is_paper or not self.feed._use_mt5:
            return self.paper_position
        if not MT5_OK or mt5 is None:
            return self.paper_position
        positions = mt5.positions_get(symbol=self.cfg.symbol)
        if not positions:
            self.paper_position = None
            return None
        for p in positions:
            if int(p.magic) != self.cfg.magic:
                continue
            side = Side.BUY if p.type == mt5.POSITION_TYPE_BUY else Side.SELL
            st = OpenState(
                side=side,
                entry=float(p.price_open),
                sl=float(p.sl),
                tp=float(p.tp),
                volume=float(p.volume),
                ticket=int(p.ticket),
            )
            self.paper_position = st
            return st
        return None
