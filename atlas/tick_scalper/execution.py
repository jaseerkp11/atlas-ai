"""
Trade execution — MARKET entries/exits only.

Supports MULTIPLE concurrent positions (same magic).
No pending grids, no averaging into losers (enforced by engine pyramid rules).
"""

from __future__ import annotations

import time
from dataclasses import dataclass

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
        self.paper_positions: list[OpenState] = []
        self._paper_ticket = 900000
        # Remember ladder targets by ticket (LIVE sync loses custom fields)
        self._targets: dict[int, float] = {}

    def _filling_modes(self) -> list[int]:
        assert mt5 is not None
        info = mt5.symbol_info(self.cfg.symbol)
        modes: list[int] = []
        if info is not None:
            fm = int(getattr(info, "filling_mode", 0) or 0)
            if fm & 1:
                modes.append(mt5.ORDER_FILLING_FOK)
            if fm & 2:
                modes.append(mt5.ORDER_FILLING_IOC)
            if fm & 4:
                modes.append(mt5.ORDER_FILLING_RETURN)
        if not modes:
            modes = [
                mt5.ORDER_FILLING_IOC,
                mt5.ORDER_FILLING_FOK,
                mt5.ORDER_FILLING_RETURN,
            ]
        ordered: list[int] = []
        for m in [mt5.ORDER_FILLING_IOC, mt5.ORDER_FILLING_FOK, mt5.ORDER_FILLING_RETURN] + modes:
            if m not in ordered and m in modes:
                ordered.append(m)
        return ordered or [mt5.ORDER_FILLING_IOC]

    def _shrink_volume(self, volume: float) -> float:
        min_lot, step, _ = self.feed.symbol_volume_limits()
        nxt = volume - step
        if nxt + 1e-12 < min_lot:
            return 0.0
        decimals = max(0, len(str(step).rstrip("0").split(".")[-1]) if "." in str(step) else 0)
        return round(max(min_lot, nxt), decimals or 2)

    def _account_free_margin(self) -> float:
        if not MT5_OK or mt5 is None:
            return 0.0
        info = mt5.account_info()
        return float(getattr(info, "margin_free", 0.0) or 0.0) if info else 0.0

    def open_market(
        self,
        signal: Signal,
        volume: float,
        sl: float,
        tp: float,
        profit_target_points: float | None = None,
    ) -> ExecResult:
        symbol = self.cfg.symbol
        if volume <= 0:
            return ExecResult(False, 0, 0.0, 0.0, "volume<=0")
        target = float(
            profit_target_points
            if profit_target_points is not None
            else self.cfg.instant_profit_points
        )

        if self.cfg.is_paper or not self.feed._use_mt5 or not MT5_OK:
            price = signal.ask if signal.side == Side.BUY else signal.bid
            self._paper_ticket += 1
            ticket = self._paper_ticket
            pos = OpenState(
                side=signal.side,
                entry=price,
                sl=sl,
                tp=tp,
                volume=volume,
                ticket=ticket,
                opened_msc=int(time.time() * 1000),
                profit_target_points=target,
            )
            self.paper_positions.append(pos)
            self._targets[ticket] = target
            self.log.trade(
                event="ENTRY",
                symbol=symbol,
                direction=signal.side.value,
                volume=volume,
                price=price,
                sl=sl,
                tp=tp,
                ticket=ticket,
                reason=f"{signal.reason} | target=+{target}pts",
                mode="PAPER",
            )
            return ExecResult(True, ticket, price, volume, "PAPER_FILL")

        assert mt5 is not None
        order_type = mt5.ORDER_TYPE_BUY if signal.side == Side.BUY else mt5.ORDER_TYPE_SELL
        price = signal.ask if signal.side == Side.BUY else signal.bid
        deviation = int(self.cfg.max_slippage_points)
        free = self._account_free_margin()
        before_tickets = {p.ticket for p in self.current_positions()}

        attempt_vol = float(volume)
        last_comment = "no attempt"
        while attempt_vol > 0:
            for filling in self._filling_modes():
                request = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": symbol,
                    "volume": float(attempt_vol),
                    "type": order_type,
                    "price": float(price),
                    "sl": float(sl),
                    "tp": float(tp),
                    "deviation": deviation,
                    "magic": self.cfg.magic,
                    "comment": "ATLAS_TICK",
                    "type_time": mt5.ORDER_TIME_GTC,
                    "type_filling": filling,
                }
                check = mt5.order_check(request)
                if check is not None and check.retcode not in (
                    0,
                    mt5.TRADE_RETCODE_DONE,
                    10009,
                ):
                    last_comment = (
                        f"check retcode={check.retcode} {check.comment} "
                        f"vol={attempt_vol} free_margin={free:.2f}"
                    )
                    if check.retcode == 10019 or "money" in str(check.comment).lower():
                        break
                    continue

                result = mt5.order_send(request)
                if result is None:
                    last_comment = f"order_send None {mt5.last_error()} vol={attempt_vol}"
                    continue
                if result.retcode == mt5.TRADE_RETCODE_DONE:
                    fill = float(result.price or price)
                    ticket = int(result.order or result.deal or 0)
                    # Prefer newly appeared position ticket
                    after = self.current_positions()
                    for p in after:
                        if p.ticket not in before_tickets:
                            ticket = p.ticket
                            break
                    self._targets[ticket] = target
                    self.log.trade(
                        event="ENTRY",
                        symbol=symbol,
                        direction=signal.side.value,
                        volume=attempt_vol,
                        price=fill,
                        sl=sl,
                        tp=tp,
                        ticket=ticket,
                        reason=f"{signal.reason} | target=+{target}pts",
                        mode="LIVE",
                    )
                    return ExecResult(
                        True, ticket, fill, float(result.volume or attempt_vol), result.comment
                    )

                last_comment = (
                    f"retcode={result.retcode} {result.comment} "
                    f"vol={attempt_vol} free_margin={free:.2f}"
                )
                if result.retcode == 10019 or "money" in str(result.comment).lower():
                    break
            nxt = self._shrink_volume(attempt_vol)
            if nxt <= 0 or nxt >= attempt_vol:
                break
            attempt_vol = nxt

        return ExecResult(False, 0, 0.0, volume, last_comment)

    def close_market(self, pos: OpenState, tick: Tick, reason: str) -> ExecResult:
        symbol = self.cfg.symbol
        if self.cfg.is_paper or not self.feed._use_mt5 or not MT5_OK:
            price = tick.bid if pos.side == Side.BUY else tick.ask
            pnl_points = (
                (price - pos.entry) if pos.side == Side.BUY else (pos.entry - price)
            ) / self.feed.symbol_point()
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
            self.paper_positions = [p for p in self.paper_positions if p.ticket != pos.ticket]
            self._targets.pop(pos.ticket, None)
            return ExecResult(True, pos.ticket, price, pos.volume, reason)

        assert mt5 is not None
        close_type = mt5.ORDER_TYPE_SELL if pos.side == Side.BUY else mt5.ORDER_TYPE_BUY
        price = tick.bid if pos.side == Side.BUY else tick.ask
        pos_id = int(pos.ticket)

        last_comment = "close failed"
        for filling in self._filling_modes():
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": float(pos.volume),
                "type": close_type,
                "position": pos_id,
                "price": float(price),
                "deviation": int(self.cfg.max_slippage_points),
                "magic": self.cfg.magic,
                "comment": f"ATLAS_X_{reason[:12]}",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": filling,
            }
            result = mt5.order_send(request)
            if result is None:
                last_comment = str(mt5.last_error())
                continue
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                fill = float(result.price or price)
                self.log.trade(
                    event="EXIT",
                    symbol=symbol,
                    direction=pos.side.value,
                    volume=pos.volume,
                    price=fill,
                    sl=pos.sl,
                    tp=pos.tp,
                    ticket=pos_id,
                    pnl=0.0,
                    reason=reason,
                    mode="LIVE",
                )
                self._targets.pop(pos_id, None)
                return ExecResult(True, pos_id, fill, pos.volume, reason)
            last_comment = f"retcode={result.retcode} {result.comment}"

        return ExecResult(False, pos.ticket, 0.0, pos.volume, last_comment)

    def current_positions(self) -> list[OpenState]:
        if self.cfg.is_paper or not self.feed._use_mt5 or not MT5_OK or mt5 is None:
            return list(self.paper_positions)

        raw = mt5.positions_get(symbol=self.cfg.symbol)
        if not raw:
            return []
        out: list[OpenState] = []
        for p in raw:
            if int(p.magic) != self.cfg.magic:
                continue
            ticket = int(p.ticket)
            side = Side.BUY if p.type == mt5.POSITION_TYPE_BUY else Side.SELL
            out.append(
                OpenState(
                    side=side,
                    entry=float(p.price_open),
                    sl=float(p.sl),
                    tp=float(p.tp),
                    volume=float(p.volume),
                    ticket=ticket,
                    profit_target_points=self._targets.get(
                        ticket, self.cfg.instant_profit_points
                    ),
                )
            )
        return out

    def current_position(self) -> OpenState | None:
        """Back-compat: first open position or None."""
        pos = self.current_positions()
        return pos[0] if pos else None
