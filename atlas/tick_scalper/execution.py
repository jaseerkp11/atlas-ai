"""
Trade execution — MARKET entries/exits only. No pending grids, no averaging.

LIVE path:
  - order_check + auto-shrink volume on margin failure
  - try broker-supported filling modes (IOC / FOK / RETURN)
  - rich failure comments (retcode, volume, free margin)
"""

from __future__ import annotations

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
        self.paper_position: OpenState | None = None
        self._paper_ticket = 900000

    def _filling_modes(self) -> list[int]:
        assert mt5 is not None
        info = mt5.symbol_info(self.cfg.symbol)
        modes: list[int] = []
        if info is not None:
            fm = int(getattr(info, "filling_mode", 0) or 0)
            # SYMBOL_FILLING_* bit flags
            if fm & 1:  # FOK
                modes.append(mt5.ORDER_FILLING_FOK)
            if fm & 2:  # IOC
                modes.append(mt5.ORDER_FILLING_IOC)
            if fm & 4:  # RETURN
                modes.append(mt5.ORDER_FILLING_RETURN)
        if not modes:
            modes = [
                mt5.ORDER_FILLING_IOC,
                mt5.ORDER_FILLING_FOK,
                mt5.ORDER_FILLING_RETURN,
            ]
        # Prefer IOC first for scalping, then others unique
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
        free = self._account_free_margin()

        # Auto-reduce volume until order_check passes or min lot fails
        attempt_vol = float(volume)
        last_comment = "no attempt"
        while attempt_vol > 0:
            filled = False
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
                    10009,  # done / placed variants on some builds
                ):
                    # 10019 = no money
                    last_comment = (
                        f"check retcode={check.retcode} {check.comment} "
                        f"vol={attempt_vol} free_margin={free:.2f}"
                    )
                    if check.retcode == 10019 or "money" in str(check.comment).lower():
                        break  # shrink volume
                    # Try next filling mode for unsupported filling etc.
                    continue

                result = mt5.order_send(request)
                if result is None:
                    last_comment = f"order_send None {mt5.last_error()} vol={attempt_vol}"
                    continue
                if result.retcode == mt5.TRADE_RETCODE_DONE:
                    fill = float(result.price or price)
                    ticket = int(result.order or result.deal or 0)
                    # Prefer position ticket if available
                    positions = mt5.positions_get(symbol=symbol)
                    if positions:
                        for p in positions:
                            if int(p.magic) == self.cfg.magic:
                                ticket = int(p.ticket)
                                break
                    self.log.trade(
                        event="ENTRY",
                        symbol=symbol,
                        direction=signal.side.value,
                        volume=attempt_vol,
                        price=fill,
                        sl=sl,
                        tp=tp,
                        ticket=ticket,
                        reason=signal.reason,
                        mode="LIVE",
                    )
                    return ExecResult(True, ticket, fill, float(result.volume or attempt_vol), result.comment)

                last_comment = (
                    f"retcode={result.retcode} {result.comment} "
                    f"vol={attempt_vol} free_margin={free:.2f}"
                )
                if result.retcode == 10019 or "money" in str(result.comment).lower():
                    break  # shrink
                # else try next filling
            # Shrink and retry
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
            self.paper_position = None
            return ExecResult(True, pos.ticket, price, pos.volume, reason)

        assert mt5 is not None
        close_type = mt5.ORDER_TYPE_SELL if pos.side == Side.BUY else mt5.ORDER_TYPE_BUY
        price = tick.bid if pos.side == Side.BUY else tick.ask
        positions = mt5.positions_get(symbol=symbol)
        pos_id = pos.ticket
        if positions:
            for p in positions:
                if int(p.magic) == self.cfg.magic:
                    pos_id = int(p.ticket)
                    break

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
                return ExecResult(True, pos_id, fill, pos.volume, reason)
            last_comment = f"retcode={result.retcode} {result.comment}"

        return ExecResult(False, pos.ticket, 0.0, pos.volume, last_comment)

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
