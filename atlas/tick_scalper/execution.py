"""
Trade execution — MARKET entries/exits with robust MT5 close (fixes 10013).

Supports MULTIPLE concurrent positions. Resolves real position tickets after
entry (order ticket ≠ position ticket on many brokers).
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
        self._targets: dict[int, float] = {}

    def _symbol_info(self):
        assert mt5 is not None
        return mt5.symbol_info(self.cfg.symbol)

    def _filling_modes(self) -> list[int | None]:
        """
        Return filling modes to try. None = omit type_filling (some brokers).
        ORDER_FILLING_FOK=0, IOC=1, RETURN=2 in MetaTrader5 package.
        """
        assert mt5 is not None
        info = self._symbol_info()
        modes: list[int | None] = []
        if info is not None:
            fm = int(getattr(info, "filling_mode", 0) or 0)
            # Bit flags on symbol: 1=FOK, 2=IOC
            if fm & 1:
                modes.append(mt5.ORDER_FILLING_FOK)
            if fm & 2:
                modes.append(mt5.ORDER_FILLING_IOC)
        # Always try RETURN + omit — fixes many "Invalid request" closes
        for m in (
            mt5.ORDER_FILLING_RETURN,
            mt5.ORDER_FILLING_IOC,
            mt5.ORDER_FILLING_FOK,
            None,
        ):
            if m not in modes:
                modes.append(m)
        return modes

    def _normalize_volume(self, volume: float) -> float:
        info = self._symbol_info() if MT5_OK and mt5 is not None else None
        if info is None:
            return round(volume, 2)
        step = float(info.volume_step or 0.01)
        vmin = float(info.volume_min or 0.01)
        vmax = float(info.volume_max or 100.0)
        if step <= 0:
            step = 0.01
        steps = int(volume / step + 1e-9)
        vol = max(vmin, min(steps * step, vmax))
        # Digits from step
        s = f"{step:.10f}".rstrip("0")
        decimals = len(s.split(".")[1]) if "." in s else 0
        return round(vol, decimals)

    def _normalize_price(self, price: float) -> float:
        info = self._symbol_info() if MT5_OK and mt5 is not None else None
        digits = int(info.digits) if info else 2
        return round(price, digits)

    def _shrink_volume(self, volume: float) -> float:
        min_lot, step, _ = self.feed.symbol_volume_limits()
        nxt = volume - step
        if nxt + 1e-12 < min_lot:
            return 0.0
        return self._normalize_volume(nxt)

    def _account_free_margin(self) -> float:
        if not MT5_OK or mt5 is None:
            return 0.0
        info = mt5.account_info()
        return float(getattr(info, "margin_free", 0.0) or 0.0) if info else 0.0

    def _resolve_new_position_ticket(
        self, before_tickets: set[int], side: Side, fill: float
    ) -> int | None:
        """Poll briefly until the new MT5 position ticket appears."""
        assert mt5 is not None
        for _ in range(25):
            raw = mt5.positions_get(symbol=self.cfg.symbol)
            if raw:
                candidates = []
                for p in raw:
                    if int(p.magic) != self.cfg.magic:
                        continue
                    ticket = int(p.ticket)
                    if ticket in before_tickets:
                        continue
                    pside = Side.BUY if p.type == mt5.POSITION_TYPE_BUY else Side.SELL
                    if pside != side:
                        continue
                    candidates.append((abs(float(p.price_open) - fill), ticket))
                if candidates:
                    candidates.sort()
                    return candidates[0][1]
            time.sleep(0.02)
        return None

    def _find_live_position(self, pos: OpenState):
        """Return MT5 position object matching our OpenState (by ticket, then heuristics)."""
        assert mt5 is not None
        raw = mt5.positions_get(symbol=self.cfg.symbol)
        if not raw:
            return None
        # Exact ticket
        for p in raw:
            if int(p.ticket) == int(pos.ticket) and int(p.magic) == self.cfg.magic:
                return p
        # Fallback: same side + nearest entry + same volume
        want_type = mt5.POSITION_TYPE_BUY if pos.side == Side.BUY else mt5.POSITION_TYPE_SELL
        best = None
        best_d = 1e18
        for p in raw:
            if int(p.magic) != self.cfg.magic:
                continue
            if int(p.type) != want_type:
                continue
            d = abs(float(p.price_open) - pos.entry)
            if d < best_d:
                best_d = d
                best = p
        return best

    def open_market(
        self,
        signal: Signal,
        volume: float,
        sl: float,
        tp: float,
        profit_target_points: float | None = None,
    ) -> ExecResult:
        symbol = self.cfg.symbol
        volume = self._normalize_volume(volume)
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
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return ExecResult(False, 0, 0.0, 0.0, "no tick")
        price = self._normalize_price(float(tick.ask if signal.side == Side.BUY else tick.bid))
        sl = self._normalize_price(sl)
        tp = self._normalize_price(tp)
        deviation = max(10, int(self.cfg.max_slippage_points))
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
                    "deviation": deviation,
                    "magic": self.cfg.magic,
                    "comment": "ATLAS_TK",
                    "type_time": mt5.ORDER_TIME_GTC,
                }
                # Attach SL/TP (normalized). Some brokers prefer separate modify — keep in request.
                request["sl"] = float(sl)
                request["tp"] = float(tp)
                if filling is not None:
                    request["type_filling"] = filling

                check = mt5.order_check(request)
                if check is not None and int(check.retcode) not in (0, 10008, 10009):
                    last_comment = (
                        f"check retcode={check.retcode} {check.comment} "
                        f"vol={attempt_vol} free_margin={free:.2f}"
                    )
                    if int(check.retcode) == 10019 or "money" in str(check.comment).lower():
                        break
                    continue

                result = mt5.order_send(request)
                if result is None:
                    last_comment = f"order_send None {mt5.last_error()} vol={attempt_vol}"
                    continue
                if result.retcode == mt5.TRADE_RETCODE_DONE:
                    fill = float(result.price or price)
                    ticket = self._resolve_new_position_ticket(
                        before_tickets, signal.side, fill
                    )
                    if ticket is None:
                        ticket = int(result.order or result.deal or 0)
                    self._targets[ticket] = target
                    # Ensure SL/TP set on real position ticket
                    self._ensure_sl_tp(ticket, sl, tp)
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

    def _ensure_sl_tp(self, ticket: int, sl: float, tp: float) -> None:
        if not MT5_OK or mt5 is None or ticket <= 0:
            return
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "symbol": self.cfg.symbol,
            "position": int(ticket),
            "sl": float(self._normalize_price(sl)),
            "tp": float(self._normalize_price(tp)),
        }
        mt5.order_send(request)

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
        live = self._find_live_position(pos)
        if live is None:
            # Already closed by broker SL/TP
            self._targets.pop(pos.ticket, None)
            return ExecResult(True, pos.ticket, tick.mid, pos.volume, "already_closed")

        pos_id = int(live.ticket)
        volume = self._normalize_volume(float(live.volume))
        close_type = (
            mt5.ORDER_TYPE_SELL
            if int(live.type) == mt5.POSITION_TYPE_BUY
            else mt5.ORDER_TYPE_BUY
        )

        fresh = mt5.symbol_info_tick(symbol)
        if fresh is None:
            return ExecResult(False, pos_id, 0.0, volume, "no tick on close")
        raw_price = float(fresh.bid if close_type == mt5.ORDER_TYPE_SELL else fresh.ask)
        price = self._normalize_price(raw_price)
        deviation = max(50, int(self.cfg.max_slippage_points))

        last_comment = "close failed"
        for filling in self._filling_modes():
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": float(volume),
                "type": close_type,
                "position": pos_id,
                "price": float(price),
                "deviation": deviation,
                "magic": self.cfg.magic,
                "comment": "ATLAS_X",
                "type_time": mt5.ORDER_TIME_GTC,
            }
            if filling is not None:
                request["type_filling"] = filling

            # Validate first
            check = mt5.order_check(request)
            if check is not None and int(check.retcode) not in (0, 10008, 10009):
                last_comment = f"check retcode={check.retcode} {check.comment} pos={pos_id}"
                continue

            result = mt5.order_send(request)
            if result is None:
                last_comment = f"None {mt5.last_error()} pos={pos_id}"
                continue
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                fill = float(result.price or price)
                self.log.trade(
                    event="EXIT",
                    symbol=symbol,
                    direction=pos.side.value,
                    volume=volume,
                    price=fill,
                    sl=pos.sl,
                    tp=pos.tp,
                    ticket=pos_id,
                    pnl=0.0,
                    reason=reason,
                    mode="LIVE",
                )
                self._targets.pop(pos_id, None)
                self._targets.pop(pos.ticket, None)
                return ExecResult(True, pos_id, fill, volume, reason)
            last_comment = f"retcode={result.retcode} {result.comment} pos={pos_id} fill={filling}"

        return ExecResult(False, pos.ticket, 0.0, volume, last_comment)

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
        pos = self.current_positions()
        return pos[0] if pos else None
