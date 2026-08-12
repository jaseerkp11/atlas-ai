"""
Buffered, low-overhead logger for the tick scalper.

Writes:
  - events.log      human-readable signal/entry/exit lines
  - ticks.csv       sampled (or every) ticks
  - trades.csv      entries / exits / modifications
"""

from __future__ import annotations

import csv
import logging
from datetime import datetime, timezone
from pathlib import Path

from atlas.tick_scalper.config import ROOT, load_tick_config

logger = logging.getLogger("tick_scalper")

TICK_FIELDS = ["time_utc", "bid", "ask", "spread_points", "last", "volume"]
TRADE_FIELDS = [
    "time_utc",
    "event",
    "symbol",
    "direction",
    "volume",
    "price",
    "sl",
    "tp",
    "ticket",
    "pnl",
    "reason",
    "mode",
]


class TickLogger:
    def __init__(self) -> None:
        cfg = load_tick_config()
        self.dir = ROOT / cfg.log_dir
        self.dir.mkdir(parents=True, exist_ok=True)
        self.ticks_path = self.dir / "ticks.csv"
        self.trades_path = self.dir / "trades.csv"
        self.events_path = self.dir / "events.log"
        self._ensure(self.ticks_path, TICK_FIELDS)
        self._ensure(self.trades_path, TRADE_FIELDS)
        self._tick_buf: list[dict] = []
        self._buf_flush_n = 64

        fh = logging.FileHandler(self.events_path, encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
        logger.setLevel(logging.INFO)
        if not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
            logger.addHandler(fh)
        if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
            sh = logging.StreamHandler()
            sh.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
            logger.addHandler(sh)

    def _ensure(self, path: Path, fields: list[str]) -> None:
        if not path.exists() or path.stat().st_size == 0:
            with open(path, "w", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=fields).writeheader()

    def event(self, msg: str) -> None:
        logger.info(msg)

    def tick(
        self,
        bid: float,
        ask: float,
        spread_points: float,
        last: float = 0.0,
        volume: int = 0,
    ) -> None:
        self._tick_buf.append(
            {
                "time_utc": datetime.now(timezone.utc).isoformat(),
                "bid": bid,
                "ask": ask,
                "spread_points": spread_points,
                "last": last,
                "volume": volume,
            }
        )
        if len(self._tick_buf) >= self._buf_flush_n:
            self.flush_ticks()

    def flush_ticks(self) -> None:
        if not self._tick_buf:
            return
        with open(self.ticks_path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=TICK_FIELDS)
            w.writerows(self._tick_buf)
        self._tick_buf.clear()

    def trade(self, **kwargs) -> None:
        self.flush_ticks()
        row = {k: kwargs.get(k, "") for k in TRADE_FIELDS}
        if not row["time_utc"]:
            row["time_utc"] = datetime.now(timezone.utc).isoformat()
        with open(self.trades_path, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=TRADE_FIELDS).writerow(row)
        self.event(
            f"TRADE {row['event']} {row['direction']} @{row['price']} "
            f"vol={row['volume']} ticket={row['ticket']} reason={row['reason']}"
        )

    def close(self) -> None:
        self.flush_ticks()
