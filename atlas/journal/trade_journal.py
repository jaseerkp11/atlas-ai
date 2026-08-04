"""
Trade journal — every closed/open/pending/rejected trade logged to CSV.

Columns: time, symbol, direction, entry, stop, target, exit, result,
R-multiple, score, reasoning — so performance can be reviewed by setup quality.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path

from atlas.config import ROOT, load_settings
from atlas.models import TradeRecord

logger = logging.getLogger(__name__)

JOURNAL_FIELDS = [
    "trade_id",
    "mode",
    "symbol",
    "direction",
    "status",
    "entry",
    "stop",
    "target",
    "exit",
    "volume",
    "score",
    "reasoning",
    "planned_rr",
    "r_multiple",
    "result",
    "opened_at",
    "filled_at",
    "closed_at",
    "mt5_ticket",
    "notes",
]


class TradeJournal:
    def __init__(self, path: Path | str | None = None) -> None:
        settings = load_settings()
        journal_dir = ROOT / settings.paths["journal_dir"]
        journal_dir.mkdir(parents=True, exist_ok=True)
        self.path = Path(path) if path else journal_dir / "trades.csv"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_header()

    def _ensure_header(self) -> None:
        if not self.path.exists() or self.path.stat().st_size == 0:
            with open(self.path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=JOURNAL_FIELDS)
                writer.writeheader()

    def log_trade(self, trade: TradeRecord) -> None:
        row = trade.to_journal_row()
        # Append (multiple status updates for same trade_id are intentional audit trail)
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=JOURNAL_FIELDS)
            writer.writerow(row)
        logger.debug("Journaled %s %s %s", trade.trade_id, trade.status.value, trade.symbol)

    def path_str(self) -> str:
        return str(self.path)
