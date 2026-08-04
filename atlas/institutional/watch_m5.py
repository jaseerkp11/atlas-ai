"""
Manual M5 scanner watch — full analysis on every NEW closed M5 candle.

Analysis only. No auto trading. You confirm on TradingView and trade yourself.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

from atlas.institutional.analyzer import InstitutionalAnalyzer
from atlas.institutional.config import InstitutionalConfig, load_institutional_config
from atlas.institutional.dashboard import render_dashboard
from atlas.live.watch import last_closed_m5_key

logger = logging.getLogger(__name__)


def _journal_decision(cfg: InstitutionalConfig, m5_key: str, decision, text: str) -> None:
    try:
        log_dir = Path(cfg.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        day = datetime.now(timezone.utc).strftime("%Y%m%d")
        path = log_dir / f"m5_{day}.jsonl"
        n = decision.narrative
        row = {
            "m5": m5_key,
            "as_of": datetime.now(timezone.utc).isoformat(),
            "setup_status": decision.action.value,
            "probability": decision.probability,
            "confidence": decision.confidence,
            "confluence": decision.confluence,
            "manual_stance": (n.extras.get("manual_stance") if n else None),
            "manual_scan": (n.extras.get("manual_scan_summary") if n else None),
            "playbook": (n.extras.get("playbook") if n else None),
        }
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        (log_dir / f"last_dashboard_{day}.txt").write_text(text, encoding="utf-8")
    except Exception as exc:
        logger.debug("journal skip: %s", exc)


class InstitutionalWatch:
    def __init__(
        self,
        cfg: InstitutionalConfig | None = None,
        execute: bool = False,
    ) -> None:
        self.cfg = cfg or load_institutional_config()
        if execute:
            print(
                "NOTE: --execute ignored. This build is a MANUAL scanner only "
                "(no auto orders). Confirm on TradingView and trade yourself."
            )
        self.analyzer = InstitutionalAnalyzer(self.cfg)

    def _run_one(self, frames, m5_key: str, cycles: int) -> int:
        print(f"\n>>> M5 CLOSE {m5_key} — manual high-prob scan #{cycles}")
        decision = self.analyzer.analyze(self.cfg.symbol, frames=frames)
        text = render_dashboard(decision)
        print(text)
        _journal_decision(self.cfg, str(m5_key), decision, text)
        return cycles

    def start(self, max_cycles: int | None = None) -> None:
        print("=" * 72)
        print("  ATLAS MANUAL HIGH-PROBABILITY M5 SCANNER")
        print(f"  Symbol : {self.cfg.symbol} | Mode: {self.cfg.mode} (analysis only)")
        print("  Output : S/R + graded BUY/SELL areas + IF/THEN triggers")
        print("  Cycle  : scan now, then again on each NEW closed M5")
        print("  Trade  : YOU decide on TradingView — bot never sends orders")
        print("=" * 72)

        if not self.analyzer.connect():
            raise RuntimeError("MT5 connection failed")

        frames = self.analyzer.load_frames(self.cfg.symbol)
        prev = last_closed_m5_key(frames.get("M5"))
        if prev is None:
            self.analyzer.disconnect()
            raise RuntimeError("No M5 bars available from MT5")

        # IMPORTANT: run immediately so the user is not left staring at "Scanning…"
        # Old bug: wait for boundary → seed that close as prev → first print only 5 min later.
        print(f"\nLast closed M5={prev} (broker bar time). Running scan now…\n")
        cycles = self._run_one(frames, prev, 1)
        if max_cycles is not None and cycles >= max_cycles:
            self.analyzer.disconnect()
            return

        print("\nWatching for next NEW M5 close (heartbeat every 30s)…\n")
        last_beat = time.time()
        try:
            while True:
                time.sleep(3.0)
                frames = self.analyzer.load_frames(self.cfg.symbol)
                cur = last_closed_m5_key(frames.get("M5"))
                now = datetime.now(timezone.utc)

                if cur is None or cur == prev:
                    if time.time() - last_beat >= 30:
                        print(
                            f"… watching | last_M5={prev} | utc={now.strftime('%H:%M:%S')} "
                            f"| waiting for next closed M5"
                        )
                        last_beat = time.time()
                    continue

                prev = cur
                cycles += 1
                cycles = self._run_one(frames, cur, cycles)
                last_beat = time.time()

                if max_cycles is not None and cycles >= max_cycles:
                    break
        except KeyboardInterrupt:
            print("\nManual scanner stopped by user")
        finally:
            self.analyzer.disconnect()
