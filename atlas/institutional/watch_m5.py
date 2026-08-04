"""
Institutional watch — full SMC analysis on every NEW closed M5 candle.

Same rules as scalp watch:
  - Do not flood mid-candle
  - Wait for next M5 boundary, then emit only when last closed M5 changes
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
from atlas.institutional.risk_manager import InstitutionalRiskManager
from atlas.live.watch import _next_m5_boundary_utc, last_closed_m5_key

logger = logging.getLogger(__name__)


def _journal_decision(cfg: InstitutionalConfig, m5_key: str, decision, text: str) -> None:
    try:
        log_dir = Path(cfg.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        day = datetime.now(timezone.utc).strftime("%Y%m%d")
        path = log_dir / f"m5_{day}.jsonl"
        sc = (decision.narrative.extras.get("scenario") if decision.narrative else None) or {}
        row = {
            "m5": m5_key,
            "as_of": datetime.now(timezone.utc).isoformat(),
            "action": decision.action.value,
            "probability": decision.probability,
            "confidence": decision.confidence,
            "confluence": decision.confluence,
            "edge": sc.get("edge_score"),
            "playbook": (decision.narrative.extras.get("playbook") if decision.narrative else None),
            "why": decision.why[:6],
        }
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        dash = log_dir / f"last_dashboard_{day}.txt"
        dash.write_text(text, encoding="utf-8")
    except Exception as exc:
        logger.debug("journal skip: %s", exc)


class InstitutionalWatch:
    def __init__(
        self,
        cfg: InstitutionalConfig | None = None,
        execute: bool = False,
    ) -> None:
        self.cfg = cfg or load_institutional_config()
        self.execute = execute
        self.analyzer = InstitutionalAnalyzer(self.cfg)
        self.risk = InstitutionalRiskManager(self.cfg)

    def start(self, max_cycles: int | None = None) -> None:
        print("=" * 72)
        print("  ATLAS INSTITUTIONAL M5-CLOSE ANALYST")
        print(f"  Symbol : {self.cfg.symbol} | Mode: {self.cfg.mode}")
        print(
            f"  Gates  : prob≥{self.cfg.min_probability} conf≥{self.cfg.min_confidence} "
            f"confl≥{self.cfg.min_confluence} playbook={'ON' if self.cfg.require_playbook else 'OFF'}"
        )
        print("  Cycle  : full institutional analysis ONLY on each new closed M5")
        print("  Style  : world SMC playbooks · AI checklist · prefer NO TRADE")
        print("=" * 72)

        if not self.analyzer.connect():
            raise RuntimeError("MT5 connection failed")

        # Wait for next M5 close — no mid-candle flood
        now = datetime.now(timezone.utc)
        nxt = _next_m5_boundary_utc(now)
        wait = max(0.0, (nxt - now).total_seconds())
        print(f"\nWaiting for next M5 close @ {nxt.isoformat()} ({wait:.0f}s)…\n")
        if wait > 0:
            time.sleep(min(wait + 1.0, 310.0))

        frames = self.analyzer.load_frames(self.cfg.symbol)
        prev = last_closed_m5_key(frames.get("M5"))
        print(f"Seeded last closed M5={prev}. Watching…\n")

        cycles = 0
        try:
            while True:
                time.sleep(3.0)
                frames = self.analyzer.load_frames(self.cfg.symbol)
                cur = last_closed_m5_key(frames.get("M5"))
                if cur is None or cur == prev:
                    continue

                prev = cur
                cycles += 1
                print(f"\n>>> NEW M5 CLOSE {cur} — institutional high-prob analysis #{cycles}")
                decision = self.analyzer.analyze(self.cfg.symbol, frames=frames)
                text = render_dashboard(decision)
                print(text)
                _journal_decision(self.cfg, str(cur), decision, text)

                if self.execute and decision.is_executable():
                    eq = 10000.0
                    try:
                        info = self.analyzer.client.account_info_dict()
                        eq = float(info.get("equity") or info.get("balance") or eq)
                    except Exception:
                        pass
                    ok, why = self.risk.allows_trade(eq, 0)
                    if not ok:
                        print(f"RISK_BLOCK {why}")
                    else:
                        print(
                            f"ARMED {decision.action.value} entry={decision.entry:.3f} "
                            f"SL={decision.stop:.3f} TP={decision.take_profit:.3f} "
                            f"(mode={self.cfg.mode})"
                        )
                        self.risk.register_trade()

                if max_cycles is not None and cycles >= max_cycles:
                    break
        except KeyboardInterrupt:
            print("\nInstitutional watch stopped by user")
        finally:
            self.analyzer.disconnect()
