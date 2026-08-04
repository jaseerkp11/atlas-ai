"""
Institutional watch — full SMC analysis on every NEW closed M5 candle.

Same rules as scalp watch:
  - Do not flood mid-candle
  - Wait for next M5 boundary, then emit only when last closed M5 changes
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone

from atlas.institutional.analyzer import InstitutionalAnalyzer
from atlas.institutional.config import InstitutionalConfig, load_institutional_config
from atlas.institutional.dashboard import render_dashboard
from atlas.institutional.risk_manager import InstitutionalRiskManager
from atlas.live.watch import _next_m5_boundary_utc, last_closed_m5_key

logger = logging.getLogger(__name__)


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
            f"confl≥{self.cfg.min_confluence}"
        )
        print("  Cycle  : analyze ONLY on each new closed M5")
        print("  Style  : high-probability playbooks · prefer NO TRADE")
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
                print(f"\n>>> NEW M5 CLOSE {cur} — running institutional analysis #{cycles}")
                decision = self.analyzer.analyze(self.cfg.symbol, frames=frames)
                print(render_dashboard(decision))

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
