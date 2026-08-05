"""
Manual M5 scanner watch — full analysis on every NEW closed M5 candle.

Analysis only. No auto trading. You confirm on TradingView and trade yourself.

Detection strategy (robust):
  1) Run one scan immediately on start (latest closed M5).
  2) Then fire primarily on wall-clock M5 boundaries (:00/:05/:10… UTC)
     with a short settle delay so MT5 can publish the new bar.
  3) Also fire if the closed-bar key advances even mid-wait (clock skew /
     broker time offset).
  4) Heartbeat shows wall UTC, next boundary, closed vs forming bar times,
     and warns if the MT5 feed looks stale.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from atlas.institutional.analyzer import InstitutionalAnalyzer
from atlas.institutional.config import InstitutionalConfig, load_institutional_config
from atlas.institutional.dashboard import render_dashboard
from atlas.live.watch import _next_m5_boundary_utc, last_closed_m5_key

logger = logging.getLogger(__name__)


def _bar_open_time(m5: pd.DataFrame | None, idx: int) -> str:
    if m5 is None or len(m5) == 0:
        return "?"
    try:
        if idx < 0 and len(m5) < abs(idx):
            return "?"
        return str(pd.Timestamp(m5["time"].iloc[idx]))
    except Exception:
        return "?"


def _parse_bar_ts(key: str | None) -> datetime | None:
    if not key or key == "?":
        return None
    try:
        ts = pd.Timestamp(key)
        if ts.tzinfo is None:
            return ts.to_pydatetime().replace(tzinfo=timezone.utc)
        return ts.to_pydatetime().astimezone(timezone.utc)
    except Exception:
        return None


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

    def _refresh_feed(self) -> None:
        """Nudge MT5 so copy_rates is not a stale terminal snapshot."""
        client = self.analyzer.client
        try:
            if getattr(client, "using_live_market_data", False):
                client.live_tick_snapshot(self.cfg.symbol)
        except Exception as exc:
            logger.debug("feed refresh skip: %s", exc)

    def _load_m5(self):
        self._refresh_feed()
        frames = self.analyzer.load_frames(self.cfg.symbol)
        m5 = frames.get("M5")
        closed = last_closed_m5_key(m5)
        forming = _bar_open_time(m5, -1)
        return frames, m5, closed, forming

    def _run_one(self, frames, m5_key: str, cycles: int) -> int:
        print(f"\n>>> M5 CLOSE {m5_key} — manual high-prob scan #{cycles}")
        print(f"    wall_utc={datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}")
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

        frames, m5, prev, forming = self._load_m5()
        if prev is None:
            self.analyzer.disconnect()
            raise RuntimeError("No M5 bars available from MT5")

        live = getattr(self.analyzer.client, "using_live_market_data", False)
        print(f"\nFeed    : {'MT5_LIVE' if live else 'NOT_LIVE / synthetic'}")
        print(f"Closed  : {prev}")
        print(f"Forming : {forming}")
        print("Running scan now…\n")
        cycles = self._run_one(frames, prev, 1)
        if max_cycles is not None and cycles >= max_cycles:
            self.analyzer.disconnect()
            return

        # After the (possibly slow) first analysis, resync closed-bar baseline
        # so we do not skip or double-fire bars that closed during the scan.
        frames, m5, cur_after, forming = self._load_m5()
        if cur_after and cur_after != prev:
            print(
                f"\nNote: M5 advanced during first scan "
                f"({prev} → {cur_after}). Catch-up scan…"
            )
            prev = cur_after
            cycles += 1
            cycles = self._run_one(frames, prev, cycles)
            if max_cycles is not None and cycles >= max_cycles:
                self.analyzer.disconnect()
                return

        next_boundary = _next_m5_boundary_utc()
        print("\nWatching for next NEW M5 close (heartbeat every 30s)…")
        print(
            f"Next wall-clock M5 boundary ≈ {next_boundary.strftime('%H:%M:%S')} UTC "
            "(+2s settle)\n"
        )
        last_beat = 0.0
        stale_warned = False
        try:
            while True:
                time.sleep(2.0)
                now = datetime.now(timezone.utc)
                frames, m5, cur, forming = self._load_m5()

                # A) Bar-key advance (works even if PC clock ≠ broker label)
                bar_advanced = cur is not None and cur != prev

                # B) Wall-clock boundary crossed → force scan after settle
                boundary_hit = now >= (next_boundary + timedelta(seconds=2))

                if not bar_advanced and not boundary_hit:
                    if time.time() - last_beat >= 30:
                        remain = max(0.0, (next_boundary - now).total_seconds())
                        print(
                            f"… watching | wall_utc={now.strftime('%H:%M:%S')} "
                            f"| next≈{next_boundary.strftime('%H:%M:%S')} "
                            f"| in {int(remain)}s "
                            f"| closed={cur or prev} | forming={forming}"
                        )
                        # Stale feed: closed key not moving for >7 minutes of wall time
                        closed_ts = _parse_bar_ts(cur or prev)
                        if closed_ts is not None:
                            # If bar times are labeled ahead of wall, compare deltas of keys only
                            age_wall = (now - closed_ts).total_seconds()
                            if age_wall > 7 * 60 and not stale_warned:
                                print(
                                    "⚠ MT5 closed-bar time looks behind wall clock by "
                                    f"{int(age_wall/60)}m — check MT5 Market Watch / "
                                    "AutoTrading connection. Will still fire on wall "
                                    "M5 boundaries."
                                )
                                stale_warned = True
                            elif age_wall < -90 * 60 and not stale_warned:
                                # Bar labeled far ahead of wall → broker time mislabeled as UTC
                                print(
                                    "⚠ Bar times look ahead of wall UTC "
                                    f"(closed={cur or prev}). Likely broker-server time "
                                    "labeled as UTC. Wall-clock boundaries still drive scans."
                                )
                                stale_warned = True
                        last_beat = time.time()
                    continue

                # Prefer the latest closed key; if feed is stale on a boundary,
                # still scan with whatever frames we have and tag with boundary time.
                scan_key = cur or prev or next_boundary.isoformat()
                if boundary_hit and not bar_advanced:
                    # One more refresh after settle
                    time.sleep(1.0)
                    frames, m5, cur2, forming = self._load_m5()
                    if cur2:
                        scan_key = cur2
                    print(
                        f"\nWall M5 boundary reached "
                        f"({next_boundary.strftime('%H:%M:%S')} UTC). "
                        f"closed_bar={scan_key} forming={forming}"
                    )
                else:
                    print(f"\nNew closed M5 detected: {scan_key} (forming={forming})")

                # Dedup: skip if we already scanned this exact closed key
                if scan_key == prev and boundary_hit and not bar_advanced:
                    # Boundary fired but same key — still analyze once (stale feed recovery)
                    pass
                elif scan_key == prev:
                    next_boundary = _next_m5_boundary_utc(now + timedelta(seconds=1))
                    continue

                prev = scan_key
                cycles += 1
                cycles = self._run_one(frames, scan_key, cycles)
                last_beat = time.time()
                stale_warned = False
                next_boundary = _next_m5_boundary_utc(
                    datetime.now(timezone.utc) + timedelta(seconds=1)
                )
                print(
                    f"\nNext wall-clock M5 boundary ≈ "
                    f"{next_boundary.strftime('%H:%M:%S')} UTC\n"
                )

                if max_cycles is not None and cycles >= max_cycles:
                    break
        except KeyboardInterrupt:
            print("\nManual scanner stopped by user")
        finally:
            self.analyzer.disconnect()
