from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

from motion_bot.config import BotConfig
from motion_bot.post_generator import PostGenerator
from motion_bot.x_client import XClient


class MotionBot:
    def __init__(self, config: BotConfig) -> None:
        self.config = config
        self.generator = PostGenerator(config)
        self.x = XClient(config)
        self.config.data_dir.mkdir(parents=True, exist_ok=True)
        self.last_post_path = self.config.data_dir / "last_post.json"

    def startup(self) -> dict:
        self.x.connect()
        identity = self.x.verify()
        mode = "DRY-RUN" if self.config.dry_run else "LIVE"
        print("=" * 56)
        print("Motion X Engagement Bot")
        print("=" * 56)
        print(f"Mode        : {mode}")
        print(f"Interval    : {self.config.interval_seconds}s")
        print(f"Account     : @{identity.get('username')}")
        print(f"Required    : {', '.join(self.config.required_tags)}")
        print(f"Site        : {self.config.motion_url}")
        print("=" * 56)
        return identity

    def _record_last(self, payload: dict) -> None:
        self.last_post_path.write_text(
            json.dumps(payload, indent=2),
            encoding="utf-8",
        )

    def post_once(self) -> bool:
        generated = self.generator.generate()
        print("\n--- Generated post ---")
        print(generated.text)
        print(f"(kind={generated.kind}, chars={len(generated.text)})")

        result = self.x.create_post(generated.text)
        now = datetime.now(timezone.utc).isoformat()
        record = {
            "at": now,
            "kind": generated.kind,
            "fingerprint": generated.fingerprint,
            "dry_run": result.dry_run,
            "ok": result.ok,
            "tweet_id": result.tweet_id,
            "error": result.error,
            "text": generated.text,
        }
        self._record_last(record)

        if not result.ok:
            print(f"POST FAILED: {result.error}")
            return False

        if result.dry_run:
            print("DRY-RUN: not published to X")
        else:
            print(f"Published tweet id={result.tweet_id}")
        return True

    def run_forever(self) -> None:
        self.startup()
        print("Starting loop. Ctrl+C to stop.")
        while True:
            started = time.monotonic()
            try:
                self.post_once()
            except Exception as exc:  # noqa: BLE001
                print(f"Cycle error: {exc}")
            elapsed = time.monotonic() - started
            sleep_for = max(1.0, self.config.interval_seconds - elapsed)
            print(f"Sleeping {sleep_for:.1f}s ...")
            time.sleep(sleep_for)
