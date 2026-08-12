#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from motion_bot.config import load_config
from motion_bot.scheduler import MotionBot


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Motion X engagement bot — unique $MOTION posts on a timer"
    )
    parser.add_argument(
        "command",
        choices=["once", "run", "verify"],
        help="once=single post, run=loop, verify=check X auth",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Force dry-run (do not publish)",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Force live publishing (overrides DRY_RUN)",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=None,
        help="Seconds between posts (default from config/env)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to config.yaml",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(config_path=args.config)

    if args.dry_run and args.live:
        print("Choose only one of --dry-run or --live")
        return 2
    if args.dry_run:
        config.dry_run = True
    if args.live:
        config.dry_run = False
    if args.interval is not None:
        config.interval_seconds = max(15, args.interval)

    bot = MotionBot(config)

    if args.command == "verify":
        bot.startup()
        print("X connection OK")
        return 0

    if args.command == "once":
        bot.startup()
        ok = bot.post_once()
        return 0 if ok else 1

    if args.command == "run":
        try:
            bot.run_forever()
        except KeyboardInterrupt:
            print("\nStopped.")
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
