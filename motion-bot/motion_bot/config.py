from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parent.parent


@dataclass
class BotConfig:
    interval_seconds: int = 60
    max_post_length: int = 280
    dry_run: bool = True
    required_tags: list[str] = field(
        default_factory=lambda: ["$MOTION", "@Motiontip", "@tipmotion"]
    )
    motion_url: str = "https://motion.tips"
    tip_bot: str = "@tipmotion"
    official_account: str = "@Motiontip"
    tier_tags: dict[str, list[str]] = field(default_factory=dict)
    howto_every_n_posts: int = 5
    history_size: int = 500
    data_dir: Path = field(default_factory=lambda: ROOT / "data")

    x_api_key: str = ""
    x_api_secret: str = ""
    x_access_token: str = ""
    x_access_token_secret: str = ""
    x_bearer_token: str = ""

    @property
    def has_x_credentials(self) -> bool:
        return all(
            [
                self.x_api_key,
                self.x_api_secret,
                self.x_access_token,
                self.x_access_token_secret,
            ]
        )


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def load_config(
    config_path: Path | None = None,
    env_path: Path | None = None,
) -> BotConfig:
    load_dotenv(env_path or ROOT / ".env")

    path = config_path or ROOT / "config.yaml"
    raw: dict[str, Any] = {}
    if path.exists():
        with path.open(encoding="utf-8") as fh:
            loaded = yaml.safe_load(fh) or {}
            if not isinstance(loaded, dict):
                raise ValueError(f"Config must be a mapping: {path}")
            raw = loaded

    data_dir = Path(raw.get("data_dir", "data"))
    if not data_dir.is_absolute():
        data_dir = ROOT / data_dir

    interval = int(
        os.getenv("INTERVAL_SECONDS", raw.get("interval_seconds", 60))
    )
    dry_run = _as_bool(os.getenv("DRY_RUN"), _as_bool(raw.get("dry_run"), True))

    return BotConfig(
        interval_seconds=max(15, interval),
        max_post_length=int(raw.get("max_post_length", 280)),
        dry_run=dry_run,
        required_tags=list(raw.get("required_tags") or ["$MOTION", "@Motiontip", "@tipmotion"]),
        motion_url=str(raw.get("motion_url", "https://motion.tips")),
        tip_bot=str(raw.get("tip_bot", "@tipmotion")),
        official_account=str(raw.get("official_account", "@Motiontip")),
        tier_tags=dict(raw.get("tier_tags") or {}),
        howto_every_n_posts=max(1, int(raw.get("howto_every_n_posts", 5))),
        history_size=max(50, int(raw.get("history_size", 500))),
        data_dir=data_dir,
        x_api_key=os.getenv("X_API_KEY", "").strip(),
        x_api_secret=os.getenv("X_API_SECRET", "").strip(),
        x_access_token=os.getenv("X_ACCESS_TOKEN", "").strip(),
        x_access_token_secret=os.getenv("X_ACCESS_TOKEN_SECRET", "").strip(),
        x_bearer_token=os.getenv("X_BEARER_TOKEN", "").strip(),
    )
