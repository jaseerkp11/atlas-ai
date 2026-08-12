from __future__ import annotations

import hashlib
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path

from motion_bot.config import BotConfig

HOOKS = [
    "Keep the timeline moving with $MOTION",
    "Create motion. Earn motion. Tip motion.",
    "TipFi energy on the timeline — $MOTION",
    "Support builders where it counts: $MOTION",
    "Allocation hits different when the community tips",
    "Graph is watching. Make every post count for $MOTION",
    "Burn cycle energy — stay loud with $MOTION",
    "Builders tip builders. That's the $MOTION loop",
    "No wallet setup needed — your X is the account for $MOTION",
    "Post. Get scored. Tip the ones who move culture — $MOTION",
    "Community compounding > solo flex. $MOTION",
    "Small tips, big signal. That's $MOTION",
]

ANGLES = [
    "tag a creator you respect and keep the flow going",
    "find a post that made you stop scrolling and tip it",
    "reply to a builder thread and leave motion behind",
    "boost someone early before the crowd piles in",
    "stack goodwill while the cycle window is open",
    "turn timeline noise into treasury burn signal",
    "show up for the people shipping in public",
    "leave value, not just vibes",
]

CTAS = [
    "Got allocation? Spread it.",
    "Let's support each other and keep the community moving",
    "Check your balance and tip someone real",
    "If you got scored this cycle, pay it forward",
    "Motion sees the graph — make your node count",
    "Stay in motion. Tip in motion.",
]

EMOJIS = ["👀", "🤝", "🔥", "⚡", "💫", "🚀", "✨", "🌀"]


HOWTO_TEMPLATE = """How to tip $MOTION {emoji}

1. Go to {url}
2. Connect your X account
3. Check your Balance
4. Find a builder, creator, or post you like
5. Reply with
{tip_bot}
 1-1000

Got allocation? Let's support each other and keep the community moving 🤝"""


@dataclass
class GeneratedPost:
    text: str
    kind: str
    fingerprint: str


class PostGenerator:
    """Build unique engaging Motion posts under the X character limit."""

    def __init__(self, config: BotConfig) -> None:
        self.config = config
        self.data_dir = config.data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.history_path = self.data_dir / "posted_hashes.json"
        self.counter_path = self.data_dir / "post_counter.json"
        self._history = self._load_history()
        self._counter = self._load_counter()

    def _load_history(self) -> list[str]:
        if not self.history_path.exists():
            return []
        try:
            data = json.loads(self.history_path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return [str(x) for x in data][-self.config.history_size :]
        except (json.JSONDecodeError, OSError):
            return []
        return []

    def _load_counter(self) -> int:
        if not self.counter_path.exists():
            return 0
        try:
            data = json.loads(self.counter_path.read_text(encoding="utf-8"))
            return int(data.get("count", 0))
        except (json.JSONDecodeError, OSError, TypeError, ValueError):
            return 0

    def _save_state(self) -> None:
        self.history_path.write_text(
            json.dumps(self._history[-self.config.history_size :], indent=2),
            encoding="utf-8",
        )
        self.counter_path.write_text(
            json.dumps({"count": self._counter, "updated_at": int(time.time())}),
            encoding="utf-8",
        )

    @staticmethod
    def fingerprint(text: str) -> str:
        normalized = " ".join(text.lower().split())
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    def _pick_tier_mentions(self) -> list[str]:
        mentions: list[str] = []
        for _tier, handles in self.config.tier_tags.items():
            if not handles:
                continue
            mentions.append(random.choice(handles))
        # Deduplicate while preserving order
        seen: set[str] = set()
        unique: list[str] = []
        for handle in mentions:
            key = handle.lower()
            if key not in seen:
                seen.add(key)
                unique.append(handle)
        return unique

    def _ensure_required(self, text: str) -> str:
        missing = [
            tag
            for tag in self.config.required_tags
            if tag.lower() not in text.lower()
        ]
        if not missing:
            return text
        suffix = " " + " ".join(missing)
        combined = (text.rstrip() + suffix).strip()
        if len(combined) <= self.config.max_post_length:
            return combined
        # Drop optional fluff by truncating body before required tags
        room = self.config.max_post_length - len(suffix) - 1
        return (text[:room].rstrip() + suffix).strip()

    def _howto_post(self) -> str:
        body = HOWTO_TEMPLATE.format(
            emoji=random.choice(EMOJIS),
            url=self.config.motion_url,
            tip_bot=self.config.tip_bot,
        )
        # Compact if over limit while keeping required signals
        if len(body) <= self.config.max_post_length:
            return self._ensure_required(body)

        compact = (
            f"How to tip $MOTION {random.choice(EMOJIS)}\n"
            f"1) {self.config.motion_url}\n"
            f"2) Connect X\n"
            f"3) Check Balance\n"
            f"4) Pick a builder/post\n"
            f"5) Reply {self.config.tip_bot} 1-1000\n"
            f"Support each other 🤝 {self.config.official_account}"
        )
        return self._ensure_required(compact)

    def _engagement_post(self) -> str:
        hook = random.choice(HOOKS)
        angle = random.choice(ANGLES)
        cta = random.choice(CTAS)
        emoji = random.choice(EMOJIS)
        tiers = self._pick_tier_mentions()
        tier_line = (" ".join(tiers) + "\n") if tiers else ""
        nonce = random.randint(1000, 9999)

        candidates = [
            (
                f"{hook} {emoji}\n"
                f"{angle.capitalize()}.\n"
                f"{cta}\n"
                f"{tier_line}"
                f"{self.config.motion_url} · #{nonce}"
            ),
            (
                f"{emoji} {hook}\n"
                f"Today's move: {angle}.\n"
                f"Tip with {self.config.tip_bot} 1-1000\n"
                f"{tier_line}"
                f"{cta} {self.config.motion_url}"
            ),
            (
                f"$MOTION pulse {emoji}\n"
                f"{angle.capitalize()} — then tip via {self.config.tip_bot}.\n"
                f"{tier_line}"
                f"{cta}\n"
                f"{self.config.official_account} | {self.config.motion_url}"
            ),
        ]
        text = random.choice(candidates)
        return self._ensure_required(text)

    def generate(self) -> GeneratedPost:
        self._counter += 1
        use_howto = self._counter % self.config.howto_every_n_posts == 0

        for _ in range(40):
            kind = "howto" if use_howto else "engagement"
            text = self._howto_post() if use_howto else self._engagement_post()
            if len(text) > self.config.max_post_length:
                text = text[: self.config.max_post_length - 1].rstrip() + "…"
                text = self._ensure_required(text)
            fp = self.fingerprint(text)
            if fp not in self._history:
                self._history.append(fp)
                self._save_state()
                return GeneratedPost(text=text, kind=kind, fingerprint=fp)
            use_howto = False  # retry with engagement variants

        # Extremely unlikely fallback
        fallback = self._ensure_required(
            f"$MOTION keep moving {int(time.time())} {self.config.motion_url} "
            f"{self.config.official_account} {self.config.tip_bot}"
        )
        fp = self.fingerprint(fallback)
        self._history.append(fp)
        self._save_state()
        return GeneratedPost(text=fallback, kind="fallback", fingerprint=fp)
