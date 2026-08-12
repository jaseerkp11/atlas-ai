from __future__ import annotations

from dataclasses import dataclass

from motion_bot.config import BotConfig


@dataclass
class PostResult:
    ok: bool
    dry_run: bool
    tweet_id: str | None
    text: str
    error: str | None = None


class XClient:
    """Thin wrapper around tweepy Client for creating posts."""

    def __init__(self, config: BotConfig) -> None:
        self.config = config
        self._client = None

    def connect(self) -> None:
        if self.config.dry_run:
            return
        if not self.config.has_x_credentials:
            raise RuntimeError(
                "Missing X API credentials. Set X_API_KEY, X_API_SECRET, "
                "X_ACCESS_TOKEN, X_ACCESS_TOKEN_SECRET in motion-bot/.env"
            )
        try:
            import tweepy
        except ImportError as exc:
            raise RuntimeError(
                "tweepy is not installed. Run: pip install -r requirements.txt"
            ) from exc

        self._client = tweepy.Client(
            consumer_key=self.config.x_api_key,
            consumer_secret=self.config.x_api_secret,
            access_token=self.config.x_access_token,
            access_token_secret=self.config.x_access_token_secret,
            bearer_token=self.config.x_bearer_token or None,
            wait_on_rate_limit=True,
        )

    def verify(self) -> dict:
        """Return basic auth identity when live; stub when dry-run."""
        if self.config.dry_run:
            return {"id": "dry-run", "username": "dry_run", "name": "Dry Run"}
        if self._client is None:
            self.connect()
        assert self._client is not None
        me = self._client.get_me(user_fields=["name", "username"])
        if me is None or me.data is None:
            raise RuntimeError("Failed to verify X credentials (get_me returned empty)")
        return {
            "id": str(me.data.id),
            "username": me.data.username,
            "name": me.data.name,
        }

    def create_post(self, text: str) -> PostResult:
        if self.config.dry_run:
            return PostResult(ok=True, dry_run=True, tweet_id=None, text=text)

        if self._client is None:
            self.connect()
        assert self._client is not None

        try:
            response = self._client.create_tweet(text=text)
            tweet_id = None
            if response is not None and getattr(response, "data", None):
                tweet_id = str(response.data.get("id"))
            return PostResult(
                ok=True,
                dry_run=False,
                tweet_id=tweet_id,
                text=text,
            )
        except Exception as exc:  # noqa: BLE001 - surface API errors cleanly
            return PostResult(
                ok=False,
                dry_run=False,
                tweet_id=None,
                text=text,
                error=str(exc),
            )
