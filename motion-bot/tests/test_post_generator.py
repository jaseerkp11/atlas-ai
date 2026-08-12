from __future__ import annotations

from pathlib import Path

import pytest

from motion_bot.config import BotConfig, load_config
from motion_bot.post_generator import PostGenerator


@pytest.fixture()
def config(tmp_path: Path) -> BotConfig:
    return BotConfig(
        interval_seconds=60,
        max_post_length=280,
        dry_run=True,
        required_tags=["$MOTION", "@Motiontip", "@tipmotion"],
        motion_url="https://motion.tips",
        tip_bot="@tipmotion",
        official_account="@Motiontip",
        tier_tags={
            "elite": ["@Motiontip"],
            "builders": ["@tipmotion"],
            "community": ["@Motiontip"],
        },
        howto_every_n_posts=3,
        history_size=100,
        data_dir=tmp_path / "data",
    )


def test_load_config_defaults():
    cfg = load_config()
    assert cfg.interval_seconds >= 15
    assert "$MOTION" in cfg.required_tags
    assert "@tipmotion" in cfg.required_tags
    assert "@Motiontip" in cfg.required_tags


def test_generated_posts_include_required_signals(config: BotConfig):
    gen = PostGenerator(config)
    seen = set()
    for _ in range(12):
        post = gen.generate()
        assert len(post.text) <= config.max_post_length
        lower = post.text.lower()
        assert "$motion" in lower
        assert "@tipmotion" in lower
        assert "@motiontip" in lower
        assert "motion.tips" in lower or "$motion" in lower
        assert post.fingerprint not in seen
        seen.add(post.fingerprint)


def test_howto_post_contains_tip_steps(config: BotConfig):
    config.howto_every_n_posts = 1
    gen = PostGenerator(config)
    post = gen.generate()
    assert post.kind in {"howto", "engagement", "fallback"}
    # With every_n=1 the first post should be howto
    assert "1" in post.text or "tip" in post.text.lower()
    assert "@tipmotion" in post.text.lower()


def test_uniqueness_across_many_generations(config: BotConfig):
    gen = PostGenerator(config)
    fingerprints = [gen.generate().fingerprint for _ in range(30)]
    assert len(set(fingerprints)) == 30
