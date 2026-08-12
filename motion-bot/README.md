# Motion X Engagement Bot

Standalone bot (separate from ATLAS trading) that connects to your X account and posts unique, engaging `$MOTION` community content on a timer so Motion's graph can index your activity.

Official refs:
- Site: https://motion.tips
- Protocol: `@Motiontip`
- Tip bot: `@tipmotion`
- Cashtag: `$MOTION`

## What it does

1. Connects to X via API (OAuth 1.0a user context)
2. Builds a **unique** post every cycle (templates + rotation + anti-repeat)
3. Always includes `$MOTION`, `@Motiontip`, `@tipmotion`, and `https://motion.tips`
4. Optionally tags tier / community accounts you configure
5. Posts every **60 seconds** by default (configurable)

Dry-run is the default until you set `DRY_RUN=false`.

## Setup

```bash
cd motion-bot
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

Fill `.env` with X developer credentials (app + user access tokens with **Read and Write**).

Create an app at https://developer.x.com — posting every minute needs a paid X API plan that allows tweet create volume (Free is too limited).

## Run

```bash
# Preview posts without publishing
python main.py once --dry-run

# Single live post
python main.py once

# Loop every 60s (dry-run)
python main.py run --dry-run

# Live loop every 60s
DRY_RUN=false python main.py run

# Custom interval (seconds)
python main.py run --interval 60
```

## Config

Edit `config.yaml`:
- `interval_seconds` — default `60`
- `tier_tags` — accounts to mention by tier
- `max_post_length` — keep under 280
- `include_howto` — occasionally include tip howto steps

## Tip howto (from Motion)

```
How to tip $MOTION

1. Go to https://motion.tips
2. Connect your X account
3. Check your Balance
4. Find a builder, creator, or post you like
5. Reply with @tipmotion 1-1000
```

## Safety notes

- Motion scores `$cashtags`, `@mentions`, quotes and replies — spammy identical posts may score poorly or get limited by X.
- This bot enforces uniqueness and keeps a local post hash history.
- Start in dry-run, then live on a schedule you are comfortable with.
