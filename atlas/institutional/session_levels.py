"""
Calendar session levels for manual chart markup.

Uses closed bars only:
  - PDH / PDL  = previous completed UTC day high / low
  - Asian H/L  = most recent 00:00–07:00 UTC range
  - London H/L = most recent 07:00–10:00 UTC range (open raid window)

These are prop-style levels traders actually draw — not rolling bar proxies.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

import pandas as pd


@dataclass
class SessionLevel:
    kind: str  # pdh | pdl | asian_high | asian_low | london_high | london_low
    price: float
    label: str
    side: str  # BUY | SELL  (support-like vs resistance-like)
    score: float = 88.0

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class SessionLevelsReport:
    levels: list[SessionLevel] = field(default_factory=list)
    pdh: float | None = None
    pdl: float | None = None
    asian_high: float | None = None
    asian_low: float | None = None
    london_high: float | None = None
    london_low: float | None = None
    summary: str = ""

    def as_dict(self) -> dict:
        return {
            "levels": [lv.as_dict() for lv in self.levels],
            "pdh": self.pdh,
            "pdl": self.pdl,
            "asian_high": self.asian_high,
            "asian_low": self.asian_low,
            "london_high": self.london_high,
            "london_low": self.london_low,
            "summary": self.summary,
        }


def _closed(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or len(df) < 2:
        return df
    return df.iloc[:-1]


def extract_session_levels(
    df: pd.DataFrame | None,
    now: datetime | None = None,
) -> SessionLevelsReport:
    """Build PDH/PDL + Asian + London session extremes from OHLC with timestamps."""
    empty = SessionLevelsReport(summary="no session levels")
    if df is None or len(df) < 20 or "time" not in df.columns:
        return empty

    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)

    try:
        work = _closed(df).copy()
        work["time"] = pd.to_datetime(work["time"], utc=True)
        work["date"] = work["time"].dt.floor("D")
        work["hour"] = work["time"].dt.hour
    except Exception:
        return empty

    levels: list[SessionLevel] = []
    pdh = pdl = None
    a_hi = a_lo = None
    l_hi = l_lo = None

    # Previous completed UTC day
    today = now.astimezone(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    prev_days = sorted(d for d in work["date"].unique() if pd.Timestamp(d).to_pydatetime().replace(tzinfo=timezone.utc) < today)
    if prev_days:
        prev = work[work["date"] == prev_days[-1]]
        if len(prev):
            pdh = float(prev["high"].max())
            pdl = float(prev["low"].min())
            levels.append(SessionLevel("pdh", pdh, "PDH (prev UTC day high)", "SELL", 92.0))
            levels.append(SessionLevel("pdl", pdl, "PDL (prev UTC day low)", "BUY", 92.0))

    # Most recent Asian window 00–07 UTC (prefer today, else last available)
    asian = work[(work["hour"] >= 0) & (work["hour"] < 7)]
    if len(asian) >= 4:
        # Prefer today's asian if enough bars, else last asian day in sample
        today_asian = asian[asian["date"] == pd.Timestamp(today)]
        use = today_asian if len(today_asian) >= 4 else asian.groupby("date").filter(lambda g: len(g) >= 4)
        if len(use):
            last_day = use["date"].max()
            block = use[use["date"] == last_day]
            a_hi = float(block["high"].max())
            a_lo = float(block["low"].min())
            levels.append(SessionLevel("asian_high", a_hi, "Asian High (00-07 UTC)", "SELL", 86.0))
            levels.append(SessionLevel("asian_low", a_lo, "Asian Low (00-07 UTC)", "BUY", 86.0))

    # London open raid window 07–10 UTC
    london = work[(work["hour"] >= 7) & (work["hour"] < 10)]
    if len(london) >= 3:
        today_lon = london[london["date"] == pd.Timestamp(today)]
        use = today_lon if len(today_lon) >= 3 else london.groupby("date").filter(lambda g: len(g) >= 3)
        if len(use):
            last_day = use["date"].max()
            block = use[use["date"] == last_day]
            l_hi = float(block["high"].max())
            l_lo = float(block["low"].min())
            levels.append(SessionLevel("london_high", l_hi, "London range high (07-10 UTC)", "SELL", 84.0))
            levels.append(SessionLevel("london_low", l_lo, "London range low (07-10 UTC)", "BUY", 84.0))

    bits = []
    if pdh is not None:
        bits.append(f"PDH={pdh:.2f}")
    if pdl is not None:
        bits.append(f"PDL={pdl:.2f}")
    if a_hi is not None:
        bits.append(f"Asian={a_lo:.2f}-{a_hi:.2f}")
    if l_hi is not None:
        bits.append(f"Lon={l_lo:.2f}-{l_hi:.2f}")
    summary = " | ".join(bits) if bits else "no session levels"

    return SessionLevelsReport(
        levels=levels,
        pdh=pdh,
        pdl=pdl,
        asian_high=a_hi,
        asian_low=a_lo,
        london_high=l_hi,
        london_low=l_lo,
        summary=summary,
    )


def session_levels_as_pools(report: SessionLevelsReport) -> list[dict[str, Any]]:
    """Shape for liquidity-style consumers."""
    out = []
    for lv in report.levels:
        out.append(
            {
                "kind": lv.kind,
                "price": lv.price,
                "significance": lv.score,
                "label": lv.label,
                "side": lv.side,
            }
        )
    return out
