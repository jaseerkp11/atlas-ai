"""Terminal institutional dashboard — narrative + decision."""

from __future__ import annotations

from atlas.institutional.models import InstitutionalDecision, MarketNarrative


def render_dashboard(decision: InstitutionalDecision) -> str:
    n = decision.narrative
    lines: list[str] = []
    lines.append("=" * 72)
    lines.append("  ATLAS INSTITUTIONAL MARKET ANALYSIS ENGINE")
    lines.append("=" * 72)
    if n:
        lines.append(f"  Symbol        : {n.symbol}")
        lines.append(f"  As of (UTC)   : {n.as_of.isoformat()}")
        lines.append(f"  Mid / ATR     : {n.mid:.3f} / {n.atr_m15:.3f}")
        lines.append(f"  Overall Bias  : {n.overall_bias.value}")
        lines.append(f"  HTF Bias      : {n.htf_bias.value}   MTF: {n.mtf_bias.value}")
        lines.append(f"  Trend Quality : {n.trend_quality}")
        lines.append(f"  Volatility    : {n.volatility_regime}")
        lines.append(f"  Best Session  : {n.best_session}")
        lines.append(f"  News Status   : {n.news_status}")
        lines.append("-" * 72)
        lines.append(f"  Structure     : {n.structure_summary}")
        lines.append(f"  Liquidity     : {n.liquidity_summary}")
        lines.append(f"  S/R           : {n.sr_summary}")
        lines.append(f"  Confluence    : {n.institutional_confluence}")
        lines.append("-" * 72)
        lines.append("  MODULE SCORES")
        for m in n.module_scores:
            flag = "OK" if m.passed else "--"
            lines.append(
                f"    [{flag}] {m.name:18} {m.score:5.1f}/100  w={m.weight:4.1f}  "
                f"{m.bias.value:8}  {m.detail[:48]}"
            )
        if n.zones:
            lines.append("-" * 72)
            lines.append("  KEY ZONES (top)")
            for z in sorted(n.zones, key=lambda x: -x.score)[:8]:
                lines.append(
                    f"    {z.strength.value:12} {z.kind:14} "
                    f"{z.bottom:.2f}-{z.top:.2f}  score={z.score:.0f}  {z.label}"
                )
        if n.fib_levels:
            lines.append("-" * 72)
            lines.append("  FIBONACCI (dynamic anchor)")
            for lv in n.fib_levels:
                mark = "*" if lv.confluence else " "
                lines.append(
                    f"   {mark} {lv.ratio:5.3f} @ {lv.price:.2f}  "
                    f"{lv.classification:18} ({lv.timeframe}) score={lv.score:.0f}"
                )
    lines.append("-" * 72)
    lines.append(decision.summary())
    lines.append("=" * 72)
    return "\n".join(lines)
