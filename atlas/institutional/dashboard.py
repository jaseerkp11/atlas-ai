"""Terminal institutional dashboard — narrative + high-probability checklist."""

from __future__ import annotations

from atlas.institutional.models import InstitutionalDecision


def render_dashboard(decision: InstitutionalDecision) -> str:
    n = decision.narrative
    lines: list[str] = []
    lines.append("=" * 72)
    lines.append("  ATLAS INSTITUTIONAL MARKET ANALYSIS ENGINE")
    lines.append("  High-probability SMC · quality over quantity · prefer NO TRADE")
    lines.append("=" * 72)
    if n:
        lines.append(f"  Symbol        : {n.symbol}")
        lines.append(f"  As of (UTC)   : {n.as_of.isoformat()}")
        lines.append(f"  Mid / ATR     : {n.mid:.3f} / {n.atr_m15:.3f}")
        h1 = n.extras.get("h1_bias", n.overall_bias.value)
        h4 = n.extras.get("h4_bias", n.htf_bias.value)
        m15 = n.extras.get("m15_bias", n.mtf_bias.value)
        lines.append(f"  Overall Bias  : {n.overall_bias.value}")
        lines.append(f"  HTF stack     : H4={h4}  H1={h1} (primary)  M15={m15}")
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

        # High-probability playbooks
        lines.append("-" * 72)
        lines.append("  HIGH-PROBABILITY PLAYBOOKS")
        hits = n.extras.get("playbook_hits") or []
        if hits:
            for h in hits:
                lines.append(
                    f"    • {h['name']:32} {h['bias']:8}  boost=+{h['boost']:.0f}"
                )
        else:
            lines.append("    (none active — stand aside)")
        if n.extras.get("playbook"):
            lines.append(f"  Stack: {n.extras.get('playbook')}")

        # Scenario / checklist
        sc = n.extras.get("scenario") or {}
        if sc:
            lines.append("-" * 72)
            lines.append("  AI SCENARIO PLAN")
            lines.append(f"  Primary      : {sc.get('primary', '')}")
            lines.append(f"  Alternate    : {sc.get('alternate', '')}")
            lines.append(f"  Invalidation : {sc.get('invalidation', '')}")
            lines.append(f"  Edge score   : {sc.get('edge_score', 0):.0f}/100  ({sc.get('summary', '')})")
            lines.append("  Checklist:")
            for c in sc.get("checklist", []):
                mark = "OK" if c.get("ok") else "--"
                lines.append(f"    [{mark}] {c.get('name')}: {c.get('detail')}")
            triggers = sc.get("next_triggers") or n.extras.get("unlock") or []
            if triggers:
                lines.append("  Next triggers:")
                for t in triggers[:5]:
                    lines.append(f"    → {t}")
        elif n.extras.get("unlock"):
            lines.append("  Unlock path:")
            for h in n.extras["unlock"][:5]:
                lines.append(f"    → {h}")

    lines.append("-" * 72)
    lines.append(decision.summary())
    lines.append("=" * 72)
    return "\n".join(lines)
