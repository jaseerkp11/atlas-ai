"""Terminal dashboard — manual high-probability scanner (no auto trade)."""

from __future__ import annotations

from atlas.institutional.models import DecisionAction, InstitutionalDecision


def _setup_status(decision: InstitutionalDecision) -> str:
    """Reframe engine action as manual setup status (never 'buy now')."""
    ms = ""
    if decision.narrative:
        ms = str(decision.narrative.extras.get("manual_stance") or "")
    if decision.action == DecisionAction.WAIT:
        return "SETUP FORMING — wait for IF/THEN trigger on TradingView"
    if decision.action == DecisionAction.NO_TRADE:
        return "STAND ASIDE — map levels only; no clean trigger yet"
    if decision.action in (DecisionAction.BUY, DecisionAction.SELL):
        side = "LONG" if decision.action == DecisionAction.BUY else "SHORT"
        return (
            f"BIAS ALIGNED ({side}) — NOT an entry signal. "
            f"Only trade if an A+/A zone IF/THEN confirms on TradingView"
        )
    return f"STATUS={decision.action.value} stance={ms}"


def render_dashboard(decision: InstitutionalDecision) -> str:
    n = decision.narrative
    lines: list[str] = []
    lines.append("=" * 72)
    lines.append("  ATLAS MANUAL HIGH-PROBABILITY MARKET SCANNER")
    lines.append("  AI analysis for YOUR TradingView decisions · NO auto trading")
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
            lines.append("  KEY S/R + IMBALANCE ZONES")
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
        lines.append("  HIGH-PROBABILITY PLAYBOOK THEMES")
        hits = n.extras.get("playbook_hits") or []
        if hits:
            for h in hits:
                lines.append(
                    f"    • {h['name']:32} {h['bias']:8}  boost=+{h['boost']:.0f}"
                )
        else:
            lines.append("    (none — wait for sweep / OTE / FVG / structure)")
        if n.extras.get("playbook"):
            lines.append(f"  Stack: {n.extras.get('playbook')}")

        sc = n.extras.get("scenario") or {}
        if sc:
            lines.append("-" * 72)
            lines.append("  AI CONTEXT")
            lines.append(f"  Primary      : {sc.get('primary', '')}")
            lines.append(f"  Alternate    : {sc.get('alternate', '')}")
            lines.append(f"  Invalidation : {sc.get('invalidation', '')}")
            triggers = sc.get("next_triggers") or n.extras.get("unlock") or []
            if triggers:
                lines.append("  Market still needs:")
                for t in triggers[:5]:
                    lines.append(f"    → {t}")

    lines.append("-" * 72)
    lines.append(f"  SETUP STATUS  : {_setup_status(decision)}")
    lines.append(
        f"  Edge readout  : prob={decision.probability:.1f}% conf={decision.confidence:.1f}% "
        f"confl={decision.confluence:.1f} risk={decision.risk_level.value}"
    )
    lines.append("  (Status is guidance only — you confirm and trade manually)")

    # Primary deliverable: graded manual scanner with IF/THEN
    from atlas.institutional.manual_scanner import ManualScanReport, SetupCard, render_manual_scan_block

    raw = (n.extras.get("manual_scan") if n else None) or {}
    report = None
    if raw:
        def _card(d: dict) -> SetupCard:
            return SetupCard(**d)

        report = ManualScanReport(
            bias=str(raw.get("bias", "")),
            stance=str(raw.get("stance", "")),
            headline=str(raw.get("headline", "")),
            cards=[_card(c) for c in raw.get("cards", [])],
            buy_cards=[_card(c) for c in raw.get("buy_cards", [])],
            sell_cards=[_card(c) for c in raw.get("sell_cards", [])],
            watchlist=list(raw.get("watchlist") or []),
            do_not=list(raw.get("do_not") or []),
            summary=str(raw.get("summary", "")),
        )
    lines.extend(render_manual_scan_block(report, mid=n.mid if n else 0.0))
    lines.append("=" * 72)
    return "\n".join(lines)
