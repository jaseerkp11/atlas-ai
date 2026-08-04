"""Terminal dashboard — short manual trader brief (optional --full details)."""

from __future__ import annotations

from atlas.institutional.models import DecisionAction, InstitutionalDecision


def _setup_status(decision: InstitutionalDecision) -> str:
    if decision.action == DecisionAction.WAIT:
        return "WAIT — setup forming (no entry yet)"
    if decision.action == DecisionAction.NO_TRADE:
        return "STAND ASIDE — map only"
    if decision.action in (DecisionAction.BUY, DecisionAction.SELL):
        side = "LONG" if decision.action == DecisionAction.BUY else "SHORT"
        return f"BIAS={side} — still NOT an entry (need IF/THEN on chart)"
    return decision.action.value


def _load_report(n):
    from atlas.institutional.manual_scanner import ManualScanReport, SetupCard

    raw = (n.extras.get("manual_scan") if n else None) or {}
    if not raw:
        return None

    def _card(d: dict) -> SetupCard:
        return SetupCard(**d)

    return ManualScanReport(
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


def render_dashboard(decision: InstitutionalDecision, full: bool = False) -> str:
    """
    Default = short trader brief (easy to read).
    full=True = old detailed modules/fib/playbooks dump.
    """
    from atlas.institutional.manual_scanner import render_manual_scan_block

    n = decision.narrative
    report = _load_report(n)
    lines: list[str] = []

    # --- SHORT BRIEF (always first) ---
    lines.append("=" * 72)
    lines.append("  ATLAS SCANNER — SHORT BRIEF  (you trade · no auto)")
    lines.append("=" * 72)
    if n:
        h1 = n.extras.get("h1_bias", n.overall_bias.value)
        h4 = n.extras.get("h4_bias", n.htf_bias.value)
        m15 = n.extras.get("m15_bias", n.mtf_bias.value)
        lines.append(f"  {n.symbol}  mid={n.mid:.2f}  ATR={n.atr_m15:.2f}  session={n.best_session}")
        lines.append(f"  Bias: {n.overall_bias.value}   H4={h4}  H1={h1}(main)  M15={m15}")
        lines.append(f"  Status: {_setup_status(decision)}")
        lines.append(f"  News: {n.news_status}")
        if report:
            focus = next(
                (
                    c
                    for c in (
                        report.buy_cards
                        if report.stance == "LONG_BIAS"
                        else report.sell_cards
                        if report.stance == "SHORT_BIAS"
                        else report.cards
                    )
                    if c.grade in ("A+", "A") and c.status != "AVOID_NOW"
                ),
                None,
            )
            if focus:
                lines.append(
                    f"  FOCUS: [{focus.grade}] {focus.side} {focus.zone_low:.2f}-{focus.zone_high:.2f}  ({focus.status})"
                )
            else:
                lines.append("  FOCUS: none — stand aside")
        lines.append("-" * 72)
        lines.append("  READ IF/THEN LIKE THIS:")
        lines.append("    1) Do NOTHING until price reaches the zone")
        lines.append("    2) Only AFTER the confirmation candle → you may consider entry")
        lines.append("    3) If cancel level hits → idea is dead")
        lines.append("    4) B / AVOID = ignore for entries (targets/danger only)")

    lines.extend(render_manual_scan_block(report, mid=n.mid if n else 0.0, brief=True))

    if full and n:
        lines.append("")
        lines.append("=" * 72)
        lines.append("  FULL DETAILS (optional)")
        lines.append("=" * 72)
        lines.append(f"  Structure : {n.structure_summary}")
        lines.append(f"  Liquidity : {n.liquidity_summary}")
        lines.append(f"  S/R       : {n.sr_summary}")
        lines.append(f"  Confluence: {n.institutional_confluence}")
        lines.append("  Modules:")
        for m in n.module_scores:
            flag = "OK" if m.passed else "--"
            lines.append(
                f"    [{flag}] {m.name:18} {m.score:5.1f}  {m.bias.value:8}  {m.detail[:42]}"
            )
        hits = n.extras.get("playbook_hits") or []
        if hits:
            lines.append("  Playbooks:")
            for h in hits[:5]:
                lines.append(f"    • {h['name']} ({h['bias']}) +{h['boost']:.0f}")
        if n.fib_levels:
            lines.append("  Fib:")
            for lv in n.fib_levels:
                lines.append(f"    {lv.ratio:.3f} @ {lv.price:.2f} ({lv.classification})")
        lines.extend(render_manual_scan_block(report, mid=n.mid, brief=False))

    lines.append("=" * 72)
    return "\n".join(lines)
