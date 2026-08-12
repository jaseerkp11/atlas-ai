"""Terminal dashboard — full institutional/manual analysis (default).

Use brief=True / --brief for a short trader card only.
"""

from __future__ import annotations

from atlas.institutional.models import DecisionAction, InstitutionalDecision


def _setup_status(decision: InstitutionalDecision) -> str:
    """Never sound like an auto-entry order."""
    if decision.action == DecisionAction.WAIT:
        return "SETUP FORMING — wait for WAIT→CONFIRM→THEN on TradingView"
    if decision.action == DecisionAction.NO_TRADE:
        return "STAND ASIDE — map levels only; no clean trigger yet"
    if decision.action in (DecisionAction.BUY, DecisionAction.SELL):
        side = "LONG" if decision.action == DecisionAction.BUY else "SHORT"
        return (
            f"BIAS ALIGNED ({side}) — NOT an entry signal. "
            "Trade only if an A+/A setup completes WAIT→CONFIRM→THEN on your chart"
        )
    return decision.action.value


def _load_report(n):
    from dataclasses import fields

    from atlas.institutional.manual_scanner import (
        FocusChecklistItem,
        FocusPlan,
        ManualScanReport,
        SetupCard,
        SideCompareBoard,
    )

    raw = (n.extras.get("manual_scan") if n else None) or {}
    if not raw:
        return None

    def _card(d: dict) -> SetupCard:
        known = {f.name for f in fields(SetupCard)}
        return SetupCard(**{k: v for k, v in d.items() if k in known})

    def _side_compare(d) -> SideCompareBoard | None:
        if not isinstance(d, dict) or not d:
            return None
        known = {f.name for f in fields(SideCompareBoard)}
        return SideCompareBoard(**{k: v for k, v in d.items() if k in known})

    def _focus(d) -> FocusPlan | None:
        if not isinstance(d, dict) or not d:
            return None
        known = {f.name for f in fields(FocusPlan)}
        data = {k: v for k, v in d.items() if k in known}
        raw_items = data.pop("checklist", []) or []
        items = []
        for it in raw_items:
            if isinstance(it, dict):
                items.append(
                    FocusChecklistItem(
                        name=str(it.get("name", "")),
                        passed=bool(it.get("passed")),
                        detail=str(it.get("detail", "")),
                    )
                )
        data["checklist"] = items
        return FocusPlan(**data)

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
        side_compare=_side_compare(raw.get("side_compare")),
        focus=_focus(raw.get("focus")),
    )


def render_dashboard(decision: InstitutionalDecision, full: bool = True, brief: bool = False) -> str:
    """
    Default = FULL long analysis (modules, zones, fib, playbooks, setups).
    brief=True = short card only.
    full kept for CLI compatibility (True by default).
    """
    from atlas.institutional.manual_scanner import render_manual_scan_block

    use_brief = brief or (not full)
    n = decision.narrative
    report = _load_report(n)
    lines: list[str] = []

    if use_brief:
        lines.append("=" * 72)
        lines.append("  ATLAS SCANNER — SHORT BRIEF  (you trade · no auto)")
        lines.append("=" * 72)
        if n:
            h1 = n.extras.get("h1_bias", n.overall_bias.value)
            h4 = n.extras.get("h4_bias", n.htf_bias.value)
            m15 = n.extras.get("m15_bias", n.mtf_bias.value)
            lines.append(
                f"  {n.symbol}  mid={n.mid:.2f}  ATR({(n.extras or {}).get('atr_tf', 'M5')}):"
                f"{n.atr_m15:.2f}  session={n.best_session}"
            )
            lines.append(f"  Bias: {n.overall_bias.value}   H4={h4}  H1={h1}(main)  M15={m15}")
            lines.append(f"  Status: {_setup_status(decision)}")
            lines.append(f"  News: {n.news_status}")
        lines.extend(render_manual_scan_block(report, mid=n.mid if n else 0.0, brief=True))
        lines.append("=" * 72)
        return "\n".join(lines)

    # -------- FULL ANALYSIS (default) --------
    lines.append("=" * 72)
    lines.append("  ATLAS MANUAL HIGH-PROBABILITY MARKET SCANNER")
    lines.append("  AI analysis for YOUR TradingView decisions · NO auto trading")
    lines.append("=" * 72)
    if n:
        h1 = n.extras.get("h1_bias", n.overall_bias.value)
        h4 = n.extras.get("h4_bias", n.htf_bias.value)
        m15 = n.extras.get("m15_bias", n.mtf_bias.value)
        lines.append(f"  Symbol        : {n.symbol}")
        lines.append(f"  As of (UTC)   : {n.as_of.isoformat()}")
        atr_tf = (n.extras or {}).get("atr_tf", "M5")
        lines.append(f"  Mid / ATR({atr_tf}): {n.mid:.3f} / {n.atr_m15:.3f}")
        pd_zone = (n.extras or {}).get("pd_zone")
        if pd_zone:
            lines.append(f"  H1 array     : {str(pd_zone).upper()}")
        sess = (n.extras or {}).get("session_levels") or {}
        if isinstance(sess, dict) and sess.get("summary"):
            lines.append(f"  Session lvls : {sess.get('summary')}")
            for lv in (sess.get("levels") or [])[:6]:
                lines.append(
                    f"    · {lv.get('label')}: {float(lv.get('price', 0)):.2f}"
                )
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
            for z in sorted(n.zones, key=lambda x: -x.score)[:10]:
                lo, hi = min(z.bottom, z.top), max(z.bottom, z.top)
                lines.append(
                    f"    {z.strength.value:12} {z.kind:14} "
                    f"{lo:.2f}-{hi:.2f}  score={z.score:.0f}  {z.label}"
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

        tls = n.extras.get("trendlines") or []
        if tls:
            lines.append("-" * 72)
            lines.append("  TRENDLINES (H1 swings — mark on TradingView)")
            for ln in tls:
                lines.append(
                    f"    {str(ln.get('kind','?')).upper():11} "
                    f"@{float(ln.get('price', 0)):.2f}  "
                    f"quality={ln.get('quality')}  touches={ln.get('touches')}  "
                    f"break_prob={float(ln.get('break_prob', 0)):.0%}"
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
    lines.append("  (Guidance only — you confirm on TradingView and trade yourself)")

    lines.append("-" * 72)
    lines.append("  HOW TO READ EACH SETUP (so you do not misunderstand):")
    lines.append("    WAIT    = what must happen first (do nothing before this)")
    lines.append("    CONFIRM = the M5 candle proof you must see")
    lines.append("    THEN    = only after CONFIRM may you consider the trade")
    lines.append("    NOW     = what to do at the current price right now")
    lines.append("    CANCEL  = if this prints, the idea is dead")

    lines.extend(render_manual_scan_block(report, mid=n.mid if n else 0.0, brief=False))
    lines.append("=" * 72)
    return "\n".join(lines)
