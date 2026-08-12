"""Institutional package exports (lazy — avoid slow pandas import on every submodule load)."""

from __future__ import annotations

__all__ = [
    "InstitutionalAnalyzer",
    "InstitutionalDecision",
    "DecisionAction",
    "render_dashboard",
]


def __getattr__(name: str):
    if name == "InstitutionalAnalyzer":
        from atlas.institutional.analyzer import InstitutionalAnalyzer

        return InstitutionalAnalyzer
    if name == "InstitutionalDecision":
        from atlas.institutional.models import InstitutionalDecision

        return InstitutionalDecision
    if name == "DecisionAction":
        from atlas.institutional.models import DecisionAction

        return DecisionAction
    if name == "render_dashboard":
        from atlas.institutional.dashboard import render_dashboard

        return render_dashboard
    raise AttributeError(f"module {__name!r} has no attribute {name!r}")
