"""Institutional package exports."""

from atlas.institutional.analyzer import InstitutionalAnalyzer
from atlas.institutional.dashboard import render_dashboard
from atlas.institutional.models import DecisionAction, InstitutionalDecision

__all__ = [
    "InstitutionalAnalyzer",
    "InstitutionalDecision",
    "DecisionAction",
    "render_dashboard",
]
