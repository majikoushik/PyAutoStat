"""Centralized Rich semantic theme for PyAutoStat terminal presentation.

Colors are chosen for semantic hierarchy, readability, and calm presentation.
Statistical significance is never coded as green (success) or red (failure).
"""

from __future__ import annotations

from rich.theme import Theme

PYAUTOSTAT_THEME = Theme(
    {
        "brand": "bold #4FC3F7",
        "section": "bold #82B1FF",
        "family.profile": "bold #26C6DA",
        "family.mean": "bold #64B5F6",
        "family.rank": "bold #BA68C8",
        "family.association": "bold #4DB6AC",
        "family.categorical": "bold #FFCA28",
        "family.regression": "bold #F06292",
        "family.repeated": "bold #9575CD",
        "family.factorial": "bold #7986CB",
        "family.reliability": "bold #66BB6A",
        "family.status": "bold #FFB74D",
        "family.planning": "bold #81C784",
        "family.reproducibility": "bold #4DD0E1",
        "family.audit": "bold #AED581",
        "family.governance": "bold #90CAF9",
        "family.descriptive": "bold #4DB6AC",
        "result.estimate": "bold",
        "result.ci": "#4DB6AC",
        "result.effect": "#CE93D8",
        "result.evidence": "#FFD54F",
        "method": "#64B5F6",
        "label": "bold",
        "muted": "dim",
        "status.success": "bold #66BB6A",
        "status.review": "bold #FFB74D",
        "status.warning": "bold #FFB74D",
        "status.error": "bold #EF5350",
        "status.missing": "bold #FFD54F",
        "limitation": "dim",
    }
)
