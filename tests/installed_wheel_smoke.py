"""Smoke the built wheel from an isolated environment, outside the source import path."""

import sys
from pathlib import Path

import pandas as pd

import pyautostat
from pyautostat import ResearchAssistant, StudyPlanner

module_path = Path(pyautostat.__file__).resolve()
assert module_path.is_relative_to(Path(sys.prefix).resolve()), module_path

independent = pd.DataFrame(
    {
        "group": ["A"] * 6 + ["B"] * 6,
        "score": [4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 1.0, 2.0, 3.0, 4.0, 5.0, 7.0],
    }
)
assistant = ResearchAssistant(independent)
profile = assistant.profile()
assert profile["overview"]["total_rows"] == 12
assert profile["resource_info"]["sampling_applied"] is False
guided = assistant.run(
    objective="compare_groups",
    outcome="score",
    predictor="group",
    design="independent",
    estimand="mean",
    variable_types={"score": "continuous"},
)
assert guided.status.value == "completed"

paired = pd.DataFrame(
    {
        "participant": [1, 1, 2, 2, 3, 3, 4, 4],
        "condition": ["before", "after"] * 4,
        "score": [12.0, 9.0, 10.0, 8.0, 15.0, 11.0, 9.0, 8.0],
    }
)
paired_guided = ResearchAssistant(paired).run(
    objective="compare_groups",
    outcome="score",
    predictor="condition",
    design="paired",
    estimand="mean",
    unit_id="participant",
    condition_order=("before", "after"),
    variable_types={"score": "continuous"},
)
assert paired_guided.status.value == "completed"

planning = StudyPlanner().independent_mean_power(
    target_difference=2,
    sd_group1=3,
    sd_group2=3,
)
assert planning.status == "available"

html = guided.report.to_html(style="apa")
assert isinstance(html, str) and "<!doctype html>" in html
print(f"installed-wheel smoke passed: {pyautostat.__version__} from {module_path}")
