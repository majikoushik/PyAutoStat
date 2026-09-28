"""Smoke the built wheel from an isolated environment, outside the source import path."""

import sys
from pathlib import Path

import pandas as pd

import pyautostat
from pyautostat import ResearchAssistant, StudyPlanner

module_path = Path(pyautostat.__file__).resolve()
assert module_path.is_relative_to(Path(sys.prefix).resolve()), module_path
assert pyautostat.__version__

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
assert list(profile["descriptive"]["score"]["percentiles"]) == [
    "p05",
    "p25",
    "p50",
    "p75",
    "p95",
]
frequency = assistant.frequency_table("group")
assert frequency["valid_n"] == 12
assert "GROUP DISTRIBUTION" in frequency["narrative"]
cross_tab = assistant.cross_tab(
    "group",
    "score",
    data_dictionary={"score": {"type": "ordinal", "ordinal_order": sorted(set(independent.score))}},
)
assert sum(map(sum, cross_tab["counts"])) == 12
assert "observed distribution only" in cross_tab["narrative"]
story = assistant.summarize(mode="story")
assert "DATASET STORY" in story
guided = assistant.run(
    objective="compare_groups",
    outcome="score",
    predictor="group",
    design="independent",
    estimand="mean",
    variable_types={"score": "continuous"},
)
assert guided.status.value == "completed"
assert guided.recommendation is not None
assert "WHY THIS TEST?" in guided.recommendation.rationale_text
explanation = guided.explain()
assert "ANALYSIS RESULT" in explanation
assert "Groups    : 'A', 'B'" in explanation

multi_group = pd.DataFrame(
    {
        "group": ["C"] * 5 + ["A"] * 5 + ["B"] * 5,
        "score": [2, 3, 4, 5, 7, 8, 10, 12, 15, 18, 1, 2, 2, 3, 5],
    }
)
multi_guided = ResearchAssistant(multi_group).run(
    objective="compare_groups",
    outcome="score",
    predictor="group",
    design="independent",
    estimand="mean",
    variable_types={"score": "continuous", "group": "nominal"},
)
assert multi_guided.analysis.method_id == "welch_anova"
assert len(multi_guided.analysis.values["pairwise_comparisons"]) == 3
assert "pairwise_comparisons" in multi_guided.report.to_csv_tables()

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

one_sample = ResearchAssistant(pd.DataFrame({"score": [48.0, 51.0, 53.0, 55.0]})).run(
    objective="compare_reference",
    outcome="score",
    reference_value=50.0,
    design="independent",
    estimand="mean",
    variable_types={"score": "continuous"},
)
assert one_sample.status.value == "completed"
assert one_sample.analysis.method_id == "one_sample_t"

paired_rank = ResearchAssistant(paired).run(
    objective="compare_groups",
    outcome="score",
    predictor="condition",
    design="paired",
    estimand="distribution",
    unit_id="participant",
    condition_order=("before", "after"),
    variable_types={"score": "continuous", "condition": "nominal"},
)
assert paired_rank.status.value == "partial"
assert paired_rank.analysis.method_id == "wilcoxon_signed_rank"

monotonic = ResearchAssistant(
    pd.DataFrame({"hours": [1, 2, 3, 4, 5, 6], "score": [2, 4, 3, 7, 8, 10]})
).run(
    objective="association",
    outcome="score",
    predictor="hours",
    design="independent",
    estimand="monotonic",
    variable_types={"score": "continuous", "hours": "continuous"},
)
assert monotonic.status.value == "completed"
assert monotonic.analysis.method_id == "spearman_correlation"

sparse = ResearchAssistant(
    pd.DataFrame(
        {
            "treatment": ["A"] * 5 + ["B"] * 7,
            "response": ["yes"] + ["no"] * 4 + ["yes"] * 5 + ["no"] * 2,
        }
    )
).run(
    objective="association",
    outcome="response",
    predictor="treatment",
    design="independent",
    estimand="categorical_independence",
    variable_types={"response": "nominal", "treatment": "nominal"},
)
assert sparse.status.value == "partial"
assert sparse.analysis.method_id == "fisher_exact"

regression = ResearchAssistant(
    pd.DataFrame(
        {
            "y": [4.2, 5.1, 6.5, 7.0, 8.4, 9.2, 10.5, 11.3],
            "x": [1.0, 2, 3, 4, 5, 6, 7, 8],
            "group": ["A", "B"] * 4,
        }
    )
).run(
    objective="regression",
    outcome="y",
    predictors=["x", "group"],
    design="independent",
    estimand="conditional_mean",
    variable_types={"y": "continuous", "x": "continuous", "group": "nominal"},
    reference_levels={"group": "A"},
    covariance_type="HC3",
)
assert regression.status.value == "completed"
assert regression.analysis.method_id == "linear_regression"
assert regression.analysis.values["covariance_type"] == "HC3"
assert "regression_diagnostics" in regression.report.to_csv_tables()
assert regression.audit.status == "passed"

planning = StudyPlanner().independent_mean_power(
    target_difference=2,
    sd_group1=3,
    sd_group2=3,
)
assert planning.status == "available"

html = guided.report.to_html(style="apa")
assert isinstance(html, str) and "<!doctype html>" in html
assert '<section class="executive-summary">' in html
print(f"installed-wheel smoke passed: {pyautostat.__version__} from {module_path}")
