"""Smoke the built wheel from an isolated environment, outside the source import path."""

import json
import math
import sys
from pathlib import Path

import pandas as pd

import pyautostat
from pyautostat import AnalysisOptions, ResearchAssistant, StudyPlanner

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
assert paired_rank.status.value == "completed"
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

phase6_options = AnalysisOptions(random_seed=7, bootstrap_samples=100)
logistic = ResearchAssistant(
    pd.DataFrame(
        {
            "event": [
                "no",
                "yes",
                "yes",
                "no",
                "yes",
                "no",
                "no",
                "yes",
                "yes",
                "no",
                "no",
                "yes",
                "no",
                "yes",
                "no",
                "no",
                "yes",
                "yes",
                "no",
                "yes",
            ],
            "x": list(range(20)),
        }
    )
).run(
    objective="regression",
    outcome="event",
    predictors=["x"],
    design="independent",
    estimand="event_probability",
    event_level="yes",
    variable_types={"event": "nominal", "x": "continuous"},
)
assert logistic.status.value == "completed"
assert logistic.analysis.method_id == "logistic_regression"
assert "ODDS RATIOS" in logistic.explain()
assert "logistic_coefficients" in logistic.report.to_csv_tables()

mcnemar_rows = []
for unit, (before, after) in enumerate([(0, 1), (0, 1), (0, 0), (1, 1), (1, 0), (0, 1)]):
    mcnemar_rows.extend(
        [
            {"unit": unit, "condition": "before", "response": before},
            {"unit": unit, "condition": "after", "response": after},
        ]
    )
mcnemar = ResearchAssistant(pd.DataFrame(mcnemar_rows)).run(
    objective="compare_groups",
    outcome="response",
    predictor="condition",
    design="paired",
    estimand="proportion",
    unit_id="unit",
    condition_order=("after", "before"),
    event_level=1,
    options=phase6_options,
    variable_types={"response": "nominal", "condition": "nominal"},
)
assert mcnemar.status.value == "completed"
assert mcnemar.analysis.method_id == "mcnemar"
assert "PAIRED BINARY COMPARISON" in mcnemar.explain()
assert "mcnemar_transition_table" in mcnemar.report.to_csv_tables()

association_frame = pd.DataFrame(
    {
        "binary": [False, False, False, False, True, True, True, True],
        "score": [1.0, 2.0, 3.0, 5.0, 4.0, 6.0, 8.0, 9.0],
        "rank": [8.0, 7.0, 7.0, 5.0, 4.0, 3.0, 2.0, 1.0],
        "control": [2.0, 1.0, 3.0, 2.0, 5.0, 4.0, 6.0, 5.0],
    }
)
point = ResearchAssistant(association_frame).run(
    objective="association",
    outcome="binary",
    predictor="score",
    design="independent",
    estimand="point_biserial",
    options=phase6_options,
    variable_types={"binary": "boolean", "score": "continuous"},
)
kendall = ResearchAssistant(association_frame).run(
    objective="association",
    outcome="score",
    predictor="rank",
    design="independent",
    estimand="monotonic",
    association_measure="kendall",
    options=phase6_options,
    variable_types={"score": "continuous", "rank": "continuous"},
)
partial = ResearchAssistant(association_frame).run(
    objective="association",
    outcome="score",
    predictor="rank",
    controls=["control"],
    design="independent",
    estimand="partial_linear",
    options=phase6_options,
    variable_types={"score": "continuous", "rank": "continuous", "control": "continuous"},
)
for workflow, method, heading in (
    (point, "point_biserial_correlation", "BINARY CODING"),
    (kendall, "kendall_tau_b", "MONOTONIC ASSOCIATION"),
    (partial, "partial_pearson_correlation", "PARTIAL ASSOCIATION"),
):
    assert workflow.status.value == "completed"
    assert workflow.analysis.method_id == method
    assert heading in workflow.explain()
    assert "<!doctype html>" in workflow.report.to_html()
    assert workflow.audit.status == "passed"
    json.loads(workflow.to_json())

reliability = ResearchAssistant(
    pd.DataFrame(
        {
            "q1": [1, 2, 3, 4, 5, 2, 4, 3],
            "q2": [1, 2, 3, 4, 4, 2, 5, 3],
            "q3": [2, 2, 3, 5, 5, 2, 4, 4],
        }
    )
).reliability(["q1", "q2", "q3"], bootstrap_samples=59)
assert reliability.status.value == "completed"
assert reliability.analysis.method_id == "cronbach_alpha"
assert math.isfinite(reliability.analysis.values["cronbach_alpha"])
assert "RELIABILITY ESTIMATE" in reliability.explain()
json.loads(reliability.to_json())
assert "reliability_items" in reliability.report.to_csv_tables()
assert reliability.audit.status == "passed"

planning = StudyPlanner().independent_mean_power(
    target_difference=2,
    sd_group1=3,
    sd_group2=3,
)
assert planning.status == "available"

repeated_smoke_df = pd.DataFrame(
    {
        "participant": [1, 2, 3, 4, 5, 6] * 3,
        "session": ["s1"] * 6 + ["s2"] * 6 + ["s3"] * 6,
        "score": [
            10.0,
            12.0,
            14.0,
            11.0,
            13.0,
            15.0,
            14.0,
            15.0,
            16.0,
            13.0,
            15.0,
            18.0,
            18.0,
            19.0,
            20.0,
            17.0,
            19.0,
            22.0,
        ],
    }
)
rm_smoke = ResearchAssistant(repeated_smoke_df).run(
    objective="compare_groups",
    outcome="score",
    predictor="session",
    design="repeated",
    estimand="mean",
    unit_id="participant",
    condition_order=("s1", "s2", "s3"),
    variable_types={"score": "continuous", "session": "ordinal"},
)
assert rm_smoke.status.value == "completed"
assert rm_smoke.analysis.method_id == "repeated_measures_anova"
assert "REPEATED-MEASURES ANALYSIS" in rm_smoke.explain()
assert "repeated_anova_omnibus" in rm_smoke.report.to_csv_tables()
assert "repeated_pairwise" in rm_smoke.report.to_csv_tables()
assert rm_smoke.audit.status == "passed"
json.dumps(rm_smoke.to_dict(), allow_nan=False)
json.loads(rm_smoke.to_json())
rm_mult = rm_smoke.analysis.values["multiplicity"]
assert rm_mult["decision_basis"] == "Holm-adjusted p-value < alpha"

friedman_smoke = ResearchAssistant(repeated_smoke_df).run(
    objective="compare_groups",
    outcome="score",
    predictor="session",
    design="repeated",
    estimand="distribution",
    unit_id="participant",
    condition_order=("s1", "s2", "s3"),
    variable_types={"score": "continuous", "session": "ordinal"},
)
assert friedman_smoke.status.value == "completed"
assert friedman_smoke.analysis.method_id == "friedman_test"
assert "OMNIBUS TEST" in friedman_smoke.explain()
assert "friedman_omnibus" in friedman_smoke.report.to_csv_tables()
assert "friedman_pairwise" in friedman_smoke.report.to_csv_tables()
assert friedman_smoke.audit.status == "passed"
json.dumps(friedman_smoke.to_dict(), allow_nan=False)
json.loads(friedman_smoke.to_json())
fr_mult = friedman_smoke.analysis.values["multiplicity"]
assert fr_mult["decision_basis"] == "Holm-adjusted p-value < alpha"

two_way_smoke_df = pd.DataFrame(
    {
        "y": [10.0, 11.0, 12.0, 14.0, 15.0, 16.0, 13.0, 12.0, 18.0, 19.0, 20.0, 21.0, 22.0, 25.0],
        "A": [
            "A1",
            "A1",
            "A1",
            "A2",
            "A2",
            "A2",
            "A2",
            "A3",
            "A3",
            "A3",
            "A3",
            "A3",
            "A3",
            "A3",
        ],
        "B": [
            "B1",
            "B1",
            "B2",
            "B1",
            "B1",
            "B2",
            "B2",
            "B1",
            "B1",
            "B1",
            "B2",
            "B2",
            "B2",
            "B2",
        ],
    }
)
two_way_smoke = ResearchAssistant(two_way_smoke_df).run(
    objective="compare_groups",
    outcome="y",
    factor_a="A",
    factor_b="B",
    design="independent",
    estimand="mean",
    variable_types={"y": "continuous", "A": "nominal", "B": "nominal"},
)
assert two_way_smoke.status.value == "completed"
assert two_way_smoke.analysis.method_id == "two_way_anova"
assert "Two-way factorial ANOVA" in two_way_smoke.explain()
assert "two_way_anova_table" in two_way_smoke.report.to_csv_tables()
assert two_way_smoke.audit.status == "passed"
json.dumps(two_way_smoke.to_dict(), allow_nan=False)
json.loads(two_way_smoke.to_json())

icc_smoke_df = pd.DataFrame(
    {
        "target": ["T1", "T1", "T2", "T2", "T3", "T3"],
        "rater": ["R1", "R2", "R1", "R2", "R1", "R2"],
        "score": [9.0, 2.0, 6.0, 1.0, 8.0, 4.0],
    }
)
icc_smoke = ResearchAssistant(icc_smoke_df).intraclass_correlation(
    target="target",
    rater="rater",
    value="score",
    model="two_way_random",
    definition="absolute_agreement",
    unit="single",
)
assert icc_smoke.status.value == "completed"
assert icc_smoke.analysis.method_id == "intraclass_correlation"
assert "ICC(2,1)" in icc_smoke.explain()
assert "icc_summary" in icc_smoke.report.to_csv_tables()
assert icc_smoke.audit.status == "passed"
json.dumps(icc_smoke.to_dict(), allow_nan=False)
json.loads(icc_smoke.to_json())

html = guided.report.to_html(style="apa")
assert isinstance(html, str) and "<!doctype html>" in html
assert '<section class="executive-summary">' in html
print(f"installed-wheel smoke passed: {pyautostat.__version__} from {module_path}")
