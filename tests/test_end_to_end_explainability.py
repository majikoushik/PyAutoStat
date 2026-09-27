"""Public end-to-end proof for deterministic researcher-facing narration."""

import json
from copy import deepcopy

import pandas as pd

from pyautostat import (
    MeaningfulEffectThreshold,
    ResearchAssistant,
    SensitivitySpecification,
    assumption_grade,
    effect_narrative,
    hypothesis_verdict,
)


def test_public_explainability_workflow_is_local_deterministic_and_nonmutating():
    frame = pd.DataFrame(
        {
            "customer_id": [f"C{i:03d}" for i in range(18)],
            "segment": ["standard"] * 9 + ["new"] * 9,
            "satisfaction": [
                61.0,
                64.0,
                66.0,
                68.0,
                69.0,
                71.0,
                73.0,
                74.0,
                None,
                68.0,
                70.0,
                72.0,
                75.0,
                77.0,
                79.0,
                81.0,
                83.0,
                None,
            ],
            "tenure_months": [3, 7, 9, 12, 18, 22, 27, 31, 36, 4, 8, 11, 17, 21, 26, 30, 35, 40],
        }
    )
    original = frame.copy(deep=True)
    assistant = ResearchAssistant(frame)

    profile = assistant.profile()
    story = assistant.summarize(mode="story")
    workflow = assistant.run(
        objective="compare_groups",
        outcome="satisfaction",
        predictor="segment",
        estimand="mean",
        design="independent",
        data_dictionary={
            "customer_id": {"type": "identifier"},
            "satisfaction": {"type": "continuous", "unit": "points"},
            "segment": {"type": "nominal"},
        },
        include_profile=True,
    )

    assert profile["overview"]["total_rows"] == 18
    assert "DATASET STORY" in story
    assert workflow.status.value == "completed"
    assert workflow.recommendation is not None
    assert workflow.analysis is not None
    assert workflow.interpretation is not None
    assert workflow.report is not None
    assert workflow.profile is not None

    structured_before = deepcopy(workflow.to_dict())
    assert json.loads(workflow.to_json()) == structured_before
    assert isinstance(structured_before["profile"]["overview"]["dtypes"]["customer_id"], str)
    assert workflow.profile["overview"]["dtypes"] == profile["overview"]["dtypes"]
    recommendation_text = workflow.recommendation.rationale_text
    explanation = workflow.explain()

    effect = workflow.analysis.values["effect_size"]
    effect_text = effect_narrative(
        effect["name"],
        effect["value"],
        n=workflow.analysis.sample_size,
        ci=effect["confidence_interval"],
        orientation=workflow.analysis.metadata["contrast"]["definition"],
        definition=effect["definition"],
    )
    hypothesis_text = hypothesis_verdict(
        workflow.analysis.values["p_value"],
        workflow.analysis.specification.options.alpha,
        effect["value"],
        effect["name"],
        n=workflow.analysis.sample_size,
    )
    diagnostics = workflow.analysis.metadata["diagnostics"]
    severity, assumption_text = assumption_grade(
        "equal_variance",
        diagnostics["equal_variance_status"],
        p_value=diagnostics["levene_p_value"],
        method_id=workflow.analysis.method_id,
    )

    practical = assistant.practical_significance(
        workflow.analysis,
        threshold=MeaningfulEffectThreshold(
            "mean_difference",
            3,
            unit="points",
            rationale="A three-point difference would change the service decision.",
        ),
    )
    sensitivity = assistant.sensitivity_analysis(
        workflow.analysis,
        scenarios=[
            SensitivitySpecification(
                "pooled-variance mean comparison",
                workflow.analysis.specification,
                method_id="student_t",
                rationale="Assess the declared pooled-variance assumption.",
                assumptions=("Equal population variances",),
            )
        ],
    )
    report = assistant.report(
        workflow.analysis,
        interpretation=workflow.interpretation,
        practical_significance=practical,
        sensitivity=sensitivity,
        title="Synthetic customer satisfaction comparison",
    )
    html = report.to_html()
    practical_text = practical.verdict
    sensitivity_text = sensitivity.verdict

    assert "WHY THIS TEST?" in recommendation_text
    assert "ANALYSIS RESULT" in explanation
    assert "Groups    : 'standard', 'new'" in explanation
    assert "HYPOTHESIS TEST" in explanation
    assert "EFFECT SIZE" in explanation
    assert "CONFIDENCE INTERVAL" in explanation
    assert "ASSUMPTIONS" in explanation
    assert "LIMITATIONS" in explanation
    assert "Cohen's d" in effect_text
    assert "p =" in hypothesis_text or "p <" in hypothesis_text
    assert severity in {"INFO", "CAUTION", "WARNING", "CRITICAL"}
    assert "equal" in assumption_text.lower()
    assert "VERDICT:" in practical_text
    assert sensitivity_text
    assert '<section class="executive-summary">' in html
    assert "Practical significance was assessed" in html
    assert "Sensitivity analysis was performed" in html

    assert story == assistant.summarize(mode="story")
    assert recommendation_text == workflow.recommendation.rationale_text
    assert explanation == workflow.explain()
    assert effect_text == effect_narrative(
        effect["name"],
        effect["value"],
        n=workflow.analysis.sample_size,
        ci=effect["confidence_interval"],
        orientation=workflow.analysis.metadata["contrast"]["definition"],
        definition=effect["definition"],
    )
    assert hypothesis_text == hypothesis_verdict(
        workflow.analysis.values["p_value"],
        workflow.analysis.specification.options.alpha,
        effect["value"],
        effect["name"],
        n=workflow.analysis.sample_size,
    )
    assert practical_text == practical.verdict
    assert sensitivity_text == sensitivity.verdict
    assert html == report.to_html()
    assert workflow.to_dict() == structured_before
    pd.testing.assert_frame_equal(frame, original)
