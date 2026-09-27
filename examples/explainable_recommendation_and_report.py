"""Show the local, deterministic researcher-facing narration workflow."""

import pandas as pd

from pyautostat import ResearchAssistant


frame = pd.DataFrame(
    {
        "group": ["standard"] * 8 + ["new"] * 8,
        "score": [62, 65, 68, 70, 72, 75, 77, 79, 67, 71, 74, 78, 80, 84, 86, 89],
    }
)
assistant = ResearchAssistant(frame)

story = assistant.summarize(mode="story")
print(story)

workflow = assistant.run(
    objective="compare_groups",
    outcome="score",
    predictor="group",
    estimand="mean",
    design="independent",
    variable_types={"score": "continuous"},
)

assert workflow.recommendation is not None
assert workflow.analysis is not None
assert workflow.report is not None

print("\n" + workflow.recommendation.rationale_text)
print("\n" + workflow.explain())

html = workflow.report.to_html()

assert workflow.analysis.method_id == workflow.recommendation.method_id
assert '<section class="executive-summary">' in html
assert "Executive Summary" in html
print(f"\nCompleted {workflow.analysis.method_label}; HTML executive summary rendered in memory.")
