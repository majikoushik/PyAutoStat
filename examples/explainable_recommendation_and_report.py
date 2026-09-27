"""Explain one deterministic recommendation and render its stored result."""

import pandas as pd

from pyautostat import ResearchAssistant


frame = pd.DataFrame(
    {
        "group": ["standard"] * 8 + ["new"] * 8,
        "score": [62, 65, 68, 70, 72, 75, 77, 79, 67, 71, 74, 78, 80, 84, 86, 89],
    }
)
assistant = ResearchAssistant(frame)
question = assistant.prepare_question(
    objective="compare_groups",
    outcome="score",
    predictor="group",
    estimand="mean",
    design="independent",
    variable_types={"score": "continuous"},
)

recommendation = assistant.recommend_test(question)
print(recommendation.rationale_text)

result = assistant.analyze(question)
report = assistant.report(result)
html = report.to_html()

assert result.method_id == recommendation.method_id
assert '<section class="executive-summary">' in html
assert "Executive Summary" in html
print(f"\nCompleted {result.method_label}; HTML executive summary rendered in memory.")
