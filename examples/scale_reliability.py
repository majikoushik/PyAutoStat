"""Run the complete non-inferential scale-reliability workflow."""

import pandas as pd

from pyautostat import ResearchAssistant, reproduce


responses = pd.DataFrame(
    {
        "clarity": [1, 2, 3, 4, 5, 4, 3, 2, 5, 4, 2, 3],
        "usefulness": [1, 2, 4, 4, 5, 4, 3, 2, 5, 3, 2, 3],
        # High scores mean difficulty, so the researcher explicitly reverses this item.
        "difficulty": [5, 4, 3, 2, 1, 2, 3, 4, 1, 2, 4, 3],
        "confidence": [1, 2, 3, 5, 5, 4, 3, 2, 5, 4, 2, 3],
    }
)
responses.loc[10, "confidence"] = None

workflow = ResearchAssistant(responses).reliability(
    items=["clarity", "usefulness", "difficulty", "confidence"],
    reverse_scoring={"difficulty": (1, 5)},
)

print(workflow.explain())
print("Audit:", workflow.audit.status)
print("Report tables:", sorted(workflow.report.to_csv_tables()))
print("Replay:", reproduce(workflow.reproducibility, data=responses).status)
