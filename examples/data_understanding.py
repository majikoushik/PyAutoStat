"""Explore synthetic numerical and categorical data without inferential claims."""

import pandas as pd

from pyautostat import ResearchAssistant, column_story

frame = pd.DataFrame(
    {
        "score": [58.0, 64.0, 72.0, 79.0, 91.0, None],
        "region": ["North", "South", "North", "West", "North", "South"],
        "rating": pd.Categorical(
            ["low", "medium", "high", "medium", "high", None],
            categories=["low", "medium", "high"],
            ordered=True,
        ),
        "purchased": [False, True, True, False, True, True],
    }
)

assistant = ResearchAssistant(frame)
profile = assistant.profile()

print("NUMERICAL PROFILE")
print(profile["descriptive"]["score"]["percentiles"])
print(column_story("score", profile["descriptive"]["score"], unit="points"))

print("\nFREQUENCY TABLE")
frequency = assistant.frequency_table("rating")
print(frequency["levels"])
print(frequency["narrative"])

print("\nDESCRIPTIVE CROSS-TAB")
cross_tab = assistant.cross_tab("region", "purchased")
print("Counts:", cross_tab["counts"])
print("Row percentages:", cross_tab["row_percent"])
print("Column percentages:", cross_tab["column_percent"])
print("Total percentages:", cross_tab["total_percent"])
print(cross_tab["narrative"])
