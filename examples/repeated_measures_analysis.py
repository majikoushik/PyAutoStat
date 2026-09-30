"""Repeated-measures workflows for 3+ conditions/time points on the same units."""

import numpy as np
import pandas as pd

from pyautostat import ResearchAssistant

# Generate synthetic repeated-measures study data (e.g., longitudinal clinical cognitive scores)
rng = np.random.default_rng(42)
n_participants = 25
baseline_scores = rng.normal(50.0, 8.0, n_participants)
week4_scores = baseline_scores + rng.normal(4.0, 3.0, n_participants)
week8_scores = week4_scores + rng.normal(3.5, 3.5, n_participants)

rows = []
for i in range(n_participants):
    pid = f"P{i + 1:03d}"
    rows.append({"participant": pid, "session": "baseline", "score": float(baseline_scores[i])})
    rows.append({"participant": pid, "session": "week4", "score": float(week4_scores[i])})
    rows.append({"participant": pid, "session": "week8", "score": float(week8_scores[i])})

repeated_df = pd.DataFrame(rows)

print("=" * 70)
print("1. REPEATED-MEASURES ANOVA (MEAN ESTIMAND)")
print("=" * 70)

# Guided repeated-measures ANOVA
rm_anova = ResearchAssistant(repeated_df).run(
    objective="compare_groups",
    outcome="score",
    predictor="session",
    design="repeated",
    estimand="mean",
    unit_id="participant",
    condition_order=("baseline", "week4", "week8"),
    variable_types={"score": "continuous", "session": "ordinal"},
)
print(rm_anova.explain())

print("\n" + "=" * 70)
print("2. FRIEDMAN TEST (DISTRIBUTION/RANK ESTIMAND)")
print("=" * 70)

# Guided Friedman test for repeated rank comparisons
friedman = ResearchAssistant(repeated_df).run(
    objective="compare_groups",
    outcome="score",
    predictor="session",
    design="repeated",
    estimand="distribution",
    unit_id="participant",
    condition_order=("baseline", "week4", "week8"),
    variable_types={"score": "continuous", "session": "ordinal"},
)
print(friedman.explain())
