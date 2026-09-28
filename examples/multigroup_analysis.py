"""Complete multi-group workflows with unconditional pairwise follow-up families."""

import pandas as pd

from pyautostat import ResearchAssistant, StatisticalAnalyzer

data = pd.DataFrame(
    {
        "programme": ["Gamma"] * 6 + ["Alpha"] * 7 + ["Beta"] * 5,
        "score": [2, 3, 4, 5, 7, 8, 8, 10, 12, 15, 18, 22, 27, 1, 2, 2, 3, 5],
    }
)

print("GUIDED WELCH ANOVA AND GAMES-HOWELL")
mean_workflow = ResearchAssistant(data).run(
    objective="compare_groups",
    outcome="score",
    predictor="programme",
    design="independent",
    estimand="mean",
    variable_types={"score": "continuous", "programme": "nominal"},
)
print(mean_workflow.explain())
print("Structured pairs:", len(mean_workflow.analysis.values["pairwise_comparisons"]))

print("\nGUIDED KRUSKAL-WALLIS AND DUNN-HOLM")
rank_workflow = ResearchAssistant(data).run(
    objective="compare_groups",
    outcome="score",
    predictor="programme",
    design="independent",
    estimand="distribution",
    variable_types={"score": "continuous", "programme": "nominal"},
)
print(rank_workflow.explain())

print("\nEXPLICIT CLASSICAL ANOVA AND TUKEY-KRAMER")
classical = StatisticalAnalyzer(data).hypothesis_tests(
    "programme", "score", test_type="anova", bootstrap_samples=0
)
print(classical["test"], classical["p_value"], classical["pairwise_method"])
for comparison in classical["pairwise_comparisons"]:
    print(
        comparison["group1"],
        "minus",
        comparison["group2"],
        "=",
        round(comparison["estimate"], 3),
        "adjusted p =",
        round(comparison["adjusted_p_value"], 4),
    )
