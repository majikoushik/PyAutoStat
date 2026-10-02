"""PyAutoStat Rich Terminal Presentation Gallery.

Demonstrates representative terminal presentations across all renderer families:
- Two-Group Comparisons (Welch t-test)
- One-Sample Comparisons (one_sample_t)
- Paired Comparisons (paired_t)
- Multi-Group Comparisons (Welch ANOVA)
- Association Analysis (Pearson correlation)
- Categorical Independence (Pearson chi-square)
- Paired Categorical Transitions (McNemar)
- Linear Regression (OLS)
- Logistic Regression
- Reliability Analysis (Cronbach's alpha)
- Repeated-Measures Analysis (Repeated ANOVA)
- Factorial Analysis (Two-way ANOVA)
- Reliability & Agreement (Intraclass Correlation - ICC)
- Descriptive Tables (frequency_table, cross_tab)
- Study Planning & Governance (StudyPlanningResult, AuditResult)
"""

from __future__ import annotations

import pandas as pd
from rich.console import Console

from pyautostat import ResearchAssistant, StudyPlanner, show

console = Console(width=90)


def gallery_two_group() -> None:
    console.rule("[bold cyan]1. Two-Group Comparison (Welch t-test)[/bold cyan]")
    df = pd.DataFrame(
        {
            "score": [12.0, 14.0, 13.5, 16.0, 15.0, 20.0, 22.5, 21.0, 23.0, 25.0],
            "group": ["Control"] * 5 + ["Treatment"] * 5,
        }
    )
    wf = ResearchAssistant(df).run(
        objective="compare_groups",
        outcome="score",
        predictor="group",
        estimand="mean",
        design="independent",
        variable_types={"score": "continuous", "group": "nominal"},
    )
    show(wf, console=console)


def gallery_one_sample() -> None:
    console.rule("[bold cyan]2. One-Sample Comparison[/bold cyan]")
    df = pd.DataFrame({"score": [10.5, 11.2, 9.8, 12.1, 10.9, 11.5, 10.2, 11.8]})
    wf = ResearchAssistant(df).run(
        objective="compare_reference",
        outcome="score",
        reference_value=10.0,
        estimand="mean",
        design="independent",
        variable_types={"score": "continuous"},
    )
    show(wf, console=console)


def gallery_paired() -> None:
    console.rule("[bold cyan]3. Paired Comparison (Paired t-test)[/bold cyan]")
    df = pd.DataFrame(
        {
            "patient": [f"P{i}" for i in range(1, 9)] * 2,
            "session": ["baseline"] * 8 + ["followup"] * 8,
            "systolic": [
                142,
                138,
                150,
                144,
                136,
                148,
                152,
                140,
                130,
                126,
                135,
                132,
                128,
                134,
                138,
                125,
            ],
        }
    )
    wf = ResearchAssistant(df).run(
        objective="compare_groups",
        outcome="systolic",
        predictor="session",
        estimand="mean",
        design="paired",
        unit_id="patient",
        condition_order=("baseline", "followup"),
        variable_types={"systolic": "continuous", "session": "nominal", "patient": "nominal"},
    )
    show(wf, console=console)


def gallery_multigroup() -> None:
    console.rule("[bold cyan]4. Multi-Group Comparison (Welch ANOVA)[/bold cyan]")
    df = pd.DataFrame(
        {
            "dose": ["Low"] * 6 + ["Medium"] * 6 + ["High"] * 6,
            "response": [
                10.0,
                11.0,
                12.0,
                10.5,
                11.5,
                12.5,
                14.0,
                15.0,
                16.0,
                14.5,
                15.5,
                16.5,
                20.0,
                22.0,
                21.0,
                23.0,
                24.0,
                22.5,
            ],
        }
    )
    wf = ResearchAssistant(df).run(
        objective="compare_groups",
        outcome="response",
        predictor="dose",
        estimand="mean",
        design="independent",
        variable_types={"response": "continuous", "dose": "nominal"},
    )
    show(wf, console=console)


def gallery_categorical() -> None:
    console.rule("[bold cyan]5. Categorical Independence (Pearson Chi-Square)[/bold cyan]")
    df = pd.DataFrame(
        {
            "dept": ["Engineering"] * 30 + ["Marketing"] * 25 + ["Sales"] * 25,
            "preference": (
                ["Remote"] * 20
                + ["Hybrid"] * 10
                + ["Remote"] * 10
                + ["Hybrid"] * 15
                + ["Remote"] * 8
                + ["Hybrid"] * 17
            ),
        }
    )
    wf = ResearchAssistant(df).run(
        objective="association",
        outcome="preference",
        predictor="dept",
        estimand="categorical_independence",
        design="independent",
        variable_types={"preference": "nominal", "dept": "nominal"},
    )
    show(wf, console=console)


def gallery_regression() -> None:
    console.rule("[bold cyan]6. Linear Regression (OLS)[/bold cyan]")
    df = pd.DataFrame(
        {
            "experience": [1.0, 2.0, 3.5, 4.0, 5.0, 6.5, 8.0, 9.5, 10.0, 12.0],
            "certifications": [0, 1, 1, 2, 2, 3, 2, 4, 3, 4],
            "salary_k": [55.0, 62.0, 68.0, 75.0, 82.0, 91.0, 105.0, 118.0, 122.0, 140.0],
        }
    )
    wf = ResearchAssistant(df).run(
        objective="regression",
        outcome="salary_k",
        predictors=["experience", "certifications"],
        estimand="conditional_mean",
        design="independent",
        variable_types={
            "salary_k": "continuous",
            "experience": "continuous",
            "certifications": "continuous",
        },
    )
    show(wf, console=console)


def gallery_reliability_and_agreement() -> None:
    console.rule("[bold cyan]7. Agreement (Intraclass Correlation - ICC)[/bold cyan]")
    df = pd.DataFrame(
        {
            "patient": ["P1", "P1", "P2", "P2", "P3", "P3", "P4", "P4"],
            "rater": ["DrA", "DrB", "DrA", "DrB", "DrA", "DrB", "DrA", "DrB"],
            "severity": [8.5, 8.0, 3.0, 3.5, 6.0, 5.5, 9.0, 9.5],
        }
    )
    icc_res = ResearchAssistant(df).intraclass_correlation(
        target="patient",
        rater="rater",
        value="severity",
        model="two_way_random",
        definition="absolute_agreement",
        unit="single",
    )
    show(icc_res, console=console)


def gallery_planning() -> None:
    console.rule("[bold cyan]8. Prospective Study Planning[/bold cyan]")
    plan = StudyPlanner().independent_mean_power(
        target_difference=2.5,
        sd_group1=3.5,
        sd_group2=3.5,
        target_power=0.85,
        alpha=0.05,
    )
    show(plan, console=console)


def main() -> None:
    console.print("\n[bold magenta]PyAutoStat Presentation Layer — Method Gallery[/bold magenta]\n")
    gallery_two_group()
    gallery_one_sample()
    gallery_paired()
    gallery_multigroup()
    gallery_categorical()
    gallery_regression()
    gallery_reliability_and_agreement()
    gallery_planning()
    console.print("\n[bold green]Gallery complete.[/bold green]\n")


if __name__ == "__main__":
    main()
