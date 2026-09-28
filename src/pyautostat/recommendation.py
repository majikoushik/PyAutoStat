"""Deterministic design checks and method recommendations; no tests are executed."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .categorical import contingency_counts
from .exceptions import InsufficientDataError, InvalidDataError
from .question_builder import QuestionDraft, QuestionStatus
from .regression import build_design_matrix
from .reliability import calculate_reliability
from .results import MissingInformation, Recommendation, RecommendationStatus
from .specifications import Objective, StudyDesign

_QUESTION_DATA_BLOCKER_KEY = "phase4_data"  # Retained for decision-trace compatibility.


@dataclass(frozen=True)
class MethodCapability:
    """What the current numerical backend actually exposes."""

    identifier: str
    name: str
    objective: str
    target: str
    variable_types: tuple[str, ...]
    designs: tuple[str, ...]
    group_count: str
    minimum_data: str
    assumptions: tuple[str, ...]
    inferential: bool
    availability: str
    limitation: str
    backend: str


METHOD_CAPABILITIES: dict[str, MethodCapability] = {
    item.identifier: item
    for item in (
        MethodCapability(
            "cronbach_alpha",
            "Scale reliability (Cronbach's alpha)",
            "reliability",
            "internal_consistency",
            ("numeric_items",),
            ("unknown",),
            "not_applicable",
            "At least two numeric items and two complete respondents",
            (
                "Researcher-declared scale membership and scoring",
                "Items are suitably scored for covariance-based internal consistency",
                "Complete respondents are relevant to the intended population",
            ),
            False,
            "runnable",
            "Alpha does not establish unidimensionality, validity, or temporal stability.",
            "PyAutoStat Cronbach alpha",
        ),
        MethodCapability(
            "one_sample_t",
            "One-sample t-test",
            "compare_reference",
            "mean",
            ("continuous",),
            ("independent",),
            "not_applicable",
            "At least 2 usable observations and a finite declared reference value",
            (
                "Independent observations",
                "Continuous outcome supporting population-mean inference",
                "Appropriate one-sample mean-inference conditions",
            ),
            True,
            "runnable",
            "The signed contrast is observed sample mean minus the declared reference value.",
            "StatisticalAnalyzer.one_sample_t_test",
        ),
        MethodCapability(
            "dataset_profile",
            "Dataset profile",
            "descriptive",
            "description",
            ("any",),
            ("not_applicable",),
            "not_applicable",
            "Nonempty DataFrame",
            (),
            False,
            "runnable",
            "Descriptive findings are not hypothesis tests.",
            "ResearchAssistant.profile",
        ),
        MethodCapability(
            "linear_regression",
            "Ordinary least-squares linear regression",
            "regression",
            "conditional_mean",
            ("continuous_outcome", "one_or_more_predictors"),
            ("independent",),
            "not_applicable",
            "Full-rank design with positive residual degrees of freedom",
            (
                "Independent observational units",
                "Linear conditional-mean specification",
                "Finite complete-case observations",
                "Appropriate residual and variance conditions for the selected covariance",
            ),
            True,
            "runnable",
            "Conditional associations are not causal or validated out-of-sample predictions.",
            "statsmodels.api.OLS",
        ),
        MethodCapability(
            "welch_t",
            "Welch independent-samples t-test",
            "compare_groups",
            "mean",
            ("quantitative", "group"),
            ("independent",),
            "exactly 2",
            "At least 2 usable values per group and representable spread",
            ("Independent observations", "Appropriate sampling and mean-inference conditions"),
            True,
            "runnable",
            "No design fact is inferred from values.",
            "StatisticalAnalyzer.hypothesis_tests(test_type='ttest', equal_var=False)",
        ),
        MethodCapability(
            "student_t",
            "Student independent-samples t-test",
            "compare_groups",
            "mean",
            ("quantitative", "group"),
            ("independent",),
            "exactly 2",
            "At least 2 usable values per group and representable spread",
            ("Independent observations", "Equal population variances", "Mean-inference conditions"),
            True,
            "runnable",
            "Explicit choice only; equal variance is not proved by Levene.",
            "StatisticalAnalyzer.hypothesis_tests(test_type='ttest', equal_var=True)",
        ),
        MethodCapability(
            "paired_t",
            "Paired-samples t-test",
            "compare_groups",
            "mean",
            ("quantitative", "condition", "unit_identifier"),
            ("paired",),
            "exactly 2 conditions",
            "At least 2 complete pairs with finite nonzero difference variance",
            (
                "Explicit paired or matched units",
                "Independent pairs",
                "Appropriate paired-difference mean-inference conditions",
            ),
            True,
            "runnable",
            "Long-format pairs require one usable observation per unit and condition.",
            "scipy.stats.ttest_rel",
        ),
        MethodCapability(
            "wilcoxon_signed_rank",
            "Paired Wilcoxon signed-rank test",
            "compare_groups",
            "distribution",
            ("ordered_numeric", "condition", "unit_identifier"),
            ("paired",),
            "exactly 2 conditions",
            "At least 2 complete pairs with at least 2 nonzero paired differences",
            (
                "Explicit paired or matched units",
                "Independent pairs",
                "Meaningful ordering and signed paired differences",
                "Symmetric paired-difference distribution for a location-shift interpretation",
            ),
            True,
            "runnable",
            "Zero differences use SciPy's explicit 'wilcox' convention.",
            "StatisticalAnalyzer.paired_wilcoxon",
        ),
        MethodCapability(
            "mann_whitney_u",
            "Mann-Whitney U",
            "compare_groups",
            "distribution",
            ("ordered_numeric", "group"),
            ("independent",),
            "exactly 2",
            "At least 2 usable numeric values per group",
            ("Independent observations", "Meaningful outcome ordering"),
            True,
            "runnable",
            "A rank-distribution comparison, not a universal median test.",
            "StatisticalAnalyzer.hypothesis_tests(test_type='mannwhitney')",
        ),
        MethodCapability(
            "welch_anova",
            "Welch one-way ANOVA with Games-Howell comparisons",
            "compare_groups",
            "mean",
            ("quantitative", "group"),
            ("independent",),
            "3 or more",
            "At least 2 usable values and positive finite variance in every group",
            ("Independent observations", "Appropriate sampling and mean-inference conditions"),
            True,
            "runnable",
            "All Games-Howell pairs are calculated regardless of the omnibus decision.",
            "StatisticalAnalyzer.welch_anova",
        ),
        MethodCapability(
            "one_way_anova",
            "Standard one-way ANOVA",
            "compare_groups",
            "mean",
            ("quantitative", "group"),
            ("independent",),
            "3 or more",
            "At least 2 usable values per group and representable spread",
            ("Independent observations", "Equal population variances", "Mean-inference conditions"),
            True,
            "runnable",
            "Explicit choice requires independently justified variance assumptions.",
            "StatisticalAnalyzer.hypothesis_tests(test_type='anova')",
        ),
        MethodCapability(
            "kruskal_wallis",
            "Kruskal-Wallis",
            "compare_groups",
            "distribution",
            ("ordered_numeric", "group"),
            ("independent",),
            "3 or more",
            "At least 5 usable values per group and two distinct outcome values",
            ("Independent observations", "Meaningful outcome ordering"),
            True,
            "runnable",
            "All Dunn-Holm pairs are calculated; no universal median claim is made.",
            "StatisticalAnalyzer.hypothesis_tests(test_type='kruskal')",
        ),
        MethodCapability(
            "pearson_correlation",
            "Pearson correlation",
            "association",
            "linear",
            ("quantitative", "quantitative"),
            ("independent",),
            "not_applicable",
            "At least 3 complete pairs and variation in each variable",
            ("Independent observational pairs", "Linear-association inference conditions"),
            True,
            "runnable",
            "The profile includes Pearson pairwise p-values; no causal claim.",
            "StatisticalAnalyzer.analyze_all()['correlation']['pearson']",
        ),
        MethodCapability(
            "spearman_correlation",
            "Spearman rank correlation",
            "association",
            "monotonic",
            ("ordered_numeric", "ordered_numeric"),
            ("independent",),
            "not_applicable",
            "At least 3 complete varying numeric pairs",
            ("Independent observational pairs", "Meaningful ordering"),
            True,
            "runnable",
            "Targets monotonic rank association; it does not establish linearity or causation.",
            "StatisticalAnalyzer.spearman_correlation",
        ),
        MethodCapability(
            "spearman_coefficient",
            "Spearman rank correlation",
            "association",
            "monotonic",
            ("numeric", "numeric"),
            ("independent",),
            "not_applicable",
            "At least 2 complete varying numeric pairs",
            ("Meaningful ordering",),
            False,
            "coefficient_only",
            "Only a coefficient matrix is exposed; no inferential p-value.",
            "StatisticalAnalyzer.analyze_all()['correlation']['spearman']",
        ),
        MethodCapability(
            "kendall_coefficient",
            "Kendall rank correlation",
            "association",
            "monotonic",
            ("numeric", "numeric"),
            ("independent",),
            "not_applicable",
            "At least 2 complete varying numeric pairs",
            ("Meaningful ordering",),
            False,
            "coefficient_only",
            "Only a coefficient matrix is exposed; no inferential p-value.",
            "StatisticalAnalyzer.analyze_all()['correlation']['kendall']",
        ),
        MethodCapability(
            "pearson_chi_square",
            "Pearson chi-square test of independence",
            "association",
            "categorical_independence",
            ("categorical", "categorical"),
            ("independent",),
            "At least 2 categories per variable",
            "Every expected cell count at least 5",
            ("Independent observations", "Adequate expected frequencies"),
            True,
            "runnable",
            "Sparse tables are blocked unless the shared observed table is 2x2.",
            "StatisticalAnalyzer.categorical_association",
        ),
        MethodCapability(
            "fisher_exact",
            "Fisher's exact test",
            "association",
            "categorical_independence",
            ("categorical", "categorical"),
            ("independent",),
            "exactly 2 categories per variable",
            "Observed 2x2 table with at least one observation in each margin",
            ("Independent observations", "Fixed or otherwise justified table margins"),
            True,
            "runnable",
            "Supported only for 2x2 tables; the odds-ratio interval is unavailable.",
            "StatisticalAnalyzer.fisher_exact",
        ),
    )
}

_QUANTITATIVE = {"continuous_numerical", "discrete_numerical"}
_CATEGORICAL = {"nominal_categorical", "ordinal_categorical", "boolean"}
_ORDERED = _QUANTITATIVE | {"ordinal_categorical"}


def _alternative(method_id: str, reason: str) -> dict[str, str]:
    item = METHOD_CAPABILITIES[method_id]
    return {
        "method_id": method_id,
        "name": item.name,
        "availability": item.availability,
        "reason": reason,
    }


def _numeric(values: pd.Series) -> np.ndarray | None:
    """Inspect representability without running a statistical test."""
    if not pd.api.types.is_numeric_dtype(values):
        return None
    try:
        result = np.asarray(values, dtype=float)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if np.isfinite(result).all() else None


def _spread(values: np.ndarray) -> float | None:
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        variance = float(np.var(values))
    return variance if math.isfinite(variance) else None


def _group_sizes(usable: pd.DataFrame, group: str) -> list[int]:
    codes, groups = pd.factorize(usable[group], sort=False)
    return np.bincount(codes, minlength=len(groups)).astype(int).tolist()


def recommend_from_draft(frame: pd.DataFrame, draft: QuestionDraft) -> Recommendation:
    """Evaluate a question draft using lightweight deterministic checks only."""
    spec = draft.specification
    question = spec.question
    objective = question.objective
    selected = list(
        dict.fromkeys(
            name
            for name in (
                question.outcome,
                question.predictor,
                *(question.predictors or ()),
                *(question.items or ()),
            )
            if name is not None
        )
    )
    recorded_types = spec.analytical_variable_types or {}
    types = {
        name: recorded_types.get(name, draft.variable_suggestions[name]["suggested_type"])
        for name in selected
    }
    context: dict[str, Any] = {
        "objective": objective.value if objective is not None else None,
        "estimand": question.estimand,
        "design": spec.design.value,
        "variable_types": types,
        "availability": draft.availability,
        "assumption_checks": [],
    }
    trace: list[dict[str, Any]] = []
    warnings = list(draft.warnings)

    def record(key: str, value: Any, reason: str) -> None:
        trace.append({"key": key, "value": value, "reason": reason})

    def finish(
        status: RecommendationStatus,
        *,
        method_id: str | None = None,
        rationale: str | None = None,
        blockers: tuple[str, ...] = (),
        missing: tuple[MissingInformation, ...] = (),
        questions: tuple[dict[str, Any], ...] = (),
        alternatives: tuple[dict[str, str], ...] = (),
    ) -> Recommendation:
        capability = METHOD_CAPABILITIES.get(method_id) if method_id else None
        if status == RecommendationStatus.READY and capability is not None:
            for assumption in capability.assumptions:
                if "Independent" in assumption:
                    continue
                context["assumption_checks"].append(
                    {
                        "assumption": assumption,
                        "category": "data_checkable"
                        if "expected frequencies" in assumption.lower()
                        else "not_fully_checkable",
                        "status": "met"
                        if "expected frequencies" in assumption.lower()
                        else "requires_review",
                    }
                )
        return Recommendation(
            status=status,
            method_id=method_id,
            method_name=capability.name if capability else None,
            rationale=rationale,
            required_assumptions=capability.assumptions if capability else (),
            missing_information=missing,
            blockers=blockers,
            warnings=tuple(warnings),
            method_availability=capability.availability if capability else "unavailable",
            decision_trace=tuple(trace),
            alternatives=alternatives,
            context=context,
            questions=questions,
        )

    record("objective", context["objective"], "Researcher-declared research objective.")
    record("variables", selected, "Selected variable roles are preserved from the specification.")
    record("types", types, "Profile suggestions or validated researcher declarations.")
    record("target", question.estimand, "The scientific target is never inferred from diagnostics.")
    record("design", spec.design.value, "Dependence is taken only from the researcher declaration.")
    record("availability", draft.availability, "Counts use rows complete for selected variables.")

    if draft.status == QuestionStatus.DATA_LIMITED or draft.status == QuestionStatus.UNSUPPORTED:
        record(
            "blocker", _QUESTION_DATA_BLOCKER_KEY, "Question intake found unusable selected data."
        )
        return finish(
            RecommendationStatus.UNSUPPORTED,
            blockers=draft.blockers,
            missing=draft.missing_information,
            questions=tuple(item.to_dict() for item in draft.questions),
            rationale="The selected dataset cannot support this question as supplied.",
        )
    if draft.status == QuestionStatus.NEEDS_INPUT:
        record(
            "clarification",
            [item.field for item in draft.questions],
            "Essential research information is unresolved.",
        )
        return finish(
            RecommendationStatus.NEEDS_INPUT,
            missing=draft.missing_information,
            questions=tuple(item.to_dict() for item in draft.questions),
            rationale="Answer the listed questions before choosing a method.",
        )

    if objective == Objective.DESCRIPTIVE:
        record("method", "dataset_profile", "Description does not require an inferential test.")
        return finish(
            RecommendationStatus.READY,
            method_id="dataset_profile",
            rationale="The existing dataset profile summarizes the selected data without "
            "an inferential hypothesis test.",
        )

    if objective == Objective.RELIABILITY:
        items = question.items or ()
        context.update(
            {
                "items": list(items),
                "missing_data_policy": "complete cases across all selected items",
                "bootstrap_samples": spec.options.bootstrap_samples,
                "random_state": 0 if spec.options.random_seed is None else spec.options.random_seed,
                "reverse_scoring": spec.options.reverse_scoring or {},
            }
        )
        if question.estimand != "internal_consistency":
            return finish(
                RecommendationStatus.UNSUPPORTED,
                blockers=("Reliability supports only estimand='internal_consistency'.",),
                rationale="The requested scale property was preserved rather than substituted.",
            )
        identifiers = [item for item in items if types.get(item) == "identifier"]
        if identifiers:
            return finish(
                RecommendationStatus.UNSUPPORTED,
                blockers=(f"Identifier columns cannot be scale items: {identifiers!r}.",),
                rationale="Identifiers are not questionnaire scores.",
            )
        unsupported = [
            item
            for item in items
            if types.get(item)
            not in {
                "continuous_numerical",
                "discrete_numerical",
                "ordinal_categorical",
                "boolean",
            }
        ]
        if unsupported:
            return finish(
                RecommendationStatus.UNSUPPORTED,
                blockers=(
                    "Cronbach's alpha requires explicitly numeric scored items; no ordinal "
                    f"label encoding is inferred for {unsupported!r}.",
                ),
                rationale="Item labels require researcher-supplied numeric scoring before use.",
            )
        for item in items:
            codes = (spec.data_dictionary or {}).get(item, {}).get("missing_codes", [])
            if codes and int(frame[item].isin(codes).sum()):
                return finish(
                    RecommendationStatus.UNSUPPORTED,
                    blockers=(
                        f"{item!r} contains declared missing-code observations; normalize "
                        "them before reliability analysis.",
                    ),
                    rationale="Declared missing codes are not silently treated as item scores.",
                )
        try:
            preview = calculate_reliability(
                frame,
                items,
                confidence_level=spec.options.confidence_level,
                bootstrap_samples=1,
                random_state=spec.options.random_seed,
                reverse_scoring=spec.options.reverse_scoring,
            )
        except (InsufficientDataError, InvalidDataError) as exc:
            return finish(
                RecommendationStatus.UNSUPPORTED,
                blockers=(str(exc),),
                rationale="The supplied item matrix cannot support a finite alpha estimate.",
            )
        context["available_observations"] = preview["sample"]["analyzed_rows"]
        context["excluded_rows"] = preview["sample"]["excluded_rows"]
        record("method", "cronbach_alpha", "Researcher-declared multi-item scale target.")
        return finish(
            RecommendationStatus.READY,
            method_id="cronbach_alpha",
            rationale=(
                "You supplied multiple scored items as one proposed scale. Cronbach's alpha "
                "estimates their internal consistency; it does not establish dimensionality "
                "or validity."
            ),
        )

    assert objective is not None
    if objective == Objective.COMPARE_REFERENCE:
        assert question.outcome is not None
        return _compare_reference(frame, question.outcome, types, spec, context, record, finish)
    if objective == Objective.REGRESSION:
        assert question.outcome is not None and question.predictors is not None
        predictors = question.predictors
        columns = [question.outcome, *predictors]
        context.update(
            {
                "outcome": question.outcome,
                "predictors": list(predictors),
                "covariance_type": spec.options.covariance_type,
                "reference_levels": spec.options.reference_levels or {},
                "intercept": True,
                "missing_data_policy": "complete cases across outcome and all predictors",
            }
        )
        if spec.design != StudyDesign.INDEPENDENT:
            return finish(
                RecommendationStatus.UNSUPPORTED,
                blockers=(
                    "OLS in this phase requires independent observational units; repeated, "
                    "paired, and clustered regression are unsupported.",
                ),
                rationale="The declared dependence structure requires another model.",
            )
        if question.estimand != "conditional_mean":
            return finish(
                RecommendationStatus.UNSUPPORTED,
                blockers=("OLS regression supports only estimand='conditional_mean'.",),
                rationale="The stated target was preserved rather than changed.",
            )
        if types.get(question.outcome) != "continuous_numerical":
            return finish(
                RecommendationStatus.UNSUPPORTED,
                blockers=("OLS regression requires a continuous numerical outcome.",),
                rationale="A categorical outcome is not silently passed to linear regression.",
            )
        allowed = _QUANTITATIVE | _CATEGORICAL
        unsupported = [name for name in predictors if types.get(name) not in allowed]
        if unsupported:
            return finish(
                RecommendationStatus.UNSUPPORTED,
                blockers=(f"Unsupported regression predictor types for: {unsupported!r}.",),
                rationale="Identifiers, datetimes, and unknown types are not model terms.",
            )
        for column in columns:
            codes = (spec.data_dictionary or {}).get(column, {}).get("missing_codes", [])
            if codes and int(frame[column].isin(codes).sum()):
                return finish(
                    RecommendationStatus.UNSUPPORTED,
                    blockers=(
                        f"{column!r} contains declared missing-code observations; normalize "
                        "them before regression.",
                    ),
                    rationale="The complete-case sample must use actual missing values.",
                )
        available = int(frame[columns].notna().all(axis=1).sum())
        if available < 3:
            return finish(
                RecommendationStatus.UNSUPPORTED,
                blockers=("Regression requires at least three complete cases.",),
                rationale="There are too few complete observations to fit the model.",
            )
        context["available_observations"] = available
        context["excluded_rows"] = int(len(frame) - available)
        try:
            design = build_design_matrix(
                frame,
                question.outcome,
                predictors,
                types,
                reference_levels=spec.options.reference_levels,
                data_dictionary=spec.data_dictionary,
            )
        except (InsufficientDataError, InvalidDataError) as exc:
            return finish(
                RecommendationStatus.UNSUPPORTED,
                blockers=(str(exc),),
                rationale=(
                    "The declared regression design cannot support identifiable OLS inference; "
                    "no predictor was dropped or recoded automatically."
                ),
            )
        context["design_feasibility"] = {
            "rank": design["rank"],
            "parameter_count": design["parameter_count"],
            "residual_degrees_of_freedom": design["residual_df"],
            "full_rank": True,
        }
        record("method", "linear_regression", "Declared conditional-mean regression target.")
        record(
            "covariance_type",
            spec.options.covariance_type,
            "Researcher-selected inference; diagnostics do not switch it.",
        )
        return finish(
            RecommendationStatus.READY,
            method_id="linear_regression",
            rationale=(
                "OLS models the continuous outcome's conditional mean using the ordered "
                "researcher-supplied predictors. Categorical predictors use recorded treatment "
                "coding; no variable selection or causal interpretation is performed."
            ),
        )
    assert question.outcome is not None and question.predictor is not None
    required_columns = [question.outcome, question.predictor]
    if spec.design == StudyDesign.PAIRED and spec.unit_id is not None:
        required_columns.append(spec.unit_id)
    usable = frame[required_columns].dropna()

    for column in required_columns:
        codes = (spec.data_dictionary or {}).get(column, {}).get("missing_codes", [])
        if codes:
            count = int(frame[column].isin(codes).sum())
            if count:
                record("blocker", "unapplied_missing_codes", "Declared codes remain observed.")
                return finish(
                    RecommendationStatus.UNSUPPORTED,
                    blockers=(
                        f"{column!r} contains {count} declared missing-code observations. "
                        "Normalize them explicitly and rebuild the assistant before "
                        "using these sample counts.",
                    ),
                    rationale="Declared missing codes are not removed from the current data.",
                )

    if spec.design == StudyDesign.PAIRED:
        if objective != Objective.COMPARE_GROUPS or question.estimand not in {
            "mean",
            "distribution",
        }:
            return finish(
                RecommendationStatus.UNSUPPORTED,
                blockers=(
                    "Paired support is limited to two-condition mean or rank-distribution "
                    "comparisons.",
                ),
                rationale="The declared paired target has no supported method.",
            )
        assert spec.unit_id is not None
        return _paired_compare(
            frame,
            question.outcome,
            question.predictor,
            spec.unit_id,
            spec.condition_order,
            types,
            context,
            record,
            finish,
            warnings,
        )

    context["assumption_checks"].append(
        {
            "assumption": "independent_observations",
            "category": "researcher_design_fact",
            "status": "confirmed" if spec.design == StudyDesign.INDEPENDENT else "unresolved",
        }
    )
    if spec.design != StudyDesign.INDEPENDENT:
        family = {
            StudyDesign.PAIRED: "paired comparison or paired association methods",
            StudyDesign.REPEATED: "repeated-measures methods",
            StudyDesign.CLUSTERED: "cluster-aware methods",
        }.get(spec.design, "a method matching the study design")
        record(
            "blocker", "dependent_design", "Current inferential methods assume independent units."
        )
        return finish(
            RecommendationStatus.UNSUPPORTED,
            blockers=(
                f"{spec.design.value} observations need {family}; the current "
                "independent-observation methods are incompatible.",
            ),
            rationale="Dependence among observations changes the sampling uncertainty.",
        )

    if any(kind == "identifier" for kind in types.values()):
        record("blocker", "identifier_type", "An identifier is not an analytical measurement.")
        return finish(
            RecommendationStatus.UNSUPPORTED,
            blockers=(
                "A selected variable is declared or suggested as an identifier. "
                "Correct its type if it is actually a measurement.",
            ),
            rationale="Numeric storage alone does not make an identifier quantitative.",
        )

    if objective == Objective.COMPARE_GROUPS:
        return _compare(
            usable,
            question.outcome,
            question.predictor,
            types,
            question.estimand,
            context,
            record,
            finish,
            warnings,
        )
    return _association(
        usable,
        question.outcome,
        question.predictor,
        types,
        question.estimand,
        context,
        record,
        finish,
        warnings,
    )


def _compare_reference(
    frame: pd.DataFrame,
    outcome: str,
    types: dict[str, str],
    specification: Any,
    context: dict[str, Any],
    record: Any,
    finish: Any,
) -> Recommendation:
    question = specification.question
    if specification.design != StudyDesign.INDEPENDENT:
        return finish(
            RecommendationStatus.UNSUPPORTED,
            blockers=("One-sample mean inference requires independent observational units.",),
            rationale="A paired, repeated, clustered, or unresolved design needs another model.",
        )
    if question.estimand != "mean":
        return finish(
            RecommendationStatus.UNSUPPORTED,
            blockers=("The one-sample workflow currently supports only a population-mean target.",),
            rationale="The declared target was preserved rather than changed.",
        )
    if types.get(outcome) != "continuous_numerical":
        return finish(
            RecommendationStatus.UNSUPPORTED,
            blockers=("One-sample t inference requires an analytically continuous outcome.",),
            rationale="A stored number is not silently treated as a continuous measurement.",
        )
    if question.reference_value is None:
        return finish(
            RecommendationStatus.NEEDS_INPUT,
            missing=(MissingInformation("reference_value", "Supply a finite reference value."),),
            rationale="The null reference must be declared by the researcher.",
        )
    codes = (specification.data_dictionary or {}).get(outcome, {}).get("missing_codes", [])
    if codes and int(frame[outcome].isin(codes).sum()):
        return finish(
            RecommendationStatus.UNSUPPORTED,
            blockers=("Declared missing codes remain observed in the outcome.",),
            rationale="Normalize declared missing codes before mean inference.",
        )
    values = _numeric(frame[outcome].dropna())
    if values is None or len(values) < 2:
        return finish(
            RecommendationStatus.UNSUPPORTED,
            blockers=("At least two finite observed outcome values are required.",),
            rationale="The mean contrast and its sampling uncertainty need observed data.",
        )
    context.update(
        {
            "reference_value": question.reference_value,
            "available_observations": int(len(values)),
            "excluded_rows": int(len(frame) - len(values)),
        }
    )
    context["assumption_checks"].append(
        {
            "assumption": "independent_observations",
            "category": "researcher_design_fact",
            "status": "confirmed",
        }
    )
    record("reference_value", question.reference_value, "Researcher-declared null mean.")
    record("method", "one_sample_t", "Continuous outcome and declared mean target.")
    return finish(
        RecommendationStatus.READY,
        method_id="one_sample_t",
        rationale=(
            "One continuous outcome is compared with the declared reference mean; the signed "
            "estimate is observed sample mean minus reference value."
        ),
    )


def _paired_compare(
    frame: pd.DataFrame,
    outcome: str,
    condition: str,
    unit_id: str,
    condition_order: tuple[Any, Any] | None,
    types: dict[str, str],
    context: dict[str, Any],
    record: Any,
    finish: Any,
    warnings: list[str],
) -> Recommendation:
    target = context.get("estimand")
    valid_types = _QUANTITATIVE if target == "mean" else _ORDERED
    if types[outcome] not in valid_types or _numeric(frame[outcome].dropna()) is None:
        return finish(
            RecommendationStatus.UNSUPPORTED,
            blockers=("A paired comparison needs an ordered numeric outcome.",),
            rationale="The declared paired target requires meaningful numeric ordering.",
        )
    observed_conditions = list(pd.unique(frame[condition].dropna()))
    if len(observed_conditions) != 2:
        return finish(
            RecommendationStatus.UNSUPPORTED,
            blockers=("Paired inference requires exactly two observed condition levels.",),
            rationale="More than two repeated conditions need a different dependent-design method.",
        )
    order = list(condition_order) if condition_order is not None else observed_conditions
    if len(order) != 2 or set(order) != set(observed_conditions):
        return finish(
            RecommendationStatus.UNSUPPORTED,
            blockers=("condition_order must name the two observed condition levels exactly.",),
            rationale="The signed paired contrast must have an explicit valid orientation.",
        )
    usable = frame[[unit_id, condition, outcome]].dropna()
    if usable.duplicated([unit_id, condition]).any():
        return finish(
            RecommendationStatus.UNSUPPORTED,
            blockers=(
                "At least one unit has multiple usable observations in the same condition; "
                "PyAutoStat will not average duplicates automatically.",
            ),
            rationale="Aggregation within a unit and condition is a scientific decision.",
        )
    pivot = usable.pivot(index=unit_id, columns=condition, values=outcome)
    complete = pivot.dropna(subset=order)
    complete_pairs = int(len(complete))
    total_units = int(frame[unit_id].dropna().nunique())
    incomplete_units = total_units - complete_pairs
    missing_unit_rows = int(frame[unit_id].isna().sum())
    order_labels = [str(item) for item in order]
    context.update(
        {
            "unit_id": unit_id,
            "condition_order": order_labels,
            "total_units": total_units,
            "complete_pairs": complete_pairs,
            "incomplete_units": incomplete_units,
            "missing_unit_rows": missing_unit_rows,
        }
    )
    record(
        "condition_order",
        order_labels,
        "Observed order or explicit researcher declaration.",
    )
    record("complete_pairs", complete_pairs, "Only units observed in both conditions are usable.")
    if incomplete_units:
        warnings.append(
            f"{incomplete_units} unit(s) without both conditions were excluded from "
            "paired analysis."
        )
    if missing_unit_rows:
        warnings.append(
            f"{missing_unit_rows} row(s) with missing unit identifiers could not be paired."
        )
    if complete_pairs < 2:
        return finish(
            RecommendationStatus.UNSUPPORTED,
            blockers=("At least two complete pairs are required for paired inference.",),
            rationale="The paired methods require complete within-unit contrasts.",
        )
    first = _numeric(complete[order[0]])
    second = _numeric(complete[order[1]])
    if first is None or second is None:
        return finish(
            RecommendationStatus.UNSUPPORTED,
            blockers=("Complete paired outcomes must be finite numeric values.",),
            rationale="The paired numerical backend cannot use nonfinite outcomes.",
        )
    differences = first - second
    if target == "distribution":
        nonzero = int(np.count_nonzero(differences))
        context["zero_differences"] = int(len(differences) - nonzero)
        context["nonzero_differences"] = nonzero
        if nonzero < 2:
            return finish(
                RecommendationStatus.UNSUPPORTED,
                blockers=(
                    "Wilcoxon signed-rank inference needs at least two nonzero paired differences.",
                ),
                rationale="Zero differences are omitted under the recorded 'wilcox' policy.",
            )
        context["assumption_checks"].append(
            {
                "assumption": "independent_pairs",
                "category": "researcher_design_fact",
                "status": "confirmed",
            }
        )
        record(
            "method",
            "wilcoxon_signed_rank",
            "Explicit paired design and two-condition rank-distribution target.",
        )
        return finish(
            RecommendationStatus.READY,
            method_id="wilcoxon_signed_rank",
            rationale=(
                "The declared paired design, unit identifier, condition order, and "
                "rank-distribution target support a Wilcoxon signed-rank analysis."
            ),
            alternatives=(
                _alternative("paired_t", "Targets the population mean paired difference."),
            ),
        )
    spread = _spread(differences)
    if spread is None or spread <= 0:
        return finish(
            RecommendationStatus.UNSUPPORTED,
            blockers=("Paired differences must have finite nonzero variation.",),
            rationale="Zero paired-difference variance makes the paired standard error undefined.",
        )
    context["assumption_checks"].append(
        {
            "assumption": "independent_pairs",
            "category": "researcher_design_fact",
            "status": "confirmed",
        }
    )
    record("method", "paired_t", "Explicit paired design and two-condition mean estimand.")
    return finish(
        RecommendationStatus.READY,
        method_id="paired_t",
        rationale=(
            "The declared paired design, explicit unit identifier, two conditions, and mean "
            "estimand support a paired-samples t-test on complete paired differences."
        ),
    )


def _compare(
    usable: pd.DataFrame,
    outcome: str,
    group: str,
    types: dict[str, str],
    target: str | None,
    context: dict[str, Any],
    record: Any,
    finish: Any,
    warnings: list[str],
) -> Recommendation:
    counts = _group_sizes(usable, group)
    context["group_sizes"] = counts
    record("group_count", len(counts), "Observed groups use rows with both variables present.")
    if len(counts) < 2:
        return finish(
            RecommendationStatus.UNSUPPORTED,
            blockers=("Fewer than two groups have usable outcome values.",),
            rationale="The current comparison methods need at least two complete groups.",
        )
    if types[group] == "datetime":
        return finish(
            RecommendationStatus.UNSUPPORTED,
            blockers=("A datetime column is not a grouping category for these methods.",),
            rationale="Choose or declare a meaningful grouping variable.",
        )
    if min(counts) < 2:
        return finish(
            RecommendationStatus.UNSUPPORTED,
            blockers=("Each observed group needs at least two usable outcomes.",),
            rationale="The current comparison implementations require two values per group.",
        )
    if target not in ("mean", "distribution"):
        return finish(
            RecommendationStatus.UNSUPPORTED,
            blockers=(f"Comparison target {target!r} is not supported by the current methods.",),
            rationale="The target was preserved rather than replaced with a different one.",
        )
    outcome_type = types[outcome]
    numeric = _numeric(usable[outcome])
    if target == "mean" and (outcome_type not in _QUANTITATIVE or numeric is None):
        return finish(
            RecommendationStatus.UNSUPPORTED,
            blockers=("Mean comparison needs a genuinely quantitative numeric outcome.",),
            rationale="An ordinal, nominal, or nonnumeric outcome is not a mean measurement.",
        )
    if target == "distribution" and (outcome_type not in _ORDERED or numeric is None):
        return finish(
            RecommendationStatus.UNSUPPORTED,
            blockers=(
                "The implemented rank comparison needs an ordered numeric outcome. "
                "Unordered labels and unencoded ordered categories cannot be used.",
            ),
            rationale="Outcome ordering must be meaningful and accepted by the backend.",
        )
    assert numeric is not None
    codes, labels = pd.factorize(usable[group], sort=False)
    values = [numeric[codes == index] for index in range(len(labels))]
    if target == "mean":
        variances = [_spread(value) for value in values]
        with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
            mean_difference = float(
                np.max([np.mean(v) for v in values]) - np.min([np.mean(v) for v in values])
            )
        if any(value is None for value in variances) or not math.isfinite(mean_difference):
            return finish(
                RecommendationStatus.UNSUPPORTED,
                blockers=(
                    "Numeric spread or mean differences are not representable "
                    "at this scale; rescale and recheck the data.",
                ),
                rationale="The current mean-comparison backend cannot guarantee a finite result.",
            )
        if all(value == 0 for value in variances):
            return finish(
                RecommendationStatus.UNSUPPORTED,
                blockers=("Every group has zero representable within-group variation.",),
                rationale="The current mean-comparison backend rejects this case.",
            )
        if len(counts) == 2:
            assert variances[0] is not None and variances[1] is not None
            with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
                pooled = ((counts[0] - 1) * variances[0] + (counts[1] - 1) * variances[1]) / (
                    counts[0] + counts[1] - 2
                )
                standardized = (
                    mean_difference / math.sqrt(pooled)
                    if math.isfinite(pooled) and pooled > 0
                    else math.inf
                )
            if not math.isfinite(standardized):
                return finish(
                    RecommendationStatus.UNSUPPORTED,
                    blockers=("The standardized mean effect is not representable at this scale.",),
                    rationale="The current t-test result contract requires a finite effect size.",
                )
            record("method", "welch_t", "Two independent groups and a stated mean target.")
            return finish(
                RecommendationStatus.READY,
                method_id="welch_t",
                rationale="Welch's test targets the difference in means across two independent "
                "groups without requiring equal population variances.",
                alternatives=(
                    _alternative(
                        "student_t", "Requires an additional justified equal-variance assumption."
                    ),
                    _alternative(
                        "mann_whitney_u", "Targets rank distributions, not the stated mean."
                    ),
                ),
            )
        if any(value == 0 for value in variances):
            return finish(
                RecommendationStatus.UNSUPPORTED,
                blockers=(
                    "Welch ANOVA and Games-Howell require positive within-group variance "
                    "in every group.",
                ),
                rationale="A zero-variance group makes variance-weighted inference undefined.",
            )
        record("method", "welch_anova", "Three or more groups and a stated mean target.")
        return finish(
            RecommendationStatus.READY,
            method_id="welch_anova",
            rationale="Welch ANOVA compares population means without requiring equal variances; "
            "Games-Howell provides the complete multiplicity-controlled pairwise family.",
            alternatives=(
                _alternative(
                    "one_way_anova",
                    "Runnable only by explicit choice with its assumptions justified.",
                ),
                _alternative("kruskal_wallis", "Targets rank distributions, not population means."),
            ),
        )
    if np.unique(numeric).size < 2:
        return finish(
            RecommendationStatus.UNSUPPORTED,
            blockers=("The ordered outcome has no observed variation.",),
            rationale="A rank-distribution comparison is undefined here.",
        )
    if len(counts) == 2:
        if min(counts) < 10 and np.unique(numeric).size < len(numeric):
            warnings.append(
                "Small tied samples may have an unreliable asymptotic Mann-Whitney p-value."
            )
        record(
            "method", "mann_whitney_u", "Two independent groups and an ordered distribution target."
        )
        return finish(
            RecommendationStatus.READY,
            method_id="mann_whitney_u",
            rationale="Mann-Whitney compares ordered rank distributions across "
            "two independent groups.",
            alternatives=(
                _alternative("welch_t", "Targets means rather than the stated distribution."),
            ),
        )
    if min(counts) < 5:
        return finish(
            RecommendationStatus.UNSUPPORTED,
            blockers=(
                "Kruskal-Wallis requires at least five usable observations "
                "per group under this library's approximation policy.",
            ),
            rationale="The current rank-test implementation rejects smaller groups.",
        )
    record("method", "kruskal_wallis", "Three or more groups and an ordered distribution target.")
    return finish(
        RecommendationStatus.READY,
        method_id="kruskal_wallis",
        rationale="Kruskal-Wallis compares rank distributions across independent groups; "
        "Dunn-Holm provides the complete multiplicity-controlled pairwise family.",
        alternatives=(
            _alternative("welch_anova", "Targets population means rather than distributions."),
            _alternative("one_way_anova", "Targets means and requires equal-variance assumptions."),
        ),
    )


def _association(
    usable: pd.DataFrame,
    first: str,
    second: str,
    types: dict[str, str],
    target: str | None,
    context: dict[str, Any],
    record: Any,
    finish: Any,
    warnings: list[str],
) -> Recommendation:
    left, right = types[first], types[second]
    numeric_pair = left in _QUANTITATIVE and right in _QUANTITATIVE
    ordered_numeric_pair = (
        left in _ORDERED
        and right in _ORDERED
        and pd.api.types.is_numeric_dtype(usable[first])
        and pd.api.types.is_numeric_dtype(usable[second])
    )
    categorical_pair = left in _CATEGORICAL and right in _CATEGORICAL
    n = len(usable)
    context["complete_pairs"] = n
    record("association_types", [left, right], "Declared or suggested analytical types.")
    if ordered_numeric_pair:
        if target is None:
            question = {
                "field": "estimand",
                "question": "What type of relationship interests you?",
                "explanation": "Linear and monotonic relationships answer different questions.",
                "input_type": "select",
                "required": True,
                "options": [
                    {"value": "linear", "label": "Linear relationship"},
                    {"value": "monotonic", "label": "Monotonic relationship"},
                    {"value": "unknown", "label": "I am not sure"},
                ],
            }
            record("clarification", "estimand", "A numeric relationship target is essential.")
            return finish(
                RecommendationStatus.NEEDS_INPUT,
                missing=(MissingInformation("estimand", str(question["question"])),),
                questions=(question,),
                rationale="Choose a relationship target before selecting a correlation method.",
            )
        if target == "monotonic":
            if n < 3 or any(usable[column].nunique() < 2 for column in (first, second)):
                return finish(
                    RecommendationStatus.UNSUPPORTED,
                    blockers=(
                        "Spearman inference needs at least three complete, varying numeric pairs.",
                    ),
                    rationale="An inferential coefficient needs paired variation and sample size.",
                )
            arrays = [_numeric(usable[column]) for column in (first, second)]
            if any(item is None for item in arrays):
                return finish(
                    RecommendationStatus.UNSUPPORTED,
                    blockers=("The ordered pair must contain finite numeric values.",),
                    rationale="The Spearman backend cannot use nonfinite or unencoded values.",
                )
            record("method", "spearman_correlation", "Explicit monotonic association target.")
            return finish(
                RecommendationStatus.READY,
                method_id="spearman_correlation",
                rationale=(
                    "Spearman inference targets the declared monotonic rank association between "
                    "the two ordered numeric variables."
                ),
                alternatives=(
                    _alternative(
                        "pearson_correlation",
                        "Targets linear rather than monotonic association.",
                    ),
                ),
            )
        if target != "linear":
            return finish(
                RecommendationStatus.UNSUPPORTED,
                blockers=(f"Association target {target!r} is not supported.",),
                rationale="The stated relationship target was not reinterpreted.",
            )
        if not numeric_pair:
            return finish(
                RecommendationStatus.UNSUPPORTED,
                blockers=(
                    "Pearson linear inference needs quantitative measurements; "
                    "an ordinal code is not automatically a continuous scale.",
                ),
                rationale="The declared ordinal measurement type is preserved.",
            )
        if n < 3:
            return finish(
                RecommendationStatus.UNSUPPORTED,
                blockers=("Pearson inference needs at least three complete pairs.",),
                rationale="The existing backend does not report a p-value below this count.",
            )
        arrays = [_numeric(usable[column]) for column in (first, second)]
        if any(item is None for item in arrays):
            return finish(
                RecommendationStatus.UNSUPPORTED,
                blockers=("The numeric pair is not representable as finite real values.",),
                rationale="Pearson inference cannot use nonfinite numerical input.",
            )
        if any(_spread(item) in (None, 0) for item in arrays if item is not None):
            return finish(
                RecommendationStatus.UNSUPPORTED,
                blockers=(
                    "Pearson correlation needs variation in both variables "
                    "at a representable numerical scale.",
                ),
                rationale="A constant or numerically undefined pair has no usable coefficient.",
            )
        warnings.append(
            "Pearson inference can be unreliable for near-constant or unusual pairs; "
            "review the resulting diagnostics before interpretation."
        )
        record("method", "pearson_correlation", "Linear target and two varying numeric variables.")
        return finish(
            RecommendationStatus.READY,
            method_id="pearson_correlation",
            rationale="Pearson addresses linear association between two quantitative variables; "
            "the existing profile reports pairwise p-values when numerically valid.",
            alternatives=(
                _alternative(
                    "spearman_correlation",
                    "Targets monotonic rather than linear association.",
                ),
            ),
        )
    if categorical_pair:
        if target not in (None, "categorical_independence"):
            return finish(
                RecommendationStatus.UNSUPPORTED,
                blockers=(f"Target {target!r} is incompatible with categorical independence.",),
                rationale="Categorical association is not a linear correlation.",
            )
        table = contingency_counts(usable, first, second)
        observed = table["counts"]
        context["contingency_shape"] = list(observed.shape)
        context["row_levels"] = table["row_levels"]
        context["column_levels"] = table["column_levels"]
        if observed.shape[0] < 2 or observed.shape[1] < 2:
            return finish(
                RecommendationStatus.UNSUPPORTED,
                blockers=(
                    "Chi-square needs at least two observed categories for each selected variable.",
                ),
                rationale="The current contingency table has only one level on an axis.",
            )
        row_totals = observed.sum(axis=1)
        col_totals = observed.sum(axis=0)
        expected = np.outer(row_totals, col_totals) / n
        minimum = float(expected.min())
        context["minimum_expected_count"] = minimum
        record(
            "expected_count_minimum",
            minimum,
            "Computed from observed margins; no chi-square test was run.",
        )
        if not math.isfinite(minimum) or minimum < 5:
            if observed.shape == (2, 2):
                record(
                    "method",
                    "fisher_exact",
                    "Sparse expected counts in a 2x2 table block chi-square under policy.",
                )
                return finish(
                    RecommendationStatus.READY,
                    method_id="fisher_exact",
                    rationale=(
                        "The declared target is categorical independence, but the observed 2x2 "
                        "table has expected counts below PyAutoStat's chi-square threshold; "
                        "Fisher's exact test addresses the same categorical association target."
                    ),
                    alternatives=(
                        _alternative(
                            "pearson_chi_square",
                            "Blocked because at least one expected count is below 5.",
                        ),
                    ),
                )
            return finish(
                RecommendationStatus.UNSUPPORTED,
                blockers=(
                    "The contingency table has an expected cell count below 5; "
                    "the current chi-square implementation rejects sparse tables. Fisher's "
                    "exact support is limited to 2x2 tables.",
                ),
                rationale="The current conservative expected-frequency policy is unmet.",
                alternatives=(
                    {
                        "method_id": "fisher_exact",
                        "name": "Fisher exact test",
                        "availability": "incompatible",
                        "reason": "The observed sparse table is larger than 2x2.",
                    },
                ),
            )
        if "ordinal_categorical" in (left, right):
            warnings.append(
                "Chi-square treats ordered categories as nominal and does not use their order."
            )
        record(
            "method", "pearson_chi_square", "Categorical variables and adequate expected counts."
        )
        return finish(
            RecommendationStatus.READY,
            method_id="pearson_chi_square",
            rationale="Chi-square assesses independence of the two categorical variables; "
            "it does not establish causation.",
        )
    if (left in _QUANTITATIVE and right in _CATEGORICAL) or (
        right in _QUANTITATIVE and left in _CATEGORICAL
    ):
        question = {
            "field": "objective",
            "question": "Do you mean to compare the numeric outcome across categories?",
            "explanation": "That is a group-comparison question and needs an explicit target.",
            "input_type": "select",
            "required": True,
            "options": [
                {"value": "compare_groups", "label": "Compare groups"},
                {"value": "association", "label": "Keep association question"},
            ],
        }
        record("clarification", "objective", "Mixed types admit different research questions.")
        return finish(
            RecommendationStatus.NEEDS_INPUT,
            missing=(MissingInformation("objective", str(question["question"])),),
            questions=(question,),
            rationale="The engine will not silently change association into group comparison.",
        )
    return finish(
        RecommendationStatus.UNSUPPORTED,
        blockers=(
            "This variable-type combination has no compatible implemented association method.",
        ),
        rationale="The selected variables cannot be reclassified or recoded automatically.",
    )
