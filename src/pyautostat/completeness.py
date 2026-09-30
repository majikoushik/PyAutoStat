"""Deterministic reporting-field completeness, separate from study quality."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .exceptions import InvalidDataError
from .research_report import ResearchReport
from .specifications import _json_value


@dataclass(frozen=True)
class CompletenessItem:
    code: str
    section: str
    description: str
    applicable: bool
    required: bool
    status: str
    source: str
    message: str

    def __post_init__(self) -> None:
        if self.status not in {"present", "missing", "partial", "not_applicable"}:
            raise InvalidDataError("Completeness item status is invalid.")

    def to_dict(self) -> dict[str, Any]:
        return _json_value(vars(self))


@dataclass(frozen=True)
class ReportingCompletenessResult:
    status: str
    style: str
    items: tuple[CompletenessItem, ...]
    limitations: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.status not in {"complete", "partial", "incomplete"}:
            raise InvalidDataError("Reporting completeness status is invalid.")
        if self.style not in {"general", "apa", "ieee"}:
            raise InvalidDataError("Reporting style must be general, apa, or ieee.")

    def to_dict(self) -> dict[str, Any]:
        return _json_value(
            {
                "schema_version": 1,
                "status": self.status,
                "style": self.style,
                "items": [item.to_dict() for item in self.items],
                "limitations": self.limitations,
            }
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2, allow_nan=False)


def assess_reporting_completeness(
    report: ResearchReport, *, style: str = "general"
) -> ReportingCompletenessResult:
    if not isinstance(report, ResearchReport):
        raise InvalidDataError("reporting_completeness requires a ResearchReport.")
    if style not in {"general", "apa", "ieee"}:
        raise InvalidDataError("style must be general, apa, or ieee.")
    payload = report.to_dict()
    sections = payload["sections"]
    method = sections["methods"].get("method_id")
    results = sections["results"]
    source_values = payload["analysis"]["values"]
    dataset = sections["dataset"]
    question = sections["research_question"]
    inference = (
        method not in {"dataset_profile", "cronbach_alpha"}
        and payload["analysis"]["status"] == "available"
    )
    df_applicable = method in {
        "welch_t",
        "student_t",
        "paired_t",
        "one_sample_t",
        "welch_anova",
        "one_way_anova",
        "kruskal_wallis",
        "pearson_chi_square",
        "linear_regression",
        "point_biserial_correlation",
        "partial_pearson_correlation",
        "repeated_measures_anova",
        "friedman_test",
    }

    def item(
        code: str,
        section: str,
        description: str,
        value: Any,
        source: str,
        *,
        applicable: bool = True,
        required: bool = True,
        unavailable: bool = False,
    ) -> CompletenessItem:
        if not applicable:
            status = "not_applicable"
            message = "This field is not mathematically applicable to the selected method."
        elif value is not None and value != [] and value != {}:
            status = "present"
            message = "The canonical report contains this field."
        elif unavailable:
            status = "partial"
            message = "The source backend does not currently provide this quantity."
        else:
            status = "missing"
            message = "An applicable reporting field is absent."
        return CompletenessItem(
            code, section, description, applicable, required, status, source, message
        )

    items: tuple[CompletenessItem, ...] = (
        item(
            "OBJECTIVE_REPORTED",
            "question",
            "Research objective",
            question.get("objective"),
            "sections.research_question.objective",
        ),
        item(
            "VARIABLES_REPORTED",
            "question",
            "Outcome and predictor roles",
            (
                "profile"
                if method == "dataset_profile"
                else question.get("items")
                if method == "cronbach_alpha"
                else [question["outcome"]]
                if method == "one_sample_t" and question.get("outcome") is not None
                else [question["outcome"], *question.get("predictors", [])]
                if method in {"linear_regression", "logistic_regression"}
                and question.get("outcome") is not None
                else [question["outcome"], question["predictor"]]
                if question.get("outcome") is not None and question.get("predictor") is not None
                else None
            ),
            "sections.research_question",
        ),
        item(
            "DESIGN_REPORTED",
            "methods",
            "Declared study design",
            sections["methods"].get("declared_design"),
            "sections.methods.declared_design",
            applicable=method not in {"dataset_profile", "cronbach_alpha"},
        ),
        item("METHOD_REPORTED", "methods", "Analysis method", method, "sections.methods.method_id"),
        item(
            "SAMPLE_SIZE_REPORTED",
            "dataset",
            "Analyzed sample size",
            dataset.get("analyzed_rows"),
            "sections.dataset.analyzed_rows",
        ),
        item(
            "EXCLUSIONS_REPORTED",
            "dataset",
            "Excluded observations",
            dataset.get("excluded_rows"),
            "sections.dataset.excluded_rows",
        ),
        item(
            "TEST_STATISTIC_REPORTED",
            "results",
            "Test statistic",
            results.get("test_statistic"),
            "sections.results.test_statistic",
            applicable=inference,
        ),
        item(
            "DEGREES_OF_FREEDOM_REPORTED",
            "results",
            "Degrees of freedom",
            results.get("degrees_of_freedom"),
            "sections.results.degrees_of_freedom",
            applicable=inference and df_applicable,
        ),
        item(
            "PVALUE_REPORTED",
            "results",
            "P-value",
            results.get("p_value"),
            "sections.results.p_value",
            applicable=inference,
        ),
        item(
            "EFFECT_ESTIMATE_REPORTED",
            "results",
            "Primary effect estimate",
            results.get("primary_estimate"),
            "sections.results.primary_estimate",
            applicable=inference
            and method
            not in {
                "welch_anova",
                "linear_regression",
                "logistic_regression",
                "repeated_measures_anova",
                "friedman_test",
            },
        ),
        item(
            "EFFECT_SIZE_REPORTED",
            "results",
            "Method-specific effect size",
            results.get("effect_size"),
            "sections.results.effect_size",
            applicable=inference
            and method not in {"welch_anova", "linear_regression", "logistic_regression"},
        ),
        item(
            "CONFIDENCE_INTERVAL_REPORTED",
            "results",
            "Confidence interval",
            results.get("confidence_interval"),
            "sections.results.confidence_interval",
            applicable=inference
            and method
            not in {
                "welch_anova",
                "linear_regression",
                "logistic_regression",
                "repeated_measures_anova",
                "friedman_test",
            },
            unavailable=(
                (
                    source_values.get("confidence_interval") is None
                    or (
                        isinstance(source_values.get("confidence_interval"), dict)
                        and source_values.get("confidence_interval", {}).get("status")
                        in {"unavailable", "uncomputable"}
                    )
                )
                and method in {"pearson_correlation", "wilcoxon_signed_rank", "fisher_exact"}
            ),
        ),
        item(
            "GROUP_SUMMARIES_REPORTED",
            "results",
            "Per-group descriptive summaries",
            results.get("group_summaries") or results.get("condition_summaries"),
            "sections.results.group_summaries",
            applicable=inference
            and method
            in {
                "welch_anova",
                "one_way_anova",
                "kruskal_wallis",
                "repeated_measures_anova",
                "friedman_test",
            },
        ),
        item(
            "PAIRWISE_COMPARISONS_REPORTED",
            "results",
            "Complete multiplicity-controlled pairwise family",
            results.get("pairwise_comparisons"),
            "sections.results.pairwise_comparisons",
            applicable=inference
            and method
            in {
                "welch_anova",
                "one_way_anova",
                "kruskal_wallis",
                "repeated_measures_anova",
                "friedman_test",
            },
        ),
        item(
            "ALPHA_REPORTED",
            "methods",
            "Significance threshold",
            sections["methods"].get("alpha"),
            "sections.methods.alpha",
            applicable=inference,
        ),
        item(
            "LIMITATIONS_REPORTED",
            "limitations",
            "Interpretive limitations",
            payload.get("limitations"),
            "limitations",
        ),
        item(
            "SENSITIVITY_REPORTED",
            "sensitivity",
            "Declared sensitivity analyses",
            payload.get("sensitivity"),
            "sensitivity",
            applicable="sensitivity_analysis" in sections,
            required=False,
        ),
        item(
            "PRACTICAL_THRESHOLD_REPORTED",
            "practical_significance",
            "Researcher-defined threshold",
            payload.get("practical_significance"),
            "practical_significance",
            applicable="practical_significance" in sections,
            required=False,
        ),
    )
    if method == "linear_regression":
        items += (
            item(
                "REGRESSION_SPECIFICATION_REPORTED",
                "results",
                "Regression model specification",
                results.get("design_matrix"),
                "sections.results.design_matrix",
            ),
            item(
                "REGRESSION_MODEL_FIT_REPORTED",
                "results",
                "R-squared, adjusted R-squared, and model fit",
                results.get("model_fit"),
                "sections.results.model_fit",
            ),
            item(
                "REGRESSION_COEFFICIENTS_REPORTED",
                "results",
                "Coefficient estimates and confidence intervals",
                results.get("coefficients"),
                "sections.results.coefficients",
            ),
            item(
                "REGRESSION_COVARIANCE_REPORTED",
                "methods",
                "Covariance estimator",
                results.get("covariance_type"),
                "sections.results.covariance_type",
            ),
            item(
                "REGRESSION_DIAGNOSTICS_REPORTED",
                "diagnostics",
                "VIF, heteroscedasticity, residual, and influence diagnostics",
                results.get("diagnostics"),
                "sections.results.diagnostics",
            ),
        )
    if method == "logistic_regression":
        logistic_coefficients = results.get("coefficients")
        odds_ratios = (
            logistic_coefficients
            if isinstance(logistic_coefficients, list)
            and logistic_coefficients
            and all(
                isinstance(item, dict) and item.get("odds_ratio") is not None
                for item in logistic_coefficients
            )
            else None
        )
        coefficient_intervals = (
            logistic_coefficients
            if isinstance(logistic_coefficients, list)
            and logistic_coefficients
            and all(
                isinstance(item, dict)
                and item.get("confidence_interval") is not None
                and item.get("odds_ratio_ci") is not None
                for item in logistic_coefficients
            )
            else None
        )
        items += (
            item(
                "LOGISTIC_EVENT_REPORTED",
                "results",
                "Modeled event and reference level",
                results.get("event_level") if results.get("non_event_level") is not None else None,
                "sections.results.event_level",
            ),
            item(
                "LOGISTIC_MODEL_FIT_REPORTED",
                "results",
                "Likelihood fit and McFadden pseudo-R-squared",
                results.get("model_fit"),
                "sections.results.model_fit",
            ),
            item(
                "LOGISTIC_COEFFICIENTS_REPORTED",
                "results",
                "Coefficients, odds ratios, and intervals",
                logistic_coefficients,
                "sections.results.coefficients",
            ),
            item(
                "LOGISTIC_ODDS_RATIOS_REPORTED",
                "results",
                "Per-term odds ratios",
                odds_ratios,
                "sections.results.coefficients.odds_ratio",
            ),
            item(
                "LOGISTIC_INTERVALS_REPORTED",
                "results",
                "Coefficient and odds-ratio confidence intervals",
                coefficient_intervals,
                "sections.results.coefficients.confidence_interval",
            ),
            item(
                "LOGISTIC_DIAGNOSTICS_REPORTED",
                "diagnostics",
                "Convergence and design diagnostics",
                results.get("diagnostics"),
                "sections.results.diagnostics",
            ),
            item(
                "LOGISTIC_COVARIANCE_REPORTED",
                "methods",
                "Coefficient covariance estimator",
                results.get("covariance_type"),
                "sections.results.covariance_type",
            ),
            item(
                "LOGISTIC_SAMPLE_COUNTS_REPORTED",
                "results",
                "Analyzed, event, and non-event counts",
                (
                    [
                        dataset.get("analyzed_rows"),
                        results.get("event_count"),
                        results.get("non_event_count"),
                    ]
                    if results.get("event_count") is not None
                    and results.get("non_event_count") is not None
                    else None
                ),
                "sections.results.event_count",
            ),
        )
    if method == "mcnemar":
        items += (
            item(
                "MCNEMAR_TRANSITIONS_REPORTED",
                "results",
                "Paired binary transition table",
                results.get("transition_table"),
                "sections.results.transition_table",
            ),
            item(
                "MCNEMAR_EVENT_REPORTED",
                "results",
                "Event and condition orientation",
                results.get("event_level") if results.get("condition_order") is not None else None,
                "sections.results.event_level",
            ),
        )
    if method == "point_biserial_correlation":
        items += (
            item(
                "POINT_BISERIAL_CODING_REPORTED",
                "results",
                "Binary positive/reference coding",
                results.get("binary_encoding"),
                "sections.results.binary_encoding",
            ),
        )
    if method == "kendall_tau_b":
        items += (
            item(
                "KENDALL_TIES_REPORTED",
                "results",
                "Tau-b variant and tie metadata",
                results.get("ties"),
                "sections.results.ties",
            ),
        )
    if method == "partial_pearson_correlation":
        items += (
            item(
                "PARTIAL_CONTROLS_REPORTED",
                "results",
                "Ordered linear-adjustment controls",
                results.get("controls"),
                "sections.results.controls",
            ),
            item(
                "PARTIAL_CONTROL_DESIGN_REPORTED",
                "results",
                "Effective control design",
                results.get("control_design"),
                "sections.results.control_design",
            ),
        )
    if method == "cronbach_alpha":
        interval = results.get("confidence_interval")
        items += (
            item(
                "RELIABILITY_ESTIMATE_REPORTED",
                "results",
                "Cronbach alpha estimate",
                results.get("cronbach_alpha"),
                "sections.results.cronbach_alpha",
            ),
            item(
                "RELIABILITY_INTERVAL_REPORTED",
                "results",
                "Respondent-row bootstrap interval",
                interval
                if isinstance(interval, dict) and interval.get("status") == "available"
                else None,
                "sections.results.confidence_interval",
                unavailable=isinstance(interval, dict) and interval.get("status") == "unavailable",
            ),
            item(
                "RELIABILITY_ITEMS_REPORTED",
                "results",
                "Per-item descriptives and diagnostics",
                results.get("item_statistics"),
                "sections.results.item_statistics",
            ),
            item(
                "INTER_ITEM_CORRELATIONS_REPORTED",
                "results",
                "Inter-item correlation matrix",
                results.get("inter_item_correlations"),
                "sections.results.inter_item_correlations",
            ),
            item(
                "RELIABILITY_MISSINGNESS_REPORTED",
                "results",
                "Per-item and complete-case missingness",
                results.get("missingness"),
                "sections.results.missingness",
            ),
        )
    if method == "repeated_measures_anova":
        items += (
            item(
                "SPHERICITY_REPORTED",
                "results",
                "Sphericity diagnostic and Mauchly test",
                results.get("sphericity"),
                "sections.results.sphericity",
            ),
            item(
                "CORRECTION_REPORTED",
                "results",
                "Greenhouse-Geisser correction and epsilon",
                results.get("greenhouse_geisser"),
                "sections.results.greenhouse_geisser",
            ),
        )
    required_statuses = [entry.status for entry in items if entry.applicable and entry.required]
    status = (
        "incomplete"
        if "missing" in required_statuses
        else "partial"
        if "partial" in required_statuses
        else "complete"
    )
    return ReportingCompletenessResult(
        status,
        style,
        items,
        (
            "Completeness means that PyAutoStat's applicable reporting fields are present; "
            "it is not a study-quality score or publication guarantee.",
        ),
    )
