"""Read-only comparisons between a source result, report and actual exports."""

from __future__ import annotations

import csv
import io
import json
import math
from dataclasses import dataclass
from typing import Any

from .exceptions import InvalidDataError, ReportError
from .practical_significance import PracticalSignificanceResult
from .provenance import content_reference
from .research_report import ResearchReport, build_research_report
from .results import AnalysisResult
from .sensitivity import SensitivityResult
from .specifications import _json_value


@dataclass(frozen=True)
class AuditFinding:
    code: str
    severity: str
    component: str
    field: str
    expected: Any
    actual: Any
    explanation: str

    def to_dict(self) -> dict[str, Any]:
        return _json_value(vars(self))


@dataclass(frozen=True)
class AuditResult:
    status: str
    findings: tuple[AuditFinding, ...]
    checked_components: tuple[str, ...]
    skipped_checks: tuple[str, ...]
    source_references: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return _json_value(
            {
                "schema_version": 1,
                "status": self.status,
                "findings": [item.to_dict() for item in self.findings],
                "checked_components": self.checked_components,
                "skipped_checks": self.skipped_checks,
                "source_references": self.source_references,
            }
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2, allow_nan=False)


def _code(path: str) -> str:
    if "comparability" in path:
        return "SENSITIVITY_COMPARABILITY_MISMATCH"
    if "scenario_results" in path and path.endswith("length"):
        return "SENSITIVITY_SCENARIO_OMITTED"
    if "sensitivity" in path and "primary_estimate" in path:
        return "SENSITIVITY_ESTIMATE_MISMATCH"
    if "practical_significance" in path and "threshold" in path:
        return "PRACTICAL_THRESHOLD_MISMATCH"
    if "practical_significance" in path and "relation" in path:
        return "PRACTICAL_RELATION_MISMATCH"
    if "p_value" in path:
        return "PVALUE_MISMATCH"
    if "test_statistic" in path:
        return "STATISTIC_MISMATCH"
    if "group_order" in path or "contrast" in path:
        return "GROUP_ORDER_MISMATCH"
    if "group_sizes" in path:
        return "GROUP_COUNT_MISMATCH"
    if any(
        part in path
        for part in (
            "sample_size",
            "original_rows",
            "analyzed_rows",
            "excluded_rows",
            "effective_pair_count",
        )
    ):
        return "SAMPLE_COUNT_MISMATCH"
    if "confidence_interval" in path or "intervals" in path:
        return "INTERVAL_QUANTITY_MISMATCH" if "quantity" in path else "INTERVAL_MISMATCH"
    if "effect_size" in path or "effect_estimates" in path:
        return "EFFECT_SIZE_MISMATCH"
    if "method_id" in path or "method_name" in path:
        return "METHOD_MISMATCH"
    if any(
        part in path
        for part in (
            "objective",
            "estimand",
            "design",
            "alpha",
            "confidence_level",
            "specification",
        )
    ):
        return "SPECIFICATION_MISMATCH"
    if "interpretation" in path:
        return "INTERPRETATION_MISMATCH"
    if "warnings" in path or "limitations" in path:
        return "MISSING_REQUIRED_WARNING"
    if "tables" in path:
        return "TABLE_MISMATCH"
    return "REPORT_VALUE_MISMATCH"


def _safe(value: Any, path: str) -> Any:
    """Do not echo possible category labels, source prose or raw profile values."""
    if isinstance(value, bool) or value is None or isinstance(value, (int, float)):
        return value
    if isinstance(value, str) and any(
        part in path for part in ("method_id", "status", "objective", "estimand", "design")
    ):
        return value
    if isinstance(value, str):
        return "<text omitted>"
    if isinstance(value, list):
        return f"<list of {len(value)} items>"
    if isinstance(value, dict):
        return f"<mapping of {len(value)} fields>"
    return "<value omitted>"


def _compare(expected: Any, actual: Any, path: str, findings: list[AuditFinding]) -> None:
    if isinstance(expected, dict) and isinstance(actual, dict):
        for key in sorted(expected.keys() | actual.keys()):
            child = f"{path}.{key}" if path else key
            if key not in expected or key not in actual:
                _mismatch(child, expected.get(key), actual.get(key), findings)
            else:
                _compare(expected[key], actual[key], child, findings)
        return
    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            _mismatch(path + ".length", len(expected), len(actual), findings)
        for index, (left, right) in enumerate(zip(expected, actual, strict=False)):
            _compare(left, right, f"{path}[{index}]", findings)
        return
    if type(expected) is not type(actual) or expected != actual:
        _mismatch(path, expected, actual, findings)


def _mismatch(path: str, expected: Any, actual: Any, findings: list[AuditFinding]) -> None:
    findings.append(
        AuditFinding(
            code=_code(path),
            severity="error",
            component=path.split(".")[0],
            field=path,
            expected=_safe(expected, path),
            actual=_safe(actual, path),
            explanation="The inspected value differs from the canonical source.",
        )
    )


def _csv_rows(value: str) -> list[list[str]]:
    return list(csv.reader(io.StringIO(value)))


def _method_contract_findings(source: AnalysisResult) -> tuple[AuditFinding, ...]:
    """Check stored method identity and accounting without recalculating statistics."""
    if source.status.value != "available":
        return ()
    if source.method_id == "dataset_profile":
        return ()
    findings: list[AuditFinding] = []
    specification = source.specification
    recommendation = source.recommendation
    if specification is None:
        _mismatch("analysis.specification", "present", None, findings)
        return tuple(findings)
    if recommendation is None or recommendation.method_id != source.method_id:
        _mismatch(
            "analysis.recommendation.method_id",
            source.method_id,
            recommendation.method_id if recommendation is not None else None,
            findings,
        )
    values = source.values
    p_value = values.get("p_value")
    allowed_missing_p_value = (
        source.method_id == "one_sample_t"
        and values.get("effect_size", {}).get("status") == "unavailable_zero_variance"
    ) or source.method_id == "cronbach_alpha"
    valid_p_value = (
        isinstance(p_value, (int, float))
        and not isinstance(p_value, bool)
        and math.isfinite(float(p_value))
        and 0 <= float(p_value) <= 1
    )
    if not valid_p_value and not allowed_missing_p_value:
        _mismatch("analysis.values.p_value", "finite probability", p_value, findings)
    statistic = values.get("test_statistic")
    allowed_missing_statistic = (
        (
            source.method_id == "one_sample_t"
            and values.get("effect_size", {}).get("status") == "unavailable_zero_variance"
        )
        or (
            source.method_id == "fisher_exact"
            and source.metadata.get("odds_ratio_status") in {"positive_infinity", "undefined"}
        )
        or source.method_id == "cronbach_alpha"
    )
    valid_statistic = (
        isinstance(statistic, (int, float))
        and not isinstance(statistic, bool)
        and math.isfinite(float(statistic))
    )
    if not valid_statistic and not allowed_missing_statistic:
        _mismatch("analysis.values.test_statistic", "finite statistic", statistic, findings)
    interval = values.get("confidence_interval")
    if isinstance(interval, dict) and interval.get("quantity") != values.get("estimate_name"):
        _mismatch(
            "analysis.values.confidence_interval.quantity",
            values.get("estimate_name"),
            interval.get("quantity"),
            findings,
        )
    sample = source.metadata.get("sample")
    if isinstance(sample, dict):
        original = sample.get("original_rows")
        if (
            isinstance(original, int)
            and source.sample_size is not None
            and source.excluded_rows is not None
            and original != source.sample_size + source.excluded_rows
        ):
            _mismatch(
                "analysis.metadata.sample.original_rows",
                source.sample_size + source.excluded_rows,
                original,
                findings,
            )
    method = source.method_id
    question = specification.question
    if method == "one_sample_t":
        reference = question.reference_value
        for path, actual in (
            ("analysis.values.reference_value", values.get("reference_value")),
            ("analysis.metadata.reference_value", source.metadata.get("reference_value")),
        ):
            if actual != reference:
                _mismatch(path, reference, actual, findings)
    if method in {"paired_t", "wilcoxon_signed_rank", "mcnemar"}:
        contrast = source.metadata.get("contrast")
        order = source.metadata.get("condition_order")
        if (
            not isinstance(order, list)
            or len(order) != 2
            or not isinstance(contrast, dict)
            or contrast.get("first") != order[0]
            or contrast.get("second") != order[1]
            or contrast.get("definition") != "first condition minus second condition"
        ):
            _mismatch("analysis.metadata.contrast", "declared condition order", contrast, findings)
        if isinstance(sample, dict) and sample.get("complete_pairs") is not None:
            expected_rows = 2 * int(sample["complete_pairs"])
            if source.sample_size != expected_rows:
                _mismatch("analysis.sample_size", expected_rows, source.sample_size, findings)
    if method == "fisher_exact":
        observed = source.metadata.get("observed_counts")
        if (
            not isinstance(observed, list)
            or len(observed) != 2
            or any(not isinstance(row, list) or len(row) != 2 for row in observed)
        ):
            _mismatch("analysis.metadata.observed_counts", "2x2 table", observed, findings)
    if method == "linear_regression":
        predictors = values.get("predictors")
        if predictors != list(question.predictors or ()):
            _mismatch(
                "analysis.values.predictors", list(question.predictors or ()), predictors, findings
            )
        design = values.get("design_matrix")
        coefficients = values.get("coefficients")
        fit = values.get("model_fit")
        diagnostics = values.get("diagnostics")
        if not isinstance(design, dict) or design.get("full_rank") is not True:
            _mismatch("analysis.values.design_matrix.full_rank", True, design, findings)
        elif isinstance(coefficients, list):
            parameter_count = design.get("parameter_count")
            if len(coefficients) != parameter_count:
                _mismatch(
                    "analysis.values.coefficients",
                    f"{parameter_count} coefficient records",
                    len(coefficients),
                    findings,
                )
            terms = [item.get("term") for item in coefficients if isinstance(item, dict)]
            if terms != design.get("term_names") or len(set(map(str, terms))) != len(terms):
                _mismatch(
                    "analysis.values.design_matrix.term_names",
                    terms,
                    design.get("term_names"),
                    findings,
                )
            for index, item in enumerate(coefficients):
                if not isinstance(item, dict):
                    _mismatch(f"analysis.values.coefficients[{index}]", "record", item, findings)
                    continue
                estimate = item.get("estimate")
                interval = item.get("confidence_interval")
                lower = interval.get("lower") if isinstance(interval, dict) else None
                upper = interval.get("upper") if isinstance(interval, dict) else None
                valid_interval = (
                    isinstance(estimate, (int, float))
                    and isinstance(lower, (int, float))
                    and isinstance(upper, (int, float))
                    and lower <= estimate <= upper
                )
                if not valid_interval:
                    _mismatch(
                        f"analysis.values.coefficients[{index}].confidence_interval",
                        "ordered interval containing estimate",
                        interval,
                        findings,
                    )
        else:
            _mismatch("analysis.values.coefficients", "coefficient records", coefficients, findings)
        if isinstance(fit, dict):
            r_squared = fit.get("r_squared")
            adjusted = fit.get("adjusted_r_squared")
            if not isinstance(r_squared, (int, float)) or not 0 <= r_squared <= 1:
                _mismatch(
                    "analysis.values.model_fit.r_squared", "value in [0, 1]", r_squared, findings
                )
            if not isinstance(adjusted, (int, float)) or not math.isfinite(float(adjusted)):
                _mismatch(
                    "analysis.values.model_fit.adjusted_r_squared",
                    "finite value",
                    adjusted,
                    findings,
                )
            if isinstance(design, dict) and source.sample_size is not None:
                expected_residual = source.sample_size - int(design.get("parameter_count", 0))
                if fit.get("residual_degrees_of_freedom") != expected_residual:
                    _mismatch(
                        "analysis.values.model_fit.residual_degrees_of_freedom",
                        expected_residual,
                        fit.get("residual_degrees_of_freedom"),
                        findings,
                    )
        else:
            _mismatch("analysis.values.model_fit", "model fit record", fit, findings)
        if values.get("covariance_type") != specification.options.covariance_type:
            _mismatch(
                "analysis.values.covariance_type",
                specification.options.covariance_type,
                values.get("covariance_type"),
                findings,
            )
        if not isinstance(diagnostics, dict) or any(
            key not in diagnostics
            for key in ("vif", "breusch_pagan", "residual_normality", "influence")
        ):
            _mismatch(
                "analysis.values.diagnostics", "complete diagnostic record", diagnostics, findings
            )
    if method == "logistic_regression":
        coefficients = values.get("coefficients")
        design = values.get("design_matrix")
        fit = values.get("model_fit")
        diagnostics = values.get("diagnostics")
        if values.get("event_level") != question.event_level:
            _mismatch(
                "analysis.values.event_level",
                question.event_level,
                values.get("event_level"),
                findings,
            )
        if values.get("predictors") != list(question.predictors or ()):
            _mismatch(
                "analysis.values.predictors",
                list(question.predictors or ()),
                values.get("predictors"),
                findings,
            )
        if not isinstance(design, dict) or design.get("full_rank") is not True:
            _mismatch("analysis.values.design_matrix.full_rank", True, design, findings)
        elif isinstance(coefficients, list):
            parameter_count = design.get("parameter_count")
            if len(coefficients) != parameter_count:
                _mismatch(
                    "analysis.values.coefficients",
                    f"{parameter_count} coefficient records",
                    len(coefficients),
                    findings,
                )
            terms = [item.get("term") for item in coefficients if isinstance(item, dict)]
            if terms != design.get("term_names"):
                _mismatch(
                    "analysis.values.design_matrix.term_names",
                    terms,
                    design.get("term_names"),
                    findings,
                )
        if not isinstance(diagnostics, dict) or diagnostics.get("converged") is not True:
            _mismatch("analysis.values.diagnostics.converged", True, diagnostics, findings)
        elif diagnostics.get("covariance_type") != specification.options.covariance_type:
            _mismatch(
                "analysis.values.diagnostics.covariance_type",
                specification.options.covariance_type,
                diagnostics.get("covariance_type"),
                findings,
            )
        event_count = values.get("event_count")
        non_event_count = values.get("non_event_count")
        if not (
            isinstance(event_count, int)
            and isinstance(non_event_count, int)
            and event_count + non_event_count == source.sample_size
        ):
            _mismatch(
                "analysis.values.event_count",
                "event and non-event counts sum to analyzed n",
                (event_count, non_event_count),
                findings,
            )
        if not isinstance(fit, dict) or fit.get("analyzed_rows") != source.sample_size:
            _mismatch(
                "analysis.values.model_fit.analyzed_rows",
                source.sample_size,
                fit,
                findings,
            )
        elif isinstance(design, dict) and source.sample_size is not None:
            parameter_count = design.get("parameter_count")
            llf = fit.get("log_likelihood")
            llnull = fit.get("null_log_likelihood")
            if (
                isinstance(parameter_count, (int, float))
                and isinstance(llf, (int, float))
                and isinstance(llnull, (int, float))
                and int(parameter_count) > 0
                and math.isfinite(float(llf))
                and math.isfinite(float(llnull))
                and float(llnull) != 0
            ):
                expected_fit = {
                    "lr_statistic": -2.0 * (float(llnull) - float(llf)),
                    "mcfadden_r2": 1.0 - float(llf) / float(llnull),
                    "aic": -2.0 * float(llf) + 2.0 * int(parameter_count),
                    "bic": -2.0 * float(llf) + math.log(source.sample_size) * int(parameter_count),
                    "model_degrees_of_freedom": int(parameter_count) - 1,
                    "residual_degrees_of_freedom": source.sample_size - int(parameter_count),
                }
                for field, expected in expected_fit.items():
                    actual = fit.get(field)
                    if not isinstance(actual, (int, float)) or not math.isclose(
                        expected, actual, rel_tol=1e-9, abs_tol=1e-12
                    ):
                        _mismatch(
                            f"analysis.values.model_fit.{field}",
                            expected,
                            actual,
                            findings,
                        )
            else:
                _mismatch(
                    "analysis.values.model_fit",
                    "finite likelihoods and parameter count",
                    fit,
                    findings,
                )
        if values.get("covariance_type") != specification.options.covariance_type:
            _mismatch(
                "analysis.values.covariance_type",
                specification.options.covariance_type,
                values.get("covariance_type"),
                findings,
            )
        if isinstance(coefficients, list):
            for index, item in enumerate(coefficients):
                if not isinstance(item, dict):
                    continue
                beta = item.get("estimate")
                odds_ratio = item.get("odds_ratio")
                ci = item.get("confidence_interval")
                odds_ci = item.get("odds_ratio_ci")
                if item.get("covariance_type") != specification.options.covariance_type:
                    _mismatch(
                        f"analysis.values.coefficients[{index}].covariance_type",
                        specification.options.covariance_type,
                        item.get("covariance_type"),
                        findings,
                    )
                if not (
                    isinstance(beta, (int, float))
                    and isinstance(odds_ratio, (int, float))
                    and math.isclose(math.exp(beta), odds_ratio, rel_tol=1e-10)
                ):
                    _mismatch(
                        f"analysis.values.coefficients[{index}].odds_ratio",
                        "exp(coefficient)",
                        odds_ratio,
                        findings,
                    )
                if isinstance(ci, dict) and isinstance(odds_ci, dict):
                    expected_interval = (math.exp(ci["lower"]), math.exp(ci["upper"]))
                    actual_lower = odds_ci.get("lower")
                    actual_upper = odds_ci.get("upper")
                    valid_odds_interval = (
                        isinstance(actual_lower, (int, float))
                        and isinstance(actual_upper, (int, float))
                        and math.isclose(expected_interval[0], actual_lower, rel_tol=1e-10)
                        and math.isclose(expected_interval[1], actual_upper, rel_tol=1e-10)
                    )
                    if not valid_odds_interval:
                        _mismatch(
                            f"analysis.values.coefficients[{index}].odds_ratio_ci",
                            expected_interval,
                            (actual_lower, actual_upper),
                            findings,
                        )
        else:
            _mismatch("analysis.values.coefficients", "coefficient records", coefficients, findings)
    if method == "mcnemar":
        table = values.get("transition_table")
        if not isinstance(table, dict) or sum(
            int(table.get(key, 0))
            for key in (
                "first_event_second_event",
                "first_event_second_non_event",
                "first_non_event_second_event",
                "first_non_event_second_non_event",
            )
        ) != int(source.metadata.get("sample", {}).get("complete_pairs", -1)):
            _mismatch(
                "analysis.values.transition_table", "counts equal complete pairs", table, findings
            )
        elif table.get("discordant_b") != table.get("first_event_second_non_event") or table.get(
            "discordant_c"
        ) != table.get("first_non_event_second_event"):
            _mismatch(
                "analysis.values.transition_table", "consistent discordant counts", table, findings
            )
        complete_pairs = source.metadata.get("sample", {}).get("complete_pairs")
        if isinstance(table, dict) and isinstance(complete_pairs, int) and complete_pairs > 0:
            first_rate = (
                table["first_event_second_event"] + table["first_event_second_non_event"]
            ) / complete_pairs
            second_rate = (
                table["first_event_second_event"] + table["first_non_event_second_event"]
            ) / complete_pairs
            difference = first_rate - second_rate
            for path, expected, actual in (
                ("first_event_proportion", first_rate, values.get("first_event_proportion")),
                ("second_event_proportion", second_rate, values.get("second_event_proportion")),
                ("primary_estimate", difference, values.get("primary_estimate")),
            ):
                if not isinstance(actual, (int, float)) or not math.isclose(
                    expected, actual, rel_tol=1e-10, abs_tol=1e-12
                ):
                    _mismatch(f"analysis.values.{path}", expected, actual, findings)
        if values.get("event_level") != question.event_level:
            _mismatch(
                "analysis.values.event_level",
                question.event_level,
                values.get("event_level"),
                findings,
            )
        expected_order = list(specification.condition_order or ())
        if values.get("condition_order") != expected_order:
            _mismatch(
                "analysis.values.condition_order",
                expected_order,
                values.get("condition_order"),
                findings,
            )
        if isinstance(table, dict):
            b = table.get("discordant_b")
            c = table.get("discordant_c")
            matched = values.get("matched_odds_ratio")
            if isinstance(b, int) and isinstance(c, int):
                expected_status = (
                    "positive_infinity"
                    if c == 0 and b > 0
                    else "zero"
                    if b == 0 and c > 0
                    else "undefined"
                    if b == 0 and c == 0
                    else "finite"
                )
                expected_value = float(b / c) if c > 0 else None
                if not isinstance(matched, dict) or matched.get("status") != expected_status:
                    _mismatch(
                        "analysis.values.matched_odds_ratio.status",
                        expected_status,
                        matched,
                        findings,
                    )
                elif matched.get("value") != expected_value:
                    _mismatch(
                        "analysis.values.matched_odds_ratio.value",
                        expected_value,
                        matched.get("value"),
                        findings,
                    )
    if method in {
        "point_biserial_correlation",
        "kendall_tau_b",
        "partial_pearson_correlation",
    }:
        estimate = values.get("primary_estimate")
        if not isinstance(estimate, (int, float)) or not -1 <= float(estimate) <= 1:
            _mismatch(
                "analysis.values.primary_estimate", "coefficient in [-1, 1]", estimate, findings
            )
        if method == "point_biserial_correlation":
            coding = values.get("binary_encoding")
            if (
                not isinstance(coding, dict)
                or coding.get("positive_level") != question.event_level
                or coding.get("negative_level") == coding.get("positive_level")
            ):
                _mismatch("analysis.values.binary_encoding", question.event_level, coding, findings)
        if method == "kendall_tau_b":
            if values.get("ties", {}).get("variant") != "b":
                _mismatch("analysis.values.ties.variant", "b", values.get("ties"), findings)
            if question.association_measure != "kendall":
                _mismatch(
                    "specification.question.association_measure",
                    "kendall",
                    question.association_measure,
                    findings,
                )
        if method == "partial_pearson_correlation":
            if values.get("controls") != list(question.controls or ()):
                _mismatch(
                    "analysis.values.controls",
                    list(question.controls or ()),
                    values.get("controls"),
                    findings,
                )
            controls = list(question.controls or ())
            expected_df = (source.sample_size or 0) - len(controls) - 2
            if values.get("degrees_of_freedom") != expected_df:
                _mismatch(
                    "analysis.values.degrees_of_freedom",
                    expected_df,
                    values.get("degrees_of_freedom"),
                    findings,
                )
            control_design = values.get("control_design")
            if not isinstance(control_design, dict) or control_design.get(
                "effective_control_terms"
            ) != len(controls):
                _mismatch(
                    "analysis.values.control_design.effective_control_terms",
                    len(controls),
                    control_design,
                    findings,
                )
            if isinstance(estimate, (int, float)):
                statistic = values.get("test_statistic")
                if abs(float(estimate)) < 1:
                    expected_statistic = float(estimate) * math.sqrt(
                        expected_df / (1.0 - float(estimate) ** 2)
                    )
                    if not isinstance(statistic, (int, float)) or not math.isclose(
                        expected_statistic, statistic, rel_tol=1e-9, abs_tol=1e-12
                    ):
                        _mismatch(
                            "analysis.values.test_statistic",
                            expected_statistic,
                            statistic,
                            findings,
                        )
                elif statistic is not None:
                    _mismatch(
                        "analysis.values.test_statistic",
                        None,
                        statistic,
                        findings,
                    )
    if method == "cronbach_alpha":
        declared_items = list(question.items or ())
        if values.get("items") != declared_items:
            _mismatch("analysis.values.items", declared_items, values.get("items"), findings)
        alpha = values.get("cronbach_alpha")
        if not (
            isinstance(alpha, (int, float))
            and not isinstance(alpha, bool)
            and math.isfinite(float(alpha))
        ):
            _mismatch("analysis.values.cronbach_alpha", "finite estimate", alpha, findings)
        item_statistics = values.get("item_statistics")
        if (
            not isinstance(item_statistics, list)
            or [item.get("item") for item in item_statistics if isinstance(item, dict)]
            != declared_items
        ):
            _mismatch(
                "analysis.values.item_statistics",
                "one ordered record per declared item",
                item_statistics,
                findings,
            )
        elif any(
            item.get("corrected_total_excludes_focal_item") is not True
            or item.get("deleted_item") != item.get("item")
            or item.get("remaining_item_count") != len(declared_items) - 1
            for item in item_statistics
        ):
            _mismatch(
                "analysis.values.item_statistics",
                "corrected-total and deletion semantics",
                item_statistics,
                findings,
            )
        interval = values.get("confidence_interval")
        if not isinstance(interval, dict) or (
            interval.get("quantity") != "Cronbach's alpha"
            or interval.get("method") != "respondent-row percentile bootstrap"
            or interval.get("level") != specification.options.confidence_level
        ):
            _mismatch(
                "analysis.values.confidence_interval",
                "alpha respondent-row bootstrap interval",
                interval,
                findings,
            )
        elif interval.get("status") == "available":
            lower, upper = interval.get("lower"), interval.get("upper")
            if not (
                isinstance(lower, (int, float))
                and isinstance(upper, (int, float))
                and math.isfinite(float(lower))
                and math.isfinite(float(upper))
                and lower <= upper
            ):
                _mismatch(
                    "analysis.values.confidence_interval",
                    "finite ordered bounds",
                    interval,
                    findings,
                )
        scoring = values.get("scoring")
        expected_scoring = specification.options.reverse_scoring or {}
        reversed_items = scoring.get("reversed_items", []) if isinstance(scoring, dict) else []
        actual_scoring = {
            item.get("item"): (item.get("lower"), item.get("upper"))
            for item in reversed_items
            if isinstance(item, dict)
        }
        if actual_scoring != expected_scoring:
            _mismatch(
                "analysis.values.scoring.reversed_items",
                expected_scoring,
                actual_scoring,
                findings,
            )
    if method in {"welch_anova", "one_way_anova", "kruskal_wallis"}:
        order = source.metadata.get("group_order")
        pairwise = values.get("pairwise_comparisons")
        summaries = values.get("group_summaries")
        if not isinstance(order, list) or len(order) < 3:
            _mismatch("analysis.metadata.group_order", "at least three groups", order, findings)
        else:
            expected_count = len(order) * (len(order) - 1) // 2
            if not isinstance(pairwise, list) or len(pairwise) != expected_count:
                _mismatch(
                    "analysis.values.pairwise_comparisons",
                    f"complete family of {expected_count} pairs",
                    pairwise,
                    findings,
                )
            else:
                expected_pairs = {
                    (str(order[first]), str(order[second]))
                    for first in range(len(order))
                    for second in range(first + 1, len(order))
                }
                observed_pairs: set[tuple[str, str]] = set()
                for index, item in enumerate(pairwise):
                    path = f"analysis.values.pairwise_comparisons[{index}]"
                    if not isinstance(item, dict):
                        _mismatch(path, "pairwise record", item, findings)
                        continue
                    observed_pairs.add((str(item.get("group1")), str(item.get("group2"))))
                    adjusted = item.get("adjusted_p_value")
                    if not (
                        isinstance(adjusted, (int, float))
                        and not isinstance(adjusted, bool)
                        and math.isfinite(float(adjusted))
                        and 0 <= float(adjusted) <= 1
                    ):
                        _mismatch(
                            f"{path}.adjusted_p_value", "finite probability", adjusted, findings
                        )
                    contrast = item.get("contrast")
                    if (
                        not isinstance(contrast, dict)
                        or contrast.get("definition") != "first group minus second group"
                        or contrast.get("first_group") != item.get("group1")
                        or contrast.get("second_group") != item.get("group2")
                    ):
                        _mismatch(
                            f"{path}.contrast", "first-minus-second orientation", contrast, findings
                        )
                if observed_pairs != expected_pairs:
                    _mismatch(
                        "analysis.values.pairwise_comparisons.pairs",
                        sorted(expected_pairs),
                        sorted(observed_pairs),
                        findings,
                    )
            if not isinstance(summaries, list) or [
                str(item.get("group")) for item in summaries if isinstance(item, dict)
            ] != [str(item) for item in order]:
                _mismatch(
                    "analysis.values.group_summaries",
                    "one summary in recorded group order",
                    summaries,
                    findings,
                )
    return tuple(findings)


class StatisticalResultAuditor:
    """Check consistency; a pass is not independent scientific validation."""

    def audit(
        self,
        report: ResearchReport,
        *,
        result: AnalysisResult | None = None,
        sensitivity: SensitivityResult | None = None,
        practical_significance: PracticalSignificanceResult | None = None,
        exports: dict[str, Any] | None = None,
    ) -> AuditResult:
        if not isinstance(report, ResearchReport):
            raise InvalidDataError("audit requires a ResearchReport.")
        if result is not None and not isinstance(result, AnalysisResult):
            raise InvalidDataError("result must be an AnalysisResult when provided.")
        if sensitivity is not None and not isinstance(sensitivity, SensitivityResult):
            raise InvalidDataError("sensitivity must be a SensitivityResult when provided.")
        if practical_significance is not None and not isinstance(
            practical_significance, PracticalSignificanceResult
        ):
            raise InvalidDataError(
                "practical_significance must be a PracticalSignificanceResult when provided."
            )
        source = result if result is not None else report._source_result
        sensitivity_source = sensitivity if sensitivity is not None else report._source_sensitivity
        practical_source = (
            practical_significance
            if practical_significance is not None
            else report._source_practical_significance
        )
        findings: list[AuditFinding] = []
        checked: list[str] = []
        skipped: list[str] = []
        references: dict[str, str] = {}
        if source is None:
            skipped.append("source_result: no independent AnalysisResult was supplied")
            return AuditResult("incomplete", (), (), tuple(skipped), {})
        source_payload = source.to_dict()
        references["analysis"] = content_reference("analysis", source_payload)
        payload = report.to_dict()
        references["report"] = content_reference("report", payload)
        try:
            expected = build_research_report(
                source,
                sensitivity=sensitivity_source,
                practical_significance=practical_source,
                title=report.title,
                include_figures=report._include_figures,
            )
        except (InvalidDataError, ReportError, KeyError, TypeError, ValueError) as exc:
            findings.append(
                AuditFinding(
                    "SOURCE_INVALID",
                    "error",
                    "analysis",
                    "analysis",
                    None,
                    None,
                    f"The source result cannot produce a valid report: {exc}",
                )
            )
            return AuditResult("failed", tuple(findings), (), (), references)
        findings.extend(_method_contract_findings(source))
        checked.append("method_contract")
        expected_payload = expected.to_dict()
        references["specification"] = content_reference(
            "specification", source_payload["specification"]
        )
        if source_payload["recommendation"] is not None:
            references["recommendation"] = content_reference(
                "recommendation", source_payload["recommendation"]
            )
        references["interpretation"] = content_reference(
            "interpretation", expected_payload["interpretation"]
        )
        if sensitivity_source is not None:
            references["sensitivity"] = content_reference(
                "sensitivity", sensitivity_source.to_dict()
            )
            checked.append("sensitivity")
        if practical_source is not None:
            references["practical_significance"] = content_reference(
                "practical_significance", practical_source.to_dict()
            )
            checked.append("practical_significance")
        _compare(expected_payload, payload, "", findings)
        checked.extend(("analysis", "interpretation", "sections", "tables", "warnings"))

        actual_exports = (
            {
                "json": report.to_json(),
                "csv": report.to_csv_tables(),
                "html": report.to_html(),
                "markdown": report.to_markdown(),
            }
            if exports is None
            else exports
        )
        if not isinstance(actual_exports, dict):
            raise InvalidDataError("exports must be a mapping of format names to content.")
        for name, supplied in actual_exports.items():
            if name == "json":
                checked.append("exports.json")
                if not isinstance(supplied, str):
                    _mismatch("exports.json", "JSON text", supplied, findings)
                    continue
                try:
                    parsed = json.loads(supplied)
                    _compare(expected_payload, parsed, "exports.json", findings)
                except (TypeError, ValueError, json.JSONDecodeError):
                    _mismatch("exports.json", "valid canonical JSON", "invalid JSON", findings)
            elif name == "csv":
                checked.append("exports.csv")
                if not isinstance(supplied, dict):
                    _mismatch("exports.csv", "CSV table mapping", supplied, findings)
                    continue
                expected_csv = expected.to_csv_tables()
                if set(supplied) != set(expected_csv):
                    _mismatch(
                        "exports.csv.table_ids", sorted(expected_csv), sorted(supplied), findings
                    )
                for table_id in sorted(expected_csv.keys() & supplied.keys()):
                    try:
                        _compare(
                            _csv_rows(expected_csv[table_id]),
                            _csv_rows(supplied[table_id]),
                            f"exports.csv.{table_id}",
                            findings,
                        )
                    except (TypeError, csv.Error):
                        _mismatch(f"exports.csv.{table_id}", "valid CSV", "invalid CSV", findings)
            elif name in {"html", "markdown", "latex"}:
                checked.append(f"exports.{name}")
                render = (
                    expected.to_html
                    if name == "html"
                    else expected.to_markdown
                    if name == "markdown"
                    else expected.to_latex
                )
                canonical = {render(style=style) for style in ("general", "apa", "ieee")}
                if not isinstance(supplied, str) or supplied not in canonical:
                    _mismatch(f"exports.{name}", "canonical rendering", supplied, findings)
            else:
                skipped.append(f"exports.{name}: unsupported format")
        status = "failed" if findings else "incomplete" if skipped else "passed"
        return AuditResult(status, tuple(findings), tuple(checked), tuple(skipped), references)
