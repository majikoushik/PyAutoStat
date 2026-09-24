"""Read-only comparisons between a source result, report and actual exports."""

from __future__ import annotations

import csv
import io
import json
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
            elif name in {"html", "markdown"}:
                checked.append(f"exports.{name}")
                canonical = expected.to_html() if name == "html" else expected.to_markdown()
                if not isinstance(supplied, str) or supplied != canonical:
                    _mismatch(f"exports.{name}", "canonical rendering", supplied, findings)
            else:
                skipped.append(f"exports.{name}: unsupported format")
        status = "failed" if findings else "incomplete" if skipped else "passed"
        return AuditResult(status, tuple(findings), tuple(checked), tuple(skipped), references)
