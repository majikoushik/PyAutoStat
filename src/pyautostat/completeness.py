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
    inference = method != "dataset_profile" and payload["analysis"]["status"] == "available"
    df_applicable = method in {
        "welch_t",
        "student_t",
        "paired_t",
        "one_way_anova",
        "kruskal_wallis",
        "pearson_chi_square",
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

    items = (
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
            applicable=method != "dataset_profile",
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
            applicable=inference,
        ),
        item(
            "EFFECT_SIZE_REPORTED",
            "results",
            "Method-specific effect size",
            results.get("effect_size"),
            "sections.results.effect_size",
            applicable=inference,
        ),
        item(
            "CONFIDENCE_INTERVAL_REPORTED",
            "results",
            "Confidence interval",
            results.get("confidence_interval"),
            "sections.results.confidence_interval",
            applicable=inference,
            unavailable=(
                source_values.get("confidence_interval") is None and method == "pearson_correlation"
            ),
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
