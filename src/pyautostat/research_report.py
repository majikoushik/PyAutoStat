"""Canonical research reports from recorded analyses."""

from __future__ import annotations

import csv
import io
import json
import math
from collections.abc import Callable
from copy import deepcopy
from html import escape
from pathlib import Path
from typing import Any

from .exceptions import InvalidDataError, ReportError
from .interpretation import InterpretationEngine, InterpretationResult, InterpretationStatus
from .narrate import executive_summary
from .practical_significance import PracticalSignificanceResult, assess_practical_significance
from .results import AnalysisResult, AnalysisStatus
from .sensitivity import SensitivityResult
from .specifications import _json_value

REPORT_SCHEMA_VERSION = 1
REPORT_STYLES = ("general", "apa", "ieee")


def _display(value: Any, *, p_value: bool = False) -> str:
    if value is None:
        return "Not available"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            return "Not available"
        if p_value and value == 0:
            return "p < 0.001 (computational zero)"
        return f"{value:.4g}" if isinstance(value, float) else str(value)
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, allow_nan=False)
    return str(value)


def _markdown(value: Any, *, cell: bool = False, p_value: bool = False) -> str:
    text = escape(_display(value, p_value=p_value), quote=False)
    for character in ("\\", chr(96), "*", "_", "[", "]", "(", ")", "#", "!", "~", "|"):
        text = text.replace(character, "\\" + character)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return text.replace("\n", "<br>" if cell else "  \n")


def _csv_cell(value: Any) -> Any:
    """Protect text beginning with a spreadsheet formula prefix."""
    if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@")):
        return "'" + value
    return value


def _style(value: str) -> str:
    if value not in REPORT_STYLES:
        raise ReportError("style must be general, apa, or ieee.")
    return value


def _styled_heading(heading: str, index: int, style: str) -> str:
    return f"{index}. {heading}" if style == "ieee" else heading


def _concise_result(data: dict[str, Any], style: str) -> str | None:
    if style == "general":
        return None
    method = data["sections"]["methods"].get("method_name")
    values = data["sections"]["results"]
    if method is None or values.get("test_statistic") is None:
        return None
    parts = [
        str(method),
        f"statistic = {_display(values.get('test_statistic'))}",
    ]
    if values.get("degrees_of_freedom") is not None:
        parts.append(f"df = {_display(values['degrees_of_freedom'])}")
    if values.get("p_value") is not None:
        parts.append(f"p = {_display(values['p_value'])}")
    if values.get("primary_estimate") is not None:
        parts.append(f"estimate = {_display(values['primary_estimate'])}")
    prefix = "APA-oriented summary: " if style == "apa" else "Technical result: "
    return prefix + ", ".join(parts) + "."


def _build_executive_summary(
    data: dict[str, Any],
    *,
    practical_verdict: str | None = None,
    sensitivity_verdict: str | None = None,
) -> tuple[str, ...]:
    """Adapt one canonical report payload to format-neutral summary content."""
    sections = data.get("sections", {})
    dataset = sections.get("dataset") if isinstance(sections, dict) else None
    methods = sections.get("methods", {}) if isinstance(sections, dict) else {}
    interpretation = sections.get("interpretation", {}) if isinstance(sections, dict) else {}
    question = sections.get("research_question", {}) if isinstance(sections, dict) else {}
    analysis = data.get("analysis", {})
    values = analysis.get("values", {}) if isinstance(analysis, dict) else {}
    profile = values.get("profile") if isinstance(values, dict) else None
    analyses: list[dict[str, Any]] = []
    if isinstance(methods, dict) and methods.get("method_id") != "dataset_profile":
        method_name = methods.get("method_name") or methods.get("method_id")
        variables = []
        if isinstance(question, dict):
            variables = [question.get("outcome"), question.get("predictor")]
            predictors = question.get("predictors")
            if isinstance(predictors, list):
                variables.extend(predictors)
        selected = [
            repr(item) for item in dict.fromkeys(item for item in variables if item is not None)
        ]
        if selected:
            method_name = f"{method_name} for {' and '.join(selected)}"
        available = methods.get("execution_status") == "available"
        if available and isinstance(interpretation, dict):
            if methods.get("method_id") == "linear_regression":
                finding = " ".join(
                    str(item).strip()
                    for item in (
                        interpretation.get("summary"),
                        interpretation.get("hypothesis"),
                    )
                    if isinstance(item, str) and item.strip()
                )
            else:
                candidate = interpretation.get("hypothesis")
                finding = (
                    candidate
                    if isinstance(candidate, str)
                    else "No reader-facing hypothesis interpretation is available."
                )
        else:
            finding = (
                "The analysis result is unavailable; no successful statistical finding is "
                "presented."
            )
        uncertainty = (
            interpretation.get("uncertainty") if isinstance(interpretation, dict) else None
        )
        if (
            available
            and methods.get("method_id") != "linear_regression"
            and isinstance(uncertainty, str)
            and uncertainty.strip()
        ):
            finding = f"{finding} {uncertainty.strip()}"
        pairwise = values.get("pairwise_comparisons") if isinstance(values, dict) else None
        if isinstance(pairwise, list) and pairwise:
            rejected = sum(
                item.get("decision") == "reject" for item in pairwise if isinstance(item, dict)
            )
            finding = (
                f"{finding} The complete pairwise family contained {len(pairwise)} comparisons; "
                f"{rejected} rejected after the recorded multiplicity control."
            )
        analyses.append(
            {
                "method_name": method_name,
                "finding": finding,
            }
        )
    practical = sections.get("practical_significance") if isinstance(sections, dict) else None
    if practical_verdict is None and isinstance(practical, dict):
        practical_verdict = practical.get("conclusion")
    sensitivity = sections.get("sensitivity_analysis") if isinstance(sections, dict) else None
    if sensitivity_verdict is None and isinstance(sensitivity, dict):
        sensitivity_verdict = sensitivity.get("comparison_note")
    diagnostics = sections.get("diagnostics") if isinstance(sections, dict) else None
    assumption_notes = (
        diagnostics.get("assumption_notes", ()) if isinstance(diagnostics, dict) else ()
    )
    limitations = [*data.get("limitations", []), *data.get("warnings", [])]
    if isinstance(assumption_notes, list):
        limitations.extend(assumption_notes)
    return executive_summary(
        profile=profile if isinstance(profile, dict) else None,
        dataset=dataset if isinstance(dataset, dict) else None,
        analyses=analyses,
        limitations=limitations,
        practical_significance=practical_verdict,
        sensitivity=sensitivity_verdict,
    )


def _latex_escape(value: Any) -> str:
    text = _display(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "{": r"\{",
        "}": r"\}",
        "$": r"\$",
        "&": r"\&",
        "#": r"\#",
        "_": r"\_",
        "%": r"\%",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(character, character) for character in text)


def _cell(value: Any, source: str | None = None) -> dict[str, Any]:
    return {"value": value, "source": source}


def _table(
    identifier: str, title: str, columns: list[str], rows: list[list[dict[str, Any]]]
) -> dict[str, Any]:
    return {"id": identifier, "title": title, "columns": columns, "rows": rows}


def _mapping_rows(mapping: dict[str, Any], prefix: str) -> list[list[dict[str, Any]]]:
    return [[_cell(key), _cell(value, f"{prefix}.{key}")] for key, value in mapping.items()]


def _write(path: str | Path, content: str, *, overwrite: bool) -> Path:
    try:
        destination = Path(path)
        if destination.exists() and not overwrite:
            raise ReportError(f"Report destination already exists: {destination}.")
        destination.write_text(content, encoding="utf-8")
        return destination
    except ReportError:
        raise
    except (OSError, ValueError, TypeError) as exc:
        raise ReportError(f"Could not write report to {path!r}: {exc}") from exc


def _omit_identifier_details(analysis: dict[str, Any]) -> bool:
    """Remove individual identifier labels from a descriptive report copy."""
    if analysis["method_id"] != "dataset_profile":
        return False
    profile = analysis["values"].get("profile")
    if not isinstance(profile, dict):
        return False
    identifiers = {
        name
        for name, role in profile.get("column_roles", {}).items()
        if isinstance(role, dict) and role.get("role") == "identifier"
    }
    identifiers.update(
        name
        for name, summary in profile.get("categorical_summary", {}).items()
        if isinstance(summary, dict)
        and isinstance(summary.get("nonmissing_count"), int)
        and summary["nonmissing_count"] >= 3
        and summary.get("observed_categories") == summary["nonmissing_count"]
    )
    declaration = analysis["specification"].get("data_dictionary") or {}
    identifiers.update(
        name
        for name, details in declaration.items()
        if isinstance(details, dict)
        and (details.get("type") == "identifier" or details.get("role") == "identifier")
    )
    for name in identifiers:
        profile.get("categorical_summary", {}).pop(name, None)
        for dictionary in (declaration, profile.get("data_dictionary", {})):
            if isinstance(dictionary, dict) and name in dictionary:
                dictionary[name] = {"type": "identifier", "report_note": "Details omitted."}
    return bool(identifiers)


class ResearchReport:
    """Effectively immutable report snapshot with safe in-memory export formats."""

    __slots__ = (
        "_payload",
        "_source_result",
        "_source_sensitivity",
        "_source_practical_significance",
        "_include_figures",
        "_on_save",
    )

    def __init__(
        self,
        payload: dict[str, Any],
        *,
        source_result: AnalysisResult | None = None,
        source_sensitivity: SensitivityResult | None = None,
        source_practical_significance: PracticalSignificanceResult | None = None,
        include_figures: bool = False,
        on_save: Callable[[str], None] | None = None,
    ) -> None:
        try:
            self._payload = _json_value(payload)
        except InvalidDataError as exc:
            raise ReportError(f"Report contains non-serializable data: {exc}") from exc
        self._source_result = deepcopy(source_result)
        self._source_sensitivity = deepcopy(source_sensitivity)
        self._source_practical_significance = deepcopy(source_practical_significance)
        self._include_figures = include_figures
        self._on_save = on_save

    @property
    def status(self) -> str:
        return str(self._payload["status"])

    @property
    def title(self) -> str:
        return str(self._payload["title"])

    def to_dict(self) -> dict[str, Any]:
        return deepcopy(self._payload)

    def to_json(self) -> str:
        return json.dumps(self._payload, ensure_ascii=False, indent=2, allow_nan=False)

    def to_csv_tables(self) -> dict[str, str]:
        outputs: dict[str, str] = {}
        for table in self._payload["tables"]:
            stream = io.StringIO(newline="")
            writer = csv.writer(stream)
            writer.writerow([_csv_cell(name) for name in table["columns"]])
            for row in table["rows"]:
                writer.writerow([_csv_cell(cell["value"]) for cell in row])
            outputs[table["id"]] = stream.getvalue()
        return outputs

    def to_markdown(self, *, style: str = "general") -> str:
        style = _style(style)
        data = self._payload
        lines = [
            f"# {_markdown(data['title'])}",
            "",
            f"**Status:** {_markdown(data['status'])}",
            "",
        ]
        summary = _concise_result(data, style)
        if summary is not None:
            lines.extend([_markdown(summary), ""])
        for index, (key, heading) in enumerate(_report_sections(data), start=1):
            lines.extend([f"## {_styled_heading(heading, index, style)}", ""])
            for label, value in data["sections"][key].items():
                if value is not None and value != [] and value != {}:
                    lines.extend(
                        [
                            f"**{_markdown(label)}:** "
                            f"{_markdown(value, p_value=key == 'results' and label == 'p_value')}",
                            "",
                        ]
                    )
        for table in data["tables"]:
            lines.extend([f"### {_markdown(table['title'])}", ""])
            columns = [_markdown(name, cell=True) for name in table["columns"]]
            lines.extend(
                [
                    "| " + " | ".join(columns) + " |",
                    "| " + " | ".join("---" for _ in columns) + " |",
                ]
            )
            for row in table["rows"]:
                cells = [
                    _markdown(
                        item["value"],
                        cell=True,
                        p_value=item["source"] == "analysis.values.p_value",
                    )
                    for item in row
                ]
                lines.append("| " + " | ".join(cells) + " |")
            lines.append("")
        for key, heading in (("limitations", "Limitations"), ("warnings", "Warnings")):
            lines.extend([f"## {heading}", ""])
            lines.extend(f"- {_markdown(item)}" for item in data[key])
            if not data[key]:
                lines.append("- None recorded.")
            lines.append("")
        return "\n".join(lines).rstrip() + "\n"

    def to_html(self, *, style: str = "general") -> str:
        style = _style(style)
        data = self._payload
        parts = [
            "<!doctype html>",
            '<html lang="en"><head><meta charset="utf-8">',
            f"<title>{escape(data['title'], quote=True)}</title>",
            "<style>body{font:16px/1.5 system-ui,sans-serif;max-width:1000px;"
            "margin:2rem auto;padding:0 1rem;color:#17212b}h1,h2{line-height:1.2}"
            "table{border-collapse:collapse;width:100%;margin:1rem 0;table-layout:fixed}"
            "th,td{border:1px solid #aeb7bf;padding:.45rem;text-align:left;"
            "vertical-align:top;overflow-wrap:anywhere}th{background:#eef2f5}"
            ".caution{border-left:4px solid #a55b00;padding:.5rem 1rem;background:#fff7e9}"
            "@media print{body{margin:0;max-width:none}.caution{break-inside:avoid}}"
            "</style></head><body><main>",
            f"<h1>{escape(data['title'], quote=True)}</h1>",
            f"<p><strong>Report status:</strong> {escape(data['status'])}</p>",
        ]
        practical_verdict = (
            self._source_practical_significance.verdict
            if self._source_practical_significance is not None
            else None
        )
        sensitivity_verdict = (
            self._source_sensitivity.verdict if self._source_sensitivity is not None else None
        )
        summary_paragraphs = _build_executive_summary(
            data,
            practical_verdict=practical_verdict,
            sensitivity_verdict=sensitivity_verdict,
        )
        parts.append('<section class="executive-summary"><h2>Executive Summary</h2>')
        parts.extend(f"<p>{escape(paragraph)}</p>" for paragraph in summary_paragraphs)
        parts.append("</section>")
        summary = _concise_result(data, style)
        if summary is not None:
            parts.append(f'<p class="oriented-summary">{escape(summary)}</p>')
        for index, (key, heading) in enumerate(_report_sections(data), start=1):
            styled = _styled_heading(heading, index, style)
            parts.append(f"<section><h2>{escape(styled)}</h2><dl>")
            for label, value in data["sections"][key].items():
                if value is not None and value != [] and value != {}:
                    display_value = escape(
                        _display(value, p_value=key == "results" and label == "p_value")
                    )
                    parts.append(
                        f"<dt><strong>{escape(label)}</strong></dt><dd>{display_value}</dd>"
                    )
            parts.append("</dl></section>")
        for table in data["tables"]:
            parts.append(f"<section><h2>{escape(table['title'])}</h2><table><thead><tr>")
            parts.extend(f'<th scope="col">{escape(name)}</th>' for name in table["columns"])
            parts.append("</tr></thead><tbody>")
            for row in table["rows"]:
                parts.append("<tr>")
                for item in row:
                    value = _display(
                        item["value"], p_value=item["source"] == "analysis.values.p_value"
                    )
                    parts.append(f"<td>{escape(value)}</td>")
                parts.append("</tr>")
            parts.append("</tbody></table></section>")
        for key, heading in (("limitations", "Limitations"), ("warnings", "Warnings")):
            parts.append(f'<section class="caution"><h2>{heading}</h2><ul>')
            parts.extend(f"<li>{escape(item)}</li>" for item in data[key])
            if not data[key]:
                parts.append("<li>None recorded.</li>")
            parts.append("</ul></section>")
        parts.append("</main></body></html>")
        return "\n".join(parts)

    def to_latex(self, *, style: str = "general") -> str:
        style = _style(style)
        data = self._payload
        lines = [
            r"\documentclass{article}",
            r"\usepackage[T1]{fontenc}",
            r"\usepackage{longtable}",
            r"\begin{document}",
            rf"\section*{{{_latex_escape(data['title'])}}}",
            rf"\textbf{{Report status:}} {_latex_escape(data['status'])}",
        ]
        summary = _concise_result(data, style)
        if summary is not None:
            lines.append(_latex_escape(summary))
        for index, (key, heading) in enumerate(_report_sections(data), start=1):
            styled = _styled_heading(heading, index, style)
            lines.append(rf"\section*{{{_latex_escape(styled)}}}")
            for label, value in data["sections"][key].items():
                if value is not None and value != [] and value != {}:
                    lines.append(rf"\textbf{{{_latex_escape(label)}:}} {_latex_escape(value)}\par")
        for key, heading in (("limitations", "Limitations"), ("warnings", "Warnings")):
            lines.extend([rf"\section*{{{heading}}}", r"\begin{itemize}"])
            values = data[key] or ["None recorded."]
            lines.extend(rf"\item {_latex_escape(item)}" for item in values)
            lines.append(r"\end{itemize}")
        lines.append(r"\end{document}")
        return "\n".join(lines) + "\n"

    def save_html(
        self, path: str | Path, *, style: str = "general", overwrite: bool = False
    ) -> Path:
        output = _write(path, self.to_html(style=style), overwrite=overwrite)
        if self._on_save is not None:
            self._on_save("html")
        return output

    def save_markdown(
        self, path: str | Path, *, style: str = "general", overwrite: bool = False
    ) -> Path:
        output = _write(path, self.to_markdown(style=style), overwrite=overwrite)
        if self._on_save is not None:
            self._on_save("markdown")
        return output

    def save_latex(
        self, path: str | Path, *, style: str = "general", overwrite: bool = False
    ) -> Path:
        output = _write(path, self.to_latex(style=style), overwrite=overwrite)
        if self._on_save is not None:
            self._on_save("latex")
        return output

    def save_json(self, path: str | Path, *, overwrite: bool = False) -> Path:
        output = _write(path, self.to_json(), overwrite=overwrite)
        if self._on_save is not None:
            self._on_save("json")
        return output

    def save_csv_tables(self, directory: str | Path, *, overwrite: bool = False) -> list[Path]:
        target = Path(directory)
        try:
            if target.exists() and not target.is_dir():
                raise ReportError(f"CSV destination is not a directory: {target}.")
            outputs = self.to_csv_tables()
            paths = [target / f"{identifier}.csv" for identifier in outputs]
            if not overwrite and any(path.exists() for path in paths):
                raise ReportError("A CSV report file exists; pass overwrite=True to replace it.")
            target.mkdir(parents=True, exist_ok=True)
            written = [
                _write(path, content, overwrite=True)
                for path, content in zip(paths, outputs.values(), strict=True)
            ]
            if self._on_save is not None:
                self._on_save("csv")
            return written
        except ReportError:
            raise
        except (OSError, ValueError, TypeError) as exc:
            raise ReportError(f"Could not save CSV report tables to {directory!r}: {exc}") from exc


_SECTIONS = (
    ("research_question", "Research question and design"),
    ("dataset", "Dataset and sample"),
    ("methods", "Methods"),
    ("diagnostics", "Diagnostics"),
    ("results", "Results"),
    ("interpretation", "Interpretation"),
)


def _report_sections(data: dict[str, Any]) -> tuple[tuple[str, str], ...]:
    additions: list[tuple[str, str]] = []
    if "sensitivity_analysis" in data.get("sections", {}):
        additions.append(("sensitivity_analysis", "Sensitivity analysis"))
    if "practical_significance" in data.get("sections", {}):
        additions.append(("practical_significance", "Practical significance"))
    return (*_SECTIONS, *additions)


def build_research_report(
    result: AnalysisResult,
    *,
    interpretation: InterpretationResult | None = None,
    sensitivity: SensitivityResult | None = None,
    practical_significance: PracticalSignificanceResult | None = None,
    title: str | None = None,
    include_figures: bool = False,
) -> ResearchReport:
    """Assemble one report from recorded analysis and interpretation records."""
    if not isinstance(result, AnalysisResult):
        raise ReportError("report() requires an AnalysisResult.")
    if title is not None and (not isinstance(title, str) or not title.strip()):
        raise ReportError("title must be a non-empty string when provided.")
    if not isinstance(include_figures, bool):
        raise ReportError("include_figures must be a Boolean.")
    if sensitivity is not None and not isinstance(sensitivity, SensitivityResult):
        raise ReportError("sensitivity must be a SensitivityResult when supplied.")
    if practical_significance is not None and not isinstance(
        practical_significance, PracticalSignificanceResult
    ):
        raise ReportError(
            "practical_significance must be a PracticalSignificanceResult when supplied."
        )
    if sensitivity is not None and sensitivity.base_result.to_dict() != result.to_dict():
        raise ReportError("The sensitivity base result does not match this report analysis.")
    if practical_significance is not None:
        expected_practical = assess_practical_significance(
            result,
            practical_significance.threshold,
            planning_status=practical_significance.threshold.planning_status,
        )
        if expected_practical.to_dict() != practical_significance.to_dict():
            raise ReportError(
                "The practical-significance result does not match this analysis and threshold."
            )
    if result.specification is None:
        raise ReportError("The AnalysisResult has no research specification.")
    try:
        analysis = result.to_dict()
    except InvalidDataError as exc:
        raise ReportError(f"The analysis contains invalid JSON data: {exc}") from exc
    identifiers_omitted = _omit_identifier_details(analysis)
    expected = InterpretationEngine().interpret(result)
    if interpretation is None:
        interpretation = expected
    elif not isinstance(interpretation, InterpretationResult):
        raise ReportError("interpretation must be an InterpretationResult.")
    elif interpretation.to_dict() != expected.to_dict():
        raise ReportError(
            "The supplied interpretation does not match this analysis. "
            "Regenerate it with ResearchAssistant.interpret(result)."
        )
    spec = analysis["specification"]
    metadata = analysis["metadata"]
    sample = metadata.get("sample", {})
    if not isinstance(sample, dict):
        raise ReportError("The analysis sample metadata is invalid.")
    original = sample.get("original_rows")
    analyzed = analysis["sample_size"]
    excluded = analysis["excluded_rows"]
    if (
        original is not None
        and analyzed is not None
        and excluded is not None
        and original != analyzed + excluded
    ):
        raise ReportError("Original, analyzed, and excluded row counts disagree.")
    if sample.get("analyzed_rows") not in (None, analyzed) or sample.get("excluded_rows") not in (
        None,
        excluded,
    ):
        raise ReportError("Sample metadata contradicts the analysis result.")
    if result.status is AnalysisStatus.AVAILABLE and analyzed is None:
        raise ReportError("An available analysis must record its analyzed row count.")
    status = (
        "unavailable"
        if result.status is AnalysisStatus.UNAVAILABLE
        or interpretation.status is InterpretationStatus.UNAVAILABLE
        else "partial"
        if interpretation.status is InterpretationStatus.PARTIAL
        or original is None
        or excluded is None
        else "complete"
    )
    if status == "unavailable":
        # Stale numbers in an unavailable envelope are not successful findings.
        analysis["values"] = {}
    values = analysis["values"] if status != "unavailable" else {}
    interpreted = interpretation.to_dict()
    validated = interpreted["metadata"]
    visible = values.copy()
    if status != "unavailable" and analysis["method_id"] != "dataset_profile":
        if analysis["method_id"] != "cronbach_alpha":
            visible["p_value"] = validated.get("p_value")
            visible["test_statistic"] = validated.get("test_statistic")
            visible["primary_estimate"] = validated.get("primary_estimate")
            visible["confidence_interval"] = validated.get("confidence_interval")
        finding_codes = {item["code"] for item in interpreted["findings"]}
        effect = values.get("effect_size")
        if analysis["method_id"] in {
            "welch_t",
            "student_t",
            "paired_t",
            "one_sample_t",
        }:
            if isinstance(effect, dict) and "standardized_effect_reported" in finding_codes:
                effect = deepcopy(effect)
                if "effect_interval_reported" not in finding_codes:
                    effect["confidence_interval"] = None
                visible["effect_size"] = effect
            else:
                visible["effect_size"] = None
        elif "effect_unavailable" in finding_codes:
            visible["effect_size"] = None
        elif isinstance(effect, dict):
            effect = deepcopy(effect)
            effect["confidence_interval"] = validated.get("confidence_interval")
            visible["effect_size"] = effect
    recommendation = analysis["recommendation"] or {}
    question = spec["question"]
    methods = {
        "method_id": analysis["method_id"],
        "method_name": metadata.get("method_name") or recommendation.get("method_name"),
        "execution_status": analysis["status"],
        "rationale": recommendation.get("rationale"),
        "declared_design": spec["design"],
        "estimand": question.get("estimand"),
        "null_hypothesis": metadata.get("null_hypothesis"),
        "alternative_hypothesis": metadata.get("alternative_hypothesis"),
        "alpha": spec["options"]["alpha"],
        "confidence_level": spec["options"]["confidence_level"],
        "effect_definition": (visible.get("effect_size") or {}).get("definition"),
        "primary_interval_method": (visible.get("confidence_interval") or {}).get("method"),
        "effective_random_seed": metadata.get("effective_random_seed")
        or (visible.get("confidence_interval") or {}).get("random_state"),
        "bootstrap_resamples": metadata.get("bootstrap_default_resamples")
        or (visible.get("confidence_interval") or {}).get("requested_resamples"),
        "required_assumptions": analysis["assumptions"],
        "unit_id": spec.get("unit_id"),
        "reference_value": question.get("reference_value"),
        "zero_method": metadata.get("zero_method"),
        "p_value_method": metadata.get("p_value_method"),
        "pairwise_method": metadata.get("pairwise_method"),
        "multiplicity_control": metadata.get("multiplicity_control"),
        "pairwise_comparison_count": metadata.get("pairwise_comparison_count"),
        "covariance_type": visible.get("covariance_type"),
        "intercept": visible.get("intercept"),
    }
    dataset = {
        "original_rows": original,
        "analyzed_rows": analyzed,
        "excluded_rows": excluded,
        "group_order": metadata.get("group_order"),
        "group_sizes": sample.get("group_sizes"),
        "effective_pair_count": sample.get("effective_pair_count"),
        "complete_pairs": sample.get("complete_pairs"),
        "total_units": sample.get("total_units"),
        "incomplete_units": sample.get("incomplete_units"),
        "excluded_units": sample.get("excluded_units"),
        "missing_unit_rows": sample.get("missing_unit_rows"),
        "complete_pair_rule": sample.get("complete_pair_rule"),
        "nonzero_differences": sample.get("nonzero_differences"),
        "zero_differences": sample.get("zero_differences"),
        "contrast": metadata.get("contrast"),
    }
    if isinstance(dataset["group_sizes"], list) and analyzed is not None:
        groups = dataset["group_sizes"]
        if any(
            not isinstance(item, dict)
            or not isinstance(item.get("size"), int)
            or isinstance(item["size"], bool)
            or item["size"] < 0
            or "group" not in item
            for item in groups
        ):
            raise ReportError("Per-group sample sizes are invalid.")
        if sum(item["size"] for item in groups) != analyzed:
            raise ReportError("Per-group sample sizes disagree with analyzed rows.")
    report_results = {
        "test_statistic": visible.get("test_statistic"),
        "degrees_of_freedom": visible.get("degrees_of_freedom"),
        "p_value": visible.get("p_value"),
        "primary_estimate": visible.get("primary_estimate"),
        "estimate_name": visible.get("estimate_name"),
        "estimate_unit": visible.get("estimate_unit"),
        "effect_size": visible.get("effect_size"),
        "confidence_interval": visible.get("confidence_interval"),
        "sample_mean": visible.get("sample_mean"),
        "reference_value": visible.get("reference_value"),
        "standard_error": visible.get("standard_error"),
        "group_summaries": visible.get("group_summaries"),
        "pairwise_comparisons": visible.get("pairwise_comparisons"),
        "outcome": visible.get("outcome"),
        "predictors": visible.get("predictors"),
        "target": visible.get("target"),
        "covariance_type": visible.get("covariance_type"),
        "intercept": visible.get("intercept"),
        "model_fit": visible.get("model_fit"),
        "coefficients": visible.get("coefficients"),
        "design_matrix": visible.get("design_matrix"),
        "diagnostics": visible.get("diagnostics"),
        "items": visible.get("items"),
        "item_count": visible.get("item_count"),
        "cronbach_alpha": visible.get("cronbach_alpha"),
        "item_statistics": visible.get("item_statistics"),
        "inter_item_correlations": visible.get("inter_item_correlations"),
        "mean_inter_item_correlation": visible.get("mean_inter_item_correlation"),
        "negative_inter_item_correlations": visible.get("negative_inter_item_correlations"),
        "missingness": visible.get("missingness"),
        "scoring": visible.get("scoring"),
        "formula": visible.get("formula"),
    }
    interpretation_section = {
        "status": interpreted["status"],
        "summary": interpreted["summary"],
        "hypothesis": interpreted["hypothesis_interpretation"],
        "effect": interpreted["effect_interpretation"],
        "uncertainty": interpreted["uncertainty_interpretation"],
        "assumptions": interpreted["assumption_notes"],
        "conclusion": interpreted["conclusion"],
    }
    warnings = list(dict.fromkeys([*analysis["warnings"], *interpreted["warnings"]]))
    limitations = list(dict.fromkeys(interpreted["limitations"]))
    if identifiers_omitted:
        limitations.append(
            "Identifier category labels were omitted from this report; "
            "small aggregate groups may still disclose information."
        )
    if status == "unavailable":
        warnings.append("No successful statistical result is presented in this report.")
    if original is None or excluded is None:
        limitations.append("Complete original/analyzed/excluded row accounting is unavailable.")
    tables: list[dict[str, Any]] = []
    sample_fields = {
        name: dataset[name]
        for name in ("original_rows", "analyzed_rows", "excluded_rows")
        if dataset[name] is not None
    }
    if sample_fields:
        tables.append(
            _table(
                "sample_accounting",
                "Sample accounting",
                ["Measure", "Rows"],
                _mapping_rows(sample_fields, "sections.dataset"),
            )
        )
    pair_fields = {
        name: dataset[name]
        for name in (
            "total_units",
            "complete_pairs",
            "incomplete_units",
            "excluded_units",
            "nonzero_differences",
            "zero_differences",
        )
        if dataset[name] is not None
    }
    if pair_fields:
        tables.append(
            _table(
                "pair_accounting",
                "Paired-unit accounting",
                ["Measure", "Count"],
                _mapping_rows(pair_fields, "sections.dataset"),
            )
        )
    if analysis["method_id"] == "cronbach_alpha" and status != "unavailable":
        interval = visible.get("confidence_interval") or {}
        tables.append(
            _table(
                "reliability_summary",
                "Reliability summary",
                [
                    "Items",
                    "Complete respondents",
                    "Cronbach alpha",
                    "CI lower",
                    "CI upper",
                    "CI status",
                    "Valid bootstrap resamples",
                    "Requested bootstrap resamples",
                ],
                [
                    [
                        _cell(visible.get("item_count"), "analysis.values.item_count"),
                        _cell(analyzed, "analysis.sample_size"),
                        _cell(visible.get("cronbach_alpha"), "analysis.values.cronbach_alpha"),
                        _cell(interval.get("lower"), "analysis.values.confidence_interval.lower"),
                        _cell(interval.get("upper"), "analysis.values.confidence_interval.upper"),
                        _cell(interval.get("status"), "analysis.values.confidence_interval.status"),
                        _cell(
                            interval.get("valid_resamples"),
                            "analysis.values.confidence_interval.valid_resamples",
                        ),
                        _cell(
                            interval.get("requested_resamples"),
                            "analysis.values.confidence_interval.requested_resamples",
                        ),
                    ]
                ],
            )
        )
        item_rows = visible.get("item_statistics")
        if isinstance(item_rows, list):
            tables.append(
                _table(
                    "reliability_items",
                    "Reliability item diagnostics",
                    [
                        "Item",
                        "Valid",
                        "Missing",
                        "Missing percent",
                        "Analyzed",
                        "Mean",
                        "SD",
                        "Minimum",
                        "Maximum",
                        "Corrected item-total correlation",
                        "Corrected item-total status",
                        "Alpha if deleted",
                        "Alpha if deleted status",
                        "Delta from full alpha",
                    ],
                    [
                        [
                            _cell(item.get(field), f"analysis.values.item_statistics[{i}].{field}")
                            for field in (
                                "item",
                                "valid_count",
                                "missing_count",
                                "missing_percentage",
                                "analyzed_count",
                                "mean",
                                "standard_deviation",
                                "minimum",
                                "maximum",
                                "corrected_item_total_correlation",
                                "corrected_item_total_status",
                                "alpha_if_deleted",
                                "alpha_if_deleted_status",
                                "delta_from_full_alpha",
                            )
                        ]
                        for i, item in enumerate(item_rows)
                        if isinstance(item, dict)
                    ],
                )
            )
        correlations = visible.get("inter_item_correlations")
        if isinstance(correlations, dict):
            names = correlations.get("items")
            matrix = correlations.get("values")
            if isinstance(names, list) and isinstance(matrix, list):
                tables.append(
                    _table(
                        "inter_item_correlations",
                        "Inter-item correlations",
                        ["Item", *names],
                        [
                            [
                                _cell(name, f"analysis.values.inter_item_correlations.items[{i}]"),
                                *[
                                    _cell(
                                        value,
                                        f"analysis.values.inter_item_correlations.values[{i}][{j}]",
                                    )
                                    for j, value in enumerate(row)
                                ],
                            ]
                            for i, (name, row) in enumerate(zip(names, matrix, strict=True))
                            if isinstance(row, list)
                        ],
                    )
                )
    groups = dataset["group_sizes"]
    if isinstance(groups, list) and groups and status != "unavailable":
        tables.append(
            _table(
                "group_sizes",
                "Group sizes",
                ["Group", "Analyzed rows"],
                [
                    [
                        _cell(item["group"], f"analysis.metadata.sample.group_sizes[{i}].group"),
                        _cell(item["size"], f"analysis.metadata.sample.group_sizes[{i}].size"),
                    ]
                    for i, item in enumerate(groups)
                ],
            )
        )
    summaries = visible.get("group_summaries")
    if isinstance(summaries, list) and summaries and status != "unavailable":
        tables.append(
            _table(
                "group_summaries",
                "Group summaries",
                ["Group", "N", "Mean", "SD", "Median"],
                [
                    [
                        _cell(item.get("group"), f"analysis.values.group_summaries[{i}].group"),
                        _cell(
                            item.get("sample_size"),
                            f"analysis.values.group_summaries[{i}].sample_size",
                        ),
                        _cell(item.get("mean"), f"analysis.values.group_summaries[{i}].mean"),
                        _cell(
                            item.get("standard_deviation"),
                            f"analysis.values.group_summaries[{i}].standard_deviation",
                        ),
                        _cell(item.get("median"), f"analysis.values.group_summaries[{i}].median"),
                    ]
                    for i, item in enumerate(summaries)
                    if isinstance(item, dict)
                ],
            )
        )
    coefficients = visible.get("coefficients")
    if isinstance(coefficients, list) and coefficients and status != "unavailable":
        tables.append(
            _table(
                "regression_coefficients",
                "Regression coefficients",
                [
                    "Term",
                    "Predictor",
                    "Comparison level",
                    "Reference level",
                    "Estimate",
                    "SE",
                    "t",
                    "p",
                    "CI lower",
                    "CI upper",
                    "Standardized beta",
                    "Decision",
                ],
                [
                    [
                        _cell(item.get("term"), f"analysis.values.coefficients[{i}].term"),
                        _cell(
                            item.get("predictor"), f"analysis.values.coefficients[{i}].predictor"
                        ),
                        _cell(item.get("level"), f"analysis.values.coefficients[{i}].level"),
                        _cell(
                            item.get("reference_level"),
                            f"analysis.values.coefficients[{i}].reference_level",
                        ),
                        _cell(item.get("estimate"), f"analysis.values.coefficients[{i}].estimate"),
                        _cell(
                            item.get("standard_error"),
                            f"analysis.values.coefficients[{i}].standard_error",
                        ),
                        _cell(
                            item.get("statistic"), f"analysis.values.coefficients[{i}].statistic"
                        ),
                        _cell(item.get("p_value"), f"analysis.values.coefficients[{i}].p_value"),
                        _cell(
                            (item.get("confidence_interval") or {}).get("lower"),
                            f"analysis.values.coefficients[{i}].confidence_interval.lower",
                        ),
                        _cell(
                            (item.get("confidence_interval") or {}).get("upper"),
                            f"analysis.values.coefficients[{i}].confidence_interval.upper",
                        ),
                        _cell(
                            item.get("standardized_beta"),
                            f"analysis.values.coefficients[{i}].standardized_beta",
                        ),
                        _cell(item.get("decision"), f"analysis.values.coefficients[{i}].decision"),
                    ]
                    for i, item in enumerate(coefficients)
                    if isinstance(item, dict)
                ],
            )
        )
        model_fit = visible.get("model_fit")
        if isinstance(model_fit, dict):
            tables.append(
                _table(
                    "regression_model_fit",
                    "Regression model fit",
                    ["Measure", "Value"],
                    _mapping_rows(model_fit, "analysis.values.model_fit"),
                )
            )
        diagnostics = visible.get("diagnostics")
        if isinstance(diagnostics, dict):
            vif = diagnostics.get("vif")
            vif_terms = vif.get("terms") if isinstance(vif, dict) else None
            if isinstance(vif_terms, list) and vif_terms:
                tables.append(
                    _table(
                        "regression_vif",
                        "Regression collinearity diagnostics",
                        ["Term", "Predictor", "VIF", "Status", "Advisory"],
                        [
                            [
                                _cell(
                                    item.get("term"),
                                    f"analysis.values.diagnostics.vif.terms[{i}].term",
                                ),
                                _cell(
                                    item.get("predictor"),
                                    f"analysis.values.diagnostics.vif.terms[{i}].predictor",
                                ),
                                _cell(
                                    item.get("value"),
                                    f"analysis.values.diagnostics.vif.terms[{i}].value",
                                ),
                                _cell(
                                    item.get("status"),
                                    f"analysis.values.diagnostics.vif.terms[{i}].status",
                                ),
                                _cell(
                                    item.get("advisory"),
                                    f"analysis.values.diagnostics.vif.terms[{i}].advisory",
                                ),
                            ]
                            for i, item in enumerate(vif_terms)
                            if isinstance(item, dict)
                        ],
                    )
                )
            bp = diagnostics.get("breusch_pagan")
            normality = diagnostics.get("residual_normality")
            influence = diagnostics.get("influence")
            diagnostic_rows = []
            if isinstance(bp, dict):
                diagnostic_rows.append(
                    [
                        _cell("Breusch-Pagan", "analysis.values.diagnostics.breusch_pagan"),
                        _cell(
                            bp.get("lm_statistic"),
                            "analysis.values.diagnostics.breusch_pagan.lm_statistic",
                        ),
                        _cell(
                            bp.get("lm_p_value"),
                            "analysis.values.diagnostics.breusch_pagan.lm_p_value",
                        ),
                        _cell(bp.get("status"), "analysis.values.diagnostics.breusch_pagan.status"),
                    ]
                )
            if isinstance(normality, dict):
                diagnostic_rows.append(
                    [
                        _cell(
                            normality.get("method"),
                            "analysis.values.diagnostics.residual_normality.method",
                        ),
                        _cell(
                            normality.get("statistic"),
                            "analysis.values.diagnostics.residual_normality.statistic",
                        ),
                        _cell(
                            normality.get("p_value"),
                            "analysis.values.diagnostics.residual_normality.p_value",
                        ),
                        _cell(
                            normality.get("status"),
                            "analysis.values.diagnostics.residual_normality.status",
                        ),
                    ]
                )
            if isinstance(influence, dict):
                diagnostic_rows.append(
                    [
                        _cell("Influence heuristics", "analysis.values.diagnostics.influence"),
                        _cell(
                            influence.get("flagged_count"),
                            "analysis.values.diagnostics.influence.flagged_count",
                        ),
                        _cell(None, "analysis.values.diagnostics.influence"),
                        _cell(
                            influence.get("status"), "analysis.values.diagnostics.influence.status"
                        ),
                    ]
                )
            if diagnostic_rows:
                tables.append(
                    _table(
                        "regression_diagnostics",
                        "Regression diagnostic summary",
                        ["Diagnostic", "Statistic or count", "p", "Status"],
                        diagnostic_rows,
                    )
                )
    pairwise = visible.get("pairwise_comparisons")
    if isinstance(pairwise, list) and pairwise and status != "unavailable":
        tables.append(
            _table(
                "pairwise_comparisons",
                "Multiplicity-controlled pairwise comparisons",
                [
                    "Procedure",
                    "First group",
                    "Second group",
                    "Estimate",
                    "Statistic",
                    "Raw p",
                    "Adjusted p",
                    "Adjustment",
                    "CI lower",
                    "CI upper",
                    "Effect",
                    "N first",
                    "N second",
                    "SE",
                    "DF",
                    "Decision",
                ],
                [
                    [
                        _cell(
                            item.get("procedure"),
                            f"analysis.values.pairwise_comparisons[{i}].procedure",
                        ),
                        _cell(
                            item.get("group1"), f"analysis.values.pairwise_comparisons[{i}].group1"
                        ),
                        _cell(
                            item.get("group2"), f"analysis.values.pairwise_comparisons[{i}].group2"
                        ),
                        _cell(
                            item.get("estimate"),
                            f"analysis.values.pairwise_comparisons[{i}].estimate",
                        ),
                        _cell(
                            item.get("statistic"),
                            f"analysis.values.pairwise_comparisons[{i}].statistic",
                        ),
                        _cell(
                            item.get("raw_p_value"),
                            f"analysis.values.pairwise_comparisons[{i}].raw_p_value",
                        ),
                        _cell(
                            item.get("adjusted_p_value"),
                            f"analysis.values.pairwise_comparisons[{i}].adjusted_p_value",
                        ),
                        _cell(
                            item.get("adjustment_method"),
                            f"analysis.values.pairwise_comparisons[{i}].adjustment_method",
                        ),
                        _cell(
                            (item.get("confidence_interval") or {}).get("lower"),
                            f"analysis.values.pairwise_comparisons[{i}].confidence_interval.lower",
                        ),
                        _cell(
                            (item.get("confidence_interval") or {}).get("upper"),
                            f"analysis.values.pairwise_comparisons[{i}].confidence_interval.upper",
                        ),
                        _cell(
                            item.get("effect_size"),
                            f"analysis.values.pairwise_comparisons[{i}].effect_size",
                        ),
                        _cell(
                            (item.get("sample_sizes") or {}).get("group1"),
                            f"analysis.values.pairwise_comparisons[{i}].sample_sizes.group1",
                        ),
                        _cell(
                            (item.get("sample_sizes") or {}).get("group2"),
                            f"analysis.values.pairwise_comparisons[{i}].sample_sizes.group2",
                        ),
                        _cell(
                            item.get("standard_error"),
                            f"analysis.values.pairwise_comparisons[{i}].standard_error",
                        ),
                        _cell(
                            item.get("degrees_of_freedom"),
                            f"analysis.values.pairwise_comparisons[{i}].degrees_of_freedom",
                        ),
                        _cell(
                            item.get("decision"),
                            f"analysis.values.pairwise_comparisons[{i}].decision",
                        ),
                    ]
                    for i, item in enumerate(pairwise)
                    if isinstance(item, dict)
                ],
            )
        )
    if status != "unavailable":
        if analysis["method_id"] == "dataset_profile":
            profile = values.get("profile", {})
            descriptive = profile.get("descriptive", {}) if isinstance(profile, dict) else {}
            if descriptive:
                limitations.append(
                    "Coefficient of variation is most interpretable for ratio-scale measurements "
                    "with a meaningful zero; no universal magnitude label is applied."
                )
                tables.append(
                    _table(
                        "descriptive_statistics",
                        "Observed descriptive statistics",
                        [
                            "Variable",
                            "N",
                            "Mean",
                            "Median",
                            "SD",
                            "P5",
                            "P25",
                            "P75",
                            "P95",
                            "CV (%)",
                        ],
                        [
                            [_cell(name)]
                            + [
                                _cell(
                                    stats.get(key),
                                    f"analysis.values.profile.descriptive.{name}.{key}",
                                )
                                for key in ("count", "mean", "median", "std")
                            ]
                            + [
                                _cell(
                                    stats.get("percentiles", {}).get(key),
                                    f"analysis.values.profile.descriptive.{name}.percentiles.{key}",
                                )
                                for key in ("p05", "p25", "p75", "p95")
                            ]
                            + [
                                _cell(
                                    stats.get("coefficient_of_variation"),
                                    "analysis.values.profile.descriptive."
                                    f"{name}.coefficient_of_variation",
                                )
                            ]
                            for name, stats in descriptive.items()
                            if isinstance(stats, dict)
                        ],
                    )
                )
            categorical = profile.get("categorical_summary", {})
            if isinstance(categorical, dict):
                frequency_index = 0
                for name, summary in categorical.items():
                    frequencies = (
                        summary.get("frequencies", []) if isinstance(summary, dict) else []
                    )
                    if frequencies:
                        frequency_index += 1
                        tables.append(
                            _table(
                                f"frequency_{frequency_index}",
                                f"Frequency summary: {name}",
                                ["Level", "Count", "Valid percent"],
                                [
                                    [
                                        _cell(row.get("value")),
                                        _cell(row.get("count")),
                                        _cell(row.get("percentage")),
                                    ]
                                    for row in frequencies
                                ],
                            )
                        )
        else:
            fields = {
                "test_statistic": visible.get("test_statistic"),
                "degrees_of_freedom": visible.get("degrees_of_freedom"),
                "p_value": visible.get("p_value"),
                "primary_estimate": visible.get("primary_estimate"),
                "sample_mean": visible.get("sample_mean"),
                "reference_value": visible.get("reference_value"),
                "standard_error": visible.get("standard_error"),
            }
            tables.append(
                _table(
                    "statistical_results",
                    "Statistical results",
                    ["Quantity", "Value"],
                    [
                        [
                            _cell(name),
                            _cell(
                                value,
                                f"analysis.values.{name}"
                                if value == values.get(name)
                                else f"interpretation.metadata.{name}",
                            ),
                        ]
                        for name, value in fields.items()
                    ],
                )
            )
            effect = visible.get("effect_size")
            if isinstance(effect, dict):
                tables.append(
                    _table(
                        "effect_estimates",
                        "Effect estimate",
                        ["Measure", "Estimate", "Definition"],
                        [
                            [
                                _cell(effect.get("name"), "analysis.values.effect_size.name"),
                                _cell(effect.get("value"), "analysis.values.effect_size.value"),
                                _cell(
                                    effect.get("definition"),
                                    "analysis.values.effect_size.definition",
                                ),
                            ]
                        ],
                    )
                )
            interval_rows = []
            for path, interval in (
                ("confidence_interval", visible.get("confidence_interval")),
                (
                    "effect_size.confidence_interval",
                    effect.get("confidence_interval")
                    if isinstance(effect, dict)
                    and analysis["method_id"] in {"welch_t", "student_t"}
                    and "effect_interval_reported" in finding_codes
                    else None,
                ),
            ):
                if isinstance(interval, dict):
                    interval_rows.append(
                        [
                            _cell(interval.get(field), f"analysis.values.{path}.{field}")
                            for field in ("quantity", "lower", "upper", "level", "method")
                        ]
                    )
            if interval_rows:
                tables.append(
                    _table(
                        "confidence_intervals",
                        "Recorded confidence intervals",
                        ["Quantity", "Lower", "Upper", "Level", "Method"],
                        interval_rows,
                    )
                )
            if analysis["method_id"] == "fisher_exact":
                observed = metadata.get("observed_counts")
                row_levels = metadata.get("row_order")
                column_levels = metadata.get("column_order")
                if (
                    isinstance(observed, list)
                    and len(observed) == 2
                    and all(isinstance(row, list) and len(row) == 2 for row in observed)
                    and isinstance(row_levels, list)
                    and len(row_levels) == 2
                    and isinstance(column_levels, list)
                    and len(column_levels) == 2
                ):
                    tables.append(
                        _table(
                            "observed_contingency_table",
                            "Observed 2x2 contingency table",
                            [metadata.get("row_variable"), *column_levels],
                            [
                                [
                                    _cell(
                                        row_levels[index],
                                        f"analysis.metadata.row_order[{index}]",
                                    ),
                                    *[
                                        _cell(
                                            count,
                                            "analysis.metadata.observed_counts"
                                            f"[{index}][{column_index}]",
                                        )
                                        for column_index, count in enumerate(row)
                                    ],
                                ]
                                for index, row in enumerate(observed)
                            ],
                        )
                    )
    figures: list[dict[str, Any]] = []
    if include_figures and analysis["method_id"] == "dataset_profile" and status != "unavailable":
        profile = values.get("profile", {})
        histograms = profile.get("histograms", {}) if isinstance(profile, dict) else {}
        for name, histogram in histograms.items():
            if not isinstance(histogram, dict):
                continue
            edges = histogram.get("bin_edges")
            counts = histogram.get("counts")
            if (
                isinstance(edges, list)
                and isinstance(counts, list)
                and len(edges) == len(counts) + 1
                and all(isinstance(x, (int, float)) and math.isfinite(x) for x in edges + counts)
            ):
                figures.append(
                    {
                        "id": f"histogram_{len(figures) + 1}",
                        "kind": "histogram_data",
                        "variable": name,
                        "bin_edges": edges,
                        "counts": counts,
                        "source": f"analysis.values.profile.histograms.{name}",
                        "note": "Descriptive bins; peaks are not a formal bimodality test.",
                    }
                )
                index = len(figures)
                tables.append(
                    _table(
                        f"histogram_{index}_bins",
                        f"Histogram bin data: {name}",
                        ["Lower edge", "Upper edge", "Count"],
                        [
                            [
                                _cell(
                                    edges[i],
                                    f"analysis.values.profile.histograms.{name}.bin_edges[{i}]",
                                ),
                                _cell(
                                    edges[i + 1],
                                    f"analysis.values.profile.histograms.{name}.bin_edges[{i + 1}]",
                                ),
                                _cell(
                                    count, f"analysis.values.profile.histograms.{name}.counts[{i}]"
                                ),
                            ]
                            for i, count in enumerate(counts)
                        ],
                    )
                )
            if len(figures) == 3:
                break
    sections = {
        "research_question": {
            "objective": question["objective"],
            "description": question["description"],
            "outcome": question["outcome"],
            "predictor": question["predictor"],
            "predictors": question.get("predictors"),
            "items": question.get("items"),
            "estimand": question["estimand"],
            "declared_design": spec["design"],
            "unit_id": spec.get("unit_id"),
            "condition_order": spec.get("condition_order"),
            "reference_value": question.get("reference_value"),
            "reference_levels": spec.get("options", {}).get("reference_levels"),
            "covariance_type": visible.get("covariance_type"),
            "data_dictionary": spec.get("data_dictionary"),
        },
        "dataset": dataset,
        "methods": methods,
        "diagnostics": {
            "recorded": metadata.get("diagnostics"),
            "assumption_notes": interpreted["assumption_notes"],
        },
        "results": report_results,
        "interpretation": interpretation_section,
    }
    sensitivity_payload = sensitivity.to_dict() if sensitivity is not None else None
    practical_payload = (
        practical_significance.to_dict() if practical_significance is not None else None
    )
    if sensitivity_payload is not None:
        sections["sensitivity_analysis"] = {
            "status": sensitivity_payload["status"],
            "base_method_id": sensitivity_payload["base_result"]["method_id"],
            "declared_scenario_count": sensitivity_payload["comparison_summary"][
                "declared_scenario_count"
            ],
            "same_estimand_scenarios": sensitivity_payload["comparison_summary"][
                "same_estimand_scenarios"
            ],
            "different_estimand_scenarios": sensitivity_payload["comparison_summary"][
                "different_estimand_scenarios"
            ],
            "comparison_note": sensitivity_payload["comparison_summary"]["note"],
        }
        tables.append(
            _table(
                "sensitivity_scenarios",
                "Declared sensitivity scenarios",
                [
                    "Scenario",
                    "Rationale",
                    "Planning status",
                    "Method",
                    "Status",
                    "Comparability",
                    "Estimate quantity",
                    "Estimate",
                    "P-value (secondary)",
                    "Analyzed rows",
                    "Warnings",
                ],
                [
                    [
                        _cell(item["name"], f"sensitivity.scenario_results[{index}].name"),
                        _cell(
                            item["specification"]["rationale"],
                            f"sensitivity.scenario_results[{index}].specification.rationale",
                        ),
                        _cell(
                            item["specification"]["planning_status"],
                            f"sensitivity.scenario_results[{index}].specification.planning_status",
                        ),
                        _cell(
                            item["method_id"],
                            f"sensitivity.scenario_results[{index}].method_id",
                        ),
                        _cell(item["status"], f"sensitivity.scenario_results[{index}].status"),
                        _cell(
                            item["comparability"],
                            f"sensitivity.scenario_results[{index}].comparability",
                        ),
                        _cell(
                            item["estimate_quantity"],
                            f"sensitivity.scenario_results[{index}].estimate_quantity",
                        ),
                        _cell(
                            item["primary_estimate"],
                            f"sensitivity.scenario_results[{index}].primary_estimate",
                        ),
                        _cell(
                            item["p_value"],
                            f"sensitivity.scenario_results[{index}].p_value",
                        ),
                        _cell(
                            item["sample_size"],
                            f"sensitivity.scenario_results[{index}].sample_size",
                        ),
                        _cell(
                            item["warnings"],
                            f"sensitivity.scenario_results[{index}].warnings",
                        ),
                    ]
                    for index, item in enumerate(sensitivity_payload["scenario_results"])
                ],
            )
        )
        warnings.extend(sensitivity_payload["warnings"])
        if status != "unavailable" and sensitivity_payload["status"] != "complete":
            status = "partial"
    if practical_payload is not None:
        threshold = practical_payload["threshold"]
        sections["practical_significance"] = {
            "status": practical_payload["status"],
            "quantity": practical_payload["quantity"],
            "estimate": practical_payload["estimate"],
            "threshold": threshold["minimum_magnitude"],
            "direction": threshold["direction"],
            "unit": threshold["unit"],
            "rationale": threshold["rationale"],
            "confidence_interval": practical_payload["confidence_interval"],
            "point_estimate_relation": practical_payload["point_estimate_relation"],
            "confidence_interval_relation": practical_payload["confidence_interval_relation"],
            "statistical_significance": practical_payload["statistical_significance"],
            "conclusion": practical_payload["conclusion"],
        }
        tables.append(
            _table(
                "practical_significance",
                "Researcher-defined meaningful-effect threshold",
                ["Field", "Value"],
                _mapping_rows(
                    sections["practical_significance"], "sections.practical_significance"
                ),
            )
        )
        warnings.extend(practical_payload["warnings"])
        if status != "unavailable" and practical_payload["status"] != "complete":
            status = "partial"
    follow_up_supplied = sensitivity_payload is not None or practical_payload is not None
    payload = {
        "schema_version": 2 if follow_up_supplied else REPORT_SCHEMA_VERSION,
        "status": status,
        "title": title.strip() if title is not None else "Statistical Research Report",
        "analysis": analysis,
        "interpretation": interpreted,
        "sections": sections,
        "tables": tables,
        "figures": figures,
        "limitations": limitations,
        "warnings": list(dict.fromkeys(warnings)),
    }
    if sensitivity_payload is not None:
        payload["sensitivity"] = sensitivity_payload
    if practical_payload is not None:
        payload["practical_significance"] = practical_payload
    return ResearchReport(
        payload,
        source_result=result,
        source_sensitivity=sensitivity,
        source_practical_significance=practical_significance,
        include_figures=include_figures,
    )
