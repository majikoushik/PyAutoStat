"""Canonical Phase 8 research reports from recorded analyses."""

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
from .practical_significance import PracticalSignificanceResult, assess_practical_significance
from .results import AnalysisResult, AnalysisStatus
from .sensitivity import SensitivityResult
from .specifications import _json_value

REPORT_SCHEMA_VERSION = 1


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
    """Effectively immutable report snapshot with four in-memory export formats."""

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

    def to_markdown(self) -> str:
        data = self._payload
        lines = [
            f"# {_markdown(data['title'])}",
            "",
            f"**Status:** {_markdown(data['status'])}",
            "",
        ]
        for key, heading in _report_sections(data):
            lines.extend([f"## {heading}", ""])
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

    def to_html(self) -> str:
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
        for key, heading in _report_sections(data):
            parts.append(f"<section><h2>{heading}</h2><dl>")
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

    def save_html(self, path: str | Path, *, overwrite: bool = False) -> Path:
        output = _write(path, self.to_html(), overwrite=overwrite)
        if self._on_save is not None:
            self._on_save("html")
        return output

    def save_markdown(self, path: str | Path, *, overwrite: bool = False) -> Path:
        output = _write(path, self.to_markdown(), overwrite=overwrite)
        if self._on_save is not None:
            self._on_save("markdown")
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
    """Assemble one report from recorded Phase 6 and Phase 7 records."""
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
        visible["p_value"] = validated.get("p_value")
        visible["test_statistic"] = validated.get("test_statistic")
        visible["primary_estimate"] = validated.get("primary_estimate")
        visible["confidence_interval"] = validated.get("confidence_interval")
        finding_codes = {item["code"] for item in interpreted["findings"]}
        effect = values.get("effect_size")
        if analysis["method_id"] in {"welch_t", "student_t"}:
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
        "effective_random_seed": metadata.get("effective_random_seed"),
        "bootstrap_resamples": metadata.get("bootstrap_default_resamples"),
        "required_assumptions": analysis["assumptions"],
    }
    dataset = {
        "original_rows": original,
        "analyzed_rows": analyzed,
        "excluded_rows": excluded,
        "group_order": metadata.get("group_order"),
        "group_sizes": sample.get("group_sizes"),
        "effective_pair_count": sample.get("effective_pair_count"),
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
    if status != "unavailable":
        if analysis["method_id"] == "dataset_profile":
            profile = values.get("profile", {})
            descriptive = profile.get("descriptive", {}) if isinstance(profile, dict) else {}
            if descriptive:
                tables.append(
                    _table(
                        "descriptive_statistics",
                        "Observed descriptive statistics",
                        ["Variable", "N", "Mean", "Median", "SD"],
                        [
                            [_cell(name)]
                            + [
                                _cell(
                                    stats.get(key),
                                    f"analysis.values.profile.descriptive.{name}.{key}",
                                )
                                for key in ("count", "mean", "median", "std")
                            ]
                            for name, stats in descriptive.items()
                            if isinstance(stats, dict)
                        ],
                    )
                )
        else:
            fields = {
                "test_statistic": visible.get("test_statistic"),
                "degrees_of_freedom": visible.get("degrees_of_freedom"),
                "p_value": visible.get("p_value"),
                "primary_estimate": visible.get("primary_estimate"),
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
            "estimand": question["estimand"],
            "declared_design": spec["design"],
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
    phase11_supplied = sensitivity_payload is not None or practical_payload is not None
    payload = {
        "schema_version": 2 if phase11_supplied else REPORT_SCHEMA_VERSION,
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
