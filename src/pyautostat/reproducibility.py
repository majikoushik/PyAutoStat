"""Explicit, data-supplied replay of existing analyses."""

from __future__ import annotations

import json
import math
import platform
import re
import zipfile
from copy import deepcopy
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Literal

import numpy as np
import pandas as pd
import scipy

from .exceptions import InvalidDataError, PyAutoStatError
from .practical_significance import (
    MeaningfulEffectThreshold,
    PracticalSignificanceResult,
    assess_practical_significance,
)
from .provenance import content_reference, dataset_fingerprint
from .results import AnalysisResult, RecommendationStatus
from .sensitivity import SensitivityResult, SensitivitySpecification
from .specifications import AnalysisSpecification, _json_value

_REL_TOL = 1e-10
_ABS_TOL = 1e-12
_FOLLOW_UP_METADATA_KEY = "phase11"  # Retained for schema-version-2 wire compatibility.
_VALUES = (
    "test_statistic",
    "degrees_of_freedom",
    "p_value",
    "primary_estimate",
    "estimate_name",
    "estimate_unit",
    "effect_size",
    "confidence_interval",
)


def _version(distribution: str) -> str | None:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return None


def runtime_environment() -> dict[str, Any]:
    """Record only relevant, observed versions; no paths or environment variables."""
    return {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "pyautostat": _version("pyautostat"),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "system": platform.system(),
        "architecture": platform.machine(),
    }


def _projection(result: AnalysisResult) -> dict[str, Any]:
    payload = result.to_dict()
    values = payload["values"]
    projection = {
        "method_id": payload["method_id"],
        "status": payload["status"],
        "sample_size": payload["sample_size"],
        "excluded_rows": payload["excluded_rows"],
        "specification": payload["specification"],
        "recommended_method": (payload["recommendation"] or {}).get("method_id"),
        "values": {key: values.get(key) for key in _VALUES},
        "sample": payload["metadata"].get("sample"),
        "group_order": payload["metadata"].get("group_order"),
        "contrast": payload["metadata"].get("contrast"),
        "warnings": payload["warnings"],
    }
    if payload["method_id"] == "dataset_profile" and "profile" in values:
        projection["profile_reference"] = content_reference("profile", values["profile"])
    return _json_value(projection)


def _differences(expected: Any, actual: Any, path: str, output: list[str]) -> None:
    if isinstance(expected, dict) and isinstance(actual, dict):
        for key in sorted(expected.keys() | actual.keys()):
            if key not in expected or key not in actual:
                output.append(f"{path}.{key}")
            else:
                _differences(expected[key], actual[key], f"{path}.{key}", output)
        return
    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            output.append(f"{path}.length")
        for index, (left, right) in enumerate(zip(expected, actual, strict=False)):
            _differences(left, right, f"{path}[{index}]", output)
        return
    if type(expected) is not type(actual):
        output.append(path)
    elif isinstance(expected, float):
        if not math.isclose(expected, actual, rel_tol=_REL_TOL, abs_tol=_ABS_TOL):
            output.append(path)
    elif expected != actual:
        output.append(path)


@dataclass(frozen=True)
class ReproductionOutcome:
    status: str
    data_status: str
    differing_fields: tuple[str, ...]
    warnings: tuple[str, ...]
    replay_reference: str | None

    def to_dict(self) -> dict[str, Any]:
        return _json_value({"schema_version": 1, **vars(self)})


class ReproducibilityRecord:
    """Serializable configuration and expected summary; no raw DataFrame."""

    def __init__(self, payload: dict[str, Any]) -> None:
        if not isinstance(payload, dict):
            raise InvalidDataError("Reproducibility record must be a mapping.")
        self._payload = _json_value(payload)
        if (
            type(self._payload.get("schema_version")) is not int
            or self._payload["schema_version"] not in (1, 2)
            or not all(
                key in self._payload
                for key in ("specification", "method_id", "expected", "environment", "stochastic")
            )
        ):
            raise InvalidDataError("Reproducibility record has missing or unsupported fields.")
        if self._payload["schema_version"] == 2 and not isinstance(
            self._payload.get(_FOLLOW_UP_METADATA_KEY), dict
        ):
            raise InvalidDataError(
                "Schema version 2 requires follow-up analysis configuration metadata."
            )
        if self._payload["schema_version"] == 2:
            follow_up = self._payload[_FOLLOW_UP_METADATA_KEY]
            if follow_up.get("automatic_replay") is not False:
                raise InvalidDataError(
                    "Follow-up reproducibility metadata must disable automatic replay."
                )
            sensitivity = follow_up.get("sensitivity")
            practical = follow_up.get("practical_significance")
            if sensitivity is None and practical is None:
                raise InvalidDataError("Follow-up metadata must contain at least one component.")
            if sensitivity is not None:
                if not isinstance(sensitivity, dict) or not isinstance(
                    sensitivity.get("configuration"), dict
                ):
                    raise InvalidDataError("Sensitivity configuration metadata is invalid.")
                configuration = sensitivity["configuration"]
                scenarios = configuration.get("scenarios")
                order = configuration.get("scenario_order")
                if not isinstance(scenarios, list) or not isinstance(order, list):
                    raise InvalidDataError(
                        "Sensitivity scenario order or specifications are invalid."
                    )
                restored = [SensitivitySpecification.from_dict(item) for item in scenarios]
                if [item.name for item in restored] != order:
                    raise InvalidDataError("Sensitivity scenario order does not match its records.")
            if practical is not None:
                if not isinstance(practical, dict) or not isinstance(
                    practical.get("threshold"), dict
                ):
                    raise InvalidDataError(
                        "Practical-significance threshold configuration is invalid."
                    )
                MeaningfulEffectThreshold.from_dict(practical["threshold"])
        if (
            not isinstance(self._payload["method_id"], str)
            or not isinstance(self._payload["expected"], dict)
            or not isinstance(self._payload["environment"], dict)
            or not isinstance(self._payload["stochastic"], dict)
        ):
            raise InvalidDataError("Reproducibility record fields have invalid types.")
        AnalysisSpecification.from_dict(self._payload["specification"])
        if self._payload["expected"].get("method_id") != self._payload["method_id"]:
            raise InvalidDataError("Recorded method disagrees with expected result.")
        fingerprint_value = self._payload.get("dataset_fingerprint")
        if fingerprint_value is not None and (
            not isinstance(fingerprint_value, dict)
            or fingerprint_value.get("algorithm") != "pyautostat-dataframe-sha256-v1"
            or not isinstance(fingerprint_value.get("digest"), str)
            or len(fingerprint_value["digest"]) != 64
        ):
            raise InvalidDataError("Dataset fingerprint has an unsupported format.")

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ReproducibilityRecord:
        return cls(payload)

    @classmethod
    def from_result(
        cls,
        result: AnalysisResult,
        *,
        data: pd.DataFrame | None = None,
        fingerprint: bool = True,
        planning_status: str = "unknown",
        sensitivity: SensitivityResult | None = None,
        practical_significance: PracticalSignificanceResult | None = None,
    ) -> ReproducibilityRecord:
        if not isinstance(result, AnalysisResult) or result.specification is None:
            raise InvalidDataError("A result with its specification is required.")
        if planning_status not in {"unknown", "planned", "exploratory"}:
            raise InvalidDataError("planning_status must be unknown, planned, or exploratory.")
        if sensitivity is not None and not isinstance(sensitivity, SensitivityResult):
            raise InvalidDataError("sensitivity must be a SensitivityResult when supplied.")
        if sensitivity is not None and sensitivity.base_result.to_dict() != result.to_dict():
            raise InvalidDataError("Sensitivity metadata does not reference this base result.")
        if practical_significance is not None and not isinstance(
            practical_significance, PracticalSignificanceResult
        ):
            raise InvalidDataError(
                "practical_significance must be a PracticalSignificanceResult when supplied."
            )
        if practical_significance is not None:
            expected_practical = assess_practical_significance(
                result,
                practical_significance.threshold,
                planning_status=practical_significance.threshold.planning_status,
            )
            if expected_practical.to_dict() != practical_significance.to_dict():
                raise InvalidDataError(
                    "Practical-significance metadata does not match this base result."
                )
        warnings = list(result.warnings)
        fingerprint_value = None
        if fingerprint and data is not None:
            try:
                fingerprint_value = dataset_fingerprint(data)
            except InvalidDataError as exc:
                warnings.append(f"Dataset fingerprint unavailable: {exc}")
        elif fingerprint:
            warnings.append("Dataset fingerprint unavailable because no DataFrame was supplied.")
        spec = result.specification.to_dict()
        metadata = result.metadata
        interval = result.values.get("effect_size") or {}
        interval = interval.get("confidence_interval") if isinstance(interval, dict) else None
        stochastic = {
            "researcher_seed": result.specification.options.random_seed,
            "effective_seed": metadata.get("effective_random_seed"),
            "bootstrap_resamples": metadata.get("bootstrap_default_resamples"),
            "bootstrap_method": interval.get("method") if isinstance(interval, dict) else None,
            "confidence_level": result.specification.options.confidence_level,
        }
        follow_up = None
        if sensitivity is not None or practical_significance is not None:
            follow_up = {
                "sensitivity": (
                    {
                        "configuration": sensitivity.reproducibility,
                        "result_reference": content_reference("sensitivity", sensitivity.to_dict()),
                    }
                    if sensitivity is not None
                    else None
                ),
                "practical_significance": (
                    {
                        "threshold": practical_significance.threshold.to_dict(),
                        "result_reference": content_reference(
                            "practical_significance", practical_significance.to_dict()
                        ),
                    }
                    if practical_significance is not None
                    else None
                ),
                "automatic_replay": False,
            }
        payload = {
            "schema_version": 2 if follow_up is not None else 1,
            "specification": spec,
            "method_id": result.method_id,
            "expected": _projection(result),
            "analysis_reference": content_reference("analysis", result.to_dict()),
            "specification_reference": content_reference("specification", spec),
            "environment": runtime_environment(),
            "stochastic": stochastic,
            "dataset_fingerprint": fingerprint_value,
            "planning_status": planning_status,
            "warnings": warnings,
            "limitations": [
                "Source data must be supplied separately for replay.",
                "Software records do not verify data authenticity or preregistration.",
            ],
        }
        if follow_up is not None:
            payload[_FOLLOW_UP_METADATA_KEY] = follow_up
        return cls(payload)

    @property
    def method_id(self) -> str:
        return str(self._payload["method_id"])

    def to_dict(self) -> dict[str, Any]:
        return deepcopy(self._payload)

    def to_json(self) -> str:
        return json.dumps(self._payload, ensure_ascii=False, indent=2, allow_nan=False)

    def save_package(
        self, path: str | Path, *, overwrite: bool = False, data_reference: str | None = None
    ) -> Path:
        """Write a metadata-only ZIP to an explicit destination."""
        destination = Path(path)
        if destination.exists() and not overwrite:
            raise InvalidDataError(
                "Reproduction package exists; pass overwrite=True to replace it."
            )
        if data_reference is not None and (
            not isinstance(data_reference, str)
            or not data_reference.strip()
            or PurePosixPath(data_reference).is_absolute()
            or PureWindowsPath(data_reference).is_absolute()
            or bool(PureWindowsPath(data_reference).drive)
            or ".." in PurePosixPath(data_reference).parts
            or ".." in PureWindowsPath(data_reference).parts
            or any(ord(character) < 32 for character in data_reference)
            or re.fullmatch(r"[\w./ -]+", data_reference) is None
        ):
            raise InvalidDataError("data_reference must be non-empty relative text without '..'.")
        readme = (
            "# PyAutoStat reproduction metadata\n\n"
            "This package has no raw dataset. Supply the original data separately and call "
            "pyautostat.reproduce(record, data=frame). A matching fingerprint is a consistency "
            "check, not proof of provenance. No included file is executed automatically.\n"
        )
        if data_reference is not None:
            readme += f"\nResearcher-supplied relative data reference: {data_reference}\n"
        files = {
            "README.md": readme,
            "analysis_specification.json": json.dumps(
                self._payload["specification"], ensure_ascii=False, indent=2, allow_nan=False
            ),
            "analysis_reference.json": json.dumps(
                {
                    "analysis_reference": self._payload["analysis_reference"],
                    "expected": self._payload["expected"],
                },
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            ),
            "reproducibility_record.json": self.to_json(),
        }
        try:
            mode: Literal["w", "x"] = "w" if overwrite else "x"
            with zipfile.ZipFile(destination, mode, compression=zipfile.ZIP_DEFLATED) as archive:
                for name, content in files.items():
                    entry = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                    entry.compress_type = zipfile.ZIP_DEFLATED
                    archive.writestr(entry, content.encode("utf-8"))
        except (OSError, ValueError, TypeError) as exc:
            raise InvalidDataError(f"Could not write reproduction package: {exc}") from exc
        return destination


def reproduce(
    record: ReproducibilityRecord,
    *,
    data: pd.DataFrame,
    allow_changed_data: bool = False,
) -> ReproductionOutcome:
    """Explicitly replay after checking data and the selected method."""
    if not isinstance(record, ReproducibilityRecord):
        raise InvalidDataError("reproduce requires a ReproducibilityRecord.")
    if not isinstance(data, pd.DataFrame):
        raise InvalidDataError("reproduce requires an explicitly supplied DataFrame.")
    payload = record.to_dict()
    warnings: list[str] = []
    if payload.get(_FOLLOW_UP_METADATA_KEY) is not None:
        warnings.append(
            "Follow-up configurations are recorded but sensitivity scenarios are not "
            "automatically replayed."
        )
    expected_fingerprint = payload.get("dataset_fingerprint")
    if expected_fingerprint is None:
        data_status = "fingerprint_unavailable"
        warnings.append("Dataset identity cannot be checked without a recorded fingerprint.")
    else:
        try:
            observed = dataset_fingerprint(data)
        except InvalidDataError as exc:
            return ReproductionOutcome(
                "unavailable", "fingerprint_unavailable", (), (str(exc),), None
            )
        data_status = "same_data" if observed == expected_fingerprint else "changed_data"
        if data_status == "changed_data" and not allow_changed_data:
            return ReproductionOutcome(
                "mismatch",
                data_status,
                ("dataset_fingerprint",),
                ("Supplied data differ from the recorded fingerprint; no analysis was run.",),
                None,
            )
    current_environment = runtime_environment()
    for name, version in payload["environment"].items():
        if current_environment.get(name) != version:
            warnings.append(f"Runtime environment differs for {name}.")
    specification = AnalysisSpecification.from_dict(payload["specification"])
    from .research_assistant import ResearchAssistant

    try:
        assistant = ResearchAssistant(data)
        recommendation = assistant.recommend_test(specification=specification)
        if (
            recommendation.status is not RecommendationStatus.READY
            or recommendation.method_id != record.method_id
        ):
            return ReproductionOutcome(
                "unavailable",
                data_status,
                ("method_id",),
                tuple(
                    [
                        *warnings,
                        "The recorded method is no longer selected and was not substituted.",
                    ]
                ),
                None,
            )
        replay = assistant.analyze(specification=specification)
    except (PyAutoStatError, ValueError) as exc:
        return ReproductionOutcome(
            "unavailable", data_status, (), tuple([*warnings, f"Replay could not run: {exc}"]), None
        )
    observed_projection = _projection(replay)
    differences: list[str] = []
    _differences(payload["expected"], observed_projection, "result", differences)
    alpha = specification.options.alpha
    old_p = payload["expected"]["values"].get("p_value")
    new_p = observed_projection["values"].get("p_value")
    if isinstance(old_p, (int, float)) and isinstance(new_p, (int, float)):
        if (old_p < alpha) != (new_p < alpha):
            differences.append("result.values.p_value_decision")
    if data_status == "changed_data":
        warnings.append("Changed-data rerun is not a same-data reproduction.")
    if data_status == "fingerprint_unavailable":
        warnings.append("Numerical agreement cannot establish same-data reproduction.")
    status = (
        "mismatch"
        if differences or data_status == "changed_data"
        else "unavailable"
        if data_status == "fingerprint_unavailable"
        else "reproduced"
    )
    return ReproductionOutcome(
        status,
        data_status,
        tuple(dict.fromkeys(differences)),
        tuple(warnings),
        content_reference("analysis", replay.to_dict()),
    )
