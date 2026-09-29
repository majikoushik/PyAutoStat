"""Typed, data-independent research configuration contracts."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .exceptions import InvalidDataError

SCHEMA_VERSION = 1
QUESTION_SCHEMA_VERSION = 2
PAIRED_SCHEMA_VERSION = 3


class Objective(str, Enum):
    DESCRIPTIVE = "descriptive"
    COMPARE_GROUPS = "compare_groups"
    COMPARE_REFERENCE = "compare_reference"
    ASSOCIATION = "association"
    REGRESSION = "regression"
    RELIABILITY = "reliability"


class StudyDesign(str, Enum):
    UNKNOWN = "unknown"
    INDEPENDENT = "independent"
    PAIRED = "paired"
    REPEATED = "repeated"
    CLUSTERED = "clustered"


def _enum(value: Any, kind: type[Enum], field_name: str) -> Any:
    try:
        return kind(value)
    except (ValueError, TypeError) as exc:
        choices = ", ".join(item.value for item in kind)
        raise InvalidDataError(
            f"{field_name} must be one of: {choices}. Choose a supported value."
        ) from exc


def _name(value: Any, field_name: str) -> None:
    if value is not None and (not isinstance(value, str) or not value.strip()):
        raise InvalidDataError(f"{field_name} must be a non-empty string when provided.")


def _probability(value: Any, field_name: str) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 0 < value < 1
    ):
        raise InvalidDataError(f"{field_name} must be a finite number strictly between 0 and 1.")


def _json_value(value: Any, field_name: str = "value") -> Any:
    """Copy a plain JSON value, rejecting unsupported and non-finite values."""
    if isinstance(value, Enum):
        return value.value
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise InvalidDataError(f"{field_name} must be finite or None for JSON serialization.")
        return value
    if isinstance(value, (list, tuple)):
        return [_json_value(item, field_name) for item in value]
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise InvalidDataError(f"{field_name} must have string keys for JSON serialization.")
        return {key: _json_value(item, f"{field_name}.{key}") for key, item in value.items()}
    raise InvalidDataError(f"{field_name} must contain only JSON-compatible values.")


@dataclass(frozen=True)
class ResearchQuestion:
    """Question and variable roles; absent fields remain explicitly unresolved."""

    objective: Objective | None = None
    outcome: str | None = None
    predictor: str | None = None
    estimand: str | None = None
    description: str | None = None
    reference_value: float | None = None
    predictors: tuple[str, ...] | None = None
    items: tuple[str, ...] | None = None
    controls: tuple[str, ...] | None = None
    event_level: Any | None = None
    association_measure: str | None = None

    def __post_init__(self) -> None:
        if self.objective is not None:
            object.__setattr__(self, "objective", _enum(self.objective, Objective, "objective"))
        for name in ("outcome", "predictor", "estimand", "description", "association_measure"):
            _name(getattr(self, name), name)
        if self.predictors is not None:
            if isinstance(self.predictors, (str, bytes)) or not isinstance(
                self.predictors, (list, tuple)
            ):
                raise InvalidDataError("predictors must be a nonempty sequence of column names.")
            checked = tuple(self.predictors)
            if not checked or any(
                not isinstance(item, str) or not item.strip() for item in checked
            ):
                raise InvalidDataError("predictors must contain nonempty column names.")
            if len(set(checked)) != len(checked):
                raise InvalidDataError("predictors must not contain duplicates.")
            object.__setattr__(self, "predictors", checked)
        if self.items is not None:
            if isinstance(self.items, (str, bytes)) or not isinstance(self.items, (list, tuple)):
                raise InvalidDataError("items must be a sequence of at least two column names.")
            checked_items = tuple(self.items)
            if len(checked_items) < 2 or any(
                not isinstance(item, str) or not item.strip() for item in checked_items
            ):
                raise InvalidDataError("items must contain at least two nonempty column names.")
            if len(set(checked_items)) != len(checked_items):
                raise InvalidDataError("items must not contain duplicates.")
            object.__setattr__(self, "items", checked_items)
        if self.controls is not None:
            if isinstance(self.controls, (str, bytes)) or not isinstance(
                self.controls, (list, tuple)
            ):
                raise InvalidDataError("controls must be a nonempty sequence of column names.")
            checked_controls = tuple(self.controls)
            if not checked_controls or any(
                not isinstance(item, str) or not item.strip() for item in checked_controls
            ):
                raise InvalidDataError("controls must contain nonempty column names.")
            if len(set(checked_controls)) != len(checked_controls):
                raise InvalidDataError("controls must not contain duplicates.")
            object.__setattr__(self, "controls", checked_controls)
        if self.event_level is not None:
            checked_event = _json_value(self.event_level, "event_level")
            if isinstance(checked_event, (list, dict)):
                raise InvalidDataError("event_level must be a non-missing JSON scalar value.")
            object.__setattr__(self, "event_level", checked_event)
        if self.predictor is not None and self.predictors is not None:
            if self.predictors != (self.predictor,):
                raise InvalidDataError(
                    "predictor and predictors cannot specify conflicting values."
                )
        if self.reference_value is not None:
            value = self.reference_value
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise InvalidDataError("reference_value must be a finite numeric value.")
            object.__setattr__(self, "reference_value", float(value))
        if self.controls is not None:
            if self.outcome in self.controls or self.predictor in self.controls:
                raise InvalidDataError(
                    "controls must differ from both the outcome and predictor variables."
                )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "objective": self.objective.value if self.objective is not None else None,
            "outcome": self.outcome,
            "predictor": self.predictor,
            "estimand": self.estimand,
            "description": self.description,
            "reference_value": self.reference_value,
        }
        if self.predictors is not None:
            payload["predictors"] = list(self.predictors)
        if self.items is not None:
            payload["items"] = list(self.items)
        if self.controls is not None:
            payload["controls"] = list(self.controls)
        if self.event_level is not None:
            payload["event_level"] = _json_value(self.event_level)
        if self.association_measure is not None:
            payload["association_measure"] = self.association_measure
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ResearchQuestion:
        return cls(**data)


@dataclass(frozen=True)
class AnalysisOptions:
    """Cross-cutting optional settings; defaults are disclosed on serialization."""

    alpha: float = 0.05
    confidence_level: float = 0.95
    random_seed: int | None = None
    covariance_type: str = "classical"
    reference_levels: dict[str, Any] | None = None
    bootstrap_samples: int = 499
    reverse_scoring: dict[str, tuple[float, float]] | None = None

    def __post_init__(self) -> None:
        _probability(self.alpha, "alpha")
        _probability(self.confidence_level, "confidence_level")
        if self.random_seed is not None and (
            isinstance(self.random_seed, bool) or not isinstance(self.random_seed, int)
        ):
            raise InvalidDataError("random_seed must be an integer or None.")
        if self.covariance_type not in {"classical", "HC3"}:
            raise InvalidDataError("covariance_type must be 'classical' or 'HC3'.")
        if self.reference_levels is not None:
            if not isinstance(self.reference_levels, dict) or any(
                not isinstance(key, str) or not key.strip() for key in self.reference_levels
            ):
                raise InvalidDataError("reference_levels must map predictor names to levels.")
            checked = _json_value(self.reference_levels, "reference_levels")
            if any(value is None or isinstance(value, (list, dict)) for value in checked.values()):
                raise InvalidDataError("Each categorical reference must be a non-missing scalar.")
            object.__setattr__(self, "reference_levels", checked)
        if isinstance(self.bootstrap_samples, bool) or not isinstance(self.bootstrap_samples, int):
            raise InvalidDataError("bootstrap_samples must be a positive integer.")
        if self.bootstrap_samples < 1:
            raise InvalidDataError("bootstrap_samples must be a positive integer.")
        if self.reverse_scoring is not None:
            if not isinstance(self.reverse_scoring, dict):
                raise InvalidDataError("reverse_scoring must map item names to (lower, upper).")
            checked_scoring: dict[str, tuple[float, float]] = {}
            for item, bounds in self.reverse_scoring.items():
                if not isinstance(item, str) or not item.strip():
                    raise InvalidDataError("reverse_scoring item names must be nonempty strings.")
                if (
                    not isinstance(bounds, (list, tuple))
                    or len(bounds) != 2
                    or any(
                        isinstance(value, bool)
                        or not isinstance(value, (int, float))
                        or not math.isfinite(value)
                        for value in bounds
                    )
                ):
                    raise InvalidDataError(
                        "Every reverse_scoring value must be two finite numeric bounds."
                    )
                lower, upper = float(bounds[0]), float(bounds[1])
                if lower >= upper:
                    raise InvalidDataError(
                        "Reverse-scoring lower bounds must be below upper bounds."
                    )
                checked_scoring[item] = (lower, upper)
            object.__setattr__(self, "reverse_scoring", checked_scoring)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "alpha": self.alpha,
            "confidence_level": self.confidence_level,
            "random_seed": self.random_seed,
        }
        if self.covariance_type != "classical" or self.reference_levels is not None:
            payload["covariance_type"] = self.covariance_type
            payload["reference_levels"] = _json_value(self.reference_levels)
        if self.bootstrap_samples != 499 or self.reverse_scoring is not None:
            payload["bootstrap_samples"] = self.bootstrap_samples
            payload["reverse_scoring"] = _json_value(self.reverse_scoring)
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AnalysisOptions:
        return cls(**data)


@dataclass(frozen=True)
class AnalysisSpecification:
    """Serializable specification; DataFrame-specific validation comes later."""

    question: ResearchQuestion = field(default_factory=ResearchQuestion)
    design: StudyDesign = StudyDesign.UNKNOWN
    options: AnalysisOptions = field(default_factory=AnalysisOptions)
    variable_metadata: dict[str, str] | None = None
    data_dictionary: dict[str, dict[str, Any]] | None = None
    unit_id: str | None = None
    condition_order: tuple[Any, Any] | None = None
    analytical_variable_types: dict[str, str] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.question, ResearchQuestion):
            raise InvalidDataError("question must be a ResearchQuestion instance.")
        object.__setattr__(self, "design", _enum(self.design, StudyDesign, "design"))
        if not isinstance(self.options, AnalysisOptions):
            raise InvalidDataError("options must be an AnalysisOptions instance.")
        if self.variable_metadata is not None:
            if not isinstance(self.variable_metadata, dict):
                raise InvalidDataError(
                    "variable_metadata must be a mapping of column names to text."
                )
            for key, value in self.variable_metadata.items():
                _name(key, "variable_metadata column")
                _name(value, f"variable_metadata[{key!r}]")
                if value is None:
                    raise InvalidDataError("variable_metadata values must be non-empty strings.")
            object.__setattr__(self, "variable_metadata", self.variable_metadata.copy())
        if self.data_dictionary is not None:
            if not isinstance(self.data_dictionary, dict) or any(
                not isinstance(key, str) or not key.strip() or not isinstance(value, dict)
                for key, value in self.data_dictionary.items()
            ):
                raise InvalidDataError(
                    "data_dictionary must map column names to metadata mappings."
                )
            object.__setattr__(self, "data_dictionary", _json_value(self.data_dictionary))
        _name(self.unit_id, "unit_id")
        if self.condition_order is not None:
            if (
                not isinstance(self.condition_order, (list, tuple))
                or len(self.condition_order) != 2
            ):
                raise InvalidDataError("condition_order must contain exactly two condition labels.")
            checked = tuple(_json_value(item, "condition_order") for item in self.condition_order)
            if any(item is None or isinstance(item, (list, dict)) for item in checked):
                raise InvalidDataError(
                    "condition_order labels must be non-missing JSON scalar values."
                )
            if checked[0] == checked[1]:
                raise InvalidDataError("condition_order labels must be distinct.")
            object.__setattr__(self, "condition_order", checked)
        if self.condition_order is not None and self.unit_id is None:
            raise InvalidDataError("condition_order requires an explicit unit_id.")
        if self.unit_id is not None and self.design is not StudyDesign.PAIRED:
            raise InvalidDataError("unit_id is supported only for design='paired'.")
        if self.analytical_variable_types is not None:
            if not isinstance(self.analytical_variable_types, dict) or any(
                not isinstance(key, str)
                or not key.strip()
                or not isinstance(value, str)
                or not value.strip()
                for key, value in self.analytical_variable_types.items()
            ):
                raise InvalidDataError(
                    "analytical_variable_types must map column names to non-empty type names."
                )
            object.__setattr__(
                self, "analytical_variable_types", self.analytical_variable_types.copy()
            )

    def to_dict(self) -> dict[str, Any]:
        schema_version = (
            PAIRED_SCHEMA_VERSION
            if self.unit_id is not None or self.condition_order is not None
            else QUESTION_SCHEMA_VERSION
            if self.data_dictionary is not None
            else SCHEMA_VERSION
        )
        payload = {
            "schema_version": schema_version,
            "question": self.question.to_dict(),
            "design": self.design.value,
            "options": self.options.to_dict(),
            "variable_metadata": _json_value(self.variable_metadata),
        }
        if self.question.objective is Objective.REGRESSION:
            payload["options"]["covariance_type"] = self.options.covariance_type
            payload["options"]["reference_levels"] = _json_value(self.options.reference_levels)
            payload["intercept"] = True
            payload["missing_data_policy"] = "complete cases across outcome and all predictors"
            payload["analytical_variable_types"] = _json_value(self.analytical_variable_types)
        if self.question.objective is Objective.RELIABILITY:
            payload["options"]["bootstrap_samples"] = self.options.bootstrap_samples
            payload["options"]["reverse_scoring"] = _json_value(self.options.reverse_scoring)
            payload["missing_data_policy"] = "complete cases across all selected items"
            payload["analytical_variable_types"] = _json_value(self.analytical_variable_types)
        if self.data_dictionary is not None:
            payload["data_dictionary"] = _json_value(self.data_dictionary)
        if schema_version == PAIRED_SCHEMA_VERSION:
            payload["unit_id"] = self.unit_id
            payload["condition_order"] = _json_value(self.condition_order)
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AnalysisSpecification:
        if (
            not isinstance(data, dict)
            or type(data.get("schema_version")) is not int
            or data["schema_version"]
            not in (SCHEMA_VERSION, QUESTION_SCHEMA_VERSION, PAIRED_SCHEMA_VERSION)
        ):
            raise InvalidDataError(
                f"schema_version must be {SCHEMA_VERSION} or "
                f"{QUESTION_SCHEMA_VERSION}, or {PAIRED_SCHEMA_VERSION} for this specification."
            )
        if data["schema_version"] == SCHEMA_VERSION and "data_dictionary" in data:
            raise InvalidDataError("data_dictionary requires schema_version 2.")
        if data["schema_version"] in (SCHEMA_VERSION, QUESTION_SCHEMA_VERSION) and any(
            key in data for key in ("unit_id", "condition_order")
        ):
            raise InvalidDataError("unit_id and condition_order require schema_version 3.")
        if data["schema_version"] == QUESTION_SCHEMA_VERSION and not isinstance(
            data.get("data_dictionary"), dict
        ):
            raise InvalidDataError("schema_version 2 requires a data_dictionary mapping.")
        if data["schema_version"] == PAIRED_SCHEMA_VERSION and (
            not isinstance(data.get("unit_id"), str) or not data["unit_id"].strip()
        ):
            raise InvalidDataError("schema_version 3 requires an explicit unit_id.")
        try:
            return cls(
                question=ResearchQuestion.from_dict(data["question"]),
                design=data["design"],
                options=AnalysisOptions.from_dict(data["options"]),
                variable_metadata=data.get("variable_metadata"),
                data_dictionary=data.get("data_dictionary"),
                unit_id=data.get("unit_id"),
                condition_order=(
                    tuple(data["condition_order"])
                    if data.get("condition_order") is not None
                    else None
                ),
                analytical_variable_types=data.get("analytical_variable_types"),
            )
        except (KeyError, TypeError) as exc:
            raise InvalidDataError(
                "AnalysisSpecification has missing or invalid question, design, or options fields."
            ) from exc
