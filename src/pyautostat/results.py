"""Future-facing records only; these do not execute statistical analyses."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .exceptions import InvalidDataError
from .specifications import SCHEMA_VERSION, _json_value


class RecommendationStatus(str, Enum):
    READY = "ready"
    NEEDS_INPUT = "needs_input"
    UNSUPPORTED = "unsupported"


class AnalysisStatus(str, Enum):
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class MissingInformation:
    field: str
    message: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.field, str)
            or not self.field.strip()
            or not isinstance(self.message, str)
            or not self.message.strip()
        ):
            raise InvalidDataError("MissingInformation field and message must be non-empty.")

    def to_dict(self) -> dict[str, str]:
        return {"field": self.field, "message": self.message}


@dataclass(frozen=True)
class Recommendation:
    status: RecommendationStatus
    method_id: str | None = None
    method_name: str | None = None
    rationale: str | None = None
    required_assumptions: tuple[str, ...] = ()
    missing_information: tuple[MissingInformation, ...] = ()
    blockers: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        try:
            object.__setattr__(self, "status", RecommendationStatus(self.status))
        except ValueError as exc:
            raise InvalidDataError("status must be ready, needs_input, or unsupported.") from exc
        if self.status is RecommendationStatus.READY and not self.method_id:
            raise InvalidDataError("method_id is required when recommendation status is ready.")
        if self.status is RecommendationStatus.NEEDS_INPUT and not self.missing_information:
            raise InvalidDataError("missing_information is required when status is needs_input.")
        if self.status is RecommendationStatus.UNSUPPORTED and not self.blockers:
            raise InvalidDataError("blockers are required when status is unsupported.")
        if any(not isinstance(item, MissingInformation) for item in self.missing_information):
            raise InvalidDataError("missing_information must contain MissingInformation records.")

    def to_dict(self) -> dict[str, Any]:
        return _json_value(
            {
                "schema_version": SCHEMA_VERSION,
                "status": self.status.value,
                "method_id": self.method_id,
                "method_name": self.method_name,
                "rationale": self.rationale,
                "required_assumptions": self.required_assumptions,
                "missing_information": [item.to_dict() for item in self.missing_information],
                "blockers": self.blockers,
                "warnings": self.warnings,
            }
        )


@dataclass(frozen=True)
class Diagnostic:
    identifier: str
    status: str
    message: str
    values: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("identifier", "status", "message"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise InvalidDataError(f"{name} must be a non-empty string.")

    def to_dict(self) -> dict[str, Any]:
        return _json_value(
            {
                "schema_version": SCHEMA_VERSION,
                "identifier": self.identifier,
                "status": self.status,
                "message": self.message,
                "values": self.values,
            }
        )


@dataclass(frozen=True)
class AnalysisResult:
    """Common envelope; method-specific numbers live in ``values``."""

    method_id: str
    status: AnalysisStatus
    sample_size: int | None = None
    excluded_rows: int | None = None
    values: dict[str, Any] = field(default_factory=dict)
    assumptions: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.method_id, str) or not self.method_id.strip():
            raise InvalidDataError("method_id must be a non-empty string.")
        try:
            object.__setattr__(self, "status", AnalysisStatus(self.status))
        except ValueError as exc:
            raise InvalidDataError("status must be available or unavailable.") from exc
        for name in ("sample_size", "excluded_rows"):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 0
            ):
                raise InvalidDataError(f"{name} must be a nonnegative integer or None.")

    def to_dict(self) -> dict[str, Any]:
        return _json_value(
            {
                "schema_version": SCHEMA_VERSION,
                "method_id": self.method_id,
                "status": self.status.value,
                "sample_size": self.sample_size,
                "excluded_rows": self.excluded_rows,
                "values": self.values,
                "assumptions": self.assumptions,
                "warnings": self.warnings,
                "metadata": self.metadata,
            }
        )
