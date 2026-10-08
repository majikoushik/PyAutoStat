"""Serializable research records; they do not execute statistical analyses."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .exceptions import InvalidDataError
from .specifications import SCHEMA_VERSION, AnalysisSpecification, _json_value


def _method_label(method_id: str | None, fallback: str | None = None) -> str:
    """Resolve a display name from the authoritative capability registry."""
    if fallback:
        return fallback
    if not method_id:
        return "Not available"
    # Imported lazily because recommendation records import this module.
    from .recommendation import METHOD_CAPABILITIES

    capability = METHOD_CAPABILITIES.get(method_id)
    return capability.name if capability is not None else method_id


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

    def __str__(self) -> str:
        """Return a human-readable description of the missing field.

        Example::

            str(item)
            # "Field 'design' requires your input: Study design is required.
            #   Next step: revised = assistant.update_question(workflow.draft, design=<your value>)
            #              workflow = assistant.run(draft=revised)"
        """
        return (
            f"Field '{self.field}' requires your input: {self.message}\n"
            f"   Next step: revised = assistant.update_question(\n"
            f"       workflow.draft, {self.field}=<your value>\n"
            f"   )\n"
            f"   workflow = assistant.run(draft=revised)"
        )

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
    method_availability: str | None = None
    decision_trace: tuple[dict[str, Any], ...] = ()
    alternatives: tuple[dict[str, Any], ...] = ()
    context: dict[str, Any] = field(default_factory=dict)
    questions: tuple[dict[str, Any], ...] = ()

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
        if self.method_availability not in (None, "runnable", "coefficient_only", "unavailable"):
            raise InvalidDataError(
                "method_availability must be runnable, coefficient_only, unavailable, or None."
            )
        if self.status is RecommendationStatus.READY and self.method_availability not in (
            None,
            "runnable",
        ):
            raise InvalidDataError("A ready recommendation cannot describe an unavailable method.")

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
                "method_availability": self.method_availability,
                "decision_trace": self.decision_trace,
                "alternatives": self.alternatives,
                "context": self.context,
                "questions": self.questions,
            }
        )

    @property
    def method_label(self) -> str:
        """Human-readable name for the recommended method.

        Returns the full method name (e.g. 'Welch independent-samples t-test')
        instead of the internal identifier ('welch_t'). Falls back to
        ``method_name`` if set, then to ``method_id``, then to 'Not available'.

        Example::

            print(recommendation.method_label)
            # 'Welch independent-samples t-test'
        """
        return _method_label(self.method_id, self.method_name)

    def explain(self, *, diagnostics: dict[str, Any] | None = None) -> str:
        """Return a deterministic researcher-readable recommendation explanation.

        ``diagnostics`` is optional because recommendation normally precedes
        execution. Only explicitly supplied recorded diagnostics are narrated;
        no test is run by this presentation method.
        """
        from .narrate import recommendation_rationale

        return recommendation_rationale(self, diagnostics=diagnostics)

    @property
    def rationale_text(self) -> str:
        """Readable rationale derived without changing the serialized record."""
        return self.explain()


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
    specification: AnalysisSpecification | None = None
    recommendation: Recommendation | None = None

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
        if self.specification is not None and not isinstance(
            self.specification, AnalysisSpecification
        ):
            raise InvalidDataError("specification must be an AnalysisSpecification or None.")
        if self.recommendation is not None and not isinstance(self.recommendation, Recommendation):
            raise InvalidDataError("recommendation must be a Recommendation or None.")

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
                "specification": self.specification.to_dict()
                if self.specification is not None
                else None,
                "recommendation": self.recommendation.to_dict()
                if self.recommendation is not None
                else None,
            }
        )

    @property
    def method_label(self) -> str:
        """Human-readable name for the executed method.

        Returns the full method name (e.g. 'Welch independent-samples t-test')
        instead of the internal identifier ('welch_t').

        Example::

            print(result.method_label)
            # 'Welch independent-samples t-test'
        """
        fallback = None
        if self.recommendation is not None and self.recommendation.method_id == self.method_id:
            fallback = self.recommendation.method_name
        if fallback is None:
            metadata_name = self.metadata.get("method_name")
            fallback = metadata_name if isinstance(metadata_name, str) else None
        return _method_label(self.method_id, fallback)

    @property
    def test_name(self) -> str:
        """Human-readable test name for the executed statistical method."""
        return self.method_label

    @property
    def statistic(self) -> float | None:
        """Primary scalar test statistic, or None if scientifically ambiguous.

        For single-hypothesis methods (e.g. t-tests, one-way ANOVA, correlations),
        returns the scalar test statistic. For complex models with multiple inferential
        tests (e.g. two-way factorial ANOVA, linear regression, logistic regression) or
        metrics without a significance test (Cronbach's alpha), returns None.
        """
        from .result_access import get_statistic

        return get_statistic(self)

    @property
    def p_value(self) -> float | None:
        """Primary scalar p-value, or None if scientifically ambiguous.

        For single-hypothesis methods, returns the canonical inferential p-value.
        For models with multiple hypothesis tests (e.g. factorial ANOVA, multiple
        regression, logistic regression) or metrics without a significance test
        (Cronbach's alpha), returns None.
        """
        from .result_access import get_p_value

        return get_p_value(self)

    @property
    def degrees_of_freedom(self) -> int | float | tuple[Any, ...] | None:
        """Degrees of freedom for the primary test, or None if not applicable."""
        from .result_access import get_degrees_of_freedom

        return get_degrees_of_freedom(self)

    @property
    def estimate(self) -> float | None:
        """Primary scalar effect estimate, or None if ambiguous or not applicable."""
        from .result_access import get_estimate

        return get_estimate(self)

    @property
    def confidence_interval(self) -> dict[str, Any] | None:
        """Confidence interval dictionary for the primary estimate, or None."""
        from .result_access import get_confidence_interval

        return get_confidence_interval(self)

    @property
    def effect_size(self) -> dict[str, Any] | None:
        """Standardized or canonical effect size record, or None."""
        from .result_access import get_effect_size

        return get_effect_size(self)

    @property
    def effect_size_confidence_interval(self) -> dict[str, Any] | None:
        """Confidence interval dictionary for the effect size, or None."""
        from .result_access import get_effect_size_ci

        return get_effect_size_ci(self)

    @property
    def sample_accounting(self) -> dict[str, Any]:
        """Summary of analyzed, excluded, and group row counts."""
        from .result_access import get_sample_accounting

        return get_sample_accounting(self)

    def primary_result(self) -> dict[str, Any]:
        """Return a deterministic, non-serialized convenience summary mapping.

        Preserves exact stored floats without recalculation. For complex models,
        includes structured sections (model_fit, terms, coefficients, comparisons)
        rather than fabricating misleading single numbers.
        """
        from .result_access import get_primary_result

        return get_primary_result(self)

    def to_dataframe(self, section: str = "primary") -> Any:
        """Convert a section of this result to a pandas DataFrame without recalculation.

        Parameters
        ----------
        section : str, default "primary"
            Supported sections: "primary", "coefficients" (regression), "terms" (factorial ANOVA),
            "comparisons" (pairwise).
        """
        from .result_access import result_to_dataframe

        return result_to_dataframe(self, section=section)

    def apa_statement(self) -> str:
        """Generate a deterministic APA-oriented statistical statement from stored results."""
        return self.statement(style="apa")

    def statement(self, style: str = "apa") -> str:
        """Generate a deterministic statistical statement from already-stored results.

        Parameters
        ----------
        style : str, default "apa"
            Statement formatting style. Defaults to "apa" (APA-oriented).
        """
        from .result_access import generate_statement

        return generate_statement(self, style=style)
