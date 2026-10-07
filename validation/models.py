"""Validation data models and field comparison routines for PyAutoStat reference validation."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

from validation.tolerances import DEFAULT_ATOL, DEFAULT_RTOL


@dataclass
class FieldValidationResult:
    case_id: str
    method_id: str
    field: str
    observed: Any
    expected: Any
    absolute_error: float | None
    relative_error: float | None
    atol: float
    rtol: float
    evidence_level: str  # "A" | "B" | "C" | "D"
    reference: str
    status: str  # "pass" | "deferred" | "discrepancy"
    shared_primitive: str | None = None
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # Ensure no NaN or Infinity is in JSON representation
        for k in ("observed", "expected", "absolute_error", "relative_error"):
            v = d[k]
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                d[k] = None
        return d

    def to_manifest_field_dict(self) -> dict[str, Any]:
        """Convert to field record for schema_version 2 manifest."""
        if self.status == "deferred":
            return {
                "field": self.field,
                "status": "deferred",
                "evidence_level": "D",
                "reason": self.notes or "Explicitly deferred in current validation tranche",
            }

        exp = self.expected
        if isinstance(exp, float) and (math.isnan(exp) or math.isinf(exp)):
            exp = None

        entry: dict[str, Any] = {
            "field": self.field,
            "expected": exp,
            "atol": self.atol,
            "rtol": self.rtol,
            "evidence_level": self.evidence_level,
            "reference": self.reference,
            "shared_primitive": self.shared_primitive,
        }
        return entry


def compare_numeric_field(
    case_id: str,
    method_id: str,
    field: str,
    observed: Any,
    expected: Any,
    evidence_level: str,
    reference: str,
    atol: float = DEFAULT_ATOL,
    rtol: float = DEFAULT_RTOL,
    shared_primitive: str | None = None,
    notes: str = "",
) -> FieldValidationResult:
    """Compare a single numerical field against expected reference value."""
    if observed is None and expected is None:
        return FieldValidationResult(
            case_id=case_id,
            method_id=method_id,
            field=field,
            observed=None,
            expected=None,
            absolute_error=0.0,
            relative_error=0.0,
            atol=atol,
            rtol=rtol,
            evidence_level=evidence_level,
            reference=reference,
            status="pass",
            shared_primitive=shared_primitive,
            notes=notes,
        )

    if observed is None or expected is None:
        return FieldValidationResult(
            case_id=case_id,
            method_id=method_id,
            field=field,
            observed=observed,
            expected=expected,
            absolute_error=None,
            relative_error=None,
            atol=atol,
            rtol=rtol,
            evidence_level=evidence_level,
            reference=reference,
            status="discrepancy",
            shared_primitive=shared_primitive,
            notes=f"Missing value discrepancy: observed={observed}, expected={expected}",
        )

    try:
        obs_val = float(observed)
        exp_val = float(expected)
    except (ValueError, TypeError) as exc:
        return FieldValidationResult(
            case_id=case_id,
            method_id=method_id,
            field=field,
            observed=observed,
            expected=expected,
            absolute_error=None,
            relative_error=None,
            atol=atol,
            rtol=rtol,
            evidence_level=evidence_level,
            reference=reference,
            status="discrepancy",
            shared_primitive=shared_primitive,
            notes=f"Non-numeric comparison failed: {exc}",
        )

    if math.isnan(obs_val) and math.isnan(exp_val):
        return FieldValidationResult(
            case_id=case_id,
            method_id=method_id,
            field=field,
            observed=None,
            expected=None,
            absolute_error=0.0,
            relative_error=0.0,
            atol=atol,
            rtol=rtol,
            evidence_level=evidence_level,
            reference=reference,
            status="pass",
            shared_primitive=shared_primitive,
            notes=notes,
        )

    abs_err = abs(obs_val - exp_val)
    denom = max(abs(exp_val), atol)
    if denom > 0:
        rel_err = abs_err / denom
    else:
        rel_err = 0.0 if abs_err == 0.0 else float("inf")

    is_pass = abs_err <= atol or rel_err <= rtol
    status = "pass" if is_pass else "discrepancy"

    return FieldValidationResult(
        case_id=case_id,
        method_id=method_id,
        field=field,
        observed=obs_val,
        expected=exp_val,
        absolute_error=abs_err,
        relative_error=rel_err,
        atol=atol,
        rtol=rtol,
        evidence_level=evidence_level,
        reference=reference,
        status=status,
        shared_primitive=shared_primitive,
        notes=notes,
    )


def compare_deferred_field(
    case_id: str,
    method_id: str,
    field: str,
    observed: Any,
    reference: str,
    notes: str,
    shared_primitive: str | None = None,
) -> FieldValidationResult:
    """Record an explicitly deferred quantity (Level D)."""
    return FieldValidationResult(
        case_id=case_id,
        method_id=method_id,
        field=field,
        observed=observed,
        expected=None,
        absolute_error=None,
        relative_error=None,
        atol=0.0,
        rtol=0.0,
        evidence_level="D",
        reference=reference,
        status="deferred",
        shared_primitive=shared_primitive,
        notes=notes,
    )
