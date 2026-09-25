"""Opt-in, local records of workflow actions actually observed by this process."""

from __future__ import annotations

import json
from collections.abc import Callable
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any

from .exceptions import InvalidDataError
from .provenance import content_reference
from .results import AnalysisResult
from .specifications import _json_value

_EVENTS = {
    "question_prepared",
    "specification_updated",
    "method_recommended",
    "analysis_executed",
    "interpretation_generated",
    "report_generated",
    "report_exported",
    "audit_performed",
    "planning_declared",
    "existing_analysis_imported",
    "sensitivity_plan_created",
    "sensitivity_scenario_attempted",
    "sensitivity_scenario_completed",
    "sensitivity_scenario_unavailable",
    "sensitivity_scenario_failed",
    "meaningful_threshold_declared",
    "analysis_plan_created",
    "analysis_plan_updated",
    "study_planning_completed",
    "plan_adherence_compared",
    "reporting_completeness_assessed",
    "session_snapshot_created",
}


class DecisionLedger:
    """Ordered events since activation, not an authenticated historical record."""

    def __init__(
        self, *, clock: Callable[[], datetime | str] | None = None, history_status: str = "recorded"
    ) -> None:
        if history_status not in {"recorded", "unavailable"}:
            raise InvalidDataError("history_status must be recorded or unavailable.")
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._history_status = history_status
        self._events: list[dict[str, Any]] = []

    @property
    def history_status(self) -> str:
        return self._history_status

    @property
    def events(self) -> tuple[dict[str, Any], ...]:
        return tuple(deepcopy(self._events))

    def _record(
        self,
        event_type: str,
        *,
        source: str = "software",
        previous_state: Any = None,
        new_state: Any = None,
        reason: str | None = None,
        references: dict[str, str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if event_type not in _EVENTS or source not in {"software", "researcher"}:
            raise InvalidDataError("Unknown ledger event type or source.")
        if reason is not None and (not isinstance(reason, str) or not reason.strip()):
            raise InvalidDataError("A supplied decision reason must be non-empty text.")
        moment = self._clock()
        timestamp = moment.isoformat() if isinstance(moment, datetime) else moment
        if not isinstance(timestamp, str) or not timestamp.strip():
            raise InvalidDataError("The ledger clock must return a datetime or timestamp text.")
        number = len(self._events) + 1
        event = _json_value(
            {
                "event_id": f"event-{number:06d}",
                "sequence": number,
                "event_type": event_type,
                "timestamp": timestamp,
                "timestamp_kind": "local_software_event",
                "source": source,
                "previous_state": previous_state,
                "new_state": new_state,
                "reason": reason,
                "references": references or {},
                "metadata": metadata or {},
            }
        )
        self._events.append(event)

    @classmethod
    def import_result(
        cls, result: AnalysisResult, *, clock: Callable[[], datetime | str] | None = None
    ) -> DecisionLedger:
        """Record only the present import; earlier decision history stays unavailable."""
        if not isinstance(result, AnalysisResult):
            raise InvalidDataError("import_result requires an AnalysisResult.")
        ledger = cls(clock=clock, history_status="unavailable")
        ledger._record(
            "existing_analysis_imported",
            references={"analysis": content_reference("analysis", result.to_dict())},
            metadata={"note": "Existing analysis imported; earlier decisions were not recorded."},
        )
        return ledger

    def to_dict(self) -> dict[str, Any]:
        return deepcopy(
            {"schema_version": 1, "history_status": self._history_status, "events": self._events}
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2, allow_nan=False)
