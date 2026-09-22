"""Public entry point for profiling and research question preparation."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, cast

import pandas as pd

from .analyzer import StatisticalAnalyzer
from .exceptions import InvalidDataError
from .profiling import complete_case_count
from .question_builder import QuestionDraft, prepare_question
from .specifications import (
    AnalysisOptions,
    AnalysisSpecification,
    Objective,
    ResearchQuestion,
    StudyDesign,
)


class ResearchAssistant:
    """Coordinate profiling and question intake without selecting an analysis.

    Construction uses StatisticalAnalyzer's validation and private DataFrame copy.
    It does not run the profiling calculations until :meth:`profile` is called.
    """

    def __init__(self, df: pd.DataFrame) -> None:
        self._analyzer = StatisticalAnalyzer(df)
        self._data_dictionary: dict | None = None

    def profile(
        self, *, data_dictionary=None, histogram_bins: int = 20, include_row_positions: bool = False
    ) -> dict:
        """Profile a DataFrame; optional declarations never mutate source values."""
        result = self._analyzer.analyze_all(
            data_dictionary=data_dictionary,
            histogram_bins=histogram_bins,
            include_row_positions=include_row_positions,
        )
        self._data_dictionary = deepcopy(result["data_dictionary"])
        return result

    def complete_case_count(self, columns: list[str]) -> dict:
        """Count rows available for a specified set of columns."""
        return complete_case_count(self._analyzer.df, columns)

    def prepare_question(
        self,
        *,
        objective: str | Objective | None = None,
        outcome: str | None = None,
        predictor: str | None = None,
        design: str | StudyDesign | None = None,
        estimand: str | None = None,
        description: str | None = None,
        options: AnalysisOptions | None = None,
        data_dictionary: dict | None = None,
        variable_types: dict[str, str] | None = None,
        specification: AnalysisSpecification | None = None,
    ) -> QuestionDraft:
        """Prepare a serializable question; return focused requests for missing facts."""
        return prepare_question(
            self._analyzer.df,
            objective=objective,
            outcome=outcome,
            predictor=predictor,
            design=design,
            estimand=estimand,
            description=description,
            options=options,
            data_dictionary=(
                data_dictionary
                if data_dictionary is not None
                else self._data_dictionary
                if specification is None
                else None
            ),
            variable_types=variable_types,
            specification=specification,
        )

    def update_question(self, draft: QuestionDraft, **changes: Any) -> QuestionDraft:
        """Reconstruct and revalidate an immutable draft after explicit answers."""
        if not isinstance(draft, QuestionDraft):
            raise InvalidDataError("draft must be a QuestionDraft.")
        allowed = {
            "objective",
            "outcome",
            "predictor",
            "design",
            "estimand",
            "description",
            "options",
            "data_dictionary",
            "variable_types",
        }
        unknown = set(changes) - allowed
        if unknown:
            raise InvalidDataError(f"Unknown question updates: {sorted(unknown)}.")
        old = draft.specification
        previous = old.question
        new_objective: Objective | None
        if "objective" in changes and changes["objective"] is not None:
            try:
                new_objective = Objective(changes["objective"])
            except (ValueError, TypeError) as exc:
                raise InvalidDataError(
                    "objective must be descriptive, compare_groups, or association."
                ) from exc
        else:
            new_objective = cast(Objective | None, changes.get("objective", previous.objective))
        switched = new_objective != previous.objective
        values: dict[str, Any] = {
            "objective": new_objective,
            "outcome": previous.outcome,
            "predictor": previous.predictor,
            "estimand": previous.estimand,
            "description": previous.description,
        }
        selected_design: StudyDesign | str = old.design
        if switched:
            values["predictor"] = None
            values["estimand"] = None
            selected_design = StudyDesign.UNKNOWN
            if new_objective == Objective.DESCRIPTIVE:
                values["outcome"] = None
        for key in ("outcome", "predictor", "estimand", "description"):
            if key in changes:
                values[key] = changes[key]
        if "design" in changes:
            selected_design = (
                StudyDesign.UNKNOWN if changes["design"] is None else changes["design"]
            )
        if new_objective == Objective.DESCRIPTIVE and any(
            key in changes for key in ("predictor", "estimand", "design")
        ):
            raise InvalidDataError(
                "Descriptive questions do not use predictor, estimand, or design."
            )
        revised = AnalysisSpecification(
            question=ResearchQuestion(**values),
            design=cast(StudyDesign, selected_design),
            options=changes.get("options", old.options),
            variable_metadata=old.variable_metadata,
            data_dictionary=changes.get("data_dictionary", old.data_dictionary),
        )
        return self.prepare_question(
            specification=revised, variable_types=changes.get("variable_types")
        )
