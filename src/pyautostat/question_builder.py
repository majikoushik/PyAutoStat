"""Dataset-aware research question intake, without method selection or execution."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, cast

import pandas as pd

from .exceptions import ColumnNotFoundError, InvalidDataError
from .profiling import complete_case_count, validate_data_dictionary, variable_intelligence_only
from .results import MissingInformation
from .specifications import (
    AnalysisOptions,
    AnalysisSpecification,
    Objective,
    ResearchQuestion,
    StudyDesign,
    _json_value,
)


class QuestionStatus(str, Enum):
    READY = "ready"
    NEEDS_INPUT = "needs_input"
    DATA_LIMITED = "data_limited"
    UNSUPPORTED = "unsupported"


@dataclass(frozen=True)
class ClarificationQuestion:
    """A GUI-ready question; option values are stable machine identifiers."""

    field: str
    question: str
    explanation: str
    input_type: str
    options: tuple[tuple[str, str], ...] = ()
    required: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "field": self.field,
            "question": self.question,
            "explanation": self.explanation,
            "input_type": self.input_type,
            "options": [{"value": value, "label": label} for value, label in self.options],
            "required": self.required,
        }


@dataclass(frozen=True)
class QuestionDraft:
    """Phase 4 completeness and data availability, without a recommended method."""

    specification: AnalysisSpecification
    status: QuestionStatus
    missing_information: tuple[MissingInformation, ...]
    questions: tuple[ClarificationQuestion, ...]
    warnings: tuple[str, ...]
    blockers: tuple[str, ...]
    variable_suggestions: dict[str, Any]
    availability: dict[str, Any] | None

    def __post_init__(self) -> None:
        if not isinstance(self.specification, AnalysisSpecification):
            raise InvalidDataError("specification must be an AnalysisSpecification.")
        try:
            object.__setattr__(self, "status", QuestionStatus(self.status))
        except (ValueError, TypeError) as exc:
            raise InvalidDataError("QuestionDraft status is invalid.") from exc
        if tuple(item.field for item in self.missing_information) != tuple(
            item.field for item in self.questions
        ):
            raise InvalidDataError("Missing-information fields must match clarification questions.")
        if self.status == QuestionStatus.READY and (
            self.missing_information or self.questions or self.blockers
        ):
            raise InvalidDataError("A ready draft cannot have missing answers or blockers.")
        if self.status == QuestionStatus.NEEDS_INPUT and not self.questions:
            raise InvalidDataError("A needs_input draft requires clarification questions.")
        if (
            self.status in (QuestionStatus.DATA_LIMITED, QuestionStatus.UNSUPPORTED)
            and not self.blockers
        ):
            raise InvalidDataError("A blocked draft requires blockers.")

    def to_dict(self) -> dict[str, Any]:
        return _json_value(
            {
                "specification": self.specification.to_dict(),
                "status": self.status.value,
                "missing_information": [item.to_dict() for item in self.missing_information],
                "questions": [item.to_dict() for item in self.questions],
                "warnings": self.warnings,
                "blockers": self.blockers,
                "variable_suggestions": self.variable_suggestions,
                "availability": self.availability,
            }
        )


_OBJECTIVE_OPTIONS = (
    ("descriptive", "Describe data"),
    ("compare_groups", "Compare groups or conditions"),
    ("association", "Study a relationship"),
)
_DESIGN_OPTIONS = (
    ("independent", "Independent observations"),
    ("paired", "Same or matched participants"),
    ("repeated", "Repeated measurements"),
    ("clustered", "Clustered observations"),
    ("unknown", "I don't know"),
)
_TYPE_OPTIONS = (
    ("continuous", "Continuous measurement"),
    ("discrete", "Numeric count or score"),
    ("nominal", "Unordered categories"),
    ("ordinal", "Ordered categories"),
    ("identifier", "Identifier"),
)


def prepare_question(
    frame: pd.DataFrame,
    *,
    objective: str | Objective | None = None,
    outcome: str | None = None,
    predictor: str | None = None,
    design: str | StudyDesign | None = None,
    estimand: str | None = None,
    description: str | None = None,
    options: AnalysisOptions | None = None,
    data_dictionary: dict[str, dict[str, Any]] | None = None,
    variable_types: dict[str, str] | None = None,
    specification: AnalysisSpecification | None = None,
    variable_metadata: dict[str, str] | None = None,
    unit_id: str | None = None,
    condition_order: tuple[Any, Any] | None = None,
) -> QuestionDraft:
    """Build or revalidate a question against the assistant's copied DataFrame."""
    if specification is not None and not isinstance(specification, AnalysisSpecification):
        raise InvalidDataError("specification must be an AnalysisSpecification.")
    base = specification or AnalysisSpecification()
    prior = base.question
    question = ResearchQuestion(
        objective=cast(Objective | None, objective if objective is not None else prior.objective),
        outcome=outcome if outcome is not None else prior.outcome,
        predictor=predictor if predictor is not None else prior.predictor,
        estimand=estimand if estimand is not None else prior.estimand,
        description=description if description is not None else prior.description,
    )
    selected_design = cast(StudyDesign, design if design is not None else base.design)
    selected_options = options if options is not None else base.options
    selected_unit_id = unit_id if unit_id is not None else base.unit_id
    selected_condition_order = (
        condition_order if condition_order is not None else base.condition_order
    )
    if not isinstance(selected_options, AnalysisOptions):
        raise InvalidDataError("options must be an AnalysisOptions instance.")
    if question.objective == Objective.DESCRIPTIVE and (
        question.predictor is not None
        or question.estimand is not None
        or selected_design not in (StudyDesign.UNKNOWN, "unknown")
    ):
        raise InvalidDataError(
            "Descriptive questions cannot retain a predictor, estimand, or inferential design."
        )
    dictionary = validate_data_dictionary(
        frame, data_dictionary if data_dictionary is not None else base.data_dictionary
    )
    if variable_types is not None:
        if not isinstance(variable_types, dict):
            raise InvalidDataError(
                "variable_types must map existing columns to Phase 3 type names."
            )
        for column, declared_type in variable_types.items():
            candidate = {key: value.copy() for key, value in dictionary.items()}
            candidate.setdefault(column, {})["type"] = declared_type
            dictionary = validate_data_dictionary(frame, candidate)
    for column, entry in dictionary.items():
        entry_type = entry.get("type")
        if entry_type in ("continuous", "discrete") and not pd.api.types.is_numeric_dtype(
            frame[column]
        ):
            raise InvalidDataError(
                f"{column!r} is declared {entry_type} but its pandas dtype "
                "is not numeric. Correct the declaration or source data."
            )
        if entry_type == "boolean" and not pd.api.types.is_bool_dtype(frame[column]):
            raise InvalidDataError(
                f"{column!r} is declared boolean but its pandas dtype is not boolean."
            )
    selected_fields: tuple[tuple[str, str | None], ...] = (
        ("outcome", question.outcome),
        ("predictor", question.predictor),
        ("unit_id", selected_unit_id),
    )
    for field_name, selected_column in selected_fields:
        if selected_column is not None and selected_column not in frame.columns:
            raise ColumnNotFoundError(
                f"{field_name} column {selected_column!r} does not exist. "
                f"Available columns: {list(frame.columns)!r}."
            )
    if (
        question.objective in (Objective.COMPARE_GROUPS, Objective.ASSOCIATION)
        and question.outcome is not None
        and question.outcome == question.predictor
    ):
        raise InvalidDataError(
            "outcome and predictor must be different columns for this objective."
        )
    spec = AnalysisSpecification(
        question=question,
        design=selected_design,
        options=selected_options,
        variable_metadata=variable_metadata
        if variable_metadata is not None
        else base.variable_metadata,
        data_dictionary=dictionary if dictionary else None,
        unit_id=selected_unit_id,
        condition_order=selected_condition_order,
    )
    selected = list(
        dict.fromkeys(
            column for column in (question.outcome, question.predictor) if column is not None
        )
    )
    availability_columns = selected.copy()
    if spec.design == StudyDesign.PAIRED and spec.unit_id is not None:
        availability_columns.append(spec.unit_id)
    hints = variable_intelligence_only(frame, dictionary) if selected else {}
    availability = complete_case_count(frame, availability_columns) if selected else None
    warnings: list[str] = []
    blockers: list[str] = []
    questions: list[ClarificationQuestion] = []

    def ask(
        field: str,
        message: str,
        explanation: str,
        input_type: str,
        choices: tuple[tuple[str, str], ...] = (),
    ) -> None:
        questions.append(ClarificationQuestion(field, message, explanation, input_type, choices))

    if question.objective is None:
        ask(
            "objective",
            "What would you like to study?",
            "Choose the research objective.",
            "select",
            _OBJECTIVE_OPTIONS,
        )
    elif question.objective in (Objective.COMPARE_GROUPS, Objective.ASSOCIATION):
        column_options = tuple((column, column) for column in frame.columns)
        if question.outcome is None:
            ask(
                "outcome",
                "Which first variable would you like to study?",
                "Choose a column; its role is your declaration, not a name-based guess.",
                "column",
                column_options,
            )
        if question.predictor is None:
            wording = (
                "Which column identifies the groups or conditions?"
                if question.objective == Objective.COMPARE_GROUPS
                else "Which second variable would you like to study?"
            )
            ask(
                "predictor",
                wording,
                "Choose a different existing column.",
                "column",
                column_options,
            )
        if question.objective == Objective.COMPARE_GROUPS and question.estimand is None:
            ask(
                "estimand",
                "What would you like to compare?",
                "The target is chosen by the researcher, not by a normality test.",
                "select",
                (("mean", "Mean values"), ("distribution", "Distributions or relative tendency")),
            )
        if spec.design == StudyDesign.UNKNOWN:
            ask(
                "design",
                "How are observations related across rows or groups?",
                "Dependence cannot be established from the values alone.",
                "select",
                _DESIGN_OPTIONS,
            )
        elif spec.design == StudyDesign.PAIRED and spec.unit_id is None:
            ask(
                "unit_id",
                "Which column identifies the same or matched unit across the two conditions?",
                "Pairing cannot be inferred from row order or identifier-like values.",
                "column",
                column_options,
            )

    if spec.unit_id is not None and spec.unit_id in {
        question.outcome,
        question.predictor,
    }:
        raise InvalidDataError("unit_id must differ from the outcome and condition columns.")

    for column in selected:
        observed = frame[column].dropna()
        info = hints[column]
        if observed.empty:
            blockers.append(f"{column!r} has no observed values in this dataset.")
        elif len(observed) > 1 and observed.nunique() == 1:
            warnings.append(f"{column!r} is constant; some later methods may be unavailable.")
        if dictionary.get(column, {}).get("missing_codes"):
            warnings.append(
                f"{column!r} has declared missing codes that are not converted "
                "to pandas missing values."
            )
        if (
            question.objective != Objective.DESCRIPTIVE
            and column == question.outcome
            and info["suggested_type"] == "identifier"
        ):
            warnings.append(f"{column!r} appears to be an identifier; confirm its analytical type.")
        if (
            question.objective != Objective.DESCRIPTIVE
            and (column == question.outcome or question.objective == Objective.ASSOCIATION)
            and (info["ambiguous_numeric_category"] or info["suggested_type"] == "identifier")
            and info["type_source"] != "declared"
        ):
            ask(
                f"variable_types.{column}",
                f"What does {column!r} measure?",
                f"Its {info['observed_dtype']} values suggest {info['suggested_type']}, "
                "but could encode categories or identifiers.",
                "select",
                _TYPE_OPTIONS,
            )
    if question.objective == Objective.COMPARE_GROUPS and question.predictor is not None:
        groups = frame[question.predictor].dropna().nunique()
        if groups < 2:
            blockers.append(
                f"Grouping column {question.predictor!r} has fewer than two observed categories."
            )
    if availability is not None and availability["available_rows"] == 0:
        blockers.append("No rows have complete observed values for the selected variables.")
    missing = tuple(MissingInformation(item.field, item.question) for item in questions)
    status = (
        QuestionStatus.DATA_LIMITED
        if blockers
        else QuestionStatus.NEEDS_INPUT
        if questions
        else QuestionStatus.READY
    )
    return QuestionDraft(
        spec,
        status,
        missing,
        tuple(questions),
        tuple(warnings),
        tuple(blockers),
        {column: hints[column] for column in selected},
        availability,
    )
