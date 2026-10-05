"""Dataset-aware research question intake, without method selection or execution."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, cast

import pandas as pd

from .exceptions import ColumnNotFoundError, InvalidDataError
from .profiling import complete_case_count, validate_data_dictionary, variable_intelligence_only
from .regression import _label as _regression_label
from .regression import _levels as _regression_levels
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
    """Check completeness and data availability without recommending a method."""

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
    ("compare_reference", "Compare a mean with a reference value"),
    ("association", "Study a relationship"),
    ("regression", "Model a conditional mean"),
    ("reliability", "Assess scale internal consistency"),
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
    ("boolean", "Boolean / binary"),
)


def prepare_question(
    frame: pd.DataFrame,
    *,
    objective: str | Objective | None = None,
    outcome: str | None = None,
    predictor: str | None = None,
    predictors: tuple[str, ...] | list[str] | None = None,
    items: tuple[str, ...] | list[str] | None = None,
    controls: tuple[str, ...] | list[str] | None = None,
    design: str | StudyDesign | None = None,
    estimand: str | None = None,
    description: str | None = None,
    options: AnalysisOptions | None = None,
    data_dictionary: dict[str, dict[str, Any]] | None = None,
    variable_types: dict[str, str] | None = None,
    specification: AnalysisSpecification | None = None,
    variable_metadata: dict[str, str] | None = None,
    unit_id: str | None = None,
    condition_order: tuple[Any, ...] | None = None,
    reference_value: float | None = None,
    covariance_type: str | None = None,
    reference_levels: dict[str, Any] | None = None,
    event_level: Any | None = None,
    association_measure: str | None = None,
    factor_a: str | None = None,
    factor_b: str | None = None,
    factors: tuple[str, ...] | list[str] | None = None,
    sum_of_squares: str | None = None,
    target: str | None = None,
    rater: str | None = None,
    model: str | None = None,
    definition: str | None = None,
    unit: str | None = None,
) -> QuestionDraft:
    """Build or revalidate a question against the assistant's copied DataFrame."""
    if specification is not None and not isinstance(specification, AnalysisSpecification):
        raise InvalidDataError("specification must be an AnalysisSpecification.")
    base = specification or AnalysisSpecification()
    prior = base.question
    selected_objective = cast(
        Objective | None, objective if objective is not None else prior.objective
    )
    selected_factor_a = factor_a if factor_a is not None else prior.factor_a
    selected_factor_b = factor_b if factor_b is not None else prior.factor_b
    selected_factors = factors if factors is not None else prior.factors
    has_new_factors = factor_a is not None or factor_b is not None or factors is not None
    selected_predictor = (
        predictor if predictor is not None else (None if has_new_factors else prior.predictor)
    )
    has_new_predictor = predictor is not None
    if has_new_predictor:
        selected_factor_a = factor_a
        selected_factor_b = factor_b
        selected_factors = factors
    selected_predictors = predictors if predictors is not None else prior.predictors
    selected_items = items if items is not None else prior.items
    selected_controls = controls if controls is not None else prior.controls
    selected_target = target if target is not None else prior.target
    selected_rater = rater if rater is not None else prior.rater
    if selected_objective == Objective.REGRESSION and selected_predictors is None:
        if selected_predictor is not None:
            selected_predictors = (selected_predictor,)
    selected_estimand = estimand if estimand is not None else prior.estimand
    if (
        selected_objective == Objective.RELIABILITY
        and (selected_target is not None or selected_rater is not None or model is not None)
        and selected_estimand is None
    ):
        selected_estimand = "intraclass_correlation"
    question = ResearchQuestion(
        objective=selected_objective,
        outcome=outcome if outcome is not None else prior.outcome,
        predictor=selected_predictor,
        predictors=tuple(selected_predictors) if selected_predictors is not None else None,
        factor_a=selected_factor_a,
        factor_b=selected_factor_b,
        factors=tuple(selected_factors) if selected_factors is not None else None,
        target=selected_target,
        rater=selected_rater,
        estimand=selected_estimand,
        description=description if description is not None else prior.description,
        reference_value=(reference_value if reference_value is not None else prior.reference_value),
        items=tuple(selected_items) if selected_items is not None else None,
        controls=tuple(selected_controls) if selected_controls is not None else None,
        event_level=event_level if event_level is not None else prior.event_level,
        association_measure=(
            association_measure if association_measure is not None else prior.association_measure
        ),
    )
    selected_design = cast(StudyDesign, design if design is not None else base.design)
    selected_options = options if options is not None else base.options
    if (
        covariance_type is not None
        or reference_levels is not None
        or sum_of_squares is not None
        or model is not None
        or definition is not None
        or unit is not None
    ):
        selected_options = AnalysisOptions(
            alpha=selected_options.alpha,
            confidence_level=selected_options.confidence_level,
            random_seed=selected_options.random_seed,
            covariance_type=(
                covariance_type if covariance_type is not None else selected_options.covariance_type
            ),
            reference_levels=(
                reference_levels
                if reference_levels is not None
                else selected_options.reference_levels
            ),
            bootstrap_samples=selected_options.bootstrap_samples,
            reverse_scoring=selected_options.reverse_scoring,
            sum_of_squares=(
                sum_of_squares if sum_of_squares is not None else selected_options.sum_of_squares
            ),
            model=model if model is not None else selected_options.model,
            definition=definition if definition is not None else selected_options.definition,
            unit=unit if unit is not None else selected_options.unit,
        )
    selected_unit_id = unit_id if unit_id is not None else base.unit_id
    selected_condition_order = (
        condition_order if condition_order is not None else base.condition_order
    )
    if not isinstance(selected_options, AnalysisOptions):
        raise InvalidDataError("options must be an AnalysisOptions instance.")
    if question.objective == Objective.DESCRIPTIVE and (
        question.predictor is not None
        or question.predictors is not None
        or question.estimand is not None
        or question.reference_value is not None
        or selected_design not in (StudyDesign.UNKNOWN, "unknown")
    ):
        raise InvalidDataError(
            "Descriptive questions cannot retain a predictor, estimand, reference value, "
            "or inferential design."
        )
    if question.objective != Objective.COMPARE_REFERENCE and question.reference_value is not None:
        raise InvalidDataError(
            "reference_value is supported only for objective='compare_reference'."
        )
    if question.objective == Objective.COMPARE_REFERENCE and question.predictor is not None:
        raise InvalidDataError("A reference comparison does not use a predictor column.")
    if question.objective != Objective.REGRESSION and question.predictors is not None:
        raise InvalidDataError("predictors is supported only for objective='regression'.")
    if question.objective != Objective.RELIABILITY and question.items is not None:
        raise InvalidDataError("items is supported only for objective='reliability'.")
    if question.objective != Objective.RELIABILITY and (
        question.target is not None or question.rater is not None
    ):
        raise InvalidDataError("target and rater are supported only for objective='reliability'.")
    if question.objective == Objective.RELIABILITY:
        if question.items is not None and any(
            value is not None
            for value in (
                question.outcome,
                question.predictor,
                question.predictors,
                question.target,
                question.rater,
            )
        ):
            raise InvalidDataError(
                "Multi-item scale reliability uses an explicit items list, "
                "not outcome, predictor, target, or rater roles."
            )
        if question.predictor is not None or question.predictors is not None:
            raise InvalidDataError("Reliability does not use predictor roles.")
    if question.objective != Objective.RELIABILITY and any(
        opt is not None
        for opt in (selected_options.model, selected_options.definition, selected_options.unit)
    ):
        raise InvalidDataError(
            "model, definition, and unit options are supported only for objective='reliability'."
        )
    if question.objective != Objective.ASSOCIATION and question.controls is not None:
        raise InvalidDataError(
            "controls is supported only for objective='association' (partial linear correlation "
            "with estimand='partial_linear'). For multiple regression, pass all explanatory "
            "variables using predictors=..."
        )
    if question.objective != Objective.ASSOCIATION and question.association_measure is not None:
        raise InvalidDataError("association_measure is supported only for objective='association'.")
    if question.objective != Objective.REGRESSION and (
        selected_options.covariance_type != "classical"
        or selected_options.reference_levels is not None
    ):
        raise InvalidDataError(
            "covariance_type and reference_levels are supported only for objective='regression'."
        )
    if (
        question.objective
        not in {
            Objective.RELIABILITY,
            Objective.ASSOCIATION,
            Objective.COMPARE_GROUPS,
            Objective.REGRESSION,
        }
        and selected_options.bootstrap_samples != 499
    ):
        raise InvalidDataError(
            "bootstrap_samples is supported only for objective='reliability', "
            "'association', 'compare_groups', or 'regression'."
        )
    if question.objective != Objective.RELIABILITY and selected_options.reverse_scoring is not None:
        raise InvalidDataError("reverse_scoring is supported only for objective='reliability'.")
    dictionary = validate_data_dictionary(
        frame, data_dictionary if data_dictionary is not None else base.data_dictionary
    )
    if variable_types is not None:
        if not isinstance(variable_types, dict):
            raise InvalidDataError(
                "variable_types must map existing columns to supported analytical type names."
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
        ("factor_a", question.factor_a),
        ("factor_b", question.factor_b),
        ("unit_id", selected_unit_id),
        ("target", question.target),
        ("rater", question.rater),
    )
    for field_name, selected_column in selected_fields:
        if selected_column is not None and selected_column not in frame.columns:
            raise ColumnNotFoundError(
                f"{field_name} column {selected_column!r} does not exist. "
                f"Available columns: {list(frame.columns)!r}."
            )
    for selected_column in question.predictors or ():
        if selected_column not in frame.columns:
            raise ColumnNotFoundError(
                f"predictor column {selected_column!r} does not exist. "
                f"Available columns: {list(frame.columns)!r}."
            )
    for selected_column in question.factors or ():
        if selected_column not in frame.columns:
            raise ColumnNotFoundError(
                f"factor column {selected_column!r} does not exist. "
                f"Available columns: {list(frame.columns)!r}."
            )
    for selected_column in question.items or ():
        if selected_column not in frame.columns:
            raise ColumnNotFoundError(
                f"item column {selected_column!r} does not exist. "
                f"Available columns: {list(frame.columns)!r}."
            )
    for selected_column in question.controls or ():
        if selected_column not in frame.columns:
            raise ColumnNotFoundError(
                f"control column {selected_column!r} does not exist. "
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
    if (
        question.objective == Objective.COMPARE_GROUPS
        and question.outcome is not None
        and question.factors is not None
        and question.outcome in question.factors
    ):
        raise InvalidDataError("outcome cannot be one of the factor columns.")
    if (
        question.objective == Objective.REGRESSION
        and question.outcome is not None
        and question.outcome in (question.predictors or ())
    ):
        raise InvalidDataError("The regression outcome cannot also appear among predictors.")
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
            column
            for column in (
                question.outcome,
                question.predictor,
                *(question.predictors or ()),
                *(question.factors or ()),
                *(question.controls or ()),
                question.target,
                question.rater,
            )
            if column is not None
        )
    )
    selected.extend(item for item in (question.items or ()) if item not in selected)
    availability_columns = selected.copy()
    if spec.design in (StudyDesign.PAIRED, StudyDesign.REPEATED) and spec.unit_id is not None:
        availability_columns.append(spec.unit_id)
    hints = variable_intelligence_only(frame, dictionary) if selected else {}
    if question.objective == Objective.REGRESSION and selected:
        resolved_references = dict(spec.options.reference_levels or {})
        predictors_set = set(question.predictors or ())
        unknown_references = set(resolved_references) - predictors_set
        if unknown_references:
            raise InvalidDataError(
                f"reference_levels names non-predictors: {sorted(unknown_references)!r}."
            )
        complete = (
            frame[[question.outcome, *(question.predictors or ())]].dropna()
            if question.outcome is not None and question.predictors
            else None
        )
        for predictor_name in question.predictors or ():
            predictor_type = hints[predictor_name]["suggested_type"]
            if predictor_type in {"continuous_numerical", "discrete_numerical"}:
                if predictor_name in resolved_references:
                    raise InvalidDataError(
                        f"reference_levels[{predictor_name!r}] is invalid because the predictor "
                        "is numerical."
                    )
                continue
            if predictor_type not in {"nominal_categorical", "ordinal_categorical", "boolean"}:
                continue
            if complete is None or complete.empty:
                continue
            levels = _regression_levels(
                complete[predictor_name], dictionary.get(predictor_name, {})
            )
            if predictor_name in resolved_references:
                if resolved_references[predictor_name] not in levels:
                    raise InvalidDataError(
                        f"Reference level {resolved_references[predictor_name]!r} is not observed "
                        f"for predictor {predictor_name!r}."
                    )
            elif levels:
                resolved_references[predictor_name] = _regression_label(levels[0])
        regression_options = AnalysisOptions(
            alpha=spec.options.alpha,
            confidence_level=spec.options.confidence_level,
            random_seed=spec.options.random_seed,
            covariance_type=spec.options.covariance_type,
            reference_levels=resolved_references or None,
            bootstrap_samples=spec.options.bootstrap_samples,
            reverse_scoring=spec.options.reverse_scoring,
        )
        spec = AnalysisSpecification(
            question=spec.question,
            design=spec.design,
            options=regression_options,
            variable_metadata=spec.variable_metadata,
            data_dictionary=spec.data_dictionary,
            unit_id=spec.unit_id,
            condition_order=spec.condition_order,
            analytical_variable_types={
                column: (
                    hints[column]["suggested_type"]
                    if hints[column]["type_source"] == "declared"
                    else "continuous_numerical"
                    if pd.api.types.is_numeric_dtype(frame[column])
                    else hints[column]["suggested_type"]
                )
                for column in selected
            },
        )
    elif question.objective in {Objective.RELIABILITY, Objective.ASSOCIATION} and selected:
        spec = AnalysisSpecification(
            question=spec.question,
            design=spec.design,
            options=spec.options,
            variable_metadata=spec.variable_metadata,
            data_dictionary=spec.data_dictionary,
            unit_id=spec.unit_id,
            condition_order=spec.condition_order,
            analytical_variable_types={
                column: hints[column]["suggested_type"] for column in selected
            },
        )
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
        if question.predictor is None and question.factors is None:
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
                (
                    ("mean", "Mean values"),
                    ("distribution", "Distributions or relative tendency"),
                    ("proportion", "Paired event proportions"),
                ),
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
        elif spec.design == StudyDesign.REPEATED and spec.unit_id is None:
            ask(
                "unit_id",
                "Which column identifies the repeated observational unit across conditions?",
                "Repeated structure cannot be inferred from row order or identifier-like values.",
                "column",
                column_options,
            )
    elif question.objective == Objective.COMPARE_REFERENCE:
        column_options = tuple((column, column) for column in frame.columns)
        if question.outcome is None:
            ask(
                "outcome",
                "Which continuous outcome should be compared with a reference value?",
                "Choose the measured outcome column.",
                "column",
                column_options,
            )
        if question.estimand is None:
            ask(
                "estimand",
                "What population quantity should be compared with the reference?",
                "The one-sample t workflow currently supports a population-mean target.",
                "select",
                (("mean", "Population mean"),),
            )
        if question.reference_value is None:
            ask(
                "reference_value",
                "What finite reference value should the population mean be compared with?",
                "This value defines the null hypothesis and must come from the researcher.",
                "number",
            )
        if spec.design == StudyDesign.UNKNOWN:
            ask(
                "design",
                "Are the observations independent units?",
                "Independence cannot be established from the values alone.",
                "select",
                _DESIGN_OPTIONS,
            )
    elif question.objective == Objective.REGRESSION:
        column_options = tuple((column, column) for column in frame.columns)
        if question.outcome is None:
            ask(
                "outcome",
                "Which continuous outcome should the model explain?",
                "OLS models a continuous conditional mean.",
                "column",
                column_options,
            )
        if not question.predictors:
            ask(
                "predictors",
                "Which one or more predictors should enter the model?",
                "Predictor choice is supplied by the researcher; no automatic selection occurs.",
                "columns",
                column_options,
            )
        if question.estimand is None:
            ask(
                "estimand",
                "What model target should be estimated?",
                "Choose the outcome-appropriate model target.",
                "select",
                (
                    ("conditional_mean", "Conditional mean"),
                    ("event_probability", "Probability / odds of a binary event"),
                ),
            )
        if spec.design == StudyDesign.UNKNOWN:
            ask(
                "design",
                "Are rows independent observational units?",
                "Independence cannot be inferred from predictor values.",
                "select",
                _DESIGN_OPTIONS,
            )
    elif question.objective == Objective.RELIABILITY:
        column_options = tuple((column, column) for column in frame.columns)
        is_icc = (
            question.target is not None
            or question.rater is not None
            or question.estimand
            in {
                "icc",
                "intraclass_correlation",
                "inter_rater_reliability",
                "agreement",
            }
            or (
                not question.items
                and (
                    question.outcome is not None
                    or spec.options.model is not None
                    or spec.options.definition is not None
                    or spec.options.unit is not None
                )
            )
        )
        if is_icc:
            if question.target is None:
                ask(
                    "target",
                    "Which column identifies the rated targets or subjects?",
                    "Target identity is a scientific design role; "
                    "it is never guessed from column names.",
                    "column",
                    column_options,
                )
            if question.rater is None:
                ask(
                    "rater",
                    "Which column identifies the raters, judges, or measurement occasions?",
                    "Rater identity is a scientific design role; "
                    "it is never guessed from column names.",
                    "column",
                    column_options,
                )
            if question.outcome is None:
                ask(
                    "outcome",
                    "Which continuous column contains the quantitative ratings or scores?",
                    "Choose the quantitative measurement column.",
                    "column",
                    column_options,
                )
            if question.estimand is None:
                ask(
                    "estimand",
                    "What reliability estimand should be evaluated?",
                    "Intraclass correlation estimates quantitative inter-rater or "
                    "test-retest reliability/agreement.",
                    "select",
                    (("intraclass_correlation", "Intraclass correlation coefficient (ICC)"),),
                )
            if spec.options.model is None:
                ask(
                    "model",
                    "How were raters sampled or assigned across targets?",
                    "Rater sampling determines whether results generalize to a broader "
                    "rater population or apply only to these specific raters.",
                    "select",
                    (
                        (
                            "two_way_random",
                            "Two-way random: Each target rated by the same random sample of "
                            "raters from a broader population",
                        ),
                        (
                            "two_way_mixed",
                            "Two-way mixed: These specific raters are the only raters of "
                            "interest (fixed raters)",
                        ),
                        (
                            "one_way_random",
                            "One-way random: Each target rated by a different set of "
                            "randomly selected raters",
                        ),
                    ),
                )
            if spec.options.model != "one_way_random" and spec.options.definition is None:
                ask(
                    "definition",
                    "Is exact numerical agreement required, or relative consistency?",
                    "Absolute agreement penalizes systematic rater level offsets; "
                    "consistency ignores systematic additive rater differences.",
                    "select",
                    (
                        (
                            "absolute_agreement",
                            "Absolute agreement: Systematic differences between raters count "
                            "as disagreement",
                        ),
                        (
                            "consistency",
                            "Consistency: Evaluates relative ordering/pattern across targets; "
                            "ignores systematic rater bias",
                        ),
                    ),
                )
            if spec.options.unit is None:
                ask(
                    "unit",
                    "Will decisions be based on a single rating or the average of all raters?",
                    "Single-measure ICC estimates reliability of one typical rating; "
                    "average-measure ICC estimates reliability of the mean of k ratings.",
                    "select",
                    (
                        ("single", "Single rating: Application relies on an individual rating"),
                        (
                            "average",
                            "Average rating: Application averages ratings across all raters",
                        ),
                    ),
                )
        else:
            if not question.items:
                ask(
                    "items",
                    "Which two or more scored items form the proposed scale?",
                    "Scale membership is declared by the researcher and is never inferred.",
                    "columns",
                    column_options,
                )
            if question.estimand is None:
                ask(
                    "estimand",
                    "What scale property should be summarized?",
                    "This workflow estimates internal consistency with Cronbach's alpha.",
                    "select",
                    (("internal_consistency", "Internal consistency"),),
                )

    def default_event_for(column: str | None) -> Any | None:
        if column is None:
            return None
        observed = frame[column].dropna()
        if observed.nunique() != 2:
            return None
        if pd.api.types.is_bool_dtype(observed):
            return True
        declared = dictionary.get(column, {}).get("type")
        values = set(observed.tolist())
        if declared in {"boolean", "nominal"} and values == {0, 1}:
            return 1
        return None

    event_variable: str | None = None
    if question.objective == Objective.REGRESSION and question.estimand == "event_probability":
        event_variable = question.outcome
    elif (
        question.objective == Objective.COMPARE_GROUPS
        and question.estimand == "proportion"
        and spec.design == StudyDesign.PAIRED
    ):
        event_variable = question.outcome
    elif (
        question.objective == Objective.ASSOCIATION
        and question.estimand in {"linear", "point_biserial"}
        and question.outcome
        and question.predictor
    ):
        outcome_kind = hints[question.outcome]["suggested_type"]
        predictor_kind = hints[question.predictor]["suggested_type"]
        if outcome_kind in {"nominal_categorical", "boolean"} and predictor_kind in {
            "continuous_numerical",
            "discrete_numerical",
        }:
            event_variable = question.outcome
        elif predictor_kind in {"nominal_categorical", "boolean"} and outcome_kind in {
            "continuous_numerical",
            "discrete_numerical",
        }:
            event_variable = question.predictor
    if event_variable is not None and question.event_level is None:
        automatic_event = default_event_for(event_variable)
        if automatic_event is not None:
            question = ResearchQuestion(**{**question.to_dict(), "event_level": automatic_event})
            spec = AnalysisSpecification(
                question=question,
                design=spec.design,
                options=spec.options,
                variable_metadata=spec.variable_metadata,
                data_dictionary=spec.data_dictionary,
                unit_id=spec.unit_id,
                condition_order=spec.condition_order,
                analytical_variable_types=spec.analytical_variable_types,
            )
        else:
            choices = tuple(
                (str(value), str(value)) for value in pd.unique(frame[event_variable].dropna())
            )
            ask(
                "event_level",
                f"Which level of {event_variable!r} is the event / positive category?",
                "The event defines coefficient and effect orientation; it is never chosen "
                "alphabetically.",
                "select",
                choices,
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
            and question.objective != Objective.RELIABILITY
            and (
                column == question.outcome
                or question.objective
                in (Objective.ASSOCIATION, Objective.REGRESSION, Objective.RELIABILITY)
            )
            and (
                info["suggested_type"] == "identifier"
                or (
                    question.objective != Objective.RELIABILITY
                    and info["ambiguous_numeric_category"]
                )
            )
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
    if question.objective == Objective.COMPARE_GROUPS and question.factors is not None:
        for factor_col in question.factors:
            groups = frame[factor_col].dropna().nunique()
            if groups < 2:
                blockers.append(
                    f"Factor column {factor_col!r} has fewer than two observed categories."
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
