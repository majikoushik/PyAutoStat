"""Public entry point for profiling, question intake, and method recommendations."""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from datetime import datetime
from typing import Any, cast

import pandas as pd

from .analysis_plan import (
    AnalysisPlanStatus,
    PlanAdherenceResult,
    StatisticalAnalysisPlan,
    compare_plan_to_result,
    planned_interval_quantity,
    planned_quantity,
)
from .analyzer import StatisticalAnalyzer
from .audit import AuditResult, StatisticalResultAuditor
from .completeness import ReportingCompletenessResult, assess_reporting_completeness
from .decision_ledger import DecisionLedger
from .exceptions import InvalidDataError, PyAutoStatError
from .execution import execute_selected_method, execute_specification
from .interpretation import InterpretationEngine, InterpretationResult
from .narrate import _dataset_story
from .practical_significance import (
    MeaningfulEffectThreshold,
    PracticalSignificanceResult,
    assess_practical_significance,
)
from .profiling import complete_case_count
from .provenance import content_reference, dataset_fingerprint
from .question_builder import QuestionDraft, prepare_question
from .recommendation import recommend_from_draft
from .reproducibility import ReproducibilityRecord
from .research_report import ResearchReport, build_research_report
from .results import AnalysisResult, Recommendation
from .sensitivity import (
    Comparability,
    ScenarioStatus,
    SensitivityResult,
    SensitivityScenarioResult,
    SensitivitySpecification,
    SensitivityStatus,
    classify_comparability,
    compare_same_estimand,
    estimate_quantity,
    scenario_values,
)
from .session import ResearchSessionSnapshot, build_session_snapshot
from .specifications import (
    AnalysisOptions,
    AnalysisSpecification,
    Objective,
    ResearchQuestion,
    StudyDesign,
)
from .study_planning import StudyPlanner, StudyPlanningResult
from .workflow import ResearchWorkflowResult, WorkflowStatus


class ResearchAssistant:
    """Recommended high-level entry point for statistical research workflows.

    Coordinates dataset profiling, question intake, design-aware method
    recommendation, validated execution, deterministic interpretation,
    audit, reproducibility records, and research reporting for pandas DataFrames.

    Key principles:
    - **DataFrame safety**: Validates the input DataFrame, preserves source data
      without modification, and uses a private copy for internal operations.
    - **No scientific guessing**: Never infers missing study designs, pairing,
      or estimands from raw values. Unresolved design facts pause the workflow
      with structured clarification requests (``needs_input``).
    - **No side effects**: Running analyses does not write files to disk. File
      export is explicit through functions such as :func:`pyautostat.save_html`
      or :func:`pyautostat.save_pdf`.
    - **Dual workflow support**: Provides both the integrated, guided :meth:`run`
      entry point and focused convenience methods (:meth:`profile`,
      :meth:`summarize`, :meth:`reliability`, :meth:`two_way_anova`,
      :meth:`intraclass_correlation`).
    - **Layered architecture**: Underlying direct calculation and profiling
      operations remain accessible through :class:`pyautostat.StatisticalAnalyzer`.
    """

    def __init__(self, df: pd.DataFrame) -> None:
        self._analyzer = StatisticalAnalyzer(df)
        self._data_dictionary: dict | None = None
        self._ledger: DecisionLedger | None = None
        self._planning_status = "unknown"
        self._last_meaningful_threshold: dict[str, Any] | None = None
        self._analysis_has_executed = False

    def enable_tracking(
        self, *, clock: Callable[[], datetime | str] | None = None
    ) -> DecisionLedger:
        """Start observing subsequent actions; no earlier decisions are reconstructed."""
        if self._ledger is None:
            self._ledger = DecisionLedger(clock=clock)
        return self._ledger

    @property
    def decision_ledger(self) -> DecisionLedger | None:
        return self._ledger

    @property
    def planning_status(self) -> str:
        return self._planning_status

    def declare_planning(self, status: str, *, reason: str | None = None) -> None:
        """Record a researcher declaration, never independent preregistration proof."""
        if status not in {"planned", "exploratory", "unknown"}:
            raise InvalidDataError("planning status must be planned, exploratory, or unknown.")
        if reason is not None and (not isinstance(reason, str) or not reason.strip()):
            raise InvalidDataError("A supplied planning reason must be non-empty text.")
        previous = self._planning_status
        if self._ledger is not None:
            self._ledger._record(
                "planning_declared",
                source="researcher",
                previous_state=previous,
                new_state=status,
                reason=reason,
            )
        self._planning_status = status

    def profile(
        self,
        *,
        data_dictionary=None,
        histogram_bins: int = 20,
        include_row_positions: bool = False,
        quantiles=(0.05, 0.25, 0.5, 0.75, 0.95),
    ) -> dict:
        """Profile the dataset structure, missingness, distributions, and data quality.

        Use `profile()` as the introductory, dataset-only entry point to inspect
        univariate distributions, skewness, normality tests, correlation pairs,
        and data-quality cues before specifying an inferential research question.

        Parameters
        ----------
        data_dictionary : dict, optional
            Variable definitions or domain notes. Optional declarations never mutate source values.
        histogram_bins : int, default 20
            Number of bins for numeric histogram profiles.
        include_row_positions : bool, default False
            Whether to include row index positions for missing values and anomalies.
        quantiles : tuple of float, default (0.05, 0.25, 0.5, 0.75, 0.95)
            Quantile cutoff positions to compute for numeric columns.

        Returns
        -------
        dict
            Structured profiling dictionary with keys for overview, distributions,
            missing_data, normality, correlation, data_quality, and analysis_warnings.
            Does not write files to disk.
        """
        result = self._analyzer.analyze_all(
            data_dictionary=data_dictionary,
            histogram_bins=histogram_bins,
            include_row_positions=include_row_positions,
            quantiles=quantiles,
        )
        self._data_dictionary = deepcopy(result["data_dictionary"])
        return result

    def summarize(
        self,
        *,
        data_dictionary=None,
        histogram_bins: int = 20,
        mode: str = "profile",
        quantiles=(0.05, 0.25, 0.5, 0.75, 0.95),
    ) -> str:
        """Return a readable plain-text overview of the dataset.

        Calls :meth:`profile` and then formats the most important findings —
        shape, missing data, distribution flags, strong correlations, and data
        quality — into a single printable block.  No new statistics are
        calculated beyond what :meth:`profile` already produces. ``mode="profile"``
        preserves the established summary. ``mode="story"`` returns the
        deterministic connected narrative.

        Example usage::

            assistant = ResearchAssistant(df)
            print(assistant.summarize())

        Returns
        -------
        str
            A multi-section plain-text summary suitable for printing or logging.
        """
        if mode not in {"profile", "story"}:
            raise InvalidDataError("summary mode must be 'profile' or 'story'.")
        result = self.profile(
            data_dictionary=data_dictionary,
            histogram_bins=histogram_bins,
            quantiles=quantiles,
        )
        if mode == "story":
            return _dataset_story(result)
        sep = "=" * 68
        thin = "-" * 68
        lines: list[str] = [sep]

        # ── Shape ────────────────────────────────────────────────────────────
        overview = result.get("overview", {})
        rows = overview.get("total_rows", "?")
        cols = overview.get("total_columns", "?")
        num_cols = overview.get("numeric_columns", 0)
        cat_cols = overview.get("categorical_columns", 0)
        lines.append(f" DATASET PROFILE | {rows} rows x {cols} columns")
        lines.append(sep)
        lines.append(f" Numeric columns     : {num_cols}")
        lines.append(f" Categorical columns : {cat_cols}")

        # ── Missing data ─────────────────────────────────────────────────────
        missing = result.get("missing_data", {})
        overall_pct = missing.get("overall_missing_percentage", 0) or 0
        by_col = missing.get("by_column", {})
        missing_cols = {
            col: info
            for col, info in by_col.items()
            if isinstance(info, dict) and (info.get("count") or 0) > 0
        }
        lines.append(thin)
        lines.append(" MISSING DATA")
        if not missing_cols:
            lines.append("   OK: No missing values; dataset is complete.")
        else:
            lines.append(f"   Overall missing rate : {overall_pct:.1f}%")
            for col, info in missing_cols.items():
                pct = info.get("percentage", 0) or 0
                cnt = info.get("count", 0) or 0
                flag = "  WARNING: >10%" if pct > 10 else ""
                lines.append(f"   {col:<22}: {pct:5.1f}%  ({cnt} values){flag}")

        # ── Distribution shape ────────────────────────────────────────────────
        distributions = result.get("distributions", {})
        normality = result.get("normality", {})
        lines.append(thin)
        lines.append(" DISTRIBUTIONS  (numeric columns)")
        if not distributions:
            lines.append("   No numeric columns to summarise.")
        else:
            for col, dist in distributions.items():
                skew_text = dist.get("skewness_interpretation", "Unknown")
                col_norm = normality.get(col, {})
                sw = col_norm.get("shapiro_wilk", {})
                dag = col_norm.get("d_agostino_pearson", {})
                norm_verdict = (
                    sw.get("verdict") or dag.get("verdict") or "Normality test not available"
                )
                bimodal = dist.get("is_bimodal")
                bimodal_note = "  [possible bimodal shape]" if bimodal else ""
                lines.append(f"   {col:<22}: {skew_text}{bimodal_note}")
                lines.append(f"   {'':>22}  Result: {norm_verdict}")

        # ── Correlations ─────────────────────────────────────────────────────
        corr_data = result.get("correlation", {})
        pearson_matrix = corr_data.get("pearson", {}).get("matrix", {})
        strong_pairs: list[tuple[str, str, float]] = []
        seen: set[frozenset[str]] = set()
        for col1, row_vals in pearson_matrix.items():
            for col2, r in (row_vals or {}).items():
                pair = frozenset({col1, col2})
                if col1 != col2 and pair not in seen and isinstance(r, (int, float)):
                    seen.add(pair)
                    if abs(r) >= 0.7:
                        strong_pairs.append((col1, col2, float(r)))
        lines.append(thin)
        lines.append(" CORRELATIONS  (|r| >= 0.70)")
        if not strong_pairs:
            lines.append("   No strong correlations detected.")
        else:
            for col1, col2, r in sorted(strong_pairs, key=lambda x: -abs(x[2])):
                direction = "positive" if r > 0 else "negative"
                lines.append(f"   {col1} <-> {col2} : r = {r:.3f}  ({direction})")

        # ── Data quality ─────────────────────────────────────────────────────
        quality = result.get("data_quality", {})
        completeness = quality.get("completeness", 1.0)
        if not isinstance(completeness, (int, float)):
            completeness = 1.0
        dupes_pct = quality.get("duplicate_rows_percentage", 0) or 0
        dupes_n = quality.get("duplicate_rows", 0) or 0
        lines.append(thin)
        lines.append(" DATA QUALITY")
        lines.append(f"   Completeness    : {completeness * 100:.1f}%")
        dup_flag = "  WARNING" if dupes_pct > 1 else "  OK"
        lines.append(f"   Duplicate rows  : {dupes_n}  ({dupes_pct:.1f}%){dup_flag}")

        # ── Warnings ─────────────────────────────────────────────────────────
        warnings_list = result.get("analysis_warnings", [])
        if warnings_list:
            lines.append(thin)
            lines.append(f" ANALYSIS WARNINGS  ({len(warnings_list)} total)")
            for w in warnings_list[:5]:
                msg = w.get("message", str(w)) if isinstance(w, dict) else str(w)
                lines.append(f"   ! {msg}")
            if len(warnings_list) > 5:
                lines.append(
                    f"   ... and {len(warnings_list) - 5} more; see profile()['analysis_warnings']"
                )

        lines.append(sep)
        return "\n".join(lines)

    def complete_case_count(self, columns: list[str]) -> dict:
        """Count rows available for a specified set of columns."""
        return complete_case_count(self._analyzer.df, columns)

    def frequency_table(self, column: str, *, data_dictionary=None) -> dict:
        """Return a categorical frequency table plus deterministic narration."""
        return self._analyzer.frequency_table(column, data_dictionary=data_dictionary)

    def cross_tab(self, row_variable: str, column_variable: str, *, data_dictionary=None) -> dict:
        """Return descriptive counts and row, column, and total percentages."""
        return self._analyzer.cross_tab(
            row_variable, column_variable, data_dictionary=data_dictionary
        )

    def reliability(
        self,
        items: list[str] | tuple[str, ...],
        *,
        confidence_level: float = 0.95,
        bootstrap_samples: int = 499,
        random_state: int | None = 0,
        reverse_scoring: dict[str, tuple[float, float]] | None = None,
        data_dictionary: dict | None = None,
        title: str | None = None,
        audit: bool = True,
        fingerprint: bool = True,
    ) -> ResearchWorkflowResult:
        """Estimate multi-item internal consistency using Cronbach's alpha.

        Focused convenience method for psychometric scale reliability. Computes
        sample-variance Cronbach's alpha, respondent-row bootstrap confidence intervals,
        corrected item-total correlations, alpha-if-item-deleted diagnostics,
        inter-item correlations, mean inter-item correlation, and explicit reverse scoring.

        Use this method instead of ``run(objective="reliability", ...)`` when
        evaluating a multi-item survey or questionnaire scale.

        Parameters
        ----------
        items : sequence of str
            Scored numeric column names belonging to the scale.
        confidence_level : float, default 0.95
            Coverage for the bootstrap confidence interval.
        bootstrap_samples : int, default 499
            Number of bootstrap resamples for the alpha confidence interval.
        random_state : int or None, default 0
            Seed for reproducible bootstrap resampling.
        reverse_scoring : dict, optional
            Mapping of column names to (min, max) bounds for negatively keyed items.
        data_dictionary : dict, optional
            Optional metadata dictionary for item descriptions.
        title : str, optional
            Title for generated workflow reports.
        audit : bool, default True
            Whether to run result audits. Setting False results in a partial workflow status.
        fingerprint : bool, default True
            Whether to compute a SHA-256 fingerprint of the analyzed columns.

        Returns
        -------
        ResearchWorkflowResult
            Integrated workflow result containing analysis values, interpretation,
            audit, and report. Does not write files to disk.
        """
        return self.run(
            objective=Objective.RELIABILITY,
            items=items,
            estimand="internal_consistency",
            options=AnalysisOptions(
                confidence_level=confidence_level,
                random_seed=random_state,
                bootstrap_samples=bootstrap_samples,
                reverse_scoring=reverse_scoring,
            ),
            data_dictionary=data_dictionary,
            title=title,
            audit=audit,
            fingerprint=fingerprint,
        )

    def two_way_anova(
        self,
        outcome: str,
        factor_a: str,
        factor_b: str,
        *,
        sum_of_squares: str = "type2",
        alpha: float = 0.05,
        confidence_level: float = 0.95,
        title: str | None = None,
        audit: bool = True,
        fingerprint: bool = True,
    ) -> ResearchWorkflowResult:
        """Run two-way factorial ANOVA for independent groups with interaction.

        Focused convenience method for evaluating two categorical grouping factors
        and their interaction on a continuous outcome. Computes main effects,
        interaction F-tests, partial eta-squared effect sizes with exact confidence
        intervals, and unweighted estimated marginal means.

        Use this method instead of ``run(objective="compare_groups", factor_a=..., factor_b=...)``
        when conducting a 2-factor between-subjects design.

        Parameters
        ----------
        outcome : str
            Continuous dependent variable column.
        factor_a : str
            First categorical grouping factor column.
        factor_b : str
            Second categorical grouping factor column.
        sum_of_squares : {"type2", "type3"}, default "type2"
            Sum of squares formulation. Type II is recommended for unbalanced
            designs without strong interaction hypotheses.
        alpha : float, default 0.05
            Significance threshold for hypothesis tests.
        confidence_level : float, default 0.95
            Coverage for partial eta-squared confidence intervals.
        title : str, optional
            Title for generated workflow reports.
        audit : bool, default True
            Whether to run result audits. Setting False results in a partial workflow status.
        fingerprint : bool, default True
            Whether to compute a SHA-256 fingerprint of the analyzed columns.

        Returns
        -------
        ResearchWorkflowResult
            Integrated workflow result containing ANOVA tables, effect sizes,
            interpretation, audit, and report. Does not write files to disk.
        """
        return self.run(
            objective=Objective.COMPARE_GROUPS,
            outcome=outcome,
            factor_a=factor_a,
            factor_b=factor_b,
            design=StudyDesign.INDEPENDENT,
            estimand="mean",
            sum_of_squares=sum_of_squares,
            options=AnalysisOptions(
                alpha=alpha,
                confidence_level=confidence_level,
                sum_of_squares=sum_of_squares,
            ),
            title=title,
            audit=audit,
            fingerprint=fingerprint,
        )

    def intraclass_correlation(
        self,
        target: str,
        rater: str,
        value: str,
        *,
        model: str | None = None,
        definition: str | None = None,
        unit: str | None = None,
        alpha: float = 0.05,
        confidence_level: float = 0.95,
        title: str | None = None,
        audit: bool = True,
        fingerprint: bool = True,
    ) -> ResearchWorkflowResult:
        """Estimate Intraclass Correlation Coefficients (ICC) for rater agreement/reliability.

        This is the canonical explicit method for ICC estimation. Supports Shrout & Fleiss
        (1979) and McGraw & Wong (1996) models: ICC(1,1), ICC(2,1), ICC(3,1), ICC(1,k),
        ICC(2,k), and ICC(3,k), computing variance components, F-tests, and confidence intervals.

        For a convenience shorthand, :meth:`icc` is an identical alias.

        Parameters
        ----------
        target : str
            Column identifying the subject, patient, or item being evaluated.
        rater : str
            Column identifying the observer, rater, or measurement device.
        value : str
            Continuous measurement or score column.
        model : {"one_way_random", "two_way_random", "two_way_mixed"}, optional
            Rater effect structure. Explicit choice is required; unresolved choices
            yield a structured needs_input workflow status.
        definition : {"absolute_agreement", "consistency"}, optional
            Whether absolute systematic differences between raters count as disagreement.
            Required for two-way models.
        unit : {"single", "average"}, optional
            Whether the measurement of interest is a single rating or the mean of k ratings.
            Explicit choice is required.
        alpha : float, default 0.05
            Significance threshold for F-tests.
        confidence_level : float, default 0.95
            Coverage for ICC confidence intervals.
        title : str, optional
            Title for generated workflow reports.
        audit : bool, default True
            Whether to run result audits. Setting False results in a partial workflow status.
        fingerprint : bool, default True
            Whether to compute a SHA-256 fingerprint of the analyzed columns.

        Returns
        -------
        ResearchWorkflowResult
            Integrated workflow result containing ICC estimates, ANOVA decomposition,
            interpretation, audit, and report. Does not write files to disk.
        """
        return self.run(
            objective=Objective.RELIABILITY,
            target=target,
            rater=rater,
            outcome=value,
            estimand="intraclass_correlation",
            model=model,
            definition=definition,
            unit=unit,
            options=AnalysisOptions(
                alpha=alpha,
                confidence_level=confidence_level,
                model=model,
                definition=definition,
                unit=unit,
            ),
            title=title,
            audit=audit,
            fingerprint=fingerprint,
        )

    def icc(
        self,
        target: str,
        rater: str,
        value: str,
        *,
        model: str | None = None,
        definition: str | None = None,
        unit: str | None = None,
        alpha: float = 0.05,
        confidence_level: float = 0.95,
        title: str | None = None,
        audit: bool = True,
        fingerprint: bool = True,
    ) -> ResearchWorkflowResult:
        """Convenience alias for :meth:`intraclass_correlation`.

        Accepts identical arguments and returns identical results. In documentation
        and public APIs, :meth:`intraclass_correlation` is the canonical name.
        """
        return self.intraclass_correlation(
            target=target,
            rater=rater,
            value=value,
            model=model,
            definition=definition,
            unit=unit,
            alpha=alpha,
            confidence_level=confidence_level,
            title=title,
            audit=audit,
            fingerprint=fingerprint,
        )

    def study_planner(self) -> StudyPlanner:
        """Return a prospective study planner for sample size, power, and effect calculation.

        Creates an independent :class:`pyautostat.StudyPlanner` instance. Study planning is
        strictly prospective and does not inspect or depend on the assistant's DataFrame.
        When a decision ledger is active, completed planning calculations are recorded
        to the ledger for scientific provenance.

        Returns
        -------
        StudyPlanner
            A planner instance supporting prospective independent-mean and paired-mean
            statistical power and sample-size precision planning.
        """

        def record(result: StudyPlanningResult) -> None:
            if self._ledger is not None:
                payload = result.to_dict()
                self._ledger._record(
                    "study_planning_completed",
                    source="researcher",
                    references={"study_planning": content_reference("study_planning", payload)},
                    metadata={
                        "status": result.status,
                        "planning_type": result.planning_type,
                        "method_family": result.method_family,
                    },
                )

        return StudyPlanner(on_result=record)

    def run(
        self,
        *,
        objective: str | Objective | None = None,
        outcome: str | None = None,
        predictor: str | None = None,
        predictors: list[str] | tuple[str, ...] | None = None,
        items: list[str] | tuple[str, ...] | None = None,
        controls: list[str] | tuple[str, ...] | None = None,
        design: str | StudyDesign | None = None,
        estimand: str | None = None,
        description: str | None = None,
        options: AnalysisOptions | None = None,
        data_dictionary: dict | None = None,
        variable_types: dict[str, str] | None = None,
        unit_id: str | None = None,
        condition_order: tuple[Any, ...] | None = None,
        reference_value: float | None = None,
        covariance_type: str | None = None,
        reference_levels: dict[str, Any] | None = None,
        event_level: Any | None = None,
        association_measure: str | None = None,
        factor_a: str | None = None,
        factor_b: str | None = None,
        factors: list[str] | tuple[str, ...] | None = None,
        sum_of_squares: str | None = None,
        target: str | None = None,
        rater: str | None = None,
        value: str | None = None,
        model: str | None = None,
        definition: str | None = None,
        unit: str | None = None,
        draft: QuestionDraft | None = None,
        specification: AnalysisSpecification | None = None,
        include_profile: bool = False,
        audit: bool = True,
        fingerprint: bool = True,
        title: str | None = None,
        include_figures: bool = False,
    ) -> ResearchWorkflowResult:
        """Execute a design-aware research workflow or return structured requests for missing facts.

        Coordinates question intake, candidate method recommendation, assumption
        validation, statistical execution, deterministic interpretation, decision auditing,
        and report assembly into a single :class:`pyautostat.ResearchWorkflowResult`.

        Scientific design facts (such as independence vs. pairing or collection intent)
        are never guessed from data values. If essential design information is missing,
        ``run()`` pauses execution with a structured ``needs_input`` status rather than
        silently choosing a default.

        Most workflows use only ``objective``, ``outcome``, ``predictor``, ``estimand``,
        and ``design``. The remaining parameters are workflow-specific and can be
        safely omitted unless your analysis requires them.

        Common question parameters
        --------------------------
        objective : str or Objective, optional
            Research objective: ``"descriptive"``, ``"compare_groups"``,
            ``"compare_reference"``, ``"association"``, ``"regression"``, or
            ``"reliability"`` (or corresponding :class:`Objective` enum values).
        outcome : str, optional
            Primary outcome variable column name. For two-variable association, the first
            variable.
        predictor : str, optional
            Single grouping variable, condition variable, or explanatory variable column name.
            In simple regression, the single predictor. For multiple regression, use `predictors`.
            When both `predictor` and `predictors` are supplied, they must be consistent
            (e.g., ``predictors=(predictor,)``); conflicting values are rejected.
        estimand : str, optional
            Scientific target of interest (e.g. ``"mean"``, ``"distribution"``,
            ``"internal_consistency"``, ``"linear"``, ``"rank"``, ``"categorical_independence"``).
            Method selection respects the declared estimand; diagnostic tests never silently
            change a mean question into a rank test.
        design : {"independent", "paired", "repeated", "clustered", "unknown"}, optional
            Researcher-declared study design (or StudyDesign enum). Supports independent groups,
            two-condition paired data, and multi-condition repeated-measures designs. Clustered
            designs are visible as unsupported rather than being silently reinterpreted.
        variable_types : dict[str, str], optional
            Explicit measurement types for columns: ``"continuous"``, ``"discrete"``,
            ``"nominal"``, ``"ordinal"``, or ``"identifier"``.
        data_dictionary : dict, optional
            Domain definitions, value labels, or notes for variables.
        description : str, optional
            Free-text description of the scientific research question.

        Regression and association parameters
        -------------------------------------
        predictors : sequence of str, optional
            Ordered collection of predictor column names for multiple regression
            (``objective="regression"``). Use `predictor` for a single explanatory variable
            or `predictors` for the complete ordered predictor set. When both are supplied,
            they must represent the same single predictor; conflicting values are rejected.
        controls : sequence of str, optional
            Ordered sequence of quantitative control variable column names for controlled
            linear association (``objective="association"``, ``estimand="partial_linear"``).
            Not used for regression; multiple regression predictors must be passed via `predictors`.
        covariance_type : {"classical", "HC3"}, optional
            Covariance matrix estimator for OLS regression (default: ``"classical"``).
            HC3 is recommended when heteroscedasticity is suspected.
        reference_levels : dict[str, Any], optional
            Baseline reference categories for categorical predictors in regression models.
        event_level : Any, optional
            Target outcome category for binary logistic regression (the event being modeled).
        association_measure : str, optional
            Specific association metric when objective is association.
        reference_value : float, optional
            Hypothesized population value for one-sample comparison tests.

        Paired and repeated-measures parameters
        ---------------------------------------
        unit_id : str, optional
            Unit, subject, or participant identifier column required for long-format
            paired and repeated-measures analyses.
        condition_order : tuple of Any, optional
            Ordered 2-tuple specifying the signed contrast direction (first minus second)
            in paired comparisons.

        Factorial-analysis parameters
        -----------------------------
        factor_a : str, optional
            First categorical factor for two-way factorial ANOVA.
        factor_b : str, optional
            Second categorical factor for two-way factorial ANOVA.
        factors : sequence of str, optional
            Alternative structured parameter accepting an ordered 2-element sequence
            ``(factor_a, factor_b)``. When both `factors` and `factor_a`/`factor_b` are
            supplied, they must be consistent (``factors == (factor_a, factor_b)``);
            conflicting values are rejected. For a dedicated interface, prefer
            :meth:`two_way_anova`.
        sum_of_squares : {"type2", "type3"}, optional
            Sum of squares formulation for factorial models (default: "type2").

        Reliability and ICC parameters
        ------------------------------
        items : sequence of str, optional
            Ordered scored column names for psychometric scale reliability (Cronbach's alpha).
            For a dedicated interface, prefer :meth:`reliability`.
        target : str, optional
            Subject, patient, or item identifier for Intraclass Correlation (ICC).
        rater : str, optional
            Observer, judge, or device identifier for Intraclass Correlation (ICC).
        value : str, optional
            Measurement score column for ICC. Mapped to `outcome` if `outcome` is omitted.
        model : {"one_way_random", "two_way_random", "two_way_mixed"}, optional
            ICC rater model specification.
        definition : {"absolute_agreement", "consistency"}, optional
            ICC definition: absolute agreement or consistency.
        unit : {"single", "average"}, optional
            ICC unit: single measurement or average of k ratings.
            For a dedicated interface, prefer :meth:`intraclass_correlation`.

        Workflow-control parameters
        ---------------------------
        draft : QuestionDraft, optional
            A previously returned draft from a `needs_input` workflow, updated with
            additional facts via :meth:`update_question`. Mutually exclusive with raw
            question arguments.
        specification : AnalysisSpecification, optional
            A pre-built, structured analysis specification. Mutually exclusive with raw
            question arguments.
        options : AnalysisOptions, optional
            Computational options such as significance level `alpha`, `confidence_level`,
            and `random_seed`.
        include_profile : bool, default False
            Whether to attach a full dataset profile to the returned workflow result.
        audit : bool, default True
            Whether to run result audits. Setting `audit=False` succeeds in analysis
            execution but marks the overall workflow status as ``partial`` because
            auditing was omitted.
        fingerprint : bool, default True
            Whether to calculate a SHA-256 fingerprint of the analyzed columns for provenance.
        title : str, optional
            Custom title for generated report and presentation views.
        include_figures : bool, default False
            Whether to embed static visual figure artifacts in the generated report.

        Returns
        -------
        ResearchWorkflowResult
            A structured result container. Check ``result.status`` (a :class:`WorkflowStatus` enum)
            to determine next actions:
            - ``WorkflowStatus.COMPLETED`` (``"completed"``): Analysis, interpretation,
              audit, and report were successfully executed.
            - ``WorkflowStatus.PARTIAL`` (``"partial"``): Analysis succeeded, but a secondary
              lifecycle component was omitted or incomplete (e.g., when ``audit=False``).
            - ``WorkflowStatus.NEEDS_INPUT`` (``"needs_input"``): Required scientific facts
              (such as design or estimand) are missing. Inspect ``result.missing_information``.
            - ``WorkflowStatus.DATA_LIMITED`` (``"data_limited"``): Data conditions (such as
              empty subsets, zero variance, or constant columns) prevent execution.
            - ``WorkflowStatus.UNSUPPORTED`` (``"unsupported"``): The declared combination of
              design, estimand, and data structure has no validated method in the library.
            - ``WorkflowStatus.FAILED`` (``"failed"``): The workflow encountered a fatal issue,
              such as an unusable numerical result, interpretation/report unavailability,
              or an audit contradiction. Inspect ``result.blockers``.

        Behavioral notes
        ----------------
        - **No file side effects**: Calling ``run()`` never writes files to disk. Use
          :func:`pyautostat.save_html`, :func:`pyautostat.save_pdf`, or
          :func:`pyautostat.save_docx` to export reports explicitly.
        - **Mutual exclusivity**: Supplying `draft` or `specification` is mutually exclusive
          with supplying raw question parameters (`objective`, `outcome`, etc.).
        - **Explicit designs**: Repeated measures, paired comparisons, and factorial designs
          are supported when explicitly declared. Unresolved designs yield ``needs_input``.
        - **Audit status**: Setting ``audit=False`` causes an otherwise successful workflow
          to return overall status ``WorkflowStatus.PARTIAL`` because scientific verification
          was bypassed.

        Examples
        --------
        >>> assistant = ResearchAssistant(df)
        >>> workflow = assistant.run(
        ...     objective="compare_groups",
        ...     outcome="score",
        ...     predictor="group",
        ...     estimand="mean",
        ...     design="independent",
        ... )
        >>> workflow.status.value
        'completed'
        """
        for name, value_check in (
            ("include_profile", include_profile),
            ("audit", audit),
            ("fingerprint", fingerprint),
            ("include_figures", include_figures),
        ):
            if not isinstance(value_check, bool):
                raise InvalidDataError(f"{name} must be a Boolean.")
        if draft is not None and specification is not None:
            raise InvalidDataError(
                "Pass either draft or specification, not both. Supply draft to resume "
                "a guided question workflow, or specification for a pre-built analysis "
                "specification."
            )
        if value is not None and outcome is None:
            outcome = value
        raw_values = (
            objective,
            outcome,
            predictor,
            predictors,
            items,
            controls,
            design,
            estimand,
            description,
            options,
            data_dictionary,
            variable_types,
            unit_id,
            condition_order,
            reference_value,
            covariance_type,
            reference_levels,
            event_level,
            association_measure,
            factor_a,
            factor_b,
            factors,
            sum_of_squares,
            target,
            rater,
            value,
            model,
            definition,
            unit,
        )
        if (draft is not None or specification is not None) and any(
            v is not None for v in raw_values
        ):
            raise InvalidDataError(
                "A supplied draft or specification cannot be combined with raw question "
                "parameters. To refine a workflow, update the draft via "
                "assistant.update_question(workflow.draft, ...) before passing "
                "draft=revised to run()."
            )
        if draft is not None:
            if not isinstance(draft, QuestionDraft):
                raise InvalidDataError("draft must be a QuestionDraft.")
            selected = prepare_question(self._analyzer.df, specification=draft.specification)
        elif specification is not None:
            selected = prepare_question(self._analyzer.df, specification=specification)
        else:
            selected = self.prepare_question(
                objective=objective,
                outcome=outcome,
                predictor=predictor,
                predictors=predictors,
                items=items,
                controls=controls,
                design=design,
                estimand=estimand,
                description=description,
                options=options,
                data_dictionary=data_dictionary,
                variable_types=variable_types,
                unit_id=unit_id,
                condition_order=condition_order,
                reference_value=reference_value,
                covariance_type=covariance_type,
                reference_levels=reference_levels,
                event_level=event_level,
                association_measure=association_measure,
                factor_a=factor_a,
                factor_b=factor_b,
                factors=factors,
                sum_of_squares=sum_of_squares,
                target=target,
                rater=rater,
                model=model,
                definition=definition,
                unit=unit,
            )

        profile = None
        is_descriptive = selected.specification.question.objective is Objective.DESCRIPTIVE
        if include_profile and not is_descriptive:
            profile = self.profile(data_dictionary=selected.specification.data_dictionary)

        def stop(
            status: WorkflowStatus,
            *,
            recommendation: Recommendation | None = None,
            analysis: AnalysisResult | None = None,
            missing_information=(),
            blockers=(),
            warnings=(),
        ) -> ResearchWorkflowResult:
            return ResearchWorkflowResult(
                status=status,
                specification=selected.specification,
                draft=selected,
                recommendation=recommendation,
                analysis=analysis,
                profile=profile,
                missing_information=tuple(missing_information),
                blockers=tuple(blockers),
                warnings=tuple(dict.fromkeys(warnings)),
            )

        if selected.status.value == "needs_input":
            return stop(
                WorkflowStatus.NEEDS_INPUT,
                missing_information=selected.missing_information,
                warnings=selected.warnings,
            )
        if selected.status.value == "data_limited":
            return stop(
                WorkflowStatus.DATA_LIMITED,
                blockers=selected.blockers,
                warnings=selected.warnings,
            )
        if selected.status.value == "unsupported":
            return stop(
                WorkflowStatus.UNSUPPORTED,
                blockers=selected.blockers,
                warnings=selected.warnings,
            )

        recommendation = self.recommend_test(selected)
        if recommendation.status.value == "needs_input":
            return stop(
                WorkflowStatus.NEEDS_INPUT,
                recommendation=recommendation,
                missing_information=recommendation.missing_information,
                warnings=(*selected.warnings, *recommendation.warnings),
            )
        if recommendation.status.value == "unsupported":
            blocked_status = (
                WorkflowStatus.DATA_LIMITED
                if _is_data_limited_recommendation(recommendation)
                else WorkflowStatus.UNSUPPORTED
            )
            return stop(
                blocked_status,
                recommendation=recommendation,
                blockers=recommendation.blockers,
                warnings=(*selected.warnings, *recommendation.warnings),
            )

        analysis = self.analyze(selected)
        if analysis.status.value != "available":
            return stop(
                WorkflowStatus.FAILED,
                recommendation=recommendation,
                analysis=analysis,
                blockers=analysis.warnings
                or ("The selected method could not produce a usable numerical result.",),
                warnings=(*selected.warnings, *recommendation.warnings, *analysis.warnings),
            )

        interpretation = self.interpret(analysis)
        report = self.report(
            analysis,
            interpretation=interpretation,
            title=title,
            include_figures=include_figures,
        )
        audit_result = self.audit(report, result=analysis) if audit else None
        reproducibility = self.reproducibility_record(analysis, fingerprint=fingerprint)
        if is_descriptive:
            profile_value = analysis.values.get("profile")
            profile = profile_value if isinstance(profile_value, dict) else None

        warnings = list(
            dict.fromkeys(
                [
                    *selected.warnings,
                    *recommendation.warnings,
                    *analysis.warnings,
                    *interpretation.warnings,
                    *report.to_dict()["warnings"],
                    *reproducibility.to_dict().get("warnings", []),
                ]
            )
        )
        blockers: tuple[str, ...] = ()
        if audit_result is not None and audit_result.status == "failed":
            status = WorkflowStatus.FAILED
            blockers = tuple(finding.explanation for finding in audit_result.findings)
        elif interpretation.status.value == "unavailable" or report.status == "unavailable":
            status = WorkflowStatus.FAILED
            blockers = ("The computed result could not be interpreted into a usable report.",)
        elif (
            interpretation.status.value == "partial"
            or report.status == "partial"
            or audit_result is None
            or audit_result.status == "incomplete"
        ):
            status = WorkflowStatus.PARTIAL
            if audit_result is None:
                warnings.append("Report auditing was disabled; no audit was performed.")
            elif audit_result.status == "incomplete":
                warnings.append("The report audit was incomplete; inspect its skipped checks.")
        else:
            status = WorkflowStatus.COMPLETED
        return ResearchWorkflowResult(
            status=status,
            specification=selected.specification,
            draft=selected,
            recommendation=recommendation,
            analysis=analysis,
            interpretation=interpretation,
            report=report,
            audit=audit_result,
            reproducibility=reproducibility,
            profile=profile,
            blockers=blockers,
            warnings=tuple(dict.fromkeys(warnings)),
        )

    def prepare_question(
        self,
        *,
        objective: str | Objective | None = None,
        outcome: str | None = None,
        predictor: str | None = None,
        predictors: list[str] | tuple[str, ...] | None = None,
        items: list[str] | tuple[str, ...] | None = None,
        controls: list[str] | tuple[str, ...] | None = None,
        design: str | StudyDesign | None = None,
        estimand: str | None = None,
        description: str | None = None,
        options: AnalysisOptions | None = None,
        data_dictionary: dict | None = None,
        variable_types: dict[str, str] | None = None,
        unit_id: str | None = None,
        condition_order: tuple[Any, Any] | None = None,
        reference_value: float | None = None,
        covariance_type: str | None = None,
        reference_levels: dict[str, Any] | None = None,
        event_level: Any | None = None,
        association_measure: str | None = None,
        factor_a: str | None = None,
        factor_b: str | None = None,
        factors: list[str] | tuple[str, ...] | None = None,
        sum_of_squares: str | None = None,
        target: str | None = None,
        rater: str | None = None,
        model: str | None = None,
        definition: str | None = None,
        unit: str | None = None,
        specification: AnalysisSpecification | None = None,
    ) -> QuestionDraft:
        """Prepare a serializable question; return focused requests for missing facts."""
        draft = prepare_question(
            self._analyzer.df,
            objective=objective,
            outcome=outcome,
            predictor=predictor,
            predictors=predictors,
            items=items,
            controls=controls,
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
            unit_id=unit_id,
            condition_order=condition_order,
            reference_value=reference_value,
            covariance_type=covariance_type,
            reference_levels=reference_levels,
            event_level=event_level,
            association_measure=association_measure,
            factor_a=factor_a,
            factor_b=factor_b,
            factors=factors,
            sum_of_squares=sum_of_squares,
            target=target,
            rater=rater,
            model=model,
            definition=definition,
            unit=unit,
            specification=specification,
        )
        if self._ledger is not None:
            payload = draft.specification.to_dict()
            self._ledger._record(
                "question_prepared",
                new_state=payload,
                references={"specification": content_reference("specification", payload)},
                metadata={"draft_status": draft.status.value},
            )
        return draft

    def update_question(
        self, draft: QuestionDraft, *, reason: str | None = None, **changes: Any
    ) -> QuestionDraft:
        """Reconstruct and revalidate an immutable draft after explicit answers."""
        if reason is not None and (not isinstance(reason, str) or not reason.strip()):
            raise InvalidDataError("A supplied revision reason must be non-empty text.")
        if not isinstance(draft, QuestionDraft):
            raise InvalidDataError("draft must be a QuestionDraft.")
        allowed = {
            "objective",
            "outcome",
            "predictor",
            "predictors",
            "items",
            "controls",
            "design",
            "estimand",
            "description",
            "options",
            "data_dictionary",
            "variable_types",
            "unit_id",
            "condition_order",
            "reference_value",
            "event_level",
            "association_measure",
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
                    "objective must be descriptive, compare_groups, compare_reference, "
                    "association, regression, or reliability."
                ) from exc
        else:
            new_objective = cast(Objective | None, changes.get("objective", previous.objective))
        switched = new_objective != previous.objective
        values: dict[str, Any] = {
            "objective": new_objective,
            "outcome": previous.outcome,
            "predictor": previous.predictor,
            "predictors": previous.predictors,
            "items": previous.items,
            "controls": previous.controls,
            "estimand": previous.estimand,
            "description": previous.description,
            "reference_value": previous.reference_value,
            "event_level": previous.event_level,
            "association_measure": previous.association_measure,
        }
        selected_design: StudyDesign | str = old.design
        if switched:
            values["predictor"] = None
            values["predictors"] = None
            values["items"] = None
            values["controls"] = None
            values["estimand"] = None
            values["reference_value"] = None
            values["event_level"] = None
            values["association_measure"] = None
            selected_design = StudyDesign.UNKNOWN
            selected_unit_id = None
            selected_condition_order = None
            if new_objective == Objective.DESCRIPTIVE:
                values["outcome"] = None
            if new_objective == Objective.RELIABILITY:
                values["outcome"] = None
        else:
            selected_unit_id = old.unit_id
            selected_condition_order = old.condition_order
        for key in (
            "outcome",
            "predictor",
            "predictors",
            "items",
            "controls",
            "estimand",
            "description",
            "reference_value",
            "event_level",
            "association_measure",
        ):
            if key in changes:
                values[key] = changes[key]
        if "design" in changes:
            selected_design = (
                StudyDesign.UNKNOWN if changes["design"] is None else changes["design"]
            )
            try:
                revised_design = StudyDesign(selected_design)
            except (TypeError, ValueError) as exc:
                raise InvalidDataError(
                    "design must be unknown, independent, paired, repeated, or clustered."
                ) from exc
            if revised_design not in (StudyDesign.PAIRED, StudyDesign.REPEATED):
                selected_unit_id = None
                selected_condition_order = None
        if changes.get("unit_id", selected_unit_id) is None:
            selected_condition_order = None
        if new_objective == Objective.DESCRIPTIVE and any(
            key in changes for key in ("predictor", "estimand", "design", "reference_value")
        ):
            raise InvalidDataError(
                "Descriptive questions do not use predictor, estimand, reference_value, or design."
            )
        revised = AnalysisSpecification(
            question=ResearchQuestion(**values),
            design=cast(StudyDesign, selected_design),
            options=changes.get("options", old.options),
            variable_metadata=old.variable_metadata,
            data_dictionary=changes.get("data_dictionary", old.data_dictionary),
            unit_id=changes.get("unit_id", selected_unit_id),
            condition_order=changes.get("condition_order", selected_condition_order),
        )
        updated = prepare_question(
            self._analyzer.df,
            specification=revised,
            variable_types=changes.get("variable_types"),
        )
        if self._ledger is not None:
            before = old.to_dict()
            after = updated.specification.to_dict()
            if before != after:
                changed = _changed_fields(before, after)
                self._ledger._record(
                    "specification_updated",
                    source="researcher",
                    previous_state=before,
                    new_state=after,
                    reason=reason,
                    references={"specification": content_reference("specification", after)},
                    metadata={"changed_fields": changed},
                )
        return updated

    def analysis_plan(
        self,
        draft_or_specification: QuestionDraft | AnalysisSpecification,
        *,
        sensitivity_scenarios: list[SensitivitySpecification]
        | tuple[SensitivitySpecification, ...]
        | None = None,
        meaningful_threshold: MeaningfulEffectThreshold | None = None,
        multiplicity_policy: str = "not_applicable",
        multiplicity_method: str | None = None,
        report_style: str = "general",
        previous_plan: StatisticalAnalysisPlan | None = None,
        reason: str | None = None,
    ) -> StatisticalAnalysisPlan:
        """Create or revise a plan without executing any numerical analysis."""
        if isinstance(draft_or_specification, QuestionDraft):
            draft = prepare_question(
                self._analyzer.df, specification=draft_or_specification.specification
            )
        elif isinstance(draft_or_specification, AnalysisSpecification):
            draft = prepare_question(self._analyzer.df, specification=draft_or_specification)
        else:
            raise InvalidDataError(
                "analysis_plan requires a QuestionDraft or AnalysisSpecification."
            )
        scenarios = tuple(sensitivity_scenarios or ())
        if any(not isinstance(item, SensitivitySpecification) for item in scenarios):
            raise InvalidDataError(
                "sensitivity_scenarios must contain SensitivitySpecification records."
            )
        if meaningful_threshold is not None and not isinstance(
            meaningful_threshold, MeaningfulEffectThreshold
        ):
            raise InvalidDataError(
                "meaningful_threshold must be a MeaningfulEffectThreshold or None."
            )
        if reason is not None and (not isinstance(reason, str) or not reason.strip()):
            raise InvalidDataError("A supplied plan-revision reason must be non-empty text.")
        recommendation = None
        if draft.status.value == "ready":
            recommendation = recommend_from_draft(self._analyzer.df, draft)
        status = (
            AnalysisPlanStatus.NEEDS_INPUT
            if draft.status.value == "needs_input"
            else AnalysisPlanStatus.READY
            if recommendation is not None and recommendation.status.value == "ready"
            else AnalysisPlanStatus.UNSUPPORTED
        )
        method_id = (
            recommendation.method_id
            if recommendation is not None and recommendation.status.value == "ready"
            else None
        )
        warnings = [*draft.warnings]
        limitations = [
            "A local plan is not proof of external preregistration or study validity.",
            "No automatic outlier deletion or missing-data modeling is planned.",
        ]
        if multiplicity_policy == "planned_method":
            limitations.append(
                "The named multiplicity procedure is recorded but is not executed by PyAutoStat."
            )
        rationale = None
        if recommendation is not None:
            rationale = recommendation.rationale
            warnings.extend(recommendation.warnings)
            limitations.extend(recommendation.blockers)
        else:
            limitations.extend(draft.blockers)
            limitations.extend(item.message for item in draft.missing_information)
        plan = StatisticalAnalysisPlan(
            specification=draft.specification,
            status=status,
            primary_method_id=method_id,
            method_rationale=rationale,
            effect_quantity=planned_quantity(method_id),
            confidence_interval_quantity=planned_interval_quantity(method_id),
            missing_data_policy="analysis-specific complete cases",
            exclusion_rule=(
                "Only rows missing variables required by the selected analysis are excluded."
            ),
            outlier_rule="No automatic outlier deletion.",
            planned_diagnostics=(
                (
                    "VIF",
                    "Breusch-Pagan",
                    "Jarque-Bera residual normality",
                    "Cook distance, leverage, and externally studentized residuals",
                    "condition number",
                )
                if method_id == "linear_regression"
                else ()
            ),
            sensitivity_scenarios=scenarios,
            meaningful_threshold=meaningful_threshold,
            multiplicity_policy=multiplicity_policy,
            multiplicity_method=multiplicity_method,
            report_style=report_style,
            planning_status=self._planning_status,
            created_after_analysis=self._analysis_has_executed,
            warnings=tuple(dict.fromkeys(warnings)),
            limitations=tuple(dict.fromkeys(limitations)),
            provenance={
                "tracking_enabled": self._ledger is not None,
                "local_record_only": True,
                "external_preregistration_verified": False,
                "timing": "after_analysis"
                if self._analysis_has_executed
                else "before_observed_analysis_in_this_assistant",
            },
        )
        if previous_plan is not None and not isinstance(previous_plan, StatisticalAnalysisPlan):
            raise InvalidDataError("previous_plan must be a StatisticalAnalysisPlan or None.")
        if self._ledger is not None:
            payload = plan.to_dict()
            previous = previous_plan.to_dict() if previous_plan is not None else None
            self._ledger._record(
                "analysis_plan_updated" if previous is not None else "analysis_plan_created",
                source="researcher",
                previous_state=previous,
                new_state=payload,
                reason=reason,
                references={"analysis_plan": content_reference("analysis_plan", payload)},
                metadata={
                    "status": plan.status.value,
                    "changed_fields": _changed_fields(previous, payload)
                    if previous is not None
                    else [],
                },
            )
        return plan

    def recommend_test(
        self,
        draft: QuestionDraft | None = None,
        *,
        specification: AnalysisSpecification | None = None,
    ) -> Recommendation:
        """Revalidate a question, then recommend a capability without executing it."""
        if (draft is None) == (specification is None):
            raise InvalidDataError("Provide either one QuestionDraft or specification.")
        if draft is not None and not isinstance(draft, QuestionDraft):
            raise InvalidDataError("draft must be a QuestionDraft.")
        selected_spec = draft.specification if draft is not None else specification
        validated = prepare_question(self._analyzer.df, specification=selected_spec)
        recommendation = recommend_from_draft(self._analyzer.df, validated)
        if self._ledger is not None:
            spec_payload = validated.specification.to_dict()
            rec_payload = recommendation.to_dict()
            self._ledger._record(
                "method_recommended",
                references={
                    "specification": content_reference("specification", spec_payload),
                    "recommendation": content_reference("recommendation", rec_payload),
                },
                metadata={
                    "status": recommendation.status.value,
                    "method_id": recommendation.method_id,
                },
            )
        return recommendation

    def analyze(
        self,
        draft: QuestionDraft | None = None,
        *,
        specification: AnalysisSpecification | None = None,
    ) -> AnalysisResult:
        """Execute a freshly validated question using its selected existing backend."""
        if (draft is None) == (specification is None):
            raise InvalidDataError("Provide either one QuestionDraft or specification.")
        if draft is not None and not isinstance(draft, QuestionDraft):
            raise InvalidDataError("draft must be a QuestionDraft.")
        selected_spec = draft.specification if draft is not None else specification
        assert selected_spec is not None
        result = execute_specification(self._analyzer, selected_spec)
        self._analysis_has_executed = result.method_id != "unselected"
        if self._ledger is not None:
            references = {"analysis": content_reference("analysis", result.to_dict())}
            if result.specification is not None:
                references["specification"] = content_reference(
                    "specification", result.specification.to_dict()
                )
            if result.recommendation is not None:
                references["recommendation"] = content_reference(
                    "recommendation", result.recommendation.to_dict()
                )
            self._ledger._record(
                "analysis_executed",
                references=references,
                metadata={
                    "status": result.status.value,
                    "method_id": result.method_id,
                    "analyzed_rows": result.sample_size,
                    "excluded_rows": result.excluded_rows,
                },
            )
        return result

    def interpret(self, result: AnalysisResult) -> InterpretationResult:
        """Explain a completed analysis from its recorded values and specification."""
        interpretation = InterpretationEngine().interpret(result)
        if self._ledger is not None:
            self._ledger._record(
                "interpretation_generated",
                references={
                    "analysis": content_reference("analysis", result.to_dict()),
                    "interpretation": content_reference("interpretation", interpretation.to_dict()),
                },
                metadata={"status": interpretation.status.value},
            )
        return interpretation

    def sensitivity_analysis(
        self,
        result: AnalysisResult,
        *,
        scenarios: list[SensitivitySpecification] | tuple[SensitivitySpecification, ...],
    ) -> SensitivityResult:
        """Execute exactly the declared scenarios, once each, in supplied order."""
        if not isinstance(result, AnalysisResult) or result.specification is None:
            raise InvalidDataError("sensitivity_analysis requires a result with its specification.")
        if result.status.value != "available":
            raise InvalidDataError("sensitivity_analysis requires an available base result.")
        if result.method_id in {"linear_regression", "logistic_regression"}:
            raise InvalidDataError(
                "Integrated regression sensitivity is not yet supported. Run separate explicit "
                "classical and HC3 specifications and compare the coefficient-level records; "
                "the current scalar sensitivity contract cannot represent a coefficient vector."
            )
        if result.method_id == "cronbach_alpha":
            raise InvalidDataError(
                "Integrated sensitivity is not supported for scale reliability; run separately "
                "declared scoring or item-set specifications and compare their complete records."
            )
        if not isinstance(scenarios, (list, tuple)) or not scenarios:
            raise InvalidDataError("scenarios must be a non-empty list or tuple.")
        if any(not isinstance(item, SensitivitySpecification) for item in scenarios):
            raise InvalidDataError("Every scenario must be a SensitivitySpecification.")
        names = [item.name for item in scenarios]
        if len(set(names)) != len(names):
            raise InvalidDataError("Sensitivity scenario names must be unique.")
        original_rows = result.metadata.get("sample", {}).get("original_rows")
        if original_rows is not None and original_rows != len(self._analyzer.df):
            raise InvalidDataError(
                "The base result row count does not match this assistant's dataset."
            )
        fingerprint_warning = None
        try:
            fingerprint = dataset_fingerprint(self._analyzer.df)
        except InvalidDataError as exc:
            fingerprint = None
            fingerprint_warning = f"Dataset fingerprint unavailable: {exc}"
        base_reference = content_reference("analysis", result.to_dict())
        plan = [item.to_dict() for item in scenarios]
        if self._ledger is not None:
            self._ledger._record(
                "sensitivity_plan_created",
                source="researcher",
                new_state=plan,
                references={"base_analysis": base_reference},
                metadata={
                    "scenario_order": names,
                    "note": "Local planning record; not external preregistration.",
                },
            )
        outputs: list[SensitivityScenarioResult] = []
        warnings: list[str] = [fingerprint_warning] if fingerprint_warning is not None else []
        base_spec = result.specification
        for scenario in scenarios:
            requested = scenario.method_id
            if self._ledger is not None:
                self._ledger._record(
                    "sensitivity_scenario_attempted",
                    source="researcher",
                    new_state=scenario.to_dict(),
                    references={"base_analysis": base_reference},
                    metadata={"name": scenario.name, "planning_status": scenario.planning_status},
                )
            incompatible_reason = _scenario_incompatibility(base_spec, scenario.specification)
            if incompatible_reason is not None:
                output = SensitivityScenarioResult(
                    name=scenario.name,
                    specification=scenario,
                    requested_method_id=requested,
                    method_id=None,
                    status=ScenarioStatus.INCOMPATIBLE,
                    comparability=Comparability.INCOMPATIBLE,
                    warnings=(incompatible_reason,),
                    error=incompatible_reason,
                )
                outputs.append(output)
                warnings.append(f"{scenario.name}: {incompatible_reason}")
                self._record_sensitivity_outcome(output, base_reference)
                continue
            method_id = requested
            if method_id is None:
                scenario_draft = prepare_question(
                    self._analyzer.df, specification=scenario.specification
                )
                recommendation = recommend_from_draft(self._analyzer.df, scenario_draft)
                method_id = (
                    recommendation.method_id if recommendation.status.value == "ready" else None
                )
                if method_id is None:
                    reason = "; ".join(recommendation.blockers) or (
                        recommendation.rationale or "No runnable method was selected."
                    )
                    output = SensitivityScenarioResult(
                        name=scenario.name,
                        specification=scenario,
                        requested_method_id=None,
                        method_id=None,
                        status=ScenarioStatus.UNAVAILABLE,
                        comparability=Comparability.UNAVAILABLE,
                        warnings=tuple(recommendation.warnings),
                        error=reason,
                    )
                    outputs.append(output)
                    warnings.append(f"{scenario.name}: {reason}")
                    self._record_sensitivity_outcome(output, base_reference)
                    continue
            if method_id in {"student_t", "one_way_anova"} and not any(
                "equal" in item.lower() and "variance" in item.lower()
                for item in scenario.assumptions
            ):
                reason = (
                    f"{method_id} requires an explicit equal-population-variance assumption "
                    "in the sensitivity scenario."
                )
                output = SensitivityScenarioResult(
                    name=scenario.name,
                    specification=scenario,
                    requested_method_id=requested,
                    method_id=method_id,
                    status=ScenarioStatus.INCOMPATIBLE,
                    comparability=Comparability.INCOMPATIBLE,
                    warnings=(reason,),
                    error=reason,
                )
                outputs.append(output)
                warnings.append(f"{scenario.name}: {reason}")
                self._record_sensitivity_outcome(output, base_reference)
                continue
            try:
                analysis = execute_selected_method(
                    self._analyzer, scenario.specification, method_id
                )
                if analysis.status.value != "available":
                    reason = analysis.warnings[-1] if analysis.warnings else "Scenario unavailable."
                    output = SensitivityScenarioResult(
                        name=scenario.name,
                        specification=scenario,
                        requested_method_id=requested,
                        method_id=method_id,
                        status=ScenarioStatus.UNAVAILABLE,
                        comparability=Comparability.UNAVAILABLE,
                        warnings=analysis.warnings,
                        error=reason,
                        analysis=analysis,
                    )
                else:
                    comparability = classify_comparability(result, analysis)
                    comparison = (
                        compare_same_estimand(result, analysis)
                        if comparability is Comparability.SAME_ESTIMAND
                        else None
                    )
                    if (
                        comparability is Comparability.SAME_ESTIMAND
                        and comparison is not None
                        and not comparison.get("available", False)
                        and "contrast" in str(comparison.get("reason", "")).lower()
                    ):
                        comparability = Comparability.INCOMPATIBLE
                    scenario_warning = list(analysis.warnings)
                    if comparability is Comparability.DIFFERENT_ESTIMAND:
                        scenario_warning.append(
                            "This scenario targets a different estimand and is not a direct "
                            "robustness replication of the base analysis."
                        )
                    elif comparability is Comparability.INCOMPATIBLE:
                        scenario_warning.append(
                            "This scenario changes the paired scientific comparison identity "
                            "or contrast orientation and is not a same-estimand robustness "
                            "comparison."
                        )
                    values = scenario_values(analysis)
                    missing_numerical = [
                        label
                        for label, value in (
                            ("primary estimate", values["primary_estimate"]),
                            ("p-value", values["p_value"]),
                        )
                        if value is None
                    ]
                    if missing_numerical:
                        reason = (
                            "Scenario returned no finite "
                            + " or ".join(missing_numerical)
                            + "; the numerical result is unreliable."
                        )
                        output = SensitivityScenarioResult(
                            name=scenario.name,
                            specification=scenario,
                            requested_method_id=requested,
                            method_id=analysis.method_id,
                            status=ScenarioStatus.FAILED,
                            comparability=Comparability.UNAVAILABLE,
                            warnings=tuple(dict.fromkeys([*scenario_warning, reason])),
                            error=reason,
                            analysis=analysis,
                            **values,
                        )
                    else:
                        output = SensitivityScenarioResult(
                            name=scenario.name,
                            specification=scenario,
                            requested_method_id=requested,
                            method_id=analysis.method_id,
                            status=ScenarioStatus.COMPLETED,
                            comparability=comparability,
                            comparison=comparison,
                            warnings=tuple(dict.fromkeys(scenario_warning)),
                            analysis=analysis,
                            **values,
                        )
                outputs.append(output)
                warnings.extend(f"{scenario.name}: {item}" for item in output.warnings)
                self._record_sensitivity_outcome(output, base_reference)
            except (PyAutoStatError, ValueError, TypeError, OverflowError, RuntimeError) as exc:
                output = SensitivityScenarioResult(
                    name=scenario.name,
                    specification=scenario,
                    requested_method_id=requested,
                    method_id=method_id,
                    status=ScenarioStatus.FAILED,
                    comparability=Comparability.UNAVAILABLE,
                    error=str(exc),
                    warnings=(f"Scenario execution failed: {exc}",),
                )
                outputs.append(output)
                warnings.extend(f"{scenario.name}: {item}" for item in output.warnings)
                self._record_sensitivity_outcome(output, base_reference)
        completed = sum(item.status is ScenarioStatus.COMPLETED for item in outputs)
        status = (
            SensitivityStatus.COMPLETE
            if completed == len(outputs)
            else SensitivityStatus.PARTIAL
            if completed
            else SensitivityStatus.UNAVAILABLE
        )
        same = [item for item in outputs if item.comparability is Comparability.SAME_ESTIMAND]
        different = [
            item for item in outputs if item.comparability is Comparability.DIFFERENT_ESTIMAND
        ]
        summary = {
            "base_method_id": result.method_id,
            "base_estimate_quantity": estimate_quantity(result.method_id),
            "declared_scenario_count": len(outputs),
            "completed_scenario_count": completed,
            "same_estimand_scenarios": len(same),
            "different_estimand_scenarios": len(different),
            "same_estimand_direction_consistent": (
                all(bool((item.comparison or {}).get("direction_consistent")) for item in same)
                if same
                else None
            ),
            "all_same_estimand_intervals_available": (
                all((item.comparison or {}).get("interval_overlap") is not None for item in same)
                if same
                else None
            ),
            "note": (
                "Comparisons are descriptive across the declared scenarios. P-values do not "
                "select, rank, or replace the primary analysis."
            ),
        }
        return SensitivityResult(
            base_result=result,
            scenario_results=tuple(outputs),
            status=status,
            comparison_summary=summary,
            warnings=tuple(dict.fromkeys(warnings)),
            provenance={
                "tracking_enabled": self._ledger is not None,
                "local_record_only": True,
                "external_preregistration_verified": False,
                "scenario_planning_statuses": [item.planning_status for item in scenarios],
            },
            reproducibility={
                "schema_version": 1,
                "base_analysis_reference": base_reference,
                "dataset_fingerprint": fingerprint,
                "scenario_order": names,
                "scenarios": plan,
                "actual_methods": [item.method_id for item in outputs],
                "scenario_statuses": [item.status.value for item in outputs],
                "random_seeds": [
                    item.specification.specification.options.random_seed for item in outputs
                ],
                "automatic_replay": False,
            },
        )

    def _record_sensitivity_outcome(
        self, output: SensitivityScenarioResult, base_reference: str
    ) -> None:
        if self._ledger is None:
            return
        event = (
            "sensitivity_scenario_completed"
            if output.status is ScenarioStatus.COMPLETED
            else "sensitivity_scenario_failed"
            if output.status is ScenarioStatus.FAILED
            else "sensitivity_scenario_unavailable"
        )
        self._ledger._record(
            event,
            references={
                "base_analysis": base_reference,
                "scenario": content_reference("sensitivity_scenario", output.to_dict()),
            },
            metadata={
                "name": output.name,
                "status": output.status.value,
                "comparability": output.comparability.value,
                "method_id": output.method_id,
            },
        )

    def practical_significance(
        self,
        result: AnalysisResult,
        *,
        threshold: MeaningfulEffectThreshold,
    ) -> PracticalSignificanceResult:
        """Assess one available quantity against a researcher-supplied threshold."""
        if not isinstance(result, AnalysisResult):
            raise InvalidDataError("practical_significance requires an AnalysisResult.")
        if result.method_id == "cronbach_alpha":
            raise InvalidDataError(
                "Cronbach's alpha is a reliability estimate, not an effect-size quantity for "
                "the practical-significance workflow."
            )
        if not isinstance(threshold, MeaningfulEffectThreshold):
            raise InvalidDataError("threshold must be a MeaningfulEffectThreshold.")
        payload = threshold.to_dict()
        if self._ledger is not None:
            self._ledger._record(
                "meaningful_threshold_declared",
                source="researcher",
                previous_state=self._last_meaningful_threshold,
                new_state=payload,
                reason=threshold.rationale,
                references={"analysis": content_reference("analysis", result.to_dict())},
                metadata={
                    "planning_status": threshold.planning_status,
                    "note": (
                        "Researcher declaration recorded locally; timing is not external proof."
                    ),
                },
            )
        self._last_meaningful_threshold = payload
        return assess_practical_significance(
            result, threshold, planning_status=threshold.planning_status
        )

    def plan_adherence(
        self,
        plan: StatisticalAnalysisPlan,
        result: AnalysisResult,
        *,
        reason: str | None = None,
        sensitivity: SensitivityResult | None = None,
        practical_significance: PracticalSignificanceResult | None = None,
    ) -> PlanAdherenceResult:
        """Compare recorded plan fields with a later result without judging conduct."""
        comparison = compare_plan_to_result(
            plan,
            result,
            reason=reason,
            sensitivity=sensitivity,
            practical_significance=practical_significance,
        )
        if self._ledger is not None:
            payload = comparison.to_dict()
            self._ledger._record(
                "plan_adherence_compared",
                references={
                    "analysis_plan": content_reference("analysis_plan", plan.to_dict()),
                    "analysis": content_reference("analysis", result.to_dict()),
                    "plan_adherence": content_reference("plan_adherence", payload),
                    **(
                        {"sensitivity": content_reference("sensitivity", sensitivity.to_dict())}
                        if sensitivity is not None
                        else {}
                    ),
                    **(
                        {
                            "practical_significance": content_reference(
                                "practical_significance", practical_significance.to_dict()
                            )
                        }
                        if practical_significance is not None
                        else {}
                    ),
                },
                metadata={"status": comparison.status},
            )
        return comparison

    def reporting_completeness(
        self, report: ResearchReport, *, style: str = "general"
    ) -> ReportingCompletenessResult:
        """Assess applicable reporting fields without scoring research quality."""
        result = assess_reporting_completeness(report, style=style)
        if self._ledger is not None:
            payload = result.to_dict()
            self._ledger._record(
                "reporting_completeness_assessed",
                references={
                    "report": content_reference("report", report.to_dict()),
                    "reporting_completeness": content_reference("reporting_completeness", payload),
                },
                metadata={"status": result.status, "style": result.style},
            )
        return result

    def session_snapshot(
        self,
        workflow: ResearchWorkflowResult,
        *,
        sensitivity: SensitivityResult | None = None,
        practical_significance: PracticalSignificanceResult | None = None,
        analysis_plan: StatisticalAnalysisPlan | None = None,
        study_planning: StudyPlanningResult | None = None,
        reporting_completeness: ReportingCompletenessResult | None = None,
    ) -> ResearchSessionSnapshot:
        """Serialize current records for UI-independent adapters without rerunning work."""
        snapshot = build_session_snapshot(
            workflow,
            sensitivity=sensitivity,
            practical_significance=practical_significance,
            analysis_plan=analysis_plan,
            study_planning=study_planning,
            reporting_completeness=reporting_completeness,
        )
        if self._ledger is not None:
            payload = snapshot.to_dict()
            self._ledger._record(
                "session_snapshot_created",
                references={"session_snapshot": content_reference("session_snapshot", payload)},
                metadata={"workflow_status": workflow.status.value},
            )
        return snapshot

    def report(
        self,
        result: AnalysisResult,
        *,
        interpretation: InterpretationResult | None = None,
        sensitivity: SensitivityResult | None = None,
        practical_significance: PracticalSignificanceResult | None = None,
        title: str | None = None,
        include_figures: bool = False,
    ) -> ResearchReport:
        """Assemble a structured ResearchReport object from recorded analysis results.

        Use this method when you need programmatic access to the report object,
        custom metadata, or direct export methods on the report. For exporting a
        completed workflow result directly to disk, prefer the top-level functions:
        :func:`pyautostat.save_html`, :func:`pyautostat.save_pdf`, or :func:`pyautostat.save_docx`.

        Parameters
        ----------
        result : AnalysisResult
            Executed statistical analysis result to report.
        interpretation : InterpretationResult, optional
            Deterministic interpretation findings.
        sensitivity : SensitivityResult, optional
            Sensitivity analysis scenario comparisons.
        practical_significance : PracticalSignificanceResult, optional
            Evaluation against researcher-defined practical thresholds.
        title : str, optional
            Custom report title.
        include_figures : bool, default False
            Whether to embed static visual figure artifacts in the report.

        Returns
        -------
        ResearchReport
            Structured, serializable report object. Does not write files to disk
            until an export method (e.g. ``save_html``) is called on it.
        """
        report = build_research_report(
            result,
            interpretation=interpretation,
            sensitivity=sensitivity,
            practical_significance=practical_significance,
            title=title,
            include_figures=include_figures,
        )
        if self._ledger is not None:
            report_reference = content_reference("report", report.to_dict())
            self._ledger._record(
                "report_generated",
                references={
                    "analysis": content_reference("analysis", result.to_dict()),
                    "interpretation": content_reference(
                        "interpretation", report.to_dict()["interpretation"]
                    ),
                    "report": report_reference,
                },
                metadata={"status": report.status},
            )
            ledger = self._ledger

            def on_save(format_name: str) -> None:
                ledger._record(
                    "report_exported",
                    references={"report": report_reference},
                    metadata={"format": format_name},
                )

            report._on_save = on_save
        return report

    def audit(
        self,
        report: ResearchReport,
        *,
        result: AnalysisResult | None = None,
        sensitivity: SensitivityResult | None = None,
        practical_significance: PracticalSignificanceResult | None = None,
        exports: dict[str, Any] | None = None,
    ) -> AuditResult:
        """Audit recorded analysis, report, and export consistency without rerunning tests.

        Uses :class:`pyautostat.StatisticalResultAuditor` to independently verify that
        reported numbers, degrees of freedom, p-values, effect sizes, and export artifacts
        are consistent with the primary analysis values.

        Parameters
        ----------
        report : ResearchReport
            The research report to audit.
        result : AnalysisResult, optional
            Primary analysis result to verify against.
        sensitivity : SensitivityResult, optional
            Sensitivity result to verify against.
        practical_significance : PracticalSignificanceResult, optional
            Practical significance result to verify against.
        exports : dict, optional
            Exported text or structure dictionaries to audit for consistency.

        Returns
        -------
        AuditResult
            Audit record with passed/warning/failed findings. Does not write files to disk.
        """
        audit = StatisticalResultAuditor().audit(
            report,
            result=result,
            sensitivity=sensitivity,
            practical_significance=practical_significance,
            exports=exports,
        )
        if self._ledger is not None:
            self._ledger._record(
                "audit_performed",
                references={
                    **audit.source_references,
                    "audit": content_reference("audit", audit.to_dict()),
                },
                metadata={"status": audit.status},
            )
        return audit

    def reproducibility_record(
        self,
        result: AnalysisResult,
        *,
        fingerprint: bool = True,
        sensitivity: SensitivityResult | None = None,
        practical_significance: PracticalSignificanceResult | None = None,
    ) -> ReproducibilityRecord:
        """Capture replay metadata and provenance for the current analysis and data.

        Constructs a :class:`pyautostat.ReproducibilityRecord` containing the exact
        specification, execution environment, package versions, random seeds, and
        optional SHA-256 dataset fingerprint needed to verify or replay the analysis.

        Parameters
        ----------
        result : AnalysisResult
            The completed analysis result to capture provenance for.
        fingerprint : bool, default True
            Whether to compute a SHA-256 fingerprint of the analyzed columns.
        sensitivity : SensitivityResult, optional
            Sensitivity scenario records to attach to the reproducibility record.
        practical_significance : PracticalSignificanceResult, optional
            Practical threshold records to attach to the reproducibility record.

        Returns
        -------
        ReproducibilityRecord
            Structured reproducibility record. Does not write files to disk.
        """
        return ReproducibilityRecord.from_result(
            result,
            data=self._analyzer.df,
            fingerprint=fingerprint,
            planning_status=self._planning_status,
            sensitivity=sensitivity,
            practical_significance=practical_significance,
        )


def _changed_fields(before: dict[str, Any], after: dict[str, Any], prefix: str = "") -> list[str]:
    fields: list[str] = []
    for key in sorted(before.keys() | after.keys()):
        path = f"{prefix}.{key}" if prefix else key
        left, right = before.get(key), after.get(key)
        if isinstance(left, dict) and isinstance(right, dict):
            fields.extend(_changed_fields(left, right, path))
        elif left != right:
            fields.append(path)
    return fields


_DATA_LIMIT_MESSAGES = (
    "fewer than two groups",
    "needs at least two usable outcomes",
    "not representable at this scale",
    "zero representable within-group variation",
    "no observed variation",
    "requires at least five usable observations",
    "at least two complete, varying numeric pairs",
    "at least three complete pairs",
    "variation in both variables",
    "at least two observed categories",
    "expected cell count below 5",
    "regression requires at least three complete cases",
    "outcome is constant",
    "has no variation in the analyzed sample",
    "design matrix is not full rank",
    "positive residual degrees of freedom",
    "at least two complete respondents",
    "at least two respondents complete",
    "total scale score has zero variance",
    "total score has zero variance",
    "nonfinite",
)


def _is_data_limited_recommendation(recommendation: Recommendation) -> bool:
    """Separate observed data insufficiency from unsupported scientific requests."""
    messages = " ".join(recommendation.blockers).lower()
    return any(fragment in messages for fragment in _DATA_LIMIT_MESSAGES)


def _scenario_incompatibility(
    base: AnalysisSpecification, scenario: AnalysisSpecification
) -> str | None:
    """Reject changes that would no longer be an analysis of the same question roles."""
    left, right = base.question, scenario.question
    if left.objective != right.objective:
        return "The scenario changes the research objective."
    if (
        left.outcome != right.outcome
        or left.predictor != right.predictor
        or left.predictors != right.predictors
        or left.items != right.items
    ):
        return "The scenario changes the outcome or predictor role."
    if base.design != scenario.design:
        return (
            "The scenario changes the declared study design; no independent-analysis method "
            "was substituted."
        )
    if (
        base.design in (StudyDesign.PAIRED, StudyDesign.REPEATED)
        and base.unit_id != scenario.unit_id
    ):
        return (
            "The scenario changes the unit_id, so it does not preserve the scientific "
            "unit definition."
        )
    return None
