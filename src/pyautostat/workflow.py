"""Structured result for the integrated, deterministic research workflow."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from typing import Any

from .audit import AuditResult
from .exceptions import InvalidDataError
from .interpretation import InterpretationResult
from .question_builder import QuestionDraft
from .reproducibility import ReproducibilityRecord
from .research_report import ResearchReport
from .results import AnalysisResult, MissingInformation, Recommendation
from .specifications import AnalysisSpecification, _json_value


def _serializable_profile(profile: dict[str, Any] | None) -> dict[str, Any] | None:
    """Copy dtype display values at the JSON boundary without changing the profile."""
    if profile is None:
        return None
    copied = dict(profile)
    overview = copied.get("overview")
    if isinstance(overview, dict):
        copied_overview = dict(overview)
        dtypes = copied_overview.get("dtypes")
        if isinstance(dtypes, dict):
            copied_overview["dtypes"] = {column: str(dtype) for column, dtype in dtypes.items()}
        copied["overview"] = copied_overview
    return copied


class WorkflowStatus(str, Enum):
    """Outcome of the integrated workflow, distinct from component statuses."""

    COMPLETED = "completed"
    PARTIAL = "partial"
    NEEDS_INPUT = "needs_input"
    DATA_LIMITED = "data_limited"
    UNSUPPORTED = "unsupported"
    FAILED = "failed"


@dataclass(frozen=True)
class ResearchWorkflowResult:
    """One inspectable view of a guided workflow without embedding source data."""

    status: WorkflowStatus
    specification: AnalysisSpecification
    draft: QuestionDraft
    recommendation: Recommendation | None = None
    analysis: AnalysisResult | None = None
    interpretation: InterpretationResult | None = None
    report: ResearchReport | None = None
    audit: AuditResult | None = None
    reproducibility: ReproducibilityRecord | None = None
    profile: dict[str, Any] | None = None
    missing_information: tuple[MissingInformation, ...] = ()
    blockers: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        try:
            object.__setattr__(self, "status", WorkflowStatus(self.status))
        except (TypeError, ValueError) as exc:
            raise InvalidDataError("Research workflow status is invalid.") from exc
        if not isinstance(self.specification, AnalysisSpecification):
            raise InvalidDataError("workflow specification must be an AnalysisSpecification.")
        if not isinstance(self.draft, QuestionDraft):
            raise InvalidDataError("workflow draft must be a QuestionDraft.")
        if any(not isinstance(item, MissingInformation) for item in self.missing_information):
            raise InvalidDataError("workflow missing_information contains an invalid record.")
        if self.status is WorkflowStatus.NEEDS_INPUT and not self.missing_information:
            raise InvalidDataError("A needs_input workflow must identify missing information.")
        if (
            self.status
            in {
                WorkflowStatus.DATA_LIMITED,
                WorkflowStatus.UNSUPPORTED,
                WorkflowStatus.FAILED,
            }
            and not self.blockers
        ):
            raise InvalidDataError("A blocked or failed workflow must explain the blocker.")
        if self.status in {WorkflowStatus.COMPLETED, WorkflowStatus.PARTIAL} and (
            self.analysis is None or self.interpretation is None or self.report is None
        ):
            raise InvalidDataError("A completed or partial workflow requires its computed outputs.")

    def to_dict(self) -> dict[str, Any]:
        """Return schema-versioned JSON-safe metadata; source rows are never included."""
        return _json_value(
            {
                "schema_version": 1,
                "status": self.status.value,
                "specification": self.specification.to_dict(),
                "draft": self.draft.to_dict(),
                "recommendation": self.recommendation.to_dict()
                if self.recommendation is not None
                else None,
                "analysis": self.analysis.to_dict() if self.analysis is not None else None,
                "interpretation": self.interpretation.to_dict()
                if self.interpretation is not None
                else None,
                "report": self.report.to_dict() if self.report is not None else None,
                "audit": self.audit.to_dict() if self.audit is not None else None,
                "reproducibility": self.reproducibility.to_dict()
                if self.reproducibility is not None
                else None,
                "profile": _serializable_profile(self.profile),
                "missing_information": [item.to_dict() for item in self.missing_information],
                "blockers": self.blockers,
                "warnings": self.warnings,
            }
        )

    def to_json(self) -> str:
        """Serialize without nonfinite JSON constants."""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2, allow_nan=False)

    def explain(self) -> str:
        """Return a human-readable summary of the entire workflow result.

        Aggregates status, method, hypothesis decision, effect size, confidence
        interval, assumptions, and limitations from the already-computed
        ``interpretation`` and ``analysis`` attributes into one readable text
        block.  No new statistics are calculated.

        Example usage::

            workflow = assistant.run(objective='compare_groups', ...)
            print(workflow.explain())

        Returns
        -------
        str
            A multi-section plain-text summary suitable for printing.
        """
        sep = "=" * 68
        thin = "-" * 68
        lines: list[str] = [sep]

        # ── Status & method ──────────────────────────────────────────────────
        status_val = self.status.value.upper().replace("_", " ")
        method_label = "Not yet determined"
        if self.analysis is not None:
            method_label = self.analysis.method_label
        elif self.recommendation is not None and self.recommendation.method_label:
            method_label = self.recommendation.method_label

        spec = self.specification
        outcome = spec.question.outcome or "not provided"
        predictor = spec.question.predictor or "not provided"
        lines.append(f" ANALYSIS RESULT | {method_label}")
        lines.append(sep)
        lines.append(f" Status    : {status_val}")

        if self.status in {WorkflowStatus.COMPLETED, WorkflowStatus.PARTIAL}:
            analysis_status_label = "Completed"
            workflow_status_label = (
                "Completed" if self.status is WorkflowStatus.COMPLETED else "Partial"
            )
        elif self.status is WorkflowStatus.FAILED:
            if self.analysis is not None and self.analysis.status.value == "available":
                analysis_status_label = "Completed"
            else:
                analysis_status_label = "Failed"
            workflow_status_label = "Failed"
        else:
            analysis_status_label = "Not run"
            workflow_status_label = {
                WorkflowStatus.NEEDS_INPUT: "Needs input",
                WorkflowStatus.DATA_LIMITED: "Data limited",
                WorkflowStatus.UNSUPPORTED: "Unsupported",
            }.get(self.status, self.status.value.replace("_", " ").title())

        lines.append(f" Analysis  : {analysis_status_label}")
        lines.append(f" Workflow  : {workflow_status_label}")

        if self.status is WorkflowStatus.PARTIAL:
            if self.audit is None:
                note = "Report auditing was disabled; no audit was performed."
            elif self.audit.status == "incomplete":
                note = "The report audit was incomplete; inspect its skipped checks."
            elif self.interpretation is not None and self.interpretation.status.value == "partial":
                note = "Statistical interpretation was partial."
            elif self.report is not None and self.report.status == "partial":
                note = "Report assembly was partial."
            else:
                note = "A workflow lifecycle component was omitted or incomplete."
            lines.append(f" Note      : {note}")
        elif self.status is WorkflowStatus.UNSUPPORTED:
            lines.append(" Note      : No substitute statistical method was run.")

        is_reliability = spec.question.objective is not None and (
            spec.question.objective.value == "reliability"
        )
        if is_reliability:
            lines.append(f" Items     : {', '.join(spec.question.items or ())}")
        else:
            lines.append(f" Outcome   : {outcome}")
        if spec.question.objective is not None and spec.question.objective.value not in {
            "compare_reference",
            "reliability",
        }:
            if spec.question.predictors:
                lines.append(f" Predictors: {', '.join(spec.question.predictors)}")
            else:
                lines.append(f" Predictor : {predictor}")
        if spec.question.reference_value is not None:
            lines.append(f" Reference : {spec.question.reference_value}")

        if self.blockers:
            lines.append(thin)
            lines.append(" BLOCKED")
            for blocker in self.blockers:
                lines.append(f"   BLOCKER: {blocker}")

        if self.missing_information:
            lines.append(thin)
            lines.append(" MISSING INFORMATION - the analysis cannot run yet")
            questions: dict[str, Any] = {item.field: item for item in self.draft.questions}
            if self.recommendation is not None:
                questions.update(
                    {
                        item["field"]: item
                        for item in self.recommendation.questions
                        if isinstance(item, dict) and isinstance(item.get("field"), str)
                    }
                )
            for item in self.missing_information:
                question = questions.get(item.field)
                lines.append(f"   Field '{item.field}': {item.message}")
                if question is not None:
                    explanation = (
                        question.get("explanation")
                        if isinstance(question, dict)
                        else question.explanation
                    )
                    options = (
                        tuple(
                            (option.get("value"), option.get("label"))
                            for option in question.get("options", ())
                            if isinstance(option, dict)
                        )
                        if isinstance(question, dict)
                        else question.options
                    )
                    if explanation:
                        lines.append(f"     Why: {explanation}")
                    if options:
                        choices = ", ".join(f"{value!r} ({label})" for value, label in options)
                        lines.append(f"     Choices: {choices}")
            fix_args = ", ".join(f"{item.field}=<your value>" for item in self.missing_information)
            lines.append(
                f"   Next step: revised = assistant.update_question(workflow.draft, {fix_args})"
            )
            lines.append("              workflow = assistant.run(draft=revised)")
        if self.analysis is not None:
            group_order = self.analysis.metadata.get("group_order")
            if isinstance(group_order, (list, tuple)) and group_order:
                groups = ", ".join(repr(group) for group in group_order)
                lines.append(f" Groups    : {groups}")
        if self.analysis is not None and self.analysis.sample_size is not None:
            excl = self.analysis.excluded_rows or 0
            lines.append(
                f" Sample    : {self.analysis.sample_size} rows analysed"
                + (f" ({excl} excluded)" if excl else "")
            )
            complete_pairs = self.analysis.metadata.get("sample", {}).get("complete_pairs")
            if isinstance(complete_pairs, int):
                lines.append(f" Pairs     : {complete_pairs} complete pairs")
            complete_units = self.analysis.metadata.get("sample", {}).get("complete_units")
            if isinstance(complete_units, int):
                lines.append(f" Units     : {complete_units} complete units")
        if self.analysis is not None and self.analysis.method_id == "cronbach_alpha":
            values = self.analysis.values
            interval = values.get("confidence_interval", {})
            lines.append(thin)
            lines.append(" RELIABILITY ESTIMATE")
            lines.append(f"   Cronbach's alpha={values.get('cronbach_alpha'):.4g}.")
            if isinstance(interval, dict) and interval.get("status") == "available":
                lines.append(
                    f"   Bootstrap CI [{interval.get('lower'):.4g}, "
                    f"{interval.get('upper'):.4g}]; valid resamples="
                    f"{interval.get('valid_resamples')}/{interval.get('requested_resamples')}."
                )
            lines.append(thin)
            lines.append(" ITEM DIAGNOSTICS")
            for item in values.get("item_statistics", []):
                lines.append(
                    f"   - {item.get('item')}: corrected item-total="
                    f"{item.get('corrected_item_total_correlation')}; "
                    f"alpha if deleted={item.get('alpha_if_deleted')}"
                )
            lines.append(thin)
            lines.append(" INTER-ITEM CORRELATIONS")
            lines.append(
                f"   Mean={values.get('mean_inter_item_correlation')}; negative pairs="
                f"{values.get('negative_inter_item_correlations', {}).get('count')}."
            )
        if self.analysis is not None and self.analysis.method_id == "intraclass_correlation":
            values = self.analysis.values
            interval = values.get("confidence_interval", {})
            f_test = values.get("f_test", {})
            lines.append(thin)
            lines.append(" RELIABILITY ESTIMATE")
            lines.append(
                f"   {values.get('variant')} ({values.get('notation')}): "
                f"ICC={values.get('intraclass_correlation'):.4g} "
                f"[{values.get('model')}, {values.get('definition')}, {values.get('unit')}]."
            )
            if isinstance(interval, dict) and interval.get("status") == "available":
                conf = int(interval.get("confidence_level", 0.95) * 100)
                lines.append(
                    f"   Analytical CI [{interval.get('lower'):.4g}, {interval.get('upper'):.4g}] "
                    f"({conf}%)."
                )
            if isinstance(f_test, dict) and f_test.get("p_value") is not None:
                lines.append(
                    f"   F({f_test.get('df1')}, {f_test.get('df2')})="
                    f"{f_test.get('statistic'):.4g}, "
                    f"p={f_test.get('p_value'):.4g} (H0: ICC = {f_test.get('null_value', 0)})."
                )
            lines.append(thin)
            lines.append(" PANEL DESIGN")
            n_obs = int(values.get("n_targets", 0) or 0) * int(values.get("n_raters", 0) or 0)
            lines.append(
                f"   Targets={values.get('n_targets')}, Raters={values.get('n_raters')}, "
                f"Observations={n_obs}."
            )
        if self.analysis is not None and self.analysis.method_id == "linear_regression":
            fit = self.analysis.values.get("model_fit", {})
            coefficients = self.analysis.values.get("coefficients", [])
            diagnostics = self.analysis.values.get("diagnostics", {})
            lines.append(thin)
            lines.append(" MODEL")
            lines.append(
                f"   Conditional mean of {outcome}; intercept included; covariance="
                f"{self.analysis.values.get('covariance_type')}."
            )
            lines.append(thin)
            lines.append(" MODEL FIT")
            lines.append(
                f"   R-squared={fit.get('r_squared'):.4g}; adjusted R-squared="
                f"{fit.get('adjusted_r_squared'):.4g}; residual SE="
                f"{fit.get('residual_standard_error'):.4g}."
            )
            lines.append(thin)
            lines.append(" COEFFICIENTS")
            for item in coefficients[:8]:
                interval = item.get("confidence_interval", {})
                lines.append(
                    f"   - {item.get('term')}: b={item.get('estimate'):.4g}, "
                    f"SE={item.get('standard_error'):.4g}, p={item.get('p_value'):.4g}, "
                    f"CI [{interval.get('lower'):.4g}, {interval.get('upper'):.4g}]"
                )
            if len(coefficients) > 8:
                lines.append(
                    f"   - {len(coefficients) - 8} additional terms are preserved in the result."
                )
            lines.append(thin)
            lines.append(" DIAGNOSTICS")
            lines.append(
                f"   Maximum VIF={diagnostics.get('vif', {}).get('maximum')}; "
                f"Breusch-Pagan p={diagnostics.get('breusch_pagan', {}).get('lm_p_value')}; "
                f"Jarque-Bera p={diagnostics.get('residual_normality', {}).get('p_value')}; "
                f"influence flags={diagnostics.get('influence', {}).get('flagged_count')}."
            )
        if self.analysis is not None and self.analysis.method_id == "logistic_regression":
            values = self.analysis.values
            fit = values.get("model_fit", {})
            lines.extend(
                [
                    thin,
                    " MODEL",
                    f"   Binary logistic model for {outcome}; event={values.get('event_level')!r}.",
                    " EVENT",
                    f"   Events={values.get('event_count')}; "
                    f"non-events={values.get('non_event_count')}.",
                    " SAMPLE",
                    f"   Complete observations={self.analysis.sample_size}; "
                    f"excluded={self.analysis.excluded_rows or 0}.",
                    " MODEL FIT",
                    f"   LR={fit.get('lr_statistic')}; p={fit.get('lr_p_value')}; "
                    f"McFadden pseudo-R-squared={fit.get('mcfadden_r2')}.",
                    " ODDS RATIOS",
                ]
            )
            for item in values.get("coefficients", []):
                if item.get("term_type") != "intercept":
                    lines.append(
                        f"   - {item.get('term_label')}: OR={item.get('odds_ratio')}; "
                        f"p={item.get('p_value')}"
                    )
            diagnostics = values.get("diagnostics", {})
            lines.extend(
                [
                    " DIAGNOSTICS",
                    f"   Converged={diagnostics.get('converged')}; covariance="
                    f"{diagnostics.get('covariance_type')}; condition number="
                    f"{diagnostics.get('condition_number')}.",
                ]
            )
        if self.analysis is not None and self.analysis.method_id == "mcnemar":
            values = self.analysis.values
            table = values.get("transition_table", {})
            lines.extend(
                [
                    thin,
                    " PAIRED BINARY COMPARISON",
                    " EVENT",
                    f"   Modeled event={values.get('event_level')!r}.",
                    " CONDITIONS",
                    f"   Ordered first-minus-second contrast={values.get('condition_order')}.",
                    " COMPLETE PAIRS",
                    f"   Complete pairs={table.get('total_pairs')}; "
                    f"incomplete units="
                    f"{self.analysis.metadata.get('sample', {}).get('incomplete_units')}.",
                    " EVENT PROPORTIONS",
                    f"   First={values.get('first_event_proportion')}; "
                    f"second={values.get('second_event_proportion')}.",
                    " DIFFERENCE",
                    f"   First minus second={values.get('primary_estimate')}.",
                    " DISCORDANT PAIRS",
                    f"   b={table.get('discordant_b')}; c={table.get('discordant_c')}.",
                    " TEST",
                    f"   Exact two-sided p={values.get('p_value')}.",
                ]
            )
        if self.analysis is not None and self.analysis.method_id in {
            "point_biserial_correlation",
            "kendall_tau_b",
            "partial_pearson_correlation",
        }:
            values = self.analysis.values
            heading = {
                "point_biserial_correlation": " ASSOCIATION",
                "kendall_tau_b": " MONOTONIC ASSOCIATION",
                "partial_pearson_correlation": " PARTIAL ASSOCIATION",
            }[self.analysis.method_id]
            lines.extend(
                [
                    thin,
                    heading,
                    f"   {values.get('estimate_name')}={values.get('primary_estimate')}; "
                    f"p={values.get('p_value')}.",
                ]
            )
            if self.analysis.method_id == "point_biserial_correlation":
                continuous = self.analysis.metadata.get("continuous_variable")
                lines.extend(
                    [
                        f" BINARY CODING: {values.get('binary_encoding')}.",
                        f" CONTINUOUS VARIABLE: {continuous}.",
                        f" r_pb: {values.get('primary_estimate')}.",
                        f" CI: {values.get('confidence_interval')}.",
                        f" p: {values.get('p_value')}.",
                    ]
                )
            if self.analysis.method_id == "kendall_tau_b":
                lines.extend(
                    [
                        f" tau-b: {values.get('primary_estimate')}.",
                        f" CI: {values.get('confidence_interval')}.",
                        f" p: {values.get('p_value')}.",
                        f" TIES: {values.get('ties')}.",
                    ]
                )
            if self.analysis.method_id == "partial_pearson_correlation":
                lines.extend(
                    [
                        f" VARIABLES: {outcome}, {predictor}.",
                        f" CONTROLS: {values.get('controls')}.",
                        f" partial r: {values.get('primary_estimate')}.",
                        f" CI: {values.get('confidence_interval')}.",
                        f" p: {values.get('p_value')}.",
                    ]
                )
        if self.analysis is not None and self.analysis.method_id == "repeated_measures_anova":
            values = self.analysis.values
            eff = values.get("effect_size", {}).get("value")
            df_vals = values.get("degrees_of_freedom")
            df_str = (
                f"({df_vals[0]}, {df_vals[1]})"
                if isinstance(df_vals, (list, tuple)) and len(df_vals) == 2
                else f"({df_vals})"
            )
            lines.extend(
                [
                    thin,
                    " REPEATED-MEASURES ANALYSIS",
                    " OMNIBUS TEST",
                    f"   F{df_str}={values.get('statistic')}; p={values.get('p_value')}; "
                    f"partial eta-squared={eff}.",
                ]
            )
            sphericity = values.get("sphericity", {})
            gg = values.get("greenhouse_geisser", {})
            if sphericity:
                lines.extend(
                    [
                        thin,
                        " SPHERICITY / CORRECTION",
                        f"   Mauchly's W={sphericity.get('mauchly_w')}; "
                        f"p={sphericity.get('p_value')}; status="
                        f"{sphericity.get('status')}.",
                        f"   Greenhouse-Geisser epsilon={gg.get('epsilon')}; "
                        f"corrected p={gg.get('corrected_p_value')}; "
                        f"policy applied={values.get('primary_inference')}.",
                    ]
                )
            summaries = values.get("condition_summaries", [])
            if summaries:
                lines.extend([thin, " CONDITION SUMMARIES"])
                for s in summaries:
                    lines.append(
                        f"   - {s.get('condition')}: mean={s.get('mean')}; "
                        f"std={s.get('std')}; median={s.get('median')}; n={s.get('n')}"
                    )
            pairwise = values.get("pairwise_comparisons", [])
            if pairwise:
                lines.extend([thin, " PAIRWISE FOLLOW-UP"])
                for pw in pairwise:
                    label = pw.get("orientation") or pw.get("contrast_id") or "contrast"
                    if pw.get("statistic") is None:
                        lines.append(
                            f"   - {label}: diff={pw.get('mean_difference')}; "
                            f"inference unavailable ({pw.get('reason') or 'zero variance'})"
                        )
                    else:
                        lines.append(
                            f"   - {label}: diff={pw.get('mean_difference')}; "
                            f"t({pw.get('degrees_of_freedom')})={pw.get('statistic')}; "
                            f"raw p={pw.get('raw_p_value')}; "
                            f"Holm p={pw.get('adjusted_p_value')}"
                        )
        if self.analysis is not None and self.analysis.method_id == "friedman_test":
            values = self.analysis.values
            eff = values.get("effect_size", {}).get("value")
            lines.extend(
                [
                    thin,
                    " REPEATED-MEASURES ANALYSIS",
                    " OMNIBUS TEST",
                    f"   Q({values.get('degrees_of_freedom')})={values.get('statistic')}; "
                    f"p={values.get('p_value')}; Kendall's W={eff}.",
                ]
            )
            summaries = values.get("condition_summaries", [])
            if summaries:
                lines.extend([thin, " CONDITION SUMMARIES"])
                for s in summaries:
                    lines.append(
                        f"   - {s.get('condition')}: median={s.get('median')}; "
                        f"IQR={s.get('iqr')}; mean={s.get('mean')}; n={s.get('n')}"
                    )
            pairwise = values.get("pairwise_comparisons", [])
            if pairwise:
                lines.extend([thin, " PAIRWISE FOLLOW-UP"])
                for pw in pairwise:
                    label = pw.get("orientation") or pw.get("contrast_id") or "contrast"
                    if pw.get("statistic") is None:
                        lines.append(
                            f"   - {label}: signed-rank inference unavailable "
                            f"({pw.get('reason') or 'insufficient nonzero differences'})"
                        )
                    else:
                        lines.append(
                            f"   - {label}: W={pw.get('statistic')}; "
                            f"raw p={pw.get('raw_p_value')}; "
                            f"Holm p={pw.get('adjusted_p_value')}"
                        )

        if self.recommendation is not None and self.recommendation.status.value == "ready":
            lines.append(thin)
            lines.append(" WHY THIS METHOD")
            lines.extend(
                f"   {line}" if line else ""
                for line in self.recommendation.explain().replace("?", "").splitlines()
            )

        # ── Interpretation sections ──────────────────────────────────────────
        interp = self.interpretation
        if interp is not None:
            if interp.hypothesis_interpretation:
                lines.append(thin)
                lines.append(
                    " OMNIBUS TEST"
                    if self.analysis is not None
                    and self.analysis.method_id
                    in {
                        "welch_anova",
                        "one_way_anova",
                        "kruskal_wallis",
                        "repeated_measures_anova",
                        "friedman_test",
                    }
                    else " HYPOTHESIS TEST"
                )
                lines.append(f"   {interp.hypothesis_interpretation.strip()}")
            if self.analysis is not None and self.analysis.method_id == "repeated_measures_anova":
                sphericity = self.analysis.values.get("sphericity")
                gg = self.analysis.values.get("greenhouse_geisser")
                if isinstance(sphericity, dict) and isinstance(gg, dict):
                    lines.append(thin)
                    lines.append(" SPHERICITY / CORRECTION")
                    w_stat = sphericity.get("statistic")
                    w_str = f"{w_stat:.4g}" if isinstance(w_stat, (int, float)) else "N/A"
                    p_stat = sphericity.get("p_value")
                    p_str = f"{p_stat:.4g}" if isinstance(p_stat, (int, float)) else "N/A"
                    lines.append(
                        f"   Mauchly W: {w_str}; p={p_str}; status={sphericity.get('status')}."
                    )
                    eps_val = gg.get("epsilon")
                    eps_str = f"{eps_val:.4g}" if isinstance(eps_val, (int, float)) else "N/A"
                    corr_p = gg.get("corrected_p_value")
                    corr_p_str = f"{corr_p:.4g}" if isinstance(corr_p, (int, float)) else "N/A"
                    lines.append(
                        f"   Greenhouse-Geisser epsilon: {eps_str}; corrected p={corr_p_str}."
                    )
                    lines.append(
                        f"   Primary inference: {self.analysis.values.get('primary_inference')}."
                    )
            if self.analysis is not None:
                summaries = self.analysis.values.get("group_summaries") or self.analysis.values.get(
                    "condition_summaries"
                )
                if isinstance(summaries, list) and summaries:
                    lines.append(thin)
                    lines.append(
                        " CONDITION SUMMARIES"
                        if self.analysis.values.get("condition_summaries")
                        else " GROUP SUMMARIES"
                    )
                    for item in summaries:
                        if not isinstance(item, dict):
                            continue
                        name = item.get("condition") or item.get("group")
                        n_c = (
                            item.get("n") if item.get("n") is not None else item.get("sample_size")
                        )
                        mean_str = (
                            f"mean={item['mean']:.4g}, " if item.get("mean") is not None else ""
                        )
                        med_str = (
                            f"median={item['median']:.4g}, "
                            if item.get("median") is not None
                            else ""
                        )
                        sd_str = (
                            f"SD={item['standard_deviation']:.4g}"
                            if item.get("standard_deviation") is not None
                            else ""
                        )
                        lines.append(
                            f"   - {name!r}: n={n_c}, {mean_str}{med_str}{sd_str}".rstrip(", ")
                        )
                pairwise = self.analysis.values.get("pairwise_comparisons")
                if isinstance(pairwise, list) and pairwise:
                    lines.append(thin)
                    lines.append(" PAIRWISE FOLLOW-UP")
                    lines.append(
                        f"   {self.analysis.metadata.get('pairwise_method')}; "
                        f"multiplicity control: "
                        f"{self.analysis.metadata.get('multiplicity_control')}."
                    )
                    for item in pairwise[:6]:
                        if not isinstance(item, dict):
                            continue
                        c1 = item.get("first_condition") or item.get("group1")
                        c2 = item.get("second_condition") or item.get("group2")
                        est_name = item.get("estimate_name") or "estimate"
                        est_val = item.get("estimate")
                        adj_p = item.get("adjusted_p_value")
                        dec = item.get("decision")
                        parts = [
                            p
                            for p in (
                                f"{est_name}={est_val:.4g}" if est_val is not None else "",
                                f"adjusted p={adj_p:.4g}" if adj_p is not None else "",
                                f"decision={dec}" if dec is not None else "",
                            )
                            if p
                        ]
                        lines.append(f"   - {c1!r} minus {c2!r}: {', '.join(parts)}")
                    if len(pairwise) > 6:
                        lines.append(
                            f"   - {len(pairwise) - 6} additional comparisons are preserved "
                            "in the structured result."
                        )
            if interp.effect_interpretation:
                lines.append(thin)
                lines.append(" EFFECT SIZE")
                lines.append(f"   {interp.effect_interpretation.strip()}")
            if interp.uncertainty_interpretation:
                lines.append(thin)
                lines.append(" CONFIDENCE INTERVAL")
                lines.append(f"   {interp.uncertainty_interpretation.strip()}")
            if interp.assumption_notes:
                lines.append(thin)
                lines.append(
                    " DIAGNOSTIC AND MISSINGNESS NOTES"
                    if self.analysis is not None
                    and self.analysis.method_id in {"cronbach_alpha", "intraclass_correlation"}
                    else " ASSUMPTIONS"
                )
                for note in interp.assumption_notes:
                    lines.append(f"   - {note}")
            if interp.limitations:
                lines.append(thin)
                lines.append(" LIMITATIONS")
                for lim in interp.limitations:
                    lines.append(f"   - {lim}")

        # ── Warnings ─────────────────────────────────────────────────────────
        if self.warnings:
            lines.append(thin)
            lines.append(" WARNINGS")
            for warning in self.warnings:
                lines.append(f"   ! {warning}")

        lines.append(sep)
        return "\n".join(lines)

    def __str__(self) -> str:
        """Delegate to :meth:`explain` for natural ``print()`` behaviour."""
        return self.explain()

    def apa_statement(self) -> str:
        """Generate a deterministic APA-oriented statistical statement from the completed workflow.

        Delegates to the computed AnalysisResult statement formatter without recalculation.
        """
        from .result_access import generate_workflow_statement

        return generate_workflow_statement(self)
