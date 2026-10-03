"""Adapters for research governance, audit, plans, completeness, and reproducibility."""

from __future__ import annotations

from ...analysis_plan import PlanAdherenceResult, StatisticalAnalysisPlan
from ...audit import AuditResult
from ...completeness import ReportingCompletenessResult
from ...decision_ledger import DecisionLedger
from ...reproducibility import ReproducibilityRecord, ReproductionOutcome
from ...session import ResearchSessionSnapshot
from ..formatting import (
    format_number,
    format_percent,
)
from ..models import (
    DisplayDiagnostic,
    DisplayMetric,
    DisplayRow,
    DisplayTable,
    TerminalView,
)


def adapt_statistical_analysis_plan(
    plan: StatisticalAnalysisPlan,
    detail: str = "standard",
) -> TerminalView:
    """Adapt a StatisticalAnalysisPlan into a TerminalView."""
    spec = plan.specification
    q = spec.question if spec else None

    design_metrics = (
        DisplayMetric("Plan Status", str(plan.status).upper(), role="method"),
        DisplayMetric("Planned Method", str(plan.primary_method_id)),
        DisplayMetric("Target Estimand", str(q.estimand if q else "Declared")),
        DisplayMetric(
            "Significance Alpha", format_number(spec.options.alpha if spec else 0.05, decimals=3)
        ),
        DisplayMetric(
            "Confidence Level",
            format_percent((spec.options.confidence_level if spec else 0.95) * 100),
        ),
        DisplayMetric(
            "Created Post-Hoc",
            "Yes (after analysis)" if plan.created_after_analysis else "No (a priori plan)",
        ),
    )

    tables: list[DisplayTable] = []
    # Policy elements table
    p_cols = ("Planning Element", "Declared Policy / Specification")
    p_rows = [
        DisplayRow(("Missing Data Policy", str(plan.missing_data_policy))),
        DisplayRow(("Exclusion Rule", str(plan.exclusion_rule))),
        DisplayRow(("Outlier Policy", str(plan.outlier_rule))),
        DisplayRow(("Multiplicity Control", str(plan.multiplicity_policy))),
        DisplayRow(("Report Style", str(plan.report_style))),
    ]
    tables.append(
        DisplayTable(title="PLANNED GOVERNANCE POLICIES", columns=p_cols, rows=tuple(p_rows))
    )

    if plan.meaningful_threshold:
        thresh = plan.meaningful_threshold
        th_cols = ("Threshold Element", "Specification")
        th_rows = [
            DisplayRow(("Quantity", str(thresh.quantity))),
            DisplayRow(
                (
                    "Minimum magnitude",
                    format_number(thresh.minimum_magnitude, decimals=2)
                    if hasattr(thresh, "minimum_magnitude")
                    else str(thresh),
                )
            ),
        ]
        if getattr(thresh, "unit", None):
            th_rows.append(DisplayRow(("Unit", str(thresh.unit))))
        if getattr(thresh, "direction", None):
            th_rows.append(DisplayRow(("Direction", str(thresh.direction))))
        if getattr(thresh, "planning_status", None):
            th_rows.append(DisplayRow(("Planning status", str(thresh.planning_status))))
        c_ord = getattr(thresh, "contrast_order", None)
        if c_ord and len(c_ord) >= 2:
            c_order = f"{c_ord[0]} - {c_ord[1]}"
            th_rows.append(DisplayRow(("Contrast", c_order)))
        if getattr(thresh, "rationale", None):
            th_rows.append(DisplayRow(("Rationale", str(thresh.rationale))))
        tables.append(
            DisplayTable(title="PRACTICAL THRESHOLD", columns=th_cols, rows=tuple(th_rows))
        )

    if plan.sensitivity_scenarios:
        s_cols = ("Scenario Name", "Planned Modification")
        s_rows = []
        for sc in plan.sensitivity_scenarios:
            s_name = getattr(sc, "name", None) or (
                sc.get("name", "") if isinstance(sc, dict) else ""
            )
            s_desc = (
                getattr(sc, "rationale", None)
                or getattr(sc, "description", None)
                or (sc.get("rationale", sc.get("description", "")) if isinstance(sc, dict) else "")
            )
            s_rows.append(DisplayRow((str(s_name), str(s_desc))))
        tables.append(
            DisplayTable(title="PLANNED SENSITIVITY SCENARIOS", columns=s_cols, rows=tuple(s_rows))
        )

    diagnostics_list = [
        DisplayDiagnostic(
            label="Pre-Analysis Protocol",
            status="PRE-SPECIFIED" if not plan.created_after_analysis else "RETROSPECTIVE",
            detail=plan.method_rationale
            or "Plan establishes immutable analytical constraints before evaluation.",
            severity="neutral" if not plan.created_after_analysis else "review",
        )
    ]

    limitations = [
        "A statistical analysis plan declares analytical protocol; "
        "it does not constitute registered external trial registration.",
        "Deviations from the declared plan must be disclosed in reporting.",
    ]

    compact_text = (
        f"Analysis Plan | Method={plan.primary_method_id} | "
        f"Status={plan.status} | PostHoc={plan.created_after_analysis}"
    )

    return TerminalView(
        title="Statistical Analysis Plan (Protocol)",
        subtitle=f"Planned: {plan.primary_method_id} (Status: {plan.status})",
        family="family.governance",
        design_metrics=design_metrics,
        key_metrics=(),
        tables=tuple(tables),
        diagnostics=tuple(diagnostics_list),
        interpretation=plan.method_rationale,
        limitations=tuple(limitations),
        warnings=plan.warnings,
        metadata={"plan": plan},
        compact_text=compact_text,
    )


def adapt_plan_adherence(
    result: PlanAdherenceResult,
    detail: str = "standard",
) -> TerminalView:
    """Adapt a PlanAdherenceResult into a TerminalView."""
    status_upper = str(result.status).upper()
    design_metrics = (
        DisplayMetric("Adherence Status", status_upper, role="method"),
        DisplayMetric("Evaluated Fields", f"{len(result.comparisons)} protocol fields"),
    )

    tables: list[DisplayTable] = []
    cols = ("Field", "Planned Policy", "Performed Analysis", "Match Status")
    rows: list[DisplayRow] = []
    for comp in result.comparisons:
        f_name = str(comp.get("field", "")).replace("_", " ").title()
        planned = str(comp.get("planned", ""))
        performed = str(comp.get("performed", ""))
        st = str(comp.get("status", "Adherent")).upper()
        rows.append(DisplayRow((f_name, planned, performed, st)))
    tables.append(
        DisplayTable(title="PROTOCOL ADHERENCE COMPARISON", columns=cols, rows=tuple(rows))
    )

    limitations = [
        "Plan adherence evaluates consistency between declared protocol and executed "
        "analysis; it does not judge scientific truth.",
    ]

    compact_text = f"Plan Adherence | Status={status_upper} | Fields={len(result.comparisons)}"

    return TerminalView(
        title="Plan Adherence Verification",
        subtitle=f"Overall Status: {status_upper}",
        family="family.governance",
        design_metrics=design_metrics,
        key_metrics=(),
        tables=tuple(tables),
        diagnostics=(),
        interpretation=result.reason,
        limitations=tuple(limitations),
        warnings=(),
        metadata={"adherence_result": result},
        compact_text=compact_text,
    )


def adapt_reporting_completeness(
    result: ReportingCompletenessResult,
    detail: str = "standard",
) -> TerminalView:
    """Adapt a ReportingCompletenessResult into a TerminalView."""
    status_upper = str(result.status).upper()
    style_name = str(result.style).upper()

    items = result.items
    n_present = sum(1 for it in items if getattr(it, "status", None) == "present")
    n_partial = sum(1 for it in items if getattr(it, "status", None) == "partial")
    n_missing = sum(1 for it in items if getattr(it, "status", None) == "missing")
    n_na = sum(1 for it in items if getattr(it, "status", None) == "not_applicable")

    design_metrics = (
        DisplayMetric("Guideline Style", style_name, role="method"),
        DisplayMetric("Overall Status", status_upper),
        DisplayMetric("Items Present", f"{n_present} items", role="result.ci"),
        DisplayMetric("Items Partial", f"{n_partial} items"),
        DisplayMetric("Items Missing", f"{n_missing} items", role="result.evidence"),
        DisplayMetric("Not Applicable", f"{n_na} items"),
    )

    tables: list[DisplayTable] = []
    cols = ("Component", "Reported Item", "Status", "Target Path")
    rows: list[DisplayRow] = []
    items_to_show = (
        items
        if detail == "full"
        else [it for it in items if getattr(it, "status", None) in ("missing", "partial")][:12]
    )
    for it in items_to_show:
        comp = str(getattr(it, "component", getattr(it, "section", ""))).title()
        desc = str(getattr(it, "description", getattr(it, "name", "")))
        st = str(getattr(it, "status", "")).upper()
        path = str(getattr(it, "target_path", ""))
        rows.append(DisplayRow((comp, desc, st, path)))
    t_title = (
        f"COMPLETENESS ITEMS ({style_name})"
        if detail == "full"
        else f"COMPLETENESS CUES (showing issues; use detail='full' for all {len(items)})"
    )
    tables.append(DisplayTable(title=t_title, columns=cols, rows=tuple(rows)))

    limitations = [
        "Completeness checking organizes recorded structural evidence; "
        "it does NOT certify study quality, journal compliance, or publication readiness.",
    ]
    limitations.extend(result.limitations)

    compact_text = (
        f"Reporting Completeness | Style={style_name} | "
        f"Present={n_present}/{len(items)} | Missing={n_missing}"
    )

    return TerminalView(
        title="Reporting Completeness Audit",
        subtitle=f"Guideline: {style_name} (Status: {status_upper})",
        family="family.governance",
        design_metrics=design_metrics,
        key_metrics=(),
        tables=tuple(tables),
        diagnostics=(),
        interpretation=None,
        limitations=tuple(limitations),
        warnings=(),
        metadata={"completeness_result": result},
        compact_text=compact_text,
    )


def adapt_audit(
    result: AuditResult,
    detail: str = "standard",
) -> TerminalView:
    """Adapt an AuditResult into a TerminalView."""
    res_status = getattr(result, "status", "")
    status_raw = str(getattr(res_status, "value", res_status)).strip().lower()

    if status_raw == "passed":
        status_upper = "PASSED"
        role_color = "status.success"
    elif status_raw == "failed":
        status_upper = "FAILED"
        role_color = "status.error"
    else:
        status_upper = "INCOMPLETE"
        role_color = "status.warning"

    findings = result.findings
    n_failures = sum(1 for f in findings if getattr(f, "severity", "") == "failure")
    n_warnings = sum(1 for f in findings if getattr(f, "severity", "") == "warning")
    n_pass = sum(1 for f in findings if getattr(f, "severity", "") == "pass")

    design_metrics = (
        DisplayMetric("Audit Outcome", status_upper, role=role_color),
        DisplayMetric("Checked Components", f"{len(result.checked_components)} components"),
        DisplayMetric("Passed Invariants", f"{n_pass} checks", role="status.success"),
        DisplayMetric("Warnings", f"{n_warnings} warnings", role="status.warning"),
        DisplayMetric(
            "Failures",
            f"{n_failures} failures",
            role="status.error" if n_failures > 0 else "default",
        ),
        DisplayMetric("Skipped Checks", f"{len(result.skipped_checks)} checks"),
    )

    tables: list[DisplayTable] = []
    cols = ("Severity", "Component", "Invariant / Check", "Detail")
    rows: list[DisplayRow] = []

    # In standard mode, show warnings and failures; full mode all
    findings_to_show = (
        findings
        if detail == "full"
        else [f for f in findings if getattr(f, "severity", "") in ("failure", "warning")]
    )
    for f in findings_to_show:
        sev = str(getattr(f, "severity", "")).upper()
        comp = str(getattr(f, "component", "")).title()
        code = str(getattr(f, "code", getattr(f, "check", "")))
        msg = str(getattr(f, "message", ""))
        rows.append(DisplayRow((sev, comp, code, msg)))

    if rows:
        t_title = (
            "AUDIT FINDINGS"
            if detail == "full"
            else f"AUDIT FINDINGS (issues: {len(rows)}; use detail='full' for all {len(findings)})"
        )
        tables.append(DisplayTable(title=t_title, columns=cols, rows=tuple(rows)))

    diagnostics_list = [
        DisplayDiagnostic(
            label="Internal Consistency",
            status=status_upper,
            detail="Verifies mathematical and methodological consistency across result records.",
            severity="success"
            if status_upper == "PASS"
            else ("error" if status_upper == "FAIL" else "warning"),
        )
    ]

    limitations = [
        "Audit verifies internal consistency, contract invariants, and numerical agreement; "
        "it does NOT certify empirical truth or study quality.",
    ]

    compact_text = (
        f"Scientific Audit | Outcome={status_upper} | "
        f"Invariants={n_pass} Pass, {n_warnings} Warn, {n_failures} Fail"
    )

    return TerminalView(
        title="Scientific Consistency Audit",
        subtitle=f"Outcome: {status_upper}",
        family="family.audit",
        design_metrics=design_metrics,
        key_metrics=(),
        tables=tuple(tables),
        diagnostics=tuple(diagnostics_list),
        interpretation=None,
        limitations=tuple(limitations),
        warnings=(),
        metadata={"audit_result": result},
        compact_text=compact_text,
    )


def adapt_reproducibility(
    record_or_outcome: ReproducibilityRecord | ReproductionOutcome,
    detail: str = "standard",
) -> TerminalView:
    """Adapt a ReproducibilityRecord or ReproductionOutcome into a TerminalView."""
    if isinstance(record_or_outcome, ReproductionOutcome):
        status_upper = str(record_or_outcome.status).upper()
        data_status = str(record_or_outcome.data_status).upper()
        design_metrics: tuple[DisplayMetric, ...] = (
            DisplayMetric(
                "Replay Outcome",
                status_upper,
                role="status.success" if status_upper == "EXACT_MATCH" else "status.warning",
            ),
            DisplayMetric("Data Match", data_status),
            DisplayMetric("Differing Fields", f"{len(record_or_outcome.differing_fields)} fields"),
        )
        tables: list[DisplayTable] = []
        if record_or_outcome.differing_fields:
            cols = ("Field", "Discrepancy Details")
            rows = [
                DisplayRow((str(f), "Recorded and recomputed values differ."))
                for f in record_or_outcome.differing_fields
            ]
            tables.append(
                DisplayTable(title="REPRODUCTION DISCREPANCIES", columns=cols, rows=tuple(rows))
            )
        compact_text = f"Reproduction | Status={status_upper} | Data={data_status}"
        title = "Deterministic Reproduction Result"
        subtitle = f"Outcome: {status_upper}"
    else:
        payload = (
            getattr(record_or_outcome, "payload", {})
            if hasattr(record_or_outcome, "payload")
            else record_or_outcome.to_dict()
        )
        pkg_ver = payload.get("package_version", "Unavailable")
        py_ver = payload.get("python_version", "Unavailable")
        m_id = payload.get("method_id", "Unavailable")
        design_metrics = (
            DisplayMetric("Method ID", str(m_id)),
            DisplayMetric("PyAutoStat Version", str(pkg_ver)),
            DisplayMetric("Python Runtime", str(py_ver)),
            DisplayMetric(
                "Fingerprint Available", "Yes" if payload.get("dataset_fingerprint") else "No"
            ),
        )
        tables = []
        compact_text = f"Reproducibility Record | Method={m_id} | PyAutoStat={pkg_ver}"
        title = "Analysis Reproducibility Record"
        subtitle = f"Method: {m_id}"

    limitations = [
        "Dataset fingerprints detect data changes but do not authenticate "
        "participant identity or legal custody.",
    ]

    return TerminalView(
        title=title,
        subtitle=subtitle,
        family="family.reproducibility",
        design_metrics=design_metrics,
        key_metrics=(),
        tables=tuple(tables),
        diagnostics=(),
        interpretation=None,
        limitations=tuple(limitations),
        warnings=(),
        metadata={"record": record_or_outcome},
        compact_text=compact_text,
    )


def adapt_decision_ledger(
    ledger: DecisionLedger,
    detail: str = "standard",
) -> TerminalView:
    """Adapt a DecisionLedger into a TerminalView."""
    events = ledger.events
    status_str = ledger.history_status

    design_metrics = (
        DisplayMetric("Ledger Status", str(status_str).upper(), role="method"),
        DisplayMetric("Total Events", f"{len(events)} recorded decisions"),
    )

    tables: list[DisplayTable] = []
    cols = ("#", "Event Type", "Summary / Details")
    rows: list[DisplayRow] = []
    events_to_show = events if detail == "full" else events[-10:]
    start_num = 1 if detail == "full" else max(1, len(events) - len(events_to_show) + 1)
    for idx, ev in enumerate(events_to_show, start=start_num):
        if isinstance(ev, dict):
            etype = str(ev.get("event_type", ev.get("type", "")))
            desc = str(ev.get("description", ev.get("summary", "")))
            if not desc:
                meta = ev.get("metadata", {})
                if isinstance(meta, dict) and meta:
                    desc = ", ".join(f"{k}={v}" for k, v in meta.items() if k != "changed_fields")
                if not desc:
                    desc = f"Sequence {ev.get('sequence', idx)}"
        else:
            etype = str(getattr(ev, "event_type", getattr(ev, "type", "")))
            desc = str(getattr(ev, "description", getattr(ev, "summary", str(ev))))
        rows.append(DisplayRow((str(idx), etype, desc)))
    t_title = (
        "DECISION LEDGER EVENTS"
        if detail == "full"
        else f"RECENT DECISIONS (last {len(rows)} of {len(events)})"
    )
    tables.append(DisplayTable(title=t_title, columns=cols, rows=tuple(rows)))

    limitations = [
        "Local decision ledger entries record sequential workflow decisions; "
        "they do not prove external preregistration or authenticated provenance.",
    ]

    compact_text = f"Decision Ledger | Events={len(events)} | Status={status_str}"

    return TerminalView(
        title="Research Decision Ledger",
        subtitle=f"{len(events)} sequential events recorded",
        family="family.governance",
        design_metrics=design_metrics,
        key_metrics=(),
        tables=tuple(tables),
        diagnostics=(),
        interpretation=None,
        limitations=tuple(limitations),
        warnings=(),
        metadata={"ledger": ledger},
        compact_text=compact_text,
    )


def adapt_session_snapshot(
    snapshot: ResearchSessionSnapshot,
    detail: str = "standard",
) -> TerminalView:
    """Adapt a ResearchSessionSnapshot into a TerminalView."""
    payload = snapshot.payload
    schema_ver = payload.get("schema_version", 1)
    status = payload.get("status", "recorded")
    q = payload.get("question", {})
    method = payload.get("method_id", payload.get("analysis", {}).get("method_id", "Not run"))

    design_metrics = (
        DisplayMetric("Snapshot Version", f"v{schema_ver}", role="method"),
        DisplayMetric("Workflow Status", str(status).upper()),
        DisplayMetric("Method", str(method)),
        DisplayMetric("Objective", str(q.get("objective", "Unspecified"))),
    )

    tables: list[DisplayTable] = []
    cols = ("Component", "Included Status", "Record Type")
    rows = [
        DisplayRow(
            (
                "Question Specification",
                "Present" if "question" in payload else "None",
                "AnalysisSpecification",
            )
        ),
        DisplayRow(
            (
                "Statistical Analysis",
                "Present" if "analysis" in payload else "None",
                "AnalysisResult",
            )
        ),
        DisplayRow(
            (
                "Deterministic Interpretation",
                "Present" if "interpretation" in payload else "None",
                "InterpretationResult",
            )
        ),
        DisplayRow(
            ("Scientific Audit", "Present" if "audit" in payload else "None", "AuditResult")
        ),
        DisplayRow(
            (
                "Decision Ledger",
                "Present" if "decision_ledger" in payload else "None",
                "DecisionLedger",
            )
        ),
    ]
    tables.append(DisplayTable(title="SESSION CONTENTS", columns=cols, rows=tuple(rows)))

    limitations = [
        "Session snapshots serialize research state independently of the active UI; "
        "raw participant data is never included.",
    ]

    compact_text = f"Session Snapshot | v{schema_ver} | Status={status} | Method={method}"

    return TerminalView(
        title="Research Session Snapshot",
        subtitle=f"Workflow Status: {status}",
        family="family.governance",
        design_metrics=design_metrics,
        key_metrics=(),
        tables=tuple(tables),
        diagnostics=(),
        interpretation=None,
        limitations=tuple(limitations),
        warnings=tuple(payload.get("warnings", ())),
        metadata={"snapshot": snapshot},
        compact_text=compact_text,
    )
