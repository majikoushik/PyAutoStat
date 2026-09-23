# Phase 9: observed decisions, audit, and explicit replay

Phase 10 calls these same facilities from `ResearchAssistant.run()`. With tracking enabled before
the call, a successful raw-input workflow records one each of question preparation,
recommendation, execution, interpretation, report, and audit. The internal revalidation performed
by execution is not recorded as a second recommendation or user decision. Supplying an existing
draft/specification does not fabricate its earlier preparation history.

`workflow.reproducibility` is created from the already computed result. This captures settings and
an optional fingerprint but never calls `reproduce()`. Default report auditing renders all four
formats in memory and can therefore cost more than `audit=False`; it still performs no statistical
recalculation. No ledger, report, record, or audit is automatically saved.

Phase 9 adds optional local tracking and separate read-only consistency checks. It does not change the Phase 6 calculations, Phase 7 rules, or Phase 8 report schema. An audit pass means only that implemented checks found no discrepancy in the inspected records and exports. It does not validate study design, raw data authenticity, causal claims, or preregistration.

## Small workflow

```python
from pyautostat import ResearchAssistant, reproduce

assistant = ResearchAssistant(frame)
assistant.enable_tracking()  # optional; only subsequent calls are observed
draft = assistant.prepare_question(
    objective="compare_groups", outcome="score", predictor="group",
    estimand="mean", design="independent",
    variable_types={"score": "continuous"},
)
recommendation = assistant.recommend_test(draft)
result = assistant.analyze(draft)
interpretation = assistant.interpret(result)
report = assistant.report(result, interpretation=interpretation)
audit = assistant.audit(report)
record = assistant.reproducibility_record(result)
outcome = reproduce(record, data=frame)  # this call explicitly reruns Phase 6
```

`decision_ledger` is `None` until tracking is enabled. It then records `question_prepared`, `specification_updated`, `method_recommended`, `analysis_executed`, `interpretation_generated`, `report_generated`, `audit_performed`, and `report_exported` for explicit report file saves. In-memory `to_html()` and similar calls are not file exports. Failed executions record their actual `unavailable` status. Internal revalidation does not create a fictional user question event. Events have stable local sequence IDs and optional software timestamps; `enable_tracking(clock=...)` permits deterministic tests. Earlier events are returned as defensive copies. `update_question(draft, reason=...)` records a supplied reason, and omitted reasons remain `null`. A revision stores both specification states and the changed field paths. `declare_planning("planned" | "exploratory" | "unknown")` is an explicit researcher declaration. The default is `unknown`; a declaration is not externally verified preregistration.

`DecisionLedger.import_result(result)` records only the present import event with `history_status="unavailable"`. A final specification cannot reconstruct earlier decisions. Neither event order nor a local timestamp proves when an external decision was made.

## Content references and dataset fingerprint

`content_reference(kind, value)` computes SHA-256 over the ASCII kind, a zero byte, and UTF-8 JSON with sorted mapping keys, compact separators, preserved list order, JSON `null` for missing fields, and nonfinite values rejected. Existing schema-version fields are included when present. These references link recorded specification, recommendation, result, interpretation, report, and audit snapshots. The report schema itself remains version 1. A content reference detects changes to its snapshot under this serialization; it does not prove authorship or prevent replacement of both the snapshot and its reference.

`dataset_fingerprint(frame)` is optional and uses `pyautostat-dataframe-sha256-v1`. It hashes, in order: column names, dtypes and categorical categories/order; index class, dtype, names and per-row labels; row count; and each row's values in column order. Supported scalar types receive explicit tokens. Integers use decimal text, floats use hexadecimal representation, timestamps use ISO text, timedeltas use their recorded duration, and missing values use one token. `None`, `NaN`, `pd.NA`, and `NaT` are treated as missing in the same position; dtype metadata still distinguishes column types. Unsupported object values cause an explicit limitation rather than being stringified. No observations are stored in the fingerprint record. Hashing has a full-data cost and matching hashes are only consistency evidence under this algorithm. Small or guessable datasets can be attacked by guessing candidate inputs. Cross-version fingerprint identity is not promised.

## Auditor

`assistant.audit(report, result=None, exports=None)` reads the result snapshot retained by reports built through `assistant.report()`. A report reconstructed directly from a dictionary needs an explicitly supplied original `AnalysisResult` for a complete audit; otherwise status is `incomplete`. The auditor rebuilds the expected Phase 8 report from that source using the existing Phase 7 interpreter, without running a statistical test. It compares the specification, recommendation, executed method, sample and group accounting, numerical values, effect and interval records, interpretation, warnings, limitations, structured sections, and table cells. This is an exact comparison of recorded canonical snapshots; no display rounding can conceal a changed p-value. Existing Phase 7 validation governs interval semantics, including valid percentile-bootstrap intervals outside the original estimate and partial interpretations with a missing effect component.

When no `exports` argument is supplied, the auditor renders and checks all four current in-memory exports. A supplied mapping checks the actual provided content. JSON is parsed and compared structurally. CSV is parsed into table IDs, headers and cells; its intended formula escaping is compared with the canonical rendering. HTML and Markdown are compared with their exact deterministic canonical rendering. This detects changed method text, conclusions, warnings, values, or missing sections, but does not claim semantic validation of arbitrary rewritten prose or alternate templates. Unsupported export formats are listed as skipped and produce `incomplete` unless another check fails. Findings include a code, component, field path, expected/actual summary, and explanation; possible category labels and prose are redacted in findings. `passed`, `failed`, and `incomplete` describe inspection outcomes, not scientific quality.

## Reproducibility record and replay

`assistant.reproducibility_record(result, fingerprint=True)` captures the existing specification, selected method, a restricted expected-result projection, source references, actual Python/PyAutoStat/pandas/NumPy/SciPy and platform metadata, recorded researcher and effective seeds, bootstrap settings, confidence level, warnings, and a fingerprint of the assistant's current DataFrame when supported. This association is a local software record, not external authentication that the result originally came from that DataFrame. The projection excludes a raw descriptive profile; for such a profile it stores a content reference. Aggregate group labels and counts may still be sensitive. The record does not include the full DataFrame, credentials, environment variables, user paths, or all installed packages. A record alone cannot reproduce an analysis without separately supplied data.

`reproduce(record, data=frame)` first checks a recorded fingerprint. A mismatch stops before execution by default. `allow_changed_data=True` permits a labelled changed-data rerun, never a same-data reproduction. Without a usable fingerprint, replay can compare numbers but returns `unavailable` for the same-data claim. It revalidates the specification, requires the same method to remain selected, then calls the existing Phase 6 engine only on explicit request. Valid Student t and one-way ANOVA adapter records cannot be silently routed through a different guided selector choice. The original result is never replaced.

Replay compares configuration, method, status, row and group accounting, statistic, degrees of freedom, raw p-value, estimate, effect and interval, warnings, and available profile reference. IDs, labels, counts, configuration and boolean decisions use exact equality. Floating point result fields use `math.isclose(rel_tol=1e-10, abs_tol=1e-12)`; the p-value decision at the recorded alpha is checked separately. The outcome lists differing field paths and discloses changed runtime versions. Agreement on a significance category alone is insufficient. Different dependency versions can legitimately produce numerical differences, and matching results do not prove scientific validity.

`record.save_package(path, data_reference=None)` writes an explicitly requested ZIP containing `README.md`, `analysis_specification.json`, `analysis_reference.json`, and `reproducibility_record.json`. The optional data reference must be relative text. No raw observations or executable script are included, and no file is uploaded or run. Existing destinations are protected unless `overwrite=True`. The researcher must supply data separately for replay.
