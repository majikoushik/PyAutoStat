"""Checks for observed decisions, report consistency, and explicit replay."""

import csv
import io
import json
import math
import zipfile
from copy import deepcopy
from dataclasses import replace
from datetime import date, datetime, time, timezone

import pandas as pd
import pytest

from pyautostat import (
    AnalysisOptions,
    DecisionLedger,
    InvalidDataError,
    ReproducibilityRecord,
    ResearchAssistant,
    ResearchReport,
    StatisticalAnalyzer,
    reproduce,
)
from pyautostat.provenance import content_reference, dataset_fingerprint


@pytest.fixture
def case():
    frame = pd.DataFrame(
        {
            "group": ["A"] * 8 + ["B"] * 8 + ["B"],
            "score": [float(i) + 0.5 for i in range(8)]
            + [float(i) + 2.5 for i in range(8)]
            + [math.nan],
            "private_id": [f"secret-{i}" for i in range(17)],
        }
    )
    assistant = ResearchAssistant(frame)
    draft = assistant.prepare_question(
        objective="compare_groups",
        outcome="score",
        predictor="group",
        estimand="mean",
        design="independent",
        variable_types={"score": "continuous"},
    )
    result = assistant.analyze(draft)
    assert result.method_id == "welch_t"
    report = assistant.report(result)
    return frame, assistant, draft, result, report


def test_tracking_records_only_observed_events_and_preserves_revisions(case, tmp_path):
    frame, assistant, draft, _, _ = case
    before = frame.copy(deep=True)
    assert assistant.decision_ledger is None
    ledger = assistant.enable_tracking(clock=lambda: datetime(2024, 1, 2, tzinfo=timezone.utc))
    assert assistant.enable_tracking() is ledger
    assert ledger.events == ()
    assert assistant.planning_status == "unknown"

    prepared = assistant.prepare_question(specification=draft.specification)
    revised = assistant.update_question(
        prepared,
        options=AnalysisOptions(alpha=0.01),
        reason="Use the stated significance threshold.",
    )
    assistant.recommend_test(revised)
    result = assistant.analyze(revised)
    interpreted = assistant.interpret(result)
    report = assistant.report(result, interpretation=interpreted)
    audit = assistant.audit(report)
    report.save_html(tmp_path / "report.html")
    assert audit.status == "passed"
    events = ledger.events
    assert [event["event_type"] for event in events] == [
        "question_prepared",
        "specification_updated",
        "method_recommended",
        "analysis_executed",
        "interpretation_generated",
        "report_generated",
        "audit_performed",
        "report_exported",
    ]
    assert [event["sequence"] for event in events] == list(range(1, 9))
    assert events[1]["metadata"]["changed_fields"] == ["options.alpha"]
    assert events[1]["reason"] == "Use the stated significance threshold."
    assert events[1]["previous_state"]["options"]["alpha"] == 0.05
    assert events[1]["new_state"]["options"]["alpha"] == 0.01
    assert events[0]["timestamp"] == "2024-01-02T00:00:00+00:00"
    assert events[0]["timestamp_kind"] == "local_software_event"
    assert events[5]["references"]["analysis"] == events[3]["references"]["analysis"]
    assert events[3]["metadata"]["analyzed_rows"] == 16
    assert events[3]["metadata"]["excluded_rows"] == 1
    assert events[7]["metadata"]["format"] == "html"
    events[0]["new_state"]["options"]["alpha"] = 0.99
    assert ledger.events[0]["new_state"]["options"]["alpha"] == 0.05
    assert ledger.to_json() == ledger.to_json()
    assert "secret-0" not in ledger.to_json()
    pd.testing.assert_frame_equal(frame, before)


def test_missing_reason_planning_and_legacy_history_are_honest(case):
    _, assistant, draft, result, _ = case
    ledger = assistant.enable_tracking(clock=lambda: "fixed-test-time")
    updated = assistant.update_question(draft, options=AnalysisOptions(alpha=0.02))
    assert ledger.events[0]["reason"] is None
    assert updated.specification.options.alpha == 0.02
    assistant.declare_planning("planned", reason="Declared by researcher.")
    assert assistant.planning_status == "planned"
    assert ledger.events[1]["source"] == "researcher"
    assert ledger.events[1]["new_state"] == "planned"
    assert "preregister" not in ledger.to_json().lower()
    imported = DecisionLedger.import_result(result, clock=lambda: "imported-now")
    assert imported.history_status == "unavailable"
    assert [event["event_type"] for event in imported.events] == ["existing_analysis_imported"]
    assert imported.events[0]["timestamp"] == "imported-now"
    with pytest.raises(InvalidDataError, match="planning status"):
        assistant.declare_planning("verified_preregistered")


def test_failed_execution_is_recorded_as_unavailable(case):
    _, assistant, draft, _, _ = case
    ledger = assistant.enable_tracking(clock=lambda: "t")
    paired = assistant.update_question(draft, design="paired")
    result = assistant.analyze(paired)
    assert result.status.value == "unavailable"
    assert ledger.events[-1]["event_type"] == "analysis_executed"
    assert ledger.events[-1]["metadata"]["status"] == "unavailable"


def test_fingerprint_content_and_unsupported_values():
    frame = pd.DataFrame({"a": [1.0, None, 3.0], "b": ["x", "y", "z"]})
    baseline = dataset_fingerprint(frame)
    assert baseline == dataset_fingerprint(frame.copy(deep=True))
    changed = frame.copy(deep=True)
    changed.loc[0, "a"] = 2.0
    assert dataset_fingerprint(changed) != baseline
    changed = frame.copy(deep=True)
    changed.loc[1, "a"], changed.loc[2, "a"] = 3.0, None
    assert dataset_fingerprint(changed) != baseline
    assert dataset_fingerprint(frame[["b", "a"]]) != baseline
    assert dataset_fingerprint(frame.iloc[::-1]) != baseline
    assert dataset_fingerprint(frame.set_index("b")) != baseline
    categorical = frame.copy(deep=True)
    categorical["b"] = pd.Categorical(categorical["b"], ordered=True)
    assert dataset_fingerprint(categorical) != baseline
    with pytest.raises(InvalidDataError, match="cannot encode"):
        dataset_fingerprint(pd.DataFrame({"x": [object()]}))
    assert content_reference("analysis", {"a": 1, "b": 2}) == content_reference(
        "analysis", {"b": 2, "a": 1}
    )


def _altered(report, result, path, value):
    payload = report.to_dict()
    target = payload
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    return ResearchReport(payload, source_result=result)


@pytest.mark.parametrize(
    ("path", "value", "code"),
    [
        (("analysis", "method_id"), "mann_whitney_u", "METHOD_MISMATCH"),
        (("sections", "research_question", "objective"), "association", "SPECIFICATION_MISMATCH"),
        (("sections", "research_question", "estimand"), "distribution", "SPECIFICATION_MISMATCH"),
        (("sections", "methods", "declared_design"), "paired", "SPECIFICATION_MISMATCH"),
        (("sections", "dataset", "group_order"), ["B", "A"], "GROUP_ORDER_MISMATCH"),
        (("sections", "dataset", "original_rows"), 99, "SAMPLE_COUNT_MISMATCH"),
        (("sections", "dataset", "analyzed_rows"), 15, "SAMPLE_COUNT_MISMATCH"),
        (("sections", "dataset", "excluded_rows"), 0, "SAMPLE_COUNT_MISMATCH"),
        (("sections", "results", "p_value"), 0.9, "PVALUE_MISMATCH"),
        (("sections", "results", "test_statistic"), 99.0, "STATISTIC_MISMATCH"),
        (
            ("sections", "interpretation", "conclusion"),
            "No effect exists.",
            "INTERPRETATION_MISMATCH",
        ),
        (("status",), "unavailable", "REPORT_VALUE_MISMATCH"),
    ],
)
def test_auditor_finds_deliberate_field_mismatches(case, path, value, code):
    _, assistant, _, result, report = case
    changed = _altered(report, result, path, value)
    finding = assistant.audit(changed).findings
    assert any(item.code == code and item.field.startswith(".".join(path)) for item in finding)


def test_auditor_checks_effects_intervals_counts_warnings_and_tables(case):
    _, assistant, _, result, report = case
    payload = report.to_dict()
    payload["sections"]["results"]["effect_size"]["value"] = 2.0
    payload["sections"]["results"]["effect_size"]["name"] = "wrong name"
    payload["sections"]["results"]["confidence_interval"]["quantity"] = "Cohen's d"
    payload["sections"]["results"]["confidence_interval"]["lower"] = -99.0
    payload["sections"]["results"]["confidence_interval"]["level"] = 0.90
    payload["sections"]["dataset"]["group_sizes"][0]["size"] = 99
    payload["limitations"] = []
    payload["warnings"] = []
    payload["tables"][0]["rows"][0][1]["value"] = 99
    audit = assistant.audit(ResearchReport(payload), result=result)
    assert audit.status == "failed"
    codes = {item.code for item in audit.findings}
    assert "EFFECT_SIZE_MISMATCH" in codes
    assert "INTERVAL_QUANTITY_MISMATCH" in codes
    assert "GROUP_COUNT_MISMATCH" in codes
    assert "MISSING_REQUIRED_WARNING" in codes
    assert "TABLE_MISMATCH" in codes
    assert all("secret-" not in json.dumps(item.to_dict()) for item in audit.findings)
    assert report.to_dict()["sections"]["dataset"]["group_sizes"][0]["size"] == 8


def test_audit_passes_valid_partial_and_bootstrap_outside_point(case, monkeypatch):
    _, assistant, _, result, report = case
    assert assistant.audit(report).status == "passed"
    values = deepcopy(result.values)
    d = values["effect_size"]["value"]
    values["effect_size"]["confidence_interval"]["lower"] = d + 0.5
    values["effect_size"]["confidence_interval"]["upper"] = d + 1.0
    changed_result = replace(result, values=values)
    changed_report = assistant.report(changed_result)
    assert assistant.audit(changed_report).status == "passed"
    partial_result = replace(result, values={**result.values, "effect_size": None})
    partial_report = assistant.report(partial_result)
    assert partial_report.status == "partial"
    assert assistant.audit(partial_report).status == "passed"

    def forbidden(*args, **kwargs):
        raise AssertionError("auditing must not rerun a statistical test")

    monkeypatch.setattr(StatisticalAnalyzer, "hypothesis_tests", forbidden)
    assert assistant.audit(report).status == "passed"
    assert assistant.audit(ResearchReport(report.to_dict())).status == "incomplete"


def test_export_audit_reads_actual_json_csv_html_and_markdown(case):
    _, assistant, _, result, report = case
    content = {
        "json": report.to_json(),
        "csv": report.to_csv_tables(),
        "html": report.to_html(),
        "markdown": report.to_markdown(),
    }
    original = deepcopy(content)
    assert assistant.audit(report, exports=content).status == "passed"

    changed = deepcopy(content)
    payload = json.loads(changed["json"])
    payload["analysis"]["values"]["p_value"] = 0.9
    payload["sections"]["dataset"]["original_rows"] = 99
    changed["json"] = json.dumps(payload)
    audit = assistant.audit(report, exports=changed)
    assert audit.status == "failed"
    assert any(item.code == "PVALUE_MISMATCH" for item in audit.findings)
    assert any(item.code == "SAMPLE_COUNT_MISMATCH" for item in audit.findings)

    changed = deepcopy(content)
    rows = list(csv.reader(io.StringIO(changed["csv"]["statistical_results"])))
    rows[1][1] = "99"
    stream = io.StringIO()
    csv.writer(stream).writerows(rows)
    changed["csv"]["statistical_results"] = stream.getvalue()
    assert assistant.audit(report, exports=changed).status == "failed"
    changed["csv"]["statistical_results"] = "wrong,heading\n"
    assert assistant.audit(report, exports=changed).status == "failed"
    changed["csv"].pop("statistical_results")
    assert assistant.audit(report, exports=changed).status == "failed"

    changed = deepcopy(content)
    changed["html"] = changed["html"].replace("welch_t", "mann_whitney_u", 1)
    assert assistant.audit(report, exports=changed).status == "failed"
    changed = deepcopy(content)
    conclusion = report.to_dict()["sections"]["interpretation"]["conclusion"]
    changed["markdown"] = changed["markdown"].replace(conclusion, "An unsupported conclusion.", 1)
    assert assistant.audit(report, exports=changed).status == "failed"
    changed = deepcopy(content)
    warning = report.to_dict()["warnings"][0]
    changed["html"] = changed["html"].replace(warning, "", 1)
    assert assistant.audit(report, exports=changed).status == "failed"
    changed = deepcopy(content)
    payload = json.loads(changed["json"])
    payload["sections"]["results"]["p_value"] += 0.0000001
    changed["json"] = json.dumps(payload)
    assert any(
        item.code == "PVALUE_MISMATCH" for item in assistant.audit(report, exports=changed).findings
    )
    assert assistant.audit(report, exports={"pdf": b"unsupported"}).status == "incomplete"
    assert content == original
    assert result.to_dict() == report.to_dict()["analysis"]


def test_reproducibility_record_replay_and_dataset_mismatch(case, tmp_path):
    frame, assistant, _, result, _ = case
    before = frame.copy(deep=True)
    record = assistant.reproducibility_record(result)
    payload = record.to_dict()
    assert payload["specification"] == result.specification.to_dict()
    assert payload["method_id"] == result.method_id
    assert payload["stochastic"]["researcher_seed"] is None
    assert payload["stochastic"]["effective_seed"] == 0
    assert payload["stochastic"]["bootstrap_resamples"] == 499
    assert payload["environment"]["pandas"] == pd.__version__
    assert payload["environment"]["python"]
    assert "secret-0" not in record.to_json()
    assert ReproducibilityRecord.from_dict(payload).to_json() == record.to_json()
    outcome = reproduce(record, data=frame)
    assert outcome.status == "reproduced"
    assert outcome.data_status == "same_data"
    assert outcome.differing_fields == ()

    changed = frame.copy(deep=True)
    changed.loc[0, "score"] = 999.0
    mismatch = reproduce(record, data=changed)
    assert mismatch.status == "mismatch"
    assert mismatch.replay_reference is None
    assert mismatch.differing_fields == ("dataset_fingerprint",)
    rerun = reproduce(record, data=changed, allow_changed_data=True)
    assert rerun.status == "mismatch"
    assert rerun.data_status == "changed_data"
    assert rerun.replay_reference is not None

    package = record.save_package(tmp_path / "reproduction.zip")
    with zipfile.ZipFile(package) as archive:
        assert set(archive.namelist()) == {
            "README.md",
            "analysis_specification.json",
            "analysis_reference.json",
            "reproducibility_record.json",
        }
        assert "secret-0" not in "".join(
            archive.read(name).decode("utf-8") for name in archive.namelist()
        )
    with pytest.raises(InvalidDataError, match="exists"):
        record.save_package(package)
    with pytest.raises(InvalidDataError, match="relative"):
        record.save_package(tmp_path / "other.zip", data_reference="../private.csv")
    pd.testing.assert_frame_equal(frame, before)


def test_replay_discloses_environment_and_numerical_discrepancies(case):
    frame, assistant, _, result, _ = case
    payload = assistant.reproducibility_record(result).to_dict()
    payload["environment"]["scipy"] = "unavailable-version"
    payload["expected"]["values"]["p_value"] = 0.0
    replay = reproduce(ReproducibilityRecord.from_dict(payload), data=frame)
    assert replay.status == "mismatch"
    assert "result.values.p_value" in replay.differing_fields
    assert "result.values.p_value_decision" in replay.differing_fields
    assert any("scipy" in warning for warning in replay.warnings)

    payload["method_id"] = "unknown_method"
    payload["expected"]["method_id"] = "unknown_method"
    unsupported = reproduce(ReproducibilityRecord.from_dict(payload), data=frame)
    assert unsupported.status == "unavailable"
    assert unsupported.differing_fields == ("method_id",)


def test_fingerprint_unavailable_prevents_same_data_claim(case):
    frame, assistant, _, result, _ = case
    record = assistant.reproducibility_record(result, fingerprint=False)
    replay = reproduce(record, data=frame)
    assert replay.status == "unavailable"
    assert replay.data_status == "fingerprint_unavailable"
    assert replay.replay_reference is not None


def test_reproduction_record_preserves_explicit_seed_and_unsupported_fingerprint(case):
    frame, _, _, result, _ = case
    assistant = ResearchAssistant(frame)
    draft = assistant.prepare_question(
        objective="compare_groups",
        outcome="score",
        predictor="group",
        estimand="mean",
        design="independent",
        variable_types={"score": "continuous"},
        options=AnalysisOptions(random_seed=42),
    )
    seeded = assistant.analyze(draft)
    record = assistant.reproducibility_record(seeded)
    assert record.to_dict()["stochastic"]["researcher_seed"] == 42
    assert record.to_dict()["stochastic"]["effective_seed"] == 42
    assert reproduce(record, data=frame).status == "reproduced"

    unsupported = ReproducibilityRecord.from_result(result, data=pd.DataFrame({"x": [object()]}))
    assert unsupported.to_dict()["dataset_fingerprint"] is None
    assert any(
        "fingerprint unavailable" in item.lower() for item in unsupported.to_dict()["warnings"]
    )


def test_package_bytes_are_stable_and_reject_private_absolute_references(case, tmp_path):
    _, assistant, _, result, _ = case
    record = assistant.reproducibility_record(result)
    first = record.save_package(tmp_path / "one.zip", data_reference="data/study.csv")
    second = record.save_package(tmp_path / "two.zip", data_reference="data/study.csv")
    assert first.read_bytes() == second.read_bytes()
    with pytest.raises(InvalidDataError, match="relative"):
        record.save_package(tmp_path / "bad.zip", data_reference="C:\\private\\study.csv")
    with pytest.raises(InvalidDataError, match="relative"):
        record.save_package(tmp_path / "bad.zip", data_reference="data\nsecret.csv")


def test_partial_and_unavailable_reports_are_audited_in_their_actual_state(case):
    _, assistant, _, result, _ = case
    unavailable = replace(result, status="unavailable")
    report = assistant.report(unavailable)
    assert report.status == "unavailable"
    assert assistant.audit(report).status == "passed"
    changed = _altered(report, unavailable, ("status",), "complete")
    assert assistant.audit(changed).status == "failed"

    pearson_frame = pd.DataFrame(
        {"hours": [1.0, 2.0, 3.0, 4.0, 5.0], "score": [2.0, 4.0, 3.0, 6.0, 7.0]}
    )
    pearson_assistant = ResearchAssistant(pearson_frame)
    draft = pearson_assistant.prepare_question(
        objective="association",
        outcome="hours",
        predictor="score",
        estimand="linear",
        design="independent",
        variable_types={"hours": "continuous", "score": "continuous"},
    )
    pearson_report = pearson_assistant.report(pearson_assistant.analyze(draft))
    assert pearson_report.status == "partial"
    assert pearson_assistant.audit(pearson_report).status == "passed"


def test_csv_formula_escaping_is_expected_by_auditor():
    frame = pd.DataFrame(
        {"group": ["=SUM(1,1)"] * 8 + ["+SUM(2,2)"] * 8, "score": [float(i) for i in range(16)]}
    )
    assistant = ResearchAssistant(frame)
    draft = assistant.prepare_question(
        objective="compare_groups",
        outcome="score",
        predictor="group",
        estimand="mean",
        design="independent",
        variable_types={"score": "continuous"},
    )
    report = assistant.report(assistant.analyze(draft))
    csv_tables = report.to_csv_tables()
    assert "'=SUM" in csv_tables["group_sizes"]
    assert assistant.audit(report, exports={"csv": csv_tables}).status == "passed"


def test_fingerprint_handles_typed_index_dates_durations_and_rejects_bad_inputs():
    frame = pd.DataFrame(
        {
            "flag": [True, False],
            "number": [1, 2],
            "when": [pd.Timestamp("2024-01-01"), pd.NaT],
            "duration": [pd.Timedelta(days=1), pd.NaT],
            "birthday": [date(2001, 1, 1), date(2002, 2, 2)],
            "clock": [time(8, 30), time(9, 45)],
        },
        index=pd.MultiIndex.from_tuples([("P", 1), ("P", 2)], names=["kind", "number"]),
    )
    reference = dataset_fingerprint(frame)
    assert reference == dataset_fingerprint(frame.copy(deep=True))
    changed = frame.copy(deep=True)
    changed.iloc[0, changed.columns.get_loc("clock")] = time(10, 0)
    assert dataset_fingerprint(changed) != reference
    with pytest.raises(InvalidDataError, match="ASCII identifier"):
        content_reference("unsafe kind", {})
    with pytest.raises(InvalidDataError, match="DataFrame"):
        dataset_fingerprint([1, 2])
    with pytest.raises(InvalidDataError, match="string column"):
        dataset_fingerprint(pd.DataFrame({1: [2]}))


def test_auditor_rejects_invalid_sources_and_export_content(case):
    _, assistant, _, result, report = case
    with pytest.raises(InvalidDataError, match="ResearchReport"):
        assistant.audit({"status": "complete"})
    with pytest.raises(InvalidDataError, match="AnalysisResult"):
        assistant.audit(report, result={})
    invalid_source = replace(result, specification=None)
    audit = assistant.audit(report, result=invalid_source)
    assert audit.status == "failed"
    assert audit.findings[0].code == "SOURCE_INVALID"

    for exports in (
        {"json": "{invalid"},
        {"json": 3},
        {"csv": "not a table mapping"},
        {"csv": {"statistical_results": None}},
    ):
        failure = assistant.audit(report, exports=exports)
        assert failure.status == "failed"
        assert failure.findings[0].field.startswith("exports.")
    assert assistant.audit(report).to_json() == assistant.audit(report).to_json()


def test_replay_record_validation_and_failure_paths(case, tmp_path):
    frame, assistant, _, result, _ = case
    good = assistant.reproducibility_record(result)
    payload = good.to_dict()
    with pytest.raises(InvalidDataError, match="mapping"):
        ReproducibilityRecord.from_dict([])
    for mutation, expected in (
        ({"schema_version": 3}, "unsupported"),
        ({"expected": []}, "invalid types"),
        ({"method_id": "different"}, "disagrees"),
        ({"dataset_fingerprint": {"algorithm": "wrong", "digest": "0" * 64}}, "fingerprint"),
    ):
        changed = deepcopy(payload)
        changed.update(mutation)
        with pytest.raises(InvalidDataError, match=expected):
            ReproducibilityRecord.from_dict(changed)
    with pytest.raises(InvalidDataError, match="DataFrame"):
        reproduce(good, data=[])
    with pytest.raises(InvalidDataError, match="ReproducibilityRecord"):
        reproduce({}, data=frame)
    with pytest.raises(InvalidDataError, match="Could not write"):
        good.save_package(tmp_path / "missing" / "package.zip")
    with pytest.raises(InvalidDataError, match="relative"):
        good.save_package(tmp_path / "bad.zip", data_reference="<script>.csv")

    payload = good.to_dict()
    payload["dataset_fingerprint"] = None
    no_fingerprint = ReproducibilityRecord.from_dict(payload)
    unavailable = reproduce(no_fingerprint, data=pd.DataFrame({"object": [object()]}))
    assert unavailable.status == "unavailable"
    assert unavailable.replay_reference is None


def test_ledger_validates_clock_reason_and_import(case):
    _, _, _, result, _ = case
    with pytest.raises(InvalidDataError, match="history_status"):
        DecisionLedger(history_status="fabricated")
    with pytest.raises(InvalidDataError, match="AnalysisResult"):
        DecisionLedger.import_result({})
    ledger = DecisionLedger(clock=lambda: "")
    with pytest.raises(InvalidDataError, match="clock"):
        ledger._record("question_prepared")
    assert ledger.events == ()
    ledger = DecisionLedger(clock=lambda: "t")
    with pytest.raises(InvalidDataError, match="reason"):
        ledger._record("question_prepared", reason=" ")
    with pytest.raises(InvalidDataError, match="event type"):
        ledger._record("fictional_preregistration")
    assert ledger.events == ()
    imported = DecisionLedger.import_result(result, clock=lambda: "t")
    assert imported.to_dict()["history_status"] == "unavailable"


def test_descriptive_replay_references_profile_without_embedding_observations():
    frame = pd.DataFrame(
        {"private_id": ["secret-a", "secret-b", "secret-c"], "score": [1.0, 2.0, 3.0]}
    )
    assistant = ResearchAssistant(frame)
    result = assistant.analyze(assistant.prepare_question(objective="descriptive"))
    record = assistant.reproducibility_record(result)
    assert "profile_reference" in record.to_dict()["expected"]
    assert "secret-a" not in record.to_json()
    assert reproduce(record, data=frame).status == "reproduced"


def test_replay_detects_missing_and_wrongly_typed_expected_fields(case):
    frame, assistant, _, result, _ = case
    baseline = assistant.reproducibility_record(result).to_dict()
    removed = deepcopy(baseline)
    removed["expected"]["values"].pop("test_statistic")
    replay = reproduce(ReproducibilityRecord.from_dict(removed), data=frame)
    assert "result.values.test_statistic" in replay.differing_fields

    changed_type = deepcopy(baseline)
    changed_type["expected"]["values"]["p_value"] = "0.1"
    replay = reproduce(ReproducibilityRecord.from_dict(changed_type), data=frame)
    assert "result.values.p_value" in replay.differing_fields

    changed_name = deepcopy(baseline)
    changed_name["expected"]["values"]["estimate_name"] = "other target"
    replay = reproduce(ReproducibilityRecord.from_dict(changed_name), data=frame)
    assert "result.values.estimate_name" in replay.differing_fields
    assert replay.to_dict()["status"] == "mismatch"

    no_frame = ReproducibilityRecord.from_result(result)
    assert no_frame.to_dict()["dataset_fingerprint"] is None
    assert any("no DataFrame" in item for item in no_frame.to_dict()["warnings"])
    unknown_data = pd.DataFrame({"x": [object()]})
    assert (
        reproduce(assistant.reproducibility_record(result), data=unknown_data).status
        == "unavailable"
    )


def test_audit_missing_fields_and_bad_export_mapping(case):
    _, assistant, _, result, report = case
    payload = report.to_dict()
    del payload["sections"]["methods"]["method_id"]
    payload["sections"]["results"]["effect_size"] = "not an effect record"
    audit = assistant.audit(ResearchReport(payload), result=result)
    assert audit.status == "failed"
    assert any(item.field == "sections.methods.method_id" for item in audit.findings)
    assert any(item.field == "sections.results.effect_size" for item in audit.findings)
    with pytest.raises(InvalidDataError, match="exports"):
        assistant.audit(report, exports=[])
