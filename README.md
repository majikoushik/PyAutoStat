# PyAutoStat

[![CI](https://github.com/majikoushik/pyautostat/actions/workflows/ci.yml/badge.svg)](https://github.com/majikoushik/pyautostat/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://github.com/majikoushik/PyAutoStat/blob/main/pyproject.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue)](https://github.com/majikoushik/PyAutoStat/blob/main/LICENSE)

PyAutoStat is a statistical research assistant for pandas DataFrames. It provides deterministic,
statistically bounded workflows for profiling data, recording a research question, recommending
and executing supported methods, interpreting results, and producing reproducible reports.

The package is in **alpha**. Researchers remain responsible for study-design facts, measurement
meaning, sampling assumptions, and the scientific use of results. PyAutoStat leaves unknown design
facts unresolved rather than guessing them.

## Installation

PyAutoStat requires Python 3.10 or newer:

```bash
python -m pip install pyautostat
```

Interactive Plotly reports are optional:

```bash
python -m pip install "pyautostat[report]"
```

## Quick start: profile a DataFrame

Profiling needs only a DataFrame. It does not require a research question.

```python
import pandas as pd

from pyautostat import ResearchAssistant

df = pd.DataFrame(
    {
        "student": ["A", "B", "C", "D", "E", "F"],
        "hours_studied": [2, 3, 4, 5, 6, 7],
        "exam_score": [55, 61, 67, 74, 81, 88],
    }
)

assistant = ResearchAssistant(df)
profile = assistant.profile()

print(profile["overview"])
print(profile["data_quality"]["issues"])
print(profile["resource_info"])
```

The profile includes descriptive summaries, missingness, duplicates, distributions, histograms,
outlier review cues, correlation records, type and role suggestions, and structured resource
advisories. Profiling never samples, truncates, imputes, or modifies the source DataFrame.

## Guided analysis

Supply the scientific target and design facts that cannot be inferred safely:

```python
scores = pd.DataFrame(
    {
        "teaching_method": ["standard"] * 6 + ["new"] * 6,
        "exam_score": [62, 65, 68, 70, 72, 75, 67, 71, 74, 78, 80, 84],
    }
)

assistant = ResearchAssistant(scores)
workflow = assistant.run(
    objective="compare_groups",
    outcome="exam_score",
    predictor="teaching_method",
    estimand="mean",
    design="independent",
    variable_types={"exam_score": "continuous"},
)

print(workflow.status)                    # completed
print(workflow.analysis.method_id)        # welch_t
print(workflow.interpretation.summary)
print(workflow.audit.status)
```

`run()` connects question validation, one method recommendation, one statistical execution,
deterministic interpretation, reporting, consistency audit, and reproducibility metadata. It does
not write files or replay the analysis automatically.

## When information is missing

Omit an essential design fact and the workflow returns a structured request without running a
test:

```python
pending = assistant.run(
    objective="compare_groups",
    outcome="exam_score",
    predictor="teaching_method",
    estimand="mean",
    variable_types={"exam_score": "continuous"},
)

print(pending.status)  # needs_input
print(pending.missing_information)

revised = assistant.update_question(
    pending.draft,
    design="independent",
)
workflow = assistant.run(draft=revised)
```

PyAutoStat does not silently default independence, pairing, clustering, unit identifiers,
estimands, or causal meaning. `needs_input`, `data_limited`, `unsupported`, `partial`, and `failed`
remain distinct from a completed workflow.

## Supported capabilities

- Dataset profiling with structured quality and resource metadata.
- Welch independent-means analysis and explicit Student t-test.
- Explicit two-condition paired t-test using a declared unit-ID column.
- Mann-Whitney U, standard one-way ANOVA, and Kruskal-Wallis calculations within their documented
  selection and explicit-use boundaries.
- Pearson numerical association and Pearson chi-square categorical association.
- Effect estimates, supported confidence intervals, sample accounting, assumptions, and warnings.
- Deterministic interpretation and canonical research reports.
- Static HTML, Markdown, JSON, CSV tables, safe LaTeX, and optional interactive Plotly HTML.
- Decision ledger, report/export audit, reproducibility metadata, and explicit supplied-data replay.
- Declared sensitivity scenarios and researcher-defined practical-significance thresholds.
- Statistical analysis plans, plan-adherence comparison, and prospective independent or paired
  mean power and precision planning.
- Reporting-completeness checks and UI-independent session snapshots.

See the [capability matrix](docs/FINAL_CAPABILITY_MATRIX.md) for exact supported and unsupported
method families.

## Advanced workflows

### Explicit paired analysis

Pairing uses an identifier, never row order:

```python
paired = pd.DataFrame(
    {
        "participant_id": [1, 1, 2, 2, 3, 3, 4, 4],
        "condition": ["before", "after"] * 4,
        "exam_score": [62, 68, 70, 74, 65, 71, 73, 79],
    }
)

paired_workflow = ResearchAssistant(paired).run(
    objective="compare_groups",
    outcome="exam_score",
    predictor="condition",
    estimand="mean",
    design="paired",
    unit_id="participant_id",
    condition_order=("after", "before"),
    variable_types={"exam_score": "continuous"},
)
```

Duplicate unit/condition rows are blocked rather than averaged. Incomplete pairs are counted and
excluded transparently. See [advanced planning and presentation](docs/ADVANCED_PLANNING_AND_PRESENTATION.md).

### Analysis and study planning

```python
from pyautostat import StudyPlanner

draft = assistant.prepare_question(
    objective="compare_groups",
    outcome="exam_score",
    predictor="teaching_method",
    estimand="mean",
    design="independent",
    variable_types={"exam_score": "continuous"},
)
plan = assistant.analysis_plan(draft, report_style="apa")

prospective = StudyPlanner().independent_mean_power(
    target_difference=5,
    sd_group1=10,
    sd_group2=12,
    target_power=0.80,
)
```

Planning inputs are researcher assumptions and are never derived automatically from an observed
result. Observed post-hoc power is not implemented.

### Sensitivity and practical significance

```python
from pyautostat import MeaningfulEffectThreshold, SensitivitySpecification

scenario = SensitivitySpecification(
    name="pooled variance",
    specification=workflow.analysis.specification,
    method_id="student_t",
    assumptions=("Equal population variances",),
)
sensitivity = assistant.sensitivity_analysis(
    workflow.analysis,
    scenarios=[scenario],
)

threshold = MeaningfulEffectThreshold(
    quantity="mean_difference",
    minimum_magnitude=5,
    direction="two_sided",
    unit="points",
)
practical = assistant.practical_significance(
    workflow.analysis,
    threshold=threshold,
)
```

Every sensitivity attempt remains visible, and different estimands or paired contrasts are not
presented as ordinary same-estimand robustness checks. Practical and statistical significance are
reported separately. Formal equivalence and noninferiority inference are not implemented. See
[sensitivity and practical significance](docs/ROBUSTNESS_AND_PRACTICAL_SIGNIFICANCE.md).

### Reports, audit, and replay

```python
report = assistant.report(
    workflow.analysis,
    sensitivity=sensitivity,
    practical_significance=practical,
)

html = report.to_html(style="apa")
markdown = report.to_markdown(style="apa")
latex = report.to_latex(style="apa")
audit = assistant.audit(report)

record = assistant.reproducibility_record(
    workflow.analysis,
    sensitivity=sensitivity,
    practical_significance=practical,
)
```

Construction and in-memory rendering write no files. Explicit `save_*` methods require a path and
protect existing files unless overwrite is requested. Replay requires the caller to supply data.
Auditing checks internal consistency, not scientific truth. See
[provenance and replay](docs/PROVENANCE_AND_REPLAY.md) and the
[research report schema](docs/RESEARCH_REPORT_SCHEMA.md).

## Resource awareness

`profile()["resource_info"]` reports row and column counts, the deep pandas memory estimate,
analytical numeric-column count, correlation-matrix dimensions, a local performance category, and
structured warnings. The default advisory policy is:

- `large` at 256 MiB;
- `very_large` at 1 GiB;
- a correlation-width advisory at 100 analytical numeric columns.

These are performance heuristics rather than scientific or universal hardware limits. Full
profiling still runs exactly as requested. If deep memory estimation fails, profiling continues
with an `unknown` resource level and an advisory warning.

## Statistical safeguards and limits

- Normality-test nonrejection is not proof of normality.
- Statistical significance is not effect magnitude, practical importance, or causation.
- Automatic selection requires the estimand and design; it does not choose a favorable p-value.
- Source values are not automatically recoded, imputed, removed, sampled, or truncated.
- Outlier flags are review cues and never automatic deletion rules.
- Repeated measures with more than two conditions, clustered models, regression, mixed models,
  survival analysis, causal inference, sparse exact-table alternatives, broad post-hoc procedures,
  and formal equivalence/noninferiority tests are unsupported.
- Current templates are publication-oriented aids, not journal or regulatory certification.

Read [scientific limitations](docs/SCIENTIFIC_LIMITATIONS.md) before applying results in research
or decisions.

## Detailed APIs and legacy interfaces

The [API reference](API_REFERENCE.md) documents `ResearchAssistant`, configuration and result
records, planning, reports, audit, replay, and the legacy `StatisticalAnalyzer`, `InsightEngine`,
and `ReportGenerator` interfaces.

Legacy calls remain available:

```python
from pyautostat import InsightEngine, ReportGenerator, StatisticalAnalyzer

analysis = StatisticalAnalyzer(df).analyze_all()
insights = InsightEngine(analysis).generate_insights()
report = ReportGenerator(analysis, insights)
```

## Examples and project documents

- [Examples guide](examples/README.md)
- [API reference](API_REFERENCE.md)
- [Product vision](PRODUCT_VISION.md)
- [Future roadmap](ROADMAP.md)
- [Architecture](docs/ARCHITECTURE.md)
- [Statistical validation](docs/STATISTICAL_VALIDATION.md)

## Development

```bash
git clone https://github.com/majikoushik/PyAutoStat.git
cd PyAutoStat
python -m pip install -e ".[dev]"
python -m ruff check src tests
python -m ruff format --check src tests
python -m mypy src/pyautostat
python -m pytest -q --cov=pyautostat --cov-report=term-missing --cov-fail-under=90
python -m build
python -m twine check dist/*
```

See [CONTRIBUTING.md](CONTRIBUTING.md) and [AGENTS.md](AGENTS.md) for contribution guidance.

## License

MIT. See [LICENSE](LICENSE).
