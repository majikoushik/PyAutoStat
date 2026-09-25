# PyAutoStat product vision

PyAutoStat aims to be an explainable and reproducible research analysis assistant: a researcher
moves from a question and dataset to a defensible recommendation, validated calculation,
qualified interpretation, and reproducible report without losing control of scientific choices.

This document contains enduring product and scientific principles. [`ROADMAP.md`](ROADMAP.md)
lists future improvement priorities, [`README.md`](README.md) introduces the package, and
[`API_REFERENCE.md`](API_REFERENCE.md) describes the shipped API.

## Product promise

PyAutoStat should make sound statistical workflows easier without pretending that software can
infer study design, domain meaning, or causal validity from numbers. The common path should be
short, while every important decision remains inspectable and advanced configuration remains
available.

The intended workflow is:

```text
dataset profile
→ research question and estimand
→ declared design facts
→ method recommendation and assumptions
→ validated execution
→ qualified interpretation
→ research report
→ audit and reproducibility record
```

Dataset profiling stands on its own and never requires a research question.

## Scientific principles

### Start with the question

Method choice begins with the objective, target quantity, variable roles, and study design.
Diagnostics can qualify a method but cannot silently replace a mean estimand with a rank estimand
or turn association into causation.

### Keep unknown facts unknown

Independence, pairing, clustering, randomization, sampling intent, and causal assumptions require
researcher knowledge. Noninteractive APIs return structured missing-information requests rather
than prompting or guessing. Future interfaces should render those same requests.

### Report uncertainty and limitations

Inferential output should identify the quantity, method, sample accounting, direction, effect,
supported uncertainty interval, assumptions, warnings, and limitations. Missing or unreliable
values remain unavailable. P-values are evidence under a model, not measures of effect magnitude,
practical importance, certainty, or causation.

### Preserve the scientific record

PyAutoStat should not silently delete outliers, impute or recode data, switch tests, change alpha,
alter pairing, sample rows, or choose a favorable result. Sensitivity analyses retain all declared
attempts and label estimand or contrast changes. Reports and exports use the same validated values
as the result object.

### Separate consistency from truth

Audits can detect contradictions among specifications, results, reports, and exports. Dataset
fingerprints can detect some changes. Neither proves that source data, design claims, collection
procedures, preregistration, or scientific conclusions are true.

## User experience

The beginner experience should begin with:

```python
assistant = ResearchAssistant(df)
profile = assistant.profile()
```

For guided inference, researchers supply or confirm the objective, outcome and predictor roles,
estimand, and essential design facts. Optional settings use transparent defaults. Advanced users
can inspect and serialize specifications, recommendations, results, plans, thresholds, reports,
audits, and replay metadata.

Plain-language errors should say what is missing, why it matters, and how to proceed. Structured
records allow notebooks, scripts, CLIs, and a future GUI to share one engine.

## Architecture vision

PyAutoStat keeps profiling, question specification, design validation, recommendation,
diagnostics, execution, interpretation, reporting, provenance, and presentation separate. One
typed, JSON-safe source of truth connects these layers. Presentation code must not choose methods
or recalculate statistics.

The package remains useful offline and should avoid mandatory cloud, AI, or GUI dependencies.
Optional capabilities must have clear installation and failure behavior.

## Privacy and responsible use

Source DataFrames are preserved by default. Reports and reproducibility packages should avoid raw
records and participant identifiers unless a future explicit contract permits them. Exports must
escape untrusted content and protect spreadsheet-readable cells. Small aggregate cells can still
be sensitive and require human review.

PyAutoStat produces publication-oriented material for expert and journal-specific review. It does
not guarantee correctness, research quality, regulatory compliance, publication acceptance, or a
causal conclusion.

## Long-term direction

The long-term product may support more research designs, stronger validation evidence, scalable
profiling, richer adapters, and additional export formats. Expansion remains subordinate to a
complete design-to-reproducibility contract and to measured evidence that the implementation is
safe, understandable, and maintainable.
