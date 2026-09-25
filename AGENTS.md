# PyAutoStat contributor and coding-agent instructions

These are permanent project-wide working rules. Follow the user's explicit task and
[`ROADMAP.md`](ROADMAP.md), and verify the actual source and tests before assuming a capability
exists.

## Mission and priorities

PyAutoStat is an explainable and reproducible research analysis assistant for pandas DataFrames.
It connects dataset profiling, explicit research specifications, defensible recommendations,
validated calculations, qualified interpretation, reporting, audit, and reproducibility. The
package is deterministic and does not require generative AI, a cloud service, or a GUI.

Work in this order:

1. Protect statistical validity and the stated estimand.
2. Keep common workflows easy to discover and use.
3. Make decisions, assumptions, exclusions, warnings, and uncertainty visible.
4. Keep one statistical engine behind every interface.
5. Make changes small, reviewable, tested, and within the assigned scope.

## Current repository facts

- Repository: `majikoushik/PyAutoStat`; import and distribution name: `pyautostat`; default
  branch: `main`.
- Read `src/pyautostat/__init__.py` for the authoritative package version. Do not duplicate or
  freeze the version in contributor guidance.
- The package uses a `src` layout, Hatchling, Python 3.10+, pandas, NumPy, and SciPy. Plotly is an
  optional reporting extra. Read `pyproject.toml` for current constraints and tool settings.
- `ResearchAssistant(df).profile()` is the beginner dataset-only entry point.
- `ResearchAssistant(df).run(...)` is the integrated guided workflow. It preserves unresolved
  design facts as structured requests and never prompts unexpectedly inside a library call.
- The public architecture includes dataset profiling; research questions and serializable
  specifications; design-aware recommendation; statistical execution; deterministic
  interpretation; research reports and styled exports; audit and explicit replay; sensitivity
  analysis; researcher-defined practical-significance thresholds; statistical analysis plans;
  prospective study planning; explicit two-condition paired analysis; reporting completeness;
  and UI-independent session snapshots.
- The legacy `StatisticalAnalyzer`, `InsightEngine`, and `ReportGenerator` APIs remain supported.
- `README.md` is the beginner introduction, `API_REFERENCE.md` is the detailed public API,
  `PRODUCT_VISION.md` contains enduring product principles, `ROADMAP.md` is the sole future-work
  roadmap, and `CHANGELOG.md` records software changes.

## Scientific safeguards

- Start with the research objective and estimand, then variable roles, design, candidate methods,
  diagnostics, execution, and reporting. A diagnostic p-value must not silently change the
  scientific question.
- Independence, pairing, clustering, randomization, causality, and collection intent cannot be
  inferred reliably from values. Require the researcher to supply essential unknown facts.
- Failing to reject a null hypothesis is not proof of no effect or normality. Statistical
  significance is not effect magnitude, practical importance, or causation.
- Every inferential result must identify its quantity, method, sample and exclusion counts,
  statistic and degrees of freedom where applicable, p-value where applicable, effect estimate,
  supported interval, direction, assumptions, limitations, and warnings.
- Never fabricate unavailable results. Handle empty, all-missing, constant, tied, sparse,
  nonfinite, extreme-scale, and undefined-denominator inputs explicitly.
- Do not automatically remove outliers, impute or recode data, change alpha, choose a favorable
  test, alter pairing, sample rows, truncate data, or switch estimands.
- Sensitivity analyses must retain every declared attempt and identify estimand or contrast
  changes. A nonsignificant superiority test is not an equivalence test.
- Planning inputs and meaningful-effect thresholds are researcher supplied. Local ledger entries
  do not prove external preregistration.
- Reporting and completeness checks organize recorded evidence; they do not certify study quality,
  journal compliance, causality, or publication readiness.

## Architecture and compatibility

- Keep profiling, specification, validation, recommendation, execution, interpretation,
  reporting, provenance, and presentation responsibilities separate.
- Use typed, documented, JSON-safe records as the source of truth. Reports, audits, exports, and
  adapters must consume the same validated values rather than recalculate statistics.
- Preserve documented public imports, signatures, schemas, and established result keys where
  scientifically defensible. Make correctness changes explicit, additive when possible, tested,
  and documented.
- Use established numerical implementations from SciPy, NumPy, and pandas rather than duplicating
  algorithms. Check behavior across supported versions where feasible and state unverified scope.
- Preserve input DataFrames. Use local reproducible random generators and record stochastic
  settings. Resource diagnostics are advisory and must not change computations.
- Optional features must fail with actionable installation guidance when their extras are absent.
  Keep the core offline and UI independent.

## Privacy and security

- Do not include raw datasets or participant identifiers in reports, snapshots, or reproducibility
  packages without an explicit opt-in contract.
- Escape untrusted text in HTML and LaTeX, and protect formula-like cells in spreadsheet-readable
  exports.
- Treat small aggregate cells as potentially sensitive. Dataset fingerprints detect changes but
  do not authenticate identity, custody, or provenance.
- Never expose credentials, environment secrets, or arbitrary executable content through records
  or exports.

## Development workflow

1. Read this file, the assigned task, `PRODUCT_VISION.md`, `ROADMAP.md`, and the relevant source,
   tests, and focused documentation.
2. Inspect the current branch and working tree. Preserve unrelated work and obey the user's Git
   constraints.
3. Describe current behavior and the bounded intended change. Resolve routine implementation
   choices from evidence in the repository.
4. Add tests that validate scientific meaning, invalid and ambiguous designs, edge cases,
   serialization, and compatibility. Avoid tests that merely assert a key exists.
5. Update the README, API reference, changelog, examples, and focused docs when shipped behavior
   changes. Do not advertise unsupported capabilities.
6. Run affected tests during development and all required quality gates before reporting
   completion.
7. Report actual commands and results, changed files, known limitations, and unverified claims.

Future work must follow the user's explicit assignment and `ROADMAP.md`. Candidate roadmap items
are not permission to implement them, and no feature is shipped merely because documentation
mentions it.

## Verification

Use `pyproject.toml` and CI as the source of truth. The standard local gates are:

```bash
python -m ruff check src tests
python -m ruff format --check src tests
python -m mypy src/pyautostat
python -m pytest -q --cov=pyautostat --cov-report=term-missing --cov-fail-under=90
python -m build
python -m twine check dist/*
```

Also run focused numerical, documentation, example, or installed-wheel checks required by the
change. Never lower thresholds or disable a failing test to obtain a pass.

## Git and release safeguards

- Do not create or switch branches, commit, push, merge, rebase, reset, clean, tag, publish, or
  create a release unless the user explicitly authorizes that action.
- Never force a Git operation or rewrite repository history without explicit authorization.
- Do not change the package version, publishing automation, or dependency floors incidentally.
- Keep release preparation and publication separate from ordinary implementation work.

## Before finishing

- Is the estimand, design, pairing, group order, and source data preserved?
- Are defaults, exclusions, warnings, unsupported cases, and missing information visible?
- Are recommendation, calculation, interpretation, report, audit, and replay records consistent?
- Does the beginner path remain short while advanced metadata remains available?
- Are source, tests, docs, examples, and changelog aligned with actual behavior?
- Can the user reproduce every reported check, and are all limitations stated?
