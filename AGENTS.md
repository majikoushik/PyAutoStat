# PyAutoStat — contributor and coding-agent instructions

> **Status:** Project-wide operating instructions for Codex and contributors. This document governs how work is done; it does **not** imply that planned features already exist. Read `PRODUCT_VISION.md` for the locked destination and `DEVELOPMENT_ROADMAP.md` for phase boundaries. Follow the user's current, explicit task and the repository's actual code over speculative descriptions.

## Mission and operating priorities

Build **PyAutoStat: an Explainable and Reproducible Research Analysis Assistant**. The intended journey is *research question → defensible recommendation → validated calculation → qualified interpretation → reproducible research report*. The initial product uses deterministic rules, **not generative AI**.

1. **Statistical validity before automation or feature count.** An analysis that cannot be justified must be blocked or marked unresolved, never presented as certain.
2. **Ease of use.** A DataFrame alone should suffice for profiling. For hypothesis testing, require only information essential to the research question and study design. Infer only defensible metadata, provide transparent defaults for optional settings, and let experts override valid options.
3. **Transparent decisions.** Explain a recommendation, record assumptions and decisions, preserve uncertainty, and never silently change the estimand, method, or dataset.
4. **One engine, multiple interfaces.** Python API first; the future GUI must consume the same serializable configuration, validation, result, and missing-information structures. Do not put statistical logic in presentation code.
5. **Small, reviewable increments.** Implement only the explicitly assigned phase; later phases are specifications, not permission for scope expansion.

## Current repository facts (verify against code before editing)

- Public repository: `majikoushik/PyAutoStat`; Python import/distribution name: `pyautostat`. `main` is the default branch. Do not confuse the case-sensitive names of publishing configuration fields.
- Existing alpha code is version `0.1.0` at the time these instructions were drafted. The authoritative current version is `src/pyautostat/__init__.py`, read by Hatchling's dynamic version setting; do not freeze a version elsewhere.
- `src` layout; Hatchling build; Python >=3.10. Runtime dependencies presently include pandas, NumPy and SciPy. Plotly is optional (`report` extra); verify actual constraints in `pyproject.toml` before introducing APIs or dependencies.
- Existing modules: `analyzer.py` (`StatisticalAnalyzer`, descriptive analysis and independent-group tests); `categorical.py` (chi-square and effect measures); `detection.py` (advisory type/role hints); `insights.py` (`InsightEngine`, data-quality-oriented recommendations); `report.py` (`ReportGenerator`, JSON/CSV/HTML and optional interactive HTML); `exceptions.py`; `__init__.py` (exports/version).
- Existing tests are in `tests/`; runnable showcase in `examples/example_usage.py`; documentation in `README.md`, `API_REFERENCE.md`, `examples/README.md`, `ROADMAP.md`, and `CHANGELOG.md`. `ROADMAP.md` records an older high-level plan: retain it until deliberately reconciled, and do not mistake it for shipped behavior.
- Current `hypothesis_tests(test_type="auto")` relies on normality/variance screens for independent-group selection. This is **existing behavior to review and correct**, not the target design. Paired/repeated-measures support, full reporting styles, and the high-level ResearchAssistant are **not yet shipped** simply because these documents describe them.

## Locked user experience

- **Quick profile:** `ResearchAssistant(df).profile()` is a *proposed* simple API; dataset profiling must not require a research question.
- **Guided research:** the researcher supplies or confirms objective, outcome/predictor/group, estimand where material, and design facts that cannot be inferred (e.g., independence, pairing, nesting). Return structured missing-information requests in noninteractive scripts. A GUI can later render these as short questions. Never unexpectedly prompt inside a library call.
- **Advanced mode:** expose validated options without burdening beginners. Use safe documented defaults for optional settings, e.g. a disclosed 0.05 alpha when applicable; do not default unknown scientific design facts.
- Make the common path short; keep expert metadata available; use plain-language errors and explanatory result objects. Keep the proposed high-level API provisional until Phase 1 formally specifies it.

## Non-negotiable statistical and ethical safeguards

- Begin with **research objective and estimand**, then variable roles, study design, candidate methods, diagnostics, execution, and reporting. Normality-test p-values alone must never choose a different scientific question or automatically switch a mean-comparison estimand to a rank-based one.
- Failing to reject a normality hypothesis is **not proof of normality**. A large p-value is **not proof of no effect**, and a small one is **not a measure of effect size, practical importance, or causation**.
- Independence, pairing, clustering, randomization, causality, and data-collection intent are not reliably deducible from numbers. Ask for essential unknowns or return a blocker. Distinguish `known/confirmed`, `violated`, `unknown`, and `not_applicable` (or equivalent documented states).
- Every inferential output must identify its target quantity, test name, sample and excluded-row counts, statistic/df as applicable, p-value when applicable, effect estimate, confidence interval when supported, group ordering/direction, assumptions, limits, and warnings. Do not fabricate unavailable results or intervals.
- Do not automatically delete outliers, impute missing values, reclassify variables, change alpha, change tests, or perform favorable alternative analyses. Analysis-specific complete-case exclusions must be clearly documented. Preserve source data by default.
- Respect multiplicity and exploratory versus confirmatory distinctions; never run all tests and pick the most favorable p-value. A decision ledger is not evidence of externally authenticated preregistration.
- Sensitivity analyses must compare scientifically defensible specifications and identify estimand changes. Researcher-defined meaningful-effect thresholds need context and justification. A nonsignificant superiority test is not an equivalence test.
- Avoid categorical promises of “publication-ready” or guaranteed correctness: produce **publication-oriented** material subject to expert and journal-specific review. Never silently infer study facts or manufacture Methods text.
- Use established numerical implementations (SciPy/NumPy/pandas and carefully justified additional dependencies) rather than duplicating tested algorithms. Verify behavior across supported dependency versions before adopting modern functions.
- Handle empty/all-missing, constant, ties, sparse cells, small samples, non-finite values, extreme numerical scale and undefined denominators explicitly. Raise an actionable package exception or return a documented unavailable outcome; never disguise an error as a normal result.

## Architecture and compatibility

- Separate **profiling**, **research specification**, **design validation**, **recommendations**, **diagnostics**, **statistical execution**, **interpretation**, **reporting**, and **provenance**. Introduce modules only when needed by the active phase; avoid premature frameworks and a monolithic analyzer.
- Create a single typed, documented, JSON-serializable source of truth for configurations and results. Tables, narrative and export must use the same validated numbers. Future GUI code must not duplicate statistical decisions.
- Preserve current public imports and result dictionaries where practical. If correctness requires a behavioral change, add regression tests, document migration, and do not silently break compatibility. Prefer a new higher-level interface over changing every established method at once.
- Avoid mandatory cloud/network/AI services or GUI dependencies. Optional features must fail with clear installation guidance when their extra is unavailable. Keep tests offline. Interactive reports currently rely on a Plotly CDN; document this rather than claiming all output is fully offline.
- Preserve input DataFrames; use local RNGs and reproducible seeds; make outputs deterministic where feasible. Protect user data: do not include raw datasets in export archives without explicit opt-in. Escape untrusted strings in HTML and protect formula-like text in exported spreadsheet-readable files.
- Prefer Python standard-library data models where sufficient; choose any new dependency only with a documented need. Use type hints, useful docstrings, `pathlib`, UTF-8, and portable code for supported Python/OS versions. Follow actual Ruff/mypy configuration in `pyproject.toml`.

## Phase discipline and Codex workflow

1. Read this file, the locked vision, the active phase in `DEVELOPMENT_ROADMAP.md`, relevant source/tests, and the user-supplied phase prompt.
2. State the current behavior and intended scope in your implementation plan; flag any conflict between instructions and the actual repository.
3. Work **only** on the requested phase; do not claim, implement, or advertise future-phase features as shipped. Small prerequisite fixes are acceptable when necessary and disclosed.
4. Define executable acceptance tests before declaring completion. Use independent reference values, tests for invalid/ambiguous designs, edge cases and compatibility; do not merely test that a result key exists.
5. Update docs and examples for changed *shipped* behavior. Keep `README.md`, `API_REFERENCE.md`, `CHANGELOG.md`, old `ROADMAP.md` and the new roadmap consistent; do not rewrite historical entries as if new work already shipped.
6. Report files changed, functionality delivered, tests run with **actual results**, checks blocked with reasons, known limitations, and deviations from the phase plan. Do not claim tests or review happened unless performed.
7. Do not bump the version, create/push tags, modify release automation, publish to PyPI, or make GitHub releases without explicit user authorization. Prefer branch/PR workflows when asked to modify the repository; never perform destructive or forced Git operations without authorization.

## Verification commands

Use the current `pyproject.toml`/CI as the source of truth. The repository currently uses commands similar to:

```bash
python -m pip install -e ".[dev]"
python -m ruff check src tests
python -m ruff format --check src tests
python -m mypy src/pyautostat
python -m pytest -q --cov=pyautostat --cov-report=term-missing --cov-fail-under=90
python -m build
python -m twine check dist/*
```

Run affected unit tests during development and all applicable checks before reporting a phase complete. Test across supported Python versions where the environment permits; if a tool or environment is unavailable, say so. Do not lower coverage thresholds or disable lint/type checks merely to achieve a passing badge. Prefer tests that validate statistical semantics as well as arithmetic and formatting.

## Before finishing

- Did the implementation preserve the stated estimand, study design and source data?
- Are default choices and requested confirmations visible, and are unsupported cases safely handled?
- Are recommendations, result values, narratives and exports consistent and traceable?
- Does the basic workflow remain simple and the implementation UI-independent?
- Are source, docs, tests and changelog aligned with what actually ships?
- Can the user reproduce what was run, and have you stated every unverified claim?
