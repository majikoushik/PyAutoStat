# PyAutoStat development guide

This file gives future contributors and coding agents the repository context and
working rules for PyAutoStat. Follow the user's current request first. Keep this
guide aligned with the code as the package evolves.

## Project context

- PyAutoStat is a Python package for automated statistical analysis of pandas
  DataFrames, aimed at researchers and analysts. The package is currently an
  alpha release (`0.1.0` in `src/pyautostat/__init__.py`).
- The package uses a `src` layout and is built with Hatchling. Python `>=3.10`
  is supported. Runtime dependencies are pandas, NumPy, and SciPy; Plotly is an
  optional dependency for interactive reports. `pyproject.toml` is the source
  of truth for package metadata, dependencies, and tool configuration.
- Public imports are defined in `src/pyautostat/__init__.py`. Preserve them and
  the shape of existing result dictionaries unless an intentional API change
  has been requested and documented.
- `StatisticalAnalyzer` in `analyzer.py` copies the input DataFrame.
  `analyze_all()` returns overview, descriptive, normality, outlier,
  correlation, missing-data, data-quality, distribution, column-role,
  column-type, histogram, and `analysis_warnings` sections. `hypothesis_tests()` is a separate
  operation; its results are not added to `analyze_all()` automatically.
- `categorical_association()` is a separate independent-sample chi-square
  comparison with Cramér's V. It requires expected cell counts of at least
  five. Cohen's h is available for 2x2 tables with an explicit success value.
- `detection.py` provides advisory column-type and name-based role heuristics,
  including contact-format and missingness hints.
  These suggestions do not alter the analysis or exclude columns.
- `InsightEngine` in `insights.py` converts analysis results into severity-rated
  findings and recommendations. `ReportGenerator` in `report.py` exports dict,
  JSON, CSV, static HTML, and optional Plotly HTML.
- `examples/example_usage.py` is the runnable end-to-end feature showcase;
  `examples/README.md` maps public features to its sections. `ROADMAP.md`
  records product goals and distinguishes shipped features from future work. Tests live
  under `tests/`; CI runs tests with coverage on Linux and Windows, and runs
  lint, formatting, mypy, and a package build on Linux.

## Source of truth and scope

- Read the relevant source and tests before changing behavior. `ROADMAP.md`
  contains proposed work, not a specification to apply wholesale. In particular,
  do not assume that streaming/chunking, post-hoc tests, imputation, regulatory
  compliance, or publication-ready formatting are implemented.
- Keep feature claims in `README.md`, `API_REFERENCE.md`, and `examples/README.md`
  consistent with shipped behavior. Treat performance and market claims as
  unverified unless measured or sourced.
- Make focused changes. Update public documentation and `CHANGELOG.md` when a
  user-visible API, result schema, behavior, dependency, or supported Python
  version changes.

## Statistical and data-handling standards

- Prefer well-defined methods from SciPy, NumPy, and pandas. State the test,
  assumptions, sample sizes, effect-size definition, confidence level, and
  direction of comparisons where applicable. Do not equate a p-value with the
  probability that a hypothesis is true, or a nonsignificant result with proof
  of no effect.
- Do not silently change an analysis based on heuristic column detection.
  Keep user-selected group and value columns explicit. Validate their types,
  distinct groups, and usable observations before running a test.
- Handle small samples, all-missing columns, constant values, non-finite
  values, and zero denominators deliberately. Return a documented unavailable
  result or raise a specific `PyAutoStatError` subclass with guidance; do not
  emit misleading statistics or silently return an empty result.
- Check the minimum sample size and other preconditions for each SciPy test.
  D'Agostino-Pearson's `normaltest` needs at least eight observations; the
  current implementation skips smaller samples. `hypothesis_tests(auto)`
  selects between standard ANOVA and Kruskal-Wallis for three or more groups
  using per-group normality and Levene screens. These screens are heuristics,
  and Kruskal-Wallis requires at least five usable observations per group.
- Preserve the meaning of missing values, group ordering, and paired versus
  independent observations. Do not describe the current independent-group
  tests as suitable for paired before/after data.
- Avoid global warning suppression in new code. Handle expected numerical
  warnings locally and make invalid or undefined outputs clear to callers.
- Escape user-supplied labels and findings before inserting them into HTML.
  Reports may contain untrusted DataFrame column names or values.

## Implementation conventions

- Keep runtime imports limited to declared dependencies. Import optional
  dependencies inside the feature that needs them and provide a clear install
  message when absent.
- Use type hints for new or substantially changed public code. Keep functions
  small enough to explain their statistical purpose, and document public
  parameters, return structures, units, and exceptions.
- Preserve input DataFrames unless a public API explicitly promises mutation.
  Prefer deterministic behavior and local random generators with fixed seeds
  in examples and tests.
- Keep exports portable across supported Python versions and operating
  systems. Use `pathlib` for new path handling, UTF-8 for text files, and
  avoid assuming output directories already exist without documenting it.
- Respect the existing Ruff configuration (`E`, `F`, `I`, `UP`, `B`, line
  length 100) and mypy configuration in `pyproject.toml`. Do not add broad
  lint/type ignores to work around a local issue.

## Verification

- Add or update focused tests when statistical behavior, result schemas, or
  error handling changes. Compare numerical results with trusted SciPy or
  pandas calculations, and test relevant boundary cases rather than only
  asserting that a key exists. Do not add tests for prose-only or trivial
  reversible changes.
- Run the checks relevant to the change. The CI commands are:

  ```text
  python -m ruff check src tests
  python -m ruff format --check src tests
  python -m mypy src/pyautostat
  python -m pytest -q --cov=pyautostat --cov-report=term-missing --cov-fail-under=90
  python -m build
  python -m twine check dist/*
  ```

- Install development dependencies with `python -m pip install -e ".[dev]"`
  when the environment permits. Report any check that could not run and why;
  do not claim it passed. CI targets Python 3.10 through 3.13 on Linux and
  Windows. The mypy target is set to Python 3.12 because of NumPy stub syntax.
- Keep tests independent of network access. Plotly is optional at runtime;
  the interactive HTML currently references Plotly JavaScript from a CDN.

## Before finishing a change

- Confirm the public behavior and examples agree with the implementation.
- Summarize what changed, why, which checks ran, and any remaining limitation.
- Do not treat this workspace as a Git checkout unless `.git` is present; a
  source snapshot may have no repository metadata.
