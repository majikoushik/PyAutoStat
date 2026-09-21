# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

### Added
- Initial packaging as `pyautostat` (src layout, pyproject.toml, tests, CI).
- `PyAutoStatError` and friendly, specific error messages for invalid data,
  unknown columns, unknown `test_type`, and too few groups in `hypothesis_tests()`.
- Effect sizes and confidence intervals in `hypothesis_tests()`: Cohen's d + CI
  for t-tests, rank-biserial correlation for Mann-Whitney, eta-squared for
  ANOVA, epsilon-squared for Kruskal-Wallis, each with a small/medium/large
  interpretation.
- `pyautostat.detect_column_types()` and `pyautostat.suggest_column_roles()`:
  heuristics for numeric-but-categorical columns, date-like text, and
  name-based role suggestions (identifier/target/datetime/economic). Both
  are included automatically in `analyze_all()`.
- `ReportGenerator.to_interactive_html()`: Plotly-based report with a
  hoverable correlation heatmap and zoomable per-column histograms.
  Requires the `report` extra (`pip install pyautostat[report]`).
- Phase 1: contact-format and missingness hints in `detect_column_types()`;
  measurement and action hints in `suggest_column_roles()`.
- Detailed per-group normality and Levene assumption statuses, selection
  reasons, and configurable deterministic percentile bootstrap intervals for
  hypothesis effect sizes. Reports can include hypothesis results in JSON,
  HTML, interactive HTML, and CSV.
- Categorical chi-square association with Cramér's V, expected-count checks,
  and optional Cohen's h for a named success outcome in a 2x2 table; both
  effect sizes support bootstrap intervals.
- Collapsible interactive report sections, table filtering and sorting, and
  chart rendering when a section is opened.

### Fixed
- `hypothesis_tests()` no longer treats a missing-value row as its own group.
- Validate DataFrame labels, scalar values, finite real numeric input, hypothesis
  test compatibility, usable group sizes, and constant or undefined test data.
- Skip normality tests that lack the required sample size or variation; preserve
  finite/undefined results explicitly and report reasons in `analysis_warnings`.
- Use pairwise observations for correlation p-values and represent undefined
  coefficients, outlier metrics, and descriptive statistics as `None`.
- Keep extreme finite values from aborting histogram peak analysis; flag
  numerical underflow and reject undefined effect sizes or confidence intervals.
- Escape untrusted HTML report text, emit standards-compliant JSON, protect
  written CSV files from spreadsheet formulas, and wrap file errors in
  `ReportError`. Removed global warning suppression.

### Changed
- Renamed the package from `autostat` to `pyautostat` everywhere (import
  path, PyPI distribution name, docs). `AutoStatError` is now `PyAutoStatError`.
- Dropped Python 3.9 support (EOL); the package now requires Python >=3.10.
- Three-or-more-group `auto` comparisons now choose ANOVA only when normality
  and equal-variance screens pass; otherwise they choose Kruskal-Wallis.
- Rank-biserial correlation now follows the first-versus-second group direction;
  negative sample epsilon-squared estimates are truncated at zero.

### Chore
- Consolidated repository documentation into a focused README, API reference,
  roadmap, changelog, contributor guide, and examples guide; removed redundant
  and unverified marketing documents.
- Reworked the runnable example to cover all public analysis and export
  features, optional Plotly output, and representative bad-data errors; added
  an examples guide and CLI smoke tests.
- Added `ruff check`, `ruff format`, and `mypy` (clean on `src/`) with a
  pre-commit config, and a GitHub Actions CI workflow that lints,
  type-checks, runs the test suite on Python 3.10-3.13 across Linux and
  Windows, and builds/validates the sdist and wheel with `twine check`.

## [0.1.0] - Unreleased

- Statistical analysis (descriptive stats, normality, outliers, correlation, hypothesis testing).
- Automated insights engine.
- Report export to JSON, HTML, CSV, and dict.
