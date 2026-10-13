# PyAutoStat 1.1.0 release notes

PyAutoStat 1.1.0 is a backward-compatible capability and hardening release for the stable 1.x
series. It preserves the established 24-method statistical catalogue and public compatibility
commitment while expanding structured metadata, sensitivity analysis, and defensive handling of
ambiguous or numerically difficult inputs.

## Highlights

- Added coefficient-level sensitivity records for linear and logistic regression without
  collapsing a fitted model into an arbitrary scalar.
- Added integrated Cronbach's-alpha sensitivity analysis with explicit construct-comparability
  boundaries.
- Added canonical method and quantity registries and public lookup/validation helpers.
- Added structured assumption metadata and post-execution diagnostic reconciliation.
- Expanded practical-significance support for bounded correlation, paired-proportion, and
  repeated-measures quantities while retaining explicit exclusions where a scalar threshold would
  be scientifically ambiguous.
- Preserved independent hypothesis alpha and confidence levels, with an advisory disclosure when
  they are not complementary.

## Independent audit remediation

The release includes a second-pass review of all 24 registered methods and their existing tests.
Corrections include:

- preserving raw typed category levels during computation while serializing public labels safely;
- rejecting Boolean, complex, and silently coerced nonnumeric values in real-quantitative roles;
- validating finite confidence levels, bootstrap counts, and nonnegative random seeds consistently;
- hardening two-way ANOVA, repeated-measures ANOVA, and ICC for extreme numerical scales and
  degenerate F statistics;
- preventing participant-level paired arrays and identifier values from leaking through public
  result, bundle, or session records;
- binding sensitivity execution to the source dataset fingerprint; and
- converting extreme-scale profiling failures into structured warnings or actionable rescaling
  errors instead of leaking raw numerical warnings or nonfinite JSON values.

## Compatibility

- No registered statistical method was removed.
- The 109 documented top-level exports remain covered by the 1.x compatibility policy.
- The legacy `StatisticalAnalyzer`, `InsightEngine`, and `ReportGenerator` APIs remain supported.
- Stable serialized records remain backward compatible; the additive sensitivity schema records
  coefficient-level estimates explicitly.

## Validation

Release preparation completed the configured Ruff, formatting, Mypy, coverage, package-build,
Twine, installed-wheel, example-gallery, and strict numerical-reference gates. The strict
reference harness represents all 24 registered methods and reports deferred Level D metadata
separately from independently or externally checked numerical fields. Passing these checks is
evidence for the tested cases, not a guarantee of universal scientific validity.

## Install

```bash
pip install pyautostat==1.1.0
```

Optional features remain available through the `report`, `figures`, `pdf`, and `docx` extras.
See the [API reference](../API_REFERENCE.md), [method contracts](STATISTICAL_METHOD_CONTRACTS.md),
[scientific limitations](SCIENTIFIC_LIMITATIONS.md), and [changelog](../CHANGELOG.md) for the full
contract and release history.
