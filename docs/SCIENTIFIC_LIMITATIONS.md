# Scientific limitations

PyAutoStat is suitable for continued controlled expert evaluation of its documented workflows.
It is not a substitute for scientific design review or subject-matter judgment.

- Study design facts, pairing, independence, clustering, randomization, and sampling intent depend
  on researcher declarations. Numerical values cannot prove them.
- Researcher-facing narration is deterministic, rule-based presentation of stored results. It
  creates no new evidence, does not verify design assumptions, and cannot establish causation.
- Conventional effect-magnitude labels are descriptive communication conventions rather than
  universal practical-importance thresholds. Practical importance remains dependent on a
  researcher-declared quantity, direction, and threshold.
- The statistical catalogue is deliberately limited. Repeated measures with more than two
  conditions, clustered models, regression families, mixed models, survival analysis, causal
  inference, exact-table extensions beyond Fisher's 2x2 test, and broad multiplicity procedures
  are unsupported.
- The one-sample t-test targets the population mean difference from a researcher-declared
  reference. With zero sample variance the raw contrast is retained, but t, p, and standardized d
  are unavailable; this is not converted into ordinary finite inference.
- Paired Wilcoxon uses explicit unit-ID matching and the `wilcox` zero-difference convention. It
  evaluates signed ranks of paired differences and is not universally a median-difference test;
  a location-shift interpretation needs an appropriate symmetry assumption.
- Inferential Spearman targets monotonic rank association. Its percentile interval resamples
  observation pairs together; ties are allowed. It neither guarantees linearity nor establishes
  causation.
- Fisher's exact test is limited to ordered 2x2 tables. Its primary effect is SciPy's
  unconditional sample odds ratio. Nonfinite zero-cell ratios are recorded as unavailable with a
  status, and no odds-ratio confidence interval is currently supplied.
- Missing data use analysis-specific complete cases. There is no automatic imputation or
  missing-data mechanism model.
- Outliers are reported for review. They are never deleted automatically.
- Coefficient of variation is a relative-spread description, not a universal quality score. It is
  most interpretable for ratio-scale measurements with a meaningful zero; numeric dtype alone
  cannot establish that meaning. Zero or numerically near-zero means and unavailable/nonfinite
  sample SDs produce an unavailable CV rather than infinity or NaN.
- Frequency tables and cross-tabs describe observed distributions. They do not establish
  association, statistical significance, causation, importance, or population representativeness.
  Cross-tabs exclude rows missing either selected variable and do not create a missing category.
- Sensitivity analysis executes only researcher-declared scenarios, preserves different
  estimands, and never selects or ranks results by p-value. It cannot supply a universal
  robustness conclusion. Matching decisions in its plain-text comparison remain descriptive.
- Practical importance requires a researcher-defined quantity and threshold. PyAutoStat supplies
  no universal threshold or generic decision rule. Nonsignificance is not equivalence; formal
  equivalence and noninferiority tests are unsupported.
- Prospective power and precision calculations depend on anticipated effects, SDs, allocation,
  and distributional approximations supplied by the researcher. They do not guarantee achieved
  power and do not implement automatic observed post-hoc power.
- General, APA-oriented, and IEEE-oriented templates organize the same canonical numbers. They do
  not guarantee journal, regulatory, accessibility, or publication compliance.
- Reporting completeness checks whether applicable implemented fields are represented. It is not
  a study-quality, bias, certainty, or publication-readiness score.
- Auditing checks internal consistency against the recorded source result. It does not establish
  that the source data, design, method choice, or scientific claim is true.
- Dataset fingerprints detect some changes but do not authenticate data, collection, identity, or
  custody. Reproduction still requires the caller to supply data explicitly.
- Decision-ledger timestamps and planning records are local software observations. They do not
  prove that a plan existed externally or before all researcher access to outcomes.
- Reports and snapshots omit raw DataFrames and participant identifier values, but small aggregate
  cells can still disclose sensitive information and require researcher review.
- Resource classifications use deep DataFrame memory and numeric-column width as local advisory
  heuristics. They do not predict peak memory or runtime on every dtype, dependency version, or
  machine. Profiling remains a full-data operation and can require substantially more temporary
  memory than the reported DataFrame size.

## Minimum dependency compatibility scope

The declared floors remain pandas 1.0, NumPy 1.19, and SciPy 1.5. A package-wide source review
found use of established pandas table, dtype, correlation, and missing-value operations; NumPy
array, quantile, random-generator, and finite-value operations; and SciPy descriptive-test,
t-distribution, chi-square, and noncentral-t functions. The closure review did not identify an API
call that clearly requires a higher version than the declared floor.

Exact minimum-version execution is still unverified. The closure checks ran on Python 3.12.7,
pandas 3.0.6, NumPy 2.2.6, and SciPy 1.13.0, plus the configured GitHub CI Python/OS matrix with
resolver-selected dependency versions. The old declared floor releases were not installed in a
separate compatible interpreter matrix, so this project does not claim verified runtime
compatibility at those exact minimum versions. The dependency declarations were not raised.
