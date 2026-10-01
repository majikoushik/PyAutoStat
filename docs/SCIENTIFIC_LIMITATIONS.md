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
- The statistical catalogue is deliberately limited. Clustered models, mixed-effects models,
  mixed ANOVA, factorial repeated-measures ANOVA, multinomial, ordinal, count, regularized,
  interaction, and polynomial regression, survival analysis, causal inference,
  exact-table extensions beyond Fisher's 2x2 test, and broad multiplicity procedures are
  unsupported.
- One-way repeated-measures ANOVA evaluates overall mean differences across 3 or more repeated
  conditions for the same units using complete-case panels. Units with missing repeated observations
  are excluded; missingness is not modeled, and missing completely at random is not assumed.
  Mauchly's test evaluates sphericity; failing to reject sphericity does not prove equal pairwise
  variance. Greenhouse-Geisser correction adjusts degrees of freedom and p-values when sphericity
  is violated; it does not change the observed F statistic. Pairwise follow-up uses per-contrast
  analytical confidence intervals alongside Holm-adjusted p-values; confidence intervals are not
  simultaneous confidence bands. Repeated-measures ANOVA does not establish causality.
- The Friedman test evaluates within-unit rank distributions across 3 or more repeated conditions
  for ordered numeric outcomes. Textual ordinal labels require explicit numeric scoring and are not
  automatically encoded. It evaluates within-unit ranks, not mean differences, and does not universally
  test equality of medians unless condition distributions have identical shapes and spreads. Complete-case
  panels are required; missingness is not modeled. Pairwise follow-up uses paired Wilcoxon signed-rank
  tests with Holm adjustment. Omnibus significance does not imply that every pairwise contrast differs.
- Two-way factorial ANOVA evaluates main effects and the interaction between exactly two categorical factors
  for independent observations and a continuous quantitative outcome. In unbalanced designs, sums of squares
  depend on the chosen convention: Type II evaluates each main effect conditional on the other main effect
  without interaction; Type III evaluates each effect conditional on all other model terms using sum-to-zero
  contrasts. When an interaction is statistically or practically meaningful, omnibus main effects do not
  describe homogeneous effects across all subgroups and must be qualified. Simple effects (effects of one
  factor within levels of the other), unweighted estimated marginal means, and (for 2x2 designs) the
  difference-of-differences interaction contrast are reported with Holm step-down multiplicity adjustment.
  Pointwise Student-t intervals for contrasts are not simultaneous confidence bands. Residual normality
  and homoscedasticity diagnostics are advisory and do not silently transform data or alter model
  specifications. Factorial ANOVA does not establish causality; repeated-measures factorial ANOVA, mixed ANOVA,
  ANCOVA, random effects, and nested factors are unsupported.
- OLS regression targets an additive conditional mean for independent observational units. It
  uses one complete-case sample and an intercept; it does not impute, transform, select, or delete
  predictors or observations. Treatment-coded nominal, Boolean, and ordinal terms are relative to
  recorded reference levels; ordinal spacing is not assumed. Coefficients are conditional
  associations, not causal effects, and R-squared is in-sample fit rather than predictive
  accuracy on new observations.
- Regression VIF, Breusch-Pagan, Jarque-Bera, Cook's distance, leverage, externally studentized
  residual, and condition-number records are diagnostics, not proofs or automatic decision rules.
  Classical or HC3 covariance must be chosen explicitly. HC3 changes uncertainty estimates, not
  the fitted OLS coefficients. Influence flags never remove rows.
- The one-sample t-test targets the population mean difference from a researcher-declared
  reference. With zero sample variance the raw contrast is retained, but t, p, and standardized d
  are unavailable; this is not converted into ordinary finite inference.
- Paired Wilcoxon uses explicit unit-ID matching and the `wilcox` zero-difference convention. It
  evaluates signed ranks of paired differences and is not universally a median-difference test;
  a location-shift interpretation needs an appropriate symmetry assumption.
- Inferential Spearman targets monotonic rank association. Its percentile interval resamples
  observation pairs together; ties are allowed. It neither guarantees linearity nor establishes
  causation.
- Binary logistic regression estimates conditional event odds under the declared model. Odds
  ratios are not constant probability differences; convergence and full rank do not prove model
  correctness, causal identification, or out-of-sample predictive performance.
- Exact McNemar inference compares paired marginal event probabilities for two conditions. It
  requires correct unit identity, condition order, and event definition and does not establish a
  causal before/after effect.
- Point-biserial sign depends on the declared positive category. Kendall tau-b describes ordinal
  concordance rather than linear association or variance explained.
- Partial Pearson describes residual linear association conditional on the included quantitative
  controls. Control selection remains a scientific decision; adjustment does not establish that
  confounding was removed or identify an independent causal effect.
- Cronbach's alpha describes covariance-based internal consistency for the exact declared item set
  and complete-case sample. It does not establish unidimensionality; construct, content,
  criterion, convergent, or discriminant validity; measurement invariance; temporal stability;
  or inter-rater reliability. Alpha depends on item count and may rise with redundant items.
  PyAutoStat applies no universal `.70` pass/fail rule, does not discover scales, and never deletes
  or reverse-scores items automatically. Numeric Likert-style scores use ordinary variances and
  correlations; ordinal/polychoric alpha, omega, factor analysis, and PCA are unsupported.
- Intraclass Correlation Coefficients (ICC) evaluate relative variance proportions across targets
  and raters on quantitative measurements. They require a fully crossed, balanced panel of at least
  2 targets and 2 raters; missingness is handled via complete-target filtering without imputation.
  Negative sample ICC values are mathematically possible in finite samples and are intentionally
  preserved without clamping to zero, as they diagnose violations of the additive variance model.
  High consistency ICC does not demonstrate absolute agreement because systematic rater differences
  are partitioned out. Statistical significance of the hypothesis F-test does not establish acceptable
  or practically adequate measurement agreement.
- Fisher's exact test is limited to ordered 2x2 tables. Its primary effect is SciPy's
  unconditional sample odds ratio. When all cells are positive, an asymptotic log-Wald confidence
  interval matching the sample odds-ratio estimator is reported. When any cell is zero, the
  confidence interval is explicitly recorded as unavailable without ad-hoc continuity corrections.
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
  and distributional approximations supplied by the researcher. Observed or post-hoc power
  calculated retrospectively from observed sample statistics is intentionally unsupported by
  scientific policy, as it is a direct 1:1 transformation of the p-value and offers no independent
  evidence of study adequacy (Hoenig & Heisey, 2001).
- Confidence intervals are reported where an independently validated method is implemented.
  Missing uncertainty is explicitly classified (`unavailable`, `not_applicable`, `not_supported`,
  or `uncomputable`) rather than silently omitted; see
  [Effect-Size Confidence Interval Gaps](EFFECT_SIZE_CI_GAPS.md) and
  [Statistical Method Contracts](STATISTICAL_METHOD_CONTRACTS.md).
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

The declared floors are pandas 1.4 (excluding 2.1.0), NumPy 1.23.5, SciPy 1.8 (excluding
1.9.2), and statsmodels 0.15; NumPy is bounded below 3 and statsmodels below 0.16. These limits
match the required OLS backend's supported numerical stack while retaining SciPy's public
studentized-range distribution for Games-Howell and Tukey-Kramer.

CI includes a Python 3.10 focused route pinned to pandas 1.5.3, NumPy 1.23.5, SciPy 1.8.0, and
statsmodels 0.15.0 for multi-group and regression tests. The full configured matrix runs on Linux
and Windows with Python 3.10 through 3.13 and resolver-selected compatible dependency versions.
