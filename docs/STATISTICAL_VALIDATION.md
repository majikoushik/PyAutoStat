# Statistical validation baseline

## Sensitivity and threshold rules

Sensitivity execution uses the execution numerical adapters. Representative pooled-variance
Student results are checked against `scipy.stats.ttest_ind(..., equal_var=True)` and the raw mean
difference is checked directly. Welch and Student share the `mean_difference` quantity when the
same outcome, grouping variable, independent design, population, and contrast are retained.
Student additionally requires an explicit equal-population-variance assumption; a Levene
nonrejection is never treated as proof.

Mann–Whitney uses `rank_biserial`, so a declared rank-distribution scenario beside a Welch mean
analysis is `different_estimand`. Paired t and paired Wilcoxon likewise retain different declared
targets, and Pearson and Spearman retain linear versus monotonic targets. No estimate difference,
relative change, or CI overlap is calculated across unlike metrics. Paired and other changed-design
specifications are incompatible; no independent test is substituted. Same-estimand comparisons use raw estimates and existing CIs,
normalize an exactly reversed two-group contrast with a disclosed sign transformation, and omit
relative change near a zero base estimate. Interval overlap is descriptive, not an inferential
test.

Practical-significance rules use only a caller-named result quantity. Signed quantities support
two-sided, positive, or negative thresholds; nonnegative metrics use a nonnegative threshold.
Finite ordered interval bounds, confidence level, named quantity, and method are validated. Point
and interval relations are separate. Missing uncertainty yields partial status. A CI within a
researcher-defined negligible region is reported descriptively and never becomes a TOST result or
formal equivalence claim. Ordinary intervals do not establish noninferiority. Statistical
significance stays in a separate field and does not choose a threshold or scenario.

## Integrated workflow invariants

The orchestration layer adds no test, effect-size calculation, or confidence-interval algorithm.
It preserves the selected method ID and declared estimand from recommendation through execution.
One successful `run()` invokes the execution path once; interpretation, reporting, audit,
and reproducibility metadata operate on the resulting object. Normality and variance diagnostics
cannot redirect a mean target to a rank target.

Recommendation-stage failures caused by observed sample limitations are surfaced as
`data_limited`; incompatible designs and unsupported scientific targets are `unsupported`.
Paired, repeated, and clustered designs never enter the independent backends. An execution that
returns an unavailable numerical result yields a failed workflow and no successful report.

The current environment exposed one baseline consistency issue: SciPy could return finite
Z-scores for a large-offset series whose variation was already considered numerically near
constant by correlation inference. Z-score outlier detection now applies the same scale-aware
floating-point threshold (`eps**0.75` relative to the mean magnitude) before reporting a count.
Near-constant values yield an unavailable count and an explicit warning. Ordinary finite varying
series continue to use SciPy's implementation; warnings and nonfinite results remain rejected.

This is an audit of the methods in the current analyzer, not a claim that any method fits a particular study. All inferential comparisons require independent observations supplied by the study design; the package cannot infer independence, pairing, randomization, or causality from values. Group and outcome rows with missing values are excluded per analysis and counted. No outlier is removed. All-missing columns have unavailable outlier counts, not zero counts.

## Method inventory and decisions

| Method | Quantity, null, and computational requirements | Audit outcome and limits |
| --- | --- | --- |
| Descriptive mean, median, sample variance/SD, quartiles/IQR, skewness, kurtosis, coefficient of variation | Numeric nonmissing values; sample variance uses `ddof=1`; quartiles use pandas interpolation; CV is `100*SD/mean` | Reference arithmetic tested. One-value variance and zero-mean CV are unavailable. Extreme-scale overflow and underflow are reported as unavailable with warnings. |
| Shapiro-Wilk | Null: a normal population; at least 3 varying values, applied here for N ≤ 5000 | Statistic and p-value retained when finite and warning-free. A nonrejection means only `not_rejected` at advisory alpha 0.05. |
| D'Agostino-Pearson | Omnibus skew/kurtosis normality null; at least 8 varying values | Finite results are retained when warning-free or accompanied only by SciPy's recognized kurtosis small-sample advisory. For N=8-19, the chi-square p-value is approximate and `analysis_warnings` records this even if a SciPy release does not warn. Other warnings or nonfinite results make the test unavailable. |
| Anderson-Darling | Normality statistic compared with supplied critical values/significance levels; at least 3 varying values by library policy | Not converted into a p-value or `is_normal` flag. When a validated 5% grid point exists, `status` and the qualified display `verdict` compare the statistic with that point. The statistic must be finite and nonnegative; the critical grid must be nonempty, finite, positive, paired with valid descending significance percentages, and increasing as significance decreases. A malformed or negative grid makes the test unavailable with a warning; SciPy's API deprecation notice is advisory. |
| Pearson correlation | Linear association; null zero population correlation for reported pairwise p-value; at least 3 paired varying values for p | Pairwise missing rows excluded. Undefined coefficients/p-values are `None`; near-constant warning makes p unavailable. |
| Spearman correlation (profile) | Descriptive rank association coefficient; at least 2 paired varying values | Matrix only; no p-values claimed. Ties use pandas/SciPy rank conventions. This remains distinct from the two-variable inferential method. |
| Kendall correlation | Concordance association coefficient; at least 2 paired varying values | Matrix only; no p-values claimed. Ties use pandas/SciPy conventions. |
| IQR outlier flag | Outside Q1−1.5 IQR or Q3+1.5 IQR | Descriptive flag only. Nonfinite bounds make count unavailable. |
| Z-score outlier flag | Absolute standardized score >3 | Zero or nonrepresentable SD makes count unavailable, including constant columns. |
| Modified Z-score/MAD flag | `0.6745*(x−median)/MAD`, absolute value >3.5 | Zero or nonfinite MAD makes count unavailable; no false zero count. |
| Welch independent t | Null: equal population means; independent groups, at least 2 usable values each | Default explicit t-test and two-group auto mean method. Uses unequal-variance standard error and Welch-Satterthwaite df. First minus second group direction. |
| Student independent t | Same mean null; additional equal-population-variance assumption | Available only by explicit `equal_var=True`; Levene does not select it. Pooled-variance SE and df `n1+n2−2`. |
| One-sample t | Population mean difference from an explicitly declared finite reference; at least 2 finite observations | Two-sided `scipy.stats.ttest_1samp`. Reports observed-minus-reference difference, SE, df, analytical raw-difference interval, and signed one-sample Cohen's d. A zero-variance sample retains the raw difference and degenerate raw interval while the t statistic, p-value, and d remain unavailable. |
| Paired Wilcoxon signed-rank | Paired-difference signed-rank distribution for an explicit first-minus-second condition contrast; at least 2 complete pairs and 2 nonzero differences | Unit-ID pairing only. Uses `zero_method="wilcox"`, omitting zero differences from ranks. Reports SciPy's two-sided statistic/p-value and matched-pairs rank-biserial correlation `(W+−W−)/(W++W−)`; no effect CI is claimed. A location-shift interpretation additionally requires a suitably symmetric difference distribution. |
| Mann-Whitney U | Null: equal underlying distributions for two independent groups; at least 2 values each | Explicit two-sided p-value. U is first-group wins plus half ties, computed from average ranks to avoid SciPy-version orientation changes. Small tied samples get an approximation warning. Not a universal median test. |
| One-way ANOVA | Null: equal group means; at least 3 groups with ≥2 observations and representable within-group variation | Standard equal-variance F test. df `(k−1, N−k)`. Explicit call only for multi-group mean comparison; Levene nonrejection does not establish equal variance. |
| Kruskal-Wallis | Null: equal group rank distributions; at least 3 groups | Tie-corrected H with chi-square approximation, df `k−1`. At least 5 per group is a conservative approximation policy, not a mathematical existence requirement. Not a universal median test. |
| Pearson chi-square | Null: categorical independence; at least 2 observed categories per variable | No Yates correction. Expected frequencies use row×column/N; df `(r−1)(c−1)`. All expected counts ≥5 is a conservative policy. The backend rejects sparse input; guided sparse 2×2 analysis selects Fisher without category merging. |
| Spearman rank inference | Null: zero population monotonic rank association for two declared numerical/ordinal variables; at least 3 complete varying pairs | Two-sided `scipy.stats.spearmanr`; ties are allowed and recorded. Rho is the signed effect. Its interval is a deterministic percentile bootstrap that resamples observation pairs together; insufficient valid resamples make only the interval unavailable. |
| Fisher exact | Null: independence in one ordered observed 2×2 table | Two-sided `scipy.stats.fisher_exact`. Reuses the shared contingency counts and retains row/column order. The effect is SciPy's unconditional sample odds ratio; a zero-cell infinite/undefined odds ratio is recorded as unavailable with status, not serialized as nonfinite. No odds-ratio CI is claimed. |
| Cohen's d | `(mean1−mean2)/pooled sample SD` | Direction reverses with group order. Pooled SD is used even alongside Welch's t-test and is not a Welch-specific standardized effect. Zero/nonrepresentable pooled SD is unavailable. |
| Rank-biserial correlation | `2*U1/(n1*n2)−1` | Positive when first group tends to have larger outcomes; ties count half. No generic qualitative magnitude band. |
| Eta-squared | Between-group sum of squares / total sum of squares | Nonnegative descriptive sample proportion; zero total SS is undefined, not a zero effect. Conventional bands remain descriptive only. |
| Rank epsilon-squared | `max(0, (H−k+1)/(N−k))` | Existing zero truncation convention retained; undefined denominator is unavailable. No correlation magnitude bands applied. |
| Cramér's V | `sqrt(chi2/(N*min(r−1,c−1)))` | Nonnegative association strength; no direction. |
| Cohen's h | `2*(asin(sqrt(p1))−asin(sqrt(p2)))` for named success outcome in 2×2 table | Sign follows first minus second group success proportion. |
| Analytical mean-difference CI | Estimate ± Student t quantile × SE with df from chosen t method | Confidence level configurable. Nonfinite estimate, SE, df, or endpoints cause a package error. This is a CI for the raw mean difference, not Cohen's d. |
| Group effect-size bootstrap CI | Percentile quantiles of independent within-group resamples | Requested/valid counts and seed recorded. At least `max(50, floor(B/2))` valid resamples required; otherwise interval `None` with warning. No BCa or multiple-comparison adjustment. |
| Categorical effect-size bootstrap CI | Percentile quantiles after resampling complete observed group/outcome rows | Preserves the observational unit, not a paired-measurement design. Empty effective categories invalidate a resample. Requested/valid counts and seed recorded; insufficient valid draws yield `None` with warning. |

The legacy `is_normal` Boolean remains for compatibility and means only that a test's p-value exceeded the fixed reference threshold. The new `status` is the preferred field, and `verdict` is a qualified presentation of the same diagnostic record. Levene uses `rejected`, `not_rejected`, or `unknown` at diagnostic alpha 0.05. Neither diagnostic establishes its null hypothesis. Explicit methods are retained even when a diagnostic raises concern, with warnings. `auto` requires `estimand="mean"` or `"distribution"`: for two groups these select Welch t or Mann-Whitney; for three or more groups, only distribution comparison selects Kruskal-Wallis. Multi-group mean auto raises an actionable error because a variance-robust mean procedure is not implemented.

## Reference cases and dependency behavior

`tests/test_statistical_correctness.py` derives expected statistics from means, sample sums of squares, U pair counts, rank sums, and contingency-cell arithmetic. Examples include Welch t and Student t on `[1,2,3,4]` versus `[3,5,7,9]`; ANOVA F=19/3 and eta-squared=38/56; Kruskal H=12.5 and rank epsilon-squared=10.5/12; chi-square=20, Cramér's V=0.5, and Cohen's h=π/3 for `[[30,10],[10,30]]`; Pearson and Spearman r=0.8 and Kendall tau=2/3 on a four-pair example. Analytical p-values use the relevant reference distribution after a separately derived statistic. Tests also cover group reversal, invalid designs, missing rows, constant data, ties, and bootstrap repeatability.

The declared minimum SciPy is 1.5. Its [Mann-Whitney documentation](https://docs.scipy.org/doc/scipy-1.5.4/reference/generated/scipy.stats.mannwhitneyu.html) states that the old default can return a half-sized two-sided p-value and a different U orientation; the implementation passes `alternative="two-sided"` and derives U1 from ranks. Newer [SciPy documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mannwhitneyu.html) describes exact versus asymptotic selection that can differ by version for small untied samples. One-sample t and Spearman pass an explicit two-sided alternative when that keyword exists; on older supported SciPy, their documented default is already two-sided. Wilcoxon passes the auto-selection option using the release's supported name (`method` on newer releases, `mode` on older releases), records that name, and makes no exactness claim. Spearman and Fisher result fields also accept the older named-result forms. [SciPy's t-test documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_ind.html) defines Welch via `equal_var=False`; [Kruskal guidance](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kruskal.html) calls five per group a typical approximation rule; [chi-square guidance](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chi2_contingency.html) treats expected-count thresholds as guidelines. Dependency constraints were not changed. The local validation environment used Python 3.12, NumPy 2.5.3, pandas 3.0.6, and SciPy 1.18.1; the full supported-version matrix requires CI or separate environments.

The guided engine provides two bounded dependent-design methods for exactly two conditions: paired t for a declared mean target and paired Wilcoxon for a declared signed-rank/distribution target. Both require an explicit unit identifier, unique unit/condition observations, and complete pairs; neither pairs rows by position. Paired t reports the first-condition-minus-second mean difference, analytical t interval, and Cohen's dz. Wilcoxon reports signed-rank inference and matched-pairs rank-biserial correlation without an effect interval. Clustered designs, repeated measures with more than two conditions, Welch ANOVA, sparse categorical tables larger than 2×2, and multiplicity adjustments remain unavailable. Extreme-scale input can still trigger NumPy/SciPy runtime warnings. The symmetric `[-1e308, 1e308]` regression case confirms that overflowed spread, normality, outlier, and histogram fields are unavailable with section warnings, while its representable mean remains available; a t-test with an undefined interval raises `InsufficientDataError`. Further numerical stabilization belongs to continued review.

## Recommendation policy

`ResearchAssistant.recommend_test()` uses the inventory above as a capability boundary. It runs no normality screen, chi-square test, correlation test, or group test. It uses complete-case counts and declared analytical types, then checks whether an existing calculation is compatible with the stated target and design. Independence and pairing are researcher-declared facts. Unknown design yields a question; a paired design requests an explicit unit identifier. Repeated and clustered designs remain blocked.

For two independent groups with a quantitative mean target, the default recommendation is Welch's t-test; the pooled Student method is an explicit alternative requiring a justified equal-variance assumption. For a distribution target and ordered numeric outcome, the recommendation is Mann-Whitney (two groups) or Kruskal-Wallis (at least three groups and five usable values per group). A multi-group mean target is blocked by default: standard ANOVA exists as a conditional explicit choice, but a variance-robust mean method does not. Normality and Levene p-values never switch the target or certify an assumption. These are methodological recommendations only; the numerical backend may still identify a warning or undefined result at execution.

An explicit finite reference, continuous outcome, independent design declaration, and mean target select one-sample t. For numeric association, an unspecified linear versus monotonic target produces one clarification question. Pearson is the runnable linear option and Spearman is the runnable monotonic option for at least three complete varying numerical/ordinal pairs; Kendall remains descriptive in the profile only. Neither marginal normality nor a diagnostic p-value switches between them. For categorical association, recommendation constructs the shared complete-case contingency grid without running a test: adequate expected counts select Pearson chi-square, while a sparse observed 2×2 table selects Fisher exact. Sparse tables larger than 2×2 remain blocked and categories are never collapsed. Observed declared missing codes block a finalized recommendation until the researcher normalizes them outside the package. Warnings and `context.assumption_checks` distinguish design confirmation, checked feasibility, and conditions that still need review.

## Execution mapping

`ResearchAssistant.analyze()` revalidates the prepared specification and recommendation, then calls one backend. It does not run alternative methods after a failure. The selected `welch_t`, `mann_whitney_u`, and `kruskal_wallis` methods map to explicit `hypothesis_tests()` calls with the stated target. One-sample t, paired Wilcoxon, and Spearman use their public `StatisticalAnalyzer` methods. Pearson maps to the existing pairwise correlation profile on the two selected columns. Chi-square maps to `categorical_association()`; Fisher exact and cross-tab reporting reuse the same shared contingency-count helper and recorded ordering. Descriptive requests map to `analyze_all()` without inferential fields. Student t and standard ANOVA remain explicit legacy backend choices; no guided override is implied by their presence in the registry.

Group sample size is the backend's complete-case count for outcome and group; excluded rows are original rows minus that count. Pearson and Spearman use complete observation pairs; chi-square and Fisher use the observed table total. Backend first-observed category order is retained. Two-group and paired contrasts are first minus second; one-sample t is observed mean minus reference. Analytical t intervals are labeled with their raw mean-difference quantities. Bootstrap effect intervals are labeled with their own metrics and never relabeled as raw differences. Spearman's rho interval uses paired-observation resampling. Wilcoxon and Fisher expose no effect interval, so their otherwise valid interpretations are partial. Pearson currently has no confidence interval. The stored alpha is an interpretation threshold; it does not change the calculated p-value.

The guided path requests 499 backend bootstrap resamples with effective seed 0 by default, or the specification's seed when provided. Both settings and valid-resample counts are recorded; a backend-unavailable interval remains `None` with its warning. The engine checks core statistics, p-values, effect estimates, interval bounds and confidence level before marking an analysis available. A known backend numerical failure returns an unavailable result with the originating specification and recommendation. These checks do not prove statistical assumptions. The separate provenance auditor checks recorded results against reports and exports without recalculating the analysis. Existing extreme-scale analyzer warnings remain possible; further numerical stabilization is outside this completion.

## Interpretation policy

`ResearchAssistant.interpret(result)` reads the original `AnalysisResult`. Supported guided methods include descriptive profile, the established group comparisons, Pearson correlation, Pearson chi-square, one-sample t, paired Wilcoxon, inferential Spearman, and Fisher exact. Valid internal Student and standard ANOVA adapter results also have templates, although the guided selector does not choose them. The profile-wide Spearman and Kendall matrices remain descriptive and are not reused as inferential results.

The unrounded recorded p-value is compared to the specification's alpha with `p < alpha`; a value equal to alpha does not reject. Missing or invalid p-values produce no threshold conclusion. Display formatting preserves the numeric source; computational zero is shown as an inequality with a machine-precision warning. A nonsignificant result is insufficient evidence against the null, not proof of no effect or equivalence. No adjustment or practical-importance threshold is claimed without recorded evidence.

The deterministic hypothesis narrative combines that unchanged decision with the existing effect
label in four presentation quadrants. Missing or unsupported effect labels do not default to a
small effect. Assumption severity is also presentation-only: rejected normality uses the stored
per-group size with boundaries below 15, 15–29, and at least 30; a rejected equal-variance
diagnostic is informational under Welch and a warning for a pooled method. These grades do not
change the selected method, estimand, diagnostic, statistic, or p-value.

All researcher-facing narration is validated as deterministic classification of stored records,
not as a new evidential procedure. Conventional magnitude names describe fixed communication
bands and do not establish practical importance. Practical-significance prose requires the
researcher's declared threshold; sensitivity prose preserves comparability and never equates
different estimands. Design-dependent conditions such as independence, pairing, sampling, and
causal identification remain for researcher verification. Narrative output cannot establish
causation and does not replace the structured result used for audit or replay.

Two-group and paired estimate direction follows the stored first-minus-second contrast; one-sample direction is observed minus reference. Analytical t intervals are for the named raw mean differences; separate bootstrap intervals belong to their named effects. Wilcoxon is described as a paired signed-rank/distribution analysis, not universally as a median test. Spearman is described as monotonic rank association, without linearity or causality claims. Fisher reports an ordered-table sample odds ratio, not a risk ratio or causal effect. Pearson, Wilcoxon, and Fisher currently lack a primary-effect CI, so interpretation is partial with an explicit uncertainty limitation. Intervals require finite ordered bounds, a valid recorded confidence level, the correct quantity, and the expected construction method. An analytical t interval must contain its raw point estimate; a percentile-bootstrap interval need not contain its original estimate. A valid raw mean difference remains interpretable if Cohen's d is unavailable, yielding partial status. Bootstrap intervals are not treated as equivalent threshold tests. Omnibus significance identifies no specific pairwise difference; Cramer's V has no sign. Diagnostic non-rejection does not prove normality or equal variance, and a declared independent design is not mathematical verification. Associations and group differences alone do not establish causation.
