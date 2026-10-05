# PyAutoStat Feature-by-Feature Usability Audit and Market Benchmarking

> **A systematic usability audit of every supported statistical and data-profiling workflow in PyAutoStat, evaluated against researcher goals, design contracts, directional clarity, sample accounting, and leading scientific software benchmarks.**

---

## 1. Executive Summary and Usability Philosophy

PyAutoStat is an explainable and reproducible research analysis assistant for pandas DataFrames. It bridges the gap between raw data manipulation and defensible scientific reporting:

$$\text{Research Question / Design} \longrightarrow \text{Recommendation} \longrightarrow \text{Validated Execution} \longrightarrow \text{Effect Size + Uncertainty} \longrightarrow \text{Deterministic Interpretation} \longrightarrow \text{Audit} \longrightarrow \text{Report}$$

### The Core Usability Mandate
1. **Protect statistical validity and the stated estimand**: A diagnostic p-value must never silently alter the researcher's scientific question (e.g. converting a mean difference question to a rank test solely because Shapiro-Wilk rejects normality).
2. **Clarify rather than guess**: Independence, pairing, clustering, and event orientations cannot be safely inferred from numerical values alone. When critical design facts are absent, PyAutoStat pauses with structured clarification requests (`needs_input`) rather than making unverified guesses.
3. **Transparent orientation and sign**: Every inferential comparison must explicitly label contrast direction ($A - B$, first minus second condition, modeled event vs non-event, or reference baseline).
4. **Complete sample accounting**: Users must always be able to trace original rows, complete analyzed cases, excluded rows, group sample sizes, and complete pairs/units.
5. **Separate analysis execution from workflow completeness**: A valid, fully executed statistical calculation can exist inside an incomplete workflow (such as when `audit=False`), and this distinction must be clear across terminal and report presentation.

---

## 2. Market Benchmarking Lens

PyAutoStat does not aim to duplicate the full algorithmic surface of general-purpose statistical libraries. Instead, it solves the workflow cohesion and scientific governance challenges where existing tools leave researchers on their own:

| Benchmark Ecosystem | Observed Strengths | Observed Usability Gaps | PyAutoStat Value Proposition |
| --- | --- | --- | --- |
| **Pingouin** | Concise, method-specific calls; pandas-friendly DataFrame inputs; easy for known methods. | Relies on the user knowing the exact procedure; lacks integrated research question clarification, automatic sample accounting, decision audit trails, and multi-format reporting. | Preserves declared estimands; validates long-format panels; handles missing design facts deterministically; produces audited, reproducible reports. |
| **Statsmodels** | Deep econometric and generalized linear modeling; rich parameter summaries and residual diagnostics. | Verbose model syntax; requires manual post-estimation for effect-size CIs; no automated design clarification; in-sample $R^2$ easily mistaken for predictive validity without guardrails. | Provides focused, unpenalized regression models with automatic heteroscedasticity diagnostics (HC3), standardized betas, VIF collinearity checks, explicit non-causal warnings, and plain-language interpretation. |
| **SciPy (`scipy.stats`)** | Foundational, high-performance numerical routines for arrays. | Low-level arrays only; strips DataFrame metadata; lacks effect sizes, confidence intervals for effects, directional labeling, sample exclusion tracking, and assumption reporting. | Wraps SciPy numerical routines inside design-aware research contracts with analytical/bootstrap effect-size CIs, contrast direction preservation, and complete row accounting. |
| **JASP / jamovi** | Excellent progressive disclosure; clear presentation tables; assumptions and effect sizes displayed alongside test statistics. | GUI-dependent; difficult to embed in automated data pipelines, CI/CD, or batch scripts. | Brings the progressive disclosure, table clarity, and assumption transparency of JASP/jamovi directly into Python consoles and automated script workflows. |
| **YData Profiling** | Fast, visual exploratory data analysis (EDA) HTML dashboards. | Focuses on exploratory inspection; disconnected from formal hypothesis testing; does not lead into defensible inferential workflows. | Lightweight, offline dataset profiling that directly informs research question specification and guides defensible inferential method selection. |

---

## 3. Usability Audit Summary Matrix

| # | Feature Family | Recommended Entry Point | Core Estimand | Contrast / Sign Orientation | Verdict | Priority | Implemented Polish |
|---|---|---|---|---|---|---|---|
| **0** | **Data Profiling & Descriptives** | `assistant.profile()` | Univariate distributions, missingness, percentiles | N/A | KEEP | P1 | Clarified advisory nature of normality and outlier review cues. |
| **1** | **One-Sample Mean vs Reference** | `run(objective="compare_reference", ...)` | Population mean relative to reference ($\mu - \mu_0$) | Observed sample mean minus reference value ($\bar{x} - \mu_0$) | POLISH | P1 | Added to cookbook table; explicit reference baseline labeling. |
| **2** | **Two Independent Groups** | `run(objective="compare_groups", ...)` | Mean difference (Welch t) or stochastic superiority (Mann-Whitney) | Explicit signed contrast: `'group1' - 'group2'` | KEEP | P1 | Clarified Welch as guided default; Student as explicit alternative. |
| **3** | **Paired Two-Condition Analysis** | `run(design="paired", ...)` | Mean paired difference ($\mu_D$) or signed-rank distribution | Ordered contrast: `'cond1' - 'cond2'` within `unit_id` | KEEP | P1 | Highlighted long-format panel requirements and complete-pair accounting. |
| **4** | **3+ Independent Groups** | `run(objective="compare_groups", ...)` | Equality of group means (Welch ANOVA) or rank distributions | Omnibus F / H; pairwise follow-ups follow declared contrast order | KEEP | P1 | Group summary table + complete multiplicity-adjusted pairwise table. |
| **5** | **Repeated Measures (3+ conditions)** | `run(design="repeated", ...)` | Equality of repeated condition means or distributions | Omnibus F; pairwise follow-ups follow condition order | KEEP | P1 | Mauchly sphericity check + Greenhouse-Geisser corrected degrees of freedom. |
| **6** | **Two-Way Factorial ANOVA** | `assistant.two_way_anova(...)` | Main factor effects and interaction effect ($A \times B$) | Unweighted estimated marginal means; simple effects | KEEP | P1 | Promoted focused facade; verified Type II/III sum of squares guidance. |
| **7** | **Bivariate & Partial Associations** | `run(objective="association", ...)` | Linear, monotonic, rank concordance, or partial correlation | Correlation sign ($-1 \le r \le +1$); binary positive level (+1) | KEEP | P1 | Emphasized non-causal association; partial correlation control accounting. |
| **8** | **Categorical Independence** | `run(objective="association", ...)` | Independence / association (Chi-square or Fisher exact) | Contingency marginals; 2x2 Odds Ratio ($ad/bc$) | KEEP | P1 | Contingency table display; explicit Fisher exact sparse fallback. |
| **9** | **OLS Linear Regression** | `run(objective="regression", ...)` | Conditional mean coefficients ($\beta_j$) and model $R^2$ | Coefficient sign relative to reference level | KEEP | P1 | Clarified HC3 covariance role; standardized betas; non-causal framing. |
| **10** | **Binary Logistic Regression** | `run(objective="regression", ...)` | Log-odds coefficients ($\beta_j$) and Odds Ratios ($e^{\beta_j}$) | Modeled event vs non-event; $OR > 1$ indicates higher odds | KEEP | P1 | Explicit event orientation; clarified absence of arbitrary classification thresholds. |
| **11** | **Scale Reliability** | `assistant.reliability(items=[...])` | Sample-variance Cronbach's alpha ($\alpha$) | Scale score orientation; explicit reverse scoring | KEEP | P1 | Respondent bootstrap CI; item-total diagnostics; anti-deletion guidance. |
| **12** | **Inter-Rater Reliability (ICC)** | `assistant.intraclass_correlation(...)` | Shrout & Fleiss / McGraw & Wong ICC variants | Reliability / agreement ratio ($-\frac{1}{k-1} \le \text{ICC} \le 1$) | KEEP | P1 | Preserved negative estimates; design clarification (`needs_input`) on missing model/unit. |

---

## 4. Feature-by-Feature Detailed Usability Audits

### Feature 0: Data Profiling and Descriptive Intelligence
- **User Goal**: Inspect distributions, percentiles, missingness patterns, categorical frequencies, and data-quality anomalies prior to inferential testing.
- **Recommended Entry Point**: `assistant.profile()` (or `assistant.summarize(mode="profile"|"story")`, `assistant.frequency_table("col")`, `assistant.cross_tab("row", "col")`).
- **Direct / Advanced Entry Point**: `StatisticalAnalyzer.analyze_all()`, `frequency_table()`, `cross_tabulation()`.
- **Required Inputs**: Input DataFrame. Specific column names for frequency tables and cross-tabulations.
- **Input Shape**: Arbitrary rectangular pandas DataFrame; preserves column types and row counts without mutation or downsampling.
- **Estimand / Target**: Descriptive statistics (mean, SD, IQR, median, P5, P25, P50, P75, P95) and categorical breakdown percentages; no inferential hypothesis test.
- **Direction / Orientation**: N/A.
- **Primary Result**: Overview counts, valid percentage, missing percentage, quantile breakdown, skewness state.
- **Effect Size / Uncertainty**: N/A (purely descriptive).
- **Diagnostics & Assumptions**: Normality tests (D'Agostino-Pearson, Shapiro-Wilk) and outlier counts are advisory screening cues. They do **not** automatically trigger row deletion or force non-parametric tests.
- **Sample Accounting**: Total rows, valid rows, missing rows, valid percentage vs total percentage.
- **Terminal Presentation**: Polished Rich tables with clear section headers via `show(profile)`.
- **Report Presentation**: Included as optional profile appendix in research exports when `include_profile=True`.
- **Error / Needs Input Experience**: Validates column existence; rejects empty DataFrames or duplicate column names.
- **Market Benchmark**: Unlike YData Profiling which renders massive visual dashboards, PyAutoStat provides clean, fast terminal and dictionary views designed to lead directly into research questions.
- **Verdict**: **KEEP** | **Priority**: P1 | **Action**: Document connection between profiling and question builder.

---

### Feature 1: One-Sample Mean Comparison vs Reference
- **User Goal**: Compare the population mean of a continuous variable against a researcher-specified historical or benchmark standard ($H_0: \mu = \mu_0$).
- **Recommended Entry Point**: `assistant.run(objective="compare_reference", outcome="score", reference_value=70.0, estimand="mean")`.
- **Direct / Advanced Entry Point**: `StatisticalAnalyzer.one_sample_t_test(outcome="score", reference_value=70.0)`.
- **Required Inputs**: `objective="compare_reference"`, `outcome`, `reference_value` (finite scalar), `estimand="mean"`. No predictor.
- **Input Shape**: Single continuous column with at least 2 usable observations.
- **Estimand / Target**: Difference between population mean and reference value ($\mu - \mu_0$).
- **Direction / Orientation**: Explicit signed contrast: observed sample mean minus reference value ($\bar{x} - \mu_0$).
- **Primary Result**: Mean difference, Student's one-sample t-statistic, degrees of freedom, p-value.
- **Effect Size / Uncertainty**: Cohen's $d = (\bar{x} - \mu_0) / s$ with exact noncentral-t confidence interval; analytical CI for mean difference.
- **Diagnostics & Assumptions**: Sample normality assessment (Shapiro-Wilk / D'Agostino-Pearson).
- **Sample Accounting**: Original rows, analyzed complete cases, excluded rows.
- **Terminal Presentation**: Explicit `Observed - Reference`, `Reference Value`, `Sample Mean`, and `Cohen's d` metrics.
- **Report Presentation**: Canonical table reporting reference baseline and sample estimate.
- **Error / Needs Input Experience**: `needs_input` if `reference_value` is missing; rejects non-finite reference values.
- **Market Benchmark**: SciPy's `ttest_1samp` returns only t and p. PyAutoStat adds noncentral-t CIs for Cohen's d, signed direction labeling, and sample accounting.
- **Verdict**: **POLISH** | **Priority**: P1 | **Action**: Added to cookbook table in API reference; ensured reference value is prominently displayed in terminal view.

---

### Feature 2: Two Independent Groups Comparison
- **User Goal**: Evaluate whether two distinct groups differ in their central tendency (mean) or rank distribution.
- **Recommended Entry Point**: `assistant.run(objective="compare_groups", outcome="score", predictor="group", estimand="mean"|"distribution", design="independent")`.
- **Direct / Advanced Entry Point**: `StatisticalAnalyzer.welch_t_test()`, `student_t_test()`, `mann_whitney_u()`.
- **Required Inputs**: `outcome`, `predictor`, `design="independent"`, `estimand` (`"mean"` for Welch t; `"distribution"` for Mann-Whitney U).
- **Input Shape**: Long format; binary grouping predictor (exactly 2 observed levels); numeric outcome.
- **Estimand / Target**: Population mean difference ($\mu_1 - \mu_2$) or rank stochastic superiority ($P(X_1 > X_2) + 0.5 P(X_1 = X_2)$).
- **Direction / Orientation**: Explicitly signed contrast: `'group1' - 'group2'` according to recorded `group_order`.
- **Primary Result**: Mean difference / Mann-Whitney U, test statistic, p-value.
- **Effect Size / Uncertainty**: Cohen's d with bootstrap or analytical CI (mean); rank-biserial correlation $r_{rb}$ with bootstrap CI (rank).
- **Diagnostics & Assumptions**: Levene's variance equality test (reported for transparency; does not alter estimand), normality per group. Welch t is the guided default for means; pooled Student t requires explicit equal-variance justification.
- **Sample Accounting**: Original rows, analyzed rows, excluded rows, group sample sizes ($n_1, n_2$).
- **Terminal Presentation**: GROUP SUMMARY table ($n_1, n_2$), Contrast metric, Mean difference, Cohen's d, p-value.
- **Report Presentation**: Comprehensive APA/IEEE style tables with group means and standard deviations.
- **Error / Needs Input Experience**: Pauses with `needs_input` if `design` is omitted; rejects predictors with $\ne 2$ levels.
- **Market Benchmark**: Pingouin requires separate function calls (`ttest` vs `mwu`); PyAutoStat uses a unified, design-first question contract where the estimand dictates the method.
- **Verdict**: **KEEP** | **Priority**: P1 | **Action**: Maintain estimand stability; Levene p-values never silently convert mean questions to rank tests.

---

### Feature 3: Paired Two-Condition Analysis
- **User Goal**: Evaluate changes or differences across two repeated measurements taken on the same observational units (e.g. pre vs post treatment).
- **Recommended Entry Point**: `assistant.run(objective="compare_groups", outcome="score", predictor="time", design="paired", unit_id="subject_id", condition_order=("pre", "post"), estimand="mean"|"distribution"|"proportion")`.
- **Direct / Advanced Entry Point**: `StatisticalAnalyzer.paired_t_test()`, `paired_wilcoxon()`, `mcnemar()`.
- **Required Inputs**: `outcome`, `predictor`, `design="paired"`, `unit_id`, `condition_order`, `estimand`.
- **Input Shape**: Long-format panel; unit identifier column; condition column with exactly two levels; outcome column.
- **Estimand / Target**: Population mean paired difference ($\mu_D = \mu_1 - \mu_2$), signed-rank difference distribution, or paired binary marginal proportion difference.
- **Direction / Orientation**: Strictly ordered: `condition_order[0] - condition_order[1]` (first condition minus second condition).
- **Primary Result**: Mean difference, Wilcoxon W, or McNemar discordant count difference.
- **Effect Size / Uncertainty**: Cohen's $d_z = \bar{D} / s_D$ with exact noncentral-t CI; rank-biserial correlation with bootstrap CI; matched odds ratio.
- **Diagnostics & Assumptions**: Distribution of paired differences; zero-difference counts; complete-pair filtering.
- **Sample Accounting**: Total rows, complete pairs retained, incomplete/unmatched units excluded, zero differences counted.
- **Terminal Presentation**: Explicit Contrast metric (`'cond1' - 'cond2' within 'unit_id'`), Complete Pairs, Incomplete Units.
- **Report Presentation**: Full table reporting paired difference, standard error, and pair accounting.
- **Error / Needs Input Experience**: Pauses with `needs_input` if `unit_id` or `condition_order` is missing; catches duplicate unit-condition rows.
- **Market Benchmark**: SciPy's `ttest_rel` operates on raw arrays and loses unit linkage. PyAutoStat enforces unit tracking, verifies complete pairs, and prevents silent ordering reversals.
- **Verdict**: **KEEP** | **Priority**: P1 | **Action**: Reinforced condition order requirement in docs and error guidance.

---

### Feature 4: 3+ Independent Groups (Multigroup Comparison)
- **User Goal**: Compare three or more independent groups on a continuous mean or ordinal rank distribution.
- **Recommended Entry Point**: `assistant.run(objective="compare_groups", outcome="score", predictor="treatment", estimand="mean"|"distribution", design="independent")`.
- **Direct / Advanced Entry Point**: `StatisticalAnalyzer.welch_anova()`, `one_way_anova()`, `kruskal_wallis()`.
- **Required Inputs**: `outcome`, `predictor` ($\ge 3$ levels), `design="independent"`, `estimand`.
- **Input Shape**: Long format; grouping predictor with 3 or more levels; numeric outcome.
- **Estimand / Target**: Equality of group means ($H_0: \mu_1 = \dots = \mu_k$) or rank distributions.
- **Direction / Orientation**: Omnibus test is non-directional; pairwise post-hoc contrasts have explicit pairwise direction ($A - B$).
- **Primary Result**: Omnibus Welch F / classical F / Kruskal-Wallis H statistic, degrees of freedom, p-value.
- **Effect Size / Uncertainty**: Eta-squared / epsilon-squared; pairwise differences with simultaneous multiplicity-controlled confidence intervals (Games-Howell, Tukey-Kramer, or Dunn-Holm).
- **Diagnostics & Assumptions**: Group variance homogeneity (Levene); positive group variances. Welch ANOVA is the guided default for means.
- **Sample Accounting**: Total rows, complete cases analyzed, excluded rows, group sample sizes ($n_1, \dots, n_k$).
- **Terminal Presentation**: Omnibus result, Group Summary table, and complete Pairwise Comparisons table with adjusted p-values and simultaneous CIs.
- **Report Presentation**: Canonical ANOVA table accompanied by post-hoc contrast matrix.
- **Error / Needs Input Experience**: Prompts for `design` (`needs_input`); blocks groups with zero variance in Welch ANOVA.
- **Market Benchmark**: Statsmodels and Scikit-posthocs require manual chaining of ANOVA and post-hoc routines; PyAutoStat returns omnibus tests and multiplicity-controlled pairwise contrasts in a single unified result.
- **Verdict**: **KEEP** | **Priority**: P1 | **Action**: Preserved complete pairwise contrasts in terminal and report outputs.

---

### Feature 5: Repeated Measures (3+ Conditions)
- **User Goal**: Evaluate continuous or ordinal differences across three or more repeated conditions or time points on the same units.
- **Recommended Entry Point**: `assistant.run(objective="compare_groups", outcome="score", predictor="session", design="repeated", unit_id="subject_id", condition_order=("t1", "t2", "t3"), estimand="mean"|"distribution")`.
- **Direct / Advanced Entry Point**: `StatisticalAnalyzer.repeated_measures_anova()`, `friedman_test()`.
- **Required Inputs**: `outcome`, `predictor`, `design="repeated"`, `unit_id`, `condition_order` ($\ge 3$ conditions), `estimand`.
- **Input Shape**: Long-format repeated measures panel; one row per unit-condition combination.
- **Estimand / Target**: Equality of condition means across repeated measures ($H_0: \mu_1 = \dots = \mu_k$) or condition rank distributions.
- **Direction / Orientation**: Omnibus is omnibus; pairwise follow-ups follow declared condition order contrasts.
- **Primary Result**: RM-ANOVA F-test or Friedman Q-statistic with degrees of freedom and p-value.
- **Effect Size / Uncertainty**: Partial eta-squared $\eta_p^2$ with exact noncentral-F CI; Kendall's W concordance with bootstrap CI; pairwise Holm-adjusted differences with CIs.
- **Diagnostics & Assumptions**: Mauchly's sphericity test; Greenhouse-Geisser and Huynh-Feldt epsilon corrections automatically reported and applied when sphericity is violated.
- **Sample Accounting**: Total observations, complete units retained across all conditions, incomplete units excluded.
- **Terminal Presentation**: Omnibus F, Mauchly's W, Greenhouse-Geisser epsilon and corrected p-value, Condition Summaries, Pairwise Contrasts.
- **Report Presentation**: Integrated ANOVA table, sphericity diagnostics, and post-hoc contrasts.
- **Error / Needs Input Experience**: Missing `unit_id` or `condition_order` yields `needs_input`; < 3 conditions yields `data_limited`/`unsupported`; duplicate unit-condition yields error.
- **Market Benchmark**: Pingouin's `rm_anova` is popular; PyAutoStat matches its sphericity handling while adding design validation, complete-unit tracking, noncentral CI for partial eta-squared, and audit trails.
- **Verdict**: **KEEP** | **Priority**: P1 | **Action**: Clarified in docs that RM-ANOVA evaluates a single within-subject factor and is distinct from mixed-model ANOVA.

---

### Feature 6: Two-Way Factorial ANOVA
- **User Goal**: Evaluate main effects of two categorical factors and their interaction on a continuous outcome.
- **Recommended Entry Point**: `assistant.two_way_anova(outcome="score", factor_a="treatment", factor_b="gender", sum_of_squares="type2"|"type3")`.
- **Direct / Advanced Entry Point**: `StatisticalAnalyzer.two_way_anova()`.
- **Required Inputs**: `outcome`, `factor_a`, `factor_b`, `design="independent"`, `estimand="mean"`.
- **Input Shape**: Independent observations; exactly two categorical factors; continuous outcome; fully crossed design.
- **Estimand / Target**: Main effects of Factor A, Factor B, and their interaction ($A \times B$).
- **Direction / Orientation**: Unweighted estimated marginal means per factor level; simple effects contrasts.
- **Primary Result**: ANOVA table with Type II or Type III Sums of Squares, df, Mean Squares, F-statistics, and p-values for main effects and interaction.
- **Effect Size / Uncertainty**: Partial eta-squared $\eta_p^2$ with exact noncentral-F confidence intervals for every term.
- **Diagnostics & Assumptions**: Cell counts, empty cell detection, residual degrees of freedom positivity.
- **Sample Accounting**: Total rows, complete cases analyzed, excluded rows, cell counts ($n_{jk}$).
- **Terminal Presentation**: Main and interaction F-tests, partial $\eta_p^2$ CIs, complete ANOVA Effects table, Cell Summaries.
- **Report Presentation**: Canonical ANOVA table, estimated marginal means, and interaction interpretation.
- **Error / Needs Input Experience**: Conflicting `factor_a`/`factor_b` vs `factors` raises clear error; empty cells raise `InsufficientDataError`.
- **Market Benchmark**: Statsmodels `ols` + `anova_lm` requires explicit formula syntax and manual effect-size calculation; PyAutoStat provides a one-line focused call with automated partial eta-squared CIs and interaction interpretation.
- **Verdict**: **KEEP** | **Priority**: P1 | **Action**: Documented preference for focused facade `assistant.two_way_anova()` over generic `run()`.

---

### Feature 7: Continuous and Ordered Bivariate Associations
- **User Goal**: Measure the strength and direction of linear, monotonic, rank-concordant, or partial association between variables.
- **Recommended Entry Point**: `assistant.run(objective="association", outcome="y", predictor="x", estimand="linear"|"monotonic"|"point_biserial"|"partial_linear")`.
- **Direct / Advanced Entry Point**: `StatisticalAnalyzer.pearson_correlation()`, `spearman_correlation()`, `kendall_tau()`, `partial_correlation()`.
- **Required Inputs**: `outcome`, `predictor`, `estimand`, `design="independent"`; `controls` for partial linear; `event_level` for point-biserial.
- **Input Shape**: Paired bivariate observations (independent rows).
- **Estimand / Target**: Population correlation coefficient: Pearson $r$, Spearman $\rho$, Kendall $\tau_b$, point-biserial $r_{pb}$, partial Pearson $r_{xy.z}$.
- **Direction / Orientation**: Sign of correlation coefficient ($-1 \le r \le +1$); point-biserial is oriented by declared `positive_level` (+1 coding).
- **Primary Result**: Correlation coefficient, test statistic (t or z), p-value.
- **Effect Size / Uncertainty**: Fisher z asymptotic confidence interval (Pearson); case-resampling bootstrap confidence interval (Spearman, Kendall, point-biserial, partial Pearson).
- **Diagnostics & Assumptions**: Bivariate normality / linear association cues; sample size checks.
- **Sample Accounting**: Total rows, complete paired rows analyzed, excluded rows.
- **Terminal Presentation**: Correlation coefficient, confidence interval, sample size, p-value; binary level breakdown for point-biserial; controls list for partial Pearson.
- **Report Presentation**: Summary table with effect size, CI, and non-causal interpretation.
- **Error / Needs Input Experience**: Missing `event_level` for binary association yields `needs_input`; non-quantitative controls raise error.
- **Market Benchmark**: SciPy correlation functions do not provide bootstrap CIs or direction metadata; PyAutoStat provides verified bootstrap CIs, explicit binary coding orientation, and partial correlation with covariate accounting.
- **Verdict**: **KEEP** | **Priority**: P1 | **Action**: Reiterate in limitations that association does not imply causation, and partial correlation conditions on quantitative covariates without establishing causal deconfounding.

---

### Feature 8: Categorical Independence (Chi-Square & Fisher Exact)
- **User Goal**: Evaluate association or independence between two categorical variables.
- **Recommended Entry Point**: `assistant.run(objective="association", outcome="y", predictor="x", estimand="categorical_independence", design="independent")`.
- **Direct / Advanced Entry Point**: `StatisticalAnalyzer.chi_square_test()`, `fisher_exact_test()`.
- **Required Inputs**: `outcome`, `predictor`, `objective="association"`, `estimand="categorical_independence"`.
- **Input Shape**: Two categorical columns; independent rows.
- **Estimand / Target**: Categorical independence / deviation from expected marginal distribution.
- **Direction / Orientation**: 2x2 odds ratio orientation is defined by row and column levels.
- **Primary Result**: Pearson Chi-square statistic or Fisher exact two-sided p-value.
- **Effect Size / Uncertainty**: Cramer's V with bootstrap CI; sample Odds Ratio with asymptotic log-Wald CI (when no cell is zero).
- **Diagnostics & Assumptions**: Minimum expected cell count; percentage of cells with expected count < 5. Fisher exact is automatically recommended as fallback for sparse 2x2 tables.
- **Sample Accounting**: Total rows, complete cases analyzed, excluded rows, contingency table marginal counts.
- **Terminal Presentation**: Contingency Table of observed counts, Expected Counts diagnostic, Cramer's V / Odds Ratio, p-value.
- **Report Presentation**: Contingency table, test statistic, effect size, and expected counts verification.
- **Error / Needs Input Experience**: Sparse tables larger than 2x2 return `unsupported` rather than unvalidated ad-hoc approximations.
- **Market Benchmark**: SciPy `chi2_contingency` returns raw arrays; PyAutoStat formats the full contingency table with row/column labels, checks expected cell policies, provides Cramer's V CIs, and falls back to Fisher for 2x2.
- **Verdict**: **KEEP** | **Priority**: P1 | **Action**: Documented odds ratio orientation clearly in Fisher exact output and audit document.

---

### Feature 9: Ordinary Least Squares (OLS) Regression
- **User Goal**: Model the conditional mean of a continuous outcome as a linear function of one or more predictors.
- **Recommended Entry Point**: `assistant.run(objective="regression", outcome="y", predictors=["x1", "x2"], estimand="conditional_mean", design="independent")`.
- **Direct / Advanced Entry Point**: `StatisticalAnalyzer.linear_regression()`.
- **Required Inputs**: `outcome` (continuous), `predictors` (sequence of column names), `objective="regression"`, `estimand="conditional_mean"`.
- **Input Shape**: Independent observations; continuous outcome; continuous or categorical predictors.
- **Estimand / Target**: Conditional mean coefficients ($\beta_j$), in-sample $R^2$, and model F.
- **Direction / Orientation**: Sign of regression coefficient indicates expected outcome change per unit increase in predictor (or relative to reference category).
- **Primary Result**: Model $R^2$, adjusted $R^2$, model F-test and p-value, coefficient estimates and standard errors.
- **Effect Size / Uncertainty**: Coefficient confidence intervals; standardized coefficients ($\beta^*$) for continuous predictors; bootstrap CI for $R^2$.
- **Diagnostics & Assumptions**: Breusch-Pagan heteroscedasticity test; Jarque-Bera residual normality test; Variance Inflation Factors (VIF) for collinearity; condition number; Cook's distance influence cues. HC3 robust covariance option supported.
- **Sample Accounting**: Total rows, complete-case analyzed observations, excluded rows.
- **Terminal Presentation**: Model summary (R2, F, p, residual SE), Coefficients table (Estimate, SE, CI, t, p), Diagnostics table (heteroscedasticity, normality, VIF).
- **Report Presentation**: Canonical regression summary, coefficients table, and diagnostics.
- **Error / Needs Input Experience**: Multicollinear or rank-deficient designs raise `InvalidDataError` or return `data_limited`; invalid `covariance_type` is rejected.
- **Market Benchmark**: Statsmodels provides massive statistical output; PyAutoStat organizes it into key metrics, coefficients with standardized betas, actionable diagnostics (VIF, heteroscedasticity), and explicit non-causal interpretations.
- **Verdict**: **KEEP** | **Priority**: P1 | **Action**: Ensured documentation highlights that HC3 is a standard error adjustment and in-sample $R^2$ is not out-of-sample predictive validity.

---

### Feature 10: Binary Logistic Regression
- **User Goal**: Model the probability / log-odds of a binary event as a function of predictors.
- **Recommended Entry Point**: `assistant.run(objective="regression", outcome="y", predictors=["x1", "x2"], event_level="Yes", estimand="event_probability", design="independent")`.
- **Direct / Advanced Entry Point**: `StatisticalAnalyzer.logistic_regression()`.
- **Required Inputs**: `outcome` (exactly 2 observed levels), `predictors`, `event_level`, `objective="regression"`, `estimand="event_probability"`.
- **Input Shape**: Independent observations; binary outcome; continuous or categorical predictors.
- **Estimand / Target**: Log-odds coefficients ($\beta_j$), odds ratios ($e^{\beta_j}$), and McFadden pseudo-$R^2$.
- **Direction / Orientation**: Explicitly oriented toward `event_level` relative to `non_event_level`; odds ratio > 1 indicates increased odds of event.
- **Primary Result**: Likelihood-ratio chi-square test, p-value, McFadden pseudo-$R^2$, odds ratios and CIs.
- **Effect Size / Uncertainty**: Odds ratios ($OR = e^\beta$) with analytical confidence intervals; log-odds coefficients with standard errors.
- **Diagnostics & Assumptions**: Model convergence status, separation checks, collinearity (condition number), event counts and event rate.
- **Sample Accounting**: Total rows, complete-case observations, excluded rows, event count, non-event count, event rate %.
- **Terminal Presentation**: Modeled Event and Event Accounting metrics, Likelihood Ratio Test, Odds Ratios table (Estimate, OR, CI, z, p).
- **Report Presentation**: Logistic regression table with explicit odds ratios and event labeling.
- **Error / Needs Input Experience**: Missing `event_level` triggers `needs_input`; non-binary outcomes (> 2 levels) raise error.
- **Market Benchmark**: Scikit-learn focuses on classification metrics (AUC, accuracy) with default regularization; PyAutoStat focuses on unpenalized inferential log-odds, likelihood-ratio tests, odds ratios with CIs, and explicit event orientation without arbitrary decision thresholds.
- **Verdict**: **KEEP** | **Priority**: P1 | **Action**: Documented that no classification cutoffs or ROC curves are generated, emphasizing the inferential focus.

---

### Feature 11: Scale Reliability (Cronbach's Alpha)
- **User Goal**: Evaluate the internal consistency and inter-item correlation structure of a multi-item psychometric scale.
- **Recommended Entry Point**: `assistant.reliability(items=["i1", "i2", "i3"])` (or `run(objective="reliability", items=[...], estimand="internal_consistency")`).
- **Direct / Advanced Entry Point**: `StatisticalAnalyzer.cronbach_alpha()`.
- **Required Inputs**: `items` (sequence of $\ge 2$ numeric column names).
- **Input Shape**: Wide format; each column is a scored item; rows are respondents.
- **Estimand / Target**: Sample-variance Cronbach's alpha ($\alpha = \frac{k}{k-1} (1 - \frac{\sum s_i^2}{s_T^2})$).
- **Direction / Orientation**: Scale direction; reverse scoring can be declared explicitly (`reverse_scoring=[...]` or `{...}`).
- **Primary Result**: Cronbach's alpha estimate, mean inter-item correlation.
- **Effect Size / Uncertainty**: Respondent-row case-resampling bootstrap confidence interval for alpha.
- **Diagnostics & Assumptions**: Corrected item-total correlations, alpha-if-deleted values, inter-item correlation matrix, item missingness.
- **Sample Accounting**: Total rows, complete respondents analyzed, excluded respondents with missing items.
- **Terminal Presentation**: Scale Items count, Respondents (N) complete cases, Cronbach's alpha, Bootstrap CI, Item-Level Diagnostics table.
- **Report Presentation**: Reliability summary table, item statistics table, inter-item correlation matrix.
- **Error / Needs Input Experience**: < 2 items raises error; non-numeric items raise error; reverse scoring out of observed bounds raises error.
- **Market Benchmark**: Pingouin's `cronbach_alpha` returns just alpha and CI; PyAutoStat provides comprehensive item diagnostics (corrected item-total, alpha-if-deleted), explicit reverse scoring, and guidance warning against mechanical item deletion.
- **Verdict**: **KEEP** | **Priority**: P1 | **Action**: Highlighted that high alpha does not prove unidimensionality or construct validity.

---

### Feature 12: Intraclass Correlation Coefficient (ICC)
- **User Goal**: Evaluate inter-rater or test-retest reliability and agreement for quantitative measurements on a balanced target-by-rater design.
- **Recommended Entry Point**: `assistant.intraclass_correlation(target="subject", rater="judge", value="score", model="two_way_random", definition="absolute_agreement", unit="single")` (or alias `assistant.icc(...)`).
- **Direct / Advanced Entry Point**: `StatisticalAnalyzer.intraclass_correlation()`.
- **Required Inputs**: `target`, `rater`, `value` (outcome score), plus explicit design choices: `model` (`"one_way_random"`, `"two_way_random"`, `"two_way_mixed"`), `unit` (`"single"`, `"average"`), and `definition` (`"absolute_agreement"`, `"consistency"` for two-way models).
- **Input Shape**: Long format panel; balanced and fully crossed design ($n \ge 2$ targets, $k \ge 2$ raters).
- **Estimand / Target**: Shrout & Fleiss / McGraw & Wong ICC variant (ICC(1,1), ICC(1,k), ICC(2,1), ICC(2,k), ICC(3,1), ICC(3,k)).
- **Direction / Orientation**: Reliability/agreement ratio ($-\frac{1}{k-1} \le \text{ICC} \le 1$). Negative estimates are preserved without clamping to zero as critical diagnostics.
- **Primary Result**: Canonical ICC estimate, confidence interval, hypothesis F-test.
- **Effect Size / Uncertainty**: Analytical exact F-inversion CI (one-way and two-way mixed consistency); Satterthwaite effective df approximation CI (two-way random absolute agreement).
- **Diagnostics & Assumptions**: ANOVA mean squares ($BMS$, $JMS$, $EMS$, $WMS$), method-of-moments variance components (target, rater, residual), systematic rater effect F-test.
- **Sample Accounting**: Total observations, retained complete targets, raters, complete panel ratings, excluded targets with missing ratings.
- **Terminal Presentation**: Canonical form metric, Model/Definition/Unit metrics, Targets x Raters, Complete Panel ratings, ANOVA Mean Squares table, Variance Components table.
- **Report Presentation**: Comprehensive ICC report with all variance components and Shrout-Fleiss notation.
- **Error / Needs Input Experience**: Missing model/definition/unit triggers `needs_input` asking researcher for explicit design facts; unbalanced/missing rating panels filter incomplete targets or return `data_limited`.
- **Market Benchmark**: Pingouin's `intraclass_corr` reports all 6 variants simultaneously; PyAutoStat asks the researcher to declare the intended design first (`needs_input`) because the choice of model and definition is a scientific design question, not an exploratory buffet.
- **Verdict**: **KEEP** | **Priority**: P1 | **Action**: Documented design choices and Shrout-Fleiss notation clearly in the audit document and API reference.

---

## 5. Cross-Cutting Usability Analysis

### Presentation Consistency
Across all 13 families, the terminal renderer (`show(workflow)`) standardizes the display hierarchy into:
1. **Header**: Title, subtitle, analysis status, and workflow status.
2. **Design Metrics**: Method name, estimand label, sample size, excluded rows, and design-specific parameters (e.g. contrast, conditions, items, factors, event level).
3. **Key Metrics**: Primary estimate, confidence interval, test statistic, p-value, effect size.
4. **Structured Tables**: Group summaries, contingency tables, ANOVA mean squares, coefficients, or pairwise comparisons.
5. **Diagnostics & Assumptions**: Explicit assumptions (variance homogeneity, sphericity, collinearity, expected counts, influence).
6. **Interpretation & Limitations**: Plain-language, rule-based qualified narration and scientific limitations.

### Sample Accounting Consistency
All methods adhere to explicit sample tracking:
- **Independent comparisons & regression**: Original rows, analyzed complete cases, excluded rows.
- **Paired two-condition**: Complete pairs, incomplete units, zero differences.
- **Repeated measures (3+)**: Complete units across all conditions, excluded incomplete units.
- **Scale reliability**: Complete respondent cases, excluded respondents with missing items.
- **ICC**: Complete crossed target panel, excluded targets with missing rater values.

### Directional and Orientation Consistency
Ambiguity of sign is eliminated across all families:
- Independent groups: `'group1' - 'group2'` (derived from `group_order`).
- Paired two-condition: `'cond1' - 'cond2'` (derived from `condition_order`).
- One-sample mean: `observed sample mean - reference value`.
- Point-biserial: explicit `positive_level` (+1 coding).
- Fisher exact: contingency row/column marginals.
- Logistic regression: modeled `event_level` vs `non_event_level`.
- Linear regression: reference categories for treatment coding.

---

## 6. Action Plan and Priority Classifications

### P0 (Critical Correctness / Contradiction) — Fixed
- Residual wording overclaims regarding `audit=False` ("completely valid") have been replaced across `README.md`, `API_REFERENCE.md`, and `docs/CAPABILITIES.md` with precise statements: the numerical calculation remains unchanged and available, while the workflow is partial because audit verification was omitted.
- No remaining P0 contradictions exist between documentation, docstrings, and verified implementation contracts.

### P1 (High-Value Usability Polish) — Implemented
- Created `docs/USABILITY_AUDIT.md` comprehensive feature audit.
- Expanded the minimal call shapes reference table in `API_REFERENCE.md` into a complete analysis cookbook covering all supported families.
- Explicitly documented entry points, estimands, input shapes, contrast directions, and sample accounting for all 13 families.

### P2 (Deferred to Future Work)
- Any new statistical modeling families (mixed models, mixed ANOVA, GEE, survival analysis).
- New power-analysis families or sample size heuristics.
- Broad exception-class taxonomy restructuring.
- Notebook UI / interactive widget development.
