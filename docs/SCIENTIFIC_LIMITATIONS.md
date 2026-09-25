# Scientific limitations

PyAutoStat is suitable for continued controlled expert evaluation of its documented workflows.
It is not a substitute for scientific design review or subject-matter judgment.

- Study design facts, pairing, independence, clustering, randomization, and sampling intent depend
  on researcher declarations. Numerical values cannot prove them.
- The statistical catalogue is deliberately limited. Repeated measures with more than two
  conditions, clustered models, regression families, mixed models, survival analysis, causal
  inference, sparse exact-table alternatives, and broad multiplicity procedures are unsupported.
- Missing data use analysis-specific complete cases. There is no automatic imputation or
  missing-data mechanism model.
- Outliers are reported for review. They are never deleted automatically.
- Sensitivity analysis executes only researcher-declared scenarios, preserves different
  estimands, and never selects or ranks results by p-value. It cannot supply a universal
  robustness conclusion.
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
- Current local validation does not by itself verify every declared Python and minimum dependency
  version. Release readiness requires the configured CI matrix and minimum-dependency review.
