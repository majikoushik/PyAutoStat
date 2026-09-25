# Final controlled capability matrix

This matrix describes the local Phase 12 implementation. A listed method still depends on a
researcher supplying the correct scientific design and on its documented numerical conditions.

## Guided execution

| Objective and target | Design | Method | Estimate and uncertainty | Status |
| --- | --- | --- | --- | --- |
| Dataset description | not applicable | Dataset profile | Descriptive records; no inferential p-value | Supported |
| Two-group population means | Independent | Welch t-test | First-minus-second mean, analytical CI, Cohen's d | Supported |
| Two-condition population mean paired difference | Paired with explicit unit ID | Paired t-test | First-minus-second paired mean, analytical CI, Cohen's dz | Supported |
| Two-group rank distributions | Independent | Mann-Whitney U | U, rank-biserial effect, optional bootstrap CI | Supported |
| Three-or-more rank distributions | Independent | Kruskal-Wallis | Omnibus H, epsilon-squared, optional bootstrap CI | Supported |
| Linear numerical association | Independent rows | Pearson correlation | r and p-value; CI unavailable | Partially supported |
| Categorical independence | Independent rows | Pearson chi-square | Chi-square, Cramer's V, optional bootstrap CI | Supported when expected cells pass policy |
| Pooled independent means | Independent | Student t-test | Existing explicit/sensitivity adapter | Runnable, not automatically selected |
| Standard multi-group means | Independent | One-way ANOVA | Existing explicit/sensitivity adapter | Runnable, not automatically selected |
| Repeated measures with more than two conditions | Repeated | None | None | Unsupported |
| Clustered, regression, mixed, survival, causal models | Corresponding designs | None | None | Unsupported |

Spearman and Kendall remain descriptive coefficient matrices without guided inference. Sparse
exact-table methods, Welch ANOVA, post-hoc families, and broad multiplicity procedures remain
unsupported.

## Prospective planning

| Family | Power planning | Precision planning | Main assumptions |
| --- | --- | --- | --- |
| Two independent means | Bounded Welch noncentral-t approximation | Bounded Welch t half-width | Explicit target difference (power), two SDs, alpha/confidence, n2/n1 allocation |
| Paired means | Bounded paired noncentral-t calculation | Bounded paired t half-width | Explicit target paired difference (power), paired-difference SD, complete pairs |

Planning is two-sided and prospective. There is no automatic observed post-hoc power and no
automatic use of a practical-significance threshold as an anticipated effect.

## Presentation and adapter capabilities

| Capability | Available contract | Limit |
| --- | --- | --- |
| Analysis plan | Schema 1, round-trip dictionary/JSON, local revision provenance | Local record is not verified preregistration |
| Reporting completeness | Machine-readable item statuses and deterministic overall status | Completeness is not scientific quality |
| Styles | General, APA-oriented, IEEE-oriented HTML/Markdown/LaTeX | No journal-compliance claim |
| LaTeX | In-memory and explicit save with escaping | No compiler, PDF, or DOCX |
| Session snapshot | Schema 1 JSON payload with questions/actions/capabilities | No GUI and no statistical logic in adapters |
| Audit | Canonical JSON/CSV/general or styled HTML/Markdown/LaTeX consistency | Checks consistency, not truth |
| Reproduction | Explicit supplied-data replay of executable base analysis | Does not authenticate data or automatically replay follow-ups |
