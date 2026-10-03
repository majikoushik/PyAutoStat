# Modern HTML and Research Report Presentation

PyAutoStat provides a scientifically faithful, self-contained HTML presentation layer that transforms structured, authoritative statistical results and research workflows into modern, responsive, and printable HTML reports.

The HTML presentation layer is strictly a **view** over existing authoritative results. It never recalculates statistics, selects methods, alters estimands, reverses contrasts, changes alpha, or invents evidence.

---

## 1. Architecture

The HTML presentation layer consumes the same normalized presentation models as the Rich terminal presentation layer:

```text
    Statistical Engine
           ↓
    AnalysisResult / ResearchWorkflowResult / ResearchReport
           ↓
    Existing Presentation Adapter
           ↓
    Normalized Presentation View (PresentationView)
           ↓
    ┌─────────────────────────────┐
    │ Rich terminal renderer      │
    │ HTML renderer               │
    │ (future notebook renderer)  │
    └─────────────────────────────┘
```

Core principles:
1. **Single statistical source of truth**: The same analysis means the exact same thing in terminal output, HTML reports, and JSON export.
2. **Zero statistical recalculation**: Generating an HTML report never re-runs model estimation or inference algorithms.
3. **Multi-renderer normalization**: Models (`PresentationView`, `DisplayMetric`, `DisplayTable`, `DisplayDiagnostic`) isolate rendering logic from backend calculation details.

---

## 2. Public API

```python
from pyautostat import to_html, save_html

# Render in-memory HTML string
html_str = to_html(workflow)

# Specify detail level and title override
html_str = to_html(
    workflow,
    detail="full",
    title="Customer Spending Analysis",
    style="general",
)

# Save directly to file with overwrite protection
saved_path = save_html(
    workflow,
    "reports/customer_spending.html",
    detail="standard",
    overwrite=True,
)
```

Direct `AnalysisResult` and `ResearchReport` objects are also accepted:

```python
# Direct analysis result
html_str = to_html(workflow.analysis)

# Canonical ResearchReport snapshot
report = workflow.report
html_str = report.to_html(detail="standard", style="apa")
report.save_html("reports/research_report.html", overwrite=True)
```

---

## 3. Detail Modes

| Mode | Target Scope | Output Sections |
|---|---|---|
| `compact` | Quick executive view | Title, key result cards, concise interpretation finding |
| `standard` (default) | Standard scientific review | Title, analysis design, key result cards, primary summary tables (bounded rows), interpretation, diagnostics, limitations, warnings |
| `full` | Complete audit & archive | All standard sections plus untruncated tables, sample accounting, reference levels, recommendation rationale, audit verification status, reproducibility metadata, multiplicity control policy, and full analysis record |

---

## 4. Reporting Styles

- **`general`** (default): Clean, modern scientific report layout with clear visual hierarchy.
- **`apa`**: Preserves APA-oriented narrative summaries and standard variable-ordering conventions.
- **`ieee`**: Numbered section headings (e.g., `1. ANALYSIS DESIGN`, `2. KEY RESULTS`, etc.) and technical formatting.

---

## 5. Offline and Static by Design

PyAutoStat HTML reports are 100% self-contained and static:
- **No external CDN dependencies**: No Google Fonts, FontAwesome, or Bootstrap CDNs.
- **No JavaScript required**: Clean semantic HTML5 renders instantly in any browser, email client, or viewer.
- **No chart library required**: Does not require Plotly or Matplotlib for static report generation.
- **Secure and portable**: Safe for air-gapped corporate and research environments.

---

## 6. Visual Design System and CSS Tokens

Reports adhere to a restrained, academic visual design system that prioritizes estimates and uncertainty over raw p-values:

- **Primary estimate**: Prominent display with dark blue accent (`#1d4ed8`).
- **Confidence intervals**: Clearly labeled with dynamic confidence levels (e.g. `90% CI`, `95% CI`, `99% CI`).
- **Effect sizes**: Distinct violet styling (`#7c3aed`).
- **Evidence / p-values**: Subdued amber tone (`#b45309`) that does not visually dominate effect sizes or estimates.
- **Diagnostics**: Explicit textual badges (`[DOCUMENTED]`, `[REVIEW]`, `[WARNING]`, `[NOT ASSUMED]`, `[NOT APPLICABLE]`) rather than color-only indicators.

---

## 7. Accessibility and Print Support

- **Accessibility**:
  - Valid HTML5 with `<html lang="en">` and proper heading hierarchy (`h1` → `h2` → `h3`).
  - Semantic tables with `<thead>`, `<tbody>`, and `<th scope="col">`.
  - Contrast ratios meeting WCAG AA standards.
  - No color-only meaning: all statuses feature bracketed text labels.
- **Print stylesheet (`@media print`)**:
  - Automatically activates on print preview or PDF export.
  - Page breaks avoided inside metric cards, diagnostic items, and table rows.
  - Headings kept with content (`break-after: avoid`).
  - Neutral white background with clean borders and zero decorative shadows.

---

## 8. Security and Text Escaping

All user-supplied content (variable names, group labels, table values, question descriptions, titles) is automatically escaped using Python's `html.escape` mechanism. Even malicious inputs such as `<script>alert(1)</script>` are safely escaped as `&lt;script&gt;alert(1)&lt;/script&gt;`, preventing Cross-Site Scripting (XSS).

---

## 9. Comprehensive Method and Object Coverage

The HTML presentation architecture covers all 24 PyAutoStat statistical analysis methods and all non-analysis governance/planning objects supported by `PresentationView`:

### Supported Statistical Analysis Methods

| Method ID | Method Family | Scientific Presentation Features |
|---|---|---|
| `one_sample_t` | Comparison | Reference value, signed difference, CI, Cohen's d, $t$, df, p-value |
| `student_t` | Comparison | Equal-variance assumption context, signed difference, CI, Cohen's d |
| `welch_t` | Comparison | Robust unequal-variance, signed difference, CI, Cohen's d with effect CI |
| `paired_t` | Comparison | Condition order preserved, complete pairs accounting, mean difference, CI |
| `mann_whitney_u` | Rank Comparison | Rank-sum/distribution context (not universal median test), U statistic, rank-biserial $r_{rb}$, CI |
| `wilcoxon_signed_rank` | Rank Comparison | Pair order, signed-rank $W$, rank-biserial $r_{rb}$, CI |
| `welch_anova` | Multigroup | Omnibus Welch $F$, Games-Howell pairwise follow-up table with simultaneous CIs |
| `one_way_anova` | Multigroup | Omnibus $F$, eta-squared $\eta^2$, Tukey-Kramer post-hoc follow-up table |
| `kruskal_wallis` | Multigroup | Omnibus $H$ statistic, epsilon-squared $\epsilon^2$, Dunn-Holm pairwise follow-up table |
| `pearson_correlation` | Association | Pearson $r$, dynamic CI, sample accounting for complete pairs, causation caveat |
| `spearman_correlation` | Association | Spearman $\rho$ notation, monotonic association context, dynamic CI |
| `kendall_tau_b` | Association | Kendall $\tau_b$ notation, tie-adjusted rank concordance, dynamic CI |
| `point_biserial_correlation` | Association | Point-biserial $r_{pb}$, positive/reference level declaration, dynamic CI |
| `partial_pearson_correlation` | Association | Partial $r$, control variables listed, dynamic CI, observational limitation |
| `pearson_chi_square` | Categorical | Contingency table, Cramer's $V$, stored minimum expected-count diagnostic |
| `fisher_exact` | Categorical | Odds ratio, exact p-value, explicit unavailable CI handling on zero cells |
| `mcnemar` | Categorical | Paired transition table, discordant pairs, proportion difference, CI |
| `linear_regression` | Regression | Model fit ($R^2$, adj $R^2$, $F$), OLS coefficient table, HC3 covariance label, Breusch-Pagan / VIF diagnostics |
| `logistic_regression` | Regression | Modeled event, Odds Ratio (OR) first coefficient table, dynamic CIs |
| `cronbach_alpha` | Reliability | Raw and standardized $\alpha$, bootstrap CI, item-deleted diagnostic table |
| `repeated_measures_anova` | Repeated Measures | Within-subject omnibus $F$, Mauchly sphericity, Greenhouse-Geisser $\epsilon$ correction |
| `friedman_test` | Repeated Measures | Friedman $\chi^2$, Kendall's $W$ concordance, Wilcoxon-Holm pairwise follow-ups |
| `two_way_anova` | Factorial | Main effects Factor A, Factor B, interaction term A×B, partial $\eta^2$ |
| `intraclass_correlation` | Reliability | ICC model and definition rendered prior to estimate, negative ICC preserved |

### Supported Descriptive, Planning, and Governance Objects

| Target Object | Presentation Features |
|---|---|
| Dataset Profile | Structured design grid (rows, columns, missing cells, duplicates, memory), numeric and categorical summaries |
| Frequency Table | Frequency, valid percentage, cumulative percentage |
| Cross-tabulation | Row/column categories, contingency counts, row/col percentages |
| Study Planning Result (`StudyPlanningResult`) | Prospective power requirements, sample-size target, explicit non-post-hoc disclaimer |
| Sensitivity Result (`SensitivityResult`) | Scenario comparisons in stable order, requested vs actual method, contrast preservation |
| Practical Significance Result (`PracticalSignificanceResult`) | Quantity, threshold magnitude/unit, observed estimate, interval relation, verdict |
| Statistical Analysis Plan (`StatisticalAnalysisPlan`) | Pre-analysis specification, planned estimand, planned methodology |
| Plan Adherence Result (`PlanAdherenceResult`) | Planned vs executed method comparisons, deviation reasons |
| Reporting Completeness Result (`ReportingCompletenessResult`) | Section completeness breakdown; states internal consistency, not study quality |
| Audit Result (`AuditResult`) | Integrity checks, verification status badges (`[PASSED]`, `[REVIEW]`, `[FAILED]`) |
| Reproducibility Record (`ReproducibilityRecord`) | Seed, runtime environment, execution metadata without raw dict dump |
| Reproduction Outcome (`ReproductionOutcome`) | Replay verification status, numerical match status |
| Decision Ledger (`DecisionLedger`) | Event ledger of research actions without unverified provenance claims |
| Research Session Snapshot (`ResearchSessionSnapshot`) | UI-independent session state, workflow status, capability matrix |

---

## 10. ResearchReport Unification

The modern `ResearchReport.to_html(...)` method reuses the exact same component architecture, CSS theme, table styling, and escaping rules as standalone `to_html(...)`:

- **Shared Visual Source**: When a `ResearchReport` contains a source `AnalysisResult`, it normalizes through `adapt(report._source_result, detail=...)` without re-running statistical algorithms.
- **Identical Numbers and Badges**: Standalone HTML and report HTML agree on primary estimates, confidence intervals, effect sizes, p-values, table rows, and diagnostic statuses.
- **Graceful Fallback**: If the source result is not attached, the renderer gracefully falls back to the canonical payload records without fabricating models.
- **Executive Summary Component**: Calm visual styling for natural-language summary paragraphs.
- **Responsive Table Bounding**: Large tables in `standard` detail mode show bounded rows with a clear pagination note, while `full` detail mode renders all rows.

```python
report = assistant.report(analysis_result)

# Export unified modern styled HTML report
html_text = report.to_html(style="general", detail="standard")
report.save_html("customer_report.html", overwrite=True)
```

---

## 11. Future Roadmap: Interactive HTML

Future releases will introduce optional, progressive enhancements:
- Interactive Plotly figures (using PyAutoStat's optional `[reporting]` extra).
- Client-side sortable tables and column filters.
- Collapsible diagnostic and reproducibility accordions.
- Forest plots for pairwise comparisons and regression coefficients.
