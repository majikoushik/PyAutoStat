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

## 9. Representative Analyses

The HTML presentation system establishes and validates presentation for four representative analysis families:

1. **Welch Independent-Samples t-test**:
   - Signed mean difference (preserves direction; never reversed or absolute-valued).
   - Dynamic estimate CI and Cohen's d effect size CI.
   - Group summary table.
   - Variance assumption explicitly marked `[NOT ASSUMED]`.
2. **Pearson Linear Correlation**:
   - Correlation coefficient `r`, dynamic CI, and two-sided p-value.
   - Sample accounting for complete paired observations.
   - Methodological limitation: association does not establish causation.
3. **Ordinary Least Squares (OLS) Linear Regression**:
   - Model fit metrics ($R^2$, adjusted $R^2$, $F$-statistic, p-value, residual standard error).
   - Coefficient table in stable source order (Term, Estimate, Std Error, dynamic CI, $t$, p-value).
   - Stored Breusch-Pagan heteroskedasticity and VIF collinearity diagnostics.
4. **Kruskal-Wallis Rank Sum Test**:
   - Omnibus $H$ statistic with degrees of freedom, p-value, and epsilon-squared effect size.
   - Real group summary schema (Group, $N$, Median; no fabricated IQR).
   - Dunn-Holm pairwise comparisons table (Contrast, Mean-Rank Diff, Dunn $z$, Rank-biserial $r$, Adjusted p-value, Decision).
   - Equal-variance diagnostic explicitly marked `[NOT APPLICABLE]`.
   - Preserves non-string/zero group labels (e.g. `0`).

---

## 10. ResearchReport Integration

The canonical `ResearchReport` object provides upgraded `to_html(...)` and `save_html(...)` methods using the modern visual design system:

```python
report = assistant.report(analysis_result)

# Export modern styled HTML report
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
