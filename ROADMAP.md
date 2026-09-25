# Product roadmap

PyAutoStat's long-term aim is a single Python library for reliable, understandable statistical analysis of tabular data. This roadmap preserves the ten themes from the original USP document while separating shipped behavior from future work. It is a planning guide, not a promise of release dates or statistical suitability for every study.

| Goal | Current status | Next useful work |
| --- | --- | --- |
| 1. Statistical rigor | Normality and variance screens, independent group tests, a bounded explicit-ID paired mean test, effect sizes, intervals and study-design guidance are available locally. | Evaluate additional dependent designs only with complete contracts; assess post-hoc and multiplicity needs separately. |
| 2. Actionable insights | `InsightEngine` rates findings and returns recommendations for several data quality and analysis issues. | Make recommendations more specific and document their evidence and limits. |
| 3. Automatic test selection | `hypothesis_tests(test_type="auto", estimand=...)` selects supported independent-group methods. The guided `run()` path checks the stated design and target, executes one ready method, and preserves the same result through interpretation, reporting, audit, and reproducibility metadata. Explicit Phase 11 sensitivity scenarios remain separate and are never selected from p-values. | Add justified multi-group mean support and broader method coverage. |
| 4. Multiple outlier methods | IQR, Z-score and MAD summaries are available. | Expose configuration and clarify behavior for highly skewed data. |
| 5. Correlation analysis | Pearson, Spearman and Kendall matrices are available; Pearson pairs have p-values. | Add clearly defined inference for other correlation methods where appropriate. |
| 6. Shareable analysis output | Dictionary, JSON, CSV, styled static HTML/Markdown, safely escaped LaTeX text, and optional interactive HTML exports are available. | Validate adapter and presentation requirements with expert users; assess PDF/DOCX separately. |
| 7. Error prevention | Input validation, specific exceptions and `analysis_warnings` cover many bad-data cases. | Add safeguards for study design and repeated testing; no library can guarantee error-free analysis. |
| 8. Performance and automation | The package can be called from Python scripts and processes an in-memory DataFrame. | Benchmark realistic datasets before making speed claims; assess batch and large-data support. |
| 9. Severity-rated findings | Insight summaries count high, medium and low severity findings. | Refine severity rules and make thresholds configurable with tests. |
| 10. Documentation and reproducibility | The README, API reference, runnable examples and tests describe the controlled workflow, explicit sensitivity plans, meaningful thresholds, audit, and replay metadata. | Add methodology references and more end-to-end research workflows. |

The [README](README.md) and [API reference](API_REFERENCE.md) describe shipped features. New work should include edge-case tests and update the [examples](examples/README.md) when it changes the public API.

The Phase 1 `ResearchAssistant(df).profile()` facade through the Phase 12 analysis-plan, study-planning, explicit paired-means, reporting-completeness, oriented-presentation, safe-LaTeX, and session-snapshot work are checked into `main`, and the corresponding GitHub CI run passed. Reporting and auditing read recorded results; only an explicitly requested replay reruns the base test. A release remains a separate owner decision. The [development roadmap](DEVELOPMENT_ROADMAP.md) defines the phase sequence; this older table remains a high-level summary.
