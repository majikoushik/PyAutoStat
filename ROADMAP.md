# Product roadmap

PyAutoStat's long-term aim is a single Python library for reliable, understandable statistical analysis of tabular data. This roadmap preserves the ten themes from the original USP document while separating shipped behavior from future work. It is a planning guide, not a promise of release dates or statistical suitability for every study.

| Goal | Current status | Next useful work |
| --- | --- | --- |
| 1. Statistical rigor | Normality and variance screens, independent group tests, effect sizes and intervals are available. | Add study-design guidance, post-hoc comparisons and multiple-testing controls. |
| 2. Actionable insights | `InsightEngine` rates findings and returns recommendations for several data quality and analysis issues. | Make recommendations more specific and document their evidence and limits. |
| 3. Automatic test selection | `hypothesis_tests(test_type="auto", estimand=...)` requires a stated mean or distribution target and selects only supported independent-group methods. | Add design validation and justified multi-group mean support in later phases. |
| 4. Multiple outlier methods | IQR, Z-score and MAD summaries are available. | Expose configuration and clarify behavior for highly skewed data. |
| 5. Correlation analysis | Pearson, Spearman and Kendall matrices are available; Pearson pairs have p-values. | Add clearly defined inference for other correlation methods where appropriate. |
| 6. Shareable analysis output | Dictionary, JSON, CSV, static HTML and optional interactive HTML exports are available. | Improve report customization and assess requirements for publication formats. |
| 7. Error prevention | Input validation, specific exceptions and `analysis_warnings` cover many bad-data cases. | Add safeguards for study design and repeated testing; no library can guarantee error-free analysis. |
| 8. Performance and automation | The package can be called from Python scripts and processes an in-memory DataFrame. | Benchmark realistic datasets before making speed claims; assess batch and large-data support. |
| 9. Severity-rated findings | Insight summaries count high, medium and low severity findings. | Refine severity rules and make thresholds configurable with tests. |
| 10. Documentation and reproducibility | The README, API reference, runnable example and tests describe current behavior. | Add methodology references and more end-to-end research workflows. |

The [README](README.md) and [API reference](API_REFERENCE.md) describe shipped features. New work should include edge-case tests and update the [examples](examples/README.md) when it changes the public API.

The Phase 1 `ResearchAssistant(df).profile()` facade and serializable research contracts are now available. They do not add guided method selection or research execution. The [development roadmap](DEVELOPMENT_ROADMAP.md) defines the current phase sequence; this older table remains a high-level summary.
