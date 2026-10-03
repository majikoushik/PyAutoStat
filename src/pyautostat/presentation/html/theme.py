"""CSS design system and tokens for PyAutoStat HTML presentations.

Modern scientific report aesthetic with accessible contrast, semantic status badges,
clean typography, responsive layouts, and dedicated print styling.
"""

from __future__ import annotations

DEFAULT_CSS = """
:root {
  --pa-bg: #f8fafc;
  --pa-surface: #ffffff;
  --pa-text: #1e293b;
  --pa-text-muted: #64748b;
  --pa-border: #e2e8f0;
  --pa-border-subtle: #f1f5f9;
  --pa-primary: #2563eb;
  --pa-primary-dark: #1d4ed8;
  --pa-secondary: #0f766e;
  --pa-effect: #7c3aed;
  --pa-evidence: #b45309;
  --pa-success: #15803d;
  --pa-warning: #b45309;
  --pa-error: #b91c1c;
  --pa-card-bg: #ffffff;
  --pa-table-stripe: #f8fafc;
  --pa-font-sans: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
    Helvetica, Arial, sans-serif;
  --pa-font-mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
}

*, *::before, *::after {
  box-sizing: border-box;
}

body {
  margin: 0;
  padding: 0;
  background-color: var(--pa-bg);
  color: var(--pa-text);
  font-family: var(--pa-font-sans);
  font-size: 16px;
  line-height: 1.5;
  -webkit-font-smoothing: antialiased;
}

.report-container {
  max-width: 960px;
  margin: 2rem auto;
  padding: 2rem 2.5rem;
  background-color: var(--pa-surface);
  border: 1px solid var(--pa-border);
  border-radius: 8px;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
}

/* Header */
.report-header {
  border-bottom: 2px solid var(--pa-border);
  padding-bottom: 1.5rem;
  margin-bottom: 2rem;
}

.report-brand {
  font-size: 0.75rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  color: var(--pa-primary);
  margin-bottom: 0.25rem;
}

.report-title {
  font-size: 1.75rem;
  font-weight: 700;
  line-height: 1.25;
  margin: 0.25rem 0 0.5rem 0;
  color: var(--pa-text);
}

.report-subtitle {
  font-size: 1rem;
  color: var(--pa-text-muted);
  margin: 0;
}

.report-status {
  font-size: 0.9375rem;
  color: var(--pa-text-muted);
  margin: 0.5rem 0 1.5rem 0;
  display: flex;
  align-items: center;
  gap: 0.5rem;
  flex-wrap: wrap;
}

.sr-only {
  position: absolute;
  width: 1px;
  height: 1px;
  padding: 0;
  margin: -1px;
  overflow: hidden;
  clip: rect(0, 0, 0, 0);
  white-space: nowrap;
  border-width: 0;
}

/* Sections */
.report-section {
  margin-bottom: 2.25rem;
}

.report-section:last-child {
  margin-bottom: 0;
}

.section-title {
  font-size: 1.125rem;
  font-weight: 600;
  letter-spacing: 0.03em;
  color: var(--pa-text);
  border-bottom: 1px solid var(--pa-border);
  padding-bottom: 0.4rem;
  margin-top: 0;
  margin-bottom: 1rem;
}

/* Metric Cards Grid */
.metric-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: 1rem;
  margin: 1rem 0;
}

.metric-card {
  background-color: var(--pa-card-bg);
  border: 1px solid var(--pa-border);
  border-radius: 6px;
  padding: 1rem 1.125rem;
  display: flex;
  flex-direction: column;
}

.metric-label {
  font-size: 0.8125rem;
  font-weight: 600;
  color: var(--pa-text-muted);
  margin-bottom: 0.35rem;
}

.metric-value {
  font-size: 1.35rem;
  font-weight: 700;
  color: var(--pa-text);
  word-break: break-word;
}

.role-estimate .metric-value {
  color: var(--pa-primary-dark);
}

.role-ci .metric-value {
  color: var(--pa-text);
}

.role-effect .metric-value {
  color: var(--pa-effect);
}

.role-evidence .metric-value {
  color: var(--pa-evidence);
}

.metric-note {
  font-size: 0.75rem;
  color: var(--pa-text-muted);
  margin-top: 0.35rem;
}

.value-unavailable {
  color: var(--pa-text-muted) !important;
  font-style: italic;
}

/* Design Grid (Definition List) */
.design-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 0.75rem 1.5rem;
  background-color: var(--pa-border-subtle);
  border: 1px solid var(--pa-border);
  border-radius: 6px;
  padding: 1rem 1.25rem;
  margin: 0 0 1rem 0;
}

.design-item {
  display: flex;
  flex-direction: column;
}

.design-label {
  font-size: 0.75rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.03em;
  color: var(--pa-text-muted);
  margin-bottom: 0.15rem;
}

.design-value {
  font-size: 0.9375rem;
  font-weight: 500;
  color: var(--pa-text);
  margin: 0;
}

/* Tables */
.table-container {
  width: 100%;
  overflow-x: auto;
  margin: 1rem 0;
  border: 1px solid var(--pa-border);
  border-radius: 6px;
  background-color: var(--pa-surface);
}

table {
  border-collapse: collapse;
  width: 100%;
  text-align: left;
  font-size: 0.875rem;
}

caption, .table-caption {
  font-size: 0.9375rem;
  font-weight: 600;
  text-align: left;
  padding: 0.65rem 0.85rem;
  color: var(--pa-text);
  border-bottom: 1px solid var(--pa-border);
  background-color: var(--pa-border-subtle);
  margin: 0;
}

th {
  background-color: var(--pa-border-subtle);
  font-weight: 600;
  color: var(--pa-text);
  padding: 0.65rem 0.85rem;
  border-bottom: 1px solid var(--pa-border);
  white-space: nowrap;
}

td {
  padding: 0.65rem 0.85rem;
  border-bottom: 1px solid var(--pa-border-subtle);
  color: var(--pa-text);
  vertical-align: top;
}

tr:last-child td {
  border-bottom: none;
}

tbody tr:nth-child(even) {
  background-color: var(--pa-table-stripe);
}

th.align-right, td.align-right {
  text-align: right;
}

th.align-left, td.align-left {
  text-align: left;
}

.cell-unavailable {
  color: var(--pa-text-muted);
  font-style: italic;
}

.table-note {
  font-size: 0.8125rem;
  color: var(--pa-text-muted);
  padding: 0.5rem 0.85rem;
  margin: 0;
  font-style: italic;
  border-top: 1px solid var(--pa-border-subtle);
}

/* Diagnostics */
.diagnostic-list {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}

.diagnostic-item {
  display: flex;
  align-items: baseline;
  gap: 0.75rem;
  padding: 0.65rem 0.85rem;
  background-color: var(--pa-surface);
  border: 1px solid var(--pa-border);
  border-radius: 6px;
}

.diagnostic-status {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  font-size: 0.6875rem;
  font-weight: 700;
  padding: 0.2rem 0.5rem;
  border-radius: 4px;
  text-transform: uppercase;
  letter-spacing: 0.04em;
  white-space: nowrap;
}

.status-documented {
  background-color: #e0f2fe;
  color: #0369a1;
  border: 1px solid #bae6fd;
}

.status-review {
  background-color: #fef3c7;
  color: #b45309;
  border: 1px solid #fde68a;
}

.status-warning {
  background-color: #ffedd5;
  color: #c2410c;
  border: 1px solid #fed7aa;
}

.status-error {
  background-color: #fee2e2;
  color: #b91c1c;
  border: 1px solid #fecaca;
}

.status-success {
  background-color: #dcfce7;
  color: #15803d;
  border: 1px solid #bbf7d0;
}

.status-neutral {
  background-color: #f1f5f9;
  color: #475569;
  border: 1px solid #e2e8f0;
}

.status-missing {
  background-color: #fef2f2;
  color: #991b1b;
  border: 1px solid #fecaca;
}

.diagnostic-label {
  font-weight: 600;
  font-size: 0.875rem;
  color: var(--pa-text);
  min-width: 180px;
}

.diagnostic-detail {
  font-size: 0.875rem;
  color: var(--pa-text-muted);
}

/* Interpretation */
.interpretation-text {
  font-size: 0.9375rem;
  line-height: 1.6;
  color: var(--pa-text);
  background-color: var(--pa-surface);
  border-left: 3px solid var(--pa-secondary);
  padding: 0.85rem 1.25rem;
  border-radius: 0 6px 6px 0;
  background-color: #f0fdfa;
}

.interpretation-text p {
  margin: 0 0 0.75rem 0;
}

.interpretation-text p:last-child {
  margin-bottom: 0;
}

/* Limitations & Warnings */
.limitations-block {
  background-color: #f8fafc;
  border: 1px solid var(--pa-border);
  border-left: 4px solid var(--pa-text-muted);
  border-radius: 0 6px 6px 0;
  padding: 0.85rem 1.25rem;
}

.limitations-block ul, .warnings-block ul {
  margin: 0;
  padding-left: 1.25rem;
}

.limitations-block li {
  margin-bottom: 0.4rem;
  font-size: 0.875rem;
  color: #334155;
}

.limitations-block li:last-child, .warnings-block li:last-child {
  margin-bottom: 0;
}

.warnings-block {
  background-color: #fffbeb;
  border: 1px solid #fde68a;
  border-left: 4px solid var(--pa-warning);
  border-radius: 0 6px 6px 0;
  padding: 0.85rem 1.25rem;
}

.warnings-block li {
  margin-bottom: 0.4rem;
  font-size: 0.875rem;
  color: #92400e;
}

/* Analysis Record / Metadata */
.record-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 0.65rem 1.25rem;
  background-color: var(--pa-border-subtle);
  border: 1px solid var(--pa-border);
  border-radius: 6px;
  padding: 1rem 1.25rem;
}

.record-item {
  display: flex;
  flex-direction: column;
}

.record-label {
  font-size: 0.75rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.03em;
  color: var(--pa-text-muted);
}

.record-value {
  font-size: 0.875rem;
  font-weight: 500;
  color: var(--pa-text);
  word-break: break-word;
}

/* Executive summary & oriented summary classes */
.executive-summary {
  background-color: #f8fafc;
  border: 1px solid var(--pa-border);
  border-radius: 6px;
  padding: 1rem 1.25rem;
  margin-bottom: 1.5rem;
}

.executive-summary h2 {
  font-size: 1.125rem;
  margin-top: 0;
  margin-bottom: 0.5rem;
}

.oriented-summary {
  font-style: italic;
  color: var(--pa-text-muted);
  font-size: 0.9375rem;
  margin-bottom: 1.25rem;
}

.caution {
  border-left: 4px solid #a55b00;
  padding: 0.75rem 1.25rem;
  background: #fff7e9;
  border-radius: 0 6px 6px 0;
  margin-bottom: 1.5rem;
}

.caution h2 {
  font-size: 1.05rem;
  margin-top: 0;
  margin-bottom: 0.5rem;
  color: #7c2d12;
}

.caution ul {
  margin: 0;
  padding-left: 1.25rem;
}

.caution li {
  margin-bottom: 0.35rem;
}

/* Responsive adjustments */
@media (max-width: 768px) {
  body {
    padding: 0;
  }
  .report-container {
    margin: 0;
    border-radius: 0;
    border-left: none;
    border-right: none;
    padding: 1.25rem 1rem;
  }
  .metric-grid {
    grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
    gap: 0.75rem;
  }
  .metric-card {
    padding: 0.75rem;
  }
  .metric-value {
    font-size: 1.15rem;
  }
  .design-grid {
    grid-template-columns: 1fr;
    gap: 0.5rem;
  }
  .diagnostic-item {
    flex-direction: column;
    gap: 0.25rem;
  }
  .diagnostic-label {
    min-width: auto;
  }
}

/* Print Styles */
@media print {
  body {
    background-color: #ffffff !important;
    color: #000000 !important;
    font-size: 11pt;
    line-height: 1.4;
    margin: 0;
    padding: 0;
  }
  .report-container {
    max-width: none !important;
    margin: 0 !important;
    padding: 0 !important;
    border: none !important;
    box-shadow: none !important;
  }
  .metric-card,
  .diagnostic-item,
  .limitations-block,
  .warnings-block,
  .caution,
  .executive-summary {
    break-inside: avoid;
    box-shadow: none !important;
    border: 1px solid #cbd5e1 !important;
  }
  h1, h2, h3, h4 {
    break-after: avoid;
  }
  table {
    break-inside: auto;
  }
  tr {
    break-inside: avoid;
  }
  .no-print {
    display: none !important;
  }
}
"""


def get_theme_css() -> str:
    """Return the complete CSS string for PyAutoStat HTML presentations."""
    return DEFAULT_CSS
