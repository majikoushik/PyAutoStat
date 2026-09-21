"""
PyAutoStat: Comprehensive Automated Statistical Analysis Library

A Python library for automatic statistical analysis of research data with:
- Comprehensive statistical tests
- Mixed data type handling
- Advanced analytics (outliers, time series, etc.)
- Actionable insights and recommendations
- Flexible output formats (JSON, HTML, Dict, CSV)
"""

from .analyzer import StatisticalAnalyzer
from .detection import detect_column_types, suggest_column_roles
from .exceptions import (
    ColumnNotFoundError,
    InsufficientDataError,
    InsufficientGroupsError,
    InvalidDataError,
    InvalidTestError,
    PyAutoStatError,
    ReportError,
)
from .insights import InsightEngine
from .report import ReportGenerator

__version__ = "0.1.0"
__author__ = "Koushik Chandra Maji"

__all__ = [
    "StatisticalAnalyzer",
    "ReportGenerator",
    "InsightEngine",
    "PyAutoStatError",
    "ColumnNotFoundError",
    "InsufficientGroupsError",
    "InsufficientDataError",
    "InvalidDataError",
    "InvalidTestError",
    "ReportError",
    "detect_column_types",
    "suggest_column_roles",
]
