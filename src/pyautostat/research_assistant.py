"""Small public entry point for existing dataset profiling."""

import pandas as pd

from .analyzer import StatisticalAnalyzer


class ResearchAssistant:
    """Coordinate research workflows; Phase 1 supports profiling only.

    Construction uses StatisticalAnalyzer's validation and private DataFrame copy.
    It does not run the profiling calculations until :meth:`profile` is called.
    """

    def __init__(self, df: pd.DataFrame) -> None:
        self._analyzer = StatisticalAnalyzer(df)

    def profile(self) -> dict:
        """Return the existing ``StatisticalAnalyzer.analyze_all()`` dictionary."""
        return self._analyzer.analyze_all()
