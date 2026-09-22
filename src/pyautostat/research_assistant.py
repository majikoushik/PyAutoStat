"""Small public entry point for existing dataset profiling."""

import pandas as pd

from .analyzer import StatisticalAnalyzer
from .profiling import complete_case_count


class ResearchAssistant:
    """Coordinate research workflows; Phase 1 supports profiling only.

    Construction uses StatisticalAnalyzer's validation and private DataFrame copy.
    It does not run the profiling calculations until :meth:`profile` is called.
    """

    def __init__(self, df: pd.DataFrame) -> None:
        self._analyzer = StatisticalAnalyzer(df)

    def profile(
        self, *, data_dictionary=None, histogram_bins: int = 20, include_row_positions: bool = False
    ) -> dict:
        """Profile a DataFrame; optional declarations never mutate source values."""
        return self._analyzer.analyze_all(
            data_dictionary=data_dictionary,
            histogram_bins=histogram_bins,
            include_row_positions=include_row_positions,
        )

    def complete_case_count(self, columns: list[str]) -> dict:
        """Count rows available for a specified set of columns."""
        return complete_case_count(self._analyzer.df, columns)
