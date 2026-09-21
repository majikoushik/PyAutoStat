"""Custom exceptions with actionable, specific error messages."""


class PyAutoStatError(Exception):
    """Base class for all PyAutoStat errors."""


class InvalidDataError(PyAutoStatError):
    """Raised when the input data isn't usable for analysis."""


class ColumnNotFoundError(PyAutoStatError):
    """Raised when a referenced column doesn't exist in the DataFrame."""


class InsufficientGroupsError(PyAutoStatError):
    """Raised when a hypothesis test needs more distinct groups than are present."""


class InsufficientDataError(InvalidDataError):
    """Raised when a requested statistic has too few usable observations."""


class InvalidTestError(PyAutoStatError):
    """Raised when a test is incompatible with the requested comparison."""


class ReportError(PyAutoStatError):
    """Raised when a report cannot be rendered or written."""
