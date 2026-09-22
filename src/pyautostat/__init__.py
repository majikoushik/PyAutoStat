"""Statistical analysis, insights, and reports for pandas DataFrames."""

from .analyzer import StatisticalAnalyzer
from .audit import AuditFinding, AuditResult, StatisticalResultAuditor
from .decision_ledger import DecisionLedger
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
from .interpretation import (
    InterpretationEngine,
    InterpretationFinding,
    InterpretationResult,
    InterpretationStatus,
)
from .question_builder import ClarificationQuestion, QuestionDraft, QuestionStatus
from .recommendation import MethodCapability
from .report import ReportGenerator
from .reproducibility import ReproducibilityRecord, ReproductionOutcome, reproduce
from .research_assistant import ResearchAssistant
from .research_report import ResearchReport
from .results import AnalysisResult, AnalysisStatus, Recommendation, RecommendationStatus
from .specifications import (
    AnalysisOptions,
    AnalysisSpecification,
    Objective,
    ResearchQuestion,
    StudyDesign,
)

__version__ = "0.1.0"
__author__ = "Koushik Chandra Maji"

__all__ = [
    "StatisticalAnalyzer",
    "ResearchAssistant",
    "ResearchReport",
    "DecisionLedger",
    "AuditFinding",
    "AuditResult",
    "StatisticalResultAuditor",
    "ReproducibilityRecord",
    "ReproductionOutcome",
    "reproduce",
    "QuestionDraft",
    "QuestionStatus",
    "ClarificationQuestion",
    "Recommendation",
    "RecommendationStatus",
    "MethodCapability",
    "AnalysisResult",
    "AnalysisStatus",
    "InterpretationEngine",
    "InterpretationFinding",
    "InterpretationResult",
    "InterpretationStatus",
    "ResearchQuestion",
    "StudyDesign",
    "Objective",
    "AnalysisOptions",
    "AnalysisSpecification",
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
