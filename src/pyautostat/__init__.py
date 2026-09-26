"""Statistical analysis, insights, and reports for pandas DataFrames."""

from .analysis_plan import (
    AnalysisPlanStatus,
    PlanAdherenceResult,
    StatisticalAnalysisPlan,
    compare_plan_to_result,
)
from .analyzer import StatisticalAnalyzer
from .audit import AuditFinding, AuditResult, StatisticalResultAuditor
from .completeness import (
    CompletenessItem,
    ReportingCompletenessResult,
    assess_reporting_completeness,
)
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
from .narrate import (
    assumption_grade,
    column_story,
    dataset_opening,
    effect_narrative,
    hypothesis_verdict,
    insight_narrative,
    interval_verdict,
    sensitivity_verdict,
)
from .practical_significance import (
    MeaningfulEffectThreshold,
    PracticalSignificanceResult,
)
from .question_builder import ClarificationQuestion, QuestionDraft, QuestionStatus
from .recommendation import MethodCapability
from .report import ReportGenerator
from .reproducibility import ReproducibilityRecord, ReproductionOutcome, reproduce
from .research_assistant import ResearchAssistant
from .research_report import ResearchReport
from .results import (
    AnalysisResult,
    AnalysisStatus,
    Recommendation,
    RecommendationStatus,
)
from .sensitivity import (
    Comparability,
    ScenarioStatus,
    SensitivityResult,
    SensitivityScenario,
    SensitivityScenarioResult,
    SensitivitySpecification,
    SensitivityStatus,
)
from .session import ResearchSessionSnapshot, build_session_snapshot, capability_payload
from .specifications import (
    AnalysisOptions,
    AnalysisSpecification,
    Objective,
    ResearchQuestion,
    StudyDesign,
)
from .study_planning import StudyPlanner, StudyPlanningResult
from .workflow import ResearchWorkflowResult, WorkflowStatus

__version__ = "0.2.0"
__author__ = "Koushik Chandra Maji"

__all__ = [
    "StatisticalAnalyzer",
    "StatisticalAnalysisPlan",
    "AnalysisPlanStatus",
    "PlanAdherenceResult",
    "compare_plan_to_result",
    "StudyPlanner",
    "StudyPlanningResult",
    "ResearchAssistant",
    "ResearchWorkflowResult",
    "WorkflowStatus",
    "ResearchReport",
    "CompletenessItem",
    "ReportingCompletenessResult",
    "assess_reporting_completeness",
    "ResearchSessionSnapshot",
    "build_session_snapshot",
    "capability_payload",
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
    "effect_narrative",
    "hypothesis_verdict",
    "assumption_grade",
    "column_story",
    "dataset_opening",
    "insight_narrative",
    "interval_verdict",
    "sensitivity_verdict",
    "SensitivitySpecification",
    "SensitivityScenario",
    "SensitivityScenarioResult",
    "SensitivityResult",
    "SensitivityStatus",
    "ScenarioStatus",
    "Comparability",
    "MeaningfulEffectThreshold",
    "PracticalSignificanceResult",
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
