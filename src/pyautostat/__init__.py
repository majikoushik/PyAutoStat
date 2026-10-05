"""Design-aware statistical workflows, interpretation, audit, and reporting
for pandas DataFrames.
"""

__version__ = "0.5.0"
__author__ = "Koushik Chandra Maji"

from .analysis_plan import (
    AnalysisPlanStatus,
    PlanAdherenceResult,
    StatisticalAnalysisPlan,
    compare_plan_to_result,
)
from .analyzer import StatisticalAnalyzer

# ---------------------------------------------------------------------------
# Level 3: Audit, reproducibility, completeness, and session snapshots
# ---------------------------------------------------------------------------
from .audit import AuditFinding, AuditResult, StatisticalResultAuditor

# ---------------------------------------------------------------------------
# Level 3: Presentation models, exports, and research bundles
# ---------------------------------------------------------------------------
from .bundle import (
    BUNDLE_SCHEMA_VERSION,
    BundleOptions,
    BundleVerificationResult,
    save_bundle,
    to_bundle,
    verify_bundle,
)
from .completeness import (
    CompletenessItem,
    ReportingCompletenessResult,
    assess_reporting_completeness,
)
from .decision_ledger import DecisionLedger

# ---------------------------------------------------------------------------
# Profiling and column-role detection helpers
# ---------------------------------------------------------------------------
from .detection import detect_column_types, suggest_column_roles

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------
from .exceptions import (
    ColumnNotFoundError,
    InsufficientDataError,
    InsufficientGroupsError,
    InvalidDataError,
    InvalidTestError,
    PyAutoStatError,
    ReportError,
)

# ---------------------------------------------------------------------------
# Legacy standalone engines
# ---------------------------------------------------------------------------
from .insights import InsightEngine

# ---------------------------------------------------------------------------
# Interpretation and narration engines and helpers
# ---------------------------------------------------------------------------
from .interpretation import (
    InterpretationEngine,
    InterpretationFinding,
    InterpretationResult,
    InterpretationStatus,
)

# ---------------------------------------------------------------------------
# Level 3: Structured results, specifications, and method contracts
# ---------------------------------------------------------------------------
from .method_contracts import METHOD_CONTRACTS, MethodContract
from .narrate import (
    assumption_grade,
    coefficient_of_variation_narrative,
    column_story,
    crosstab_narrative,
    dataset_opening,
    effect_narrative,
    executive_summary,
    frequency_narrative,
    hypothesis_verdict,
    insight_narrative,
    interval_verdict,
    percentile_narrative,
    recommendation_rationale,
    sensitivity_verdict,
)
from .practical_significance import (
    MeaningfulEffectThreshold,
    PracticalSignificanceResult,
)
from .presentation import (
    PresentationView,
    StaticFigureArtifact,
    UnsupportedPresentationError,
    save_docx,
    save_html,
    save_interactive_html,
    save_pdf,
    save_static_figures,
    show,
    to_docx,
    to_html,
    to_interactive_html,
    to_pdf,
    to_static_figures,
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

__all__ = [
    # Level 1: Recommended high-level workflow and primary presentation
    "ResearchAssistant",
    "show",
    "save_html",
    "save_pdf",
    "save_docx",
    # Level 2: Direct / advanced statistical analysis, planning, and sensitivity
    "StatisticalAnalyzer",
    "StudyPlanner",
    "StudyPlanningResult",
    "StatisticalAnalysisPlan",
    "AnalysisPlanStatus",
    "PlanAdherenceResult",
    "compare_plan_to_result",
    "MeaningfulEffectThreshold",
    "PracticalSignificanceResult",
    "SensitivitySpecification",
    "SensitivityScenario",
    "SensitivityScenarioResult",
    "SensitivityResult",
    "SensitivityStatus",
    "ScenarioStatus",
    "Comparability",
    # Level 3: Structured results, specifications, and method contracts
    "ResearchWorkflowResult",
    "WorkflowStatus",
    "AnalysisResult",
    "AnalysisStatus",
    "AnalysisSpecification",
    "AnalysisOptions",
    "ResearchQuestion",
    "StudyDesign",
    "Objective",
    "Recommendation",
    "RecommendationStatus",
    "QuestionDraft",
    "QuestionStatus",
    "ClarificationQuestion",
    "MethodCapability",
    "MethodContract",
    "METHOD_CONTRACTS",
    # Level 3: Audit, reproducibility, completeness, and session snapshots
    "DecisionLedger",
    "AuditFinding",
    "AuditResult",
    "StatisticalResultAuditor",
    "ReproducibilityRecord",
    "ReproductionOutcome",
    "reproduce",
    "CompletenessItem",
    "ReportingCompletenessResult",
    "assess_reporting_completeness",
    "ResearchSessionSnapshot",
    "build_session_snapshot",
    "capability_payload",
    # Level 3: Presentation models, exports, and research bundles
    "ResearchReport",
    "PresentationView",
    "UnsupportedPresentationError",
    "to_html",
    "to_pdf",
    "to_docx",
    "to_interactive_html",
    "save_interactive_html",
    "to_static_figures",
    "save_static_figures",
    "StaticFigureArtifact",
    "BUNDLE_SCHEMA_VERSION",
    "BundleOptions",
    "BundleVerificationResult",
    "save_bundle",
    "to_bundle",
    "verify_bundle",
    # Interpretation and narration engines and helpers
    "InterpretationEngine",
    "InterpretationFinding",
    "InterpretationResult",
    "InterpretationStatus",
    "assumption_grade",
    "coefficient_of_variation_narrative",
    "column_story",
    "crosstab_narrative",
    "dataset_opening",
    "effect_narrative",
    "executive_summary",
    "frequency_narrative",
    "hypothesis_verdict",
    "insight_narrative",
    "interval_verdict",
    "percentile_narrative",
    "recommendation_rationale",
    "sensitivity_verdict",
    # Profiling and column-role detection helpers
    "detect_column_types",
    "suggest_column_roles",
    # Legacy standalone engines
    "ReportGenerator",
    "InsightEngine",
    # Exceptions
    "PyAutoStatError",
    "ColumnNotFoundError",
    "InsufficientDataError",
    "InsufficientGroupsError",
    "InvalidDataError",
    "InvalidTestError",
    "ReportError",
]
