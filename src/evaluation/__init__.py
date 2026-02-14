"""
Evaluation Request Workflow

Processes cross-project evaluation requests: loads YAML configs,
runs candidate models, applies custom scorers, evaluates acceptance
criteria, and produces comparison reports with recommendations.

Usage:
    from src.evaluation import EvalRequestConfig, EvaluationRunner

    config = EvalRequestConfig.from_yaml(Path("evaluations/requests/my-request.yaml"))
    runner = EvaluationRunner()
    result = await runner.run(config)
"""

from .config import AcceptanceCriterion, CandidateModel, EvalRequestConfig, Scenario
from .report import EvaluationReportGenerator
from .runner import (
    CriterionResult,
    EvaluationResult,
    EvaluationRunner,
    ModelEvalResult,
)
from .scorers import (
    SCORER_REGISTRY,
    CitationAccuracyScorer,
    JsonValidityScorer,
    LatencyScorer,
    PatternAccuracyScorer,
    ScorerResult,
    VerbatimQuoteScorer,
    get_scorer,
)

__all__ = [
    # Config
    "EvalRequestConfig",
    "AcceptanceCriterion",
    "Scenario",
    "CandidateModel",
    # Runner
    "EvaluationRunner",
    "EvaluationResult",
    "ModelEvalResult",
    "CriterionResult",
    # Report
    "EvaluationReportGenerator",
    # Scorers
    "ScorerResult",
    "JsonValidityScorer",
    "LatencyScorer",
    "PatternAccuracyScorer",
    "CitationAccuracyScorer",
    "VerbatimQuoteScorer",
    "SCORER_REGISTRY",
    "get_scorer",
]
