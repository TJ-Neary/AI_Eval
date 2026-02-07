"""
Scoring Module

Provides evaluation metrics and scoring utilities for LLM benchmarks.

Components:
- pass_k: Code generation evaluation with pass@k metric
- llm_judge: LLM-as-Judge evaluation with bias mitigation
- rag_metrics: RAG pipeline evaluation using DeepEval

Usage:
    from src.scoring import (
        # Pass@k for code
        calculate_pass_at_k,
        evaluate_code_generation,
        # LLM-as-Judge
        LLMJudge,
        JudgingCriteria,
        # RAG metrics
        RAGEvaluator,
    )
"""

from .llm_judge import (
    JudgingConfig,
    JudgingCriteria,
    JudgingResult,
    LLMJudge,
    PairwiseResult,
)
from .pass_k import (
    CodeEvaluationConfig,
    CodeExecutionResult,
    PassKResult,
    calculate_pass_at_k,
    calculate_pass_at_k_batch,
    evaluate_code_generation,
    evaluate_code_samples,
    execute_python_code,
)
from .rag_metrics import (
    RAGEvaluationResult,
    RAGEvaluator,
    RAGEvaluatorConfig,
    RAGMetricResult,
    RAGTestCase,
)

__all__ = [
    # Pass@k
    "calculate_pass_at_k",
    "calculate_pass_at_k_batch",
    "evaluate_code_samples",
    "evaluate_code_generation",
    "execute_python_code",
    "CodeExecutionResult",
    "CodeEvaluationConfig",
    "PassKResult",
    # LLM-as-Judge
    "LLMJudge",
    "JudgingCriteria",
    "JudgingConfig",
    "JudgingResult",
    "PairwiseResult",
    # RAG metrics
    "RAGEvaluator",
    "RAGEvaluatorConfig",
    "RAGTestCase",
    "RAGMetricResult",
    "RAGEvaluationResult",
]
