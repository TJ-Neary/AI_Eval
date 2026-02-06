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

from .pass_k import (
    calculate_pass_at_k,
    calculate_pass_at_k_batch,
    evaluate_code_samples,
    evaluate_code_generation,
    execute_python_code,
    CodeExecutionResult,
    CodeEvaluationConfig,
    PassKResult,
)

from .llm_judge import (
    LLMJudge,
    JudgingCriteria,
    JudgingConfig,
    JudgingResult,
    PairwiseResult,
)

from .rag_metrics import (
    RAGEvaluator,
    RAGEvaluatorConfig,
    RAGTestCase,
    RAGMetricResult,
    RAGEvaluationResult,
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
