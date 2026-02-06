"""
RAG Evaluation Metrics

Integration with DeepEval for comprehensive RAG pipeline evaluation.
Supports metrics for retrieval quality, answer generation, and hallucination detection.

Metrics available:
- Answer Relevancy: How relevant is the answer to the question?
- Faithfulness: Is the answer grounded in the retrieved context?
- Contextual Precision: Are relevant contexts ranked higher?
- Contextual Recall: Are all relevant contexts retrieved?
- Hallucination: Does the answer contain fabricated information?

Based on RAGAS methodology and DeepEval framework.

Usage:
    from src.scoring.rag_metrics import RAGEvaluator, RAGTestCase

    evaluator = RAGEvaluator()
    result = await evaluator.evaluate(
        question="What is the capital of France?",
        answer="Paris is the capital of France.",
        contexts=["France is a country in Europe. Paris is its capital."],
    )
    print(result.faithfulness, result.answer_relevancy)
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class RAGTestCase:
    """Test case for RAG evaluation."""

    question: str
    answer: str
    contexts: List[str]
    expected_answer: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RAGMetricResult:
    """Result for a single RAG metric."""

    name: str
    score: float  # 0-1 normalized
    reason: str = ""
    threshold: float = 0.5
    passed: bool = True

    def __post_init__(self) -> None:
        self.passed = self.score >= self.threshold


@dataclass
class RAGEvaluationResult:
    """Complete RAG evaluation results."""

    test_case: RAGTestCase
    answer_relevancy: Optional[RAGMetricResult] = None
    faithfulness: Optional[RAGMetricResult] = None
    contextual_precision: Optional[RAGMetricResult] = None
    contextual_recall: Optional[RAGMetricResult] = None
    hallucination: Optional[RAGMetricResult] = None

    @property
    def overall_score(self) -> float:
        """Average of all computed metrics."""
        scores = []
        for metric in [
            self.answer_relevancy,
            self.faithfulness,
            self.contextual_precision,
            self.contextual_recall,
        ]:
            if metric is not None:
                scores.append(metric.score)

        # Hallucination is inverted (lower is better)
        if self.hallucination is not None:
            scores.append(1.0 - self.hallucination.score)

        return sum(scores) / len(scores) if scores else 0.0

    @property
    def passed(self) -> bool:
        """Whether all metrics pass their thresholds."""
        for metric in [
            self.answer_relevancy,
            self.faithfulness,
            self.contextual_precision,
            self.contextual_recall,
        ]:
            if metric is not None and not metric.passed:
                return False

        # Hallucination should be LOW to pass
        if self.hallucination is not None:
            if self.hallucination.score > (1.0 - self.hallucination.threshold):
                return False

        return True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "question": self.test_case.question,
            "answer": self.test_case.answer,
            "overall_score": self.overall_score,
            "passed": self.passed,
            "metrics": {
                "answer_relevancy": self._metric_to_dict(self.answer_relevancy),
                "faithfulness": self._metric_to_dict(self.faithfulness),
                "contextual_precision": self._metric_to_dict(self.contextual_precision),
                "contextual_recall": self._metric_to_dict(self.contextual_recall),
                "hallucination": self._metric_to_dict(self.hallucination),
            },
        }

    @staticmethod
    def _metric_to_dict(metric: Optional[RAGMetricResult]) -> Optional[Dict[str, Any]]:
        if metric is None:
            return None
        return {
            "score": metric.score,
            "reason": metric.reason,
            "passed": metric.passed,
        }


@dataclass
class RAGEvaluatorConfig:
    """Configuration for RAG evaluation."""

    # Which metrics to compute
    compute_answer_relevancy: bool = True
    compute_faithfulness: bool = True
    compute_contextual_precision: bool = True
    compute_contextual_recall: bool = True
    compute_hallucination: bool = True

    # Thresholds
    answer_relevancy_threshold: float = 0.5
    faithfulness_threshold: float = 0.5
    contextual_precision_threshold: float = 0.5
    contextual_recall_threshold: float = 0.5
    hallucination_threshold: float = 0.5  # Max acceptable hallucination

    # LLM settings (if using LLM-based metrics)
    model: str = "gpt-4o-mini"  # DeepEval default
    use_local_model: bool = False
    local_model: str = "ollama/qwen2.5:32b"


class RAGEvaluator:
    """
    RAG pipeline evaluator using DeepEval metrics.

    Provides standardized evaluation for:
    - Retrieval quality (precision, recall)
    - Answer quality (relevancy, faithfulness)
    - Safety (hallucination detection)
    """

    def __init__(self, config: Optional[RAGEvaluatorConfig] = None):
        """
        Args:
            config: Evaluation configuration.
        """
        self.config = config or RAGEvaluatorConfig()
        self._deepeval_available = self._check_deepeval()

    def _check_deepeval(self) -> bool:
        """Check if DeepEval is available."""
        try:
            import deepeval
            return True
        except ImportError:
            logger.warning("DeepEval not installed. Using fallback metrics.")
            return False

    async def evaluate(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        expected_answer: Optional[str] = None,
    ) -> RAGEvaluationResult:
        """
        Evaluate a RAG response.

        Args:
            question: The user's question.
            answer: The generated answer.
            contexts: Retrieved context documents.
            expected_answer: Optional ground truth answer.

        Returns:
            RAGEvaluationResult with all computed metrics.
        """
        test_case = RAGTestCase(
            question=question,
            answer=answer,
            contexts=contexts,
            expected_answer=expected_answer,
        )

        if self._deepeval_available:
            return await self._evaluate_with_deepeval(test_case)
        else:
            return await self._evaluate_fallback(test_case)

    async def _evaluate_with_deepeval(
        self, test_case: RAGTestCase
    ) -> RAGEvaluationResult:
        """Evaluate using DeepEval metrics."""
        from deepeval import evaluate
        from deepeval.test_case import LLMTestCase

        # Create DeepEval test case
        deepeval_test_case = LLMTestCase(
            input=test_case.question,
            actual_output=test_case.answer,
            retrieval_context=test_case.contexts,
            expected_output=test_case.expected_answer,
        )

        result = RAGEvaluationResult(test_case=test_case)
        cfg = self.config

        # Answer Relevancy
        if cfg.compute_answer_relevancy:
            try:
                from deepeval.metrics import AnswerRelevancyMetric

                metric = AnswerRelevancyMetric(
                    threshold=cfg.answer_relevancy_threshold,
                    model=cfg.model,
                )
                metric.measure(deepeval_test_case)
                result.answer_relevancy = RAGMetricResult(
                    name="answer_relevancy",
                    score=metric.score or 0.0,
                    reason=metric.reason or "",
                    threshold=cfg.answer_relevancy_threshold,
                )
            except Exception as e:
                logger.warning(f"Answer relevancy metric failed: {e}")

        # Faithfulness
        if cfg.compute_faithfulness:
            try:
                from deepeval.metrics import FaithfulnessMetric

                metric = FaithfulnessMetric(
                    threshold=cfg.faithfulness_threshold,
                    model=cfg.model,
                )
                metric.measure(deepeval_test_case)
                result.faithfulness = RAGMetricResult(
                    name="faithfulness",
                    score=metric.score or 0.0,
                    reason=metric.reason or "",
                    threshold=cfg.faithfulness_threshold,
                )
            except Exception as e:
                logger.warning(f"Faithfulness metric failed: {e}")

        # Contextual Precision
        if cfg.compute_contextual_precision:
            try:
                from deepeval.metrics import ContextualPrecisionMetric

                metric = ContextualPrecisionMetric(
                    threshold=cfg.contextual_precision_threshold,
                    model=cfg.model,
                )
                metric.measure(deepeval_test_case)
                result.contextual_precision = RAGMetricResult(
                    name="contextual_precision",
                    score=metric.score or 0.0,
                    reason=metric.reason or "",
                    threshold=cfg.contextual_precision_threshold,
                )
            except Exception as e:
                logger.warning(f"Contextual precision metric failed: {e}")

        # Contextual Recall
        if cfg.compute_contextual_recall and test_case.expected_answer:
            try:
                from deepeval.metrics import ContextualRecallMetric

                metric = ContextualRecallMetric(
                    threshold=cfg.contextual_recall_threshold,
                    model=cfg.model,
                )
                metric.measure(deepeval_test_case)
                result.contextual_recall = RAGMetricResult(
                    name="contextual_recall",
                    score=metric.score or 0.0,
                    reason=metric.reason or "",
                    threshold=cfg.contextual_recall_threshold,
                )
            except Exception as e:
                logger.warning(f"Contextual recall metric failed: {e}")

        # Hallucination
        if cfg.compute_hallucination:
            try:
                from deepeval.metrics import HallucinationMetric

                metric = HallucinationMetric(
                    threshold=cfg.hallucination_threshold,
                    model=cfg.model,
                )
                metric.measure(deepeval_test_case)
                result.hallucination = RAGMetricResult(
                    name="hallucination",
                    score=metric.score or 0.0,
                    reason=metric.reason or "",
                    threshold=cfg.hallucination_threshold,
                )
            except Exception as e:
                logger.warning(f"Hallucination metric failed: {e}")

        return result

    async def _evaluate_fallback(self, test_case: RAGTestCase) -> RAGEvaluationResult:
        """Fallback evaluation when DeepEval is not available."""
        # Simple heuristic-based fallback metrics
        result = RAGEvaluationResult(test_case=test_case)

        # Answer relevancy: Check if answer mentions key terms from question
        question_terms = set(test_case.question.lower().split())
        answer_terms = set(test_case.answer.lower().split())
        overlap = len(question_terms & answer_terms)
        relevancy_score = min(1.0, overlap / max(len(question_terms), 1))

        result.answer_relevancy = RAGMetricResult(
            name="answer_relevancy",
            score=relevancy_score,
            reason="Fallback: term overlap heuristic",
            threshold=self.config.answer_relevancy_threshold,
        )

        # Faithfulness: Check if answer terms appear in contexts
        context_text = " ".join(test_case.contexts).lower()
        answer_words = test_case.answer.lower().split()
        grounded_words = sum(1 for w in answer_words if w in context_text)
        faithfulness_score = grounded_words / max(len(answer_words), 1)

        result.faithfulness = RAGMetricResult(
            name="faithfulness",
            score=faithfulness_score,
            reason="Fallback: context grounding heuristic",
            threshold=self.config.faithfulness_threshold,
        )

        return result

    async def evaluate_batch(
        self, test_cases: List[RAGTestCase]
    ) -> List[RAGEvaluationResult]:
        """Evaluate multiple test cases."""
        results = []
        for tc in test_cases:
            result = await self.evaluate(
                question=tc.question,
                answer=tc.answer,
                contexts=tc.contexts,
                expected_answer=tc.expected_answer,
            )
            results.append(result)
        return results
