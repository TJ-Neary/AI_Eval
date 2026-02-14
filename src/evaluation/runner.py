"""
Evaluation Runner

Orchestrates evaluation requests: runs candidate models through
BenchmarkRunner, applies custom scorers, evaluates acceptance
criteria, and produces comparison results with recommendations.
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from ..benchmarks.datasets import Dataset, TestCase
from ..benchmarks.runner import (
    BenchmarkResult,
    BenchmarkRunner,
    RunConfig,
)
from ..profiling import HardwareProfile, detect_hardware
from ..providers.base import BaseProvider, ProviderFactory
from .config import AcceptanceCriterion, CandidateModel, EvalRequestConfig
from .scorers import Scorer, ScorerResult, get_scorer

logger = logging.getLogger(__name__)


@dataclass
class CriterionResult:
    """Result of evaluating one acceptance criterion across all scenarios."""

    criterion: AcceptanceCriterion
    measured_value: float  # Aggregated (e.g., rate, average)
    passed: bool
    per_scenario: List[ScorerResult] = field(default_factory=list)


@dataclass
class ModelEvalResult:
    """Complete evaluation result for one candidate model."""

    model: str
    provider: str
    benchmark_result: BenchmarkResult
    criterion_results: List[CriterionResult] = field(default_factory=list)
    all_criteria_passed: bool = False
    custom_scores: Dict[str, List[ScorerResult]] = field(default_factory=dict)
    skipped: bool = False
    skip_reason: str = ""

    @property
    def criteria_passed_count(self) -> int:
        return sum(1 for c in self.criterion_results if c.passed)

    @property
    def criteria_total(self) -> int:
        return len(self.criterion_results)


@dataclass
class EvaluationResult:
    """Complete evaluation across all candidate models."""

    request_config: EvalRequestConfig
    model_results: List[ModelEvalResult] = field(default_factory=list)
    recommended_model: Optional[str] = None  # None = no model qualifies
    recommendation_reason: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    hardware: Optional[HardwareProfile] = None
    duration_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to JSON-compatible dict."""
        return {
            "request_id": self.request_config.request_id,
            "requesting_project": self.request_config.requesting_project,
            "use_case": self.request_config.use_case,
            "timestamp": self.timestamp.isoformat(),
            "duration_seconds": self.duration_seconds,
            "recommended_model": self.recommended_model,
            "recommendation_reason": self.recommendation_reason,
            "model_results": [
                {
                    "model": r.model,
                    "provider": r.provider,
                    "skipped": r.skipped,
                    "skip_reason": r.skip_reason,
                    "all_criteria_passed": r.all_criteria_passed,
                    "criteria_passed": r.criteria_passed_count,
                    "criteria_total": r.criteria_total,
                    "overall_score": r.benchmark_result.overall_score if not r.skipped else 0.0,
                    "criterion_details": [
                        {
                            "name": c.criterion.name,
                            "metric": c.criterion.metric,
                            "threshold": c.criterion.threshold,
                            "operator": c.criterion.operator,
                            "measured": c.measured_value,
                            "passed": c.passed,
                        }
                        for c in r.criterion_results
                    ],
                }
                for r in self.model_results
            ],
        }


class EvaluationRunner:
    """Orchestrates evaluation request execution.

    Wraps BenchmarkRunner with custom scoring and acceptance criteria.
    Does not modify or subclass BenchmarkRunner — composes with it.
    """

    def __init__(self) -> None:
        self._runner = BenchmarkRunner()

    async def run(self, config: EvalRequestConfig) -> EvaluationResult:
        """Execute full evaluation pipeline.

        1. Convert config to Dataset + RunConfig
        2. For each candidate model:
           a. Create provider and check health
           b. Run BenchmarkRunner
           c. Apply custom scorers
           d. Evaluate acceptance criteria
        3. Compare candidates and select recommendation
        """
        start_time = time.time()

        hardware = detect_hardware()
        dataset = config.to_dataset()
        run_config = config.to_run_config()

        model_results: List[ModelEvalResult] = []

        for candidate in config.candidates:
            result = await self._evaluate_model(
                candidate=candidate,
                dataset=dataset,
                run_config=run_config,
                config=config,
            )
            model_results.append(result)

        # Select recommendation
        recommended, reason = self._select_recommendation(model_results)

        duration = time.time() - start_time

        return EvaluationResult(
            request_config=config,
            model_results=model_results,
            recommended_model=recommended,
            recommendation_reason=reason,
            timestamp=datetime.now(),
            hardware=hardware,
            duration_seconds=duration,
        )

    async def _evaluate_model(
        self,
        candidate: CandidateModel,
        dataset: Dataset,
        run_config: RunConfig,
        config: EvalRequestConfig,
    ) -> ModelEvalResult:
        """Run benchmarks + custom scoring for one candidate model."""
        # Create provider
        try:
            provider: BaseProvider = ProviderFactory.create(
                candidate.provider, model=candidate.name
            )
        except Exception as e:
            logger.warning("Failed to create provider for %s: %s", candidate.name, e)
            return self._skipped_result(candidate, f"Provider creation failed: {e}")

        # Health check
        try:
            healthy = await provider.health_check()
        except Exception:
            healthy = False

        if not healthy:
            logger.warning("Model %s not available, skipping", candidate.name)
            return self._skipped_result(candidate, f"Model not available ({candidate.provider})")

        # Run benchmark
        try:
            benchmark_result = await self._runner.run(
                provider=provider,
                dataset=dataset,
                config=run_config,
            )
        except Exception as e:
            logger.error("Benchmark failed for %s: %s", candidate.name, e)
            return self._skipped_result(candidate, f"Benchmark failed: {e}")

        # Apply custom scorers
        custom_scores = self._apply_custom_scorers(
            benchmark_result=benchmark_result,
            config=config,
            dataset=dataset,
        )

        # Evaluate acceptance criteria
        criterion_results = self._evaluate_acceptance_criteria(
            benchmark_result=benchmark_result,
            custom_scores=custom_scores,
            criteria=config.acceptance_criteria,
        )

        all_passed = all(c.passed for c in criterion_results) if criterion_results else True

        return ModelEvalResult(
            model=candidate.name,
            provider=candidate.provider,
            benchmark_result=benchmark_result,
            criterion_results=criterion_results,
            all_criteria_passed=all_passed,
            custom_scores=custom_scores,
        )

    def _apply_custom_scorers(
        self,
        benchmark_result: BenchmarkResult,
        config: EvalRequestConfig,
        dataset: Dataset,
    ) -> Dict[str, List[ScorerResult]]:
        """Apply all configured custom scorers to benchmark test results."""
        if not config.custom_scorers:
            return {}

        # Build scorer instances
        scorers: List[Scorer] = []
        for scorer_name in config.custom_scorers:
            scorer_cfg = config.scorer_config.get(scorer_name, {})
            try:
                scorers.append(get_scorer(scorer_name, scorer_cfg if scorer_cfg else None))
            except KeyError:
                logger.warning("Unknown scorer: %s, skipping", scorer_name)

        if not scorers:
            return {}

        # Build test case lookup for metadata
        test_lookup: Dict[str, TestCase] = {tc.id: tc for tc in dataset.tests}

        # Apply each scorer to each test result
        scores: Dict[str, List[ScorerResult]] = {s.name: [] for s in scorers}

        for cat_result in benchmark_result.categories.values():
            for test_result in cat_result.tests:
                # Build context from test result + test case metadata
                context: Dict[str, Any] = {
                    "generation_time_ms": test_result.generation_time_ms,
                    "tokens_per_second": test_result.tokens_per_second,
                }

                # Add test case metadata (source_text, expected_patterns, etc.)
                test_case = test_lookup.get(test_result.test_id)
                if test_case:
                    context["expected_patterns"] = test_case.expected_patterns
                    context.update(test_case.metadata)

                # Run each scorer
                for scorer in scorers:
                    try:
                        result = scorer.score(test_result.response, context)
                        scores[scorer.name].append(result)
                    except Exception as e:
                        logger.warning(
                            "Scorer %s failed on %s: %s",
                            scorer.name,
                            test_result.test_id,
                            e,
                        )

        return scores

    def _evaluate_acceptance_criteria(
        self,
        benchmark_result: BenchmarkResult,
        custom_scores: Dict[str, List[ScorerResult]],
        criteria: List[AcceptanceCriterion],
    ) -> List[CriterionResult]:
        """Check each acceptance criterion against measured values."""
        results: List[CriterionResult] = []

        for criterion in criteria:
            scorer_results = custom_scores.get(criterion.metric, [])

            if not scorer_results:
                # Try to compute from benchmark result directly
                measured = self._measure_from_benchmark(benchmark_result, criterion.metric)
                passed = criterion.evaluate(measured)
                results.append(
                    CriterionResult(
                        criterion=criterion,
                        measured_value=measured,
                        passed=passed,
                    )
                )
                continue

            # Aggregate scorer results based on unit type
            if criterion.unit in ("seconds", "per_minute", "per_second"):
                # For latency/throughput: use average of raw values
                raw_values = [
                    r.raw_value
                    for r in scorer_results
                    if r.raw_value is not None and isinstance(r.raw_value, (int, float))
                ]
                measured = sum(raw_values) / len(raw_values) if raw_values else 0.0
            else:
                # For rates: proportion of passes
                if scorer_results:
                    measured = sum(r.score for r in scorer_results) / len(scorer_results)
                else:
                    measured = 0.0

            passed = criterion.evaluate(measured)
            results.append(
                CriterionResult(
                    criterion=criterion,
                    measured_value=measured,
                    passed=passed,
                    per_scenario=scorer_results,
                )
            )

        return results

    def _measure_from_benchmark(self, benchmark_result: BenchmarkResult, metric: str) -> float:
        """Extract a measurement from BenchmarkResult for criteria without custom scorers."""
        if metric == "latency":
            # Average generation time in seconds
            all_times: list[float] = []
            for cat in benchmark_result.categories.values():
                all_times.extend(
                    t.generation_time_ms / 1000.0 for t in cat.tests if t.generation_time_ms > 0
                )
            return sum(all_times) / len(all_times) if all_times else 0.0

        if metric == "tokens_per_second":
            return benchmark_result.avg_tokens_per_second

        if metric == "pass_rate":
            total = benchmark_result.total_tests
            return benchmark_result.total_passed / total if total > 0 else 0.0

        if metric == "overall_score":
            return benchmark_result.overall_score / 100.0  # Normalize to 0-1

        return 0.0

    def _select_recommendation(
        self, model_results: List[ModelEvalResult]
    ) -> Tuple[Optional[str], str]:
        """Pick best qualifying model or determine none qualifies."""
        # Filter to non-skipped results
        evaluated = [r for r in model_results if not r.skipped]

        if not evaluated:
            return None, "All candidate models were unavailable or failed during testing."

        # Find models that pass all criteria
        qualifying = [r for r in evaluated if r.all_criteria_passed]

        if qualifying:
            # Pick the one with highest overall benchmark score
            best = max(qualifying, key=lambda r: r.benchmark_result.overall_score)
            runner_up = sorted(
                qualifying, key=lambda r: r.benchmark_result.overall_score, reverse=True
            )
            if len(runner_up) > 1:
                reason = (
                    f"{best.model} passes all acceptance criteria with the highest "
                    f"overall score ({best.benchmark_result.overall_score:.1f}/100). "
                    f"Runner-up: {runner_up[1].model} "
                    f"({runner_up[1].benchmark_result.overall_score:.1f}/100)."
                )
            else:
                reason = (
                    f"{best.model} is the only model that passes all acceptance criteria "
                    f"(score: {best.benchmark_result.overall_score:.1f}/100)."
                )
            return best.model, reason

        # No model passes all criteria — report which criteria each failed
        failure_details: list[str] = []
        for r in evaluated:
            failed_criteria = [c for c in r.criterion_results if not c.passed]
            failed_names = [c.criterion.name for c in failed_criteria]
            failure_details.append(
                f"{r.model}: failed {', '.join(failed_names)} "
                f"({r.criteria_passed_count}/{r.criteria_total} passed)"
            )

        reason = (
            "No candidate model meets all acceptance criteria. "
            "Recommend keeping the current model for this task. "
            "Details: " + "; ".join(failure_details)
        )
        return None, reason

    def _skipped_result(self, candidate: CandidateModel, reason: str) -> ModelEvalResult:
        """Create a placeholder result for a skipped model."""
        empty_benchmark = BenchmarkResult(
            model=candidate.name,
            provider=candidate.provider.upper(),
            timestamp=datetime.now(),
            duration_seconds=0.0,
            hardware=detect_hardware(),
            config=RunConfig(),
        )
        return ModelEvalResult(
            model=candidate.name,
            provider=candidate.provider,
            benchmark_result=empty_benchmark,
            skipped=True,
            skip_reason=reason,
        )
