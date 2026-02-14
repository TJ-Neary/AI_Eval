"""Tests for evaluation runner orchestration."""

from datetime import datetime
from unittest.mock import MagicMock

from src.benchmarks.runner import BenchmarkResult, CategoryResult, RunConfig, TestResult, TestStatus
from src.evaluation.config import AcceptanceCriterion, CandidateModel, EvalRequestConfig, Scenario
from src.evaluation.runner import (
    CriterionResult,
    EvaluationResult,
    EvaluationRunner,
    ModelEvalResult,
)
from src.evaluation.scorers import ScorerResult


def _make_hardware() -> MagicMock:
    hw = MagicMock()
    hw.chip_name = "Apple M4 Max"
    hw.chip_type.name = "APPLE_SILICON"
    hw.ram_gb = 48.0
    hw.to_dict.return_value = {"chip_name": "Apple M4 Max", "ram_gb": 48.0}
    return hw


def _make_test_result(
    test_id: str = "test-001",
    response: str = '{"claims": []}',
    generation_time_ms: float = 5000.0,
    tokens_per_second: float = 20.0,
    score: float = 80.0,
    passed: bool = True,
) -> TestResult:
    return TestResult(
        test_id=test_id,
        status=TestStatus.PASSED if passed else TestStatus.FAILED,
        prompt="test prompt",
        response=response,
        generation_time_ms=generation_time_ms,
        tokens_per_second=tokens_per_second,
        score=score,
    )


def _make_benchmark_result(
    model: str = "qwen2.5:32b",
    test_results: list[TestResult] | None = None,
) -> BenchmarkResult:
    if test_results is None:
        test_results = [_make_test_result()]
    hw = _make_hardware()
    return BenchmarkResult(
        model=model,
        provider="OLLAMA",
        timestamp=datetime.now(),
        duration_seconds=10.0,
        hardware=hw,
        config=RunConfig(),
        categories={
            "text-generation": CategoryResult(category="text-generation", tests=test_results)
        },
        total_tests=len(test_results),
        total_passed=sum(1 for t in test_results if t.status == TestStatus.PASSED),
    )


def _make_eval_config(
    candidates: list[CandidateModel] | None = None,
    criteria: list[AcceptanceCriterion] | None = None,
    scorers: list[str] | None = None,
) -> EvalRequestConfig:
    if candidates is None:
        candidates = [CandidateModel(name="qwen2.5:32b", provider="ollama")]
    if criteria is None:
        criteria = [
            AcceptanceCriterion(
                name="JSON validity", metric="json_validity", threshold=0.9, operator=">="
            ),
        ]
    return EvalRequestConfig(
        request_id="test-eval",
        requesting_project="TestProject",
        date="2026-02-09",
        use_case="Test evaluation",
        model_capability="text-generation",
        task_description="Test task",
        candidates=candidates,
        acceptance_criteria=criteria,
        scenarios=[
            Scenario(
                id="test-001",
                description="test scenario",
                prompt="Answer with JSON",
                metadata={"source_text": "Evidence on Page 3"},
            ),
        ],
        custom_scorers=scorers if scorers is not None else ["json_validity"],
    )


class TestEvaluationRunnerApplyScorers:
    def test_applies_custom_scorers(self) -> None:
        runner = EvaluationRunner()
        config = _make_eval_config(scorers=["json_validity"])
        benchmark = _make_benchmark_result(
            test_results=[_make_test_result(response='{"claims": []}')]
        )
        dataset = config.to_dataset()

        scores = runner._apply_custom_scorers(benchmark, config, dataset)

        assert "json_validity" in scores
        assert len(scores["json_validity"]) == 1
        assert scores["json_validity"][0].passed is True

    def test_applies_multiple_scorers(self) -> None:
        runner = EvaluationRunner()
        config = _make_eval_config(scorers=["json_validity", "latency"])
        benchmark = _make_benchmark_result(
            test_results=[_make_test_result(response='{"data": 1}', generation_time_ms=5000)]
        )
        dataset = config.to_dataset()

        scores = runner._apply_custom_scorers(benchmark, config, dataset)

        assert "json_validity" in scores
        assert "latency" in scores

    def test_unknown_scorer_skipped(self) -> None:
        runner = EvaluationRunner()
        config = _make_eval_config(scorers=["nonexistent_scorer"])
        benchmark = _make_benchmark_result()
        dataset = config.to_dataset()

        scores = runner._apply_custom_scorers(benchmark, config, dataset)
        assert scores == {}

    def test_no_scorers_returns_empty(self) -> None:
        runner = EvaluationRunner()
        config = _make_eval_config(scorers=[])
        benchmark = _make_benchmark_result()
        dataset = config.to_dataset()

        scores = runner._apply_custom_scorers(benchmark, config, dataset)
        assert scores == {}


class TestAcceptanceCriteriaEvaluation:
    def test_criteria_pass(self) -> None:
        runner = EvaluationRunner()
        benchmark = _make_benchmark_result()
        criteria = [
            AcceptanceCriterion(name="JSON", metric="json_validity", threshold=0.5, operator=">="),
        ]
        custom_scores = {
            "json_validity": [ScorerResult(name="json_validity", score=1.0, passed=True)],
        }

        results = runner._evaluate_acceptance_criteria(benchmark, custom_scores, criteria)

        assert len(results) == 1
        assert results[0].passed is True
        assert results[0].measured_value == 1.0

    def test_criteria_fail(self) -> None:
        runner = EvaluationRunner()
        benchmark = _make_benchmark_result()
        criteria = [
            AcceptanceCriterion(name="JSON", metric="json_validity", threshold=0.95, operator=">="),
        ]
        custom_scores = {
            "json_validity": [
                ScorerResult(name="json_validity", score=1.0, passed=True),
                ScorerResult(name="json_validity", score=0.0, passed=False),
            ],
        }

        results = runner._evaluate_acceptance_criteria(benchmark, custom_scores, criteria)

        assert len(results) == 1
        assert results[0].passed is False
        assert results[0].measured_value == 0.5  # Average of 1.0 and 0.0

    def test_latency_criterion_uses_raw_values(self) -> None:
        runner = EvaluationRunner()
        benchmark = _make_benchmark_result()
        criteria = [
            AcceptanceCriterion(
                name="Latency", metric="latency", threshold=15.0, operator="<=", unit="seconds"
            ),
        ]
        custom_scores = {
            "latency": [
                ScorerResult(name="latency", score=1.0, passed=True, raw_value=5.0),
                ScorerResult(name="latency", score=1.0, passed=True, raw_value=10.0),
            ],
        }

        results = runner._evaluate_acceptance_criteria(benchmark, custom_scores, criteria)

        assert results[0].passed is True
        assert results[0].measured_value == 7.5  # Average of 5.0 and 10.0

    def test_missing_scorer_falls_back_to_benchmark(self) -> None:
        runner = EvaluationRunner()
        benchmark = _make_benchmark_result(
            test_results=[_make_test_result(generation_time_ms=8000.0)]
        )
        criteria = [
            AcceptanceCriterion(
                name="Latency", metric="latency", threshold=15.0, operator="<=", unit="seconds"
            ),
        ]

        results = runner._evaluate_acceptance_criteria(benchmark, {}, criteria)

        assert results[0].passed is True
        assert results[0].measured_value == 8.0  # 8000ms = 8s


class TestRecommendation:
    def test_recommends_best_qualifying(self) -> None:
        runner = EvaluationRunner()
        results = [
            ModelEvalResult(
                model="model-a",
                provider="ollama",
                benchmark_result=_make_benchmark_result(model="model-a"),
                criterion_results=[
                    CriterionResult(
                        criterion=AcceptanceCriterion(name="t", metric="m", threshold=0.5),
                        measured_value=0.8,
                        passed=True,
                    ),
                ],
                all_criteria_passed=True,
            ),
            ModelEvalResult(
                model="model-b",
                provider="ollama",
                benchmark_result=_make_benchmark_result(model="model-b"),
                criterion_results=[
                    CriterionResult(
                        criterion=AcceptanceCriterion(name="t", metric="m", threshold=0.5),
                        measured_value=0.9,
                        passed=True,
                    ),
                ],
                all_criteria_passed=True,
            ),
        ]

        recommended, reason = runner._select_recommendation(results)

        # Both pass, pick by overall_score (which is same since same benchmark_result)
        assert recommended is not None
        assert "passes all acceptance criteria" in reason

    def test_no_model_qualifies(self) -> None:
        runner = EvaluationRunner()
        results = [
            ModelEvalResult(
                model="model-a",
                provider="ollama",
                benchmark_result=_make_benchmark_result(model="model-a"),
                criterion_results=[
                    CriterionResult(
                        criterion=AcceptanceCriterion(
                            name="JSON", metric="json_validity", threshold=0.95
                        ),
                        measured_value=0.5,
                        passed=False,
                    ),
                ],
                all_criteria_passed=False,
            ),
        ]

        recommended, reason = runner._select_recommendation(results)

        assert recommended is None
        assert "No candidate model meets all acceptance criteria" in reason
        assert "model-a" in reason

    def test_all_skipped(self) -> None:
        runner = EvaluationRunner()
        results = [
            ModelEvalResult(
                model="model-a",
                provider="ollama",
                benchmark_result=_make_benchmark_result(),
                skipped=True,
                skip_reason="Not available",
            ),
        ]

        recommended, reason = runner._select_recommendation(results)

        assert recommended is None
        assert "unavailable" in reason


class TestEvaluationResultSerialization:
    def test_to_dict(self) -> None:
        config = _make_eval_config()
        result = EvaluationResult(
            request_config=config,
            model_results=[
                ModelEvalResult(
                    model="qwen2.5:32b",
                    provider="ollama",
                    benchmark_result=_make_benchmark_result(),
                    criterion_results=[
                        CriterionResult(
                            criterion=AcceptanceCriterion(
                                name="JSON", metric="json_validity", threshold=0.9
                            ),
                            measured_value=0.95,
                            passed=True,
                        ),
                    ],
                    all_criteria_passed=True,
                ),
            ],
            recommended_model="qwen2.5:32b",
            recommendation_reason="Best model",
        )

        d = result.to_dict()

        assert d["request_id"] == "test-eval"
        assert d["recommended_model"] == "qwen2.5:32b"
        assert len(d["model_results"]) == 1
        assert d["model_results"][0]["all_criteria_passed"] is True
        assert d["model_results"][0]["criterion_details"][0]["passed"] is True
