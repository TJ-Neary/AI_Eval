"""Tests for evaluation report generation."""

from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

from src.benchmarks.runner import BenchmarkResult, CategoryResult, RunConfig, TestResult, TestStatus
from src.evaluation.config import AcceptanceCriterion, CandidateModel, EvalRequestConfig, Scenario
from src.evaluation.report import EvaluationReportGenerator
from src.evaluation.runner import CriterionResult, EvaluationResult, ModelEvalResult
from src.reporting.report_generator import ReportConfig


def _make_hardware() -> MagicMock:
    hw = MagicMock()
    hw.chip_name = "Apple M4 Max"
    hw.chip_type.name = "APPLE_SILICON"
    hw.ram_gb = 48.0
    hw.os_name = "macOS"
    hw.os_version = "15.3"
    hw.to_dict.return_value = {"chip_name": "Apple M4 Max", "ram_gb": 48.0}
    return hw


def _make_eval_result(recommended: str | None = "qwen2.5:32b") -> EvaluationResult:
    hw = _make_hardware()
    benchmark = BenchmarkResult(
        model="qwen2.5:32b",
        provider="OLLAMA",
        timestamp=datetime.now(),
        duration_seconds=30.0,
        hardware=hw,
        config=RunConfig(),
        categories={
            "text-generation": CategoryResult(
                category="text-generation",
                tests=[
                    TestResult(
                        test_id="test-001",
                        status=TestStatus.PASSED,
                        prompt="test",
                        response='{"claims": []}',
                        generation_time_ms=5000.0,
                        tokens_per_second=20.0,
                        score=80.0,
                    )
                ],
            )
        },
        total_tests=1,
        total_passed=1,
    )

    config = EvalRequestConfig(
        request_id="test-eval-001",
        requesting_project="TestProject",
        date="2026-02-09",
        use_case="Test evaluation",
        model_capability="text-generation",
        task_description="Test task description",
        input_description="Test input",
        output_description="Test output",
        complexity="MEDIUM",
        current_model="Claude Sonnet 4",
        candidates=[CandidateModel(name="qwen2.5:32b")],
        acceptance_criteria=[
            AcceptanceCriterion(name="JSON validity", metric="json_validity", threshold=0.9),
        ],
        scenarios=[Scenario(id="test-001", description="test", prompt="test")],
    )

    model_result = ModelEvalResult(
        model="qwen2.5:32b",
        provider="ollama",
        benchmark_result=benchmark,
        criterion_results=[
            CriterionResult(
                criterion=AcceptanceCriterion(
                    name="JSON validity", metric="json_validity", threshold=0.9
                ),
                measured_value=0.95,
                passed=True,
            ),
        ],
        all_criteria_passed=True,
    )

    return EvaluationResult(
        request_config=config,
        model_results=[model_result],
        recommended_model=recommended,
        recommendation_reason="Best model for the task",
        hardware=hw,
        duration_seconds=30.0,
    )


class TestEvaluationReportGenerator:
    def test_generates_markdown(self, tmp_path: Path) -> None:
        report_config = ReportConfig(report_dir=tmp_path, formats=["markdown"])
        generator = EvaluationReportGenerator(config=report_config)
        result = _make_eval_result()

        path = generator.generate(result)

        assert path.exists()
        assert path.suffix == ".md"
        content = path.read_text()
        assert "Test evaluation" in content
        assert "TestProject" in content

    def test_generates_json(self, tmp_path: Path) -> None:
        report_config = ReportConfig(report_dir=tmp_path, formats=["json"])
        generator = EvaluationReportGenerator(config=report_config)
        result = _make_eval_result()

        path = generator.generate(result)

        json_path = path.with_suffix(".json")
        assert json_path.exists()

    def test_includes_recommendation(self, tmp_path: Path) -> None:
        report_config = ReportConfig(report_dir=tmp_path, formats=["markdown"])
        generator = EvaluationReportGenerator(config=report_config)
        result = _make_eval_result(recommended="qwen2.5:32b")

        path = generator.generate(result)
        content = path.read_text()

        assert "qwen2.5:32b" in content
        assert "Recommended" in content

    def test_no_model_qualifies_message(self, tmp_path: Path) -> None:
        report_config = ReportConfig(report_dir=tmp_path, formats=["markdown"])
        generator = EvaluationReportGenerator(config=report_config)
        result = _make_eval_result(recommended=None)
        result.recommendation_reason = "No model meets all acceptance criteria"

        path = generator.generate(result)
        content = path.read_text()

        assert "No model meets all acceptance criteria" in content

    def test_includes_acceptance_criteria(self, tmp_path: Path) -> None:
        report_config = ReportConfig(report_dir=tmp_path, formats=["markdown"])
        generator = EvaluationReportGenerator(config=report_config)
        result = _make_eval_result()

        path = generator.generate(result)
        content = path.read_text()

        assert "JSON validity" in content
        assert "PASS" in content

    def test_includes_task_description(self, tmp_path: Path) -> None:
        report_config = ReportConfig(report_dir=tmp_path, formats=["markdown"])
        generator = EvaluationReportGenerator(config=report_config)
        result = _make_eval_result()

        path = generator.generate(result)
        content = path.read_text()

        assert "Test task description" in content
        assert "Claude Sonnet 4" in content
