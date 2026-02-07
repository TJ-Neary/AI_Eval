"""Tests for the report generator module."""

import json
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.reporting.report_generator import (
    ReportConfig,
    ReportGenerator,
    _calculate_fitness_scores,
    _sanitize_model_name,
    _score_rating,
)


# ── Fixtures ────────────────────────────────────────────────────────────────


def _make_test_result(status_name: str = "PASSED", score: float = 85.0) -> MagicMock:
    """Create a mock TestResult."""
    t = MagicMock()
    t.test_id = "test-001"
    t.status.name = status_name
    t.score = score
    t.generation_time_ms = 1200.0
    t.tokens_per_second = 45.3
    t.prompt_tokens = 50
    t.completion_tokens = 150
    return t


def _make_category_result(
    name: str = "text-generation",
    avg_score: float = 80.0,
    passed: int = 4,
    total: int = 5,
    tps: float = 42.0,
) -> MagicMock:
    """Create a mock CategoryResult."""
    test = _make_test_result()
    cat = MagicMock()
    cat.category = name
    cat.avg_score = avg_score
    cat.passed = passed
    cat.total_tests = total
    cat.pass_rate = passed / total if total > 0 else 0.0
    cat.avg_tokens_per_second = tps
    cat.tests = [test]
    return cat


def _make_hardware() -> MagicMock:
    """Create a mock HardwareProfile."""
    hw = MagicMock()
    hw.chip_name = "Apple M4 Max"
    hw.chip_type.name = "APPLE_SILICON"
    hw.ram_gb = 48.0
    hw.gpu_cores = 40
    hw.memory_bandwidth_gbps = 546.0
    hw.os_name = "Darwin"
    hw.os_version = "25.2.0"
    return hw


def _make_benchmark_result() -> MagicMock:
    """Create a mock BenchmarkResult with full data."""
    result = MagicMock()
    result.model = "qwen2.5:32b"
    result.provider = "ollama"
    result.timestamp = datetime(2026, 2, 5, 14, 30, 0)
    result.duration_seconds = 123.4
    result.hardware = _make_hardware()
    result.overall_score = 78.5
    result.avg_tokens_per_second = 42.1
    result.total_tests = 10
    result.total_passed = 8
    result.total_tokens = 5000
    result.config.temperature = 0.1
    result.config.max_tokens = 2048
    result.config.warmup_queries = 3
    result.config.repetitions = 1
    result.config.timeout_seconds = 120.0
    result.config.use_llm_judge = True
    result.categories = {
        "text-generation": _make_category_result("text-generation", 82.0, 4, 5, 45.0),
        "code-generation": _make_category_result("code-generation", 75.0, 3, 5, 38.0),
    }
    result.to_dict.return_value = {
        "model": "qwen2.5:32b",
        "provider": "ollama",
        "timestamp": "2026-02-05T14:30:00",
        "overall_score": 78.5,
    }
    return result


# ── Helper Function Tests ───────────────────────────────────────────────────


class TestSanitizeModelName:
    def test_replaces_colons(self) -> None:
        assert _sanitize_model_name("qwen2.5:32b") == "qwen2.5_32b"

    def test_replaces_slashes(self) -> None:
        assert _sanitize_model_name("org/model:tag") == "org_model_tag"

    def test_preserves_dots_and_dashes(self) -> None:
        assert _sanitize_model_name("llama-3.2") == "llama-3.2"


class TestScoreRating:
    def test_excellent(self) -> None:
        assert _score_rating(95.0) == "Excellent"

    def test_good(self) -> None:
        assert _score_rating(80.0) == "Good"

    def test_adequate(self) -> None:
        assert _score_rating(65.0) == "Adequate"

    def test_marginal(self) -> None:
        assert _score_rating(45.0) == "Marginal"

    def test_poor(self) -> None:
        assert _score_rating(20.0) == "Poor"

    def test_custom_thresholds(self) -> None:
        thresholds = {"excellent": 95, "good": 80, "adequate": 65, "marginal": 50}
        assert _score_rating(82.0, thresholds) == "Good"
        assert _score_rating(90.0, thresholds) == "Good"  # Below 95


class TestCalculateFitnessScores:
    def test_calculates_weighted_scores(self) -> None:
        category_scores = {"text-generation": 80.0, "code-generation": 90.0}
        profiles = {
            "code-assistant": {"text-generation": 0.3, "code-generation": 0.7},
        }
        result = _calculate_fitness_scores(category_scores, profiles)
        expected = (80.0 * 0.3 + 90.0 * 0.7) / 1.0
        assert abs(result["code-assistant"] - expected) < 0.01

    def test_handles_missing_categories(self) -> None:
        category_scores = {"text-generation": 80.0}
        profiles = {
            "test": {"text-generation": 0.5, "missing-category": 0.5},
        }
        result = _calculate_fitness_scores(category_scores, profiles)
        # Only text-generation contributes: 80.0 * 0.5 / 0.5 = 80.0
        assert abs(result["test"] - 80.0) < 0.01

    def test_returns_zero_for_no_matching_categories(self) -> None:
        result = _calculate_fitness_scores({}, {"test": {"missing": 1.0}})
        assert result["test"] == 0.0

    def test_multiple_profiles(self) -> None:
        scores = {"text-generation": 70.0, "code-generation": 90.0}
        profiles = {
            "profile-a": {"text-generation": 1.0},
            "profile-b": {"code-generation": 1.0},
        }
        result = _calculate_fitness_scores(scores, profiles)
        assert abs(result["profile-a"] - 70.0) < 0.01
        assert abs(result["profile-b"] - 90.0) < 0.01


# ── Report Generator Tests ─────────────────────────────────────────────────


class TestReportGenerator:
    def test_generates_markdown_report(self, tmp_path: Path) -> None:
        config = ReportConfig(report_dir=tmp_path, formats=["markdown"])
        gen = ReportGenerator(config=config)
        result = _make_benchmark_result()

        report_path = gen.generate(result)

        assert report_path.exists()
        assert report_path.suffix == ".md"
        content = report_path.read_text()
        assert "qwen2.5:32b" in content
        assert "text-generation" in content

    def test_generates_json_report(self, tmp_path: Path) -> None:
        config = ReportConfig(report_dir=tmp_path, formats=["json"])
        gen = ReportGenerator(config=config)
        result = _make_benchmark_result()

        report_path = gen.generate(result)

        json_path = report_path.with_suffix(".json")
        assert json_path.exists()
        data = json.loads(json_path.read_text())
        assert data["model"] == "qwen2.5:32b"
        assert "fitness_profiles" in data

    def test_generates_both_formats(self, tmp_path: Path) -> None:
        config = ReportConfig(report_dir=tmp_path, formats=["markdown", "json"])
        gen = ReportGenerator(config=config)
        result = _make_benchmark_result()

        report_path = gen.generate(result)

        assert report_path.exists()
        assert report_path.with_suffix(".json").exists()

    def test_creates_report_dir(self, tmp_path: Path) -> None:
        report_dir = tmp_path / "nested" / "reports"
        config = ReportConfig(report_dir=report_dir, formats=["markdown"])
        gen = ReportGenerator(config=config)
        result = _make_benchmark_result()

        gen.generate(result)

        assert report_dir.exists()

    def test_filename_contains_model_and_timestamp(self, tmp_path: Path) -> None:
        config = ReportConfig(report_dir=tmp_path, formats=["markdown"])
        gen = ReportGenerator(config=config)
        result = _make_benchmark_result()

        report_path = gen.generate(result)

        assert "qwen2.5_32b" in report_path.name
        assert "20260205" in report_path.name

    def test_report_contains_hardware_info(self, tmp_path: Path) -> None:
        config = ReportConfig(report_dir=tmp_path, formats=["markdown"])
        gen = ReportGenerator(config=config)
        result = _make_benchmark_result()

        report_path = gen.generate(result)
        content = report_path.read_text()

        assert "Apple M4 Max" in content
        assert "48GB" in content

    def test_report_contains_category_breakdown(self, tmp_path: Path) -> None:
        config = ReportConfig(report_dir=tmp_path, formats=["markdown"])
        gen = ReportGenerator(config=config)
        result = _make_benchmark_result()

        report_path = gen.generate(result)
        content = report_path.read_text()

        assert "text-generation" in content
        assert "code-generation" in content
        assert "82.0" in content

    def test_report_contains_fitness_profiles(self, tmp_path: Path) -> None:
        config = ReportConfig(report_dir=tmp_path, formats=["markdown"])
        gen = ReportGenerator(config=config)
        result = _make_benchmark_result()

        report_path = gen.generate(result)
        content = report_path.read_text()

        assert "Fitness Profile" in content
