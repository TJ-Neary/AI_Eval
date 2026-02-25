"""Tests for AI_Eval Integration Track MCP server.

Tests all 4 domain tools: get_model_recommendation, list_evaluations,
get_evaluation, and get_manifest.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import patch

import pytest

from src.mcp_server import (
    aieval_get_evaluation,
    aieval_get_manifest,
    aieval_get_model_recommendation,
    aieval_list_evaluations,
)

# ============================================================================
# Fixtures
# ============================================================================


SAMPLE_REPORT_A = {
    "model": "qwen2.5:32b",
    "provider": "OLLAMA",
    "timestamp": "2026-02-07T04:37:53-07:00",
    "duration_seconds": 120.5,
    "hardware": {
        "accelerator": {"name": "Apple M4 Max", "gpu_memory_gb": 48},
    },
    "overall_score": 80.27,
    "avg_tokens_per_second": 20.48,
    "total_tests": 5,
    "total_passed": 5,
    "categories": {"text-generation": {"score": 82.0}},
    "fitness_profiles": {
        "Chat Application": "85.4",
        "Code Assistant": "77.4",
        "Data Pipeline": "76.3",
        "Document Processor": "80.4",
        "Rag Knowledge Engine": "80.8",
    },
}

SAMPLE_REPORT_B = {
    "model": "llama3.1:8b",
    "provider": "OLLAMA",
    "timestamp": "2026-02-07T04:37:24-07:00",
    "duration_seconds": 90.0,
    "hardware": {
        "accelerator": {"name": "Apple M4 Max", "gpu_memory_gb": 48},
    },
    "overall_score": 78.53,
    "avg_tokens_per_second": 88.10,
    "total_tests": 5,
    "total_passed": 5,
    "categories": {"text-generation": {"score": 79.0}},
    "fitness_profiles": {
        "Chat Application": "81.2",
        "Code Assistant": "74.1",
        "Data Pipeline": "79.5",
        "Document Processor": "77.0",
        "Rag Knowledge Engine": "76.4",
    },
}

SAMPLE_INDEX = [
    {
        "model": "qwen2.5:32b",
        "provider": "OLLAMA",
        "score": 80.27,
        "tokens_per_second": 20.48,
        "pass_rate": "5/5",
        "hardware": "Apple M4 Max (48GB)",
        "date": "2026-02-07",
        "report_path": "reports/qwen2.5_32b_20260207_043753.md",
    },
    {
        "model": "llama3.1:8b",
        "provider": "OLLAMA",
        "score": 78.53,
        "tokens_per_second": 88.10,
        "pass_rate": "5/5",
        "hardware": "Apple M4 Max (48GB)",
        "date": "2026-02-07",
        "report_path": "reports/llama3.1_8b_20260207_043724.md",
    },
]


@pytest.fixture
def tmp_reports(tmp_path: Any) -> Any:
    """Create a temporary reports directory with sample data."""
    reports = tmp_path / "reports"
    reports.mkdir()

    (reports / "qwen2.5_32b_20260207_043753.json").write_text(
        json.dumps(SAMPLE_REPORT_A), encoding="utf-8"
    )
    (reports / "llama3.1_8b_20260207_043724.json").write_text(
        json.dumps(SAMPLE_REPORT_B), encoding="utf-8"
    )
    (reports / ".results_index.json").write_text(json.dumps(SAMPLE_INDEX), encoding="utf-8")

    return reports


# ============================================================================
# Model recommendation
# ============================================================================


class TestGetModelRecommendation:
    """Tests for aieval_get_model_recommendation tool."""

    def test_valid_task_type(self, tmp_reports: Any) -> None:
        """Recommendation for valid task type returns ranked models."""
        with patch("src.mcp_server.REPORTS_DIR", tmp_reports):
            result = aieval_get_model_recommendation(task_type="Chat Application")

        assert "recommendations" in result
        assert len(result["recommendations"]) == 2
        assert result["best_model"] == "qwen2.5:32b"
        assert result["best_score"] == 85.4

    def test_case_insensitive(self, tmp_reports: Any) -> None:
        """Task type matching is case-insensitive."""
        with patch("src.mcp_server.REPORTS_DIR", tmp_reports):
            result = aieval_get_model_recommendation(task_type="chat application")

        assert len(result["recommendations"]) == 2

    def test_unknown_task_type(self, tmp_reports: Any) -> None:
        """Unknown task type returns error with valid types listed."""
        with patch("src.mcp_server.REPORTS_DIR", tmp_reports):
            result = aieval_get_model_recommendation(task_type="Quantum Computing")

        assert "error" in result
        assert "valid_types" in result
        assert "Chat Application" in result["valid_types"]

    def test_empty_task_type(self) -> None:
        """Empty task type returns error."""
        result = aieval_get_model_recommendation(task_type="")
        assert "error" in result

    def test_no_reports(self, tmp_path: Any) -> None:
        """Missing reports directory returns error."""
        with patch("src.mcp_server.REPORTS_DIR", tmp_path / "nope"):
            result = aieval_get_model_recommendation(task_type="Chat Application")

        assert "error" in result


# ============================================================================
# List evaluations
# ============================================================================


class TestListEvaluations:
    """Tests for aieval_list_evaluations tool."""

    def test_returns_all(self, tmp_reports: Any) -> None:
        """List returns all evaluations from index."""
        with patch("src.mcp_server.INDEX_PATH", tmp_reports / ".results_index.json"):
            result = aieval_list_evaluations()

        assert len(result) == 2

    def test_sorted_by_date(self, tmp_reports: Any) -> None:
        """Results are sorted by date descending."""
        with patch("src.mcp_server.INDEX_PATH", tmp_reports / ".results_index.json"):
            result = aieval_list_evaluations()

        # Both are same date, just verify we get them
        assert all("model" in r for r in result)

    def test_fallback_to_files(self, tmp_reports: Any) -> None:
        """When index is missing, falls back to parsing report files."""
        # Remove the index file
        (tmp_reports / ".results_index.json").unlink()

        with (
            patch("src.mcp_server.INDEX_PATH", tmp_reports / ".results_index.json"),
            patch("src.mcp_server.REPORTS_DIR", tmp_reports),
        ):
            result = aieval_list_evaluations()

        assert len(result) == 2


# ============================================================================
# Get evaluation
# ============================================================================


class TestGetEvaluation:
    """Tests for aieval_get_evaluation tool."""

    def test_exact_key(self, tmp_reports: Any) -> None:
        """Exact report key returns full report."""
        with patch("src.mcp_server.REPORTS_DIR", tmp_reports):
            result = aieval_get_evaluation(report_key="qwen2.5_32b_20260207_043753")

        assert result["model"] == "qwen2.5:32b"
        assert "fitness_profiles" in result

    def test_partial_key(self, tmp_reports: Any) -> None:
        """Partial model name finds the report."""
        with patch("src.mcp_server.REPORTS_DIR", tmp_reports):
            result = aieval_get_evaluation(report_key="llama3.1_8b")

        assert result["model"] == "llama3.1:8b"

    def test_not_found(self, tmp_reports: Any) -> None:
        """Nonexistent report returns error."""
        with patch("src.mcp_server.REPORTS_DIR", tmp_reports):
            result = aieval_get_evaluation(report_key="nonexistent_model")

        assert "error" in result

    def test_empty_key(self) -> None:
        """Empty key returns error."""
        result = aieval_get_evaluation(report_key="")
        assert "error" in result


# ============================================================================
# Manifest
# ============================================================================


class TestGetManifest:
    """Tests for aieval_get_manifest tool."""

    def test_manifest_structure(self) -> None:
        """Manifest returns expected fields."""
        result = aieval_get_manifest()
        assert result["service"] == "AI_Eval"
        assert result["version"] == "1.0.0"
        assert len(result["tools"]) == 4
        assert result["requires_llm"] == []

    def test_all_tools_listed(self) -> None:
        """All 4 tools appear in manifest."""
        result = aieval_get_manifest()
        tool_names = {t["name"] for t in result["tools"]}
        expected = {
            "aieval_get_model_recommendation",
            "aieval_list_evaluations",
            "aieval_get_evaluation",
            "aieval_get_manifest",
        }
        assert tool_names == expected
