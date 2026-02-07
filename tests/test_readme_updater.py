"""Tests for the README updater module."""

import json
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.reporting.readme_updater import _merge_result, _render_results_table, update_readme_results

# ── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def readme_with_markers(tmp_path: Path) -> Path:
    """Create a README.md with marker section."""
    f = tmp_path / "README.md"
    f.write_text(
        "# AI_Eval\n\n"
        "## Benchmark Results\n\n"
        "<!-- AI_EVAL:BEGIN -->\n"
        "*No benchmark results yet.*\n"
        "<!-- AI_EVAL:END -->\n\n"
        "## Footer\n"
    )
    return f


@pytest.fixture
def readme_without_markers(tmp_path: Path) -> Path:
    """Create a README.md without markers."""
    f = tmp_path / "plain.md"
    f.write_text("# AI_Eval\n\nJust a normal file.\n")
    return f


def _make_mock_result(
    model: str = "qwen2.5:32b",
    provider: str = "ollama",
    score: float = 78.5,
    tps: float = 42.0,
    passed: int = 8,
    total: int = 10,
) -> MagicMock:
    """Create a mock BenchmarkResult."""
    result = MagicMock()
    result.model = model
    result.provider = provider
    result.overall_score = score
    result.avg_tokens_per_second = tps
    result.total_passed = passed
    result.total_tests = total
    result.timestamp = datetime(2026, 2, 5, 14, 30, 0)
    result.hardware.chip_name = "Apple M4 Max"
    result.hardware.ram_gb = 48.0
    return result


# ── Merge Tests ─────────────────────────────────────────────────────────────


class TestMergeResult:
    def test_adds_new_entry(self) -> None:
        existing: list = []
        new = {"model": "qwen2.5:32b", "provider": "ollama", "score": 80.0}
        merged = _merge_result(existing, new)
        assert len(merged) == 1
        assert merged[0]["model"] == "qwen2.5:32b"

    def test_replaces_same_model_provider(self) -> None:
        existing = [{"model": "qwen2.5:32b", "provider": "ollama", "score": 70.0}]
        new = {"model": "qwen2.5:32b", "provider": "ollama", "score": 85.0}
        merged = _merge_result(existing, new)
        assert len(merged) == 1
        assert merged[0]["score"] == 85.0

    def test_keeps_different_models(self) -> None:
        existing = [{"model": "llama3.2:3b", "provider": "ollama", "score": 60.0}]
        new = {"model": "qwen2.5:32b", "provider": "ollama", "score": 80.0}
        merged = _merge_result(existing, new)
        assert len(merged) == 2

    def test_sorts_by_score_descending(self) -> None:
        existing = [{"model": "model-a", "provider": "ollama", "score": 60.0}]
        new = {"model": "model-b", "provider": "ollama", "score": 90.0}
        merged = _merge_result(existing, new)
        assert merged[0]["score"] == 90.0
        assert merged[1]["score"] == 60.0

    def test_keeps_different_providers(self) -> None:
        existing = [{"model": "gemini-pro", "provider": "google", "score": 85.0}]
        new = {"model": "gemini-pro", "provider": "ollama", "score": 70.0}
        merged = _merge_result(existing, new)
        assert len(merged) == 2


# ── Render Table Tests ──────────────────────────────────────────────────────


class TestRenderResultsTable:
    def test_renders_empty_placeholder(self) -> None:
        table = _render_results_table([])
        assert "No benchmark results yet" in table

    def test_renders_table_with_headers(self) -> None:
        summaries = [
            {
                "model": "qwen2.5:32b",
                "provider": "ollama",
                "score": 78.5,
                "tokens_per_second": 42.0,
                "pass_rate": "8/10",
                "hardware": "Apple M4 Max (48GB)",
                "date": "2026-02-05",
                "report_path": "",
            }
        ]
        table = _render_results_table(summaries)
        assert "| Model |" in table
        assert "qwen2.5:32b" in table
        assert "78.5" in table

    def test_renders_report_link(self) -> None:
        summaries = [
            {
                "model": "qwen2.5:32b",
                "provider": "ollama",
                "score": 80.0,
                "tokens_per_second": 40.0,
                "pass_rate": "8/10",
                "hardware": "Test HW",
                "date": "2026-02-05",
                "report_path": "reports/qwen2.5_32b.md",
            }
        ]
        table = _render_results_table(summaries)
        assert "[qwen2.5:32b](reports/qwen2.5_32b.md)" in table

    def test_renders_multiple_rows(self) -> None:
        summaries = [
            {
                "model": "model-a",
                "provider": "ollama",
                "score": 90.0,
                "tokens_per_second": 50.0,
                "pass_rate": "9/10",
                "hardware": "HW",
                "date": "2026-02-05",
                "report_path": "",
            },
            {
                "model": "model-b",
                "provider": "google",
                "score": 70.0,
                "tokens_per_second": 30.0,
                "pass_rate": "7/10",
                "hardware": "HW",
                "date": "2026-02-05",
                "report_path": "",
            },
        ]
        table = _render_results_table(summaries)
        assert "model-a" in table
        assert "model-b" in table


# ── Integration Tests ───────────────────────────────────────────────────────


class TestUpdateReadmeResults:
    def test_updates_readme_with_results(self, readme_with_markers: Path, tmp_path: Path) -> None:
        result = _make_mock_result()
        index_path = tmp_path / "index.json"

        success = update_readme_results(
            result,
            readme_path=readme_with_markers,
            index_path=index_path,
        )

        assert success is True
        content = readme_with_markers.read_text()
        assert "qwen2.5:32b" in content
        assert "78.5" in content
        assert "No benchmark results yet" not in content

    def test_returns_false_without_markers(
        self, readme_without_markers: Path, tmp_path: Path
    ) -> None:
        result = _make_mock_result()
        index_path = tmp_path / "index.json"

        success = update_readme_results(
            result,
            readme_path=readme_without_markers,
            index_path=index_path,
        )

        assert success is False

    def test_creates_index_file(self, readme_with_markers: Path, tmp_path: Path) -> None:
        result = _make_mock_result()
        index_path = tmp_path / "reports" / "index.json"

        update_readme_results(
            result,
            readme_path=readme_with_markers,
            index_path=index_path,
        )

        assert index_path.exists()
        data = json.loads(index_path.read_text())
        assert len(data) == 1

    def test_accumulates_multiple_results(self, readme_with_markers: Path, tmp_path: Path) -> None:
        index_path = tmp_path / "index.json"

        result1 = _make_mock_result(model="model-a", score=80.0)
        result2 = _make_mock_result(model="model-b", score=70.0)

        update_readme_results(result1, readme_path=readme_with_markers, index_path=index_path)
        update_readme_results(result2, readme_path=readme_with_markers, index_path=index_path)

        data = json.loads(index_path.read_text())
        assert len(data) == 2

        content = readme_with_markers.read_text()
        assert "model-a" in content
        assert "model-b" in content

    def test_replaces_same_model_on_rerun(self, readme_with_markers: Path, tmp_path: Path) -> None:
        index_path = tmp_path / "index.json"

        result_v1 = _make_mock_result(score=70.0)
        result_v2 = _make_mock_result(score=85.0)

        update_readme_results(result_v1, readme_path=readme_with_markers, index_path=index_path)
        update_readme_results(result_v2, readme_path=readme_with_markers, index_path=index_path)

        data = json.loads(index_path.read_text())
        assert len(data) == 1
        assert data[0]["score"] == 85.0

    def test_preserves_content_outside_markers(
        self, readme_with_markers: Path, tmp_path: Path
    ) -> None:
        result = _make_mock_result()
        index_path = tmp_path / "index.json"

        update_readme_results(result, readme_path=readme_with_markers, index_path=index_path)

        content = readme_with_markers.read_text()
        assert "# AI_Eval" in content
        assert "## Footer" in content

    def test_includes_report_link(self, readme_with_markers: Path, tmp_path: Path) -> None:
        result = _make_mock_result()
        index_path = tmp_path / "index.json"
        report = Path("reports/test_report.md")

        update_readme_results(
            result,
            readme_path=readme_with_markers,
            report_path=report,
            index_path=index_path,
        )

        content = readme_with_markers.read_text()
        assert "reports/test_report.md" in content
