"""Tests for the catalog exporter module."""

from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

from src.export.catalog_exporter import (
    ExportConfig,
    _classify_model,
    _ensure_monthly_marker,
    _extract_param_count,
    _render_catalog_row,
    _render_decision_entry,
    _render_hardware_entry,
    _render_log_entry,
    _render_summary_row,
    _update_log_history,
    _update_marker_section,
    export_to_catalog,
)

# ── Fixtures ────────────────────────────────────────────────────────────────


def _make_hardware() -> MagicMock:
    hw = MagicMock()
    hw.chip_name = "Apple M4 Max"
    hw.chip_type.name = "APPLE_SILICON"
    hw.ram_gb = 48.0
    hw.gpu_cores = 40
    hw.memory_bandwidth_gbps = 546.0
    hw.os_name = "Darwin"
    hw.os_version = "25.2.0"
    return hw


def _make_category(name: str, score: float) -> MagicMock:
    cat = MagicMock()
    cat.category = name
    cat.avg_score = score
    cat.tests = [MagicMock()]
    cat.avg_tokens_per_second = 42.0
    return cat


def _make_result(model: str = "qwen2.5:32b", score: float = 80.0) -> MagicMock:
    result = MagicMock()
    result.model = model
    result.provider = "OLLAMA"
    result.overall_score = score
    result.avg_tokens_per_second = 20.5
    result.total_passed = 5
    result.total_tests = 5
    result.timestamp = datetime(2026, 2, 7, 14, 30, 0)
    result.hardware = _make_hardware()
    result.categories = {
        "text-generation": _make_category("text-generation", 94.0),
        "code-generation": _make_category("code-generation", 63.0),
        "conversational": _make_category("conversational", 89.0),
        "structured-output": _make_category("structured-output", 75.0),
    }
    return result


def _make_evaluation_log(eval_dir: Path) -> Path:
    """Create a mock EVALUATION_LOG.md with markers."""
    log = eval_dir / "EVALUATION_LOG.md"
    log.write_text(
        "# Evaluation Log\n\n"
        "## Summary Table\n\n"
        "<!-- AI_EVAL:BEGIN eval-summary -->\n"
        "| _(no evaluations yet)_ | | | | | | | |\n"
        "<!-- AI_EVAL:END eval-summary -->\n\n"
        "## Evaluation History\n\n"
        "Detailed records.\n\n"
        "### 2026-02\n\n"
        "<!-- AI_EVAL:BEGIN history-2026-02 -->\n"
        "_(No evaluations recorded yet.)_\n"
        "<!-- AI_EVAL:END history-2026-02 -->\n"
    )
    return log


def _make_evaluations_dir(tmp_path: Path) -> Path:
    """Create a mock _HQ/evaluations/ directory with catalog files."""
    eval_dir = tmp_path / "evaluations"
    eval_dir.mkdir()
    (eval_dir / "reports").mkdir()
    (eval_dir / "data").mkdir()

    # MODEL_CATALOG.md
    catalog = eval_dir / "MODEL_CATALOG.md"
    catalog.write_text(
        "# Model Catalog\n\n"
        "## Large\n\n"
        "<!-- AI_EVAL:BEGIN local-large -->\n"
        "| _Example_ | _32B_ | _Q4_ | _20 GB_ | _18_ | _85_ | _88_ | _84_ | _86_ | _Example_ | _→_ |\n"
        "<!-- AI_EVAL:END local-large -->\n\n"
        "## Midrange\n\n"
        "<!-- AI_EVAL:BEGIN local-midrange -->\n"
        "| _Example_ | _7B_ | _Q4_ | _5 GB_ | _45_ | _74_ | _72_ | _76_ | _78_ | _Example_ | _→_ |\n"
        "<!-- AI_EVAL:END local-midrange -->\n"
    )

    # DECISION_MATRIX.md
    matrix = eval_dir / "DECISION_MATRIX.md"
    matrix.write_text(
        "# Decision Matrix\n\n"
        "<!-- AI_EVAL:BEGIN decision-matrix -->\n"
        "_Pending._\n"
        "<!-- AI_EVAL:END decision-matrix -->\n"
    )

    # HARDWARE_PROFILES.md
    hw = eval_dir / "HARDWARE_PROFILES.md"
    hw.write_text(
        "# Hardware Profiles\n\n"
        "<!-- AI_EVAL:BEGIN hardware-profiles -->\n"
        "_Pending._\n"
        "<!-- AI_EVAL:END hardware-profiles -->\n"
    )

    return eval_dir


# ── Unit Tests ──────────────────────────────────────────────────────────────


class TestExtractParamCount:
    def test_extracts_from_colon_suffix(self) -> None:
        assert _extract_param_count("qwen2.5:32b") == "32B"

    def test_extracts_from_name(self) -> None:
        assert _extract_param_count("llama3.1:8b") == "8B"

    def test_extracts_decimal(self) -> None:
        assert _extract_param_count("phi-3-mini:3.8b") == "3.8B"

    def test_returns_none_for_no_params(self) -> None:
        assert _extract_param_count("nomic-embed-text:latest") is None

    def test_extracts_from_name_without_colon(self) -> None:
        assert _extract_param_count("gemma2:27b") == "27B"


class TestClassifyModel:
    def test_compact(self) -> None:
        assert _classify_model("llama3.2:3b", "OLLAMA") == "local-compact"

    def test_midrange(self) -> None:
        assert _classify_model("llama3.1:8b", "OLLAMA") == "local-midrange"

    def test_large(self) -> None:
        assert _classify_model("qwen2.5:32b", "OLLAMA") == "local-large"

    def test_large_27b(self) -> None:
        assert _classify_model("gemma2:27b", "OLLAMA") == "local-large"

    def test_code_specialized(self) -> None:
        assert _classify_model("deepseek-coder:33b", "OLLAMA") == "local-code"

    def test_vision(self) -> None:
        assert _classify_model("llama3.2-vision:latest", "OLLAMA") == "local-vision"

    def test_fallback_no_params(self) -> None:
        assert _classify_model("nomic-embed-text:latest", "OLLAMA") == "local-midrange"


class TestRenderCatalogRow:
    def test_renders_row_with_scores(self) -> None:
        result = _make_result()
        row = _render_catalog_row(result, "reports/test.md", {})
        assert "qwen2.5:32b" in row
        assert "32B" in row
        assert "20" in row  # TPS rounded (20.5 -> 20)
        assert "[→](reports/test.md)" in row

    def test_renders_dash_for_missing_category(self) -> None:
        result = _make_result()
        # Remove document-analysis (not in our test data)
        row = _render_catalog_row(result, "", {})
        assert "—" in row  # document-analysis missing


class TestRenderHardwareEntry:
    def test_renders_entry(self) -> None:
        result = _make_result()
        entry = _render_hardware_entry(result)
        assert "Apple M4 Max" in entry
        assert "48GB" in entry
        assert "qwen2.5:32b" in entry
        assert "2026-02-07" in entry


class TestRenderDecisionEntry:
    def test_renders_entry(self) -> None:
        result = _make_result()
        entry = _render_decision_entry(result)
        assert "qwen2.5:32b" in entry
        assert "32B" in entry
        assert "80/100" in entry
        assert "Good" in entry


# ── Marker Update Tests ────────────────────────────────────────────────────


class TestUpdateMarkerSection:
    def test_replaces_example_rows(self, tmp_path: Path) -> None:
        f = tmp_path / "test.md"
        f.write_text(
            "# Test\n\n"
            "<!-- AI_EVAL:BEGIN test-section -->\n"
            "| _example_ | _data_ |\n"
            "<!-- AI_EVAL:END test-section -->\n"
        )
        _update_marker_section(f, "test-section", "| real | data |")
        content = f.read_text()
        assert "| real | data |" in content
        assert "_example_" not in content

    def test_replaces_same_model(self, tmp_path: Path) -> None:
        f = tmp_path / "test.md"
        f.write_text(
            "# Test\n\n"
            "<!-- AI_EVAL:BEGIN test-section -->\n"
            "| qwen2.5:32b | old data |\n"
            "<!-- AI_EVAL:END test-section -->\n"
        )
        _update_marker_section(f, "test-section", "| qwen2.5:32b | new data |")
        content = f.read_text()
        assert "new data" in content
        assert "old data" not in content

    def test_keeps_different_models(self, tmp_path: Path) -> None:
        f = tmp_path / "test.md"
        f.write_text(
            "# Test\n\n"
            "<!-- AI_EVAL:BEGIN test-section -->\n"
            "| llama3.1:8b | existing |\n"
            "<!-- AI_EVAL:END test-section -->\n"
        )
        _update_marker_section(f, "test-section", "| qwen2.5:32b | new |")
        content = f.read_text()
        assert "llama3.1:8b" in content
        assert "qwen2.5:32b" in content

    def test_adds_header_if_missing(self, tmp_path: Path) -> None:
        f = tmp_path / "test.md"
        f.write_text(
            "# Test\n\n"
            "<!-- AI_EVAL:BEGIN test-section -->\n"
            "_Pending._\n"
            "<!-- AI_EVAL:END test-section -->\n"
        )
        header = "| Model | Score |\n|-------|-------|"
        _update_marker_section(f, "test-section", "| test | 80 |", table_header=header)
        content = f.read_text()
        assert "| Model | Score |" in content
        assert "| test | 80 |" in content

    def test_returns_false_for_missing_markers(self, tmp_path: Path) -> None:
        f = tmp_path / "test.md"
        f.write_text("No markers here.\n")
        assert _update_marker_section(f, "nonexistent", "| data |") is False


# ── Integration Tests ───────────────────────────────────────────────────────


class TestExportToCatalog:
    def test_exports_to_all_files(self, tmp_path: Path) -> None:
        eval_dir = _make_evaluations_dir(tmp_path)
        config = ExportConfig(evaluations_path=eval_dir, copy_reports=False)
        result = _make_result()

        outcomes = export_to_catalog(result, config=config)

        assert outcomes["catalog"] is True
        assert outcomes["matrix"] is True
        assert outcomes["hardware"] is True

    def test_updates_catalog_correct_section(self, tmp_path: Path) -> None:
        eval_dir = _make_evaluations_dir(tmp_path)
        config = ExportConfig(evaluations_path=eval_dir, copy_reports=False)
        result = _make_result("qwen2.5:32b")

        export_to_catalog(result, config=config)

        catalog = (eval_dir / "MODEL_CATALOG.md").read_text()
        # Should be in local-large section, not midrange
        large_start = catalog.find("AI_EVAL:BEGIN local-large")
        large_end = catalog.find("AI_EVAL:END local-large")
        large_section = catalog[large_start:large_end]
        assert "qwen2.5:32b" in large_section

    def test_updates_midrange_for_8b(self, tmp_path: Path) -> None:
        eval_dir = _make_evaluations_dir(tmp_path)
        config = ExportConfig(evaluations_path=eval_dir, copy_reports=False)
        result = _make_result("llama3.1:8b")

        export_to_catalog(result, config=config)

        catalog = (eval_dir / "MODEL_CATALOG.md").read_text()
        mid_start = catalog.find("AI_EVAL:BEGIN local-midrange")
        mid_end = catalog.find("AI_EVAL:END local-midrange")
        mid_section = catalog[mid_start:mid_end]
        assert "llama3.1:8b" in mid_section

    def test_copies_report_files(self, tmp_path: Path) -> None:
        eval_dir = _make_evaluations_dir(tmp_path)
        config = ExportConfig(evaluations_path=eval_dir)
        result = _make_result()

        # Create fake report files
        report = tmp_path / "test_report.md"
        report.write_text("# Report")
        report_json = tmp_path / "test_report.json"
        report_json.write_text('{"test": true}')

        export_to_catalog(result, config=config, report_path=report)

        assert (eval_dir / "reports" / "test_report.md").exists()
        assert (eval_dir / "data" / "test_report.json").exists()

    def test_returns_false_for_missing_eval_dir(self, tmp_path: Path) -> None:
        config = ExportConfig(evaluations_path=tmp_path / "nonexistent")
        result = _make_result()

        outcomes = export_to_catalog(result, config=config)

        assert all(v is False for v in outcomes.values())

    def test_updates_decision_matrix(self, tmp_path: Path) -> None:
        eval_dir = _make_evaluations_dir(tmp_path)
        config = ExportConfig(evaluations_path=eval_dir, copy_reports=False)
        result = _make_result()

        export_to_catalog(result, config=config)

        matrix = (eval_dir / "DECISION_MATRIX.md").read_text()
        assert "qwen2.5:32b" in matrix
        assert "80/100" in matrix

    def test_updates_hardware_profiles(self, tmp_path: Path) -> None:
        eval_dir = _make_evaluations_dir(tmp_path)
        config = ExportConfig(evaluations_path=eval_dir, copy_reports=False)
        result = _make_result()

        export_to_catalog(result, config=config)

        hw = (eval_dir / "HARDWARE_PROFILES.md").read_text()
        assert "Apple M4 Max" in hw
        assert "qwen2.5:32b" in hw

    def test_accumulates_multiple_models(self, tmp_path: Path) -> None:
        eval_dir = _make_evaluations_dir(tmp_path)
        config = ExportConfig(evaluations_path=eval_dir, copy_reports=False)

        r1 = _make_result("qwen2.5:32b", 80.0)
        r2 = _make_result("gemma2:27b", 67.0)

        export_to_catalog(r1, config=config)
        export_to_catalog(r2, config=config)

        catalog = (eval_dir / "MODEL_CATALOG.md").read_text()
        assert "qwen2.5:32b" in catalog
        assert "gemma2:27b" in catalog


# ── Evaluation Log Tests ──────────────────────────────────────────────────


class TestRenderSummaryRow:
    def test_renders_row_with_project(self) -> None:
        result = _make_result()
        row = _render_summary_row(result, "VA_Assistant", "reports/test.md", {})
        assert "qwen2.5:32b" in row
        assert "VA_Assistant" in row
        assert "80" in row
        assert "2026-02-07" in row
        assert "[report](reports/test.md)" in row

    def test_renders_dash_for_no_project(self) -> None:
        result = _make_result()
        row = _render_summary_row(result, "", "", {})
        # No requesting project → "—"
        assert "| — |" in row

    def test_renders_dash_for_no_report(self) -> None:
        result = _make_result()
        row = _render_summary_row(result, "CoreRag", "", {})
        assert "| — |" in row


class TestRenderLogEntry:
    def test_renders_entry_with_all_fields(self) -> None:
        result = _make_result()
        entry = _render_log_entry(
            result, "VA_Assistant", "RAG pipeline evaluation", "reports/test.md", {}
        )
        assert "#### 2026-02-07: qwen2.5:32b" in entry
        assert "VA_Assistant" in entry
        assert "RAG pipeline evaluation" in entry
        assert "80.0/100" in entry
        assert "Good" in entry
        assert "20.5" in entry  # TPS
        assert "Text 94" in entry
        assert "Code 63" in entry
        assert "[Full report](reports/test.md)" in entry

    def test_renders_defaults_for_missing_fields(self) -> None:
        result = _make_result()
        entry = _render_log_entry(result, "", "", "", {})
        assert "AI_Eval" in entry  # default project
        assert "General evaluation" in entry  # default use case


class TestEnsureMonthlyMarker:
    def test_existing_marker_returns_true(self, tmp_path: Path) -> None:
        eval_dir = tmp_path / "evaluations"
        eval_dir.mkdir()
        log = _make_evaluation_log(eval_dir)
        assert _ensure_monthly_marker(log, "2026-02") is True

    def test_creates_new_monthly_section(self, tmp_path: Path) -> None:
        eval_dir = tmp_path / "evaluations"
        eval_dir.mkdir()
        log = _make_evaluation_log(eval_dir)
        assert _ensure_monthly_marker(log, "2026-03") is True

        content = log.read_text()
        assert "<!-- AI_EVAL:BEGIN history-2026-03 -->" in content
        assert "<!-- AI_EVAL:END history-2026-03 -->" in content

    def test_returns_false_without_history_heading(self, tmp_path: Path) -> None:
        f = tmp_path / "no_heading.md"
        f.write_text("# Something Else\n\nNo history here.\n")
        assert _ensure_monthly_marker(f, "2026-02") is False


class TestUpdateLogHistory:
    def test_appends_entry(self, tmp_path: Path) -> None:
        eval_dir = tmp_path / "evaluations"
        eval_dir.mkdir()
        log = _make_evaluation_log(eval_dir)

        entry = "#### 2026-02-07: qwen2.5:32b\n- **Score:** 80/100"
        result = _update_log_history(log, "history-2026-02", entry)

        assert result is True
        content = log.read_text()
        assert "#### 2026-02-07: qwen2.5:32b" in content
        assert "_(No evaluations recorded yet.)_" not in content

    def test_appends_multiple_entries(self, tmp_path: Path) -> None:
        eval_dir = tmp_path / "evaluations"
        eval_dir.mkdir()
        log = _make_evaluation_log(eval_dir)

        entry1 = "#### 2026-02-07: qwen2.5:32b\n- **Score:** 80/100"
        entry2 = "#### 2026-02-08: llama3.1:8b\n- **Score:** 65/100"
        _update_log_history(log, "history-2026-02", entry1)
        _update_log_history(log, "history-2026-02", entry2)

        content = log.read_text()
        assert "qwen2.5:32b" in content
        assert "llama3.1:8b" in content

    def test_returns_false_for_missing_marker(self, tmp_path: Path) -> None:
        f = tmp_path / "empty.md"
        f.write_text("No markers.\n")
        assert _update_log_history(f, "nonexistent", "data") is False


class TestExportToCatalogWithLog:
    def test_exports_to_log(self, tmp_path: Path) -> None:
        eval_dir = _make_evaluations_dir(tmp_path)
        _make_evaluation_log(eval_dir)
        config = ExportConfig(
            evaluations_path=eval_dir,
            copy_reports=False,
            requesting_project="VA_Assistant",
            use_case="RAG pipeline",
        )
        result = _make_result()

        outcomes = export_to_catalog(result, config=config)

        assert outcomes["log"] is True
        log_content = (eval_dir / "EVALUATION_LOG.md").read_text()
        assert "qwen2.5:32b" in log_content
        assert "VA_Assistant" in log_content

    def test_log_summary_replaces_placeholder(self, tmp_path: Path) -> None:
        eval_dir = _make_evaluations_dir(tmp_path)
        _make_evaluation_log(eval_dir)
        config = ExportConfig(evaluations_path=eval_dir, copy_reports=False)
        result = _make_result()

        export_to_catalog(result, config=config)

        log_content = (eval_dir / "EVALUATION_LOG.md").read_text()
        assert "_(no evaluations yet)_" not in log_content

    def test_log_history_replaces_placeholder(self, tmp_path: Path) -> None:
        eval_dir = _make_evaluations_dir(tmp_path)
        _make_evaluation_log(eval_dir)
        config = ExportConfig(evaluations_path=eval_dir, copy_reports=False)
        result = _make_result()

        export_to_catalog(result, config=config)

        log_content = (eval_dir / "EVALUATION_LOG.md").read_text()
        assert "_(No evaluations recorded yet.)_" not in log_content

    def test_re_evaluation_updates_summary(self, tmp_path: Path) -> None:
        eval_dir = _make_evaluations_dir(tmp_path)
        _make_evaluation_log(eval_dir)
        config = ExportConfig(evaluations_path=eval_dir, copy_reports=False)

        r1 = _make_result("qwen2.5:32b", 80.0)
        r2 = _make_result("qwen2.5:32b", 85.0)

        export_to_catalog(r1, config=config)
        export_to_catalog(r2, config=config)

        log_content = (eval_dir / "EVALUATION_LOG.md").read_text()
        # Summary should have latest score only (85, not 80)
        summary_start = log_content.find("AI_EVAL:BEGIN eval-summary")
        summary_end = log_content.find("AI_EVAL:END eval-summary")
        summary = log_content[summary_start:summary_end]
        assert "85" in summary
        # But history should have both entries
        history_start = log_content.find("AI_EVAL:BEGIN history-2026-02")
        history_end = log_content.find("AI_EVAL:END history-2026-02")
        history = log_content[history_start:history_end]
        assert history.count("qwen2.5:32b") == 2

    def test_skip_log_when_disabled(self, tmp_path: Path) -> None:
        eval_dir = _make_evaluations_dir(tmp_path)
        _make_evaluation_log(eval_dir)
        config = ExportConfig(evaluations_path=eval_dir, copy_reports=False, update_log=False)
        result = _make_result()

        outcomes = export_to_catalog(result, config=config)

        assert outcomes["log"] is False
