"""Tests for model discovery and catalog management."""

from pathlib import Path
from unittest.mock import patch

import pytest

from src.evaluation.model_discovery import (
    ModelCatalog,
    ModelEntry,
    _detect_capabilities,
    _extract_param_count,
    _parse_size_to_gb,
    check_for_updates,
    refresh_local_models,
)


class TestModelEntry:
    def test_defaults(self) -> None:
        entry = ModelEntry(name="qwen2.5:32b")
        assert entry.name == "qwen2.5:32b"
        assert entry.size_gb == 0.0
        assert entry.capabilities == []
        assert entry.is_local is False

    def test_full_entry(self) -> None:
        entry = ModelEntry(
            name="qwen2.5:32b",
            size_gb=19.0,
            parameter_count="32B",
            family="qwen2.5",
            capabilities=["text", "code"],
            quantization="Q4_K_M",
            is_local=True,
            last_seen="2026-02-09",
        )
        assert entry.size_gb == 19.0
        assert entry.family == "qwen2.5"
        assert "code" in entry.capabilities


class TestModelCatalog:
    def test_save_and_load(self, tmp_path: Path) -> None:
        catalog = ModelCatalog(
            models=[
                ModelEntry(name="qwen2.5:32b", size_gb=19.0, is_local=True),
                ModelEntry(name="llama3.2:3b", size_gb=2.0, is_local=True),
            ],
            last_refreshed="2026-02-09",
        )
        path = tmp_path / "catalog.json"
        catalog.save(path)

        loaded = ModelCatalog.load(path)
        assert len(loaded.models) == 2
        assert loaded.models[0].name == "qwen2.5:32b"
        assert loaded.models[0].size_gb == 19.0
        assert loaded.last_refreshed == "2026-02-09"

    def test_load_missing_file(self, tmp_path: Path) -> None:
        loaded = ModelCatalog.load(tmp_path / "nonexistent.json")
        assert len(loaded.models) == 0
        assert loaded.last_refreshed == ""

    def test_find(self) -> None:
        catalog = ModelCatalog(
            models=[
                ModelEntry(name="qwen2.5:32b"),
                ModelEntry(name="llama3.2:3b"),
            ]
        )
        found = catalog.find("qwen2.5:32b")
        assert found is not None
        assert found.name == "qwen2.5:32b"

        assert catalog.find("nonexistent") is None

    def test_search_by_capability(self) -> None:
        catalog = ModelCatalog(
            models=[
                ModelEntry(name="qwen2.5:32b", capabilities=["text"]),
                ModelEntry(name="llava:13b", capabilities=["text", "vision"]),
                ModelEntry(name="nomic-embed-text", capabilities=["embedding"]),
            ]
        )
        vision = catalog.search(capability="vision")
        assert len(vision) == 1
        assert vision[0].name == "llava:13b"

    def test_search_by_size(self) -> None:
        catalog = ModelCatalog(
            models=[
                ModelEntry(name="small", size_gb=2.0),
                ModelEntry(name="medium", size_gb=8.0),
                ModelEntry(name="large", size_gb=19.0),
            ]
        )
        results = catalog.search(max_size_gb=10.0)
        assert len(results) == 2
        assert results[0].name == "small"
        assert results[1].name == "medium"

        results = catalog.search(min_size_gb=5.0)
        assert len(results) == 2
        assert results[0].name == "medium"

    def test_search_local_only(self) -> None:
        catalog = ModelCatalog(
            models=[
                ModelEntry(name="local", is_local=True, size_gb=1.0),
                ModelEntry(name="remote", is_local=False, size_gb=2.0),
            ]
        )
        results = catalog.search(local_only=True)
        assert len(results) == 1
        assert results[0].name == "local"

    def test_search_combined_filters(self) -> None:
        catalog = ModelCatalog(
            models=[
                ModelEntry(
                    name="qwen2.5:7b",
                    size_gb=4.0,
                    capabilities=["text", "code"],
                    is_local=True,
                ),
                ModelEntry(
                    name="qwen2.5:32b",
                    size_gb=19.0,
                    capabilities=["text", "code"],
                    is_local=True,
                ),
                ModelEntry(
                    name="llava:13b",
                    size_gb=8.0,
                    capabilities=["text", "vision"],
                    is_local=False,
                ),
            ]
        )
        results = catalog.search(capability="code", max_size_gb=10.0, local_only=True)
        assert len(results) == 1
        assert results[0].name == "qwen2.5:7b"


class TestParseSizeToGb:
    def test_gb(self) -> None:
        assert _parse_size_to_gb("19 GB") == 19.0
        assert _parse_size_to_gb("4.7 GB") == 4.7

    def test_mb(self) -> None:
        result = _parse_size_to_gb("512 MB")
        assert result == pytest.approx(0.5, abs=0.01)

    def test_tb(self) -> None:
        assert _parse_size_to_gb("1 TB") == 1024.0

    def test_no_unit(self) -> None:
        assert _parse_size_to_gb("19") == 19.0

    def test_empty(self) -> None:
        assert _parse_size_to_gb("") == 0.0

    def test_invalid(self) -> None:
        assert _parse_size_to_gb("not a number") == 0.0


class TestDetectCapabilities:
    def test_text_model(self) -> None:
        caps = _detect_capabilities("qwen2.5:32b")
        assert caps == ["text"]

    def test_vision_model(self) -> None:
        caps = _detect_capabilities("llava:13b")
        assert "vision" in caps
        assert "text" in caps

    def test_code_model(self) -> None:
        caps = _detect_capabilities("deepseek-coder:33b")
        assert "code" in caps

    def test_embedding_model(self) -> None:
        caps = _detect_capabilities("nomic-embed-text")
        assert caps == ["embedding"]

    def test_reasoning_model(self) -> None:
        caps = _detect_capabilities("deepseek-r1:32b")
        assert "reasoning" in caps


class TestExtractParamCount:
    def test_with_tag(self) -> None:
        assert _extract_param_count("qwen2.5:32b") == "32B"
        assert _extract_param_count("llama3.2:3b") == "3B"

    def test_with_decimal(self) -> None:
        assert _extract_param_count("model:1.5b") == "1.5B"

    def test_no_tag(self) -> None:
        assert _extract_param_count("nomic-embed-text") == ""

    def test_non_numeric_tag(self) -> None:
        assert _extract_param_count("model:latest") == ""


OLLAMA_LIST_OUTPUT = """NAME                    ID              SIZE    MODIFIED
qwen2.5:32b             abc123          19 GB   2 days ago
llama3.2:3b             def456          2.0 GB  5 days ago
deepseek-r1:32b         ghi789          19 GB   1 day ago
"""


class TestRefreshLocalModels:
    @pytest.mark.asyncio
    async def test_refresh_parses_ollama_output(self, tmp_path: Path) -> None:
        catalog_path = tmp_path / "catalog.json"
        with patch("src.evaluation.model_discovery.subprocess.run") as mock_run:
            mock_run.return_value.returncode = 0
            mock_run.return_value.stdout = OLLAMA_LIST_OUTPUT
            mock_run.return_value.stderr = ""

            catalog = await refresh_local_models(catalog_path)

        assert len(catalog.models) == 3
        assert catalog.find("qwen2.5:32b") is not None
        assert catalog.find("qwen2.5:32b").is_local is True  # type: ignore[union-attr]
        assert catalog.find("llama3.2:3b") is not None
        assert catalog.find("deepseek-r1:32b") is not None

        # Verify saved
        assert catalog_path.exists()

    @pytest.mark.asyncio
    async def test_refresh_updates_existing_entries(self, tmp_path: Path) -> None:
        catalog_path = tmp_path / "catalog.json"

        # Pre-populate with an existing model
        existing = ModelCatalog(
            models=[ModelEntry(name="qwen2.5:32b", size_gb=19.0, is_local=False)]
        )
        existing.save(catalog_path)

        with patch("src.evaluation.model_discovery.subprocess.run") as mock_run:
            mock_run.return_value.returncode = 0
            mock_run.return_value.stdout = OLLAMA_LIST_OUTPUT
            mock_run.return_value.stderr = ""

            catalog = await refresh_local_models(catalog_path)

        # Existing entry should be updated, not duplicated
        qwen_entries = [m for m in catalog.models if m.name == "qwen2.5:32b"]
        assert len(qwen_entries) == 1
        assert qwen_entries[0].is_local is True

        # New entries added
        assert len(catalog.models) == 3

    @pytest.mark.asyncio
    async def test_refresh_handles_ollama_not_found(self, tmp_path: Path) -> None:
        catalog_path = tmp_path / "catalog.json"
        with patch("src.evaluation.model_discovery.subprocess.run") as mock_run:
            mock_run.side_effect = FileNotFoundError()

            catalog = await refresh_local_models(catalog_path)

        assert len(catalog.models) == 0

    @pytest.mark.asyncio
    async def test_refresh_handles_ollama_failure(self, tmp_path: Path) -> None:
        catalog_path = tmp_path / "catalog.json"
        with patch("src.evaluation.model_discovery.subprocess.run") as mock_run:
            mock_run.return_value.returncode = 1
            mock_run.return_value.stderr = "connection refused"

            catalog = await refresh_local_models(catalog_path)

        assert len(catalog.models) == 0


class TestCheckForUpdates:
    def test_finds_alternatives(self) -> None:
        catalog = ModelCatalog(
            models=[
                ModelEntry(name="qwen2.5:7b", family="qwen2.5", is_local=True),
                ModelEntry(name="qwen2.5:32b", family="qwen2.5", is_local=True),
                ModelEntry(name="llama3.2:3b", family="llama3.2", is_local=True),
            ]
        )
        advisories = check_for_updates(catalog, ["qwen2.5:7b"])
        assert len(advisories) == 1
        assert advisories[0]["current"] == "qwen2.5:7b"
        assert "qwen2.5:32b" in advisories[0]["alternatives"]

    def test_no_alternatives(self) -> None:
        catalog = ModelCatalog(
            models=[
                ModelEntry(name="qwen2.5:32b", family="qwen2.5", is_local=True),
                ModelEntry(name="llama3.2:3b", family="llama3.2", is_local=True),
            ]
        )
        advisories = check_for_updates(catalog, ["qwen2.5:32b"])
        assert len(advisories) == 0

    def test_unknown_model(self) -> None:
        catalog = ModelCatalog(models=[])
        advisories = check_for_updates(catalog, ["nonexistent"])
        assert len(advisories) == 0
