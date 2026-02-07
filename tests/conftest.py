"""
Shared test fixtures for AI_Eval.

Provides common setup: temporary directories, mock configurations,
and test data factories.
"""

from pathlib import Path

import pytest


@pytest.fixture
def tmp_project_dir(tmp_path: Path) -> Path:
    """Create a temporary project directory with standard structure."""
    dirs = ["data", "logs", "config"]
    for d in dirs:
        (tmp_path / d).mkdir()
    return tmp_path


@pytest.fixture
def mock_env(monkeypatch: pytest.MonkeyPatch, tmp_project_dir: Path) -> Path:
    """Set environment variables pointing to temporary directories."""
    monkeypatch.setenv("AI_EVAL_STATE_DIR", str(tmp_project_dir))
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("DEBUG", "true")
    return tmp_project_dir


@pytest.fixture
def sample_text() -> str:
    """Return synthetic sample text for testing."""
    return (
        "This is a sample document for testing purposes. "
        "It contains no real personal information. "
        "John Doe lives at 123 Main Street, Anytown, ST 12345. "
        "Contact: john.doe@example.com or (555) 555-0100."
    )


@pytest.fixture
def sample_file(tmp_path: Path, sample_text: str) -> Path:
    """Create a temporary text file with sample content."""
    f = tmp_path / "sample.txt"
    f.write_text(sample_text)
    return f
