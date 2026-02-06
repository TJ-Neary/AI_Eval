"""
Example test file demonstrating testing patterns for AI_Eval.

Delete or rename this file once real tests are in place.
"""

import pytest
from pathlib import Path


class TestProjectSetup:
    """Verify the project is configured correctly."""

    def test_project_directory_exists(self, tmp_project_dir):
        """Temporary project dir is created by fixture."""
        assert tmp_project_dir.exists()
        assert (tmp_project_dir / "data").is_dir()

    def test_sample_file_created(self, sample_file):
        """Sample file fixture creates a readable file."""
        assert sample_file.exists()
        content = sample_file.read_text()
        assert "sample document" in content

    def test_environment_configured(self, mock_env):
        """Mock environment variables are set."""
        import os
        assert os.getenv("AI_EVAL_STATE_DIR") is not None
        assert os.getenv("LOG_LEVEL") == "DEBUG"


class TestExampleUnit:
    """Example unit tests — replace with real tests."""

    def test_addition(self):
        assert 1 + 1 == 2

    @pytest.mark.parametrize("input_val,expected", [
        ("hello", 5),
        ("", 0),
        ("world!", 6),
    ])
    def test_string_length(self, input_val, expected):
        assert len(input_val) == expected

    def test_path_operations(self, tmp_path):
        """Demonstrate using tmp_path for file-based tests."""
        test_file = tmp_path / "output.txt"
        test_file.write_text("test output")
        assert test_file.read_text() == "test output"