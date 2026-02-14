"""Tests for evaluation request configuration loading and conversion."""

from pathlib import Path

import pytest
import yaml

from src.benchmarks.datasets import Dataset, TestCase
from src.benchmarks.runner import RunConfig
from src.evaluation.config import (
    AcceptanceCriterion,
    EvalRequestConfig,
    Scenario,
)


def _make_yaml_data() -> dict:
    """Create minimal valid YAML data for testing."""
    return {
        "request_id": "test-eval-001",
        "requesting_project": "TestProject",
        "date": "2026-02-09",
        "use_case": "Test evaluation",
        "model_capability": "text-generation",
        "task": {
            "description": "Test task",
            "input": "text input",
            "output": "JSON output",
            "complexity": "MEDIUM",
        },
        "candidates": [
            {"name": "qwen2.5:32b", "provider": "ollama", "notes": "test"},
            {"name": "llama3:8b", "provider": "ollama"},
        ],
        "acceptance_criteria": [
            {
                "name": "JSON validity",
                "metric": "json_validity",
                "threshold": 0.90,
                "operator": ">=",
            },
            {
                "name": "Latency",
                "metric": "latency",
                "threshold": 15.0,
                "operator": "<=",
                "unit": "seconds",
            },
        ],
        "custom_scorers": ["json_validity", "latency"],
        "scorer_config": {"json_validity": {"required_fields": ["claims"]}},
        "run_config": {
            "temperature": 0.2,
            "max_tokens": 4096,
            "timeout_seconds": 30.0,
            "repetitions": 2,
        },
        "scenarios": [
            {
                "id": "scenario-001",
                "description": "Basic test",
                "prompt": "Answer this: What is 2+2?",
                "expected_patterns": [r"\b4\b"],
                "tags": ["basic"],
            },
        ],
    }


class TestAcceptanceCriterion:
    def test_evaluate_gte(self) -> None:
        c = AcceptanceCriterion(name="test", metric="m", threshold=0.9, operator=">=")
        assert c.evaluate(0.95) is True
        assert c.evaluate(0.9) is True
        assert c.evaluate(0.89) is False

    def test_evaluate_lte(self) -> None:
        c = AcceptanceCriterion(name="test", metric="m", threshold=15.0, operator="<=")
        assert c.evaluate(10.0) is True
        assert c.evaluate(15.0) is True
        assert c.evaluate(15.1) is False

    def test_evaluate_gt(self) -> None:
        c = AcceptanceCriterion(name="test", metric="m", threshold=0.5, operator=">")
        assert c.evaluate(0.6) is True
        assert c.evaluate(0.5) is False

    def test_evaluate_lt(self) -> None:
        c = AcceptanceCriterion(name="test", metric="m", threshold=10.0, operator="<")
        assert c.evaluate(9.9) is True
        assert c.evaluate(10.0) is False

    def test_evaluate_eq(self) -> None:
        c = AcceptanceCriterion(name="test", metric="m", threshold=1.0, operator="==")
        assert c.evaluate(1.0) is True
        assert c.evaluate(0.9) is False

    def test_evaluate_unknown_operator_raises(self) -> None:
        c = AcceptanceCriterion(name="test", metric="m", threshold=1.0, operator="!!")
        with pytest.raises(ValueError, match="Unknown operator"):
            c.evaluate(1.0)


class TestScenario:
    def test_to_test_case(self) -> None:
        s = Scenario(
            id="s1",
            description="test",
            prompt="What is 2+2?",
            expected_patterns=[r"\b4\b"],
            tags=["math"],
            metadata={"key": "val"},
        )
        tc = s.to_test_case(category="text-generation")

        assert isinstance(tc, TestCase)
        assert tc.id == "s1"
        assert tc.prompt == "What is 2+2?"
        assert tc.category == "text-generation"
        assert tc.subcategory == "evaluation"
        assert tc.expected_patterns == [r"\b4\b"]
        assert tc.tags == ["math"]
        assert tc.metadata == {"key": "val"}

    def test_to_test_case_minimal(self) -> None:
        s = Scenario(id="s2", description="minimal", prompt="Hello")
        tc = s.to_test_case(category="conversational")
        assert tc.id == "s2"
        assert tc.expected is None
        assert tc.expected_patterns == []


class TestEvalRequestConfigFromYaml:
    def test_loads_from_yaml(self, tmp_path: Path) -> None:
        data = _make_yaml_data()
        config_file = tmp_path / "test.yaml"
        config_file.write_text(yaml.dump(data))

        config = EvalRequestConfig.from_yaml(config_file)

        assert config.request_id == "test-eval-001"
        assert config.requesting_project == "TestProject"
        assert config.use_case == "Test evaluation"
        assert config.model_capability == "text-generation"
        assert config.task_description == "Test task"
        assert len(config.candidates) == 2
        assert config.candidates[0].name == "qwen2.5:32b"
        assert config.candidates[1].provider == "ollama"
        assert len(config.acceptance_criteria) == 2
        assert config.acceptance_criteria[0].threshold == 0.90
        assert config.acceptance_criteria[1].operator == "<="
        assert config.custom_scorers == ["json_validity", "latency"]
        assert config.temperature == 0.2
        assert config.max_tokens == 4096
        assert config.repetitions == 2

    def test_resolves_input_file(self, tmp_path: Path) -> None:
        # Create input file
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        input_file = data_dir / "evidence.txt"
        input_file.write_text("Patient shows left knee ROM of 90 degrees.")

        data = _make_yaml_data()
        data["scenarios"] = [
            {
                "id": "file-test",
                "description": "test with file input",
                "prompt": "Evidence: {input}\nQuestion: What is the ROM?",
                "input_file": "data/evidence.txt",
                "expected_patterns": [r"90"],
            },
        ]
        config_file = tmp_path / "test.yaml"
        config_file.write_text(yaml.dump(data))

        config = EvalRequestConfig.from_yaml(config_file)

        assert len(config.scenarios) == 1
        assert "Patient shows left knee ROM of 90 degrees." in config.scenarios[0].prompt
        assert "{input}" not in config.scenarios[0].prompt
        assert (
            config.scenarios[0].metadata["source_text"]
            == "Patient shows left knee ROM of 90 degrees."
        )

    def test_missing_input_file_raises(self, tmp_path: Path) -> None:
        data = _make_yaml_data()
        data["scenarios"] = [
            {
                "id": "missing-file",
                "description": "test",
                "prompt": "Input: {input}",
                "input_file": "nonexistent.txt",
            },
        ]
        config_file = tmp_path / "test.yaml"
        config_file.write_text(yaml.dump(data))

        with pytest.raises(FileNotFoundError, match="nonexistent.txt"):
            EvalRequestConfig.from_yaml(config_file)

    def test_missing_config_file_raises(self) -> None:
        with pytest.raises(FileNotFoundError, match="Config file not found"):
            EvalRequestConfig.from_yaml(Path("/nonexistent/path.yaml"))

    def test_defaults_for_optional_fields(self, tmp_path: Path) -> None:
        """Minimal YAML with only required fields."""
        data = {
            "request_id": "minimal",
            "requesting_project": "Test",
            "use_case": "Test",
        }
        config_file = tmp_path / "minimal.yaml"
        config_file.write_text(yaml.dump(data))

        config = EvalRequestConfig.from_yaml(config_file)
        assert config.model_capability == "text-generation"
        assert config.temperature == 0.1
        assert config.candidates == []
        assert config.scenarios == []


class TestEvalRequestConfigConversions:
    def test_to_dataset(self, tmp_path: Path) -> None:
        data = _make_yaml_data()
        config_file = tmp_path / "test.yaml"
        config_file.write_text(yaml.dump(data))

        config = EvalRequestConfig.from_yaml(config_file)
        dataset = config.to_dataset()

        assert isinstance(dataset, Dataset)
        assert dataset.name == "test-eval-001"
        assert dataset.category == "text-generation"
        assert len(dataset.tests) == 1
        assert dataset.tests[0].id == "scenario-001"
        assert dataset.tests[0].category == "text-generation"

    def test_to_run_config(self, tmp_path: Path) -> None:
        data = _make_yaml_data()
        config_file = tmp_path / "test.yaml"
        config_file.write_text(yaml.dump(data))

        config = EvalRequestConfig.from_yaml(config_file)
        run_config = config.to_run_config()

        assert isinstance(run_config, RunConfig)
        assert run_config.temperature == 0.2
        assert run_config.max_tokens == 4096
        assert run_config.timeout_seconds == 30.0
        assert run_config.repetitions == 2
        assert run_config.use_llm_judge is True
