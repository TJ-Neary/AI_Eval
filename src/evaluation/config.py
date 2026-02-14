"""
Evaluation Request Configuration

Dataclasses for structured evaluation requests and YAML loading.
Converts request configs into BenchmarkRunner-compatible Dataset + RunConfig.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from ..benchmarks.datasets import Dataset, TestCase
from ..benchmarks.runner import RunConfig

logger = logging.getLogger(__name__)


@dataclass
class AcceptanceCriterion:
    """A single pass/fail acceptance criterion for an evaluation."""

    name: str  # Human-readable: "JSON validity rate"
    metric: str  # Scorer name: "json_validity"
    threshold: float  # e.g., 0.90 for >=90%
    operator: str = ">="  # ">=", "<=", "==", ">"
    unit: str = ""  # "rate", "seconds", "per_minute"
    notes: str = ""

    def evaluate(self, measured: float) -> bool:
        """Check if measured value meets this criterion."""
        if self.operator == ">=":
            return measured >= self.threshold
        elif self.operator == "<=":
            return measured <= self.threshold
        elif self.operator == ">":
            return measured > self.threshold
        elif self.operator == "<":
            return measured < self.threshold
        elif self.operator == "==":
            return measured == self.threshold
        else:
            raise ValueError(f"Unknown operator: {self.operator}")


@dataclass
class Scenario:
    """A single test scenario within an evaluation request."""

    id: str
    description: str
    prompt: str  # May contain {input} placeholder
    input_file: Optional[str] = None  # Path relative to config YAML
    expected: Optional[str] = None
    expected_patterns: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)

    def to_test_case(self, category: str) -> TestCase:
        """Convert to a BenchmarkRunner TestCase."""
        return TestCase(
            id=self.id,
            prompt=self.prompt,
            category=category,
            subcategory="evaluation",
            expected=self.expected,
            expected_patterns=self.expected_patterns,
            tags=self.tags,
            metadata=self.metadata,
        )


@dataclass
class CandidateModel:
    """A model to evaluate."""

    name: str
    provider: str = "ollama"
    notes: str = ""


@dataclass
class EvalRequestConfig:
    """Complete evaluation request configuration, loaded from YAML."""

    # Metadata
    request_id: str
    requesting_project: str
    date: str
    use_case: str
    model_capability: str  # text-generation, vision-ocr, thinking-reasoning, etc.

    # Task
    task_description: str
    input_description: str = ""
    output_description: str = ""
    complexity: str = "MEDIUM"
    current_model: str = ""

    # Models
    candidates: List[CandidateModel] = field(default_factory=list)

    # Acceptance criteria
    acceptance_criteria: List[AcceptanceCriterion] = field(default_factory=list)

    # Test scenarios
    scenarios: List[Scenario] = field(default_factory=list)

    # Scorers
    custom_scorers: List[str] = field(default_factory=list)
    scorer_config: Dict[str, Any] = field(default_factory=dict)

    # RunConfig overrides
    temperature: float = 0.1
    max_tokens: int = 2048
    timeout_seconds: float = 120.0
    warmup_queries: int = 3
    repetitions: int = 3
    use_llm_judge: bool = True

    # Hardware constraints
    max_ram_gb: Optional[float] = None
    min_tokens_per_second: Optional[float] = None

    @classmethod
    def from_yaml(cls, path: Path) -> "EvalRequestConfig":
        """Load evaluation request from YAML file.

        Resolves input_file references relative to the YAML file's directory.
        Substitutes {input} placeholders in prompts with file contents.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path) as f:
            data = yaml.safe_load(f)

        config_dir = path.parent

        # Parse task section
        task = data.get("task", {})

        # Parse candidates
        candidates = [
            CandidateModel(
                name=c["name"],
                provider=c.get("provider", "ollama"),
                notes=c.get("notes", ""),
            )
            for c in data.get("candidates", [])
        ]

        # Parse acceptance criteria
        criteria = [
            AcceptanceCriterion(
                name=c["name"],
                metric=c["metric"],
                threshold=float(c["threshold"]),
                operator=c.get("operator", ">="),
                unit=c.get("unit", ""),
                notes=c.get("notes", ""),
            )
            for c in data.get("acceptance_criteria", [])
        ]

        # Parse scenarios with input_file resolution
        scenarios = []
        for s in data.get("scenarios", []):
            prompt = s["prompt"]
            metadata = dict(s.get("metadata", {}))
            input_file = s.get("input_file")

            if input_file:
                input_path = config_dir / input_file
                if not input_path.exists():
                    raise FileNotFoundError(
                        f"Scenario input file not found: {input_path} "
                        f"(referenced in scenario '{s['id']}')"
                    )
                source_text = input_path.read_text()
                prompt = prompt.replace("{input}", source_text)
                metadata["source_text"] = source_text
                metadata["input_file"] = str(input_file)

            scenarios.append(
                Scenario(
                    id=s["id"],
                    description=s.get("description", ""),
                    prompt=prompt,
                    input_file=input_file,
                    expected=s.get("expected"),
                    expected_patterns=s.get("expected_patterns", []),
                    metadata=metadata,
                    tags=s.get("tags", []),
                )
            )

        # Parse run_config overrides
        run_cfg = data.get("run_config", {})

        # Parse hardware constraints
        hardware = data.get("hardware_constraints", {})

        return cls(
            request_id=data["request_id"],
            requesting_project=data["requesting_project"],
            date=data.get("date", ""),
            use_case=data["use_case"],
            model_capability=data.get("model_capability", "text-generation"),
            task_description=task.get("description", ""),
            input_description=task.get("input", ""),
            output_description=task.get("output", ""),
            complexity=task.get("complexity", "MEDIUM"),
            current_model=task.get("current_model", ""),
            candidates=candidates,
            acceptance_criteria=criteria,
            scenarios=scenarios,
            custom_scorers=data.get("custom_scorers", []),
            scorer_config=data.get("scorer_config", {}),
            temperature=run_cfg.get("temperature", 0.1),
            max_tokens=run_cfg.get("max_tokens", 2048),
            timeout_seconds=run_cfg.get("timeout_seconds", 120.0),
            warmup_queries=run_cfg.get("warmup_queries", 3),
            repetitions=run_cfg.get("repetitions", 3),
            use_llm_judge=run_cfg.get("use_llm_judge", True),
            max_ram_gb=hardware.get("max_ram_gb"),
            min_tokens_per_second=hardware.get("min_tokens_per_second"),
        )

    def to_dataset(self) -> Dataset:
        """Convert scenarios to a BenchmarkRunner Dataset."""
        tests = [s.to_test_case(category=self.model_capability) for s in self.scenarios]

        return Dataset(
            name=self.request_id,
            version="1.0",
            category=self.model_capability,
            description=self.use_case,
            tests=tests,
            metadata={
                "requesting_project": self.requesting_project,
                "custom_scorers": self.custom_scorers,
            },
        )

    def to_run_config(self) -> RunConfig:
        """Build a RunConfig from request overrides."""
        return RunConfig(
            warmup_queries=self.warmup_queries,
            repetitions=self.repetitions,
            timeout_seconds=self.timeout_seconds,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            use_llm_judge=self.use_llm_judge,
            verbose=True,
        )
