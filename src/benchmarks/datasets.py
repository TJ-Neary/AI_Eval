"""
Benchmark Dataset Management

Loads and manages test prompts and expected outputs for evaluation.
Supports YAML-based dataset definitions with versioning.

Dataset structure:
    datasets/
    ├── text-generation/
    │   ├── instruction-following.yaml
    │   └── summarization.yaml
    ├── code-generation/
    │   ├── python.yaml
    │   └── sql.yaml
    └── manifest.yaml

Usage:
    from src.benchmarks.datasets import DatasetManager

    manager = DatasetManager()
    dataset = manager.load("text-generation/instruction-following")
    for test in dataset.tests:
        print(test.prompt, test.expected)
"""

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)


@dataclass
class TestCase:
    """A single test case in a dataset."""

    id: str
    prompt: str
    category: str
    subcategory: str = ""
    expected: Optional[str] = None
    expected_patterns: List[str] = field(default_factory=list)  # Regex patterns
    scoring_rubric: str = ""
    difficulty: str = "medium"  # easy, medium, hard
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # For code tests
    test_code: Optional[str] = None  # Assertions to run
    language: str = "python"

    # For multi-turn
    conversation: List[Dict[str, str]] = field(default_factory=list)


@dataclass
class Dataset:
    """A collection of test cases."""

    name: str
    version: str
    category: str
    description: str = ""
    tests: List[TestCase] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def content_hash(self) -> str:
        """SHA256 hash of dataset contents for versioning."""
        content = f"{self.name}:{self.version}:{len(self.tests)}"
        for test in self.tests:
            content += f":{test.id}:{test.prompt[:50]}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def filter_by_difficulty(self, difficulty: str) -> "Dataset":
        """Return a new dataset with only tests of given difficulty."""
        filtered = Dataset(
            name=self.name,
            version=self.version,
            category=self.category,
            description=self.description,
            tests=[t for t in self.tests if t.difficulty == difficulty],
            metadata=self.metadata,
        )
        return filtered

    def filter_by_tags(self, tags: List[str]) -> "Dataset":
        """Return a new dataset with tests matching any of the tags."""
        tag_set = set(tags)
        filtered = Dataset(
            name=self.name,
            version=self.version,
            category=self.category,
            description=self.description,
            tests=[t for t in self.tests if tag_set & set(t.tags)],
            metadata=self.metadata,
        )
        return filtered


@dataclass
class BenchmarkSuite:
    """A complete benchmark suite with multiple datasets."""

    name: str
    version: str
    description: str = ""
    datasets: List[Dataset] = field(default_factory=list)
    config: Dict[str, Any] = field(default_factory=dict)

    @property
    def total_tests(self) -> int:
        return sum(len(d.tests) for d in self.datasets)

    @property
    def categories(self) -> List[str]:
        return list(set(d.category for d in self.datasets))


class DatasetManager:
    """
    Manages loading and caching of benchmark datasets.
    """

    def __init__(self, datasets_dir: Optional[Path] = None):
        """
        Args:
            datasets_dir: Directory containing dataset YAML files.
                         Defaults to src/benchmarks/datasets/
        """
        if datasets_dir is None:
            datasets_dir = Path(__file__).parent / "datasets"
        self.datasets_dir = Path(datasets_dir)
        self._cache: Dict[str, Dataset] = {}

    def load(self, name: str) -> Dataset:
        """
        Load a dataset by name.

        Args:
            name: Dataset name (e.g., "text-generation/instruction-following")

        Returns:
            Loaded Dataset object.
        """
        if name in self._cache:
            return self._cache[name]

        # Try with and without .yaml extension
        path = self.datasets_dir / f"{name}.yaml"
        if not path.exists():
            path = self.datasets_dir / name
            if not path.exists():
                raise FileNotFoundError(f"Dataset not found: {name}")

        dataset = self._load_yaml(path)
        self._cache[name] = dataset
        return dataset

    def _load_yaml(self, path: Path) -> Dataset:
        """Load dataset from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)

        tests: list[TestCase] = []
        for test_data in data.get("tests", []):
            test = TestCase(
                id=test_data.get("id", f"test_{len(tests)}"),
                prompt=test_data["prompt"],
                category=data.get("category", "unknown"),
                subcategory=test_data.get("subcategory", ""),
                expected=test_data.get("expected"),
                expected_patterns=test_data.get("expected_patterns", []),
                scoring_rubric=test_data.get("scoring_rubric", ""),
                difficulty=test_data.get("difficulty", "medium"),
                tags=test_data.get("tags", []),
                metadata=test_data.get("metadata", {}),
                test_code=test_data.get("test_code"),
                language=test_data.get("language", "python"),
                conversation=test_data.get("conversation", []),
            )
            tests.append(test)

        return Dataset(
            name=data.get("name", path.stem),
            version=data.get("version", "1.0"),
            category=data.get("category", "unknown"),
            description=data.get("description", ""),
            tests=tests,
            metadata=data.get("metadata", {}),
        )

    def list_datasets(self) -> List[str]:
        """List all available datasets."""
        if not self.datasets_dir.exists():
            return []

        datasets = []
        for path in self.datasets_dir.rglob("*.yaml"):
            if path.name != "manifest.yaml":
                rel_path = path.relative_to(self.datasets_dir)
                name = str(rel_path).replace(".yaml", "")
                datasets.append(name)
        return sorted(datasets)

    def load_suite(self, config_path: Path) -> BenchmarkSuite:
        """
        Load a complete benchmark suite from config.

        Args:
            config_path: Path to suite configuration YAML.

        Returns:
            BenchmarkSuite with all referenced datasets.
        """
        with open(config_path) as f:
            config = yaml.safe_load(f)

        datasets = []
        for category in config.get("categories", []):
            # Load all datasets in category
            category_dir = self.datasets_dir / category
            if category_dir.is_dir():
                for yaml_file in category_dir.glob("*.yaml"):
                    dataset = self._load_yaml(yaml_file)
                    datasets.append(dataset)

        return BenchmarkSuite(
            name=config.get("name", config_path.stem),
            version=config.get("version", "1.0"),
            description=config.get("description", ""),
            datasets=datasets,
            config=config,
        )


# Built-in minimal dataset for quick testing
QUICK_TEST_DATASET = Dataset(
    name="quick-test",
    version="1.0",
    category="mixed",
    description="Minimal dataset for quick sanity checks",
    tests=[
        TestCase(
            id="text_simple",
            prompt="What is the capital of France?",
            category="text-generation",
            subcategory="factual",
            expected="Paris",
            difficulty="easy",
        ),
        TestCase(
            id="code_simple",
            prompt="Write a Python function that returns the sum of two numbers.",
            category="code-generation",
            subcategory="python",
            expected_patterns=[r"def\s+\w+\s*\(", r"return"],
            test_code="assert add(2, 3) == 5\nassert add(-1, 1) == 0",
            difficulty="easy",
        ),
        TestCase(
            id="reasoning_simple",
            prompt="If all roses are flowers and some flowers fade quickly, can we conclude that some roses fade quickly?",
            category="text-generation",
            subcategory="reasoning",
            expected_patterns=[r"(no|cannot|can't)", r"(some|all)"],
            difficulty="medium",
        ),
        TestCase(
            id="structured_json",
            prompt="Convert this to JSON: Name: John, Age: 30, City: New York",
            category="structured-output",
            subcategory="json",
            expected_patterns=[r'\{.*"name".*:.*"John"', r'"age".*:.*30'],
            difficulty="easy",
        ),
        TestCase(
            id="chat_greeting",
            prompt="Hello! How are you today?",
            category="conversational",
            subcategory="greeting",
            expected_patterns=[r"(hello|hi|hey|good|well|great|fine)"],
            difficulty="easy",
        ),
    ],
)
