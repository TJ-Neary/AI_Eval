"""
Benchmark Runner

Executes evaluation suites against LLM providers and collects metrics.
Supports warmup phases, timeout handling, progress display, and statistical analysis.

Usage:
    from src.benchmarks import BenchmarkRunner
    from src.providers import OllamaProvider

    runner = BenchmarkRunner()
    provider = OllamaProvider(model="qwen2.5:32b")

    results = await runner.run(
        provider=provider,
        dataset=my_dataset,
        config=RunConfig(warmup_queries=3, repetitions=3),
    )
    print(results.summary())
"""

import asyncio
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional

import numpy as np
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from .datasets import Dataset, TestCase, QUICK_TEST_DATASET
from ..providers.base import BaseProvider, GenerationConfig, GenerationResponse
from ..profiling import detect_hardware, get_memory_usage, HardwareProfile
from ..scoring.llm_judge import LLMJudge, JudgingCriteria, JudgingResult
from ..scoring.pass_k import evaluate_code_generation, PassKResult

logger = logging.getLogger(__name__)
console = Console()


class TestStatus(Enum):
    """Status of a test execution."""

    PENDING = auto()
    RUNNING = auto()
    PASSED = auto()
    FAILED = auto()
    TIMEOUT = auto()
    ERROR = auto()
    SKIPPED = auto()


@dataclass
class RunConfig:
    """Configuration for a benchmark run."""

    # Execution
    warmup_queries: int = 3
    repetitions: int = 1  # Times to repeat each test for consistency
    timeout_seconds: float = 120.0
    max_concurrent: int = 1  # Concurrent test executions

    # Generation
    temperature: float = 0.1
    max_tokens: int = 2048
    seed: Optional[int] = None  # For reproducibility

    # Scoring
    use_llm_judge: bool = True
    judge_model: Optional[str] = None  # If None, use same provider
    pass_k_samples: int = 5  # For code generation

    # Output
    save_responses: bool = True
    verbose: bool = True


@dataclass
class TestResult:
    """Result of a single test execution."""

    test_id: str
    status: TestStatus
    prompt: str
    response: str = ""
    expected: Optional[str] = None

    # Timing
    generation_time_ms: float = 0.0
    time_to_first_token_ms: Optional[float] = None
    tokens_per_second: float = 0.0

    # Tokens
    prompt_tokens: int = 0
    completion_tokens: int = 0

    # Scoring
    score: Optional[float] = None  # 0-100
    judge_result: Optional[JudgingResult] = None
    pass_k_result: Optional[PassKResult] = None
    pattern_matches: Dict[str, bool] = field(default_factory=dict)

    # Metadata
    repetition: int = 0
    timestamp: datetime = field(default_factory=datetime.now)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "test_id": self.test_id,
            "status": self.status.name,
            "score": self.score,
            "generation_time_ms": self.generation_time_ms,
            "tokens_per_second": self.tokens_per_second,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "response": self.response[:500] if self.response else "",
            "error": self.error,
        }


@dataclass
class CategoryResult:
    """Aggregated results for a category."""

    category: str
    tests: List[TestResult] = field(default_factory=list)

    @property
    def total_tests(self) -> int:
        return len(self.tests)

    @property
    def passed(self) -> int:
        return len([t for t in self.tests if t.status == TestStatus.PASSED])

    @property
    def failed(self) -> int:
        return len([t for t in self.tests if t.status == TestStatus.FAILED])

    @property
    def pass_rate(self) -> float:
        if not self.tests:
            return 0.0
        return self.passed / self.total_tests

    @property
    def avg_score(self) -> float:
        scores = [t.score for t in self.tests if t.score is not None]
        return np.mean(scores) if scores else 0.0

    @property
    def avg_tokens_per_second(self) -> float:
        tps = [t.tokens_per_second for t in self.tests if t.tokens_per_second > 0]
        return np.mean(tps) if tps else 0.0

    @property
    def avg_generation_time_ms(self) -> float:
        times = [t.generation_time_ms for t in self.tests if t.generation_time_ms > 0]
        return np.mean(times) if times else 0.0


@dataclass
class BenchmarkResult:
    """Complete results from a benchmark run."""

    # Metadata
    model: str
    provider: str
    timestamp: datetime
    duration_seconds: float
    hardware: HardwareProfile

    # Config
    config: RunConfig

    # Results by category
    categories: Dict[str, CategoryResult] = field(default_factory=dict)

    # Aggregate metrics
    total_tests: int = 0
    total_passed: int = 0
    total_tokens: int = 0

    @property
    def overall_score(self) -> float:
        """Weighted average score across categories."""
        # Default weights from TEST_SUITE_SPEC.md
        weights = {
            "text-generation": 0.20,
            "code-generation": 0.25,
            "document-analysis": 0.20,
            "structured-output": 0.15,
            "conversational": 0.20,
        }
        total_weight = 0.0
        weighted_score = 0.0

        for cat_name, cat_result in self.categories.items():
            weight = weights.get(cat_name, 0.20)
            if cat_result.tests:
                weighted_score += cat_result.avg_score * weight
                total_weight += weight

        return weighted_score / total_weight if total_weight > 0 else 0.0

    @property
    def avg_tokens_per_second(self) -> float:
        all_tps = []
        for cat in self.categories.values():
            all_tps.extend(t.tokens_per_second for t in cat.tests if t.tokens_per_second > 0)
        return np.mean(all_tps) if all_tps else 0.0

    def summary(self) -> str:
        """Generate a text summary of results."""
        lines = [
            f"═══ Benchmark Results: {self.model} ═══",
            f"Provider: {self.provider}",
            f"Hardware: {self.hardware.chip_name} ({self.hardware.ram_gb:.0f}GB RAM)",
            f"Duration: {self.duration_seconds:.1f}s",
            f"",
            f"Overall Score: {self.overall_score:.1f}/100",
            f"Avg Throughput: {self.avg_tokens_per_second:.1f} tokens/sec",
            f"Tests: {self.total_passed}/{self.total_tests} passed",
            f"",
            "Category Breakdown:",
        ]

        for cat_name, cat_result in sorted(self.categories.items()):
            lines.append(
                f"  {cat_name}: {cat_result.avg_score:.1f}/100 "
                f"({cat_result.passed}/{cat_result.total_tests} passed, "
                f"{cat_result.avg_tokens_per_second:.1f} t/s)"
            )

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "provider": self.provider,
            "timestamp": self.timestamp.isoformat(),
            "duration_seconds": self.duration_seconds,
            "hardware": self.hardware.to_dict(),
            "overall_score": self.overall_score,
            "avg_tokens_per_second": self.avg_tokens_per_second,
            "total_tests": self.total_tests,
            "total_passed": self.total_passed,
            "categories": {
                name: {
                    "score": cat.avg_score,
                    "pass_rate": cat.pass_rate,
                    "avg_tps": cat.avg_tokens_per_second,
                    "tests": [t.to_dict() for t in cat.tests],
                }
                for name, cat in self.categories.items()
            },
        }


class BenchmarkRunner:
    """
    Executes benchmarks against LLM providers.
    """

    def __init__(self):
        self._judge: Optional[LLMJudge] = None

    async def run(
        self,
        provider: BaseProvider,
        dataset: Optional[Dataset] = None,
        config: Optional[RunConfig] = None,
    ) -> BenchmarkResult:
        """
        Run a benchmark.

        Args:
            provider: LLM provider to benchmark.
            dataset: Dataset to use. Defaults to QUICK_TEST_DATASET.
            config: Run configuration.

        Returns:
            BenchmarkResult with all metrics.
        """
        if dataset is None:
            dataset = QUICK_TEST_DATASET
        if config is None:
            config = RunConfig()

        # Detect hardware
        hardware = detect_hardware()

        # Initialize judge if needed
        if config.use_llm_judge:
            self._judge = LLMJudge(provider=provider)

        start_time = time.perf_counter()
        timestamp = datetime.now()

        # Run warmup
        if config.warmup_queries > 0 and config.verbose:
            console.print(f"[dim]Running {config.warmup_queries} warmup queries...[/dim]")
            await self._run_warmup(provider, config)

        # Execute tests
        results_by_category: Dict[str, List[TestResult]] = {}

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
            disable=not config.verbose,
        ) as progress:
            task = progress.add_task(
                f"[cyan]Benchmarking {provider.model}...",
                total=len(dataset.tests) * config.repetitions,
            )

            for test in dataset.tests:
                category = test.category
                if category not in results_by_category:
                    results_by_category[category] = []

                for rep in range(config.repetitions):
                    result = await self._run_test(provider, test, config, rep)
                    results_by_category[category].append(result)
                    progress.advance(task)

        # Build result
        duration = time.perf_counter() - start_time

        categories = {
            name: CategoryResult(category=name, tests=tests)
            for name, tests in results_by_category.items()
        }

        total_tests = sum(len(c.tests) for c in categories.values())
        total_passed = sum(c.passed for c in categories.values())
        total_tokens = sum(
            t.prompt_tokens + t.completion_tokens
            for c in categories.values()
            for t in c.tests
        )

        result = BenchmarkResult(
            model=provider.model,
            provider=provider.provider_type.name,
            timestamp=timestamp,
            duration_seconds=duration,
            hardware=hardware,
            config=config,
            categories=categories,
            total_tests=total_tests,
            total_passed=total_passed,
            total_tokens=total_tokens,
        )

        if config.verbose:
            console.print(f"\n{result.summary()}")

        return result

    async def _run_warmup(self, provider: BaseProvider, config: RunConfig) -> None:
        """Run warmup queries to prime the model."""
        warmup_prompt = "Hello, how are you today?"
        for _ in range(config.warmup_queries):
            await provider.generate(
                warmup_prompt,
                config=GenerationConfig(
                    temperature=config.temperature,
                    max_tokens=50,
                ),
            )

    async def _run_test(
        self,
        provider: BaseProvider,
        test: TestCase,
        config: RunConfig,
        repetition: int = 0,
    ) -> TestResult:
        """Execute a single test case."""
        result = TestResult(
            test_id=test.id,
            status=TestStatus.RUNNING,
            prompt=test.prompt,
            expected=test.expected,
            repetition=repetition,
        )

        try:
            # Generate response
            gen_config = GenerationConfig(
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                seed=config.seed,
            )

            response = await asyncio.wait_for(
                provider.generate(test.prompt, config=gen_config),
                timeout=config.timeout_seconds,
            )

            if not response.success:
                result.status = TestStatus.ERROR
                result.error = response.error
                return result

            result.response = response.text
            result.generation_time_ms = response.metrics.total_duration_ms
            result.time_to_first_token_ms = response.metrics.time_to_first_token_ms
            result.tokens_per_second = response.metrics.tokens_per_second
            result.prompt_tokens = response.metrics.prompt_tokens
            result.completion_tokens = response.metrics.completion_tokens

            # Score the response
            result.score = await self._score_response(provider, test, response, config)

            # Determine pass/fail
            if result.score is not None and result.score >= 60:
                result.status = TestStatus.PASSED
            else:
                result.status = TestStatus.FAILED

        except asyncio.TimeoutError:
            result.status = TestStatus.TIMEOUT
            result.error = f"Timed out after {config.timeout_seconds}s"

        except Exception as e:
            result.status = TestStatus.ERROR
            result.error = str(e)
            logger.error(f"Test {test.id} failed: {e}")

        return result

    async def _score_response(
        self,
        provider: BaseProvider,
        test: TestCase,
        response: GenerationResponse,
        config: RunConfig,
    ) -> float:
        """Score a test response."""
        scores: List[float] = []

        # Pattern matching
        if test.expected_patterns:
            matches = 0
            for pattern in test.expected_patterns:
                if re.search(pattern, response.text, re.IGNORECASE):
                    matches += 1
            pattern_score = (matches / len(test.expected_patterns)) * 100
            scores.append(pattern_score)

        # Exact match (partial)
        if test.expected:
            if test.expected.lower() in response.text.lower():
                scores.append(100)
            else:
                # Fuzzy partial match
                expected_words = set(test.expected.lower().split())
                response_words = set(response.text.lower().split())
                overlap = len(expected_words & response_words)
                word_score = (overlap / len(expected_words)) * 100 if expected_words else 0
                scores.append(word_score)

        # Code execution (for code tests)
        if test.test_code and test.category == "code-generation":
            try:
                pass_k_result = evaluate_code_generation(
                    problem_id=test.id,
                    code_samples=[response.text],
                    test_code=test.test_code,
                    k_values=[1],
                )
                code_score = pass_k_result.pass_at_k.get(1, 0) * 100
                scores.append(code_score)
            except Exception as e:
                logger.warning(f"Code evaluation failed for {test.id}: {e}")

        # LLM-as-Judge
        if config.use_llm_judge and self._judge:
            try:
                # Pick criteria based on category
                criteria_map = {
                    "text-generation": JudgingCriteria.HELPFULNESS,
                    "code-generation": JudgingCriteria.CODE_QUALITY,
                    "document-analysis": JudgingCriteria.CORRECTNESS,
                    "structured-output": JudgingCriteria.CORRECTNESS,
                    "conversational": JudgingCriteria.HELPFULNESS,
                }
                criteria = criteria_map.get(test.category, JudgingCriteria.HELPFULNESS)

                judge_result = await self._judge.evaluate(
                    question=test.prompt,
                    response=response.text,
                    criteria=criteria,
                    reference_answer=test.expected,
                )
                # Convert 1-10 to 0-100
                judge_score = (judge_result.score - 1) / 9 * 100
                scores.append(judge_score)
            except Exception as e:
                logger.warning(f"LLM judge failed for {test.id}: {e}")

        # Return average of all scores
        return np.mean(scores) if scores else 50.0

    async def quick_test(
        self,
        provider: BaseProvider,
        prompt: str = "What is 2 + 2?",
    ) -> Dict[str, Any]:
        """
        Run a quick single-prompt test.

        Args:
            provider: LLM provider.
            prompt: Test prompt.

        Returns:
            Dict with response and metrics.
        """
        start = time.perf_counter()
        response = await provider.generate(prompt)
        elapsed = time.perf_counter() - start

        return {
            "model": provider.model,
            "prompt": prompt,
            "response": response.text,
            "success": response.success,
            "tokens_per_second": response.metrics.tokens_per_second,
            "generation_time_ms": response.metrics.total_duration_ms,
            "wall_time_seconds": elapsed,
            "prompt_tokens": response.metrics.prompt_tokens,
            "completion_tokens": response.metrics.completion_tokens,
        }
