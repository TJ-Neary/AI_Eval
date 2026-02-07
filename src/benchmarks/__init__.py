"""
Benchmark Module

Test suites and dataset management for LLM evaluation.

Usage:
    from src.benchmarks import BenchmarkRunner, DatasetManager, QUICK_TEST_DATASET
    from src.providers import OllamaProvider

    runner = BenchmarkRunner()
    provider = OllamaProvider(model="qwen2.5:32b")

    # Quick test
    result = await runner.quick_test(provider)
    print(f"Response: {result['response']}")
    print(f"Speed: {result['tokens_per_second']:.1f} t/s")

    # Full benchmark
    results = await runner.run(provider)
    print(results.summary())
"""

from .datasets import QUICK_TEST_DATASET, BenchmarkSuite, Dataset, DatasetManager, TestCase
from .runner import (
    BenchmarkResult,
    BenchmarkRunner,
    CategoryResult,
    RunConfig,
    TestResult,
    TestStatus,
)

__all__ = [
    # Datasets
    "Dataset",
    "TestCase",
    "BenchmarkSuite",
    "DatasetManager",
    "QUICK_TEST_DATASET",
    # Runner
    "BenchmarkRunner",
    "BenchmarkResult",
    "CategoryResult",
    "TestResult",
    "TestStatus",
    "RunConfig",
]
