"""
Pass@k Scoring for Code Generation

Implementation of the pass@k metric used by HumanEval and other code benchmarks.
Measures the probability that at least one of k generated code samples passes all tests.

Based on the methodology from:
- OpenAI's HumanEval paper: "Evaluating Large Language Models Trained on Code"
- BigCode's HumanEvalPack (used by benchllama)

Usage:
    from src.scoring.pass_k import calculate_pass_at_k, evaluate_code_samples

    # Calculate pass@k from raw counts
    pass_1 = calculate_pass_at_k(n=10, c=3, k=1)   # 3/10 passed, what's pass@1?
    pass_5 = calculate_pass_at_k(n=10, c=3, k=5)   # 3/10 passed, what's pass@5?

    # Evaluate multiple samples
    results = [True, False, True, False, False]  # 2/5 passed
    scores = evaluate_code_samples(results, k_values=[1, 5, 10])
"""

import logging
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


def calculate_pass_at_k(n: int, c: int, k: int) -> float:
    """
    Calculate pass@k using the unbiased estimator.

    This is the standard formula from the HumanEval paper:
    pass@k = 1 - C(n-c, k) / C(n, k)

    Where:
    - n = total number of samples generated
    - c = number of samples that passed (correct)
    - k = the k in pass@k

    Args:
        n: Total number of samples.
        c: Number of correct samples.
        k: The k value for pass@k.

    Returns:
        pass@k probability between 0 and 1.

    Example:
        >>> calculate_pass_at_k(n=10, c=3, k=1)
        0.3  # Exactly 3/10
        >>> calculate_pass_at_k(n=10, c=3, k=5)
        0.738...  # Higher because we get 5 tries
    """
    if n < k:
        return 1.0 if c > 0 else 0.0

    if c == 0:
        return 0.0

    if c >= n:
        return 1.0

    # Use log to avoid overflow with large numbers
    # pass@k = 1 - prod_{i=0}^{k-1} (n-c-i) / (n-i)
    result = 1.0
    for i in range(k):
        result *= (n - c - i) / (n - i)

    return 1.0 - result


def calculate_pass_at_k_batch(results: List[List[bool]], k_values: List[int]) -> Dict[int, float]:
    """
    Calculate pass@k for a batch of problems, each with multiple samples.

    Args:
        results: List of problems, each containing a list of pass/fail results.
        k_values: List of k values to compute (e.g., [1, 5, 10]).

    Returns:
        Dict mapping k to average pass@k across all problems.

    Example:
        >>> results = [
        ...     [True, False, True],   # Problem 1: 2/3 passed
        ...     [False, False, False], # Problem 2: 0/3 passed
        ...     [True, True, True],    # Problem 3: 3/3 passed
        ... ]
        >>> calculate_pass_at_k_batch(results, k_values=[1])
        {1: 0.555...}  # Average of (2/3, 0, 1) = 5/9
    """
    scores: Dict[int, List[float]] = {k: [] for k in k_values}

    for problem_results in results:
        n = len(problem_results)
        c = sum(problem_results)

        for k in k_values:
            if k <= n:
                score = calculate_pass_at_k(n, c, k)
                scores[k].append(score)

    return {k: np.mean(v) if v else 0.0 for k, v in scores.items()}


def evaluate_code_samples(
    results: List[bool], k_values: Optional[List[int]] = None
) -> Dict[int, float]:
    """
    Evaluate pass@k for a single problem with multiple samples.

    Args:
        results: List of pass/fail results for each sample.
        k_values: k values to compute. Defaults to [1, 5, 10].

    Returns:
        Dict mapping k to pass@k score.
    """
    if k_values is None:
        k_values = [1, 5, 10]

    n = len(results)
    c = sum(results)

    return {k: calculate_pass_at_k(n, c, k) for k in k_values if k <= n}


@dataclass
class CodeExecutionResult:
    """Result of executing a code sample."""

    passed: bool
    output: str = ""
    error: str = ""
    execution_time_ms: float = 0.0
    timed_out: bool = False


@dataclass
class CodeEvaluationConfig:
    """Configuration for code execution and evaluation."""

    timeout_seconds: float = 10.0
    language: str = "python"
    test_framework: str = "pytest"  # or "unittest", "doctest", "assert"
    sandbox: bool = True  # Use restricted execution
    max_output_length: int = 10000


def execute_python_code(
    code: str,
    test_code: str,
    config: Optional[CodeEvaluationConfig] = None,
) -> CodeExecutionResult:
    """
    Execute Python code with test assertions.

    Args:
        code: The generated code to evaluate.
        test_code: Test code with assertions to run against the generated code.
        config: Execution configuration.

    Returns:
        CodeExecutionResult with pass/fail status.
    """
    cfg = config or CodeEvaluationConfig()

    # Combine code and tests
    full_code = f"{code}\n\n{test_code}"

    with tempfile.TemporaryDirectory() as tmpdir:
        code_file = Path(tmpdir) / "solution.py"
        code_file.write_text(full_code)

        try:
            import time

            start = time.perf_counter()

            result = subprocess.run(
                ["python", str(code_file)],
                capture_output=True,
                text=True,
                timeout=cfg.timeout_seconds,
                cwd=tmpdir,
            )

            elapsed_ms = (time.perf_counter() - start) * 1000

            passed = result.returncode == 0
            output = result.stdout[: cfg.max_output_length]
            error = result.stderr[: cfg.max_output_length]

            return CodeExecutionResult(
                passed=passed,
                output=output,
                error=error,
                execution_time_ms=elapsed_ms,
            )

        except subprocess.TimeoutExpired:
            return CodeExecutionResult(
                passed=False,
                error=f"Execution timed out after {cfg.timeout_seconds}s",
                timed_out=True,
            )
        except Exception as e:
            return CodeExecutionResult(
                passed=False,
                error=str(e),
            )


@dataclass
class PassKResult:
    """Complete pass@k evaluation result."""

    problem_id: str
    num_samples: int
    num_passed: int
    pass_at_k: Dict[int, float]
    execution_results: List[CodeExecutionResult] = field(default_factory=list)

    @property
    def pass_rate(self) -> float:
        """Raw pass rate (passed / total)."""
        return self.num_passed / self.num_samples if self.num_samples > 0 else 0.0


def evaluate_code_generation(
    problem_id: str,
    code_samples: List[str],
    test_code: str,
    k_values: Optional[List[int]] = None,
    config: Optional[CodeEvaluationConfig] = None,
) -> PassKResult:
    """
    Full pass@k evaluation for a code generation problem.

    Args:
        problem_id: Identifier for the problem.
        code_samples: List of generated code samples to evaluate.
        test_code: Test code with assertions.
        k_values: k values for pass@k. Defaults to [1, 5, 10].
        config: Execution configuration.

    Returns:
        PassKResult with scores and execution details.
    """
    if k_values is None:
        k_values = [1, 5, 10]

    execution_results: List[CodeExecutionResult] = []

    for code in code_samples:
        result = execute_python_code(code, test_code, config)
        execution_results.append(result)

    passed_flags = [r.passed for r in execution_results]
    num_passed = sum(passed_flags)

    pass_at_k = evaluate_code_samples(passed_flags, k_values)

    return PassKResult(
        problem_id=problem_id,
        num_samples=len(code_samples),
        num_passed=num_passed,
        pass_at_k=pass_at_k,
        execution_results=execution_results,
    )
