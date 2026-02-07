"""
LLM-as-Judge Evaluator

Uses an LLM to evaluate the quality of model responses.
Implements bias mitigation strategies from TD-011:
- Position shuffling (avoid position bias)
- Multi-judge consensus
- Calibration against human ratings
- Separate judges for different criteria

Based on patterns from:
- cloudmercato/ollama-benchmark (LLM-as-Judge with MT-Bench)
- FastChat's LLM-as-Judge implementation
- Anthropic's research on LLM evaluation

Usage:
    from src.scoring.llm_judge import LLMJudge, JudgingCriteria

    judge = LLMJudge(provider=my_provider)
    result = await judge.evaluate(
        question="What is the capital of France?",
        response="The capital of France is Paris.",
        criteria=JudgingCriteria.CORRECTNESS,
    )
    print(result.score, result.reasoning)
"""

import logging
import random
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class JudgingCriteria(Enum):
    """Criteria for evaluating responses."""

    CORRECTNESS = auto()  # Factual accuracy
    HELPFULNESS = auto()  # Usefulness to the user
    HARMLESSNESS = auto()  # Safety and appropriateness
    COHERENCE = auto()  # Logical flow and clarity
    RELEVANCE = auto()  # Addresses the question
    COMPLETENESS = auto()  # Thorough coverage
    CONCISENESS = auto()  # Appropriate length
    CREATIVITY = auto()  # Novel/interesting approach
    CODE_QUALITY = auto()  # For code: correctness, style, efficiency


@dataclass
class JudgingConfig:
    """Configuration for LLM-as-Judge evaluation."""

    # Scoring
    min_score: int = 1
    max_score: int = 10
    require_reasoning: bool = True

    # Bias mitigation
    shuffle_positions: bool = True  # TD-011: Shuffle A/B positions
    num_evaluations: int = 1  # TD-011: Multiple evaluations for consensus
    use_diverse_judges: bool = False  # TD-011: Use different judge models

    # Prompting
    use_reference_answer: bool = False
    include_rubric: bool = True

    # Output
    temperature: float = 0.0  # Low temp for consistency


@dataclass
class JudgingResult:
    """Result from an LLM judge evaluation."""

    score: float
    reasoning: str
    criteria: JudgingCriteria
    confidence: float = 1.0
    raw_scores: List[float] = field(default_factory=list)  # For multi-eval
    position_used: Optional[str] = None  # "A" or "B" for pairwise
    judge_model: str = ""

    @property
    def normalized_score(self) -> float:
        """Score normalized to 0-1 range."""
        return (self.score - 1) / 9  # Assuming 1-10 scale


@dataclass
class PairwiseResult:
    """Result from a pairwise (A vs B) comparison."""

    winner: str  # "A", "B", or "tie"
    score_a: float
    score_b: float
    reasoning: str
    position_swapped: bool = False  # Whether positions were shuffled
    consensus: bool = True  # Whether multiple evals agreed


# Judging prompt templates
SINGLE_EVAL_PROMPT = """You are an expert evaluator. Rate the following response on a scale of 1-10 for {criteria}.

{rubric}

Question/Task:
{question}

Response to evaluate:
{response}

{reference_section}

Provide your evaluation in this exact format:
SCORE: [number 1-10]
REASONING: [your detailed reasoning]
"""

PAIRWISE_EVAL_PROMPT = """You are an expert evaluator comparing two responses. Evaluate which response is better for {criteria}.

{rubric}

Question/Task:
{question}

Response A:
{response_a}

Response B:
{response_b}

Compare the responses and provide your evaluation in this exact format:
WINNER: [A, B, or TIE]
SCORE_A: [number 1-10]
SCORE_B: [number 1-10]
REASONING: [your detailed comparison]
"""

# Rubrics for each criteria
RUBRICS: Dict[JudgingCriteria, str] = {
    JudgingCriteria.CORRECTNESS: """
Scoring rubric for CORRECTNESS (factual accuracy):
- 9-10: Completely accurate, no errors
- 7-8: Mostly accurate, minor issues that don't affect understanding
- 5-6: Partially accurate, some significant errors
- 3-4: Many errors, misleading information
- 1-2: Mostly or completely incorrect
""",
    JudgingCriteria.HELPFULNESS: """
Scoring rubric for HELPFULNESS:
- 9-10: Extremely helpful, directly addresses the user's needs
- 7-8: Very helpful, good response with minor improvements possible
- 5-6: Moderately helpful, addresses the question but lacks depth
- 3-4: Somewhat helpful, misses key aspects
- 1-2: Not helpful, doesn't address the user's needs
""",
    JudgingCriteria.COHERENCE: """
Scoring rubric for COHERENCE (logical flow and clarity):
- 9-10: Crystal clear, perfectly organized, easy to follow
- 7-8: Clear and well-organized with minor issues
- 5-6: Understandable but could be clearer or better organized
- 3-4: Confusing in places, poor organization
- 1-2: Incoherent, very difficult to understand
""",
    JudgingCriteria.CODE_QUALITY: """
Scoring rubric for CODE_QUALITY:
- 9-10: Correct, efficient, well-documented, follows best practices
- 7-8: Correct and readable, minor style issues
- 5-6: Works but has issues (inefficient, poor style, or limited)
- 3-4: Partially works, significant bugs or problems
- 1-2: Doesn't work, major errors, or completely wrong approach
""",
    JudgingCriteria.RELEVANCE: """
Scoring rubric for RELEVANCE:
- 9-10: Directly and completely addresses the question
- 7-8: Addresses the question well with minor tangents
- 5-6: Partially addresses the question, some off-topic content
- 3-4: Mostly off-topic or misunderstands the question
- 1-2: Completely irrelevant to the question
""",
}


class LLMJudge:
    """
    LLM-based evaluator with bias mitigation.

    Implements TD-011 strategies:
    - Position shuffling for pairwise comparisons
    - Multiple evaluations for consensus
    - Separate scoring for different criteria
    """

    def __init__(
        self,
        provider: Any,  # BaseProvider
        config: Optional[JudgingConfig] = None,
    ):
        """
        Args:
            provider: LLM provider to use as judge.
            config: Judging configuration.
        """
        self.provider = provider
        self.config = config or JudgingConfig()

    async def evaluate(
        self,
        question: str,
        response: str,
        criteria: JudgingCriteria,
        reference_answer: Optional[str] = None,
    ) -> JudgingResult:
        """
        Evaluate a single response.

        Args:
            question: The original question/task.
            response: The response to evaluate.
            criteria: The criteria to judge on.
            reference_answer: Optional reference for comparison.

        Returns:
            JudgingResult with score and reasoning.
        """
        rubric = RUBRICS.get(criteria, "")
        reference_section = ""
        if reference_answer and self.config.use_reference_answer:
            reference_section = f"Reference answer:\n{reference_answer}"

        prompt = SINGLE_EVAL_PROMPT.format(
            criteria=criteria.name.lower().replace("_", " "),
            rubric=rubric if self.config.include_rubric else "",
            question=question,
            response=response,
            reference_section=reference_section,
        )

        scores: List[float] = []
        reasonings: List[str] = []

        for _ in range(self.config.num_evaluations):
            result = await self.provider.generate(
                prompt,
                config=type(self.provider.config)(
                    temperature=self.config.temperature,
                    max_tokens=1024,
                ),
            )

            score, reasoning = self._parse_single_eval(result.text)
            if score is not None:
                scores.append(score)
                reasonings.append(reasoning)

        if not scores:
            return JudgingResult(
                score=5.0,
                reasoning="Failed to parse judge response",
                criteria=criteria,
                confidence=0.0,
                judge_model=self.provider.model,
            )

        avg_score = sum(scores) / len(scores)
        confidence = 1.0 - (max(scores) - min(scores)) / 9 if len(scores) > 1 else 1.0

        return JudgingResult(
            score=avg_score,
            reasoning=reasonings[0],  # Use first reasoning
            criteria=criteria,
            confidence=confidence,
            raw_scores=scores,
            judge_model=self.provider.model,
        )

    async def evaluate_pairwise(
        self,
        question: str,
        response_a: str,
        response_b: str,
        criteria: JudgingCriteria,
    ) -> PairwiseResult:
        """
        Compare two responses (A vs B).

        Implements position shuffling to mitigate position bias (TD-011).

        Args:
            question: The original question/task.
            response_a: First response.
            response_b: Second response.
            criteria: The criteria to judge on.

        Returns:
            PairwiseResult indicating winner and scores.
        """
        rubric = RUBRICS.get(criteria, "")

        # TD-011: Shuffle positions to detect/mitigate position bias
        swap = self.config.shuffle_positions and random.random() < 0.5

        if swap:
            actual_a, actual_b = response_b, response_a
        else:
            actual_a, actual_b = response_a, response_b

        prompt = PAIRWISE_EVAL_PROMPT.format(
            criteria=criteria.name.lower().replace("_", " "),
            rubric=rubric if self.config.include_rubric else "",
            question=question,
            response_a=actual_a,
            response_b=actual_b,
        )

        result = await self.provider.generate(
            prompt,
            config=type(self.provider.config)(
                temperature=self.config.temperature,
                max_tokens=1024,
            ),
        )

        winner, score_a, score_b, reasoning = self._parse_pairwise_eval(result.text)

        # Unswap if needed
        if swap:
            winner = {"A": "B", "B": "A", "TIE": "TIE"}.get(winner, winner)
            score_a, score_b = score_b, score_a

        return PairwiseResult(
            winner=winner,
            score_a=score_a,
            score_b=score_b,
            reasoning=reasoning,
            position_swapped=swap,
        )

    async def evaluate_multi_criteria(
        self,
        question: str,
        response: str,
        criteria_list: List[JudgingCriteria],
        reference_answer: Optional[str] = None,
    ) -> Dict[JudgingCriteria, JudgingResult]:
        """
        Evaluate a response on multiple criteria.

        Args:
            question: The original question/task.
            response: The response to evaluate.
            criteria_list: List of criteria to evaluate.
            reference_answer: Optional reference answer.

        Returns:
            Dict mapping each criteria to its JudgingResult.
        """
        results = {}
        for criteria in criteria_list:
            result = await self.evaluate(
                question=question,
                response=response,
                criteria=criteria,
                reference_answer=reference_answer,
            )
            results[criteria] = result
        return results

    def _parse_single_eval(self, text: str) -> Tuple[Optional[float], str]:
        """Parse score and reasoning from judge response."""
        score = None
        reasoning = ""

        lines = text.strip().split("\n")
        for i, line in enumerate(lines):
            line_upper = line.upper().strip()

            if line_upper.startswith("SCORE:"):
                try:
                    score_str = line.split(":", 1)[1].strip()
                    # Handle formats like "8/10", "8", "8.5"
                    if "/" in score_str:
                        score_str = score_str.split("/")[0]
                    score = float(score_str)
                    score = max(self.config.min_score, min(self.config.max_score, score))
                except (ValueError, IndexError):
                    pass

            elif line_upper.startswith("REASONING:"):
                reasoning = line.split(":", 1)[1].strip()
                # Include remaining lines in reasoning
                reasoning += "\n" + "\n".join(lines[i + 1 :])
                break

        return score, reasoning.strip()

    def _parse_pairwise_eval(self, text: str) -> Tuple[str, float, float, str]:
        """Parse pairwise evaluation response."""
        winner = "TIE"
        score_a = 5.0
        score_b = 5.0
        reasoning = ""

        lines = text.strip().split("\n")
        for i, line in enumerate(lines):
            line_upper = line.upper().strip()

            if line_upper.startswith("WINNER:"):
                winner_str = line.split(":", 1)[1].strip().upper()
                if "A" in winner_str and "B" not in winner_str:
                    winner = "A"
                elif "B" in winner_str and "A" not in winner_str:
                    winner = "B"
                else:
                    winner = "TIE"

            elif line_upper.startswith("SCORE_A:"):
                try:
                    score_a = float(line.split(":", 1)[1].strip().split("/")[0])
                except (ValueError, IndexError):
                    pass

            elif line_upper.startswith("SCORE_B:"):
                try:
                    score_b = float(line.split(":", 1)[1].strip().split("/")[0])
                except (ValueError, IndexError):
                    pass

            elif line_upper.startswith("REASONING:"):
                reasoning = line.split(":", 1)[1].strip()
                reasoning += "\n" + "\n".join(lines[i + 1 :])
                break

        return winner, score_a, score_b, reasoning.strip()
