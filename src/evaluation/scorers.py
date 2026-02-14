"""
Custom Scorers for Evaluation Requests

Task-specific scoring functions that evaluate model responses beyond
standard benchmark scoring. Each scorer implements the Scorer protocol
and produces a normalized ScorerResult.
"""

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@dataclass
class ScorerResult:
    """Result from a custom scorer."""

    name: str
    score: float  # 0.0-1.0 normalized
    passed: bool
    details: str = ""
    raw_value: Any = None  # The measured value (e.g., latency in seconds)


@runtime_checkable
class Scorer(Protocol):
    """Protocol for custom scorers."""

    name: str

    def score(self, response: str, context: Dict[str, Any]) -> ScorerResult:
        """Score a model response.

        Args:
            response: The model's text output.
            context: Additional context including:
                - generation_time_ms: float
                - tokens_per_second: float
                - source_text: str (from input_file, for citation/quote checking)
                - expected_patterns: List[str]
                - Any scorer-specific config from scorer_config

        Returns:
            ScorerResult with normalized 0-1 score and pass/fail.
        """
        ...


class JsonValidityScorer:
    """Checks if response contains valid JSON.

    Optionally validates that specific fields exist. Extracts JSON from
    markdown code blocks if the response wraps JSON in ```json ... ```.
    """

    name = "json_validity"

    def __init__(self, required_fields: Optional[List[str]] = None) -> None:
        self.required_fields = required_fields or []

    def score(self, response: str, context: Dict[str, Any]) -> ScorerResult:
        """Score JSON validity of the response."""
        extracted = self._extract_json(response)
        if extracted is None:
            return ScorerResult(
                name=self.name,
                score=0.0,
                passed=False,
                details="No valid JSON found in response",
                raw_value=None,
            )

        try:
            parsed = json.loads(extracted)
        except (json.JSONDecodeError, TypeError):
            return ScorerResult(
                name=self.name,
                score=0.0,
                passed=False,
                details="JSON parsing failed",
                raw_value=None,
            )

        # Check required fields
        if self.required_fields and isinstance(parsed, dict):
            missing = [f for f in self.required_fields if f not in parsed]
            if missing:
                return ScorerResult(
                    name=self.name,
                    score=0.5,
                    passed=False,
                    details=f"Missing required fields: {', '.join(missing)}",
                    raw_value=parsed,
                )

        return ScorerResult(
            name=self.name,
            score=1.0,
            passed=True,
            details="Valid JSON with all required fields",
            raw_value=parsed,
        )

    def _extract_json(self, text: str) -> Optional[str]:
        """Extract JSON from response, handling markdown code blocks."""
        # Try markdown code block first
        code_block = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
        if code_block:
            return code_block.group(1).strip()

        # Try to find JSON object or array directly
        text = text.strip()
        if text.startswith(("{", "[")):
            return text

        # Look for embedded JSON object/array
        for pattern in [r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", r"\[.*\]"]:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                return match.group(0)

        return None


class LatencyScorer:
    """Checks generation time against a threshold."""

    name = "latency"

    def __init__(self, max_seconds: float = 15.0) -> None:
        self.max_seconds = max_seconds

    def score(self, response: str, context: Dict[str, Any]) -> ScorerResult:
        """Score latency against threshold."""
        generation_ms = context.get("generation_time_ms", 0.0)
        latency_seconds = generation_ms / 1000.0

        passed = latency_seconds <= self.max_seconds
        # Score: 1.0 if at or under threshold, degrades linearly up to 2x threshold
        if passed:
            score = 1.0
        elif latency_seconds <= self.max_seconds * 2:
            score = 1.0 - (latency_seconds - self.max_seconds) / self.max_seconds
        else:
            score = 0.0

        return ScorerResult(
            name=self.name,
            score=max(0.0, score),
            passed=passed,
            details=f"{latency_seconds:.1f}s (threshold: {self.max_seconds}s)",
            raw_value=latency_seconds,
        )


class PatternAccuracyScorer:
    """Checks what fraction of expected patterns appear in the response."""

    name = "pattern_accuracy"

    def score(self, response: str, context: Dict[str, Any]) -> ScorerResult:
        """Score pattern matching accuracy."""
        patterns = context.get("expected_patterns", [])
        if not patterns:
            return ScorerResult(
                name=self.name,
                score=1.0,
                passed=True,
                details="No patterns to check",
            )

        matches = 0
        matched_patterns: list[str] = []
        missed_patterns: list[str] = []

        for pattern in patterns:
            if re.search(pattern, response, re.IGNORECASE | re.DOTALL):
                matches += 1
                matched_patterns.append(pattern)
            else:
                missed_patterns.append(pattern)

        score = matches / len(patterns)
        details_parts = [f"{matches}/{len(patterns)} patterns matched"]
        if missed_patterns:
            details_parts.append(f"missed: {missed_patterns}")

        return ScorerResult(
            name=self.name,
            score=score,
            passed=score >= 0.5,
            details="; ".join(details_parts),
            raw_value={"matched": matched_patterns, "missed": missed_patterns},
        )


class CitationAccuracyScorer:
    """Checks that citations reference valid pages from the source text.

    Expects citations in format [File.pdf, Page N] or similar.
    Verifies referenced pages exist in the source_text metadata.
    """

    name = "citation_accuracy"

    def __init__(self, citation_pattern: str = r"\[([^\]]+\.pdf),\s*Page\s*(\d+)\]") -> None:
        self.citation_pattern = citation_pattern

    def score(self, response: str, context: Dict[str, Any]) -> ScorerResult:
        """Score citation accuracy against source text."""
        source_text = context.get("source_text", "")
        if not source_text:
            return ScorerResult(
                name=self.name,
                score=0.0,
                passed=False,
                details="No source_text in context for citation verification",
            )

        # Extract citations from response
        citations = re.findall(self.citation_pattern, response)
        if not citations:
            # Check if response has any citation-like patterns at all
            any_citation = re.search(r"\[[^\]]*[Pp]age\s*\d+[^\]]*\]", response)
            if any_citation:
                citations = re.findall(r"\[[^\]]*[Pp]age\s*\d+[^\]]*\]", response)
                # Can't verify format but found something
                return ScorerResult(
                    name=self.name,
                    score=0.5,
                    passed=False,
                    details=f"Found {len(citations)} citations but format doesn't match expected pattern",
                    raw_value={"found": len(citations), "verified": 0},
                )
            return ScorerResult(
                name=self.name,
                score=0.0,
                passed=False,
                details="No citations found in response",
                raw_value={"found": 0, "verified": 0},
            )

        # Verify citations: check if referenced pages appear in source text
        verified = 0
        for citation in citations:
            if isinstance(citation, tuple):
                filename, page_num = citation
                # Check that the specific page reference exists in source
                if f"Page {page_num}" in source_text:
                    verified += 1
            else:
                # Single match — check if the citation text references source content
                if str(citation) in source_text:
                    verified += 1

        total = len(citations)
        accuracy = verified / total if total > 0 else 0.0

        return ScorerResult(
            name=self.name,
            score=accuracy,
            passed=accuracy >= 0.5,
            details=f"{verified}/{total} citations verified against source",
            raw_value={"found": total, "verified": verified},
        )


class VerbatimQuoteScorer:
    """Checks that quoted text actually appears in the source input.

    Extracts text between quotation marks and verifies each is a substring
    of the source_text from context.
    """

    name = "verbatim_quote"

    def __init__(self, min_quote_length: int = 10) -> None:
        self.min_quote_length = min_quote_length

    def score(self, response: str, context: Dict[str, Any]) -> ScorerResult:
        """Score verbatim quote accuracy against source text."""
        source_text = context.get("source_text", "")
        if not source_text:
            return ScorerResult(
                name=self.name,
                score=0.0,
                passed=False,
                details="No source_text in context for quote verification",
            )

        # Extract quoted text — match both "..." and values in JSON "quote": "..."
        quotes = self._extract_quotes(response)
        if not quotes:
            return ScorerResult(
                name=self.name,
                score=0.0,
                passed=False,
                details="No quotes found in response",
                raw_value={"found": 0, "verified": 0},
            )

        # Verify each quote exists in source
        verified = 0
        source_lower = source_text.lower()
        for quote in quotes:
            if quote.lower() in source_lower:
                verified += 1

        total = len(quotes)
        accuracy = verified / total if total > 0 else 0.0

        return ScorerResult(
            name=self.name,
            score=accuracy,
            passed=accuracy >= 0.5,
            details=f"{verified}/{total} quotes found verbatim in source",
            raw_value={"found": total, "verified": verified},
        )

    def _extract_quotes(self, text: str) -> List[str]:
        """Extract substantial quoted text from response."""
        # Match JSON string values that look like quotes
        # and standalone quoted text
        quotes: list[str] = []

        # Pattern for "verbatim_quote": "..." or "quote": "..." in JSON
        json_quotes = re.findall(
            r'"(?:verbatim_quote|quote|quotes?|excerpt)"\s*:\s*"([^"]+)"', text
        )
        quotes.extend(json_quotes)

        # Pattern for standalone quoted text (not JSON keys)
        if not quotes:
            standalone = re.findall(r'"([^"]{10,})"', text)
            # Filter out things that look like JSON keys
            quotes.extend(q for q in standalone if not re.match(r"^[a-z_]+$", q))

        # Filter by minimum length
        return [q for q in quotes if len(q) >= self.min_quote_length]


# --- Scorer Registry ---

SCORER_REGISTRY: Dict[str, type] = {
    "json_validity": JsonValidityScorer,
    "latency": LatencyScorer,
    "pattern_accuracy": PatternAccuracyScorer,
    "citation_accuracy": CitationAccuracyScorer,
    "verbatim_quote": VerbatimQuoteScorer,
}


def get_scorer(name: str, config: Optional[Dict[str, Any]] = None) -> Scorer:
    """Create a scorer by name with optional configuration.

    Args:
        name: Scorer name from SCORER_REGISTRY.
        config: Optional kwargs passed to scorer constructor.

    Returns:
        Configured Scorer instance.

    Raises:
        KeyError: If scorer name is not registered.
    """
    if name not in SCORER_REGISTRY:
        available = ", ".join(sorted(SCORER_REGISTRY.keys()))
        raise KeyError(f"Unknown scorer: '{name}'. Available: {available}")

    scorer_cls = SCORER_REGISTRY[name]
    instance: Scorer
    if config:
        instance = scorer_cls(**config)
    else:
        instance = scorer_cls()
    return instance
