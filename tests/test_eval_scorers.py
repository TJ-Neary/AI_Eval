"""Tests for custom evaluation scorers."""

import pytest

from src.evaluation.scorers import (
    SCORER_REGISTRY,
    CitationAccuracyScorer,
    JsonValidityScorer,
    LatencyScorer,
    PatternAccuracyScorer,
    VerbatimQuoteScorer,
    get_scorer,
)


class TestJsonValidityScorer:
    def test_valid_json_passes(self) -> None:
        scorer = JsonValidityScorer()
        result = scorer.score('{"name": "test", "value": 42}', {})
        assert result.passed is True
        assert result.score == 1.0

    def test_invalid_json_fails(self) -> None:
        scorer = JsonValidityScorer()
        result = scorer.score("This is not JSON at all", {})
        assert result.passed is False
        assert result.score == 0.0

    def test_extracts_from_markdown_code_block(self) -> None:
        scorer = JsonValidityScorer()
        response = '```json\n{"claims": [{"text": "test"}]}\n```'
        result = scorer.score(response, {})
        assert result.passed is True
        assert result.score == 1.0

    def test_required_fields_present(self) -> None:
        scorer = JsonValidityScorer(required_fields=["claims", "analysis"])
        response = '{"claims": [], "analysis": "none"}'
        result = scorer.score(response, {})
        assert result.passed is True
        assert result.score == 1.0

    def test_required_fields_missing(self) -> None:
        scorer = JsonValidityScorer(required_fields=["claims", "analysis"])
        response = '{"claims": []}'
        result = scorer.score(response, {})
        assert result.passed is False
        assert result.score == 0.5
        assert "analysis" in result.details

    def test_json_array(self) -> None:
        scorer = JsonValidityScorer()
        result = scorer.score("[1, 2, 3]", {})
        assert result.passed is True

    def test_json_embedded_in_text(self) -> None:
        scorer = JsonValidityScorer()
        result = scorer.score('Here is the answer: {"result": true}', {})
        assert result.passed is True


class TestLatencyScorer:
    def test_under_threshold_passes(self) -> None:
        scorer = LatencyScorer(max_seconds=15.0)
        result = scorer.score("response", {"generation_time_ms": 5000.0})
        assert result.passed is True
        assert result.score == 1.0
        assert result.raw_value == 5.0

    def test_at_threshold_passes(self) -> None:
        scorer = LatencyScorer(max_seconds=15.0)
        result = scorer.score("response", {"generation_time_ms": 15000.0})
        assert result.passed is True

    def test_over_threshold_fails(self) -> None:
        scorer = LatencyScorer(max_seconds=15.0)
        result = scorer.score("response", {"generation_time_ms": 20000.0})
        assert result.passed is False
        assert result.score < 1.0
        assert result.score > 0.0  # Degrades linearly

    def test_way_over_threshold_zero(self) -> None:
        scorer = LatencyScorer(max_seconds=15.0)
        result = scorer.score("response", {"generation_time_ms": 60000.0})
        assert result.passed is False
        assert result.score == 0.0

    def test_missing_generation_time(self) -> None:
        scorer = LatencyScorer(max_seconds=15.0)
        result = scorer.score("response", {})
        assert result.passed is True  # 0ms < 15s
        assert result.raw_value == 0.0


class TestPatternAccuracyScorer:
    def test_all_patterns_match(self) -> None:
        scorer = PatternAccuracyScorer()
        context = {"expected_patterns": [r'"claims"', r'"citations"']}
        response = '{"claims": [], "citations": []}'
        result = scorer.score(response, context)
        assert result.score == 1.0
        assert result.passed is True

    def test_partial_match(self) -> None:
        scorer = PatternAccuracyScorer()
        context = {"expected_patterns": [r'"claims"', r'"citations"', r'"missing_field"']}
        response = '{"claims": [], "citations": []}'
        result = scorer.score(response, context)
        assert abs(result.score - 2.0 / 3.0) < 0.01
        assert result.passed is True  # >= 0.5

    def test_no_match(self) -> None:
        scorer = PatternAccuracyScorer()
        context = {"expected_patterns": [r'"impossible_field"']}
        response = '{"something": "else"}'
        result = scorer.score(response, context)
        assert result.score == 0.0
        assert result.passed is False

    def test_no_patterns(self) -> None:
        scorer = PatternAccuracyScorer()
        result = scorer.score("any response", {"expected_patterns": []})
        assert result.score == 1.0
        assert result.passed is True

    def test_no_patterns_in_context(self) -> None:
        scorer = PatternAccuracyScorer()
        result = scorer.score("any response", {})
        assert result.score == 1.0


class TestCitationAccuracyScorer:
    def test_valid_citations(self) -> None:
        scorer = CitationAccuracyScorer()
        source = "Report.pdf Page 3 shows left knee ROM. Report.pdf Page 7 shows medications."
        response = "Citations: [Report.pdf, Page 3] and [Report.pdf, Page 7]"
        result = scorer.score(response, {"source_text": source})
        assert result.passed is True
        assert result.score == 1.0

    def test_invalid_citations(self) -> None:
        scorer = CitationAccuracyScorer()
        source = "Report.pdf Page 3 shows findings."
        response = "Citations: [Report.pdf, Page 99]"  # Page 99 not in source
        result = scorer.score(response, {"source_text": source})
        assert result.score < 1.0

    def test_no_citations_found(self) -> None:
        scorer = CitationAccuracyScorer()
        result = scorer.score("No citations here.", {"source_text": "some source"})
        assert result.passed is False
        assert result.score == 0.0

    def test_no_source_text(self) -> None:
        scorer = CitationAccuracyScorer()
        result = scorer.score("[Report.pdf, Page 1]", {})
        assert result.passed is False
        assert "No source_text" in result.details


class TestVerbatimQuoteScorer:
    def test_exact_quote_found(self) -> None:
        scorer = VerbatimQuoteScorer()
        source = "The patient demonstrates flexion to 90 degrees with pain noted at endpoint."
        response = '{"verbatim_quote": "flexion to 90 degrees with pain noted at endpoint"}'
        result = scorer.score(response, {"source_text": source})
        assert result.passed is True
        assert result.score == 1.0

    def test_quote_not_in_source(self) -> None:
        scorer = VerbatimQuoteScorer()
        source = "The patient is stable."
        response = '{"verbatim_quote": "patient shows significant deterioration in condition"}'
        result = scorer.score(response, {"source_text": source})
        assert result.passed is False
        assert result.score == 0.0

    def test_partial_quotes(self) -> None:
        scorer = VerbatimQuoteScorer()
        source = "ROM limited to 90 degrees. Pain at endpoint. No instability noted."
        response = '{"quotes": ["ROM limited to 90 degrees", ' '"severe instability observed"]}'
        result = scorer.score(response, {"source_text": source})
        assert result.score == 0.5  # 1 of 2 found

    def test_no_quotes_found(self) -> None:
        scorer = VerbatimQuoteScorer()
        result = scorer.score("Just text, no quotes", {"source_text": "source"})
        assert result.passed is False

    def test_no_source_text(self) -> None:
        scorer = VerbatimQuoteScorer()
        result = scorer.score('"some quote here for testing"', {})
        assert result.passed is False
        assert "No source_text" in result.details

    def test_min_quote_length_filter(self) -> None:
        scorer = VerbatimQuoteScorer(min_quote_length=20)
        source = "Short text here and some longer text that should be matched"
        response = '{"quote": "short", "verbatim_quote": "some longer text that should be matched"}'
        result = scorer.score(response, {"source_text": source})
        # "short" is under min length, only the long one counts
        assert result.score == 1.0


class TestScorerRegistry:
    def test_get_known_scorer(self) -> None:
        scorer = get_scorer("json_validity")
        assert isinstance(scorer, JsonValidityScorer)

    def test_get_scorer_with_config(self) -> None:
        scorer = get_scorer("latency", {"max_seconds": 30.0})
        assert isinstance(scorer, LatencyScorer)
        assert scorer.max_seconds == 30.0

    def test_get_scorer_with_json_config(self) -> None:
        scorer = get_scorer("json_validity", {"required_fields": ["claims"]})
        assert isinstance(scorer, JsonValidityScorer)
        assert scorer.required_fields == ["claims"]

    def test_unknown_scorer_raises(self) -> None:
        with pytest.raises(KeyError, match="Unknown scorer"):
            get_scorer("nonexistent_scorer")

    def test_all_registry_entries_instantiate(self) -> None:
        for name in SCORER_REGISTRY:
            scorer = get_scorer(name)
            assert hasattr(scorer, "name")
            assert hasattr(scorer, "score")
