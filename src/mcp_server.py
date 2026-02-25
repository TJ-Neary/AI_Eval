"""AI_Eval Integration Track — MCP Domain Server.

Exposes 4 read-only tools for ecosystem agents to query evaluation results,
get model recommendations, and discover capabilities.

All tools are tier: local (no LLM calls, pure data queries).

Transport: stdio (default for FastMCP)

Usage:
    python src/mcp_server.py

Registration (Claude Code):
    claude mcp add aieval python3 src/mcp_server.py
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from fastmcp import FastMCP

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = PROJECT_ROOT / "reports"
INDEX_PATH = REPORTS_DIR / ".results_index.json"

# ---------------------------------------------------------------------------
# FastMCP instance
# ---------------------------------------------------------------------------
mcp = FastMCP(name="AI_Eval Domain MCP")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _load_index() -> list[dict[str, Any]]:
    """Load the results index file."""
    if not INDEX_PATH.exists():
        return []
    try:
        return json.loads(INDEX_PATH.read_text(encoding="utf-8"))  # type: ignore[no-any-return]
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to read results index: %s", exc)
        return []


def _load_report(model_key: str) -> dict[str, Any] | None:
    """Load a JSON report by model key (e.g., 'qwen2.5_32b_20260207_043753')."""
    json_file = REPORTS_DIR / f"{model_key}.json"
    if not json_file.exists():
        # Try matching partial model name
        for f in REPORTS_DIR.glob("*.json"):
            if f.name.startswith("."):
                continue
            if model_key in f.stem:
                json_file = f
                break
        else:
            return None

    try:
        return json.loads(json_file.read_text(encoding="utf-8"))  # type: ignore[no-any-return]
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to read report %s: %s", json_file.name, exc)
        return None


def _load_all_reports() -> list[dict[str, Any]]:
    """Load all JSON reports from the reports directory."""
    reports: list[dict[str, Any]] = []
    if not REPORTS_DIR.exists():
        return reports
    for f in sorted(REPORTS_DIR.glob("*.json")):
        if f.name.startswith("."):
            continue
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            data["_report_key"] = f.stem
            reports.append(data)
        except (json.JSONDecodeError, OSError):
            continue
    return reports


# ---------------------------------------------------------------------------
# Tool 1: Get model recommendation
# ---------------------------------------------------------------------------


@mcp.tool()  # type: ignore[misc]
def aieval_get_model_recommendation(task_type: str) -> dict[str, Any]:
    """Recommend the best model for a given task type based on evaluation results.

    Uses fitness profile scores from completed evaluations to rank models.

    Args:
        task_type: The task type to optimize for.
            Valid types: Chat Application, Code Assistant, Data Pipeline,
            Document Processor, Rag Knowledge Engine.

    Returns:
        Dict with recommendations sorted by fitness score (highest first).
    """
    if not task_type or not task_type.strip():
        return {"error": "task_type parameter is required"}

    reports = _load_all_reports()
    if not reports:
        return {"error": "No evaluation reports found", "recommendations": []}

    # Normalize task type for matching (case-insensitive)
    task_lower = task_type.lower().strip()

    recommendations = []
    valid_types: set[str] = set()

    for report in reports:
        profiles = report.get("fitness_profiles", {})
        for profile_name, score_str in profiles.items():
            valid_types.add(profile_name)
            if profile_name.lower() == task_lower:
                try:
                    score = float(score_str)
                except (ValueError, TypeError):
                    score = 0.0
                recommendations.append(
                    {
                        "model": report.get("model", "unknown"),
                        "provider": report.get("provider", "unknown"),
                        "fitness_score": score,
                        "overall_score": report.get("overall_score", 0.0),
                        "avg_tokens_per_second": report.get("avg_tokens_per_second", 0.0),
                        "hardware": report.get("hardware", {})
                        .get("accelerator", {})
                        .get("name", "unknown"),
                        "evaluated_at": report.get("timestamp", ""),
                    }
                )

    if not recommendations:
        return {
            "error": f"No evaluations found for task type '{task_type}'",
            "valid_types": sorted(valid_types),
            "recommendations": [],
        }

    recommendations.sort(key=lambda r: r["fitness_score"], reverse=True)

    return {
        "task_type": task_type,
        "recommendations": recommendations,
        "best_model": recommendations[0]["model"],
        "best_score": recommendations[0]["fitness_score"],
    }


# ---------------------------------------------------------------------------
# Tool 2: List evaluations
# ---------------------------------------------------------------------------


@mcp.tool()  # type: ignore[misc]
def aieval_list_evaluations() -> list[dict[str, Any]]:
    """List all completed evaluation runs with summary data.

    Returns:
        List of evaluation summaries sorted by date (newest first).
    """
    index = _load_index()
    if index:
        return sorted(index, key=lambda r: r.get("date", ""), reverse=True)

    # Fallback: build from report files
    reports = _load_all_reports()
    summaries = []
    for report in reports:
        summaries.append(
            {
                "model": report.get("model", "unknown"),
                "provider": report.get("provider", "unknown"),
                "score": report.get("overall_score", 0.0),
                "tokens_per_second": report.get("avg_tokens_per_second", 0.0),
                "pass_rate": f"{report.get('total_passed', 0)}/{report.get('total_tests', 0)}",
                "hardware": report.get("hardware", {})
                .get("accelerator", {})
                .get("name", "unknown"),
                "date": report.get("timestamp", "")[:10],
                "report_key": report.get("_report_key", ""),
            }
        )

    return sorted(summaries, key=lambda r: r.get("date", ""), reverse=True)


# ---------------------------------------------------------------------------
# Tool 3: Get evaluation
# ---------------------------------------------------------------------------


@mcp.tool()  # type: ignore[misc]
def aieval_get_evaluation(report_key: str) -> dict[str, Any]:
    """Get full results for a specific evaluation run.

    Args:
        report_key: The report identifier (filename stem, e.g.,
            'qwen2.5_32b_20260207_043753') or partial model name
            (e.g., 'qwen2.5_32b').

    Returns:
        Full evaluation report dict, or error if not found.
    """
    if not report_key or not report_key.strip():
        return {"error": "report_key parameter is required"}

    report = _load_report(report_key.strip())
    if not report:
        return {"error": f"Evaluation report not found: {report_key}"}

    # Remove internal key if present
    report.pop("_report_key", None)
    return report


# ---------------------------------------------------------------------------
# Tool 4: Get manifest
# ---------------------------------------------------------------------------


@mcp.tool()  # type: ignore[misc]
def aieval_get_manifest() -> dict[str, Any]:
    """Return service capabilities and version information for discovery.

    Returns:
        Manifest dict with service name, version, tool list, data types.
    """
    return {
        "service": "AI_Eval",
        "version": "1.0.0",
        "description": "LLM evaluation framework — benchmarks, fitness profiles, model recommendations",
        "tools": [
            {"name": "aieval_get_model_recommendation", "type": "query"},
            {"name": "aieval_list_evaluations", "type": "query"},
            {"name": "aieval_get_evaluation", "type": "query"},
            {"name": "aieval_get_manifest", "type": "status"},
        ],
        "data_types": ["evaluation_report", "fitness_profile", "model_recommendation"],
        "requires_llm": [],
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    mcp.run(transport="stdio")
