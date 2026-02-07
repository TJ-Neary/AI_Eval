"""
README.md results updater.

Maintains a results index and renders a summary table between
markers in README.md so GitHub visitors see real benchmark results.
"""

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from utils.marker_parser import has_markers, replace_marker_content

logger = logging.getLogger(__name__)

DEFAULT_README = Path("README.md")
DEFAULT_INDEX = Path("reports/.results_index.json")


@dataclass
class ResultSummary:
    """Lightweight summary of a benchmark run for the results table."""

    model: str
    provider: str
    score: float
    tokens_per_second: float
    pass_rate: str  # e.g. "8/10"
    hardware: str  # e.g. "Apple M4 Max (48GB)"
    date: str  # ISO date
    report_path: str = ""  # Relative path to full report


def _load_index(index_path: Path) -> List[Dict[str, Any]]:
    """Load the results index file."""
    if not index_path.exists():
        return []
    with open(index_path) as f:
        return json.load(f)


def _save_index(index_path: Path, entries: List[Dict[str, Any]]) -> None:
    """Save the results index file."""
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with open(index_path, "w") as f:
        json.dump(entries, f, indent=2)


def _merge_result(
    existing: List[Dict[str, Any]], new: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Merge a new result into the index, replacing same model+provider."""
    key = (new["model"], new["provider"])
    filtered = [e for e in existing if (e["model"], e["provider"]) != key]
    filtered.append(new)
    # Sort by score descending
    filtered.sort(key=lambda x: x.get("score", 0), reverse=True)
    return filtered


def _render_results_table(summaries: List[Dict[str, Any]]) -> str:
    """Render a markdown table from result summaries."""
    if not summaries:
        return "*No benchmark results yet. Run `python -m src run` to generate.*\n"

    lines = [
        "| Model | Provider | Score | Tokens/sec | Pass Rate | Hardware | Date |",
        "|-------|----------|-------|------------|-----------|----------|------|",
    ]
    for s in summaries:
        report_link = s["model"]
        if s.get("report_path"):
            report_link = f"[{s['model']}]({s['report_path']})"
        lines.append(
            f"| {report_link} "
            f"| {s['provider']} "
            f"| {s['score']:.1f} "
            f"| {s['tokens_per_second']:.1f} "
            f"| {s['pass_rate']} "
            f"| {s['hardware']} "
            f"| {s['date']} |"
        )
    lines.append("")  # Trailing newline before end marker
    return "\n".join(lines) + "\n"


def update_readme_results(
    result: Any,
    readme_path: Path = DEFAULT_README,
    report_path: Optional[Path] = None,
    index_path: Path = DEFAULT_INDEX,
) -> bool:
    """
    Update README.md with benchmark results from a BenchmarkResult.

    Maintains an accumulating results index so multiple runs build up
    a comparison table.

    Args:
        result: BenchmarkResult from the benchmark runner.
        readme_path: Path to README.md with markers.
        report_path: Path to the generated report (for linking).
        index_path: Path to the results index JSON.

    Returns:
        True if README was updated, False if markers not found.
    """
    if not has_markers(readme_path):
        logger.warning(f"No markers found in {readme_path}")
        return False

    # Build summary
    hw = result.hardware
    hardware_str = f"{hw.chip_name} ({hw.ram_gb:.0f}GB)"

    summary = ResultSummary(
        model=result.model,
        provider=result.provider,
        score=float(result.overall_score),
        tokens_per_second=float(result.avg_tokens_per_second),
        pass_rate=f"{result.total_passed}/{result.total_tests}",
        hardware=hardware_str,
        date=result.timestamp.strftime("%Y-%m-%d"),
        report_path=str(report_path) if report_path else "",
    )

    # Load, merge, save index
    existing = _load_index(index_path)
    merged = _merge_result(existing, asdict(summary))
    _save_index(index_path, merged)

    # Render and update README
    table = _render_results_table(merged)
    replace_marker_content(readme_path, table)
    logger.info(f"Updated {readme_path} with {len(merged)} result(s)")

    return True
