"""
Catalog exporter for _HQ integration.

Pushes benchmark results to _HQ/evaluations/ by:
1. Copying report files to evaluations/reports/ and data/
2. Updating MODEL_CATALOG.md with model rows in the correct section
3. Updating DECISION_MATRIX.md with benchmark-backed recommendations
4. Updating HARDWARE_PROFILES.md with tested configurations
5. Updating EVALUATION_LOG.md with history entries and summary rows
"""

import logging
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from src.reporting.report_generator import _calculate_fitness_scores, _score_rating
from utils.exceptions import ExportError
from utils.marker_parser import has_markers, read_marker_content, replace_marker_content

logger = logging.getLogger(__name__)

DEFAULT_EVALUATIONS_PATH = Path.home() / "Tech_Projects" / "_HQ" / "evaluations"


@dataclass
class ExportConfig:
    """Configuration for catalog export."""

    evaluations_path: Path = DEFAULT_EVALUATIONS_PATH
    update_catalog: bool = True
    update_matrix: bool = True
    update_hardware: bool = True
    update_log: bool = True
    copy_reports: bool = True
    requesting_project: str = ""
    use_case: str = ""


def _extract_param_count(model: str) -> Optional[str]:
    """Extract parameter count from model name (e.g. 'qwen2.5:32b' -> '32B')."""
    match = re.search(r"(\d+\.?\d*)[bB]", model)
    if match:
        num = match.group(1)
        # Drop trailing .0
        if num.endswith(".0"):
            num = num[:-2]
        return f"{num}B"
    return None


def _classify_model(model: str, provider: str) -> str:
    """Classify a model into a catalog section based on name and size.

    Returns the marker suffix (e.g. 'local-midrange', 'local-large').
    """
    # Keyword-based classification (checked before size, since some models lack param counts)
    vision_keywords = ["vision", "llava", "moondream"]
    if any(kw in model.lower() for kw in vision_keywords):
        return "local-vision"

    code_keywords = ["coder", "codestral", "starcoder", "deepseek-coder"]
    if any(kw in model.lower() for kw in code_keywords):
        return "local-code"

    # Size-based classification
    params = _extract_param_count(model)
    if not params:
        return "local-midrange"  # Default fallback

    num_match = re.match(r"(\d+\.?\d*)", params)
    if not num_match:
        return "local-midrange"
    size = float(num_match.group(1))

    # Size-based classification
    if size <= 3:
        return "local-compact"
    elif size <= 14:
        return "local-midrange"
    else:
        return "local-large"


def _get_category_score(result: Any, category: str) -> str:
    """Get a category score from results, or '—' if not tested."""
    cat = result.categories.get(category)
    if cat and cat.tests:
        return f"{cat.avg_score:.0f}"
    return "—"


def _determine_best_for(result: Any, scoring_config: Dict[str, Any]) -> str:
    """Determine best use cases from fitness profile scores."""
    category_scores = {
        name: float(cat.avg_score) for name, cat in result.categories.items() if cat.tests
    }
    profiles = scoring_config.get("fitness_profiles", {})
    fitness = _calculate_fitness_scores(category_scores, profiles)

    # Use cases where score >= 75
    good_uses = [
        name.replace("-", " ")
        .replace("rag knowledge engine", "RAG")
        .replace("code assistant", "Code")
        .replace("document processor", "Document processing")
        .replace("chat application", "Chat")
        .replace("data pipeline", "Data pipelines")
        for name, score in fitness.items()
        if score >= 75
    ]
    return ", ".join(good_uses) if good_uses else "General purpose"


def _render_catalog_row(result: Any, report_relpath: str, scoring_config: Dict[str, Any]) -> str:
    """Render a MODEL_CATALOG.md table row for a local model."""
    params = _extract_param_count(result.model) or "?"
    quant = "Q4_K_M"  # Default for Ollama models
    ram = f"{_estimate_ram(params)} GB"
    tps = f"{result.avg_tokens_per_second:.0f}"

    code = _get_category_score(result, "code-generation")
    analysis = _get_category_score(result, "document-analysis")
    chat = _get_category_score(result, "conversational")
    structured = _get_category_score(result, "structured-output")
    best_for = _determine_best_for(result, scoring_config)
    report_link = f"[→]({report_relpath})" if report_relpath else "—"

    return (
        f"| {result.model} | {params} | {quant} | {ram} | {tps} "
        f"| {code} | {analysis} | {chat} | {structured} "
        f"| {best_for} | {report_link} |"
    )


def _estimate_ram(params: Optional[str]) -> str:
    """Estimate minimum RAM for Q4 quantization."""
    if not params:
        return "?"
    match = re.match(r"(\d+\.?\d*)", params)
    if not match:
        return "?"
    size = float(match.group(1))
    # ~0.5 GB per billion params (Q4) + 2 GB overhead
    ram = size * 0.5 + 2
    return f"{ram:.0f}"


def _render_hardware_entry(result: Any) -> str:
    """Render a HARDWARE_PROFILES.md entry for tested hardware."""
    hw = result.hardware
    tps = f"{result.avg_tokens_per_second:.1f}"
    score = f"{result.overall_score:.1f}"
    date = result.timestamp.strftime("%Y-%m-%d")

    return (
        f"| {hw.chip_name} | {hw.ram_gb:.0f}GB | {result.model} "
        f"| {score}/100 | {tps} t/s | {date} |"
    )


def _render_decision_entry(result: Any) -> str:
    """Render a DECISION_MATRIX.md entry with benchmark data."""
    params = _extract_param_count(result.model) or "?"
    score = f"{result.overall_score:.0f}"
    tps = f"{result.avg_tokens_per_second:.0f}"
    rating = _score_rating(float(result.overall_score))
    hw = result.hardware

    return (
        f"| {result.model} | {params} | {hw.chip_name} ({hw.ram_gb:.0f}GB) "
        f"| {score}/100 ({rating}) | {tps} t/s |"
    )


def _render_summary_row(
    result: Any,
    requesting_project: str,
    report_relpath: str,
    scoring_config: Dict[str, Any],
) -> str:
    """Render a summary table row for EVALUATION_LOG.md."""
    ram = _estimate_ram(_extract_param_count(result.model))
    tps = f"{result.avg_tokens_per_second:.0f}"
    score = f"{result.overall_score:.0f}"
    date = result.timestamp.strftime("%Y-%m-%d")
    best_for = _determine_best_for(result, scoring_config)
    report_link = f"[report]({report_relpath})" if report_relpath else "—"
    project = requesting_project or "—"

    return (
        f"| {result.model} | {score} | {tps} | {ram} GB "
        f"| {best_for} | {project} | {date} | {report_link} |"
    )


def _render_log_entry(
    result: Any,
    requesting_project: str,
    use_case: str,
    report_relpath: str,
    scoring_config: Dict[str, Any],
) -> str:
    """Render a detailed history entry for EVALUATION_LOG.md."""
    date = result.timestamp.strftime("%Y-%m-%d")
    score = f"{result.overall_score:.1f}"
    rating = _score_rating(float(result.overall_score))
    tps = f"{result.avg_tokens_per_second:.1f}"
    ram = _estimate_ram(_extract_param_count(result.model))
    project = requesting_project or "AI_Eval"
    use_desc = use_case or "General evaluation"

    # Category scores
    text = _get_category_score(result, "text-generation")
    code = _get_category_score(result, "code-generation")
    analysis = _get_category_score(result, "document-analysis")
    structured = _get_category_score(result, "structured-output")
    chat = _get_category_score(result, "conversational")

    # Best fitness profiles
    category_scores = {
        name: float(cat.avg_score) for name, cat in result.categories.items() if cat.tests
    }
    profiles = scoring_config.get("fitness_profiles", {})
    fitness = _calculate_fitness_scores(category_scores, profiles)
    fitness_str = ", ".join(
        f"{name.replace('-', ' ').title()} {s:.1f}"
        for name, s in sorted(fitness.items(), key=lambda x: -x[1])
        if s >= 60
    )
    if not fitness_str:
        fitness_str = "None above 60"

    # Report links
    report_md = f"[Full report]({report_relpath})" if report_relpath else "—"
    data_relpath = (
        report_relpath.replace("reports/", "data/").replace(".md", ".json")
        if report_relpath
        else ""
    )
    data_md = f" | [Raw data]({data_relpath})" if data_relpath else ""

    lines = [
        f"#### {date}: {result.model}",
        f"- **Requested by:** {project} | **Use case:** {use_desc}",
        f"- **Overall:** {score}/100 ({rating}) | **TPS:** {tps} | **RAM:** {ram} GB",
        f"- **Category scores:** Text {text} | Code {code} | Analysis {analysis} | Structured {structured} | Chat {chat}",
        f"- **Fitness:** {fitness_str}",
        f"- **Report:** {report_md}{data_md}",
    ]

    return "\n".join(lines)


def _ensure_monthly_marker(file_path: Path, year_month: str) -> bool:
    """Ensure a monthly history marker section exists in EVALUATION_LOG.md.

    Creates the marker section if it doesn't exist yet.
    Returns True if markers are available (existing or newly created).
    """
    begin = f"<!-- AI_EVAL:BEGIN history-{year_month} -->"
    end = f"<!-- AI_EVAL:END history-{year_month} -->"

    if has_markers(file_path, begin, end):
        return True

    # Need to create the section — find insertion point
    content = file_path.read_text()

    # Look for "## Evaluation History" heading
    history_heading = "## Evaluation History"
    heading_idx = content.find(history_heading)
    if heading_idx == -1:
        logger.warning("Could not find '## Evaluation History' in EVALUATION_LOG.md")
        return False

    # Find the end of the heading line
    heading_end = content.find("\n", heading_idx)
    if heading_end == -1:
        heading_end = len(content)

    # Skip past any text after the heading until the first ### or marker
    insert_idx = heading_end + 1
    # Skip blank lines and description text
    while insert_idx < len(content):
        line_end = content.find("\n", insert_idx)
        if line_end == -1:
            line_end = len(content)
        line = content[insert_idx:line_end].strip()
        if line.startswith("###") or line.startswith("<!--") or line.startswith("---"):
            break
        insert_idx = line_end + 1

    # Insert the new monthly section before existing content
    new_section = (
        f"\n### {year_month}\n\n" f"{begin}\n" f"_(No evaluations recorded yet.)_\n" f"{end}\n"
    )

    updated = content[:insert_idx] + new_section + content[insert_idx:]
    file_path.write_text(updated)
    logger.info(f"Created monthly marker section: history-{year_month}")
    return True


def _update_log_history(
    file_path: Path,
    marker_name: str,
    new_entry: str,
) -> bool:
    """Append a history entry to a marker section (does not deduplicate)."""
    begin = f"<!-- AI_EVAL:BEGIN {marker_name} -->"
    end = f"<!-- AI_EVAL:END {marker_name} -->"

    if not has_markers(file_path, begin, end):
        return False

    existing = read_marker_content(file_path, begin, end) or ""

    # Remove placeholder text
    lines = existing.strip().split("\n") if existing.strip() else []
    filtered = [line for line in lines if not line.strip().startswith("_(")]

    # Append new entry
    if filtered:
        filtered.append("")  # blank line separator
    filtered.append(new_entry)

    new_content = "\n".join(filtered) + "\n"
    return replace_marker_content(file_path, "\n" + new_content, begin, end)


def _update_marker_section(
    file_path: Path,
    marker_name: str,
    new_row: str,
    table_header: Optional[str] = None,
) -> bool:
    """Update a named marker section, appending a row (replacing if same model exists)."""
    begin = f"<!-- AI_EVAL:BEGIN {marker_name} -->"
    end = f"<!-- AI_EVAL:END {marker_name} -->"

    if not has_markers(file_path, begin, end):
        logger.warning(f"Markers '{marker_name}' not found in {file_path}")
        return False

    existing = read_marker_content(file_path, begin, end) or ""

    # Parse model name from the new row (first cell after |)
    new_model_match = re.match(r"\|\s*([^|]+?)\s*\|", new_row)
    new_model = new_model_match.group(1).strip() if new_model_match else None

    # Filter out example rows (italic) and rows for same model
    lines = existing.strip().split("\n") if existing.strip() else []
    filtered = []
    for line in lines:
        # Skip example/placeholder rows
        if line.strip().startswith("| _"):
            continue
        # Skip header/separator rows (keep them)
        if line.strip().startswith("| Model") or line.strip().startswith("|---"):
            filtered.append(line)
            continue
        # Skip existing entry for same model
        if new_model:
            model_match = re.match(r"\|\s*([^|]+?)\s*\|", line)
            if model_match and model_match.group(1).strip() == new_model:
                continue
        filtered.append(line)

    # If no header exists and one was provided, add it
    if table_header and not any(
        line.strip().startswith("| Model") or line.strip().startswith("| Chip") for line in filtered
    ):
        header_lines = table_header.strip().split("\n")
        filtered = header_lines + filtered

    # Append new row
    filtered.append(new_row)

    new_content = "\n".join(filtered) + "\n"
    return replace_marker_content(file_path, "\n" + new_content, begin, end)


def export_to_catalog(
    result: Any,
    config: Optional[ExportConfig] = None,
    report_path: Optional[Path] = None,
    scoring_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, bool]:
    """
    Export benchmark results to _HQ/evaluations/ catalog files.

    Args:
        result: BenchmarkResult from the benchmark runner.
        config: Export configuration.
        report_path: Path to the generated report (for copying).
        scoring_config: Scoring config dict (fitness profiles, thresholds).

    Returns:
        Dict mapping file names to whether they were updated.
    """
    if config is None:
        config = ExportConfig()

    if scoring_config is None:
        scoring_config = {}

    outcomes: Dict[str, bool] = {
        "catalog": False,
        "matrix": False,
        "hardware": False,
        "reports": False,
        "log": False,
    }

    eval_path = config.evaluations_path
    if not eval_path.exists():
        logger.warning(f"Evaluations directory not found: {eval_path}")
        return outcomes

    try:
        # Copy report files
        if config.copy_reports and report_path and report_path.exists():
            reports_dir = eval_path / "reports"
            data_dir = eval_path / "data"
            reports_dir.mkdir(parents=True, exist_ok=True)
            data_dir.mkdir(parents=True, exist_ok=True)

            shutil.copy2(report_path, reports_dir / report_path.name)
            json_path = report_path.with_suffix(".json")
            if json_path.exists():
                shutil.copy2(json_path, data_dir / json_path.name)
            outcomes["reports"] = True
            logger.info(f"Copied reports to {eval_path}")

        # Determine report relative path for catalog links
        report_relpath = ""
        if report_path:
            report_relpath = f"reports/{report_path.name}"

        # Update MODEL_CATALOG.md
        catalog_path = eval_path / "MODEL_CATALOG.md"
        if config.update_catalog and catalog_path.exists():
            section = _classify_model(result.model, result.provider)
            row = _render_catalog_row(result, report_relpath, scoring_config)
            outcomes["catalog"] = _update_marker_section(catalog_path, section, row)
            if outcomes["catalog"]:
                logger.info(f"Updated MODEL_CATALOG.md section '{section}'")

        # Update DECISION_MATRIX.md
        matrix_path = eval_path / "DECISION_MATRIX.md"
        if config.update_matrix and matrix_path.exists():
            entry = _render_decision_entry(result)
            header = (
                "| Model | Params | Hardware | Score | Throughput |\n"
                "|-------|--------|----------|-------|------------|"
            )
            outcomes["matrix"] = _update_marker_section(
                matrix_path, "decision-matrix", entry, table_header=header
            )
            if outcomes["matrix"]:
                logger.info("Updated DECISION_MATRIX.md")

        # Update HARDWARE_PROFILES.md
        hw_path = eval_path / "HARDWARE_PROFILES.md"
        if config.update_hardware and hw_path.exists():
            hw_entry = _render_hardware_entry(result)
            header = (
                "| Chip | RAM | Model Tested | Score | Throughput | Date |\n"
                "|------|-----|-------------|-------|------------|------|"
            )
            outcomes["hardware"] = _update_marker_section(
                hw_path, "hardware-profiles", hw_entry, table_header=header
            )
            if outcomes["hardware"]:
                logger.info("Updated HARDWARE_PROFILES.md")

        # Update EVALUATION_LOG.md
        log_path = eval_path / "EVALUATION_LOG.md"
        if config.update_log and log_path.exists():
            # Update summary table (replace existing row for same model)
            summary_row = _render_summary_row(
                result, config.requesting_project, report_relpath, scoring_config
            )
            summary_header = (
                "| Model | Overall | TPS | RAM | Best Profile "
                "| Requested By | Date | Report |\n"
                "|-------|---------|-----|-----|-------------- "
                "|--------------|------|--------|"
            )
            _update_marker_section(
                log_path, "eval-summary", summary_row, table_header=summary_header
            )

            # Add history entry (append, don't replace)
            year_month = result.timestamp.strftime("%Y-%m")
            if _ensure_monthly_marker(log_path, year_month):
                log_entry = _render_log_entry(
                    result,
                    config.requesting_project,
                    config.use_case,
                    report_relpath,
                    scoring_config,
                )
                _update_log_history(log_path, f"history-{year_month}", log_entry)

            outcomes["log"] = True
            logger.info("Updated EVALUATION_LOG.md")

    except Exception as e:
        raise ExportError(f"Catalog export failed: {e}") from e

    return outcomes
