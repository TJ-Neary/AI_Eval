"""
Evaluation Report Generator

Generates evaluation-specific reports with acceptance criteria results,
candidate comparisons, and recommendations. Extends the existing
ReportGenerator infrastructure.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from jinja2 import Environment, FileSystemLoader

from ..reporting.report_generator import TEMPLATE_DIR, ReportConfig, _sanitize_model_name
from .runner import EvaluationResult

logger = logging.getLogger(__name__)


class EvaluationReportGenerator:
    """Generates evaluation reports with acceptance criteria and recommendations."""

    def __init__(self, config: Optional[ReportConfig] = None) -> None:
        self.config = config or ReportConfig()
        self._env = Environment(
            loader=FileSystemLoader(str(TEMPLATE_DIR)),
            trim_blocks=True,
            lstrip_blocks=True,
            autoescape=True,
        )

    def generate(self, result: EvaluationResult) -> Path:
        """Generate evaluation report.

        Returns:
            Path to the generated markdown report.
        """
        self.config.report_dir.mkdir(parents=True, exist_ok=True)

        context = self._build_context(result)
        timestamp_str = result.timestamp.strftime("%Y%m%d_%H%M%S")
        stem = f"eval_{_sanitize_model_name(result.request_config.request_id)}_{timestamp_str}"

        # Render markdown
        md_path = self.config.report_dir / f"{stem}.md"
        if "markdown" in self.config.formats:
            template = self._env.get_template("evaluation_report.md.j2")
            rendered = template.render(**context)
            md_path.write_text(rendered)
            logger.info("Evaluation report saved to %s", md_path)

        # Save JSON
        json_path = self.config.report_dir / f"{stem}.json"
        if "json" in self.config.formats:
            with open(json_path, "w") as f:
                json.dump(result.to_dict(), f, indent=2, default=str)
            logger.info("Evaluation JSON saved to %s", json_path)

        return md_path

    def _build_context(self, result: EvaluationResult) -> Dict[str, Any]:
        """Build Jinja2 template context from EvaluationResult."""
        config = result.request_config

        # Build model results for template
        model_results = []
        for mr in result.model_results:
            model_data: Dict[str, Any] = {
                "model": mr.model,
                "provider": mr.provider,
                "skipped": mr.skipped,
                "skip_reason": mr.skip_reason,
            }

            if not mr.skipped:
                model_data.update(
                    {
                        "overall_score": f"{mr.benchmark_result.overall_score:.1f}",
                        "tps": f"{mr.benchmark_result.avg_tokens_per_second:.1f}",
                        "criteria_passed": mr.criteria_passed_count,
                        "criteria_total": mr.criteria_total,
                        "verdict": (
                            "RECOMMENDED"
                            if mr.model == result.recommended_model
                            else ("PASS" if mr.all_criteria_passed else "FAIL")
                        ),
                        "criteria": [
                            {
                                "name": cr.criterion.name,
                                "operator": cr.criterion.operator,
                                "threshold": cr.criterion.threshold,
                                "measured": f"{cr.measured_value:.3f}",
                                "result": "PASS" if cr.passed else "FAIL",
                            }
                            for cr in mr.criterion_results
                        ],
                    }
                )

            model_results.append(model_data)

        # Hardware
        hw = result.hardware
        hardware = {}
        if hw:
            hardware = {
                "chip_name": hw.chip_name or "Unknown",
                "ram_gb": f"{hw.ram_gb:.0f}",
                "os": f"{hw.os_name} {hw.os_version}" if hasattr(hw, "os_name") else "Unknown",
            }

        return {
            "request_id": config.request_id,
            "requesting_project": config.requesting_project,
            "use_case": config.use_case,
            "model_capability": config.model_capability,
            "timestamp": result.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "duration": f"{result.duration_seconds:.1f}",
            "recommended_model": result.recommended_model,
            "recommendation_reason": result.recommendation_reason,
            "model_results": model_results,
            "task_description": config.task_description,
            "input_description": config.input_description,
            "output_description": config.output_description,
            "complexity": config.complexity,
            "current_model": config.current_model,
            "hardware": hardware,
            "version": "0.1.0",
        }
