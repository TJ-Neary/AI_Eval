"""
Benchmark report generator.

Takes BenchmarkResult and renders markdown/JSON reports using Jinja2 templates.
Calculates fitness profile scores from config weights.
"""

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from jinja2 import Environment, FileSystemLoader

from utils.exceptions import ReportingError

logger = logging.getLogger(__name__)

TEMPLATE_DIR = Path(__file__).parent / "templates"
DEFAULT_CONFIG_PATH = Path(__file__).parents[2] / "configs" / "default.yaml"


@dataclass
class ReportConfig:
    """Configuration for report generation."""

    report_dir: Path = Path("./reports")
    formats: List[str] = field(default_factory=lambda: ["markdown", "json"])
    config_path: Path = DEFAULT_CONFIG_PATH


def _sanitize_model_name(model: str) -> str:
    """Convert model name to safe filename (colons/slashes to underscores)."""
    return re.sub(r"[:/\\]", "_", model)


def _score_rating(score: float, thresholds: Optional[Dict[str, int]] = None) -> str:
    """Map a numeric score to a rating string."""
    if thresholds is None:
        thresholds = {"excellent": 90, "good": 75, "adequate": 60, "marginal": 40}

    if score >= thresholds["excellent"]:
        return "Excellent"
    elif score >= thresholds["good"]:
        return "Good"
    elif score >= thresholds["adequate"]:
        return "Adequate"
    elif score >= thresholds["marginal"]:
        return "Marginal"
    else:
        return "Poor"


def _load_scoring_config(config_path: Path) -> Dict[str, Any]:
    """Load scoring thresholds and fitness profiles from config YAML."""
    if not config_path.exists():
        return {}
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config.get("scoring", {})


def _calculate_fitness_scores(
    category_scores: Dict[str, float],
    fitness_profiles: Dict[str, Dict[str, float]],
) -> Dict[str, float]:
    """Calculate fitness profile scores from category scores and profile weights."""
    results = {}
    for profile_name, weights in fitness_profiles.items():
        total_weight = 0.0
        weighted_score = 0.0
        for category, weight in weights.items():
            if category in category_scores:
                weighted_score += category_scores[category] * weight
                total_weight += weight
        results[profile_name] = weighted_score / total_weight if total_weight > 0 else 0.0
    return results


class ReportGenerator:
    """Generates benchmark reports from BenchmarkResult data."""

    def __init__(self, config: Optional[ReportConfig] = None):
        self.config = config or ReportConfig()
        self._env = Environment(
            loader=FileSystemLoader(str(TEMPLATE_DIR)),
            trim_blocks=True,
            lstrip_blocks=True,
        )
        self._scoring = _load_scoring_config(self.config.config_path)
        self._thresholds = {
            "excellent": self._scoring.get("excellent", 90),
            "good": self._scoring.get("good", 75),
            "adequate": self._scoring.get("adequate", 60),
            "marginal": self._scoring.get("marginal", 40),
        }

    def generate(self, result: Any) -> Path:
        """
        Generate reports from a BenchmarkResult.

        Args:
            result: BenchmarkResult from the benchmark runner.

        Returns:
            Path to the generated markdown report.

        Raises:
            ReportingError: If report generation fails.
        """
        try:
            self.config.report_dir.mkdir(parents=True, exist_ok=True)

            # Build template context
            context = self._build_context(result)

            # Generate file stem
            safe_model = _sanitize_model_name(result.model)
            timestamp_str = result.timestamp.strftime("%Y%m%d_%H%M%S")
            stem = f"{safe_model}_{timestamp_str}"

            # Render markdown
            md_path = self.config.report_dir / f"{stem}.md"
            if "markdown" in self.config.formats:
                template = self._env.get_template("benchmark_report.md.j2")
                rendered = template.render(**context)
                md_path.write_text(rendered)
                logger.info(f"Markdown report saved to {md_path}")

            # Save JSON
            json_path = self.config.report_dir / f"{stem}.json"
            if "json" in self.config.formats:
                json_data = result.to_dict()
                json_data["fitness_profiles"] = {
                    p["name"]: p["score"] for p in context["fitness_profiles"]
                }
                with open(json_path, "w") as f:
                    json.dump(json_data, f, indent=2, default=str)
                logger.info(f"JSON report saved to {json_path}")

            return md_path

        except Exception as e:
            raise ReportingError(f"Failed to generate report: {e}") from e

    def _build_context(self, result: Any) -> Dict[str, Any]:
        """Build the Jinja2 template context from a BenchmarkResult."""
        # Category data
        category_scores: Dict[str, float] = {}
        categories = []
        for cat_name, cat_result in sorted(result.categories.items()):
            score = float(cat_result.avg_score)
            category_scores[cat_name] = score
            categories.append({
                "name": cat_name,
                "score": f"{score:.1f}",
                "passed": cat_result.passed,
                "total": cat_result.total_tests,
                "pass_rate": f"{cat_result.pass_rate * 100:.0f}",
                "tokens_per_second": f"{cat_result.avg_tokens_per_second:.1f}",
                "rating": _score_rating(score, self._thresholds),
                "tests": [
                    {
                        "id": t.test_id,
                        "status": t.status.name,
                        "score": f"{t.score:.1f}" if t.score is not None else "N/A",
                        "time_ms": f"{t.generation_time_ms:.0f}",
                        "tokens_per_second": f"{t.tokens_per_second:.1f}",
                    }
                    for t in cat_result.tests
                ],
            })

        # Fitness profiles
        fitness_weights = self._scoring.get("fitness_profiles", {})
        fitness_scores = _calculate_fitness_scores(category_scores, fitness_weights)
        fitness_profiles = [
            {
                "name": name.replace("-", " ").title(),
                "score": f"{score:.1f}",
                "rating": _score_rating(score, self._thresholds),
            }
            for name, score in sorted(fitness_scores.items())
        ]

        # Hardware dict
        hw = result.hardware
        hardware = {
            "chip_name": hw.chip_name or "Unknown",
            "chip_type": hw.chip_type.name,
            "ram_gb": f"{hw.ram_gb:.0f}",
            "gpu_cores": hw.gpu_cores,
            "memory_bandwidth_gbps": f"{hw.memory_bandwidth_gbps:.0f}",
            "os": f"{hw.os_name} {hw.os_version}",
        }

        overall = float(result.overall_score)

        return {
            "model": result.model,
            "provider": result.provider,
            "timestamp": result.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "duration": f"{result.duration_seconds:.1f}",
            "hardware": hardware,
            "overall_score": f"{overall:.1f}",
            "overall_rating": _score_rating(overall, self._thresholds),
            "avg_tokens_per_second": f"{result.avg_tokens_per_second:.1f}",
            "total_passed": result.total_passed,
            "total_tests": result.total_tests,
            "total_tokens": result.total_tokens,
            "categories": categories,
            "fitness_profiles": fitness_profiles,
            "config": {
                "temperature": result.config.temperature,
                "max_tokens": result.config.max_tokens,
                "warmup_queries": result.config.warmup_queries,
                "repetitions": result.config.repetitions,
                "timeout_seconds": result.config.timeout_seconds,
                "use_llm_judge": result.config.use_llm_judge,
            },
            "version": "0.1.0",
        }
