"""
Benchmark reporting module.

Generates markdown/JSON reports from benchmark results and updates
README.md with accumulated results.
"""

from .readme_updater import update_readme_results
from .report_generator import ReportConfig, ReportGenerator

__all__ = ["ReportGenerator", "ReportConfig", "update_readme_results"]
