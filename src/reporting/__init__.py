"""
Benchmark reporting module.

Generates markdown/JSON reports from benchmark results and updates
README.md with accumulated results.
"""

from .report_generator import ReportGenerator, ReportConfig
from .readme_updater import update_readme_results

__all__ = ["ReportGenerator", "ReportConfig", "update_readme_results"]
