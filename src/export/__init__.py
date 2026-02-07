"""
Catalog export module.

Pushes benchmark results to _HQ/evaluations/ catalog files,
updating MODEL_CATALOG.md, DECISION_MATRIX.md, and HARDWARE_PROFILES.md
via marker-based content replacement.
"""

from .catalog_exporter import ExportConfig, export_to_catalog

__all__ = ["export_to_catalog", "ExportConfig"]
