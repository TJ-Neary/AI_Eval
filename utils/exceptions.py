"""
Custom exception hierarchy for AI_Eval.

All project-specific exceptions inherit from AiEvalError.
"""


class AiEvalError(Exception):
    """Base exception for AI_Eval."""

    pass


class ConfigError(AiEvalError):
    """Invalid or missing configuration."""

    pass


class ProcessingError(AiEvalError):
    """Error during data processing."""

    pass


class DatabaseError(AiEvalError):
    """Error with database operations."""

    pass


class ReportingError(AiEvalError):
    """Error during report generation or export."""

    pass


class ExportError(AiEvalError):
    """Error during catalog export to _HQ."""

    pass
