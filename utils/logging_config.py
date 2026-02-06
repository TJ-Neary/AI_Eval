"""
Logging configuration for AI_Eval.

Provides structured logging with multiple outputs:
- Colored console output (human-readable)
- Rotating file log (human-readable)
- JSON structured log (machine-parseable)
- Error-only log (quick problem identification)
"""

import json
import logging
import logging.handlers
import os
import sys
import traceback
import functools
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional


class StructuredFormatter(logging.Formatter):
    """JSON-structured log formatter for machine parsing."""

    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": traceback.format_exception(*record.exc_info),
            }
        if hasattr(record, "extra_data"):
            log_data["data"] = record.extra_data
        return json.dumps(log_data)


class ColoredFormatter(logging.Formatter):
    """Colored console formatter for human readability."""

    COLORS = {
        "DEBUG": "\033[36m",     # Cyan
        "INFO": "\033[32m",      # Green
        "WARNING": "\033[33m",   # Yellow
        "ERROR": "\033[31m",     # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, "")
        record.levelname = f"{color}{record.levelname}{self.RESET}"
        return super().format(record)


def setup_logging(
    level: str = "INFO",
    log_dir: Optional[Path] = None,
    console: bool = True,
    json_logs: bool = True,
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 5,
) -> logging.Logger:
    """
    Set up logging configuration.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        log_dir: Directory for log files
        console: Enable console output
        json_logs: Enable JSON structured logs
        max_bytes: Max size per log file
        backup_count: Number of backup files to keep

    Returns:
        Root logger for the project
    """
    log_dir = log_dir or Path.home() / ".ai_eval" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    root_logger = logging.getLogger("ai_eval")
    root_logger.setLevel(getattr(logging, level.upper()))
    root_logger.handlers.clear()

    # Console handler with colors
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        if sys.stdout.isatty():
            fmt = ColoredFormatter(
                "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                datefmt="%H:%M:%S",
            )
        else:
            fmt = logging.Formatter(
                "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                datefmt="%H:%M:%S",
            )
        console_handler.setFormatter(fmt)
        root_logger.addHandler(console_handler)

    # Rotating file handler (human-readable)
    file_handler = logging.handlers.RotatingFileHandler(
        log_dir / "ai_eval.log",
        maxBytes=max_bytes,
        backupCount=backup_count,
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(module)s:%(lineno)d | %(message)s"
        )
    )
    root_logger.addHandler(file_handler)

    # JSON structured logs
    if json_logs:
        json_handler = logging.handlers.RotatingFileHandler(
            log_dir / "ai_eval.json.log",
            maxBytes=max_bytes,
            backupCount=backup_count,
        )
        json_handler.setLevel(logging.DEBUG)
        json_handler.setFormatter(StructuredFormatter())
        root_logger.addHandler(json_handler)

    # Error-only log
    error_handler = logging.handlers.RotatingFileHandler(
        log_dir / "ai_eval.error.log",
        maxBytes=max_bytes,
        backupCount=backup_count,
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(module)s:%(lineno)d\n"
            "%(message)s\n---"
        )
    )
    root_logger.addHandler(error_handler)

    return root_logger


class LogContext:
    """Context manager for adding structured context to log records."""

    def __init__(self, logger: logging.Logger, **context: Any):
        self.logger = logger
        self.context = context
        self._old_factory = None

    def __enter__(self):
        self._old_factory = logging.getLogRecordFactory()
        context = self.context

        def record_factory(*args, **kwargs):
            record = self._old_factory(*args, **kwargs)
            record.extra_data = context
            return record

        logging.setLogRecordFactory(record_factory)
        return self

    def __exit__(self, *args):
        logging.setLogRecordFactory(self._old_factory)


def log_performance(logger: Optional[logging.Logger] = None):
    """Decorator to log function execution time."""

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            log = logger or logging.getLogger(func.__module__)
            start = time.time()
            try:
                result = func(*args, **kwargs)
                elapsed = (time.time() - start) * 1000
                log.debug(f"{func.__name__} completed in {elapsed:.1f}ms")
                return result
            except Exception as e:
                elapsed = (time.time() - start) * 1000
                log.error(f"{func.__name__} failed after {elapsed:.1f}ms: {e}", exc_info=True)
                raise

        return wrapper

    return decorator


class DebugTimer:
    """Context manager for timing code blocks with optional checkpoints."""

    def __init__(self, name: str, logger: Optional[logging.Logger] = None):
        self.name = name
        self.logger = logger or logging.getLogger("ai_eval.debug")
        self.start_time = None
        self.checkpoints: list = []

    def __enter__(self):
        self.start_time = time.time()
        return self

    def checkpoint(self, name: str):
        elapsed = time.time() - self.start_time
        self.checkpoints.append((name, elapsed))
        self.logger.debug(f"[{self.name}] {name}: {elapsed * 1000:.1f}ms")

    def __exit__(self, *args):
        elapsed = time.time() - self.start_time
        self.logger.debug(f"[{self.name}] Total: {elapsed * 1000:.1f}ms")


def get_logger(name: str) -> logging.Logger:
    """Get a logger for a module."""
    return logging.getLogger(f"ai_eval.{name}")