"""
Centralized configuration for AI_Eval.

Loads environment variables from .env and provides validated paths and settings.
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def get_path_var(var_name: str, default: str | None = None, required: bool = True) -> Path | None:
    """Retrieve a path from environment variables, resolving to absolute."""
    value = os.getenv(var_name, default)
    if not value:
        if required:
            print(f"Error: Missing required environment variable '{var_name}' in .env file.")
            sys.exit(1)
        return None
    return Path(value).expanduser().resolve()


# -- Paths -------------------------------------------------------------------


STATE_DIR = Path(os.getenv("AI_EVAL_STATE_DIR", str(Path.home() / ".ai_eval")))

# -- Settings -----------------------------------------------------------------

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
DEBUG = os.getenv("DEBUG", "false").lower() in ("true", "1", "yes")


def validate_config() -> None:
    """Validate that critical paths exist or can be created."""
    for path_var in [STATE_DIR]:
        if not path_var.exists():
            try:
                path_var.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                print(f"Warning: Could not create {path_var}: {e}")
