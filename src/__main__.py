"""
Entry point for running AI_Eval as a module.

Usage:
    python -m src quick-test --model qwen2.5:32b
    python -m src run --model qwen2.5:32b
    python -m src hardware
"""

from .cli import main

if __name__ == "__main__":
    exit(main())
