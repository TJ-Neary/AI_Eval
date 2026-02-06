"""
AI_Eval — LLM Evaluation and Benchmarking

A comprehensive tool for evaluating local and API-based language models
across standardized benchmarks.

Main components:
- providers: Unified interface for LLM backends (Ollama, Google, Anthropic, OpenAI)
- scoring: Evaluation metrics (pass@k, LLM-as-Judge, RAG metrics)
- benchmarks: Test suites and dataset management
- profiling: Hardware detection and performance monitoring
- reporting: Report generation with Jinja2 templates
"""

__version__ = "0.1.0"
