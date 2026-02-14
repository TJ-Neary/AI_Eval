# StartHere.md — AI_Eval Project Guide

> **Purpose**: This is the single entry point for understanding the AI_Eval project.
> Read this document first, then follow links to deeper documentation as needed.
> Designed for both human developers and LLM agents.

**AI_Eval** is an LLM evaluation and benchmarking framework that compares local models (Ollama) against cloud APIs (Google Gemini, Anthropic Claude, OpenAI GPT) across standardized test suites. It generates hardware-aware recommendations using three scoring methodologies (pass@k, LLM-as-Judge, RAG metrics) and produces fitness scores weighted to specific deployment use cases.

**Status**: Stable — all core subsystems complete. Provider layer, scoring engine, benchmark runner, hardware profiling, evaluation workflow, reporting, and catalog export are implemented.

**Quick orientation**: Start with the [README](./README.md) for features and installation, then review the [Architecture](#system-architecture) section below for the full component map.

**Last Updated**: 2026-02-14

---

## Document Map

### Primary References

| Document | Path | Purpose |
|----------|------|---------|
| README | [README.md](./README.md) | Project overview, installation, quick start |
| CLAUDE.md | [CLAUDE.md](./CLAUDE.md) | AI assistant guidance, architecture, commands |
| DevPlan | [DevPlan.md](./DevPlan.md) | Development phases, technical decisions (TD-001–TD-013), task tracker |
| Security | [SECURITY.md](./SECURITY.md) | Security policy, secrets management, incident response |

### Research & Specifications

| Document | Path | Purpose |
|----------|------|---------|
| Test Suite Spec | [docs/TEST_SUITE_SPEC.md](./docs/TEST_SUITE_SPEC.md) | 5 benchmark categories, metrics, weights |
| Research Synthesis | [docs/RESEARCH_SYNTHESIS.md](./docs/RESEARCH_SYNTHESIS.md) | Unified findings from multi-AI research |

> **Archived**: Raw research docs (Claude, ChatGPT, Gemini, RESEARCH_FINDINGS, BENCHMARK_RESEARCH) moved to `_archive/docs/` — findings fully synthesized into RESEARCH_SYNTHESIS.md and DevPlan.md.

### Configuration & Build

| File | Path | Purpose |
|------|------|---------|
| pyproject.toml | [pyproject.toml](./pyproject.toml) | Package config, pytest, black/ruff/mypy settings |
| requirements.txt | [requirements.txt](./requirements.txt) | Runtime dependencies |
| requirements-dev.txt | [requirements-dev.txt](./requirements-dev.txt) | Development dependencies |
| .pre-commit-config.yaml | [.pre-commit-config.yaml](./.pre-commit-config.yaml) | Pre-commit hooks (black, ruff, isort, mypy) |
| .gitignore | [.gitignore](./.gitignore) | Git exclusions |
| .env.example | [.env.example](./.env.example) | Environment variable template |
| default.yaml | [configs/default.yaml](./configs/default.yaml) | Evaluation config: models, suites, scoring |

### CI/CD

| File | Path | Purpose |
|------|------|---------|
| ci.yml | [.github/workflows/ci.yml](./.github/workflows/ci.yml) | GitHub Actions CI (lint, test, security, build) |
| release.yml | [.github/workflows/release.yml](./.github/workflows/release.yml) | GitHub Actions release pipeline |

### Security Utilities

| File | Path | Purpose |
|------|------|---------|
| security_scan.sh | [security_scan.sh](./security_scan.sh) | Pre-commit security scanner (secrets, PII, paths) |
| pii_terms.example.yaml | [pii_terms.example.yaml](./pii_terms.example.yaml) | PII term detection template |
| .security_terms.example | [.security_terms.example](./.security_terms.example) | Private term detection template |

---

## System Architecture

```
                              ┌─────────────────────────────────────────────────────┐
                              │                    CLI Entry Point                   │
                              │                     src/cli.py                       │
                              └─────────────────────────────────────────────────────┘
                                                       │
                    ┌──────────────────────────────────┼──────────────────────────────────┐
                    │                                  │                                  │
                    ▼                                  ▼                                  ▼
     ┌──────────────────────────┐    ┌──────────────────────────┐    ┌──────────────────────────┐
     │        Providers         │    │       Benchmarks         │    │        Profiling         │
     │    src/providers/        │    │    src/benchmarks/       │    │     src/profiling/       │
     ├──────────────────────────┤    ├──────────────────────────┤    ├──────────────────────────┤
     │ • OllamaProvider (local) │    │ • BenchmarkRunner        │    │ • detect_hardware()      │
     │ • GoogleProvider (API)   │    │ • DatasetManager         │    │ • HardwareProfile        │
     │ • [Anthropic] (extend)   │    │ • YAML test suites       │    │ • Apple Silicon tiers    │
     │ • [OpenAI] (extend)      │    │                          │    │ • NVIDIA/AMD detection   │
     └──────────────────────────┘    └──────────────────────────┘    └──────────────────────────┘
                    │                              │                              │
                    │                              ▼                              │
                    │         ┌──────────────────────────────────┐                │
                    │         │           Scoring                │                │
                    │         │        src/scoring/              │                │
                    │         ├──────────────────────────────────┤                │
                    │         │ • pass_k.py (HumanEval-style)    │                │
                    │         │ • llm_judge.py (bias mitigation) │                │
                    │         │ • rag_metrics.py (DeepEval)      │                │
                    │         └──────────────────────────────────┘                │
                    │                              │                              │
                    └──────────────────────────────┼──────────────────────────────┘
                                                   ▼
     ┌──────────────────────────┐    ┌──────────────────────────┐    ┌──────────────────────────┐
     │       Evaluation         │    │       Reporting          │    │         Export           │
     │    src/evaluation/       │    │    src/reporting/        │    │      src/export/         │
     ├──────────────────────────┤    ├──────────────────────────┤    ├──────────────────────────┤
     │ • Config (YAML loader)   │    │ • Jinja2 templates       │    │ • MODEL_CATALOG.md       │
     │ • EvaluationRunner       │    │ • Markdown/JSON output   │    │ • DECISION_MATRIX.md     │
     │ • Custom scorers         │    │ • README updater         │    │ • Marker-based updates   │
     │ • Model discovery        │    │                          │    │                          │
     └──────────────────────────┘    └──────────────────────────┘    └──────────────────────────┘

     ┌──────────────────────────┐
     │        Utilities         │
     │         utils/           │
     ├──────────────────────────┤
     │ ✓ logging_config.py      │
     │ ✓ exceptions.py          │
     │ ✓ retry.py               │
     │ ✓ marker_parser.py       │
     │ ○ rate_limiter.py        │
     │ ○ state_machine.py       │
     │ ○ plugin_loader.py       │
     └──────────────────────────┘
        ✓ = Wired  ○ = Reference implementations
```

---

## Source Code Map

### Core Modules (src/)

| Directory/File | Purpose | Key Classes/Functions | Status |
|----------------|---------|----------------------|--------|
| `src/cli.py` | CLI entry point | `cmd_quick_test()`, `cmd_run()`, `cmd_compare()`, `cmd_hardware()`, `cmd_models()`, `cmd_evaluate()`, `main()` | **Wired** |
| `src/__main__.py` | Package entry point | Imports and runs `main()` | **Wired** |

### Provider Layer (src/providers/)

| File | Purpose | Key Classes | Status |
|------|---------|-------------|--------|
| `base.py` | Abstract provider interface | `BaseProvider`, `ProviderFactory`, `GenerationConfig`, `GenerationResponse`, `GenerationMetrics`, `ModelInfo`, `Message`, `ProviderType` | **Wired** |
| `ollama_provider.py` | Local Ollama inference | `OllamaProvider` — async generate/chat with throughput metrics | **Wired** |
| `google_provider.py` | Google Gemini API | `GoogleProvider` — cloud inference with context window management | **Wired** |

### Benchmark Layer (src/benchmarks/)

| File | Purpose | Key Classes | Status |
|------|---------|-------------|--------|
| `runner.py` | Benchmark executor | `BenchmarkRunner`, `BenchmarkResult`, `TestResult`, `RunConfig`, `CategoryResult` | **Wired** |
| `datasets.py` | Dataset management | `Dataset`, `TestCase`, `DatasetManager`, `BenchmarkSuite`, `QUICK_TEST_DATASET` | **Wired** |

### Scoring Layer (src/scoring/)

| File | Purpose | Key Classes | Status |
|------|---------|-------------|--------|
| `pass_k.py` | HumanEval-style code eval | `calculate_pass_at_k()`, `evaluate_code_generation()`, `PassKResult`, `CodeExecutionResult` | **Wired** |
| `llm_judge.py` | LLM-as-Judge (TD-011) | `LLMJudge`, `JudgingCriteria`, `JudgingResult`, `JudgingConfig` — position shuffling, multi-eval consensus | **Wired** |
| `rag_metrics.py` | RAG evaluation (DeepEval) | `RAGEvaluator`, `RAGEvaluationResult`, `RAGTestCase` — AnswerRelevancy, Faithfulness, ContextualPrecision, Hallucination | **Wired** |

### Profiling Layer (src/profiling/)

| File | Purpose | Key Classes | Status |
|------|---------|-------------|--------|
| `hardware.py` | Hardware detection | `detect_hardware()`, `HardwareProfile`, `ChipType`, `AppleSiliconTier`, `get_memory_usage()`, `can_run_model_size()` | **Wired** |

### Evaluation Layer (src/evaluation/)

| File | Purpose | Key Classes | Status |
|------|---------|-------------|--------|
| `config.py` | YAML config loading | `EvalRequestConfig`, `Scenario`, `AcceptanceCriterion`, `ScorerConfig` | **Wired** |
| `runner.py` | Evaluation orchestration | `EvaluationRunner` — candidate testing, acceptance criteria, result aggregation | **Wired** |
| `scorers.py` | Custom scorers | `JsonValidityScorer`, `LatencyScorer`, `PatternAccuracyScorer`, `CitationAccuracyScorer`, `VerbatimQuoteScorer` | **Wired** |
| `model_discovery.py` | Model catalog management | `ModelCatalog`, `ModelEntry` — Ollama integration, version detection, classification | **Wired** |
| `report.py` | Evaluation report generator | `EvaluationReportGenerator` — Jinja2-based evaluation reports | **Wired** |

### Reporting Layer (src/reporting/)

| File | Purpose | Key Classes | Status |
|------|---------|-------------|--------|
| `report_generator.py` | Jinja2 report generation | `ReportGenerator` — markdown/JSON benchmark reports | **Wired** |
| `readme_updater.py` | README results table updater | `ReadmeUpdater` — marker-based results table updates | **Wired** |
| `templates/` | Jinja2 templates | Benchmark and evaluation report templates | **Wired** |

### Export Layer (src/export/)

| File | Purpose | Key Classes | Status |
|------|---------|-------------|--------|
| `catalog_exporter.py` | Catalog export | `CatalogExporter` — marker-based updates to MODEL_CATALOG, DECISION_MATRIX, HARDWARE_PROFILES | **Wired** |

### Utilities (utils/)

| File | Purpose | Key Classes | Status |
|------|---------|-------------|--------|
| `exceptions.py` | Exception hierarchy | `AiEvalError`, `ConfigError`, `ProcessingError`, `DatabaseError` | **Wired** |
| `logging_config.py` | Structured logging | `setup_logging()`, `get_logger()`, `LogContext`, `DebugTimer`, `log_performance()` | **Wired** |
| `retry.py` | Retry with circuit breaker | `retry_with_backoff()`, `RetryBudget`, `CircuitBreaker`, `RetryStrategies` | **Wired** |
| `rate_limiter.py` | Sliding window rate limiter | `RateLimiter` — per-operation API throttling | **Orphaned** |
| `state_machine.py` | Async state machine | `StateMachine[S]`, `StateTransition` — workflow management | **Orphaned** |
| `plugin_loader.py` | Dynamic plugin system | `PluginLoader`, `BasePlugin` — auto-discovery | **Orphaned** |

### Configuration

| File | Purpose | Status |
|------|---------|--------|
| `config.py` | Config loader | `get_path_var()`, `STATE_DIR`, env var loading via python-dotenv | **Wired** |
| `configs/default.yaml` | Evaluation config | Models, suites, parameters, scoring thresholds, fitness profiles | **Wired** |

### Tests (tests/)

| File | Purpose | Status |
|------|---------|--------|
| `conftest.py` | Shared pytest fixtures | `tmp_project_dir`, `mock_env`, `sample_text`, `sample_file` | **Wired** |
| `test_tdd_example.py` | TDD methodology demo | Demonstrates Red-Green-Refactor cycle | **Documentation** |
| `test_eval_config.py` | Evaluation config tests | 15 tests for YAML loading, validation | **Wired** |
| `test_eval_runner.py` | Evaluation runner tests | 12 tests for orchestration, criteria | **Wired** |
| `test_eval_scorers.py` | Custom scorer tests | 32 tests for all 5 scorers | **Wired** |
| `test_eval_report.py` | Eval report tests | 6 tests for report generation | **Wired** |
| `test_model_discovery.py` | Model discovery tests | 31 tests for catalog, classification | **Wired** |
| `test_catalog_exporter.py` | Export tests | 45 tests for marker-based catalog export | **Wired** |
| `test_report_generator.py` | Report tests | 21 tests for Jinja2 reporting | **Wired** |
| `test_readme_updater.py` | README updater tests | 16 tests for results table updates | **Wired** |
| `test_marker_parser.py` | Marker parser tests | 17 tests for marker matching | **Wired** |

---

## Known Issues & Discrepancies

| Issue | Location | Description | Status |
|-------|----------|-------------|--------|
| Reference utilities | `utils/rate_limiter.py`, `state_machine.py`, `plugin_loader.py` | Implemented but not wired into main codebase | Documented as reference implementations |
| Extensible providers | `ProviderType` enum | ANTHROPIC, OPENAI defined but not implemented | Architecture supports extension via `BaseProvider` |

---

## Development Status

All core subsystems are complete. The framework is stable and suitable for production evaluation workflows.

| Component | Status | Notes |
|-----------|--------|-------|
| Provider abstraction | Complete | BaseProvider + Factory pattern |
| Ollama provider | Complete | Async generate/chat with metrics |
| Google provider | Complete | Gemini API integration |
| Scoring — pass@k | Complete | HumanEval methodology |
| Scoring — LLM-as-Judge | Complete | TD-011 bias mitigation |
| Scoring — RAG metrics | Complete | DeepEval integration |
| Hardware detection | Complete | Apple Silicon, NVIDIA, AMD, CPU |
| Benchmark runner | Complete | Warmup, concurrent execution |
| Evaluation workflow | Complete | Config-driven runner, custom scorers, model discovery |
| CLI | Complete | 7 commands: run, quick-test, compare, list-models, hardware, models, evaluate |
| Reporting | Complete | Jinja2 markdown/JSON reports, README results table |
| Catalog export | Complete | Marker-based updates to MODEL_CATALOG, DECISION_MATRIX, HARDWARE_PROFILES |

---

## Roadmap

Core development is complete. The architecture is extensible for future providers and scoring methods. See [DevPlan.md](./DevPlan.md) for the full development history and technical decisions (TD-001 through TD-013).

---

## Quick Reference

### Setup & Run

```bash
# Environment setup
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt      # For development
cp .env.example .env                      # Add API keys

# Run tests
pytest                                    # All tests with coverage
pytest -m "not slow"                      # Skip slow tests
pytest -m "not integration"              # Skip integration tests

# Code quality
black src/ tests/ --line-length 100
ruff check src/ tests/
mypy src/

# Pre-commit hooks (one-time setup)
pre-commit install
```

### CLI Commands

```bash
# Quick sanity test
python -m src quick-test --provider ollama --model qwen2.5:7b

# Full benchmark
python -m src run --config configs/default.yaml

# Compare models
python -m src compare --models "qwen2.5:7b,llama3:8b" --provider ollama

# List available models
python -m src list-models --provider ollama

# Show hardware profile
python -m src hardware

# Manage model catalog
python -m src models --discover --provider ollama

# Run config-driven evaluation
python -m src evaluate --config evaluations/requests/my-request.yaml
```

### Key Entry Points

| Entry Point | Transport | Module |
|-------------|-----------|--------|
| CLI | Terminal | `src/cli.py` → `main()` |
| Package | Python import | `src/__main__.py` |

### Key Configuration Files

| File | Location | Gitignored |
|------|----------|------------|
| .env | Project root | Yes |
| configs/default.yaml | configs/ | No |
| pii_terms.yaml | ~/.ai_eval/ | Yes (user-specific) |
| .security_terms | Project root | Yes |

### Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `GOOGLE_API_KEY` | — | Google Gemini API key |
| `ANTHROPIC_API_KEY` | — | Anthropic Claude API key |
| `OPENAI_API_KEY` | — | OpenAI GPT API key |
| `DATA_DIR` | ~/.ai_eval | State directory |
| `LOG_LEVEL` | INFO | Logging verbosity |
| `DEBUG` | false | Debug mode |

---

## Technology Stack

| Component | Technology | Version/Detail |
|-----------|------------|----------------|
| Language | Python | 3.12+ |
| Async | asyncio | Native |
| Local LLM | Ollama | Via `ollama` SDK |
| Cloud APIs | google-genai, anthropic, openai | Latest |
| RAG Metrics | DeepEval | RAGAS integration |
| Hardware | psutil | Cross-platform |
| Templates | Jinja2 | Report generation |
| CLI Output | rich, tabulate | Colored terminal |
| HTTP | httpx | Async HTTP client |
| Testing | pytest | pytest-asyncio, pytest-cov |
| Formatting | black | 100 char line length |
| Linting | ruff | Fast Python linter |
| Types | mypy | Static type checking |
| Security | bandit, safety, pip-audit | Vulnerability scanning |

---

## Runtime & Generated Files

Files created during execution (gitignored):

| File/Directory | Purpose |
|----------------|---------|
| `.env` | API keys and local config |
| `logs/` | Application logs |
| `*.log` | Log files |
| `.coverage`, `htmlcov/`, `coverage.xml` | Coverage reports |
| `.pytest_cache/` | Pytest cache |
| `build/`, `dist/`, `*.egg-info/` | Build artifacts |
| `temp/`, `tmp/` | Temporary files |
| `__pycache__/` | Python bytecode |
| `~/.ai_eval/` | State directory (outside project) |

---

## Template System Integration

This project uses a shared template and standards system for cross-project consistency. Standards, guides, and templates are synced from a central repository.

### Synced Assets

| Asset | Version | Purpose |
|-------|---------|---------|
| `standards/TDD.md` | v2 | Test-driven development standard (mandatory) |
| `guides/universal/TESTING.md` | v3 | Testing strategy with TDD and mutation testing |
| `templates/tests-skeleton/test_tdd_example.py.j2` | v1 | TDD example test template |
| `security_scan.sh` | v2 | Pre-commit security scanner |

### Evaluation Outputs

AI_Eval generates and maintains the following shared evaluation files:

| File | Purpose |
|------|---------|
| `MODEL_CATALOG.md` | LLM model catalog with benchmark results |
| `HARDWARE_PROFILES.md` | Hardware reference configurations |
| `DECISION_MATRIX.md` | Local vs API decision framework |

---

*Updated 2026-02-14*
