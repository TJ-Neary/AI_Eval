# StartHere.md — AI_Eval Project Guide

> **Purpose**: This is the single entry point for understanding the AI_Eval project.
> Read this document first, then follow links to deeper documentation as needed.
> Designed for both human developers and LLM agents.

**AI_Eval** is an LLM evaluation and benchmarking tool that compares local models (Ollama) against cloud APIs (Google Gemini, Anthropic Claude, OpenAI GPT) across standardized test suites. It generates hardware-aware recommendations and exports results to `~/Tech_Projects/_HQ/evaluations/` for cross-project model selection.

**Last Updated**: 2026-02-05

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
     │ • [Anthropic] (planned)  │    │ • YAML test suites       │    │ • Apple Silicon tiers    │
     │ • [OpenAI] (planned)     │    │                          │    │ • NVIDIA/AMD detection   │
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
     │       Reporting          │    │         Export           │    │        Utilities         │
     │    src/reporting/        │    │      src/export/         │    │         utils/           │
     │      [PLANNED]           │    │      [PLANNED]           │    ├──────────────────────────┤
     │ • Jinja2 templates       │    │ • MODEL_CATALOG.md       │    │ ✓ logging_config.py      │
     │ • Markdown/JSON output   │    │ • DECISION_MATRIX.md     │    │ ✓ exceptions.py          │
     │                          │    │ • Marker-based updates   │    │ ✓ retry.py               │
     └──────────────────────────┘    └──────────────────────────┘    │ ○ rate_limiter.py        │
                                                                     │ ○ code_validator.py      │
                                                                     │ ○ state_machine.py       │
                                                                     │ ○ path_guard.py          │
                                                                     │ ○ plugin_loader.py       │
                                                                     └──────────────────────────┘
                                                                        ✓ = Wired  ○ = Orphaned
```

---

## Source Code Map

### Core Modules (src/)

| Directory/File | Purpose | Key Classes/Functions | Status |
|----------------|---------|----------------------|--------|
| `src/cli.py` | CLI entry point | `cmd_quick_test()`, `cmd_run()`, `cmd_compare()`, `cmd_hardware()`, `main()` | **Wired** |
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

### Planned Modules

| Directory | Purpose | Status |
|-----------|---------|--------|
| `src/reporting/` | Jinja2 report generation (markdown/JSON) | **Planned** (directory exists, empty) |
| `src/export/` | Export to `_HQ/evaluations/` with markers | **Planned** (directory exists, empty) |

### Utilities (utils/)

| File | Purpose | Key Classes | Status |
|------|---------|-------------|--------|
| `exceptions.py` | Exception hierarchy | `AiEvalError`, `ConfigError`, `ProcessingError`, `DatabaseError` | **Wired** |
| `logging_config.py` | Structured logging | `setup_logging()`, `get_logger()`, `LogContext`, `DebugTimer`, `log_performance()` | **Wired** |
| `retry.py` | Retry with circuit breaker | `retry_with_backoff()`, `RetryBudget`, `CircuitBreaker`, `RetryStrategies` | **Wired** |
| `rate_limiter.py` | Sliding window rate limiter | `RateLimiter` — per-operation API throttling | **Orphaned** |
| `code_validator.py` | AST-based code safety | `CodeValidator`, `validate_code()` — blocked imports/calls | **Orphaned** |
| `state_machine.py` | Async state machine | `StateMachine[S]`, `StateTransition` — workflow management | **Orphaned** |
| `path_guard.py` | Filesystem sandbox | `PathGuard` — write boundary enforcement | **Orphaned** |
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
| `test_example.py` | Placeholder tests | `TestProjectSetup`, `TestExampleUnit` — to be replaced | **Placeholder** |

---

## Known Issues & Discrepancies

| Issue | Location | Description | Recommendation |
|-------|----------|-------------|----------------|
| Orphaned utilities | `utils/rate_limiter.py`, `utils/code_validator.py`, `utils/state_machine.py`, `utils/path_guard.py`, `utils/plugin_loader.py` | Implemented but never imported | Integrate into providers (rate limiting), pass_k (code validation), runner (state machine), or document as optional |
| Empty directories | `src/reporting/`, `src/export/` | Directories exist but contain no code | Implement per DevPlan Phase 3 |
| Hardcoded dataset | `src/cli.py:108` | TODO comment — loads `QUICK_TEST_DATASET` instead of config | Load from `RunConfig` or `configs/default.yaml` |
| Placeholder tests | `tests/test_example.py` | Example tests only | Add unit tests for providers, scoring, benchmarks |
| Missing providers | `ProviderType` enum | ANTHROPIC, OPENAI defined but not implemented | Implement in Phase 2 |

---

## Development Status

| Phase | Target | Key Deliverables | Status |
|-------|--------|------------------|--------|
| **MVP** | Personal Mac, RAG testing | Provider layer, scoring module, benchmark runner, hardware detection | **In Progress** |
| **Phase 2** | Multi-provider | Anthropic + OpenAI providers, reporting, export | Planned |
| **Phase 3** | Family machines | Windows support, NVIDIA/AMD optimization | Planned |
| **Phase 4** | Consulting clients | Multi-tenancy, dashboards | Planned |
| **Phase 5+** | Commercial | SaaS deployment, paid tiers | Future |

### Implemented Components

| Component | Completion | Notes |
|-----------|------------|-------|
| Provider abstraction | 100% | BaseProvider + Factory pattern |
| Ollama provider | 100% | Async generate/chat with metrics |
| Google provider | 100% | Gemini API integration |
| Scoring — pass@k | 100% | HumanEval methodology |
| Scoring — LLM-as-Judge | 100% | TD-011 bias mitigation |
| Scoring — RAG metrics | 100% | DeepEval integration |
| Hardware detection | 100% | Apple Silicon, NVIDIA, AMD, CPU |
| Benchmark runner | 100% | Warmup, concurrent execution |
| CLI | 80% | Commands work, config loading incomplete |
| Reporting | 0% | Not started |
| Export | 0% | Not started |

---

## Roadmap

Full roadmap in [DevPlan.md](./DevPlan.md). Summary:

| Priority | Item | Status |
|----------|------|--------|
| P0 | Provider abstraction layer | Done |
| P0 | Scoring module (pass@k, LLM-Judge, RAG) | Done |
| P0 | Hardware detection | Done |
| P0 | Benchmark runner | Done |
| P1 | Config-driven dataset loading | TODO |
| P1 | Reporting module (Jinja2) | Not started |
| P1 | Export to `_HQ/evaluations/` | Not started |
| P2 | Anthropic provider | Not started |
| P2 | OpenAI provider | Not started |
| P2 | Real test suite | Not started |

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
| Templates | Jinja2 | For reporting (planned) |
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

## Project Templates Integration

This project is registered in [`~/Tech_Projects/_HQ/SYNC_STATUS.yaml`](~/Tech_Projects/_HQ/SYNC_STATUS.yaml).

**Sync Status**: Up to date (all assets at version 1)

### Available Commands

| Command | Purpose |
|---------|---------|
| `/sync` | Pull template updates, contribute patterns, update portfolio |
| `/new-project` | Scaffold a new project from templates |

### Relevant Guides

Based on this project's features, these guides from `_HQ` are applicable:

| Guide | Path | Relevance |
|-------|------|-----------|
| LLM Evaluation | [guides/universal/LLM_EVALUATION.md](~/Tech_Projects/_HQ/guides/universal/LLM_EVALUATION.md) | Core methodology |
| Logging | [guides/universal/LOGGING.md](~/Tech_Projects/_HQ/guides/universal/LOGGING.md) | Structured logging patterns |
| Error Handling | [guides/universal/ERROR_HANDLING.md](~/Tech_Projects/_HQ/guides/universal/ERROR_HANDLING.md) | Exception hierarchy, retry, circuit breaker |
| Apple Silicon | [guides/universal/APPLE_SILICON.md](~/Tech_Projects/_HQ/guides/universal/APPLE_SILICON.md) | Hardware optimization |
| RAG + HITL | [guides/universal/RAG_HITL_PIPELINE.md](~/Tech_Projects/_HQ/guides/universal/RAG_HITL_PIPELINE.md) | RAG evaluation context |
| PII Detection | [guides/universal/PII_DETECTION.md](~/Tech_Projects/_HQ/guides/universal/PII_DETECTION.md) | Three-layer detection |
| Testing | [guides/universal/TESTING.md](~/Tech_Projects/_HQ/guides/universal/TESTING.md) | Testing strategy |
| CI/CD | [guides/universal/CI_CD.md](~/Tech_Projects/_HQ/guides/universal/CI_CD.md) | Pipeline design |
| Env Configuration | [guides/universal/ENV_CONFIGURATION.md](~/Tech_Projects/_HQ/guides/universal/ENV_CONFIGURATION.md) | .env patterns |

### Evaluation Outputs

AI_Eval is the source of truth for these `_HQ` evaluation files:

| File | Purpose |
|------|---------|
| `evaluations/MODEL_CATALOG.md` | LLM model catalog with benchmarks |
| `evaluations/HARDWARE_PROFILES.md` | Hardware reference configurations |
| `evaluations/DECISION_MATRIX.md` | Local vs API decision framework |

---

*Updated 2026-02-05*
