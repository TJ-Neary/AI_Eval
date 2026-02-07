# AI_Eval

![AI_Eval Architecture](assets/ai_eval_banner.png)

**A comprehensive LLM evaluation and benchmarking framework for comparing local and cloud-hosted language models across standardized test suites with hardware-aware performance profiling.**

![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)
![Status: Active Development](https://img.shields.io/badge/status-active%20development-yellow)

---

## Overview

Selecting the right language model for a production workload requires more than reading leaderboard scores. Published benchmarks rarely account for the hardware the model will actually run on, the specific task domain it needs to serve, or the cost trade-offs between local inference and API calls. AI_Eval addresses this gap by providing a structured, reproducible evaluation pipeline that benchmarks LLMs under real-world conditions and produces actionable, hardware-contextualized recommendations.

AI_Eval supports both **local models** via [Ollama](https://ollama.ai/) and **cloud API models** from Google, Anthropic, and OpenAI. Every benchmark run captures a complete hardware profile — chip architecture, memory bandwidth, GPU cores, thermal state — so that results are always grounded in the environment that produced them. This makes it possible to compare a 7B model running on a MacBook Air against the same model on a workstation GPU, or to quantify exactly when a cloud API becomes more cost-effective than local inference.

The framework evaluates models across five benchmark categories (text generation, code generation, document analysis, conversational ability, and structured output), applies three distinct scoring methodologies, and generates fitness scores weighted to specific use cases such as RAG knowledge engines, code assistants, and document processing pipelines.

---

## Key Features

### Multi-Provider Architecture

A unified provider abstraction layer enables seamless evaluation across inference backends. Each provider implements a standardized interface for generation, chat, model metadata retrieval, and resource measurement — making it straightforward to add new backends.

| Provider | Backend | Status |
|----------|---------|--------|
| Ollama | Local inference (Apple Silicon, NVIDIA, AMD, CPU) | Implemented |
| Google Gemini | Cloud API via `google-genai` SDK | Implemented |
| Anthropic Claude | Cloud API via `anthropic` SDK | Planned |
| OpenAI GPT | Cloud API via `openai` SDK | Planned |

### Scoring Methodology

AI_Eval employs three complementary scoring approaches, each designed for a different class of evaluation task:

- **pass@k (Code Generation)** — HumanEval-style functional correctness scoring. Generated code is executed in a sandboxed subprocess with timeout enforcement, and pass rates are calculated across *k* samples using the unbiased estimator from the [Evaluating Large Language Models Trained on Code](https://arxiv.org/abs/2107.03374) paper.

- **LLM-as-Judge (Subjective Quality)** — Automated quality assessment using a separate judge model with bias mitigation protocols per [TD-011](DevPlan.md#td-011-llm-as-judge-bias-mitigation). Position shuffling eliminates ordering bias in A/B comparisons, explicit rubrics penalize verbosity bias, and self-evaluation is prohibited (the judge model never scores its own output). Multi-evaluation consensus across multiple judge calls further reduces noise.

- **RAG Metrics (Retrieval-Augmented Generation)** — Integration with [DeepEval](https://github.com/confident-ai/deepeval) for RAGAS-based evaluation of retrieval pipelines. Metrics include Answer Relevancy (embedding-based semantic similarity), Faithfulness (factual grounding against retrieved context), Contextual Precision (retrieval signal-to-noise ratio), and Hallucination detection.

### Benchmark Categories

Models are evaluated across five standardized categories, each with defined subcategories, test prompts, and scoring rubrics:

| Category | Weight | Subcategories | Focus |
|----------|--------|---------------|-------|
| **Text Generation** | 20% | Instruction following, reasoning, summarization, creative writing | General writing quality and coherence |
| **Code Generation** | 25% | Python, SQL, bug fixing, code explanation, multi-language | Functional correctness and code quality |
| **Document Analysis** | 30% | Information extraction, classification, question answering, long context | Comprehension and extraction accuracy |
| **Conversational** | 10% | Multi-turn context, persona consistency, clarification handling | Dialogue quality and context retention |
| **Structured Output** | 15% | JSON/YAML generation, schema compliance, output consistency | Format adherence and reliability |

Full test definitions, scoring rubrics, and pass criteria are documented in [docs/TEST_SUITE_SPEC.md](docs/TEST_SUITE_SPEC.md).

### Fitness Profiles

Raw category scores are aggregated into **fitness scores** — weighted composites tuned to specific deployment scenarios. This allows the same benchmark data to answer different questions: "Which model is best for my RAG pipeline?" produces a different ranking than "Which model is best for code generation?"

| Profile | Primary Categories | Use Case |
|---------|-------------------|----------|
| RAG Knowledge Engine | Document Analysis (40%), Structured Output (25%) | Retrieval-augmented Q&A systems |
| Code Assistant | Code Generation (50%), Conversational (15%) | IDE integration, code review, generation |
| Document Processor | Document Analysis (50%), Structured Output (30%) | Classification, extraction, compliance |
| Chat Application | Conversational (40%), Text Generation (30%) | Customer support, conversational agents |
| Data Pipeline | Structured Output (35%), Document Analysis (30%) | ETL, schema transformation, data cleaning |

### Hardware-Aware Profiling

Every benchmark run begins with automatic hardware detection, embedding a complete system profile in the results:

- **Apple Silicon** — Chip identification (M1–M4, Pro/Max/Ultra tiers), GPU core count, Neural Engine cores, unified memory bandwidth
- **NVIDIA GPU** — VRAM, CUDA cores, driver version via `pynvml` / `nvidia-smi`
- **AMD GPU** — ROCm detection via `rocm-smi`
- **CPU Fallback** — Core count, clock speed, available RAM, architecture

This enables cross-machine comparison: the same model evaluated on different hardware produces results that can be meaningfully compared when normalized against the hardware profile.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              AI_Eval CLI                                    │
│  ai-eval run | quick-test | compare | hardware | list-models               │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Orchestrator                                      │
│  Config loading · Model discovery · Test coordination · Result aggregation  │
└─────────────────────────────────────────────────────────────────────────────┘
          │                    │                    │                    │
          ▼                    ▼                    ▼                    ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  Providers   │    │  Benchmarks  │    │  Profiling   │    │   Scoring    │
│              │    │              │    │              │    │              │
│  Ollama      │    │  Suites      │    │  Hardware    │    │  pass@k      │
│  Google      │    │  Datasets    │    │  Memory      │    │  LLM Judge   │
│  Anthropic*  │    │  Runner      │    │  Thermal     │    │  RAG Metrics │
│  OpenAI*     │    │              │    │              │    │              │
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Reporting & Export                                │
│  Jinja2 report templates · MODEL_CATALOG.md · DECISION_MATRIX.md · JSON    │
└─────────────────────────────────────────────────────────────────────────────┘
                                                              * = Planned
```

### Design Principles

- **Provider abstraction** — `BaseProvider` abstract class with `ProviderFactory` registry. Adding a new backend requires implementing `generate()`, `chat()`, and `get_model_info()`.
- **YAML-driven configuration** — Test suites, model lists, scoring thresholds, and fitness weights are all defined in `configs/default.yaml`, making evaluations fully reproducible.
- **Marker-based catalog updates** — Export module uses `<!-- AI_EVAL:BEGIN -->` / `<!-- AI_EVAL:END -->` markers to update shared catalog files without overwriting manual annotations.
- **Statistical rigor** — Bootstrap confidence intervals, minimum sample sizes per category, and paired-difference analysis for model comparisons (see [TD-010](DevPlan.md#td-010-statistical-rigor-requirements)).

---

## Requirements

- **Python** 3.12 or higher
- **macOS** (Apple Silicon optimized), **Linux**, or **Windows**
- **[Ollama](https://ollama.ai/)** installed and running (for local model evaluation)
- API keys for cloud providers (optional — only required for API model evaluation)

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/AI_Eval.git
cd AI_Eval

# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows

# Install runtime dependencies
pip install -r requirements.txt

# Install development dependencies (testing, linting, type checking)
pip install -r requirements-dev.txt

# Or install as editable package with dev extras
pip install -e ".[dev]"

# Configure environment variables
cp .env.example .env
# Edit .env with your API keys (optional for local-only evaluation)
```

## Quick Start

```bash
# Run a quick sanity test on a single model
python -m src quick-test --provider ollama --model qwen2.5:7b

# Execute a full benchmark suite
python -m src run --config configs/default.yaml

# Compare two models head-to-head
python -m src compare --models "qwen2.5:7b,llama3:8b" --provider ollama

# List available models from a provider
python -m src list-models --provider ollama

# Display hardware profile for the current machine
python -m src hardware
```

## Configuration

Evaluation behavior is controlled through two configuration layers:

| File | Purpose |
|------|---------|
| `.env` | API keys, data directory path, log level (gitignored — copy from `.env.example`) |
| `configs/default.yaml` | Model lists, test suite definitions, scoring thresholds, fitness profile weights, output formats |

See [.env.example](.env.example) for all supported environment variables and [configs/default.yaml](configs/default.yaml) for the full configuration schema.

---

## Project Structure

```
AI_Eval/
├── src/                        # Source code
│   ├── cli.py                  # CLI entry point and command definitions
│   ├── providers/              # LLM provider abstraction layer
│   │   ├── base.py             #   BaseProvider, ProviderFactory, data classes
│   │   ├── ollama_provider.py  #   Ollama local inference with throughput metrics
│   │   └── google_provider.py  #   Google Gemini API integration
│   ├── benchmarks/             # Benchmark execution engine
│   │   ├── runner.py           #   BenchmarkRunner with warmup and concurrency
│   │   └── datasets.py         #   Dataset management and test case loading
│   ├── scoring/                # Scoring methodologies
│   │   ├── pass_k.py           #   HumanEval-style pass@k for code generation
│   │   ├── llm_judge.py        #   LLM-as-Judge with bias mitigation (TD-011)
│   │   └── rag_metrics.py      #   DeepEval RAG evaluation metrics
│   ├── profiling/              # Hardware detection and resource monitoring
│   │   └── hardware.py         #   Platform-aware hardware profiling
│   ├── reporting/              # Report generation
│   │   ├── report_generator.py #   Jinja2-based markdown/JSON report generation
│   │   ├── readme_updater.py   #   README.md results table updater
│   │   └── templates/          #   Jinja2 report templates
│   └── export/                 # Catalog export (planned)
├── utils/                      # Shared utilities
│   ├── exceptions.py           #   Custom exception hierarchy (AiEvalError base, ReportingError)
│   ├── marker_parser.py        #   Marker-based content replacement for catalog updates
│   ├── logging_config.py       #   Multi-handler structured logging
│   ├── retry.py                #   Retry with backoff, circuit breaker, retry budget
│   ├── rate_limiter.py         #   Sliding window API rate limiting
│   ├── state_machine.py        #   Async state machine for workflow management
│   └── plugin_loader.py        #   Dynamic plugin discovery system
├── configs/                    # Evaluation configuration files (YAML)
│   └── default.yaml            #   Default evaluation configuration
├── tests/                      # Test suite (pytest)
├── docs/                       # Specifications and research
│   ├── TEST_SUITE_SPEC.md      #   Detailed test definitions and scoring rubrics
│   └── RESEARCH_SYNTHESIS.md   #   Synthesized research findings
├── config.py                   # Environment and path configuration
├── pyproject.toml              # Package metadata, tool configuration
├── DevPlan.md                  # Development roadmap and technical decisions
├── StartHere.md                # Developer onboarding guide
└── SECURITY.md                 # Security policy
```

---

## Benchmark Results

<!-- AI_EVAL:BEGIN -->
| Model | Provider | Score | Tokens/sec | Pass Rate | Hardware | Date |
|-------|----------|-------|------------|-----------|----------|------|
| [llama3.1:8b](reports/llama3.1_8b_20260207_042743.md) | OLLAMA | 66.0 | 88.8 | 4/5 | Apple M4 Max (48GB) | 2026-02-07 |

<!-- AI_EVAL:END -->

---

## Development Status

AI_Eval is in **active development**. The core evaluation infrastructure — provider abstraction, benchmark runner, scoring engine, and hardware detection — is implemented and functional. Reporting, export, and additional API providers are the next development priorities.

| Component | Status | Notes |
|-----------|--------|-------|
| Provider abstraction layer | Complete | `BaseProvider` + `ProviderFactory` pattern |
| Ollama provider | Complete | Async generate/chat with throughput metrics |
| Google Gemini provider | Complete | Cloud inference via `google-genai` SDK |
| Anthropic / OpenAI providers | Planned | SDK dependencies included |
| Benchmark runner | Complete | Warmup phase, concurrent execution, timeout handling |
| Scoring — pass@k | Complete | HumanEval methodology with sandboxed execution |
| Scoring — LLM-as-Judge | Complete | Bias-mitigated evaluation (TD-011) |
| Scoring — RAG metrics | Complete | DeepEval integration (RAGAS) |
| Hardware detection | Complete | Apple Silicon, NVIDIA, AMD, CPU fallback |
| CLI | Functional | Core commands working, config loading in progress |
| Reporting | Complete | Jinja2 markdown/JSON reports, README results table |
| Catalog export | Not started | Marker-based catalog file updates |

See [DevPlan.md](DevPlan.md) for the full development roadmap, technical decisions (TD-001 through TD-013), and task tracker.

---

## Testing

AI_Eval follows test-driven development (TDD) practices with the Red-Green-Refactor cycle. All new tests must be observed to fail before writing the implementation that makes them pass.

```bash
# Run the full test suite with coverage reporting
pytest

# Run a specific test file
pytest tests/test_example.py

# Skip slow or integration tests
pytest -m "not slow"
pytest -m "not integration"

# Run mutation testing to validate test effectiveness
mutmut run
mutmut results
```

Testing is configured in `pyproject.toml` with `--cov=src --cov-report=term-missing` and `asyncio_mode = "auto"`.

---

## Documentation

| Document | Description |
|----------|-------------|
| [DevPlan.md](DevPlan.md) | Development phases, technical decisions (TD-001–TD-013), task tracker, and roadmap |
| [StartHere.md](StartHere.md) | Comprehensive developer onboarding guide with architecture walkthrough |
| [docs/TEST_SUITE_SPEC.md](docs/TEST_SUITE_SPEC.md) | Detailed test definitions, scoring rubrics, and performance thresholds |
| [docs/RESEARCH_SYNTHESIS.md](docs/RESEARCH_SYNTHESIS.md) | Synthesized findings from multi-source evaluation methodology research |
| [SECURITY.md](SECURITY.md) | Security policy, secrets management, and incident response procedures |

---

## Contributing

Contributions are welcome. Please follow these guidelines:

1. **Code style** — Format with `black` (100 char line length), lint with `ruff`, type check with `mypy`
2. **Testing** — Follow the Red-Green-Refactor TDD cycle. Write the test first, observe it fail, then implement
3. **Commit format** — Use conventional commits: `<type>: <description>` (feat, fix, docs, refactor, test, chore, perf)
4. **Pre-commit hooks** — Install with `pre-commit install` to run automated checks before each commit
5. **Security** — Read [SECURITY.md](SECURITY.md) before committing. No secrets, PII, or hardcoded paths in tracked files

---

## License

MIT License — See [LICENSE](LICENSE) for details.
