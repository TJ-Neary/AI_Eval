# AI_Eval — Development Plan

> LLM evaluation and benchmarking tool for comparing local and API models.
> Exports results to `_HQ/evaluations/` for cross-project model selection.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Architecture](#architecture)
3. [Development Phases](#development-phases)
4. [Task Tracker](#task-tracker)
5. [Technical Decisions](#technical-decisions)
6. [Template Folder Outputs](#template-folder-outputs)
7. [Quality Standards](#quality-standards)

---

## Project Overview

**Project**: AI_Eval  
**Python**: 3.12+  
**Created**: 2026-02-02  
**Template Output**: `~/Tech_Projects/_HQ/evaluations/`

### Goals

1. **Benchmark local LLM models** against standardized test suites (text, code, analysis, chat, structured output)
2. **Compare local vs API performance** to determine when API is necessary
3. **Generate hardware-aware recommendations** so models can be matched to target deployment machines
4. **Maintain a living catalog** (`MODEL_CATALOG.md`, `DECISION_MATRIX.md`) updated automatically after each evaluation

### Extended Goals (Future)

5. **RAG Pipeline Evaluation** — Test retrieval quality, faithfulness, answer relevancy (RAGAS)
6. **Embedding Model Comparison** — MTEB-style benchmarks for vector embeddings
7. **Multi-machine support** — Run evaluations on any machine (macOS, Linux, Windows, VMs), aggregate results
8. **Model update monitoring** — Detect new model releases, notify user, manual trigger to run tests
9. **Portable distribution** — Package as standalone tool for client machine testing
10. **Vision & Multimodal models** — Image understanding, image-to-text, visual QA
11. **Prompt Engineering Testing** — Measure prompt sensitivity, model-specific optimization
12. **Audio/Speech models** — STT, TTS (when local options mature)
13. **Image generation models** — Stable Diffusion quality/speed benchmarks

### Target Use Cases & Roadmap

| Phase | Target | Focus |
|-------|--------|-------|
| **MVP** | Personal Mac | Core evaluation, RAG testing, my projects |
| **Phase 2** | Family Windows | Cross-platform validation, practice consulting |
| **Phase 3** | Consulting Clients | Run on client hardware, branded reports |
| **Phase 4+** | Commercial Tool | Enterprise features, compliance, SOC 2 |

### Licensing & Distribution

- **Current**: Private repository, undecided license
- **Decision Point**: After MVP proven, evaluate MIT vs commercial
- **Design Principle**: Build for potential commercialization without over-engineering

### Model Types Covered

| Type | Examples | Status |
|------|----------|--------|
| **Text LLMs** | qwen2.5, llama3.1, deepseek | Primary focus (v1.0) |
| **Code LLMs** | deepseek-coder, codestral | Primary focus (v1.0) |
| **Vision/Multimodal** | llama3.2-vision, gemini-pro-vision | v1.1 |
| **Embeddings** | nomic-embed-text, bge-m3 | v1.0 (separate track) |
| **Audio/STT** | Whisper, faster-whisper | v2.0 |
| **Image Gen** | Stable Diffusion, SDXL | v2.0 |

### Non-Goals (v1.0)

- Real-time model serving (see `MODEL_SERVING.md` guide instead)
- Training or fine-tuning models
- Managing Ollama installations (assumes Ollama is pre-installed)
- Automatic test execution on model updates (manual trigger only)

---

## Architecture

### High-Level Design

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              AI_Eval CLI                                    │
│  ai-eval run --config configs/default.yaml --models qwen2.5:32b            │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Orchestrator                                      │
│  - Loads config                                                             │
│  - Coordinates test runs                                                    │
│  - Aggregates results                                                       │
└─────────────────────────────────────────────────────────────────────────────┘
          │                    │                    │                    │
          ▼                    ▼                    ▼                    ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  Providers   │    │  Benchmarks  │    │  Profiling   │    │   Scoring    │
│              │    │              │    │              │    │              │
│ - Ollama     │    │ - Suites     │    │ - Hardware   │    │ - Category   │
│ - Google     │    │ - Datasets   │    │ - Memory     │    │ - Fitness    │
│ - Anthropic  │    │ - Runner     │    │ - Timing     │    │ - Weighted   │
│ - OpenAI     │    │              │    │              │    │              │
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Reporting & Export                                │
│  - Render model_eval_report.md.j2                                          │
│  - Update MODEL_CATALOG.md (<!-- AI_EVAL:BEGIN/END --> markers)            │
│  - Update DECISION_MATRIX.md                                                │
│  - Save raw data to evaluations/data/                                       │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Key Components

| Component | Location | Purpose | Status |
|-----------|----------|---------|--------|
| **Providers** | `src/providers/` | Abstraction layer for LLM backends | Partial (Ollama, Google done; Anthropic, OpenAI pending) |
| **Benchmarks** | `src/benchmarks/` | Test suites and dataset management | Implemented |
| **Profiling** | `src/profiling/` | Hardware detection and resource monitoring | Implemented |
| **System Monitor** | `src/monitor/` | Real-time safety monitoring, interference detection | Not Started (directory not created) |
| **Scoring** | `src/scoring/` | Score calculation and normalization | Implemented |
| **Reporting** | `src/reporting/` | Report generation from Jinja2 templates | Not Started (directory exists, empty) |
| **Export** | `src/export/` | Update `_HQ/evaluations/` | Not Started (directory exists, empty) |
| **CLI** | `src/cli.py` | Command-line interface | Implemented (config loading incomplete) |

### Data Flow

```
1. User runs: ai-eval run --config configs/rag-eval.yaml
2. Orchestrator loads config, discovers models
3. For each model:
   a. Provider.get_model_info() → metadata
   b. Profiling.detect_hardware() → machine specs
   c. For each test in suite:
      - Provider.generate(prompt) with timing
      - Profiling.measure_resources() → RAM, GPU
      - Scoring.score_response(response, expected)
4. Scoring.calculate_fitness() → weighted scores
5. Reporting.render() → markdown reports
6. Export.update_catalog() → update template files
```

---

## Pre-Implementation Research

> ✅ **Research Complete** — See [RESEARCH_SYNTHESIS.md](docs/RESEARCH_SYNTHESIS.md) for synthesized findings. Raw research archived in `_archive/docs/`.

### R-001: Competitive Analysis ✅
- Open LLM Leaderboard v2 uses `lm-evaluation-harness` with MMLU-Pro, GPQA, MUSR, MATH, IFEval, BBH
- Normalized scoring (0 = random, 100 = perfect)
- AI_Eval should use lm-eval for standard benchmarks, add custom tests on top

### R-002: API Provider Documentation ✅
| Provider | Safety Testing | Benchmarking | Data Retention |
|----------|---------------|--------------|----------------|
| Google | ✅ Allowed | ✅ Allowed | Opt-in only |
| OpenAI | ⚠️ Conditional* | ✅ Allowed | 30 days |
| Anthropic | ✅ Explicit allow | ✅ Allowed | Per ToS |

*OpenAI: "Unsolicited safety testing" prohibited. Formal red-team program required.

### R-003: Legal & Licensing ✅
- **Recommendation**: MIT license (maximum adoption, simple)
- Benchmark publication: Generally allowed
- lm-eval: Apache 2.0 (compatible)

### R-004: Benchmark Datasets ✅
| Dataset | License | Use in AI_Eval |
|---------|---------|----------------|
| HumanEval | MIT | ✅ Code generation |
| MBPP | CC-BY-4.0 | ✅ Simpler code |
| MMLU | MIT | ✅ Via lm-eval |
| BigBench | Apache 2.0 | ✅ Reasoning |

### R-005: Hardware Detection ✅
| Platform | Method |
|----------|--------|
| All | `psutil` for CPU/RAM |
| Apple Silicon | `system_profiler`, `ioreg -l \| grep gpu-core-count`, `sysctl -n hw.memsize` |
| NVIDIA | `pynvml` or `nvidia-smi` |
| AMD | `rocm-smi` |

### R-006: API Pricing ✅
| Model | Input $/1M | Output $/1M |
|-------|------------|-------------|
| Gemini 2.5 Pro | $1.25-2.50 | $10-15 |
| Gemini 2.5 Flash | $0.30 | $2.50 |
| GPT-4o | $2.50 | $10.00 |
| Claude 3.5 Sonnet | $3.00 | $15.00 |

---

## Development Phases

### Phase 1: Foundation (Week 1)
Core infrastructure that everything else depends on.

- [x] Project scaffolding and configuration
- [x] **Provider abstraction layer**
  - [x] `BaseProvider` abstract class with `generate()`, `get_model_info()`, `measure_resources()`
  - [x] `OllamaProvider` implementation
  - [x] `GoogleProvider` implementation (google-genai SDK)
  - [ ] `AnthropicProvider` implementation
  - [ ] `OpenAIProvider` implementation
- [x] **Hardware detection module**
  - [x] Apple Silicon detection (chip type, GPU cores, Neural Engine, unified memory)
  - [x] Linux/Windows GPU detection
  - [x] CPU-only fallback detection
  - [x] `HardwareProfile` dataclass
- [ ] **Test dataset structure**
  - [ ] YAML schema for test prompts + expected outputs
  - [ ] Dataset versioning with SHA256 hash
  - [ ] Initial dataset: 5 prompts per category (text, code, analysis, chat, structured)
- [ ] **Cold start timing** (TD-012)
  - [ ] Measure model load time separately from inference
  - [ ] Track first-query latency vs subsequent averages

### Phase 2: Core Benchmarking (Week 2)
The actual evaluation engine.

- [x] **Benchmark runner**
  - [x] Load test suites from `configs/*.yaml`
  - [x] Execute tests with warmup phase (discard first N queries)
  - [x] Timeout handling and graceful failure
  - [x] Progress display with `rich`
- [x] **Metrics collection**
  - [x] Total generation time
  - [x] Time-to-first-token (TTFT)
  - [x] Tokens per second
  - [x] Peak RAM usage
  - [x] GPU/MPS memory usage
- [x] **Scoring system**
  - [x] Category scores (1-100) for each of 5 categories
  - [x] Subcategory breakdowns
  - [x] Weighted fitness scores by use case (RAG, code-assistant, chat, etc.)
- [ ] **Context window stress tests**
  - [ ] Test at 25%, 50%, 75%, 100% of advertised context
  - [ ] Measure performance degradation curve
- [ ] **Statistical rigor** (TD-010)
  - [ ] Bootstrap confidence intervals (1000 resamples) on all metrics
  - [ ] Minimum 100 test cases per category
  - [ ] Report format: `Score = 85.2% (95% CI: 82.1-88.3%)`
- [ ] **Memory bandwidth tracking** (TD-012)
  - [ ] Calculate MBU: `(Model_Size_GB * TPS) / Memory_Bandwidth_GB/s`
  - [ ] Track theoretical max vs actual efficiency
- [ ] **Soak test mode** (TD-012)
  - [ ] 30-minute continuous inference option
  - [ ] Sample GPU temp/clock every 30 seconds
  - [ ] Calculate degradation %: `(TPS_initial - TPS_final) / TPS_initial`

### Phase 2.5: RAG Evaluation (Week 2-3)
RAG pipeline and embedding model testing — core use case.

- [ ] **RAGAS metrics integration**
  - [ ] Faithfulness (factual accuracy vs context)
  - [ ] Answer Relevancy (embedding similarity)
  - [ ] Context Precision (retrieval signal-to-noise)
  - [ ] Context Recall (requires ground truth)
- [ ] **Embedding model benchmarks** (MTEB subset)
  - [ ] Retrieval accuracy tests
  - [ ] nomic-embed-text, mxbai-embed-large comparisons
  - [ ] Embedding dimension vs quality analysis
- [ ] **RAG pipeline end-to-end test**
  - [ ] Document → chunk → embed → retrieve → generate → score
  - [ ] Measure retrieval latency, generation latency separately
- [ ] **RAG test suite config**
  - [ ] Create `configs/rag-eval.yaml`
  - [ ] CLI: `ai-eval rag-test --pipeline <name>`

### Phase 3: Reporting & Export (Week 3)
Generate outputs and update template catalog.

- [ ] **Report generation**
  - [ ] Render `model_eval_report.md.j2` for individual models
  - [ ] Render `comparison_report.md.j2` for head-to-head comparisons
  - [ ] Save reports to `reports/`
- [ ] **Template catalog updates**
  - [ ] Parse `<!-- AI_EVAL:BEGIN -->` / `<!-- AI_EVAL:END -->` markers
  - [ ] Insert/update model entries in `MODEL_CATALOG.md`
  - [ ] Update `DECISION_MATRIX.md` with new recommendations
  - [ ] Update `HARDWARE_PROFILES.md` with tested configurations
- [ ] **Data persistence**
  - [ ] Save raw benchmark data as JSON to `evaluations/data/`
  - [ ] Include dataset version hash in all reports
  - [ ] Store historical scores for regression detection

### Phase 4: CLI & UX (Week 4)
User-facing interface and quality of life.

- [ ] **CLI commands**
  - [ ] `ai-eval run` — full benchmark run
  - [ ] `ai-eval quick-test --model <name>` — interactive single-model test
  - [ ] `ai-eval compare --models <a>,<b>` — head-to-head comparison
  - [ ] `ai-eval list-models` — show available models
  - [ ] `ai-eval export` — update template folder without re-running tests
  - [ ] `ai-eval estimate-cost` — project API costs before running (TD-009)
  - [ ] `ai-eval rag-test` — dedicated RAG pipeline evaluation
- [ ] **Visualization**
  - [ ] Generate PNG/SVG charts for score comparisons
  - [ ] Radar charts for category breakdowns
  - [ ] Performance degradation curves
- [ ] **Regression detection**
  - [ ] Compare current scores to historical baseline
  - [ ] Alert on significant score changes (>5% delta)
  - [ ] Track model version changes from Ollama

### Phase 5: Polish & Extensions (Week 5+)
Nice-to-haves and advanced features.

- [ ] **New template outputs**
  - [ ] `EMBEDDING_CATALOG.md` — separate catalog for embedding models
  - [ ] `QUANTIZATION_GUIDE.md` — Q4_K_M vs Q8_0 vs FP16 analysis
  - [ ] `PROMPT_FORMATS.md` — chat templates per model family
  - [ ] `MIGRATION_PATHS.md` — model upgrade recommendations
  - [ ] `CHANGELOG.md` — evaluation history
- [ ] **Pre-built eval configs**
  - [ ] `configs/rag-eval.yaml` — RAG/knowledge engine focus
  - [ ] `configs/code-eval.yaml` — code generation focus
  - [ ] `configs/chat-eval.yaml` — conversational focus
  - [ ] `configs/quick-screen.yaml` — fast 5-minute screening
- [ ] **Advanced benchmarks**
  - [ ] Custom datasets from CoreRag prompts (document classification, PII advisory)
  - [ ] Failure mode documentation (OOM behavior, timeout patterns)
  - [ ] API cost tracking with break-even analysis
- [ ] **Model serving guide**
  - [ ] Create `guides/universal/MODEL_SERVING.md`
  - [ ] Cover Ollama vs vLLM vs llama.cpp vs MLX-LM

### Phase 6: Advanced Enhancements (Future)
Features for multi-machine and commercial use.

- [ ] **Prompt format management**
  - [ ] Store chat templates per model family (ChatML, Llama, Qwen)
  - [ ] Auto-detect format from model name
  - [ ] Validate format before running tests
- [ ] **Variance/consistency testing**
  - [ ] Run same prompt N times, measure output variance
  - [ ] Report standard deviation per model
  - [ ] Flag high-variance models as "unstable"
- [ ] **Temperature sensitivity**
  - [ ] Test at temp 0.0, 0.3, 0.7, 1.0
  - [ ] Find optimal temp per use case
- [ ] **Latency percentiles**
  - [ ] Report p50, p95, p99 (not just mean)
  - [ ] Track tail latency separately
- [ ] **Resume capability**
  - [ ] Checkpoint after each test
  - [ ] Resume from last successful test on crash
- [ ] **Model registry monitoring**
  - [ ] Check Ollama library for new model releases
  - [ ] Notify user of available updates
  - [ ] Manual trigger to pull and test
  - [ ] Track model weight hashes for change detection
- [ ] **Multi-machine support**
  - [ ] Platform-agnostic hardware detection (macOS, Linux, Windows)
  - [ ] Portable results format (JSON + markdown)
  - [ ] Compare results across machines
  - [ ] VM/container testing support
- [ ] **Power/efficiency metrics** (Apple Silicon)
  - [ ] Track wattage via `powermetrics`
  - [ ] Report "tokens per watt"
  - [ ] Battery impact analysis
- [ ] **Failure mode catalog**
  - [ ] Document OOM behavior per model
  - [ ] Timeout patterns
  - [ ] Hallucination triggers
  - [ ] Token limit violations
- [ ] **Packaging for distribution**
  - [ ] PyPI package
  - [ ] Standalone macOS app (PyInstaller)
  - [ ] Docker container for Linux testing

### Phase 7: Strategic Enhancements (v2.0+)
Features for differentiation and commercial viability.

- [ ] **Use case profiles** (Quick Start)
  - [ ] Pre-configured profiles: `rag-engine`, `coding-assistant`, `customer-chat`, `document-processor`
  - [ ] CLI: `ai-eval profile <name>` selects tests, metrics, weights automatically
  - [ ] Custom profile creation: `ai-eval profile create myprofile`
- [ ] **Quantization comparison**
  - [ ] Test same model at Q4_K_M, Q8_0, FP16
  - [ ] Measure quality delta, speed delta, RAM delta
  - [ ] Recommend optimal quantization per use case
- [ ] **Historical tracking & regression alerts**
  - [ ] Store all results in local SQLite database
  - [ ] Compare current run to previous runs
  - [ ] Alert if quality dropped >5% after model update
  - [ ] Trend visualization over time
- [ ] **Hardware recommendations in reports**
  - [ ] Minimum RAM to run
  - [ ] Recommended RAM for good performance
  - [ ] Optimal hardware tier classification
- [ ] **Local web dashboard**
  - [ ] localhost web UI for browsing results
  - [ ] Side-by-side model comparison
  - [ ] Filter by use case, date, score
  - [ ] Export charts as PNG/SVG
- [ ] **Plugin architecture**
  - [ ] `@ai_eval.register_test()` decorator for custom tests
  - [ ] Load plugins from `~/.ai_eval/plugins/`
  - [ ] Plugin API documentation
- [ ] **Community benchmarks** (Opt-in)
  - [ ] Upload anonymized hardware + scores
  - [ ] Compare to community averages
  - [ ] Leaderboard by hardware tier
- [ ] **Vision/Multimodal test suite**
  - [ ] Image understanding tests (describe image, extract text)
  - [ ] Visual QA tests (answer questions about image)
  - [ ] Image accuracy scoring rubrics
- [ ] **Embedding test suite** 
  - [ ] Retrieval accuracy (given query, find relevant docs)
  - [ ] Clustering quality
  - [ ] MTEB subset for local testing
- [ ] **CI/CD integration**
  - [ ] GitHub Action for model regression testing
  - [ ] Pre-commit hook for quality gates
  - [ ] Webhook notifications

---

## Commercial Readiness Checklist

Items required before public release (OSS or commercial):

| Item | Status | Priority |
|------|--------|----------|
| **Licensing decision** (MIT, Apache, proprietary) | Not Started | Critical |
| **Project name/branding** | Using "AI_Eval" | Medium |
| **Documentation site** | Not Started | High |
| **JSON schema versioning** | Not Started | High |
| **Contributor guidelines** | Not Started | Medium |
| **Security audit** | Not Started | High |
| **Test coverage >80%** | Not Started | Medium |
| **PyPI package registration** | Not Started | Medium |
| **Logo/visual identity** | Not Started | Low |

---

## Model Testing Priority

> **Strategy**: Local models first, APIs later. Start with currently installed models.

### Tier 1: Currently Installed (Test First)

Models already on this machine (M4 Max 48GB):

| Model | Size | Parameters | Category | Priority |
|-------|------|------------|----------|----------|
| `qwen2.5:32b` | 19 GB | 32B | General/Code | 🔴 High |
| `deepseek-r1:32b` | 19 GB | 32B | Reasoning | 🔴 High |
| `deepseek-coder:33b` | 18 GB | 33B | Code | 🔴 High |
| `gemma2:27b` | 15 GB | 27B | General | 🟡 Medium |
| `llama3.1:70b` | 42 GB | 70B | General | 🟡 Medium (RAM limit) |
| `llama3.1:8b` | 4.9 GB | 8B | General (fast) | 🟡 Medium |
| `llama3.2-vision:latest` | 7.8 GB | — | Vision/Multimodal | 🟡 Medium |
| `dolphin-mixtral:latest` | 26 GB | MoE | Uncensored | 🟢 Low |
| `wizard-vicuna-uncensored:30b` | 18 GB | 30B | Uncensored | 🟢 Low |
| `gpt-oss:20b` | 13 GB | 20B | General | 🟢 Low |
| `nomic-embed-text:latest` | 274 MB | — | Embeddings | 🔴 High (separate tests) |

### Tier 2: High-Priority Models to Pull

Models to download and test after Tier 1:

| Model | Estimated Size | Rationale |
|-------|----------------|-----------|
| `qwen2.5-coder:32b` | ~19 GB | Code-specialized Qwen |
| `codestral:22b` | ~13 GB | Mistral's code model |
| `phi-4:14b` | ~8 GB | Microsoft's efficient model |
| `mistral:7b` | ~4 GB | Fast baseline comparison |
| `llama3.2:3b` | ~2 GB | Ultra-lightweight option |

### Tier 3: API Models (After Local Complete)

Test after local model evaluation is stable:

| Provider | Model | Purpose |
|----------|-------|---------|
| Google | `gemini-2.0-flash` | Fast API baseline |
| Google | `gemini-1.5-pro` | Quality ceiling |
| Anthropic | `claude-3.5-sonnet` | Code/analysis reference |
| OpenAI | `gpt-4o` | Industry benchmark |

### Testing Order

```
1. Tier 1 installed models (foundation data)
   └─ Start with: qwen2.5:32b, deepseek-r1:32b, llama3.1:8b
   
2. nomic-embed-text (embedding tests, separate category)

3. Tier 2 models (expand local coverage)
   └─ Pull and test incrementally
   
4. Tier 3 APIs (establish quality ceiling)
   └─ Only after local benchmarks are solid
```

---

## Task Tracker

### Critical Priority

| Task | Status | Owner | Notes |
|------|--------|-------|-------|
| BaseProvider abstraction | Implemented | — | Blocks all model testing |
| OllamaProvider | Implemented | — | Primary local model interface |
| Hardware detection | Implemented | — | Required for meaningful comparisons |

### High Priority

| Task | Status | Owner | Notes |
|------|--------|-------|-------|
| Dataset schema + initial prompts | In Progress | — | 25 prompts (5 per category) |
| Benchmark runner | Implemented | — | Core evaluation loop with warmup, concurrency |
| Scoring system | Implemented | — | pass@k, LLM-as-Judge (TD-011), RAG metrics |
| Real test suite | Not Started | — | Replace placeholder tests in `tests/` |

### Medium Priority

| Task | Status | Owner | Notes |
|------|--------|-------|-------|
| GoogleProvider | Implemented | — | google-genai SDK |
| AnthropicProvider | Not Started | — | anthropic SDK |
| OpenAIProvider | Not Started | — | openai SDK |
| Report generation | Not Started | — | Jinja2 templates (`src/reporting/` directory exists, empty) |
| Export to templates | Not Started | — | Update `MODEL_CATALOG.md` (`src/export/` directory exists, empty) |
| CLI interface | Implemented | — | `ai-eval run` command (config loading TODO) |

### Low Priority

| Task | Status | Owner | Notes |
|------|--------|-------|-------|
| Visualization charts | Not Started | — | matplotlib PNG export |
| Regression detection | Not Started | — | Compare to historical baseline |
| Embedding catalog | Not Started | — | Separate from LLM catalog |

### Code Health (from StartHere.md audit 2026-02-04)

Issues discovered during comprehensive code review:

| Issue | Location | Status | Action Required |
|-------|----------|--------|-----------------|
| Hardcoded dataset | `src/cli.py:108` | TODO | Load from `RunConfig` or `configs/default.yaml` |
| Empty reporting dir | `src/reporting/` | Not Started | Implement Jinja2 report generation (Phase 3) |
| Empty export dir | `src/export/` | Not Started | Implement catalog export (Phase 3) |
| Placeholder tests | `tests/test_example.py` | Placeholder | Add real unit tests for providers, scoring, benchmarks |
| Missing Anthropic provider | `src/providers/` | Not Started | Implement `AnthropicProvider` |
| Missing OpenAI provider | `src/providers/` | Not Started | Implement `OpenAIProvider` |
| Incomplete LLM judge rubrics | `src/scoring/llm_judge.py` | Partial | Only 5/9 criteria have rubrics (missing HARMLESSNESS, COMPLETENESS, CONCISENESS, CREATIVITY) |
| RAG metrics not wired | `src/benchmarks/runner.py` | Not Integrated | `rag_metrics.py` module exists but `_score_response()` never calls it for RAG-type tests |

**Orphaned Utility Modules** (implemented but never imported):

| Module | Location | Purpose | Recommendation |
|--------|----------|---------|----------------|
| `rate_limiter.py` | `utils/` | Sliding window API throttling | Integrate into API providers |
| `code_validator.py` | `utils/` | AST-based code safety | Integrate into `pass_k.py` for sandboxing |
| `state_machine.py` | `utils/` | Async workflow state tracking | Integrate into `BenchmarkRunner` lifecycle |
| `path_guard.py` | `utils/` | Filesystem write boundary | Integrate into code execution sandboxing |
| `plugin_loader.py` | `utils/` | Dynamic plugin discovery | Wire for future provider plugins (Phase 5+) |

> **Note**: These utilities were scaffolded from `_HQ` templates and are production-ready.
> Decide: integrate into active code paths OR document as optional/future features.

### Recommendations from /sync (2026-02-04)

Specific integration tasks for orphaned utilities:

| Task | Priority | What | Where | Why |
|------|----------|------|-------|-----|
| Wire rate_limiter into providers | **High** | Add `RateLimiter` to API provider base class | `src/providers/google_provider.py`, future `anthropic_provider.py`, `openai_provider.py` | Prevent API rate limit errors during benchmark runs; each provider has different limits |
| Wire code_validator into pass_k | **High** | Validate LLM-generated code before execution | `src/scoring/pass_k.py` — call `validate_code()` before `subprocess.run()` | Prevent malicious/dangerous code from executing during code generation benchmarks |
| Wire path_guard into pass_k | **Medium** | Sandbox temp directories for code execution | `src/scoring/pass_k.py` — wrap temp dir creation with `PathGuard` | Ensure generated code can't write outside designated sandbox |
| Wire state_machine into runner | **Medium** | Replace manual state tracking with `StateMachine` | `src/benchmarks/runner.py` — states: INIT→WARMUP→RUNNING→SCORING→COMPLETE/ERROR | Cleaner lifecycle management, transition history for debugging |
| Wire retry.py into providers | **Medium** | Use existing retry utility in provider API calls | `src/providers/google_provider.py` — wrap API calls with `retry_with_backoff()` | Already have the utility but providers don't use it; follows ERROR_HANDLING.md guide |
| Defer plugin_loader | **Low** | Keep for Phase 5+ plugin architecture | `src/plugins/` (future) | Not needed until custom evaluation plugins are implemented |

**Integration Order** (recommended):
1. `code_validator` + `path_guard` → `pass_k.py` (security-critical for code execution)
2. `rate_limiter` + `retry.py` → providers (reliability for API calls)
3. `state_machine` → `runner.py` (code quality improvement)
4. `plugin_loader` → defer to Phase 5+

---

## Technical Decisions

### TD-001: Provider Abstraction Pattern

- **Decision**: Use abstract base class `BaseProvider` with standardized interface
- **Rationale**: Makes adding new providers trivial; enforces consistent metrics collection
- **Interface**:
  ```python
  class BaseProvider(ABC):
      @abstractmethod
      def generate(self, prompt: str, **kwargs) -> GenerationResult: ...
      @abstractmethod
      def get_model_info(self) -> ModelInfo: ...
      @abstractmethod
      def measure_resources(self) -> ResourceMetrics: ...
  ```
- **Date**: 2026-02-02

### TD-002: Dataset Versioning

- **Decision**: Each dataset file includes SHA256 hash; reports include hash for reproducibility
- **Rationale**: Ollama updates models frequently; must know which prompts produced which scores
- **Format**: `datasets/v1.0.0-abc123.yaml`
- **Date**: 2026-02-02

### TD-003: Marker-Based Catalog Updates

- **Decision**: Use `<!-- AI_EVAL:BEGIN -->` / `<!-- AI_EVAL:END -->` markers in template files
- **Rationale**: Allows manual annotations outside markers to persist; AI_Eval only touches its section
- **Files affected**: `MODEL_CATALOG.md`, `DECISION_MATRIX.md`, `HARDWARE_PROFILES.md`
- **Date**: 2026-02-02

### TD-004: Token-Level Metrics

- **Decision**: Measure TTFT (time-to-first-token) separately from total generation time
- **Rationale**: TTFT matters for real-time UX (chat, streaming); total time matters for batch processing
- **Implementation**: Use streaming API when available
- **Date**: 2026-02-02

### TD-005: Mandatory Hardware Auto-Detection

- **Decision**: Auto-detect full hardware profile before every benchmark run; embed in every report
- **Rationale**: Results are meaningless without hardware context; enables multi-machine comparison
- **Detection includes**:
  - Platform (macOS/Linux/Windows)
  - CPU/chip (Apple M4 Max, Intel i9, AMD Ryzen, etc.)
  - Total RAM, available RAM
  - GPU type, VRAM/unified memory
  - GPU cores, Neural Engine cores (Apple Silicon)
  - Disk type (SSD/HDD), free space
  - Power source (AC/battery)
  - Thermal state
- **When**: Runs automatically at benchmark start, no user action needed
- **Output**: `hardware_profile` section in every JSON result and markdown report
- **Date**: 2026-02-02

### TD-006: Dual Output Format

- **Decision**: Generate both markdown and JSON for every report
- **Rationale**: Markdown for human reading, JSON for machine parsing and cross-project integration
- **Output files**:
  - `reports/<model>_<date>.md` — human-readable with tables, prose, summaries
  - `reports/<model>_<date>.json` — structured data, all scores, raw metrics
  - `data/<model>_<date>_raw.json` — complete raw test results for archival
- **JSON schema**: Versioned schema so consumers can handle format changes
- **Template folder**: Markdown goes to `evaluations/reports/`, JSON goes to `evaluations/data/`
- **Date**: 2026-02-02

### TD-007: Scientific Experimental Methodology

- **Decision**: Follow rigorous experimental methodology for reproducible, credible results
- **Rationale**: Results must be statistically valid and reproducible across runs

**Pre-Test Protocol**:
1. Close heavy applications, connect to AC power
2. Thermal cooldown if system recently under load (wait 2 min if temp elevated)
3. Capture system baseline snapshot
4. Load model, run 3 warmup queries (results discarded)

**During-Test Protocol**:
1. Randomize test order (avoid ordering effects)
2. Run each test 3x minimum (configurable via `replication_count`)
3. One model loaded at a time; full unload between models
4. Include control prompts with known-correct answers
5. Timestamp every query for event correlation
6. For subjective scoring: blind evaluation (scorer doesn't see model name)

**Post-Test Analysis**:
1. Calculate mean ± standard deviation for all metrics
2. Flag outliers (>2σ from mean) for investigation
3. Report confidence intervals, not just raw scores
4. Note statistical significance when comparing models
5. Document any excluded tests and reasons

**Report Requirements**:
- Methodology section: date, versions, conditions, replication count
- Limitations section: what test doesn't measure, caveats
- Reproducibility checklist: exact commands to replicate

- **Date**: 2026-02-02

### TD-008: Security & Safety Protocol

- **Decision**: Implement comprehensive security measures and model safety testing
- **Rationale**: Protect user machine, API credentials, and evaluate model safety characteristics

**API Key Protection**:
- Store keys in `.env` file, load via `python-dotenv`
- Never log keys; redact from all output
- Pre-commit hook scans for key patterns (blocks commits with secrets)
- `.gitignore` includes `.env`, `*.key`, `*credentials*`

**Code Execution Sandboxing**:
- Generated code runs in subprocess with 30s timeout
- No file system access outside temp directory
- Whitelist allowed imports (no `os.system`, `subprocess`, `eval`)

**Network Monitoring** (Optional paranoid mode):
- Log all outbound connections during model tests
- Alert on unexpected network activity

**Safety Guardrail Testing** (Local Models Only):
- Test refusal consistency (same harmful prompt → same refusal?)
- Jailbreak resistance tests (standard bypass attempts)
- Content policy boundary probing
- Document what each model allows/refuses

**API Compliance**:
- Research each provider's ToS before testing
- Document which providers allow safety testing
- Flag providers that prohibit red-team testing
- Never run jailbreak tests on APIs without explicit permission

**Data Policy Documentation** (per API provider):
- Data retention period
- Training data reuse policy
- Privacy rating (1-5 scale)
- Jurisdiction / legal considerations

- **Date**: 2026-02-02

### TD-009: Cost Tracking & Token Calculator

- **Decision**: Track all token usage and provide cost estimation tools
- **Rationale**: Enable budget planning and API vs. local cost comparison

**Token Counting**:
- Count input/output tokens for every API call
- Store in test results for aggregation
- Use provider's tokenizer when available

**Cost Calculation**:
- Maintain pricing table per model (updated manually)
- Calculate cost per test run: `(input_tokens × input_price) + (output_tokens × output_price)`
- Report total cost at end of benchmark run

**Pre-Run Estimation**:
- Command: `ai-eval estimate-cost --model gemini-pro --suite full`
- Estimate tokens based on test suite size
- Show projected cost before user confirms

**Break-Even Analysis**:
- Compare API cost to local model quality
- Calculate: "At X queries/month, local model saves $Y vs API"
- Include in comparison reports

**Pricing Data Source**:
- `configs/api_pricing.yaml` — user-maintained pricing table
- CLI command to update: `ai-eval update-pricing`

- **Date**: 2026-02-02

### TD-010: Statistical Rigor Requirements

- **Decision**: Implement Anthropic's statistical framework for benchmark reliability
- **Rationale**: Simple mean±std insufficient; need confidence intervals and power analysis
- **Reference**: [RESEARCH_SYNTHESIS.md](docs/RESEARCH_SYNTHESIS.md)

**Minimum Requirements**:
- 100+ test cases per category minimum (800+ for 3% effect detection at 80% power)
- Bootstrap confidence intervals on all metrics (1000 resamples)
- Clustered standard errors for non-independent questions
- Paired-difference analysis when comparing models
- Cohen's d effect size for model comparisons

**Reporting**:
- Report: `Score = 85.2% (95% CI: 82.1-88.3%)`
- Include sample size (n) in all metrics
- Flag results with wide CIs (>±5%) as "low confidence"

- **Date**: 2026-02-02

### TD-011: LLM-as-Judge Bias Mitigation

- **Decision**: Implement bias mitigation for all LLM-based evaluation
- **Rationale**: Research shows position bias, verbosity bias, and self-evaluation bias

**Mitigation Strategies**:
| Bias Type | Mitigation |
|-----------|------------|
| Position Bias | Randomize answer order in A/B comparisons |
| Verbosity Bias | Explicit rubric penalizing unnecessary length |
| Self-Evaluation | Never use model to judge its own output |
| Cost | Support local judge models (Prometheus, fine-tuned Llama) |

**Implementation**:
- Config: `judge_model`, `bias_mitigation: true`
- Calibrate judge against human ratings (target κ > 0.6)
- Log disagreement rate between automated and human scores

- **Date**: 2026-02-02

### TD-012: Hardware Profiling & Thermal Testing

- **Decision**: Add memory bandwidth metrics and thermal soak testing
- **Rationale**: Inference is memory-bandwidth bound, not compute bound; thermal throttling causes 30%+ performance drops

**Memory Bandwidth Utilization (MBU)**:
- Formula: `MBU = (Model_Size_GB * TPS) / Memory_Bandwidth_GB/s`
- Track theoretical max: `Max_TPS = Bandwidth / Model_Size`
- Report actual vs theoretical efficiency

**Soak Test Mode**:
- Duration: 30 minutes continuous inference
- Sample: GPU temp, clock speed every 30 seconds
- Calculate: `Degradation% = (TPS_initial - TPS_final) / TPS_initial`
- Flag models with >15% degradation

**Cold Start Tracking**:
- Separate `model_load_time` from `inference_time`
- Report first-query latency vs subsequent averages

- **Date**: 2026-02-02

### TD-013: Benchmark Contamination Strategy

- **Decision**: Detect and mitigate benchmark contamination effects
- **Rationale**: Research shows 11-22% performance inflation on contaminated benchmarks

**Detection Methods**:
1. Compare performance on original vs fresh variants
2. Check n-gram overlap (50+ chars threshold)
3. Embedding similarity check (0.4-0.8 thresholds)

**De-Emphasis List** (saturated/contaminated):
- MMLU (saturated Sep 2024, 9%+ question error rate)
- HumanEval (164 problems widely seen in training)

**Preferred Fresh Benchmarks**:
- LiveCodeBench (weekly new problems)
- SWE-Bench Verified (500 human-validated)
- GPQA Diamond (PhD-level science)
- Custom/synthetic test generation

- **Date**: 2026-02-02

---

## Template Folder Outputs

Files in `~/Tech_Projects/_HQ/evaluations/` that AI_Eval will create or update:

| File | Action | Frequency |
|------|--------|-----------|
| `MODEL_CATALOG.md` | Update via markers | After each eval run |
| `DECISION_MATRIX.md` | Update via markers | After each eval run |
| `HARDWARE_PROFILES.md` | Update via markers | After each eval run |
| `reports/<model>_<date>.md` | Create new | After each eval run |
| `data/<model>_<date>.json` | Create new | After each eval run |
| `EMBEDDING_CATALOG.md` | Create (Phase 5) | After embedding eval |
| `CHANGELOG.md` | Append | After each eval run |

---

## Quality Standards

- **Testing**: pytest with coverage (`--cov=src --cov-report=term-missing`)
- **Formatting**: black (100 char line length)
- **Linting**: ruff
- **Type checking**: mypy (strict on new code)
- **Security**: bandit + security_scan.sh pre-commit hook
- **Commit format**: `<type>: <description>` (feat/fix/docs/refactor/test/chore/perf)

---

*Last updated: 2026-02-05*