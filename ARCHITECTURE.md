# System Architecture

> Technical architecture documentation for AI_Eval.
> Follows the [C4 Model](https://c4model.com/) with Mermaid.js diagrams.

---

## 1. System Context (C4 Level 1)

How AI_Eval fits into the broader ecosystem.

```mermaid
C4Context
    title System Context — AI_Eval

    Person(user, "Developer", "Evaluates LLMs for project-specific use cases")

    System(ai_eval, "AI_Eval", "LLM evaluation and benchmarking framework")

    System_Ext(ollama, "Ollama", "Local LLM inference server")
    System_Ext(gemini, "Google Gemini API", "Cloud LLM inference")
    System_Ext(hq, "Evaluations Catalog", "Shared model catalog and decision matrix")

    Rel(user, ai_eval, "Runs evaluations", "CLI")
    Rel(ai_eval, ollama, "Generate/Chat", "HTTP API")
    Rel(ai_eval, gemini, "Generate/Chat", "HTTPS API")
    Rel(ai_eval, hq, "Exports results", "Marker-based file updates")
```

---

## 2. Container Architecture (C4 Level 2)

The logical modules that compose the system.

```mermaid
C4Container
    title Container Diagram — AI_Eval

    Person(user, "Developer", "")

    Container(cli, "CLI", "Python / argparse", "7 commands: run, quick-test, compare, list-models, hardware, models, evaluate")

    Container(providers, "Provider Layer", "Python", "BaseProvider abstraction with Ollama and Google implementations")
    Container(benchmarks, "Benchmark Engine", "Python", "Test suite runner with warmup, concurrency, timeout handling")
    Container(scoring, "Scoring Engine", "Python", "pass@k, LLM-as-Judge, RAG metrics via DeepEval")
    Container(profiling, "Hardware Profiler", "Python / psutil", "Apple Silicon, NVIDIA, AMD, CPU detection")
    Container(evaluation, "Evaluation Workflow", "Python / YAML", "Config-driven runner, custom scorers, model discovery")
    Container(reporting, "Reporting", "Python / Jinja2", "Markdown and JSON report generation")
    Container(export, "Catalog Export", "Python", "Marker-based updates to shared catalog files")

    Rel(user, cli, "Invokes", "Terminal")
    Rel(cli, providers, "Selects provider")
    Rel(cli, benchmarks, "Runs benchmarks")
    Rel(cli, evaluation, "Runs evaluations")
    Rel(cli, profiling, "Detects hardware")
    Rel(benchmarks, scoring, "Scores results")
    Rel(evaluation, scoring, "Scores candidates")
    Rel(evaluation, providers, "Queries models")
    Rel(benchmarks, reporting, "Generates reports")
    Rel(reporting, export, "Exports to catalog")
```

---

## 3. Key Design Decisions

Significant architectural choices and their rationale. Full details in [DevPlan.md](DevPlan.md) technical decisions.

| Decision | Choice | Why | Reference |
|----------|--------|-----|-----------|
| Provider abstraction | `BaseProvider` ABC + `ProviderFactory` | Decouple evaluation logic from inference backends; adding providers requires only implementing the interface | [TD-001](DevPlan.md) |
| Marker-based export | `<!-- AI_EVAL:BEGIN -->` / `<!-- AI_EVAL:END -->` delimiters | Update shared catalog files without overwriting manual annotations | [TD-003](DevPlan.md) |
| LLM-as-Judge bias mitigation | Position shuffling, self-evaluation prohibition, multi-eval consensus | Eliminate known biases in LLM judge evaluations | [TD-011](DevPlan.md) |
| Config-driven evaluation | YAML request configs with dataclass validation | Reproducible evaluations; configs stored alongside results | [TD-012](DevPlan.md) |
| Hardware-aware profiling | Auto-detect chip, memory, GPU at benchmark start | Ground results in the hardware that produced them for cross-machine comparison | [TD-006](DevPlan.md) |
| Scoring methodology | Three complementary approaches (pass@k, Judge, RAG) | Different evaluation tasks require different measurement methods | [TD-010](DevPlan.md) |

---

## 4. Data Flow

Primary data flow for a benchmark evaluation run.

```mermaid
flowchart TD
    A[CLI Command] --> B[Load Config]
    B --> C[Detect Hardware]
    B --> D[Initialize Provider]
    C --> E[Benchmark Runner]
    D --> E
    E --> F{For Each Test}
    F --> G[Generate Response]
    G --> H[Score Response]
    H --> F
    F -->|Complete| I[Aggregate Results]
    I --> J[Generate Report]
    J --> K[Export to Catalog]
    K --> L[Update MODEL_CATALOG.md]
    K --> M[Update DECISION_MATRIX.md]
    K --> N[Update HARDWARE_PROFILES.md]
```

### Evaluation Workflow Data Flow

```mermaid
flowchart TD
    A[YAML Request Config] --> B[EvaluationRunner]
    B --> C[Model Discovery]
    C --> D[Filter Candidates]
    D --> E{For Each Candidate}
    E --> F[Run Scenarios]
    F --> G[Apply Custom Scorers]
    G --> H[Check Acceptance Criteria]
    H --> E
    E -->|Complete| I[Rank Results]
    I --> J[Generate Evaluation Report]
```

---

## 5. Security Posture

| Concern | Approach |
|---------|----------|
| API Key Management | Environment variables via `.env` (gitignored); `.env.example` template provided |
| Code Execution | pass@k scorer runs generated code in sandboxed subprocess with timeout enforcement |
| PII Protection | `security_scan.sh` (10 phases) checks for names, emails, paths before commit |
| Data Sanitization | Test fixtures use only synthetic data; no real user data in tracked files |
| Dependency Scanning | `bandit` (static analysis), `pip-audit` (vulnerability DB), `safety` (CVE checks) |
| Pre-commit Hooks | black, ruff, isort, mypy, security scanner run automatically before each commit |
| CI Pipeline | GitHub Actions runs lint, type check, security scan, tests on every push |

See [SECURITY.md](SECURITY.md) for the full security policy.

---

## 6. Technology Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Language | Python 3.12+ | Primary implementation |
| CLI | argparse | Command-line interface |
| Local Inference | Ollama SDK | Local LLM model serving |
| Cloud Inference | google-genai SDK | Google Gemini API access |
| RAG Evaluation | DeepEval | RAGAS metrics (relevancy, faithfulness, precision) |
| Hardware Detection | psutil | Cross-platform system profiling |
| Templating | Jinja2 | Report generation |
| HTTP | httpx | Async HTTP client |
| Terminal Output | rich, tabulate | Colored CLI output |
| Testing | pytest | pytest-asyncio, pytest-cov, pytest-mock |
| Formatting | black (100 chars) | Code formatting |
| Linting | ruff, isort | Fast linting and import sorting |
| Type Checking | mypy | Static type analysis |
| Security | bandit, pip-audit, safety | Vulnerability scanning |
| CI/CD | GitHub Actions | Automated lint, test, security, build |

---

## 7. Component Interaction

Sequence diagram for a typical benchmark run.

```mermaid
sequenceDiagram
    participant U as Developer
    participant CLI as CLI
    participant HW as Hardware Profiler
    participant P as Provider
    participant BR as Benchmark Runner
    participant S as Scoring Engine
    participant R as Report Generator
    participant E as Catalog Exporter

    U->>CLI: python -m src run --config default.yaml
    CLI->>HW: detect_hardware()
    HW-->>CLI: HardwareProfile
    CLI->>P: ProviderFactory.create("ollama")
    P-->>CLI: OllamaProvider
    CLI->>BR: run(provider, config)
    loop Each Test Case
        BR->>P: generate(prompt)
        P-->>BR: GenerationResponse
        BR->>S: score(response, expected)
        S-->>BR: Score
    end
    BR-->>CLI: BenchmarkResult
    CLI->>R: generate_report(result, hardware)
    R-->>CLI: Report (markdown + JSON)
    CLI->>E: export(result)
    E-->>CLI: Catalog files updated
```

---

*This document is updated when architectural decisions change.*
