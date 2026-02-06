# AI_Eval

> LLM evaluation and benchmarking tool for comparing local and API models.

## Features

- **Multi-provider support**: Ollama (local), Google Gemini — Anthropic and OpenAI planned
- **Hardware-aware benchmarks**: Apple Silicon optimized, NVIDIA/AMD GPU detection
- **Scoring methods**: pass@k (code generation), LLM-as-Judge (bias-mitigated), RAG metrics (DeepEval)
- **Fitness profiles**: Weighted scoring for RAG, code-assistant, chat, document-processor, data-pipeline use cases
- **Benchmark runner**: Warmup phase, concurrent execution, category scoring

## Requirements

- Python 3.12+
- macOS (Apple Silicon), Linux, or Windows
- [Ollama](https://ollama.ai/) installed for local models

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd AI_Eval

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate  # macOS/Linux
# or: .venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your API keys (optional for local-only testing)
```

## Quick Start

```bash
# Quick sanity test
python -m src quick-test --provider ollama --model qwen2.5:7b

# Full benchmark
python -m src run --config configs/default.yaml

# Compare models
python -m src compare --models "qwen2.5:7b,llama3:8b" --provider ollama

# Show hardware profile
python -m src hardware
```

## Documentation

| Document | Description |
|----------|-------------|
| [DevPlan.md](DevPlan.md) | Development phases, technical decisions, roadmap |
| [StartHere.md](StartHere.md) | Developer onboarding guide |
| [docs/TEST_SUITE_SPEC.md](docs/TEST_SUITE_SPEC.md) | Test categories and metrics |
| [docs/RESEARCH_SYNTHESIS.md](docs/RESEARCH_SYNTHESIS.md) | Research findings and methodology |

## Project Structure

```
AI_Eval/
├── src/                    # Source code
│   ├── providers/          # LLM provider abstraction (Ollama, Google)
│   ├── benchmarks/         # Benchmark runner and dataset management
│   ├── profiling/          # Hardware detection
│   ├── scoring/            # pass@k, LLM-as-Judge, RAG metrics
│   ├── reporting/          # Report generation (planned)
│   ├── export/             # Catalog export (planned)
│   └── cli.py              # CLI entry point
├── utils/                  # Logging, retry, exceptions, rate limiter, etc.
├── configs/                # Evaluation configurations (YAML)
├── tests/                  # Test suite
├── docs/                   # Specifications and research
├── .github/workflows/      # CI/CD pipelines
├── config.py               # Environment and path configuration
└── DevPlan.md              # Development plan
```

## License

TBD — Private repository, licensing undecided.
