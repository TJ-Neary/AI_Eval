# Contributing to AI_Eval

Thank you for your interest in contributing. This guide covers the development workflow, code standards, and conventions used in this project.

## Development Setup

```bash
git clone https://github.com/TJ-Neary/AI_Eval.git
cd AI_Eval
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt
cp .env.example .env
pre-commit install
```

## Code Standards

- **Formatter**: `black` with 100 character line length
- **Linter**: `ruff` for fast Python linting
- **Import sorting**: `isort` with black-compatible profile
- **Type checking**: `mypy` with strict mode on `src/`
- **Python**: 3.12+, type hints on all function signatures
- **Data structures**: Dataclasses or Pydantic models

```bash
black src/ tests/ --line-length 100
ruff check src/ tests/
mypy src/
```

## Testing

This project follows **test-driven development** (TDD) with the Red-Green-Refactor cycle:

1. Write a failing test that defines the expected behavior
2. Write the minimum code to make the test pass
3. Refactor while keeping tests green

```bash
pytest                          # Full suite with coverage
pytest tests/test_file.py       # Single file
pytest -k "test_name"           # Single test
pytest -m "not slow"            # Skip slow tests
pytest -m "not integration"     # Skip integration tests
```

Test markers: `@pytest.mark.slow`, `@pytest.mark.integration`

## Commit Convention

Use conventional commit format: `<type>: <description>`

| Type | Use For |
|------|---------|
| `feat` | New feature |
| `fix` | Bug fix |
| `docs` | Documentation only |
| `refactor` | Code change that neither fixes a bug nor adds a feature |
| `test` | Adding or updating tests |
| `chore` | Maintenance tasks |
| `perf` | Performance improvement |

## Pre-commit Hooks

Pre-commit hooks run automatically on `git commit`:

- **black** — code formatting
- **ruff** — linting
- **isort** — import sorting
- **mypy** — type checking

Install with `pre-commit install`. To run manually: `pre-commit run --all-files`.

## Security

Read [SECURITY.md](SECURITY.md) before committing. Key rules:

- No API keys, passwords, or secrets in tracked files
- No real names, emails, or PII in code or test fixtures
- No hardcoded absolute paths (use `config.py` path resolution)
- Run `./security_scan.sh` before pushing

## Adding a New Provider

To add support for a new LLM backend:

1. Create `src/providers/your_provider.py`
2. Implement the `BaseProvider` abstract class:
   - `generate()` — single prompt completion
   - `chat()` — multi-turn conversation
   - `get_model_info()` — model metadata
   - `get_available_models()` — list available models
3. Register in `ProviderFactory` in `src/providers/base.py`
4. Add tests in `tests/test_your_provider.py`
5. Update `ProviderType` enum if needed
