# Evaluations Directory

This directory stores evaluation request configurations and results. It is **gitignored** because request configs may contain project-specific details, synthetic test data, and results.

## Structure

```
evaluations/
  requests/              # YAML evaluation request configs
    <project>-<task>.yaml
    data/                # Synthetic test data for scenarios
  results/               # Generated evaluation results (JSON)
  README.md              # This file (committed)
```

## Usage

```bash
# Run an evaluation from a request config
python -m src.cli evaluate --config evaluations/requests/my-request.yaml

# Results and reports are generated in reports/ and evaluations/results/
```

## Creating Request Configs

See `src/evaluation/config.py` for the YAML format, dataclass definitions, and examples.
