# Benchmark Reports

This directory contains generated benchmark reports from AI_Eval runs.

## File Format

Each benchmark run generates two files:

- **`<model>_<timestamp>.md`** — Human-readable markdown report with hardware profile, category breakdown, fitness scores, and test details.
- **`<model>_<timestamp>.json`** — Machine-readable raw data for downstream processing.

## Results Index

The `.results_index.json` file tracks accumulated results across runs. It is used to render the summary table in the project README.

## Generating Reports

```bash
# Run a benchmark (automatically generates report)
python -m src run --provider ollama --model qwen2.5:32b

# Skip report generation
python -m src run --provider ollama --model qwen2.5:32b --no-report

# Skip README update
python -m src run --provider ollama --model qwen2.5:32b --no-readme

# Custom report directory
python -m src run --provider ollama --model qwen2.5:32b --report-dir ./custom_reports
```

## Note

All report files except this README are gitignored. The results summary is embedded in the project [README.md](../README.md) between `<!-- AI_EVAL:BEGIN -->` / `<!-- AI_EVAL:END -->` markers.
