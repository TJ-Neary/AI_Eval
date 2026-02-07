"""
AI_Eval CLI

Command-line interface for running LLM benchmarks.

Usage:
    # Quick test a single model
    python -m src.cli quick-test --model qwen2.5:32b

    # Run full benchmark
    python -m src.cli run --model qwen2.5:32b

    # Compare models
    python -m src.cli compare --models qwen2.5:32b,llama3.2:3b

    # List available models
    python -m src.cli list-models
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table

console = Console()


async def cmd_quick_test(args: argparse.Namespace) -> int:
    """Run a quick single-prompt test."""
    from .benchmarks import BenchmarkRunner
    from .providers import GoogleProvider, OllamaProvider, ProviderFactory

    console.print(f"[cyan]Quick test: {args.model}[/cyan]")

    # Create provider
    if args.provider == "ollama":
        provider = OllamaProvider(model=args.model)
    elif args.provider == "google":
        provider = GoogleProvider(model=args.model)
    else:
        provider = ProviderFactory.create(args.provider, model=args.model)

    # Check health
    if not await provider.health_check():
        console.print(f"[red]Error: Could not connect to {args.provider}[/red]")
        return 1

    # Run test
    runner = BenchmarkRunner()
    result = await runner.quick_test(provider, prompt=args.prompt)

    # Display results
    console.print(f"\n[bold]Model:[/bold] {result['model']}")
    console.print(f"[bold]Prompt:[/bold] {result['prompt']}")
    console.print(f"\n[bold]Response:[/bold]\n{result['response']}")
    console.print("\n[dim]───────────────────────────────────────[/dim]")
    console.print(f"[green]Tokens/sec:[/green] {result['tokens_per_second']:.1f}")
    console.print(f"[green]Generation time:[/green] {result['generation_time_ms']:.0f}ms")
    console.print(
        f"[green]Tokens:[/green] {result['prompt_tokens']} prompt + {result['completion_tokens']} completion"
    )

    return 0


async def cmd_run(args: argparse.Namespace) -> int:
    """Run a full benchmark."""
    from .benchmarks import QUICK_TEST_DATASET, BenchmarkRunner, RunConfig
    from .profiling import detect_hardware
    from .providers import GoogleProvider, OllamaProvider

    console.print(f"[cyan]Benchmark: {args.model}[/cyan]")

    # Detect hardware
    hardware = detect_hardware()
    console.print(f"[dim]Hardware: {hardware.chip_name} ({hardware.ram_gb:.0f}GB RAM)[/dim]")

    # Create provider
    if args.provider == "ollama":
        provider = OllamaProvider(model=args.model)
    elif args.provider == "google":
        provider = GoogleProvider(model=args.model)
    else:
        console.print(f"[red]Unknown provider: {args.provider}[/red]")
        return 1

    # Check health
    if not await provider.health_check():
        console.print(f"[red]Error: Could not connect to {args.provider}[/red]")
        return 1

    # Configure run
    config = RunConfig(
        warmup_queries=args.warmup,
        repetitions=args.repetitions,
        timeout_seconds=args.timeout,
        use_llm_judge=not args.no_judge,
        verbose=True,
    )

    # Run benchmark
    runner = BenchmarkRunner()
    result = await runner.run(
        provider=provider,
        dataset=QUICK_TEST_DATASET,  # TODO: Load from config
        config=config,
    )

    # Save raw JSON results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(result.to_dict(), f, indent=2)
        console.print(f"\n[green]Results saved to {output_path}[/green]")

    # Generate report
    report_path = None
    if not args.no_report:
        from .reporting import ReportConfig, ReportGenerator

        report_config = ReportConfig(
            report_dir=Path(args.report_dir),
            formats=["markdown", "json"],
        )
        generator = ReportGenerator(config=report_config)
        report_path = generator.generate(result)
        console.print(f"[green]Report saved to {report_path}[/green]")

    # Update README with results table
    if not args.no_readme:
        from .reporting import update_readme_results

        readme_path = Path("README.md")
        if update_readme_results(result, readme_path=readme_path, report_path=report_path):
            console.print("[green]README.md updated with results[/green]")

    # Export to _HQ catalog
    if not args.no_export:
        from .export import ExportConfig, export_to_catalog

        export_config = ExportConfig()
        outcomes = export_to_catalog(result, config=export_config, report_path=report_path)
        if any(outcomes.values()):
            updated = [k for k, v in outcomes.items() if v]
            console.print(f"[green]Catalog updated: {', '.join(updated)}[/green]")

    return 0


async def cmd_list_models(args: argparse.Namespace) -> int:
    """List available models."""
    from .providers import OllamaProvider

    console.print("[cyan]Available Models[/cyan]\n")

    # Ollama models
    provider = OllamaProvider(model="")
    if await provider.health_check():
        models = await provider.list_models()

        table = Table(title="Ollama Models")
        table.add_column("Model", style="cyan")
        table.add_column("Size", style="green")

        for model in models:
            # Parse size from model name if present
            size = ""
            if ":" in model:
                parts = model.split(":")
                if len(parts) > 1:
                    size = parts[1]
            table.add_row(model, size)

        console.print(table)
    else:
        console.print("[yellow]Ollama not available[/yellow]")

    return 0


async def cmd_compare(args: argparse.Namespace) -> int:
    """Compare multiple models."""
    from .benchmarks import QUICK_TEST_DATASET, BenchmarkRunner, RunConfig
    from .providers import OllamaProvider

    models = [m.strip() for m in args.models.split(",")]
    console.print(f"[cyan]Comparing: {', '.join(models)}[/cyan]\n")

    runner = BenchmarkRunner()
    config = RunConfig(
        warmup_queries=2,
        repetitions=1,
        use_llm_judge=not args.no_judge,
        verbose=False,
    )

    results = []
    for model in models:
        console.print(f"[dim]Testing {model}...[/dim]")
        provider = OllamaProvider(model=model)

        if not await provider.health_check():
            console.print(f"[yellow]Skipping {model} (not available)[/yellow]")
            continue

        result = await runner.run(
            provider=provider,
            dataset=QUICK_TEST_DATASET,
            config=config,
        )
        results.append(result)

    # Display comparison table
    if results:
        table = Table(title="Model Comparison")
        table.add_column("Model", style="cyan")
        table.add_column("Score", style="green")
        table.add_column("Tokens/sec", style="yellow")
        table.add_column("Pass Rate", style="blue")

        for r in sorted(results, key=lambda x: x.overall_score, reverse=True):
            table.add_row(
                r.model,
                f"{r.overall_score:.1f}",
                f"{r.avg_tokens_per_second:.1f}",
                f"{r.total_passed}/{r.total_tests}",
            )

        console.print(table)

    return 0


async def cmd_hardware(args: argparse.Namespace) -> int:
    """Show hardware profile."""
    from .profiling import detect_hardware

    profile = detect_hardware()

    console.print("[cyan]Hardware Profile[/cyan]\n")
    console.print(f"[bold]System:[/bold] {profile.hostname}")
    console.print(f"[bold]OS:[/bold] {profile.os_name} {profile.os_version}")
    console.print(f"[bold]Platform:[/bold] {profile.platform}")
    console.print()
    console.print(f"[bold]Chip:[/bold] {profile.chip_name}")
    console.print(f"[bold]Type:[/bold] {profile.chip_type.name}")
    console.print(f"[bold]GPU Cores:[/bold] {profile.gpu_cores}")
    console.print(f"[bold]Neural Engine:[/bold] {profile.neural_engine_cores} cores")
    console.print()
    console.print(
        f"[bold]RAM:[/bold] {profile.ram_gb:.1f}GB total, {profile.ram_available_gb:.1f}GB available"
    )
    console.print(f"[bold]Memory Bandwidth:[/bold] {profile.memory_bandwidth_gbps:.0f} GB/s")
    console.print()
    console.print("[bold]Capabilities:[/bold]")
    console.print(f"  MPS: {'✓' if profile.supports_mps else '✗'}")
    console.print(f"  MLX: {'✓' if profile.supports_mlx else '✗'}")
    console.print(f"  CUDA: {'✓' if profile.supports_cuda else '✗'}")
    console.print()

    # Model recommendations
    console.print("[bold]Model Recommendations:[/bold]")
    sizes = ["7B", "13B", "32B", "70B"]
    for size in sizes:
        can_run = profile.can_run_model_size(size)
        quant = profile.recommended_quantization(size)
        status = "✓" if can_run else "✗"
        console.print(f"  {size}: {status} ({quant})")

    return 0


def main() -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="ai-eval",
        description="LLM Evaluation and Benchmarking Tool",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # quick-test
    quick_parser = subparsers.add_parser("quick-test", help="Run a quick single-prompt test")
    quick_parser.add_argument("--model", "-m", required=True, help="Model name")
    quick_parser.add_argument(
        "--provider", "-p", default="ollama", help="Provider (ollama, google)"
    )
    quick_parser.add_argument("--prompt", default="What is 2 + 2?", help="Test prompt")

    # run
    run_parser = subparsers.add_parser("run", help="Run a full benchmark")
    run_parser.add_argument("--model", "-m", required=True, help="Model name")
    run_parser.add_argument("--provider", "-p", default="ollama", help="Provider (ollama, google)")
    run_parser.add_argument("--config", "-c", help="Path to config YAML")
    run_parser.add_argument("--output", "-o", help="Output path for results JSON")
    run_parser.add_argument("--warmup", type=int, default=3, help="Warmup queries")
    run_parser.add_argument("--repetitions", type=int, default=1, help="Repetitions per test")
    run_parser.add_argument("--timeout", type=float, default=120, help="Timeout in seconds")
    run_parser.add_argument("--no-judge", action="store_true", help="Disable LLM-as-Judge")
    run_parser.add_argument("--no-report", action="store_true", help="Skip report generation")
    run_parser.add_argument("--no-readme", action="store_true", help="Skip README update")
    run_parser.add_argument("--report-dir", default="./reports", help="Report output directory")
    run_parser.add_argument("--no-export", action="store_true", help="Skip catalog export to _HQ")

    # compare
    compare_parser = subparsers.add_parser("compare", help="Compare multiple models")
    compare_parser.add_argument("--models", "-m", required=True, help="Comma-separated model names")
    compare_parser.add_argument("--no-judge", action="store_true", help="Disable LLM-as-Judge")

    # list-models
    subparsers.add_parser("list-models", help="List available models")

    # hardware
    subparsers.add_parser("hardware", help="Show hardware profile")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 0

    # Run async command
    if args.command == "quick-test":
        return asyncio.run(cmd_quick_test(args))
    elif args.command == "run":
        return asyncio.run(cmd_run(args))
    elif args.command == "compare":
        return asyncio.run(cmd_compare(args))
    elif args.command == "list-models":
        return asyncio.run(cmd_list_models(args))
    elif args.command == "hardware":
        return asyncio.run(cmd_hardware(args))
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
