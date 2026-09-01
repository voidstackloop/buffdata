import asyncio
from pathlib import Path
from typing import Optional
import typer
import yaml
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from buffdata.engine.checkpoint import CheckpointManager
from buffdata.engine.client import GeminiClient
from buffdata.engine.limiter import AsyncRateLimiter
from buffdata.models.formats import read_dataset, write_dataset
from buffdata.models.schemas import DatasetItem, PipelineConfig
from buffdata.optimizers.dedup import Deduplicator
from buffdata.optimizers.evolver import DataEvolver
from buffdata.optimizers.preference import PreferenceBuilder
from buffdata.optimizers.refiner import DataRefiner
from buffdata.optimizers.scorer import QualityScorer
from buffdata.report.generator import ReportGenerator

app = typer.Typer(
    name="buffdata",
    help="🚀 buffdata: AI Training Data Optimizer powered by Google Gemini API",
    add_completion=False,
)
console = Console()


@app.command("score")
def score_cmd(
    input_file: Path = typer.Argument(..., help="Path to input dataset (.jsonl, .json, .parquet, .csv)"),
    output_file: Optional[Path] = typer.Option(None, "-o", "--output", help="Path to output dataset"),
    min_score: float = typer.Option(7.0, "--min-score", help="Filter out samples below this quality score"),
    filter_data: bool = typer.Option(False, "--filter", help="Drop items that do not meet min_score"),
    model: str = typer.Option("gemini-3.7-flash", "--model", help="Gemini model for evaluation"),
    concurrency: int = typer.Option(10, "--concurrency", help="Max concurrent requests"),
    max_rpm: int = typer.Option(60, "--rpm", help="Max requests per minute"),
    max_rows: Optional[int] = typer.Option(None, "--max-rows", help="Limit number of rows to process"),
):
    """Score and evaluate training dataset quality using Gemini LLM-as-a-judge."""
    console.print(Panel(f"[bold cyan]BuffData Scorer[/bold cyan]\nInput: {input_file}\nModel: {model}", border_style="cyan"))

    items = read_dataset(input_file, max_rows=max_rows)
    console.print(f"[green]Loaded {len(items)} items.[/green]")

    client = GeminiClient(default_model=model)
    limiter = AsyncRateLimiter(max_rpm=max_rpm, concurrency=concurrency)
    scorer = QualityScorer(client=client, limiter=limiter, model=model)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
    ) as progress:
        task = progress.add_task("[cyan]Scoring dataset items...", total=len(items))

        def on_step():
            progress.advance(task, 1)

        scored_items = asyncio.run(scorer.score_batch_async(items, on_progress=on_step))

    kept, dropped = scorer.filter_items(scored_items, min_score=min_score)

    table = Table(title="Scoring Summary", border_style="cyan")
    table.add_column("Category", style="bold")
    table.add_column("Count", justify="right")
    table.add_row("Total Processed", str(len(items)))
    table.add_row("Passed (>= min_score)", f"[green]{len(kept)}[/green]")
    table.add_row("Dropped (< min_score)", f"[red]{len(dropped)}[/red]")
    console.print(table)

    save_items = kept if filter_data else scored_items
    if output_file:
        write_dataset(save_items, output_file)
        console.print(f"[bold green]✓ Wrote {len(save_items)} items to {output_file}[/bold green]")


@app.command("refine")
def refine_cmd(
    input_file: Path = typer.Argument(..., help="Path to input dataset"),
    output_file: Path = typer.Option(..., "-o", "--output", help="Path to output dataset"),
    mode: str = typer.Option("all", "--mode", help="Refine mode: all, response_only, prompt_only"),
    model: str = typer.Option("gemini-3.7-flash", "--model", help="Gemini model to use"),
    concurrency: int = typer.Option(10, "--concurrency", help="Max concurrent requests"),
    max_rpm: int = typer.Option(60, "--rpm", help="Max requests per minute"),
    max_rows: Optional[int] = typer.Option(None, "--max-rows", help="Limit number of rows to process"),
):
    """Clean, expand reasoning, fix syntax, and strip AI boilerplate from dataset."""
    console.print(Panel(f"[bold magenta]BuffData Refiner[/bold magenta]\nInput: {input_file}\nMode: {mode}", border_style="magenta"))

    items = read_dataset(input_file, max_rows=max_rows)
    client = GeminiClient(default_model=model)
    limiter = AsyncRateLimiter(max_rpm=max_rpm, concurrency=concurrency)
    refiner = DataRefiner(client=client, limiter=limiter, model=model)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
    ) as progress:
        task = progress.add_task("[magenta]Refining items...", total=len(items))

        def on_step():
            progress.advance(task, 1)

        refined = asyncio.run(refiner.refine_batch_async(items, mode=mode, on_progress=on_step))

    write_dataset(refined, output_file)
    console.print(f"[bold green]✓ Successfully refined and saved {len(refined)} items to {output_file}[/bold green]")


@app.command("evolve")
def evolve_cmd(
    input_file: Path = typer.Argument(..., help="Path to input dataset"),
    output_file: Path = typer.Option(..., "-o", "--output", help="Path to output dataset"),
    strategy: str = typer.Option("deepen_reasoning", "--strategy", help="deepen_reasoning, add_constraints, concretize, in_breadth"),
    model: str = typer.Option("gemini-3.7-flash", "--model", help="Gemini model to use"),
    concurrency: int = typer.Option(10, "--concurrency", help="Max concurrent requests"),
    max_rpm: int = typer.Option(60, "--rpm", help="Max requests per minute"),
    max_rows: Optional[int] = typer.Option(None, "--max-rows", help="Limit number of rows to process"),
):
    """Evolve prompts into higher complexity and reasoning depth (Evol-Instruct)."""
    console.print(Panel(f"[bold blue]BuffData Evolver[/bold blue]\nStrategy: {strategy}", border_style="blue"))

    items = read_dataset(input_file, max_rows=max_rows)
    client = GeminiClient(default_model=model)
    limiter = AsyncRateLimiter(max_rpm=max_rpm, concurrency=concurrency)
    evolver = DataEvolver(client=client, limiter=limiter, model=model)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
    ) as progress:
        task = progress.add_task("[blue]Evolving dataset items...", total=len(items))

        def on_step():
            progress.advance(task, 1)

        evolved = asyncio.run(evolver.evolve_batch_async(items, strategies=[strategy], on_progress=on_step))

    write_dataset(evolved, output_file)
    console.print(f"[bold green]✓ Successfully evolved and saved {len(evolved)} items to {output_file}[/bold green]")


@app.command("dpo")
def dpo_cmd(
    input_file: Path = typer.Argument(..., help="Path to input SFT dataset"),
    output_file: Path = typer.Option(..., "-o", "--output", help="Path to output DPO dataset"),
    model: str = typer.Option("gemini-3.7-flash", "--model", help="Gemini model to use"),
    concurrency: int = typer.Option(10, "--concurrency", help="Max concurrent requests"),
    max_rpm: int = typer.Option(60, "--rpm", help="Max requests per minute"),
    max_rows: Optional[int] = typer.Option(None, "--max-rows", help="Limit number of rows to process"),
):
    """Generate chosen and rejected preference pairs for DPO/RLHF alignment."""
    console.print(Panel(f"[bold yellow]BuffData DPO Preference Builder[/bold yellow]", border_style="yellow"))

    items = read_dataset(input_file, max_rows=max_rows)
    client = GeminiClient(default_model=model)
    limiter = AsyncRateLimiter(max_rpm=max_rpm, concurrency=concurrency)
    dpo_builder = PreferenceBuilder(client=client, limiter=limiter, model=model)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
    ) as progress:
        task = progress.add_task("[yellow]Synthesizing DPO pairs...", total=len(items))

        def on_step():
            progress.advance(task, 1)

        dpo_items = asyncio.run(dpo_builder.build_dpo_batch_async(items, on_progress=on_step))

    write_dataset(dpo_items, output_file)
    console.print(f"[bold green]✓ Successfully generated {len(dpo_items)} DPO pairs to {output_file}[/bold green]")


@app.command("dedup")
def dedup_cmd(
    input_file: Path = typer.Argument(..., help="Path to input dataset"),
    output_file: Path = typer.Option(..., "-o", "--output", help="Path to output deduplicated dataset"),
    method: str = typer.Option("minhash", "--method", help="exact, minhash, semantic"),
    threshold: float = typer.Option(0.85, "--threshold", help="Similarity threshold (0.0 to 1.0)"),
):
    """Remove exact, lexical, or semantic duplicates from the dataset."""
    console.print(Panel(f"[bold red]BuffData Deduplicator[/bold red]\nMethod: {method}\nThreshold: {threshold}", border_style="red"))

    items = read_dataset(input_file)
    deduper = Deduplicator()

    if method == "exact":
        kept, dropped = deduper.deduplicate_exact(items)
    elif method == "minhash":
        kept, dropped = deduper.deduplicate_minhash(items, threshold=threshold)
    elif method == "semantic":
        kept, dropped = deduper.deduplicate_semantic(items, threshold=threshold)
    else:
        raise ValueError(f"Unknown dedup method: {method}")

    console.print(f"[green]Kept: {len(kept)} items[/green] | [red]Dropped Duplicates: {len(dropped)} items[/red]")
    write_dataset(kept, output_file)
    console.print(f"[bold green]✓ Wrote deduplicated dataset to {output_file}[/bold green]")


@app.command("stats")
def stats_cmd(
    input_file: Path = typer.Argument(..., help="Path to dataset to inspect"),
):
    """Display statistics and quality metrics of a dataset."""
    items = read_dataset(input_file)
    summary_md = ReportGenerator.generate_markdown_summary(items)
    console.print(summary_md)


@app.command("report")
def report_cmd(
    input_file: Path = typer.Argument(..., help="Path to dataset with quality scores"),
    output_html: Path = typer.Option(Path("report.html"), "-o", "--output", help="Output HTML file path"),
):
    """Generate interactive HTML audit report."""
    items = read_dataset(input_file)
    out = ReportGenerator.generate_html_report(items, output_html)
    console.print(f"[bold green]✓ Generated interactive HTML audit report at {out.resolve()}[/bold green]")


@app.command("pipeline")
def pipeline_cmd(
    config_file: Path = typer.Argument(..., help="Path to pipeline YAML configuration file"),
    input_file: Path = typer.Option(..., "-i", "--input", help="Input dataset path"),
    output_file: Path = typer.Option(..., "-o", "--output", help="Final output dataset path"),
):
    """Execute a full declarative multi-stage optimization pipeline."""
    console.print(Panel(f"[bold green]BuffData Multi-Stage Pipeline[/bold green]\nConfig: {config_file}", border_style="green"))

    with open(config_file, "r", encoding="utf-8") as f:
        cfg_dict = yaml.safe_load(f)
    config = PipelineConfig(**cfg_dict)

    items = read_dataset(input_file)
    console.print(f"[cyan]Loaded {len(items)} initial items.[/cyan]")

    client = GeminiClient(default_model=config.model, embedding_model=config.embedding_model)
    limiter = AsyncRateLimiter(max_rpm=config.max_rpm, concurrency=config.concurrency)

    # 1. Dedup
    deduper = Deduplicator(client=client)
    items, dropped_dups = deduper.deduplicate_minhash(items, threshold=config.dedup_threshold)
    console.print(f"[bold]Step 1 (Dedup):[/bold] {len(items)} unique items ({len(dropped_dups)} dropped)")

    # 2. Refine
    refiner = DataRefiner(client=client, limiter=limiter, model=config.model)
    items = asyncio.run(refiner.refine_batch_async(items, mode=config.refine_mode))
    console.print(f"[bold]Step 2 (Refine):[/bold] {len(items)} items refined")

    # 3. Score & Filter
    scorer = QualityScorer(client=client, limiter=limiter, model=config.model)
    scored = asyncio.run(scorer.score_batch_async(items))
    kept, dropped_low = scorer.filter_items(scored, min_score=config.filter_min_score)
    console.print(f"[bold]Step 3 (Scoring & Filtering):[/bold] {len(kept)} items passed ({len(dropped_low)} filtered out)")

    write_dataset(kept, output_file)
    console.print(f"[bold green]🎉 Pipeline completed! Saved {len(kept)} high-quality items to {output_file}[/bold green]")
