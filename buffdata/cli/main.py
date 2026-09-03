import asyncio
import os
from pathlib import Path
from typing import List, Optional
import typer
import yaml
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from buffdata.engine.checkpoint import CheckpointManager
from buffdata.engine.client import create_llm_client
from buffdata.engine.limiter import AsyncRateLimiter
from buffdata.models.formats import _is_cloud_url, read_dataset, write_dataset, write_dataset_atomic
from buffdata.models.schemas import ClassificationMode, DatasetItem, PipelineConfig
from buffdata.optimizers.dedup import Deduplicator
from buffdata.optimizers.evolver import DataEvolver
from buffdata.optimizers.preference import PreferenceBuilder
from buffdata.optimizers.refiner import DataRefiner
from buffdata.optimizers.scorer import QualityScorer
from buffdata.report.generator import ReportGenerator

app = typer.Typer(
    name="buffdata",
    help="🚀 buffdata: provider-neutral AI training-data optimizer",
    add_completion=False,
)
productivity_app = typer.Typer(help="Documentation and code-graph tools for productive dataset work.")
app.add_typer(productivity_app, name="productivity")
from buffdata.runs.cli import app as runs_app
app.add_typer(runs_app, name="runs")
console = Console()


@app.callback()
def security_options(
    ctx: typer.Context,
    security_policy: Optional[Path] = typer.Option(None, "--security-policy", envvar="BUFFDATA_SECURITY_POLICY"),
    identity_token_file: Optional[str] = typer.Option(None, "--identity-token-file", help="JWT file, or - for stdin; never put tokens in argv"),
    identity_oidc_config: Optional[Path] = typer.Option(None, "--identity-oidc-config"),
    identity_policy: Optional[Path] = typer.Option(None, "--identity-policy"),
):
    """Optional managed boundary applying to every legacy command, including pipeline."""
    from buffdata.security.policy import ExecutionContext, SecurityPolicy, execution_context, SecretFilter
    import logging
    if identity_token_file or identity_oidc_config or identity_policy:
        if not all([identity_token_file, identity_oidc_config, identity_policy]):
            raise typer.BadParameter("Identity token file, OIDC configuration, and access policy must be supplied together")
        from buffdata.runs.cli import read_token
        _enforce_access_policy(None, identity_policy, "unrestricted", bearer_token=read_token(identity_token_file),
                               oidc_config_file=identity_oidc_config)
    if security_policy:
        policy = SecurityPolicy(**(yaml.safe_load(security_policy.read_text()) or {}))
        ctx.with_resource(execution_context(ExecutionContext(policy=policy)))
    for handler in logging.getLogger().handlers:
        handler.addFilter(SecretFilter())


def _enforce_access_policy(
    actor: Optional[str],
    policy_file: Optional[Path],
    network_policy: str,
    *,
    bearer_token: Optional[str] = None,
    oidc_config_file: Optional[Path] = None,
) -> None:
    """Opt-in authorization check: does nothing unless an identity (--actor, or a verified
    --bearer-token) and --policy are given, so every command's default behavior is
    completely unchanged.

    Two ways to establish identity:
      - --actor <name>: trust-based -- you're asserting who's running this.
      - --bearer-token <jwt> --oidc-config <file>: verified -- the token's signature,
        issuer, audience, and expiry are checked against the IdP's real JWKS
        (buffdata.governance.oidc) before its claims are trusted for anything. If both are
        given, the verified identity must match --actor, or this refuses to guess which one
        is right.

    Once identity is established, every actor needs the 'run' permission, and using
    anything other than network_policy=strict (i.e., actually reaching an external
    provider) additionally needs 'use_external_providers' -- letting a policy restrict some
    actors to the zero-network guarantee from buffdata.engine.client.NetworkForbiddenClient
    regardless of what CLI flags they pass.
    """
    verified_actor: Optional[str] = None
    if bearer_token is not None or oidc_config_file is not None:
        if bearer_token is None or oidc_config_file is None:
            raise typer.BadParameter("--bearer-token and --oidc-config must be given together.")
        from buffdata.governance import OIDCVerificationError, load_oidc_config, verify_bearer_token

        oidc_config = load_oidc_config(oidc_config_file)
        try:
            identity = verify_bearer_token(bearer_token, config=oidc_config)
        except OIDCVerificationError as exc:
            raise typer.BadParameter(f"Bearer token verification failed: {exc}") from exc
        verified_actor = identity.actor
        if actor is not None and actor != verified_actor:
            raise typer.BadParameter(
                f"--actor '{actor}' does not match the identity verified from --bearer-token "
                f"('{verified_actor}')."
            )

    resolved_actor = verified_actor or actor
    if resolved_actor is None and policy_file is None:
        return
    if resolved_actor is None or policy_file is None:
        raise typer.BadParameter(
            "An identity (--actor, or --bearer-token together with --oidc-config) and --policy "
            "must be given together."
        )
    from buffdata.governance import Policy, check_permission

    policy = Policy.from_yaml(policy_file)
    check_permission(policy, resolved_actor, "run")
    if network_policy != "strict":
        check_permission(policy, resolved_actor, "use_external_providers")


@productivity_app.command("graph-build")
def graph_build_cmd(
    project_path: Path = typer.Argument(Path("."), help="Repository to map into a local knowledge graph"),
):
    """Build a local AST-only Graphify knowledge graph for this repository."""
    from buffdata.integrations.graphify import GraphifyManager

    graph_path, _ = GraphifyManager().build(project_path)
    console.print(f"[bold green]Graph ready:[/bold green] {graph_path}")
    console.print("Query it with: buffdata productivity graph-query \"how does optimization reach scoring?\"")


@productivity_app.command("graph-query")
def graph_query_cmd(
    question: str = typer.Argument(..., help="Natural-language question about the codebase"),
    project_path: Path = typer.Option(Path("."), "--project", help="Repository containing graphify-out/graph.json"),
):
    """Answer a codebase question through the local Graphify graph."""
    from buffdata.integrations.graphify import GraphifyManager

    console.print(GraphifyManager().query(question, project_path))


@productivity_app.command("docs-search")
def docs_search_cmd(query: str = typer.Argument(..., help="Library or framework name to resolve")):
    """Search Context7 for a version-specific documentation library id."""
    from buffdata.integrations.context7 import Context7Client

    console.print_json(data=Context7Client().search_libraries(query))


@productivity_app.command("docs-context")
def docs_context_cmd(
    library_id: str = typer.Argument(..., help="Context7 library id, for example /pydantic/pydantic"),
    query: str = typer.Argument(..., help="Focused documentation question"),
):
    """Retrieve focused current documentation from Context7."""
    from buffdata.integrations.context7 import Context7Client

    console.print_json(data=Context7Client().get_context(library_id, query))


@app.command("score")
def score_cmd(
    input_file: str = typer.Argument(..., help="Path to input dataset, or a cloud URL (s3://, gs://, az://, ...)"),
    output_file: Optional[str] = typer.Option(None, "-o", "--output", help="Path to output dataset, or a cloud URL"),
    min_score: float = typer.Option(7.0, "--min-score", help="Filter out samples below this quality score"),
    filter_data: bool = typer.Option(False, "--filter", help="Drop items that do not meet min_score"),
    provider: str = typer.Option("gemini", "--provider", help="gemini, openai, anthropic, a private endpoint (azure_openai, bedrock_anthropic, openai_compatible), or a local LLM server (ollama, lmstudio, vllm, llamacpp)"),
    base_url: Optional[str] = typer.Option(None, "--base-url", help="Endpoint for openai_compatible/local-server providers, e.g. http://192.168.1.50:11434/v1 for another machine's Ollama -- overrides that provider's default local port"),
    network_policy: str = typer.Option("unrestricted", "--network-policy", help="unrestricted; local to allow only a provider whose endpoint you control on a loopback/private address; or strict to guarantee zero network calls at all"),
    actor: Optional[str] = typer.Option(None, "--actor", help="Check this actor's permissions against --policy before running (requires --policy too)"),
    policy_file: Optional[Path] = typer.Option(None, "--policy", help="Access policy YAML to check --actor against (requires --actor too)"),
    bearer_token: Optional[str] = typer.Option(None, "--bearer-token", help="OIDC bearer token proving identity, verified against --oidc-config before --policy is checked (requires --oidc-config too)"),
    oidc_config_file: Optional[Path] = typer.Option(None, "--oidc-config", help="OIDC issuer/audience/JWKS config YAML to verify --bearer-token against (requires --bearer-token too)"),
    model: Optional[str] = typer.Option(None, "--model", help="Provider model override"),
    concurrency: int = typer.Option(10, "--concurrency", help="Max concurrent requests"),
    max_rpm: int = typer.Option(60, "--rpm", help="Max requests per minute"),
    max_rows: Optional[int] = typer.Option(None, "--max-rows", help="Limit number of rows to process"),
):
    """Score and evaluate training dataset quality using an LLM-as-a-judge."""
    _enforce_access_policy(
        actor, policy_file, network_policy, bearer_token=bearer_token, oidc_config_file=oidc_config_file
    )
    client = create_llm_client(provider, model, base_url=base_url, network_policy=network_policy)
    model = model or client.default_model
    console.print(Panel(f"[bold cyan]BuffData Scorer[/bold cyan]\nInput: {input_file}\nProvider: {provider}\nModel: {model}", border_style="cyan"))

    items = read_dataset(input_file, max_rows=max_rows)
    console.print(f"[green]Loaded {len(items)} items.[/green]")

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
    input_file: str = typer.Argument(..., help="Path to input dataset, or a cloud URL (s3://, gs://, az://, ...)"),
    output_file: str = typer.Option(..., "-o", "--output", help="Path to output dataset, or a cloud URL"),
    mode: str = typer.Option("all", "--mode", help="Refine mode: all, response_only, prompt_only"),
    provider: str = typer.Option("gemini", "--provider", help="gemini, openai, anthropic, a private endpoint (azure_openai, bedrock_anthropic, openai_compatible), or a local LLM server (ollama, lmstudio, vllm, llamacpp)"),
    base_url: Optional[str] = typer.Option(None, "--base-url", help="Endpoint for openai_compatible/local-server providers, e.g. http://192.168.1.50:11434/v1 for another machine's Ollama -- overrides that provider's default local port"),
    network_policy: str = typer.Option("unrestricted", "--network-policy", help="unrestricted; local to allow only a provider whose endpoint you control on a loopback/private address; or strict to guarantee zero network calls at all"),
    actor: Optional[str] = typer.Option(None, "--actor", help="Check this actor's permissions against --policy before running (requires --policy too)"),
    policy_file: Optional[Path] = typer.Option(None, "--policy", help="Access policy YAML to check --actor against (requires --actor too)"),
    bearer_token: Optional[str] = typer.Option(None, "--bearer-token", help="OIDC bearer token proving identity, verified against --oidc-config before --policy is checked (requires --oidc-config too)"),
    oidc_config_file: Optional[Path] = typer.Option(None, "--oidc-config", help="OIDC issuer/audience/JWKS config YAML to verify --bearer-token against (requires --bearer-token too)"),
    model: Optional[str] = typer.Option(None, "--model", help="Provider model override"),
    concurrency: int = typer.Option(10, "--concurrency", help="Max concurrent requests"),
    max_rpm: int = typer.Option(60, "--rpm", help="Max requests per minute"),
    batch_size: int = typer.Option(20, "--batch-size", min=1, max=100, help="Records grouped into one structured request"),
    max_rows: Optional[int] = typer.Option(None, "--max-rows", help="Limit number of rows to process"),
):
    """Clean, expand reasoning, fix syntax, and strip AI boilerplate from dataset."""
    console.print(Panel(f"[bold magenta]BuffData Refiner[/bold magenta]\nInput: {input_file}\nMode: {mode}", border_style="magenta"))

    items = read_dataset(input_file, max_rows=max_rows)
    _enforce_access_policy(
        actor, policy_file, network_policy, bearer_token=bearer_token, oidc_config_file=oidc_config_file
    )
    client = create_llm_client(provider, model, base_url=base_url, network_policy=network_policy)
    model = model or client.default_model
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

        refined = asyncio.run(refiner.refine_batch_async(items, mode=mode, batch_size=batch_size, on_progress=on_step))

    write_dataset(refined, output_file)
    console.print(f"[bold green]✓ Successfully refined and saved {len(refined)} items to {output_file}[/bold green]")


@app.command("evolve")
def evolve_cmd(
    input_file: str = typer.Argument(..., help="Path to input dataset, or a cloud URL (s3://, gs://, az://, ...)"),
    output_file: str = typer.Option(..., "-o", "--output", help="Path to output dataset, or a cloud URL"),
    strategy: str = typer.Option("deepen_reasoning", "--strategy", help="deepen_reasoning, add_constraints, concretize, in_breadth"),
    provider: str = typer.Option("gemini", "--provider", help="gemini, openai, anthropic, a private endpoint (azure_openai, bedrock_anthropic, openai_compatible), or a local LLM server (ollama, lmstudio, vllm, llamacpp)"),
    base_url: Optional[str] = typer.Option(None, "--base-url", help="Endpoint for openai_compatible/local-server providers, e.g. http://192.168.1.50:11434/v1 for another machine's Ollama -- overrides that provider's default local port"),
    network_policy: str = typer.Option("unrestricted", "--network-policy", help="unrestricted; local to allow only a provider whose endpoint you control on a loopback/private address; or strict to guarantee zero network calls at all"),
    actor: Optional[str] = typer.Option(None, "--actor", help="Check this actor's permissions against --policy before running (requires --policy too)"),
    policy_file: Optional[Path] = typer.Option(None, "--policy", help="Access policy YAML to check --actor against (requires --actor too)"),
    bearer_token: Optional[str] = typer.Option(None, "--bearer-token", help="OIDC bearer token proving identity, verified against --oidc-config before --policy is checked (requires --oidc-config too)"),
    oidc_config_file: Optional[Path] = typer.Option(None, "--oidc-config", help="OIDC issuer/audience/JWKS config YAML to verify --bearer-token against (requires --bearer-token too)"),
    model: Optional[str] = typer.Option(None, "--model", help="Provider model override"),
    concurrency: int = typer.Option(10, "--concurrency", help="Max concurrent requests"),
    max_rpm: int = typer.Option(60, "--rpm", help="Max requests per minute"),
    batch_size: int = typer.Option(20, "--batch-size", min=1, max=100, help="Records grouped into one structured request"),
    max_rows: Optional[int] = typer.Option(None, "--max-rows", help="Limit number of rows to process"),
):
    """Evolve prompts into higher complexity and reasoning depth (Evol-Instruct)."""
    console.print(Panel(f"[bold blue]BuffData Evolver[/bold blue]\nStrategy: {strategy}", border_style="blue"))

    items = read_dataset(input_file, max_rows=max_rows)
    _enforce_access_policy(
        actor, policy_file, network_policy, bearer_token=bearer_token, oidc_config_file=oidc_config_file
    )
    client = create_llm_client(provider, model, base_url=base_url, network_policy=network_policy)
    model = model or client.default_model
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

        evolved = asyncio.run(evolver.evolve_batch_async(items, strategies=[strategy], batch_size=batch_size, on_progress=on_step))

    write_dataset(evolved, output_file)
    console.print(f"[bold green]✓ Successfully evolved and saved {len(evolved)} items to {output_file}[/bold green]")


@app.command("dpo")
def dpo_cmd(
    input_file: str = typer.Argument(..., help="Path to input SFT dataset, or a cloud URL"),
    output_file: str = typer.Option(..., "-o", "--output", help="Path to output DPO dataset, or a cloud URL"),
    provider: str = typer.Option("gemini", "--provider", help="gemini, openai, anthropic, a private endpoint (azure_openai, bedrock_anthropic, openai_compatible), or a local LLM server (ollama, lmstudio, vllm, llamacpp)"),
    base_url: Optional[str] = typer.Option(None, "--base-url", help="Endpoint for openai_compatible/local-server providers, e.g. http://192.168.1.50:11434/v1 for another machine's Ollama -- overrides that provider's default local port"),
    network_policy: str = typer.Option("unrestricted", "--network-policy", help="unrestricted; local to allow only a provider whose endpoint you control on a loopback/private address; or strict to guarantee zero network calls at all"),
    actor: Optional[str] = typer.Option(None, "--actor", help="Check this actor's permissions against --policy before running (requires --policy too)"),
    policy_file: Optional[Path] = typer.Option(None, "--policy", help="Access policy YAML to check --actor against (requires --actor too)"),
    bearer_token: Optional[str] = typer.Option(None, "--bearer-token", help="OIDC bearer token proving identity, verified against --oidc-config before --policy is checked (requires --oidc-config too)"),
    oidc_config_file: Optional[Path] = typer.Option(None, "--oidc-config", help="OIDC issuer/audience/JWKS config YAML to verify --bearer-token against (requires --bearer-token too)"),
    model: Optional[str] = typer.Option(None, "--model", help="Provider model override"),
    concurrency: int = typer.Option(10, "--concurrency", help="Max concurrent requests"),
    max_rpm: int = typer.Option(60, "--rpm", help="Max requests per minute"),
    batch_size: int = typer.Option(20, "--batch-size", min=1, max=100, help="Records grouped into one structured request"),
    max_rows: Optional[int] = typer.Option(None, "--max-rows", help="Limit number of rows to process"),
):
    """Generate chosen and rejected preference pairs for DPO/RLHF alignment."""
    console.print(Panel(f"[bold yellow]BuffData DPO Preference Builder[/bold yellow]", border_style="yellow"))

    items = read_dataset(input_file, max_rows=max_rows)
    _enforce_access_policy(
        actor, policy_file, network_policy, bearer_token=bearer_token, oidc_config_file=oidc_config_file
    )
    client = create_llm_client(provider, model, base_url=base_url, network_policy=network_policy)
    model = model or client.default_model
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

        dpo_items = asyncio.run(dpo_builder.build_dpo_batch_async(items, batch_size=batch_size, on_progress=on_step))

    write_dataset(dpo_items, output_file)
    console.print(f"[bold green]✓ Successfully generated {len(dpo_items)} DPO pairs to {output_file}[/bold green]")


@app.command("dedup")
def dedup_cmd(
    input_file: str = typer.Argument(..., help="Path to input dataset, or a cloud URL (s3://, gs://, az://, ...)"),
    output_file: str = typer.Option(..., "-o", "--output", help="Path to output deduplicated dataset, or a cloud URL"),
    method: str = typer.Option("minhash", "--method", help="exact, minhash, semantic, semantic-local"),
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
    elif method == "semantic-local":
        kept, dropped = deduper.deduplicate_semantic_local(items, threshold=threshold)
    elif method == "semantic":
        kept, dropped = deduper.deduplicate_semantic(items, threshold=threshold)
    else:
        raise ValueError(f"Unknown dedup method: {method}")

    console.print(f"[green]Kept: {len(kept)} items[/green] | [red]Dropped Duplicates: {len(dropped)} items[/red]")
    write_dataset(kept, output_file)
    console.print(f"[bold green]✓ Wrote deduplicated dataset to {output_file}[/bold green]")


@app.command("stats")
def stats_cmd(
    input_file: str = typer.Argument(..., help="Path to dataset to inspect, or a cloud URL"),
):
    """Display statistics and quality metrics of a dataset."""
    items = read_dataset(input_file)
    summary_md = ReportGenerator.generate_markdown_summary(items)
    console.print(summary_md)


@app.command("report")
def report_cmd(
    input_file: str = typer.Argument(..., help="Path to dataset with quality scores, or a cloud URL"),
    output_html: Path = typer.Option(Path("report.html"), "-o", "--output", help="Output HTML file path"),
):
    """Generate interactive HTML audit report."""
    items = read_dataset(input_file)
    out = ReportGenerator.generate_html_report(items, output_html)
    console.print(f"[bold green]✓ Generated interactive HTML audit report at {out.resolve()}[/bold green]")


@app.command("pipeline")
def pipeline_cmd(
    config_file: Path = typer.Argument(..., help="Path to pipeline YAML configuration file"),
    input_file: str = typer.Option(..., "-i", "--input", help="Input dataset path"),
    output_file: str = typer.Option(..., "-o", "--output", help="Final output dataset path"),
):
    """Execute the adaptive pipeline from a YAML configuration."""
    if _is_cloud_url(input_file) or _is_cloud_url(output_file):
        raise typer.BadParameter(
            "buffdata pipeline doesn't support cloud storage URLs yet -- its checkpoint, "
            "fingerprint, and companion-artifact (.rejected.jsonl/.report.json) paths all "
            "assume a local filesystem. Download the input locally, run the pipeline, then "
            "upload the output -- or use buffdata score/refine/evolve/dpo/dedup/augment/"
            "scrub/classify/filter-ppl, which already accept s3://, gs://, and az:// directly."
        )
    console.print(Panel(f"[bold green]BuffData Multi-Stage Pipeline[/bold green]\nConfig: {config_file}", border_style="green"))

    with open(config_file, "r", encoding="utf-8") as f:
        cfg_dict = yaml.safe_load(f)
    config = PipelineConfig(**cfg_dict)

    from buffdata.engine.pipeline import OptimizationPipeline

    result = asyncio.run(OptimizationPipeline(config).run_file(Path(input_file), Path(output_file)))
    console.print(f"[bold green]Pipeline completed: {len(result.accepted)} accepted, {len(result.rejected)} quarantined.[/bold green]")
    console.print(f"[green]Output:[/green] {result.output_path}")
    console.print(f"[yellow]Rejected:[/yellow] {result.rejected_path}")
    console.print(f"[cyan]Report:[/cyan] {result.report_path}")


@app.command("optimize")
def optimize_cmd(
    input_file: str = typer.Argument(..., help="Input dataset"),
    output_file: str = typer.Option(..., "-o", "--output", help="Optimized dataset"),
    config_file: Optional[Path] = typer.Option(None, "--config", help="Optional YAML configuration"),
    provider: Optional[str] = typer.Option(None, "--provider", help="gemini, openai, anthropic, a private endpoint (azure_openai, bedrock_anthropic, openai_compatible), or a local LLM server (ollama, lmstudio, vllm, llamacpp)"),
    base_url: Optional[str] = typer.Option(None, "--base-url", help="Endpoint for openai_compatible/local-server providers, e.g. http://192.168.1.50:11434/v1 for another machine's Ollama -- overrides that provider's default local port"),
    model: Optional[str] = typer.Option(None, "--model", help="Provider model override"),
    classification: Optional[str] = typer.Option(None, "--classification", help="auto, off, binary, multi-class, or multi-label"),
    classes: Optional[str] = typer.Option(None, "--classes", help="Comma-separated class names"),
    min_score: Optional[float] = typer.Option(None, "--min-score", help="Final quality threshold"),
    quality_mode: Optional[str] = typer.Option(None, "--quality-mode", help="llm, sampled, or off"),
    quality_sample_size: Optional[int] = typer.Option(None, "--quality-sample-size", help="Rows audited in sampled mode"),
    classification_pii_mode: Optional[str] = typer.Option(
        None,
        "--classification-pii-mode",
        help="For labeled classification: identifiers, all, or off",
    ),
    accuracy_contract: Optional[str] = typer.Option(
        None,
        "--accuracy-contract",
        help="balanced or strict; strict forbids labeled text/label mutation",
    ),
    network_policy: Optional[str] = typer.Option(
        None,
        "--network-policy",
        help="unrestricted; local to allow only a provider whose endpoint you control on a loopback/private address; or strict to guarantee zero network calls (requires quality-mode off and dedup-method != semantic)",
    ),
    actor: Optional[str] = typer.Option(None, "--actor", help="Check this actor's permissions against --policy before running (requires --policy too)"),
    policy_file: Optional[Path] = typer.Option(None, "--policy", help="Access policy YAML to check --actor against (requires --actor too)"),
    bearer_token: Optional[str] = typer.Option(None, "--bearer-token", help="OIDC bearer token proving identity, verified against --oidc-config before --policy is checked (requires --oidc-config too)"),
    oidc_config_file: Optional[Path] = typer.Option(None, "--oidc-config", help="OIDC issuer/audience/JWKS config YAML to verify --bearer-token against (requires --bearer-token too)"),
    validation_file: Optional[str] = typer.Option(
        None,
        "--validation-file",
        help="Labeled held-out dataset used by the positive accuracy gate",
    ),
    require_positive_gain: bool = typer.Option(
        False,
        "--require-positive-gain",
        help="Publish output only when candidate accuracy improves on every seed",
    ),
    accuracy_min_gain: float = typer.Option(
        0.0,
        "--accuracy-min-gain",
        min=0.0,
        help="Required fractional accuracy gain on the mean and every seed",
    ),
    accuracy_seeds: str = typer.Option("17,29,43", "--accuracy-seeds"),
    accuracy_epochs: int = typer.Option(6, "--accuracy-epochs", min=1),
    accuracy_max_train_rows: int = typer.Option(
        100_000,
        "--accuracy-max-train-rows",
        min=100,
        help="Maximum rows from each condition used by the gate",
    ),
):
    """Run the dataset-aware quality, privacy, and classification pipeline."""
    if _is_cloud_url(input_file) or _is_cloud_url(output_file) or (validation_file and _is_cloud_url(validation_file)):
        raise typer.BadParameter(
            "buffdata optimize doesn't support cloud storage URLs yet -- its checkpoint, "
            "fingerprint, and companion-artifact (.rejected.jsonl/.report.json) paths all "
            "assume a local filesystem. Download the input locally, run optimize, then "
            "upload the output -- or use buffdata score/refine/evolve/dpo/dedup/augment/"
            "scrub/classify/filter-ppl, which already accept s3://, gs://, and az:// directly."
        )
    input_file = Path(input_file)
    output_file = Path(output_file)
    validation_file = Path(validation_file) if validation_file else None
    config_data = {}
    if config_file:
        with open(config_file, "r", encoding="utf-8") as handle:
            config_data = yaml.safe_load(handle) or {}
    if provider is not None:
        config_data["provider"] = provider
    if base_url is not None:
        config_data["base_url"] = base_url
    if model is not None:
        config_data["model"] = model
    if classification is not None:
        config_data["classification"] = classification
    if classes is not None:
        config_data["classes"] = [value.strip() for value in classes.split(",") if value.strip()]
    if min_score is not None:
        config_data["filter_min_score"] = min_score
    if quality_mode is not None:
        config_data["quality_mode"] = quality_mode
    if quality_sample_size is not None:
        config_data["quality_sample_size"] = quality_sample_size
    if classification_pii_mode is not None:
        config_data["classification_pii_mode"] = classification_pii_mode
    if accuracy_contract is not None:
        config_data["accuracy_contract"] = accuracy_contract
    if network_policy is not None:
        config_data["network_policy"] = network_policy
    config = PipelineConfig(**config_data)
    _enforce_access_policy(
        actor, policy_file, config.network_policy, bearer_token=bearer_token, oidc_config_file=oidc_config_file
    )

    from buffdata.engine.pipeline import OptimizationPipeline

    console.print(Panel(
        f"[bold green]BuffData Adaptive Optimizer[/bold green]\nProvider: {config.provider}\nClassification: {config.classification.value}",
        border_style="green",
    ))
    pipeline = OptimizationPipeline(config)
    if require_positive_gain:
        if validation_file is None:
            raise typer.BadParameter("--validation-file is required with --require-positive-gain")
        if output_file.exists():
            raise typer.BadParameter(
                "Positive-gain output must not already exist; choose a new path so a failed gate cannot leave stale data"
            )
        from buffdata.evaluation.accuracy_gate import evaluate_accuracy_gain

        original_items = read_dataset(input_file)
        validation_items = read_dataset(validation_file)
        result = asyncio.run(pipeline.run([item.model_copy(deep=True) for item in original_items]))
        seeds = [int(value.strip()) for value in accuracy_seeds.split(",") if value.strip()]
        gate = evaluate_accuracy_gain(
            original_items,
            result.accepted,
            validation_items,
            seeds=seeds,
            epochs=accuracy_epochs,
            minimum_gain=accuracy_min_gain,
            max_train_rows=accuracy_max_train_rows,
        )
        table = Table(title="Original vs candidate accuracy gate")
        table.add_column("Seed")
        table.add_column("Original", justify="right")
        table.add_column("Candidate", justify="right")
        table.add_column("Gain", justify="right")
        for index, seed in enumerate(gate["seeds"]):
            original_accuracy = gate["original_runs"][index]["accuracy"]
            candidate_accuracy = gate["candidate_runs"][index]["accuracy"]
            table.add_row(
                str(seed),
                f"{original_accuracy:.4f}",
                f"{candidate_accuracy:.4f}",
                f"{candidate_accuracy - original_accuracy:+.4f}",
            )
        table.add_row(
            "mean",
            f"{gate['original_accuracy_mean']:.4f}",
            f"{gate['candidate_accuracy_mean']:.4f}",
            f"{gate['accuracy_gain']:+.4f}",
        )
        console.print(table)
        if not gate["accepted"]:
            console.print(
                "[bold red]Accuracy gate failed. No optimized output was published.[/bold red]"
            )
            raise typer.Exit(code=2)
        result.metrics["accuracy_gate"] = gate
        rejected_path, report_path, _ = pipeline.artifact_paths(output_file)
        write_dataset_atomic(result.accepted, output_file)
        write_dataset_atomic(result.rejected, rejected_path)
        pipeline._atomic_json(
            report_path,
            {
                "profile": result.profile.model_dump(mode="json"),
                "metrics": result.metrics,
                "rejection_reasons": pipeline._rejection_counts(result.rejected),
            },
        )
        result.output_path = str(output_file)
        result.rejected_path = str(rejected_path)
        result.report_path = str(report_path)
    else:
        result = asyncio.run(pipeline.run_file(input_file, output_file))
    console.print(f"[bold green]Accepted {len(result.accepted)} records.[/bold green]")
    console.print(f"[yellow]Quarantined {len(result.rejected)} records to {result.rejected_path}.[/yellow]")
    console.print(f"[cyan]Audit report: {result.report_path}[/cyan]")


@app.command()
def synthesize(
    source: str = typer.Argument(..., help="Path to PDF, TXT, or a URL"),
    output: str = typer.Option("synthetic.jsonl", "-o", "--output", help="Output JSONL path"),
    num_pairs: int = typer.Option(10, "-n", help="Number of pairs to generate per batch"),
    provider: str = typer.Option("gemini", "--provider", help="gemini, openai, anthropic, a private endpoint (azure_openai, bedrock_anthropic, openai_compatible), or a local LLM server (ollama, lmstudio, vllm, llamacpp)"),
    base_url: Optional[str] = typer.Option(None, "--base-url", help="Endpoint for openai_compatible/local-server providers, e.g. http://192.168.1.50:11434/v1 for another machine's Ollama -- overrides that provider's default local port"),
    network_policy: str = typer.Option("unrestricted", "--network-policy", help="unrestricted; local to allow only a provider whose endpoint you control on a loopback/private address; or strict to guarantee zero network calls at all"),
    actor: Optional[str] = typer.Option(None, "--actor", help="Check this actor's permissions against --policy before running (requires --policy too)"),
    policy_file: Optional[Path] = typer.Option(None, "--policy", help="Access policy YAML to check --actor against (requires --actor too)"),
    bearer_token: Optional[str] = typer.Option(None, "--bearer-token", help="OIDC bearer token proving identity, verified against --oidc-config before --policy is checked (requires --oidc-config too)"),
    oidc_config_file: Optional[Path] = typer.Option(None, "--oidc-config", help="OIDC issuer/audience/JWKS config YAML to verify --bearer-token against (requires --bearer-token too)"),
    model: Optional[str] = typer.Option(None, "--model", help="Provider model override"),
):
    """Generate synthetic instruction datasets from raw documents."""
    import asyncio
    import json
    from buffdata.synthesizer.generator import DataSynthesizer, extract_text_from_pdf, extract_text_from_url

    _enforce_access_policy(
        actor, policy_file, network_policy, bearer_token=bearer_token, oidc_config_file=oidc_config_file
    )
    client = create_llm_client(provider, model, base_url=base_url, network_policy=network_policy)
    synthesizer = DataSynthesizer(client)

    async def run_synth():
        if source.startswith("http"):
            text = await extract_text_from_url(source)
        elif source.endswith(".pdf"):
            text = await extract_text_from_pdf(source)
        else:
            with open(source, "r", encoding="utf-8") as f:
                text = f.read()

        console.print(f"[green]Extracted {len(text)} characters from {source}. Generating {num_pairs} pairs...[/green]")
        items = await synthesizer.synthesize_from_text(text, max_pairs=num_pairs)

        with open(output, "w", encoding="utf-8") as f:
            for item in items:
                f.write(json.dumps(item) + "\n")
        console.print(f"[bold green]Synthetic dataset generated at {output}![/bold green]")

    asyncio.run(run_synth())

@app.command()
def push(
    dataset: str = typer.Argument(..., help="Path to optimized JSONL dataset"),
    repo_id: str = typer.Argument(..., help="HuggingFace repo ID (e.g. user/my-dataset)"),
    token: str = typer.Option(None, "--token", envvar="HF_TOKEN", help="HuggingFace Write Token")
):
    """Push an optimized dataset directly to the HuggingFace Hub."""
    from buffdata.integrations.huggingface import push_to_hub
    console.print(f"Preparing to push {dataset} to {repo_id}...")
    push_to_hub(dataset, repo_id, token)

@app.command()
def export(
    dataset: str = typer.Argument(..., help="Path to JSONL dataset"),
    output: str = typer.Argument(..., help="Output JSON path"),
    format: str = typer.Option("unsloth", "--format", help="Target training format (e.g., 'unsloth' ShareGPT)")
):
    """Export dataset into specialized formats for Axolotl or Unsloth training."""
    if format.lower() == "unsloth":
        from buffdata.integrations.train_export import export_unsloth_sharegpt
        export_unsloth_sharegpt(dataset, output)
    else:
        console.print(f"[red]Unsupported export format: {format}[/red]")


@app.command()
def scrub(
    input_file: str = typer.Argument(..., help="Path to input dataset, or a cloud URL (s3://, gs://, az://, ...)"),
    output_file: str = typer.Option(..., "-o", "--output", help="Path to output scrubbed dataset, or a cloud URL"),
):
    """Detect and redact PII (Personally Identifiable Information) from a dataset."""
    from buffdata.optimizers.scrubber import PIIScrubber
    console.print(Panel("[bold yellow]BuffData PII Scrubber (Presidio)[/bold yellow]", border_style="yellow"))

    items = read_dataset(input_file)
    scrubber = PIIScrubber()

    with Progress(
        SpinnerColumn(), TextColumn("[progress.description]{task.description}"), BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%")
    ) as progress:
        task = progress.add_task("[cyan]Scrubbing PII...", total=len(items))
        scrubbed_items = []
        for item in items:
            scrubbed_items.append(scrubber.scrub_item(item))
            progress.advance(task, 1)

    write_dataset(scrubbed_items, output_file)
    console.print(f"[bold green]Successfully scrubbed {len(items)} items. Saved to {output_file}[/bold green]")

@app.command()
def validate(
    input_file: str = typer.Argument(..., help="Path to input dataset, or a cloud URL (s3://, gs://, az://, ...)"),
):
    """Validate dataset schema and structural integrity."""
    from buffdata.engine.validator import DatasetValidator
    console.print(Panel(f"[bold blue]BuffData Validator[/bold blue]\nFile: {input_file}", border_style="blue"))

    items = read_dataset(input_file)
    is_valid, errors = DatasetValidator.validate(items)

    if is_valid:
        console.print(f"[bold green]Dataset is perfectly valid! ({len(items)} rows)[/bold green]")
    else:
        console.print(f"[bold red]Validation failed with {len(errors)} errors:[/bold red]")
        for err in errors[:10]: # Print top 10
            console.print(f"  - [red]{err}[/red]")
        if len(errors) > 10:
            console.print(f"  - ... and {len(errors) - 10} more.")


@app.command("filter-ppl")
def filter_ppl_cmd(
    input_file: str = typer.Argument(..., help="Path to input dataset, or a cloud URL (s3://, gs://, az://, ...)"),
    output_file: str = typer.Option(..., "-o", "--output", help="Path to output filtered dataset, or a cloud URL"),
    max_ppl: float = typer.Option(100.0, "--max-ppl", help="Maximum perplexity threshold. Higher = more lenient."),
):
    """Filter out low-quality/garbled text using PyTorch Perplexity evaluation."""
    from buffdata.optimizers.perplexity import PerplexityFilter
    console.print(Panel(f"[bold magenta]BuffData PyTorch Perplexity Filter[/bold magenta]\nMax PPL: {max_ppl}", border_style="magenta"))

    items = read_dataset(input_file)

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}")) as progress:
        progress.add_task("[magenta]Loading PyTorch Model (GPT-2)...", total=None)
        ppl_filter = PerplexityFilter()

    console.print("[cyan]Evaluating perplexity across dataset...[/cyan]")
    kept, dropped = ppl_filter.filter_batch(items, max_ppl)

    console.print(f"[green]Kept: {len(kept)} items[/green] | [red]Dropped (Garbage/High PPL): {len(dropped)} items[/red]")
    write_dataset(kept, output_file)


@app.command()
def classify(
    input_file: str = typer.Argument(..., help="Path to input dataset, or a cloud URL (s3://, gs://, az://, ...)"),
    output_file: str = typer.Option(..., "-o", "--output", help="Path to output classified dataset, or a cloud URL"),
    type: str = typer.Option("auto", "--type", "--classification", help="auto, off, binary, multi-class, or multi-label"),
    classes: Optional[str] = typer.Option(None, "--classes", help="Comma-separated class names"),
    provider: str = typer.Option("gemini", "--provider", help="gemini, openai, anthropic, a private endpoint (azure_openai, bedrock_anthropic, openai_compatible), or a local LLM server (ollama, lmstudio, vllm, llamacpp)"),
    base_url: Optional[str] = typer.Option(None, "--base-url", help="Endpoint for openai_compatible/local-server providers, e.g. http://192.168.1.50:11434/v1 for another machine's Ollama -- overrides that provider's default local port"),
    network_policy: str = typer.Option("unrestricted", "--network-policy", help="unrestricted; local to allow only a provider whose endpoint you control on a loopback/private address; or strict to guarantee zero network calls at all"),
    actor: Optional[str] = typer.Option(None, "--actor", help="Check this actor's permissions against --policy before running (requires --policy too)"),
    policy_file: Optional[Path] = typer.Option(None, "--policy", help="Access policy YAML to check --actor against (requires --actor too)"),
    bearer_token: Optional[str] = typer.Option(None, "--bearer-token", help="OIDC bearer token proving identity, verified against --oidc-config before --policy is checked (requires --oidc-config too)"),
    oidc_config_file: Optional[Path] = typer.Option(None, "--oidc-config", help="OIDC issuer/audience/JWKS config YAML to verify --bearer-token against (requires --bearer-token too)"),
    model: Optional[str] = typer.Option(None, "--model", help="Provider model override"),
    confidence: float = typer.Option(0.75, "--confidence", help="Minimum auto-detection confidence"),
    sample_size: int = typer.Option(100, "--sample-size", help="Representative records to inspect"),
):
    """Automatically classify datasets (Binary, Multi-class, Multi-label)."""
    import asyncio
    from buffdata.optimizers.classifier import DatasetClassifier
    from buffdata.engine.limiter import AsyncRateLimiter

    console.print(Panel(f"[bold cyan]BuffData Classifier[/bold cyan]\nMode: {type}", border_style="cyan"))
    items = read_dataset(input_file)
    if not items:
        console.print("[red]Dataset is empty![/red]")
        return

    _enforce_access_policy(
        actor, policy_file, network_policy, bearer_token=bearer_token, oidc_config_file=oidc_config_file
    )
    client = create_llm_client(provider, model, base_url=base_url, network_policy=network_policy)
    limiter = AsyncRateLimiter(max_rpm=60, concurrency=5)
    classifier = DatasetClassifier(client, limiter, model)

    async def run_classification():
        requested_classes = [c.strip() for c in classes.split(",") if c.strip()] if classes else []
        schema = await classifier.resolve_schema(
            items,
            mode=ClassificationMode(type),
            classes=requested_classes,
            sample_size=sample_size,
        )
        console.print(f"[bold green]Applicable:[/bold green] {schema.applicable}")
        console.print(f"[bold green]Detected Task:[/bold green] {schema.task_type}")
        console.print(f"[bold green]Detected Classes:[/bold green] {schema.classes}")
        console.print(f"[green]Confidence:[/green] {schema.confidence:.2f}")
        console.print(f"[green]Reasoning:[/green] {schema.reasoning}")
        if not schema.applicable or schema.confidence < confidence:
            write_dataset(items, output_file)
            console.print(f"[yellow]Classification skipped; unchanged records saved to {output_file}.[/yellow]")
            return

        console.print(f"[cyan]Classifying {len(items)} items...[/cyan]")
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}")) as progress:
            progress.add_task(f"[cyan]Calling {provider} for classification...", total=None)
            classified_items = await classifier.classify_batch(items, schema)

        write_dataset(classified_items, output_file)
        console.print(f"[bold green]Successfully classified and saved to {output_file}[/bold green]")

    asyncio.run(run_classification())

@app.command("augment")
def augment_cmd(
    input_file: str = typer.Argument(..., help="Input dataset, or a cloud URL"),
    output_file: str = typer.Option(..., "-o", "--output", help="Augmented output dataset, or a cloud URL"),
    multiplier: int = typer.Option(1, "--multiplier", "-m", min=1, max=3, help="Accepted variations per record"),
    chunk_size: int = typer.Option(20, "--chunk-size", min=1, max=100, help="Source records per provider request"),
    min_label_confidence: float = typer.Option(0.55, "--min-label-confidence", min=0.0, max=1.0),
    concurrency: int = typer.Option(10, "--concurrency", help="Max concurrent requests"),
    max_rpm: int = typer.Option(60, "--rpm", help="Max requests per minute"),
    provider: Optional[str] = typer.Option(None, "--provider", help="gemini, openai, anthropic, a private endpoint (azure_openai, bedrock_anthropic, openai_compatible), or a local LLM server (ollama, lmstudio, vllm, llamacpp)"),
    base_url: Optional[str] = typer.Option(None, "--base-url", help="Endpoint for openai_compatible/local-server providers, e.g. http://192.168.1.50:11434/v1 for another machine's Ollama -- overrides that provider's default local port"),
    network_policy: str = typer.Option("unrestricted", "--network-policy", help="unrestricted; local to allow only a provider whose endpoint you control on a loopback/private address; or strict to guarantee zero network calls at all"),
    actor: Optional[str] = typer.Option(None, "--actor", help="Check this actor's permissions against --policy before running (requires --policy too)"),
    policy_file: Optional[Path] = typer.Option(None, "--policy", help="Access policy YAML to check --actor against (requires --actor too)"),
    bearer_token: Optional[str] = typer.Option(None, "--bearer-token", help="OIDC bearer token proving identity, verified against --oidc-config before --policy is checked (requires --oidc-config too)"),
    oidc_config_file: Optional[Path] = typer.Option(None, "--oidc-config", help="OIDC issuer/audience/JWKS config YAML to verify --bearer-token against (requires --bearer-token too)"),
    model: Optional[str] = typer.Option(None, "--model", help="Provider model override"),
):
    """Generate label-guarded synthetic variations for a labeled classification dataset."""
    import asyncio
    from buffdata.engine.client import create_llm_client
    from buffdata.engine.limiter import AsyncRateLimiter
    from buffdata.models.formats import read_dataset, write_dataset
    from buffdata.optimizers.augmenter import DataAugmenter
    from rich.console import Console

    console = Console()
    console.print(f"[bold cyan]BuffData Synthetic Augmenter (x{multiplier})[/bold cyan]")

    selected_provider = provider or "gemini"
    _enforce_access_policy(
        actor, policy_file, network_policy, bearer_token=bearer_token, oidc_config_file=oidc_config_file
    )
    client = create_llm_client(selected_provider, model=model, allow_mock=False, base_url=base_url, network_policy=network_policy)
    limiter = AsyncRateLimiter(max_rpm=max_rpm, concurrency=concurrency)

    items = read_dataset(input_file)
    console.print(f"Loaded {len(items)} original records.")

    async def run():
        augmenter = DataAugmenter(client=client, limiter=limiter, model=model)
        return await augmenter.augment_batch_async(
            items,
            multiplier=multiplier,
            chunk_size=chunk_size,
            min_label_confidence=min_label_confidence,
        )

    augmented_items = asyncio.run(run())
    write_dataset(augmented_items, output_file)
    generated = sum(bool(item.metadata.get("is_synthetic")) for item in augmented_items)
    console.print(
        f"[bold green]Augmentation complete.[/bold green] "
        f"Accepted {generated} guarded synthetic records; output has {len(augmented_items)} total records."
    )


@app.command("generate")
def generate_cmd(
    input_file: str = typer.Argument(..., help="Labeled classification dataset to generate more data from"),
    output_file: str = typer.Option(..., "-o", "--output", help="Accuracy-gated generated dataset"),
    validation_file: str = typer.Option(
        ...,
        "--validation-file",
        help="Held-out labeled dataset used to measure the accuracy gain",
    ),
    min_relative_gain: float = typer.Option(
        0.10,
        "--min-relative-gain",
        min=0.0,
        help="Required relative accuracy gain over the original dataset (0.10 = 10%)",
    ),
    max_iterations: int = typer.Option(5, "--max-iterations", min=1, max=20),
    multiplier: int = typer.Option(1, "--multiplier", "-m", min=1, max=3, help="Variations per record per round"),
    chunk_size: int = typer.Option(20, "--chunk-size", min=1, max=100, help="Source records per provider request"),
    min_label_confidence: float = typer.Option(0.55, "--min-label-confidence", min=0.0, max=1.0),
    accuracy_seeds: str = typer.Option("17,29,43", "--accuracy-seeds"),
    accuracy_epochs: int = typer.Option(6, "--accuracy-epochs", min=1),
    weak_class_count: int = typer.Option(2, "--weak-class-count", min=1, help="Classes targeted per retry round"),
    concurrency: int = typer.Option(10, "--concurrency", help="Max concurrent requests"),
    max_rpm: int = typer.Option(60, "--rpm", help="Max requests per minute"),
    provider: Optional[str] = typer.Option(None, "--provider", help="gemini, openai, anthropic, a private endpoint (azure_openai, bedrock_anthropic, openai_compatible), or a local LLM server (ollama, lmstudio, vllm, llamacpp)"),
    base_url: Optional[str] = typer.Option(None, "--base-url", help="Endpoint for openai_compatible/local-server providers, e.g. http://192.168.1.50:11434/v1 for another machine's Ollama -- overrides that provider's default local port"),
    network_policy: str = typer.Option("unrestricted", "--network-policy", help="unrestricted; local to allow only a provider whose endpoint you control on a loopback/private address; or strict to guarantee zero network calls at all"),
    actor: Optional[str] = typer.Option(None, "--actor", help="Check this actor's permissions against --policy before running (requires --policy too)"),
    policy_file: Optional[Path] = typer.Option(None, "--policy", help="Access policy YAML to check --actor against (requires --actor too)"),
    bearer_token: Optional[str] = typer.Option(None, "--bearer-token", help="OIDC bearer token proving identity, verified against --oidc-config before --policy is checked (requires --oidc-config too)"),
    oidc_config_file: Optional[Path] = typer.Option(None, "--oidc-config", help="OIDC issuer/audience/JWKS config YAML to verify --bearer-token against (requires --bearer-token too)"),
    model: Optional[str] = typer.Option(None, "--model", help="Provider model override"),
):
    """Iteratively augment a labeled dataset until it clears a minimum relative accuracy
    gain over the original data, or report the shortfall honestly after the iteration budget.
    """
    from buffdata.optimizers.gated_generator import AccuracyGatedGenerator

    if _is_cloud_url(input_file) or _is_cloud_url(output_file) or _is_cloud_url(validation_file):
        raise typer.BadParameter(
            "buffdata generate doesn't support cloud storage URLs yet -- its companion "
            ".accuracy_gate.json report path assumes a local filesystem. Download the input "
            "and validation files locally, run generate, then upload the output -- or use "
            "buffdata augment directly (no accuracy gate, but accepts s3://, gs://, az:// now)."
        )
    output_file = Path(output_file)
    input_file = Path(input_file)
    validation_file = Path(validation_file)

    if output_file.exists():
        raise typer.BadParameter(
            "Generated output must not already exist; choose a new path so a stale run cannot be mistaken for a fresh one"
        )

    selected_provider = provider or "gemini"
    _enforce_access_policy(
        actor, policy_file, network_policy, bearer_token=bearer_token, oidc_config_file=oidc_config_file
    )
    client = create_llm_client(selected_provider, model=model, allow_mock=False, base_url=base_url, network_policy=network_policy)
    limiter = AsyncRateLimiter(max_rpm=max_rpm, concurrency=concurrency)
    items = read_dataset(input_file)
    validation_items = read_dataset(validation_file)
    seeds = [int(value.strip()) for value in accuracy_seeds.split(",") if value.strip()]

    console.print(Panel(
        f"[bold cyan]BuffData Accuracy-Gated Generator[/bold cyan]\n"
        f"Input: {input_file} ({len(items)} rows)\n"
        f"Target: >= {min_relative_gain:.0%} relative accuracy gain, up to {max_iterations} round(s)",
        border_style="cyan",
    ))

    generator = AccuracyGatedGenerator(client=client, limiter=limiter, model=model)

    async def run():
        return await generator.generate(
            items,
            validation_items,
            min_relative_gain=min_relative_gain,
            max_iterations=max_iterations,
            multiplier=multiplier,
            chunk_size=chunk_size,
            min_label_confidence=min_label_confidence,
            accuracy_seeds=seeds,
            accuracy_epochs=accuracy_epochs,
            weak_class_count=weak_class_count,
        )

    candidate, report = asyncio.run(run())

    table = Table(title="Accuracy-gated generation")
    table.add_column("Round")
    table.add_column("Rows", justify="right")
    table.add_column("Accuracy", justify="right")
    table.add_column("Relative gain", justify="right")
    table.add_column("Weak classes")
    table.add_column("Gate")
    for metric in report.per_iteration:
        table.add_row(
            str(metric.iteration),
            str(metric.pool_size),
            f"{metric.accuracy:.4f}",
            f"{metric.relative_gain:+.1%}",
            ", ".join(metric.weak_labels) or "-",
            "[green]PASS[/green]" if metric.accepted else "[red]FAIL[/red]",
        )
    console.print(table)

    write_dataset(candidate, output_file)
    gate_report_path = output_file.with_name(f"{output_file.stem}.accuracy_gate.json")
    gate_report_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")

    if report.passed:
        console.print(
            f"[bold green]Accuracy gate PASSED[/bold green]: {report.baseline_accuracy:.4f} -> "
            f"{report.final_accuracy:.4f} ({report.relative_gain:+.1%}, target {report.target_relative_gain:.0%}) "
            f"in {report.iterations_used} round(s). Saved {len(candidate)} rows to {output_file}."
        )
    else:
        console.print(
            f"[bold yellow]Accuracy gate did not reach the target after {report.iterations_used} round(s).[/bold yellow] "
            f"Best result: {report.baseline_accuracy:.4f} -> {report.final_accuracy:.4f} "
            f"({report.relative_gain:+.1%} vs target {report.target_relative_gain:.0%}). "
            f"Saved the best candidate found ({len(candidate)} rows) to {output_file} -- see {gate_report_path} for details."
        )


@app.command("shard")
def shard_cmd(
    input_file: str = typer.Argument(..., help="Dataset to split into partitions"),
    output_dir: str = typer.Option(..., "-o", "--output-dir", help="Directory to write partition-NNNN.jsonl files into"),
    num_partitions: int = typer.Option(..., "-n", "--num-partitions", min=1, help="How many partition files to create"),
    no_stratify: bool = typer.Option(False, "--no-stratify", help="Disable label-proportional splitting; use plain round-robin"),
):
    """Split a dataset into N files for orchestrator-level parallelism (Airflow/Dagster/k8s
    Job array/xargs -P): run this once, then N independent buffdata invocations in parallel
    (one per partition), then `buffdata merge` the outputs back into one dataset.
    """
    from buffdata.integrations.sharding import partition_dataset

    console.print(Panel(
        f"[bold cyan]BuffData Sharder[/bold cyan]\nInput: {input_file}\nPartitions: {num_partitions}",
        border_style="cyan",
    ))
    paths = partition_dataset(
        input_file, output_dir, num_partitions, stratify_by_label=not no_stratify,
    )
    console.print(f"[bold green]Wrote {len(paths)} partition(s) to {output_dir}:[/bold green]")
    for path in paths:
        console.print(f"  {path}")


@app.command("merge")
def merge_cmd(
    partition_outputs: List[str] = typer.Argument(..., help="Partition output files to combine, in any order"),
    output_file: str = typer.Option(..., "-o", "--output", help="Final combined dataset path"),
):
    """Recombine N independently-processed partition outputs (from `buffdata shard` +
    N parallel buffdata runs) into one final dataset, merging each partition's
    .rejected.jsonl / .report.json artifacts along the way.
    """
    from buffdata.integrations.sharding import merge_partition_results

    console.print(Panel(
        f"[bold cyan]BuffData Merger[/bold cyan]\nCombining {len(partition_outputs)} partition(s)",
        border_style="cyan",
    ))
    summary = merge_partition_results(partition_outputs, output_file)
    console.print(
        f"[bold green]Merged {summary['partitions_merged']} partition(s):[/bold green] "
        f"{summary['accepted_records']} accepted, {summary['rejected_records']} rejected. "
        f"Saved to {output_file}."
    )


@app.command("run-ray")
def run_ray_cmd(
    config_file: Path = typer.Argument(..., help="Pipeline YAML configuration file (same format as `buffdata pipeline`)"),
    partitions: List[str] = typer.Argument(..., help="Partition files to process, e.g. from `buffdata shard`"),
    output_dir: str = typer.Option(..., "-o", "--output-dir", help="Directory to write each partition's processed output into"),
    ray_address: Optional[str] = typer.Option(None, "--ray-address", help="Ray cluster address (ray://host:10001); omit to use/start a local Ray runtime"),
    num_cpus: Optional[int] = typer.Option(None, "--num-cpus", help="CPUs for a local Ray runtime (ignored when --ray-address is given)"),
):
    """Process each partition (from `buffdata shard`) as an independent Ray task, distributed
    across whatever Ray cluster is connected -- local by default, or a real remote cluster via
    --ray-address. An in-process alternative to the shard -> external-orchestrator -> merge
    pattern for teams that already run a Ray cluster: run this in place of the external
    orchestrator step, then `buffdata merge` the outputs same as always.
    """
    from buffdata.integrations.ray_executor import run_partitions_with_ray

    with open(config_file, "r", encoding="utf-8") as handle:
        config = PipelineConfig(**(yaml.safe_load(handle) or {}))

    console.print(Panel(
        f"[bold cyan]BuffData Ray Executor[/bold cyan]\nPartitions: {len(partitions)}\n"
        f"Ray: {ray_address or 'local runtime'}",
        border_style="cyan",
    ))
    outputs = run_partitions_with_ray(
        partitions, output_dir, config, ray_address=ray_address, num_cpus=num_cpus,
    )
    console.print(f"[bold green]Processed {len(outputs)} partition(s) into {output_dir}:[/bold green]")
    for path in outputs:
        console.print(f"  {path}")


contract_app = typer.Typer(help="Check a completed run's artifacts against a versioned Data Contract.")
app.add_typer(contract_app, name="contract")


@contract_app.command("check")
def contract_check_cmd(
    contract_file: Path = typer.Argument(..., help="Path to the Data Contract YAML file"),
    report: Path = typer.Option(..., "--report", help="Path to the run's *.report.json"),
    accuracy_gate: Optional[Path] = typer.Option(
        None,
        "--accuracy-gate",
        help="Path to a *.accuracy_gate.json (from buffdata generate) when the report itself doesn't carry one",
    ),
):
    """Check an already-completed buffdata run against a Data Contract -- fast, deterministic,
    no provider credentials needed, suitable for a CI gate. Exits 0 when every requirement is
    met, 1 otherwise (with each violation printed), so `buffdata contract check ... || exit 1`
    composes directly into a build pipeline.
    """
    from buffdata.governance import DataContract, check_contract

    contract = DataContract.from_yaml(contract_file)
    result = check_contract(contract, report_path=report, accuracy_gate_path=accuracy_gate)

    console.print(Panel(
        f"[bold cyan]Data Contract:[/bold cyan] {result.contract_name} (v{result.contract_version})",
        border_style="cyan",
    ))
    if result.passed:
        console.print("[bold green]PASSED[/bold green] -- every requirement met.")
        return

    table = Table(title="Contract violations")
    table.add_column("Requirement")
    table.add_column("Expected")
    table.add_column("Actual")
    for violation in result.violations:
        table.add_row(violation.requirement, violation.expected, violation.actual)
    console.print(table)
    console.print(f"[bold red]FAILED[/bold red] -- {len(result.violations)} requirement(s) not met.")
    raise typer.Exit(code=1)


audit_app = typer.Typer(help="A durable, queryable log of buffdata runs (SQLite-backed by default).")
app.add_typer(audit_app, name="audit")

DEFAULT_AUDIT_DB = "buffdata_audit.db"


def _audit_db_path(db: Optional[str]) -> str:
    import os

    return db or os.getenv("BUFFDATA_AUDIT_DB", DEFAULT_AUDIT_DB)


@audit_app.command("record")
def audit_record_cmd(
    report: Path = typer.Argument(..., help="Path to the run's *.report.json"),
    command: str = typer.Option(..., "--command", help="Which buffdata command produced this report (optimize, generate, score, ...)"),
    db: Optional[str] = typer.Option(None, "--db", help=f"Audit database path (default: {DEFAULT_AUDIT_DB}, or $BUFFDATA_AUDIT_DB)"),
    contract_file: Optional[Path] = typer.Option(None, "--contract", help="Also check against a Data Contract and record pass/fail"),
    accuracy_gate: Optional[Path] = typer.Option(None, "--accuracy-gate", help="*.accuracy_gate.json, if the contract needs it and the report doesn't carry one"),
):
    """Ingest a completed run's report.json into the durable audit log."""
    from buffdata.governance import DataContract, check_contract, record_from_report, SQLiteAuditStore

    contract_name = None
    contract_passed = None
    if contract_file is not None:
        contract = DataContract.from_yaml(contract_file)
        result = check_contract(contract, report_path=report, accuracy_gate_path=accuracy_gate)
        contract_name, contract_passed = result.contract_name, result.passed

    store = SQLiteAuditStore(_audit_db_path(db))
    entry = record_from_report(store, report, command=command, contract_name=contract_name, contract_passed=contract_passed)
    console.print(f"[bold green]Recorded run {entry.run_id}[/bold green] ({command}) to {_audit_db_path(db)}")


@audit_app.command("query")
def audit_query_cmd(
    db: Optional[str] = typer.Option(None, "--db", help=f"Audit database path (default: {DEFAULT_AUDIT_DB}, or $BUFFDATA_AUDIT_DB)"),
    command: Optional[str] = typer.Option(None, "--command", help="Filter to one buffdata command"),
    provider: Optional[str] = typer.Option(None, "--provider", help="Filter to one provider"),
    contract: Optional[str] = typer.Option(None, "--contract", help="Filter to one contract name"),
    since: Optional[str] = typer.Option(None, "--since", help="ISO timestamp lower bound (inclusive)"),
    limit: int = typer.Option(50, "--limit", min=1, max=1000),
):
    """List recorded runs, most recent first."""
    from buffdata.governance import SQLiteAuditStore

    store = SQLiteAuditStore(_audit_db_path(db))
    records = store.query(command=command, provider=provider, contract_name=contract, since=since, limit=limit)

    if not records:
        console.print("[yellow]No matching runs recorded.[/yellow]")
        return

    table = Table(title=f"Audit log ({len(records)} run(s))")
    table.add_column("Recorded")
    table.add_column("Command")
    table.add_column("Provider/Model")
    table.add_column("Accepted", justify="right")
    table.add_column("Rejected", justify="right")
    table.add_column("Tokens", justify="right")
    table.add_column("Network")
    table.add_column("Contract")
    for r in records:
        contract_cell = "-" if not r.contract_name else f"{r.contract_name} ({'PASS' if r.contract_passed else 'FAIL'})"
        table.add_row(
            r.recorded_at, r.command, f"{r.provider or '-'}/{r.model or '-'}",
            str(r.accepted_records), str(r.rejected_records), str(r.total_tokens),
            r.network_policy or "-", contract_cell,
        )
    console.print(table)


@audit_app.command("usage-report")
def audit_usage_report_cmd(
    db: Optional[str] = typer.Option(None, "--db", help=f"Audit database path (default: {DEFAULT_AUDIT_DB}, or $BUFFDATA_AUDIT_DB)"),
    command: Optional[str] = typer.Option(None, "--command", help="Filter to one buffdata command"),
    provider: Optional[str] = typer.Option(None, "--provider", help="Filter to one provider"),
    since: Optional[str] = typer.Option(None, "--since", help="ISO timestamp lower bound (inclusive)"),
    pricing: Optional[Path] = typer.Option(None, "--pricing", help="YAML/JSON {\"provider/model\": {input_per_1k, output_per_1k}} to also estimate cost"),
):
    """Aggregate token usage across recorded runs. No dollar amounts are shown unless you
    supply --pricing with your own rates -- this command never guesses at provider pricing.
    """
    from buffdata.governance import SQLiteAuditStore, estimate_cost, summarize_usage

    store = SQLiteAuditStore(_audit_db_path(db))
    records = store.query(command=command, provider=provider, since=since, limit=100_000)
    if not records:
        console.print("[yellow]No matching runs recorded.[/yellow]")
        return

    usage = summarize_usage(records)
    console.print(Panel(
        f"[bold cyan]Usage across {usage['runs']} run(s)[/bold cyan]\n"
        f"Input tokens:  {usage['totals']['input_tokens']:,}\n"
        f"Output tokens: {usage['totals']['output_tokens']:,}\n"
        f"Total tokens:  {usage['totals']['total_tokens']:,}",
        border_style="cyan",
    ))

    table = Table(title="By provider/model")
    table.add_column("Provider/Model")
    table.add_column("Input tokens", justify="right")
    table.add_column("Output tokens", justify="right")
    table.add_column("Total tokens", justify="right")
    for key, totals in usage["by_provider_model"].items():
        table.add_row(key, f"{totals['input_tokens']:,}", f"{totals['output_tokens']:,}", f"{totals['total_tokens']:,}")
    console.print(table)

    if pricing is not None:
        rates = yaml.safe_load(pricing.read_text(encoding="utf-8")) or {}
        cost = estimate_cost(records, rates)
        console.print(f"\n[bold green]Estimated cost: ${cost['total_cost_usd']:,.2f}[/bold green]")
        if cost["unpriced_provider_models"]:
            console.print(
                f"[yellow]No pricing given for: {', '.join(cost['unpriced_provider_models'])} "
                f"({cost['unpriced_tokens']:,} tokens excluded from the estimate)[/yellow]"
            )


@app.command("sbom")
def sbom_cmd(
    output: Path = typer.Option("sbom.json", "-o", "--output", help="Where to write the CycloneDX SBOM"),
):
    """Generate a CycloneDX 1.5 Software Bill of Materials from the packages actually
    installed in this environment -- for a vendor security review or a compliance package.
    """
    import json

    from buffdata.governance import generate_sbom

    sbom = generate_sbom()
    output.write_text(json.dumps(sbom, indent=2), encoding="utf-8")
    console.print(
        f"[bold green]Wrote SBOM[/bold green] ({len(sbom['components'])} components) to {output}"
    )


auth_app = typer.Typer(
    help="Store or remove provider API keys in the OS-native credential store (Windows "
    "Credential Manager / macOS Keychain / Linux Secret Service), instead of a plaintext "
    "file or a shell history entry."
)
app.add_typer(auth_app, name="auth")

# Not exhaustive -- any name works, since the OS keyring just stores whatever key you give
# it -- but this is every secret name a stock BuffData install actually looks for, so
# `buffdata auth status` has something concrete to check.
_KNOWN_SECRET_NAMES = [
    "GEMINI_API_KEY", "GOOGLE_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY",
    "AZURE_OPENAI_API_KEY", "OPENAI_COMPATIBLE_API_KEY",
    "OLLAMA_API_KEY", "LMSTUDIO_API_KEY", "VLLM_API_KEY", "LLAMACPP_API_KEY",
]


def _keyring_service_name() -> str:
    return os.getenv("BUFFDATA_KEYRING_SERVICE", "buffdata")


@auth_app.command("set")
def auth_set_cmd(
    key_name: str = typer.Argument(
        ...,
        help=f"Secret name to store, e.g. one of: {', '.join(_KNOWN_SECRET_NAMES)}",
    ),
):
    """Store a provider API key in the OS keyring -- prompted with hidden input, never
    echoed to the terminal, never written to any file or shell history. Every buffdata
    command that needs it picks it up automatically afterward with no other
    configuration: the default secret backend checks the OS keyring whenever the
    matching environment variable isn't already set (see buffdata/engine/secrets.py).
    """
    try:
        import keyring
    except ImportError:
        console.print("[bold red]Install keyring (pip install keyring) to use `buffdata auth`.[/bold red]")
        raise typer.Exit(1)

    value = typer.prompt(f"Value for {key_name}", hide_input=True, confirmation_prompt=True)
    if not value:
        console.print("[yellow]Empty value -- nothing stored.[/yellow]")
        raise typer.Exit(1)
    try:
        keyring.set_password(_keyring_service_name(), key_name, value)
    except Exception as exc:
        console.print(f"[bold red]Could not store {key_name} in the OS keyring: {exc}[/bold red]")
        raise typer.Exit(1)
    console.print(
        f"[bold green]Stored {key_name} in the OS keyring.[/bold green] No .env file or "
        "BUFFDATA_SECRET_BACKEND change needed -- it's used automatically from here on."
    )


@auth_app.command("remove")
def auth_remove_cmd(
    key_name: str = typer.Argument(..., help="Secret name to remove, as previously passed to `buffdata auth set`"),
):
    """Remove a key previously stored via `buffdata auth set`."""
    try:
        import keyring
        from keyring.errors import PasswordDeleteError
    except ImportError:
        console.print("[bold red]Install keyring (pip install keyring) to use `buffdata auth`.[/bold red]")
        raise typer.Exit(1)

    try:
        keyring.delete_password(_keyring_service_name(), key_name)
        console.print(f"[bold green]Removed {key_name} from the OS keyring.[/bold green]")
    except PasswordDeleteError:
        console.print(f"[yellow]{key_name} was not set in the OS keyring.[/yellow]")


@auth_app.command("status")
def auth_status_cmd():
    """Show which known provider secrets currently resolve, and from where (environment
    variable vs. OS keyring) -- the values themselves are never displayed, here or
    anywhere else in buffdata.
    """
    table = Table(title="Secret resolution status", border_style="cyan")
    table.add_column("Name", style="bold")
    table.add_column("Resolves from")
    for name in _KNOWN_SECRET_NAMES:
        if os.getenv(name):
            source = "[green]environment[/green]"
        else:
            found = None
            try:
                import keyring

                found = keyring.get_password(_keyring_service_name(), name)
            except Exception:
                pass
            source = "[cyan]OS keyring[/cyan]" if found else "[dim]not set[/dim]"
        table.add_row(name, source)
    console.print(table)
