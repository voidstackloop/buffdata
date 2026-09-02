import pytest
from typer.testing import CliRunner

from buffdata.cli.main import app
from buffdata.engine.client import (
    LLMProvider,
    NetworkForbiddenClient,
    ProviderError,
    create_llm_client,
)
from buffdata.engine.pipeline import OptimizationPipeline
from buffdata.models.formats import write_dataset
from buffdata.models.schemas import DatasetItem, PipelineConfig


# --- NetworkForbiddenClient itself --------------------------------------------------------

def test_network_forbidden_client_raises_on_every_method():
    client = NetworkForbiddenClient(would_be_provider=LLMProvider.GEMINI)
    with pytest.raises(ProviderError, match="strict.*blocked"):
        client.generate_structured("p", dict)
    with pytest.raises(ProviderError, match="strict.*blocked"):
        client.generate_text("p")
    with pytest.raises(ProviderError, match="strict.*blocked"):
        client.embed_texts(["a"])


@pytest.mark.asyncio
async def test_network_forbidden_client_raises_on_async_methods():
    client = NetworkForbiddenClient(would_be_provider=LLMProvider.ANTHROPIC)
    with pytest.raises(ProviderError, match="strict.*blocked"):
        await client.generate_structured_async("p", dict)
    with pytest.raises(ProviderError, match="strict.*blocked"):
        await client.generate_text_async("p")
    with pytest.raises(ProviderError, match="strict.*blocked"):
        await client.embed_texts_async(["a"])


def test_network_forbidden_client_error_names_the_provider_it_would_have_used():
    client = NetworkForbiddenClient(would_be_provider=LLMProvider.OPENAI)
    with pytest.raises(ProviderError, match="provider 'openai'"):
        client.generate_text("p")


# --- create_llm_client(network_policy="strict") -------------------------------------------

def test_create_llm_client_strict_returns_network_forbidden_client_without_any_credentials(monkeypatch):
    # No API key anywhere -- proves strict mode never even attempts to resolve a secret.
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    client = create_llm_client("gemini", network_policy="strict")
    assert isinstance(client, NetworkForbiddenClient)
    assert client.provider == LLMProvider.GEMINI


def test_create_llm_client_strict_still_validates_the_provider_name():
    with pytest.raises(ProviderError, match="Unknown provider"):
        create_llm_client("not-a-real-provider", network_policy="strict")


def test_create_llm_client_rejects_unknown_network_policy():
    with pytest.raises(ProviderError, match="network_policy"):
        create_llm_client("gemini", network_policy="paranoid")


# --- PipelineConfig fast pre-flight check ---------------------------------------------------

def test_pipeline_config_accepts_strict_with_compatible_settings():
    config = PipelineConfig(network_policy="strict", quality_mode="off", dedup_method="exact")
    assert config.network_policy == "strict"


def test_pipeline_config_rejects_strict_with_llm_quality_mode():
    with pytest.raises(ValueError, match="quality_mode='off'"):
        PipelineConfig(network_policy="strict", quality_mode="llm")


def test_pipeline_config_rejects_strict_with_sampled_quality_mode():
    with pytest.raises(ValueError, match="quality_mode='off'"):
        PipelineConfig(network_policy="strict", quality_mode="sampled")


def test_pipeline_config_rejects_strict_with_semantic_dedup():
    with pytest.raises(ValueError, match="dedup_method"):
        PipelineConfig(network_policy="strict", quality_mode="off", dedup_method="semantic")


def test_pipeline_config_strict_allows_auto_dedup():
    # "auto" only ever resolves to exact/minhash internally, never semantic -- must not be
    # rejected by the same guard that blocks the explicit "semantic" value.
    config = PipelineConfig(network_policy="strict", quality_mode="off", dedup_method="auto")
    assert config.dedup_method == "auto"


# --- End-to-end pipeline runs ---------------------------------------------------------------

def _labeled_items(n: int = 10) -> list[DatasetItem]:
    return [
        DatasetItem.from_dict({"text": f"apple orchard fruit stand {i}", "label": 0})
        if i % 2 == 0
        else DatasetItem.from_dict({"text": f"banana market fruit stand {i}", "label": 1})
        for i in range(n)
    ]


def _unlabeled_items(n: int = 10) -> list[DatasetItem]:
    return [DatasetItem.from_dict({"text": f"some unlabeled sentence number {i}"}) for i in range(n)]


@pytest.mark.asyncio
async def test_strict_pipeline_completes_with_zero_network_calls_on_already_labeled_data():
    # Labeled data means profile/classify both skip their remote call entirely (see
    # buffdata/engine/profiler.py), and quality_mode=off / dedup exact are both local --
    # this run must complete cleanly through a client that raises on any network attempt.
    config = PipelineConfig(
        provider="gemini", network_policy="strict", quality_mode="off",
        dedup_method="exact", classification="auto", scrub_pii=False,
    )
    pipeline = OptimizationPipeline(config)
    assert isinstance(pipeline.client, NetworkForbiddenClient)

    result = await pipeline.run(_labeled_items())
    assert len(result.accepted) == 10


@pytest.mark.asyncio
async def test_strict_pipeline_fails_clearly_when_unlabeled_data_needs_classification():
    # This is the case the config-only pre-flight check can't rule out (it's data-dependent),
    # so NetworkForbiddenClient is the thing that actually has to catch it.
    config = PipelineConfig(
        provider="gemini", network_policy="strict", quality_mode="off",
        dedup_method="exact", classification="auto", scrub_pii=False,
    )
    pipeline = OptimizationPipeline(config)

    with pytest.raises(ProviderError, match="strict.*blocked"):
        await pipeline.run(_unlabeled_items())


# --- CLI wiring -------------------------------------------------------------------------

def test_cli_score_with_strict_network_policy_fails_clearly(tmp_path):
    source = tmp_path / "train.jsonl"
    output = tmp_path / "scored.jsonl"
    write_dataset(_labeled_items(4), source)

    result = CliRunner().invoke(app, [
        "score", str(source), "-o", str(output), "--network-policy", "strict",
    ])

    # ProviderError isn't a click.UsageError, so Click doesn't format it into result.output --
    # it surfaces as an unhandled exception CliRunner captures in result.exception instead.
    assert result.exit_code != 0
    assert result.exception is not None
    assert "Network policy 'strict' blocked" in str(result.exception)


def test_cli_optimize_with_strict_network_policy_and_labeled_data_succeeds(tmp_path):
    source = tmp_path / "train.jsonl"
    output = tmp_path / "optimized.jsonl"
    write_dataset(_labeled_items(4), source)

    result = CliRunner().invoke(app, [
        "optimize", str(source), "-o", str(output),
        "--network-policy", "strict", "--quality-mode", "off", "--classification", "auto",
    ])

    assert result.exit_code == 0, result.output
    assert output.exists()


def test_cli_optimize_rejects_incompatible_strict_settings_before_doing_any_work(tmp_path):
    source = tmp_path / "train.jsonl"
    output = tmp_path / "optimized.jsonl"
    write_dataset(_labeled_items(4), source)

    result = CliRunner().invoke(app, [
        "optimize", str(source), "-o", str(output),
        "--network-policy", "strict", "--quality-mode", "llm",
    ])

    # PipelineConfig's validator raises pydantic.ValidationError, which -- like ProviderError
    # above -- isn't a click.UsageError, so it lands in result.exception, not result.output.
    assert result.exit_code != 0
    assert result.exception is not None
    assert "quality_mode='off'" in str(result.exception)
    assert not output.exists()
