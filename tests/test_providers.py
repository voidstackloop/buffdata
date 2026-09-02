import pytest
from types import SimpleNamespace
from pydantic import BaseModel

from buffdata.engine.client import (
    AnthropicClient,
    AzureOpenAIClient,
    BedrockAnthropicClient,
    GeminiClient,
    LLMProvider,
    MissingAPIKeyError,
    OpenAIClient,
    OpenAICompatibleClient,
    ProviderError,
    create_llm_client,
)


@pytest.mark.parametrize(
    ("provider", "client_type", "model"),
    [
        ("gemini", GeminiClient, "gemini-test"),
        ("openai", OpenAIClient, "gpt-test"),
        ("anthropic", AnthropicClient, "claude-test"),
        ("azure_openai", AzureOpenAIClient, "my-azure-deployment"),
        ("bedrock_anthropic", BedrockAnthropicClient, "claude-bedrock-test"),
        ("openai_compatible", OpenAICompatibleClient, "local-llama"),
    ],
)
def test_provider_factory_selects_exact_provider(provider, client_type, model):
    client = create_llm_client(provider, model=model, allow_mock=True)
    assert isinstance(client, client_type)
    assert client.default_model == model
    assert client.provider == LLMProvider(provider)


def test_bedrock_provider_gets_a_default_model_without_being_asked(monkeypatch):
    monkeypatch.delenv("BUFFDATA_DEFAULT_MODEL", raising=False)
    client = create_llm_client("bedrock_anthropic", allow_mock=True)
    assert client.default_model == "us.anthropic.claude-sonnet-4-6-v1:0"


@pytest.mark.parametrize("provider", ["azure_openai", "openai_compatible"])
def test_private_endpoint_providers_refuse_to_guess_a_model(provider, monkeypatch):
    monkeypatch.delenv("BUFFDATA_DEFAULT_MODEL", raising=False)
    with pytest.raises(ProviderError, match="no universal default model"):
        create_llm_client(provider, allow_mock=True)


def test_provider_factory_does_not_fallback(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    client = create_llm_client("openai", allow_mock=False)
    with pytest.raises(MissingAPIKeyError, match="OPENAI_API_KEY"):
        _ = client.client


class StructuredFixture(BaseModel):
    value: str


def test_openai_adapter_uses_responses_parse():
    captured = {}

    def parse(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(output_parsed=StructuredFixture(value="ok"), usage=None)

    client = OpenAIClient(api_key="test", default_model="gpt-test")
    client._client = SimpleNamespace(responses=SimpleNamespace(parse=parse))
    result = client.generate_structured("input", StructuredFixture)
    assert result.value == "ok"
    assert captured["text_format"] is StructuredFixture
    assert captured["model"] == "gpt-test"


def test_anthropic_adapter_uses_supported_parse_arguments():
    captured = {}

    def parse(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(parsed_output=StructuredFixture(value="ok"), usage=None)

    client = AnthropicClient(api_key="test", default_model="claude-test")
    client._client = SimpleNamespace(messages=SimpleNamespace(parse=parse))
    result = client.generate_structured("input", StructuredFixture, temperature=0.4)
    assert result.value == "ok"
    assert captured["output_format"] is StructuredFixture
    assert "temperature" not in captured


def test_anthropic_adapter_marks_system_prompt_cacheable():
    captured = {}

    def parse(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(parsed_output=StructuredFixture(value="ok"), usage=None)

    client = AnthropicClient(api_key="test", default_model="claude-test")
    client._client = SimpleNamespace(messages=SimpleNamespace(parse=parse))
    client.generate_structured("input", StructuredFixture, system_instruction="Static system prompt.")

    assert captured["system"] == [
        {"type": "text", "text": "Static system prompt.", "cache_control": {"type": "ephemeral"}}
    ]


def test_anthropic_adapter_omits_system_when_not_given():
    captured = {}

    def parse(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(parsed_output=StructuredFixture(value="ok"), usage=None)

    client = AnthropicClient(api_key="test", default_model="claude-test")
    client._client = SimpleNamespace(messages=SimpleNamespace(parse=parse))
    client.generate_structured("input", StructuredFixture)

    assert "system" not in captured


def test_azure_openai_adapter_uses_chat_completions_parse():
    captured = {}

    def parse(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(parsed=StructuredFixture(value="ok")))],
            usage=None,
        )

    client = AzureOpenAIClient(
        api_key="test", default_model="my-deployment",
        azure_endpoint="https://example.openai.azure.com", api_version="2026-01-01-preview",
    )
    client._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(parse=parse)))
    result = client.generate_structured("input", StructuredFixture, system_instruction="Be terse.")

    assert result.value == "ok"
    assert captured["model"] == "my-deployment"
    assert captured["response_format"] is StructuredFixture
    assert captured["messages"] == [
        {"role": "system", "content": "Be terse."},
        {"role": "user", "content": "input"},
    ]


def test_azure_openai_requires_endpoint_even_with_a_key():
    client = AzureOpenAIClient(api_key="test", default_model="my-deployment", azure_endpoint=None)
    with pytest.raises(ProviderError, match="AZURE_OPENAI_ENDPOINT"):
        _ = client.client


def test_openai_compatible_adapter_targets_the_configured_base_url():
    captured = {}

    def parse(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(parsed=StructuredFixture(value="ok")))],
            usage=None,
        )

    client = OpenAICompatibleClient(default_model="local-llama", base_url="http://localhost:8000/v1")
    client._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(parse=parse)))
    result = client.generate_structured("input", StructuredFixture)

    assert result.value == "ok"
    assert captured["model"] == "local-llama"
    assert client.base_url == "http://localhost:8000/v1"


def test_openai_compatible_requires_a_base_url():
    client = OpenAICompatibleClient(default_model="local-llama", base_url=None)
    with pytest.raises(ProviderError, match="base URL is required"):
        _ = client.client


def test_bedrock_anthropic_reuses_anthropic_request_shape_and_caching():
    captured = {}

    def parse(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(parsed_output=StructuredFixture(value="ok"), usage=None)

    client = BedrockAnthropicClient(default_model="us.anthropic.claude-sonnet-4-6-v1:0")
    client._client = SimpleNamespace(messages=SimpleNamespace(parse=parse))
    result = client.generate_structured("input", StructuredFixture, system_instruction="Static prompt.")

    assert result.value == "ok"
    assert captured["model"] == "us.anthropic.claude-sonnet-4-6-v1:0"
    assert captured["system"] == [
        {"type": "text", "text": "Static prompt.", "cache_control": {"type": "ephemeral"}}
    ]


def test_bedrock_anthropic_has_no_api_key_concept():
    # Bedrock authenticates through the AWS credential chain, not a static key -- unlike
    # every other client here, there should be nothing resembling an api_key attribute.
    client = BedrockAnthropicClient(default_model="us.anthropic.claude-sonnet-4-6-v1:0")
    assert not hasattr(client, "api_key")


def test_gemini_usage_metadata_names_are_recorded():
    client = GeminiClient(api_key="test")
    client._record_usage(
        SimpleNamespace(
            prompt_token_count=11,
            candidates_token_count=7,
            total_token_count=21,
        )
    )

    assert client.usage == {"input_tokens": 11, "output_tokens": 7, "total_tokens": 21}
