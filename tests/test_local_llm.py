import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
from typer.testing import CliRunner

from buffdata.cli.main import app
from buffdata.engine.client import (
    LLMProvider,
    OpenAICompatibleClient,
    ProviderError,
    _is_local_network_host,
    create_llm_client,
)
from buffdata.models.formats import write_dataset
from buffdata.models.schemas import DatasetItem, PipelineConfig


# --- _is_local_network_host: pure logic, no network involved ------------------------------

@pytest.mark.parametrize("host", [
    "localhost",
    "127.0.0.1",
    "127.5.5.5",
    "::1",
    "10.0.0.1",
    "10.255.255.255",
    "172.16.0.1",
    "172.31.255.255",
    "192.168.1.1",
    "192.168.0.50",
    "169.254.1.1",  # link-local
    "myhost.local",
    "MyHost.LOCAL",  # case-insensitive
    "LOCALHOST",
])
def test_is_local_network_host_accepts_loopback_and_private_addresses(host):
    assert _is_local_network_host(host) is True


@pytest.mark.parametrize("host", [
    "8.8.8.8",
    "1.1.1.1",
    "93.184.216.34",
    "api.openai.com",
    "generativelanguage.googleapis.com",
    "myserver.example.com",
    "172.32.0.1",  # just outside the 172.16/12 private range
    "172.15.255.255",  # just below it
    "",
])
def test_is_local_network_host_rejects_public_hosts(host):
    assert _is_local_network_host(host) is False


# --- provider aliases resolve to OpenAICompatibleClient with the right preset -------------

@pytest.mark.parametrize(("provider", "expected_url"), [
    ("ollama", "http://localhost:11434/v1"),
    ("lmstudio", "http://localhost:1234/v1"),
    ("vllm", "http://localhost:8000/v1"),
    ("llamacpp", "http://localhost:8080/v1"),
])
def test_local_provider_aliases_resolve_to_their_preset_base_url(provider, expected_url):
    client = create_llm_client(provider, model="llama3.1")
    assert isinstance(client, OpenAICompatibleClient)
    assert client.base_url == expected_url
    assert client.provider == LLMProvider(provider)  # reports its own name, not "openai_compatible"


def test_local_provider_explicit_base_url_overrides_the_preset():
    client = create_llm_client("ollama", model="llama3.1", base_url="http://192.168.1.50:11434/v1")
    assert client.base_url == "http://192.168.1.50:11434/v1"


def test_local_provider_env_var_overrides_the_preset_but_not_an_explicit_base_url(monkeypatch):
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://gpu-box.local:11434/v1")
    client = create_llm_client("ollama", model="llama3.1")
    assert client.base_url == "http://gpu-box.local:11434/v1"

    client_explicit = create_llm_client("ollama", model="llama3.1", base_url="http://other:11434/v1")
    assert client_explicit.base_url == "http://other:11434/v1"


def test_local_provider_without_a_model_raises_naming_the_specific_provider(monkeypatch):
    monkeypatch.delenv("BUFFDATA_DEFAULT_MODEL", raising=False)
    with pytest.raises(ProviderError, match="provider 'ollama' has no universal default model"):
        create_llm_client("ollama")


def test_local_provider_api_key_defaults_to_not_required_when_unset(monkeypatch):
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
    client = create_llm_client("ollama", model="llama3.1")
    assert client.api_key == "not-required"


def test_local_provider_reads_its_own_api_key_env_var(monkeypatch):
    monkeypatch.setenv("VLLM_API_KEY", "team-shared-token")
    client = create_llm_client("vllm", model="mistral")
    assert client.api_key == "team-shared-token"


# --- network_policy="local": structural guarantee ------------------------------------------

@pytest.mark.parametrize("provider", ["gemini", "openai", "anthropic", "azure_openai", "bedrock_anthropic"])
def test_network_policy_local_rejects_cloud_providers_outright(provider):
    with pytest.raises(ProviderError, match="requires a provider whose endpoint you control"):
        create_llm_client(provider, model="whatever", network_policy="local")


def test_network_policy_local_accepts_a_loopback_base_url():
    client = create_llm_client("ollama", model="llama3.1", network_policy="local")
    assert isinstance(client, OpenAICompatibleClient)


def test_network_policy_local_accepts_a_private_lan_base_url():
    client = create_llm_client(
        "openai_compatible", model="llama3.1",
        base_url="http://192.168.1.50:8000/v1", network_policy="local",
    )
    assert client.base_url == "http://192.168.1.50:8000/v1"


def test_network_policy_local_blocks_a_public_base_url():
    with pytest.raises(ProviderError, match="blocked base_url"):
        create_llm_client(
            "openai_compatible", model="llama3.1",
            base_url="https://api.some-cloud-llm.com/v1", network_policy="local",
        )


def test_network_policy_local_requires_a_resolvable_base_url(monkeypatch):
    monkeypatch.delenv("OPENAI_COMPATIBLE_BASE_URL", raising=False)
    with pytest.raises(ProviderError, match="requires a base URL to validate"):
        create_llm_client("openai_compatible", model="llama3.1", network_policy="local")


def test_network_policy_accepts_only_the_three_known_values():
    with pytest.raises(ProviderError, match="network_policy must be"):
        create_llm_client("ollama", model="llama3.1", network_policy="paranoid")


# --- PipelineConfig: fast pre-flight for network_policy="local" ---------------------------

def test_pipeline_config_accepts_local_network_policy_with_an_eligible_provider():
    config = PipelineConfig(provider="ollama", model="llama3.1", network_policy="local", quality_mode="off")
    assert config.network_policy == "local"


def test_pipeline_config_rejects_local_network_policy_with_a_cloud_provider():
    with pytest.raises(ValueError, match="requires a provider whose endpoint you control"):
        PipelineConfig(provider="gemini", network_policy="local")


def test_pipeline_config_accepts_a_base_url_field():
    config = PipelineConfig(provider="ollama", base_url="http://192.168.1.50:11434/v1")
    assert config.base_url == "http://192.168.1.50:11434/v1"


# --- Real end-to-end round trip against a local OpenAI-compatible server ------------------
# A minimal HTTP server standing in for Ollama/LM Studio/vLLM's real /v1/chat/completions
# endpoint -- this proves the real `openai` SDK client actually sends a real HTTP request to
# 127.0.0.1 and parses a real HTTP response, not a mocked object.

class _FakeLocalLLMHandler(BaseHTTPRequestHandler):
    received_requests: list = []

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length))
        _FakeLocalLLMHandler.received_requests.append({"path": self.path, "body": body})
        response = {
            "id": "chatcmpl-fake-local-1",
            "object": "chat.completion",
            "created": 1700000000,
            "model": body.get("model", "unknown"),
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "Hello from the fake local LLM server!"},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 12, "completion_tokens": 8, "total_tokens": 20},
        }
        payload = json.dumps(response).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format, *args):
        pass  # silence default request logging


@pytest.fixture
def fake_local_llm_server():
    _FakeLocalLLMHandler.received_requests = []
    server = HTTPServer(("127.0.0.1", 0), _FakeLocalLLMHandler)
    port = server.server_port
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{port}/v1"
    finally:
        server.shutdown()
        thread.join(timeout=5)


def test_ollama_provider_makes_a_real_http_round_trip_to_a_local_server(fake_local_llm_server):
    client = create_llm_client("ollama", model="llama3.1", base_url=fake_local_llm_server)

    result = client.generate_text("What model are you?")

    assert result == "Hello from the fake local LLM server!"
    assert len(_FakeLocalLLMHandler.received_requests) == 1
    sent = _FakeLocalLLMHandler.received_requests[0]
    assert sent["path"] == "/v1/chat/completions"
    assert sent["body"]["model"] == "llama3.1"
    assert sent["body"]["messages"][0]["content"] == "What model are you?"
    assert client.usage["total_tokens"] == 20


@pytest.mark.asyncio
async def test_ollama_provider_makes_a_real_async_http_round_trip(fake_local_llm_server):
    client = create_llm_client("ollama", model="llama3.1", base_url=fake_local_llm_server)

    result = await client.generate_text_async("ping")

    assert result == "Hello from the fake local LLM server!"


def test_network_policy_local_end_to_end_against_a_real_loopback_server(fake_local_llm_server):
    # network_policy="local" doesn't just permit the client to be constructed -- confirm a
    # real call through it still reaches the real (loopback) server successfully.
    client = create_llm_client("ollama", model="llama3.1", base_url=fake_local_llm_server, network_policy="local")
    result = client.generate_text("hi")
    assert result == "Hello from the fake local LLM server!"


# --- CLI wiring -------------------------------------------------------------------------

def test_cli_score_reaches_a_real_local_server_via_base_url_flag(fake_local_llm_server, tmp_path):
    # score_cmd uses generate_structured_async (chat.completions.parse), which the fake
    # server's plain-text response can't satisfy -- so this deliberately fails downstream,
    # the same way tests/test_network_policy.py's CLI tests let strict-mode runs fail for an
    # unrelated, expected reason. What this proves is that --provider/--base-url actually
    # reached a real client that made a real HTTP request to the local server (confirmed via
    # the server's own request log), rather than being blocked or short-circuited earlier.
    source = tmp_path / "train.jsonl"
    output = tmp_path / "scored.jsonl"
    write_dataset([DatasetItem.from_dict({"text": "a sample row"})], source)

    CliRunner().invoke(app, [
        "score", str(source), "-o", str(output),
        "--provider", "ollama", "--base-url", fake_local_llm_server, "--model", "llama3.1",
    ])

    assert len(_FakeLocalLLMHandler.received_requests) >= 1
    assert _FakeLocalLLMHandler.received_requests[0]["body"]["model"] == "llama3.1"


def test_cli_score_with_network_policy_local_blocks_a_public_base_url(tmp_path):
    source = tmp_path / "train.jsonl"
    output = tmp_path / "scored.jsonl"
    write_dataset([DatasetItem.from_dict({"text": "a sample row"})], source)

    result = CliRunner().invoke(app, [
        "score", str(source), "-o", str(output),
        "--provider", "openai_compatible", "--base-url", "https://not-local.example.com/v1",
        "--model", "llama3.1", "--network-policy", "local",
    ])

    assert result.exit_code != 0
    assert "blocked base_url" in str(result.output) + str(result.exception)
    assert not output.exists()
