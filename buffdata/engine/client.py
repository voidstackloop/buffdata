"""Provider-neutral language-model clients used by BuffData."""

from __future__ import annotations

import asyncio
import os
import threading
from enum import Enum
from typing import Any, List, Optional, Protocol, Type, TypeVar, runtime_checkable

from dotenv import load_dotenv
from pydantic import BaseModel

from buffdata.engine.secrets import get_default_secret_resolver

load_dotenv()

T = TypeVar("T", bound=BaseModel)

# None of the provider SDKs default to a bounded request timeout that's actually safe for
# an unattended CLI/batch run: google-genai's default is unbounded (a stalled connection
# hangs forever, confirmed by hand -- a single-row classify call never returned), and
# openai/anthropic default to 600s, long enough to stall a whole run over one bad request.
# Every client below applies this instead, so a real network hiccup fails fast and lets
# tenacity's retry (buffdata/engine/limiter.py) actually do its job, rather than the
# request never completing in the first place.
DEFAULT_REQUEST_TIMEOUT_SECONDS = float(os.getenv("BUFFDATA_REQUEST_TIMEOUT", "120"))


def _resolve_secret(*keys: str) -> Optional[str]:
    """Look up each key, in order, through the configured secret backend (plain
    environment variables by default -- see engine/secrets.py). Every client's API key
    resolution goes through this instead of os.getenv directly, so pointing
    BUFFDATA_SECRET_BACKEND at Vault/AWS/GCP/Azure Key Vault changes every provider's
    credential source without touching any client class.
    """
    resolver = get_default_secret_resolver()
    for key in keys:
        value = resolver.get(key)
        if value:
            from buffdata.security.policy import remember_secret
            return remember_secret(value)
    return None


class LLMProvider(str, Enum):
    GEMINI = "gemini"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    AZURE_OPENAI = "azure_openai"
    BEDROCK_ANTHROPIC = "bedrock_anthropic"
    OPENAI_COMPATIBLE = "openai_compatible"
    OLLAMA = "ollama"
    LMSTUDIO = "lmstudio"
    VLLM = "vllm"
    LLAMACPP = "llamacpp"


# Private-endpoint providers (AZURE_OPENAI, OPENAI_COMPATIBLE, and the four local-server
# providers below) deliberately have no entry here: an Azure "model" is a customer-created
# deployment name, a gateway's model name is whatever that gateway exposes, and a local
# server's model name is whatever the operator pulled/loaded -- there is no universal
# default that wouldn't silently be wrong.
DEFAULT_MODELS = {
    LLMProvider.GEMINI: "gemini-3.7-flash",
    LLMProvider.OPENAI: "gpt-5.4-mini",
    LLMProvider.ANTHROPIC: "claude-sonnet-4-6",
    LLMProvider.BEDROCK_ANTHROPIC: "us.anthropic.claude-sonnet-4-6-v1:0",
}

# Providers for an LLM server running on the user's own machine or another machine on their
# network (Ollama, LM Studio, vLLM, llama.cpp's server) -- all speak the same OpenAI
# chat-completions protocol underneath (OpenAICompatibleClient), so these exist as their own
# named providers only so `--provider ollama` fills in that tool's well-known default local
# port automatically instead of requiring `--provider openai_compatible --base-url
# http://localhost:11434/v1` spelled out by hand. Every default here is just a default:
# --base-url (or the provider's own *_BASE_URL env var below) points at any other host on the
# network, e.g. --provider ollama --base-url http://192.168.1.50:11434/v1 for a shared GPU box.
_LOCAL_PROVIDER_DEFAULTS: dict["LLMProvider", tuple[str, str, str]] = {
    LLMProvider.OLLAMA: ("http://localhost:11434/v1", "OLLAMA_BASE_URL", "OLLAMA_API_KEY"),
    LLMProvider.LMSTUDIO: ("http://localhost:1234/v1", "LMSTUDIO_BASE_URL", "LMSTUDIO_API_KEY"),
    LLMProvider.VLLM: ("http://localhost:8000/v1", "VLLM_BASE_URL", "VLLM_API_KEY"),
    LLMProvider.LLAMACPP: ("http://localhost:8080/v1", "LLAMACPP_BASE_URL", "LLAMACPP_API_KEY"),
}


def _is_local_network_host(host: str) -> bool:
    """True only when `host` is an unambiguous, DNS-free indicator of a loopback or private
    network address: a literal IP in a loopback/private/link-local range, the literal
    "localhost", or an mDNS-style ".local" hostname. Deliberately never resolves DNS to make
    this determination -- a hostname that isn't obviously local by its own text is treated as
    NOT local even if it happens to resolve to a private address today (behind a VPN, a
    corporate resolver, ...), since that resolution could silently change later without this
    check's guarantee changing with it. Used only to gate network_policy="local"; has no
    bearing on "unrestricted", where any host is allowed.
    """
    import ipaddress

    normalized = host.strip().strip("[]").lower()
    if normalized == "localhost" or normalized.endswith(".local"):
        return True
    try:
        parsed_ip = ipaddress.ip_address(normalized)
    except ValueError:
        return False
    return parsed_ip.is_loopback or parsed_ip.is_private or parsed_ip.is_link_local


class ProviderError(RuntimeError):
    """Base error for provider failures."""


class MissingAPIKeyError(ProviderError):
    """Raised when a selected provider has no API key."""


@runtime_checkable
class LLMClient(Protocol):
    provider: LLMProvider
    default_model: str
    usage: dict[str, int]

    def generate_structured(
        self,
        prompt: str,
        response_schema: Type[T],
        model: Optional[str] = None,
        system_instruction: Optional[str] = None,
        temperature: float = 0.2,
    ) -> T: ...

    async def generate_structured_async(
        self,
        prompt: str,
        response_schema: Type[T],
        model: Optional[str] = None,
        system_instruction: Optional[str] = None,
        temperature: float = 0.2,
    ) -> T: ...

    def generate_text(
        self,
        prompt: str,
        model: Optional[str] = None,
        system_instruction: Optional[str] = None,
        temperature: float = 0.7,
    ) -> str: ...

    async def generate_text_async(
        self,
        prompt: str,
        model: Optional[str] = None,
        system_instruction: Optional[str] = None,
        temperature: float = 0.7,
    ) -> str: ...


class NetworkForbiddenClient:
    """A structural guarantee that no network call happens, not a convention every stage has
    to keep honoring correctly forever. When --network-policy strict is set, create_llm_client
    returns this instead of a real provider client -- every method that would reach a network
    raises immediately, with a clear explanation, rather than reaching a provider. A security
    reviewer can verify "this run cannot call out" by reading this one class instead of
    auditing every optimizer stage's internal logic, today and after every future change.
    """

    default_model = "network-forbidden"
    usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    def __init__(self, would_be_provider: Optional["LLMProvider"] = None):
        self.provider = would_be_provider

    def _raise(self, method: str) -> None:
        target = f" (would have gone to provider '{self.provider.value}')" if self.provider else ""
        raise ProviderError(
            f"Network policy 'strict' blocked a {method} call{target}. Remove --network-policy "
            "strict, or adjust configuration so no stage needs a remote call: quality_mode=off, "
            "dedup_method not 'semantic', and classification=off (or already-labeled data, "
            "which skips classification and profiling calls automatically)."
        )

    def generate_structured(self, *args: Any, **kwargs: Any) -> Any:
        self._raise("generate_structured")

    async def generate_structured_async(self, *args: Any, **kwargs: Any) -> Any:
        self._raise("generate_structured_async")

    def generate_text(self, *args: Any, **kwargs: Any) -> Any:
        self._raise("generate_text")

    async def generate_text_async(self, *args: Any, **kwargs: Any) -> Any:
        self._raise("generate_text_async")

    def embed_texts(self, *args: Any, **kwargs: Any) -> Any:
        self._raise("embed_texts")

    async def embed_texts_async(self, *args: Any, **kwargs: Any) -> Any:
        self._raise("embed_texts_async")


class _ClientBase:
    provider: LLMProvider

    def __init__(self, default_model: str, allow_mock: bool = False):
        self.default_model = default_model
        self.allow_mock = allow_mock
        self.usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        self._usage_lock = threading.Lock()

    def _record_usage(self, usage: Any) -> None:
        if usage is None:
            return
        input_tokens = int(
            getattr(usage, "input_tokens", None)
            or getattr(usage, "prompt_tokens", None)
            or getattr(usage, "prompt_token_count", None)
            or 0
        )
        output_tokens = int(
            getattr(usage, "output_tokens", None)
            or getattr(usage, "completion_tokens", None)
            or getattr(usage, "candidates_token_count", None)
            or 0
        )
        total_tokens = int(
            getattr(usage, "total_tokens", None)
            or getattr(usage, "total_token_count", None)
            or input_tokens + output_tokens
        )
        with self._usage_lock:
            self.usage["input_tokens"] += input_tokens
            self.usage["output_tokens"] += output_tokens
            self.usage["total_tokens"] += total_tokens

    async def generate_structured_async(
        self,
        prompt: str,
        response_schema: Type[T],
        model: Optional[str] = None,
        system_instruction: Optional[str] = None,
        temperature: float = 0.2,
    ) -> T:
        return await asyncio.to_thread(
            self.generate_structured,
            prompt,
            response_schema,
            model,
            system_instruction,
            temperature,
        )

    async def generate_text_async(
        self,
        prompt: str,
        model: Optional[str] = None,
        system_instruction: Optional[str] = None,
        temperature: float = 0.7,
    ) -> str:
        return await asyncio.to_thread(
            self.generate_text,
            prompt,
            model,
            system_instruction,
            temperature,
        )

    def _mock_structured(self, schema_cls: Type[T]) -> T:
        from buffdata.models.schemas import (
            EvolutionResult,
            PreferenceResult,
            QualityScore,
            RefinementResult,
        )

        if schema_cls == QualityScore:
            return QualityScore(
                overall_score=8.5,
                clarity=9.0,
                factual_accuracy=8.5,
                reasoning_depth=8.0,
                instruction_following=9.0,
                is_safe=True,
                issues=["Minor brevity"],
                recommendations="Expand the explanation.",
            )
        if schema_cls == RefinementResult:
            return RefinementResult(
                refined_prompt="Clarified prompt with clear instructions.",
                refined_response="High-quality refined response with clean formatting.",
                reasoning_added=True,
                formatting_fixed=True,
                artifacts_removed=["Certainly!"],
                explanation_of_changes="Removed boilerplate and improved structure.",
            )
        if schema_cls == EvolutionResult:
            return EvolutionResult(
                evolved_prompt="Complex prompt with explicit constraints.",
                evolved_response="Detailed response satisfying the evolved prompt.",
                evolution_type="deepen_reasoning",
                complexity_score=8.5,
            )
        if schema_cls == PreferenceResult:
            return PreferenceResult(
                prompt="Explain backpropagation.",
                chosen="Backpropagation computes gradients using the chain rule.",
                rejected="Backpropagation changes weights.",
                rejection_reason="The rejected answer is incomplete.",
            )
        if schema_cls.__name__ == "ClassificationSchema":
            return schema_cls(
                applicable=True,
                task_type="multi-class",
                classes=["World", "Sports", "Business", "Sci/Tech"],
                confidence=0.9,
                reasoning="Detected a closed-set news-topic dataset.",
            )
        if schema_cls.__name__ == "ClassificationResult":
            return schema_cls(labels=["World"], confidence=0.8)
        if schema_cls.__name__ == "SyntheticDataset":
            return schema_cls(items=[])
        return schema_cls.model_construct()


class _ChatCompletionsClient(_ClientBase):
    """Shared OpenAI Chat Completions structured-output logic for any provider that speaks
    the OpenAI API surface but isn't OpenAI's own public endpoint. Chat Completions (not the
    newer Responses API that OpenAIClient uses) is the surface Azure OpenAI and third-party
    gateways/self-hosted servers (vLLM, TGI, LiteLLM proxies) actually implement. Subclasses
    provide only `__init__` and the `client` property.
    """

    @staticmethod
    def _messages(prompt: str, system_instruction: Optional[str]) -> list[dict[str, str]]:
        messages = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": "user", "content": prompt})
        return messages

    def generate_structured(
        self,
        prompt: str,
        response_schema: Type[T],
        model: Optional[str] = None,
        system_instruction: Optional[str] = None,
        temperature: float = 0.2,
    ) -> T:
        if self._mock_mode:
            return self._mock_structured(response_schema)
        response = self.client.chat.completions.parse(
            model=model or self.default_model,
            messages=self._messages(prompt, system_instruction),
            response_format=response_schema,
            temperature=temperature,
        )
        self._record_usage(getattr(response, "usage", None))
        parsed = response.choices[0].message.parsed
        if parsed is None:
            raise ProviderError(f"{self.provider.value} returned no parsed structured output.")
        return parsed

    def generate_text(
        self,
        prompt: str,
        model: Optional[str] = None,
        system_instruction: Optional[str] = None,
        temperature: float = 0.7,
    ) -> str:
        if self._mock_mode:
            return f"[Simulated {self.provider.value} response to: {prompt[:40]}...]"
        response = self.client.chat.completions.create(
            model=model or self.default_model,
            messages=self._messages(prompt, system_instruction),
            temperature=temperature,
        )
        self._record_usage(getattr(response, "usage", None))
        return response.choices[0].message.content


class AzureOpenAIClient(_ChatCompletionsClient):
    """Routes through the customer's own Azure OpenAI resource instead of OpenAI's public
    API -- the model, the request, and the response all stay inside the customer's Azure
    tenant and region.
    """

    provider = LLMProvider.AZURE_OPENAI

    def __init__(
        self,
        api_key: Optional[str] = None,
        default_model: Optional[str] = None,
        azure_endpoint: Optional[str] = None,
        api_version: Optional[str] = None,
        allow_mock: bool = False,
    ):
        if not default_model:
            raise ProviderError(
                "provider 'azure_openai' has no universal default model: pass --model with "
                "your Azure deployment name, or set BUFFDATA_DEFAULT_MODEL."
            )
        super().__init__(default_model=default_model, allow_mock=allow_mock)
        self.api_key = api_key or _resolve_secret("AZURE_OPENAI_API_KEY")
        self.azure_endpoint = azure_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
        self.api_version = api_version or os.getenv("AZURE_OPENAI_API_VERSION", "2026-01-01-preview")
        self._client = None
        self._mock_mode = not bool(self.api_key) and allow_mock

    @property
    def client(self):
        if self._client is None:
            if not self.api_key:
                raise MissingAPIKeyError("AZURE_OPENAI_API_KEY is required for provider 'azure_openai'.")
            if not self.azure_endpoint:
                raise ProviderError("AZURE_OPENAI_ENDPOINT is required for provider 'azure_openai'.")
            try:
                from openai import AzureOpenAI
            except ImportError as exc:
                raise ProviderError("Install openai to use provider 'azure_openai'.") from exc
            self._client = AzureOpenAI(
                api_key=self.api_key,
                azure_endpoint=self.azure_endpoint,
                api_version=self.api_version,
                timeout=DEFAULT_REQUEST_TIMEOUT_SECONDS,
            )
        return self._client


class OpenAICompatibleClient(_ChatCompletionsClient):
    """Routes to any server implementing the OpenAI Chat Completions API surface: an internal
    LLM gateway, a self-hosted vLLM/TGI deployment, a proxy like LiteLLM, or -- via the
    OLLAMA/LMSTUDIO/VLLM/LLAMACPP provider names, which all construct this same class with a
    different preset default base_url -- an LLM running on the user's own machine or another
    machine on their network. The escape hatch for organizations that have already stood up
    their own approved model-serving layer and don't want BuffData talking to any public
    provider at all.
    """

    provider = LLMProvider.OPENAI_COMPATIBLE

    def __init__(
        self,
        api_key: Optional[str] = None,
        default_model: Optional[str] = None,
        base_url: Optional[str] = None,
        allow_mock: bool = False,
        provider_label: Optional["LLMProvider"] = None,
    ):
        if not default_model:
            label = provider_label.value if provider_label else "openai_compatible"
            raise ProviderError(
                f"provider '{label}' has no universal default model: pass --model matching "
                "whatever your gateway/server exposes (e.g. the name shown by `ollama list`)."
            )
        super().__init__(default_model=default_model, allow_mock=allow_mock)
        if provider_label is not None:
            self.provider = provider_label  # instance override -- reporting/audit sees e.g. "ollama", not the generic "openai_compatible"
        api_key_env = "OPENAI_COMPATIBLE_API_KEY"
        base_url_env = "OPENAI_COMPATIBLE_BASE_URL"
        if provider_label in _LOCAL_PROVIDER_DEFAULTS:
            preset_url, base_url_env, api_key_env = _LOCAL_PROVIDER_DEFAULTS[provider_label]
        else:
            preset_url = None
        self.api_key = api_key or _resolve_secret(api_key_env) or "not-required"
        self.base_url = base_url or os.getenv(base_url_env) or preset_url
        self._client = None
        self._mock_mode = not bool(self.base_url) and allow_mock

    @property
    def client(self):
        if self._client is None:
            if not self.base_url:
                raise ProviderError(f"A base URL is required for provider '{self.provider.value}' -- pass --base-url.")
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise ProviderError(f"Install openai to use provider '{self.provider.value}'.") from exc
            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=DEFAULT_REQUEST_TIMEOUT_SECONDS)
        return self._client


class GeminiClient(_ClientBase):
    """Google Gemini adapter. Mock mode remains available for legacy offline tests."""

    provider = LLMProvider.GEMINI

    def __init__(
        self,
        api_key: Optional[str] = None,
        default_model: str = DEFAULT_MODELS[LLMProvider.GEMINI],
        embedding_model: str = "text-embedding-004",
        allow_mock: bool = True,
    ):
        super().__init__(default_model=default_model, allow_mock=allow_mock)
        self.api_key = api_key or _resolve_secret("GEMINI_API_KEY", "GOOGLE_API_KEY")
        self.embedding_model = embedding_model
        self._client = None
        self._mock_mode = not bool(self.api_key) and allow_mock

    @property
    def client(self):
        if self._client is None:
            if not self.api_key:
                raise MissingAPIKeyError("GEMINI_API_KEY is required for provider 'gemini'.")
            try:
                from google import genai
                from google.genai import types as genai_types
            except ImportError as exc:
                raise ProviderError("Install google-genai to use provider 'gemini'.") from exc
            self._client = genai.Client(
                api_key=self.api_key,
                http_options=genai_types.HttpOptions(timeout=int(DEFAULT_REQUEST_TIMEOUT_SECONDS * 1000)),
            )
        return self._client

    def generate_structured(
        self,
        prompt: str,
        response_schema: Type[T],
        model: Optional[str] = None,
        system_instruction: Optional[str] = None,
        temperature: float = 0.2,
    ) -> T:
        if self._mock_mode:
            return self._mock_structured(response_schema)
        from google.genai import types

        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=response_schema,
            temperature=temperature,
            system_instruction=system_instruction,
        )
        response = self.client.models.generate_content(
            model=model or self.default_model,
            contents=prompt,
            config=config,
        )
        self._record_usage(getattr(response, "usage_metadata", None))
        parsed = getattr(response, "parsed", None)
        return parsed if parsed is not None else response_schema.model_validate_json(response.text)

    def generate_text(
        self,
        prompt: str,
        model: Optional[str] = None,
        system_instruction: Optional[str] = None,
        temperature: float = 0.7,
    ) -> str:
        if self._mock_mode:
            return f"[Simulated Gemini response to: {prompt[:40]}...]"
        from google.genai import types

        response = self.client.models.generate_content(
            model=model or self.default_model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=temperature,
                system_instruction=system_instruction,
            ),
        )
        self._record_usage(getattr(response, "usage_metadata", None))
        return response.text

    def embed_texts(self, texts: List[str], model: Optional[str] = None) -> List[List[float]]:
        if self._mock_mode:
            import numpy as np

            return [np.zeros(768).tolist() for _ in texts]
        results: List[List[float]] = []
        for text in texts:
            response = self.client.models.embed_content(
                model=model or self.embedding_model,
                contents=text,
            )
            results.append(response.embedding.values)
        return results

    async def embed_texts_async(
        self, texts: List[str], model: Optional[str] = None
    ) -> List[List[float]]:
        return await asyncio.to_thread(self.embed_texts, texts=texts, model=model)


class OpenAIClient(_ClientBase):
    """OpenAI Responses API adapter using Pydantic structured outputs."""

    provider = LLMProvider.OPENAI

    def __init__(
        self,
        api_key: Optional[str] = None,
        default_model: str = DEFAULT_MODELS[LLMProvider.OPENAI],
        allow_mock: bool = False,
    ):
        super().__init__(default_model=default_model, allow_mock=allow_mock)
        self.api_key = api_key or _resolve_secret("OPENAI_API_KEY")
        self._client = None
        self._mock_mode = not bool(self.api_key) and allow_mock

    @property
    def client(self):
        if self._client is None:
            if not self.api_key:
                raise MissingAPIKeyError("OPENAI_API_KEY is required for provider 'openai'.")
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise ProviderError("Install openai to use provider 'openai'.") from exc
            self._client = OpenAI(api_key=self.api_key, timeout=DEFAULT_REQUEST_TIMEOUT_SECONDS)
        return self._client

    @staticmethod
    def _input(prompt: str, system_instruction: Optional[str]) -> list[dict[str, str]]:
        messages = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": "user", "content": prompt})
        return messages

    def generate_structured(
        self,
        prompt: str,
        response_schema: Type[T],
        model: Optional[str] = None,
        system_instruction: Optional[str] = None,
        temperature: float = 0.2,
    ) -> T:
        if self._mock_mode:
            return self._mock_structured(response_schema)
        response = self.client.responses.parse(
            model=model or self.default_model,
            input=self._input(prompt, system_instruction),
            text_format=response_schema,
            temperature=temperature,
        )
        self._record_usage(getattr(response, "usage", None))
        parsed = getattr(response, "output_parsed", None)
        if parsed is None:
            raise ProviderError("OpenAI returned no parsed structured output.")
        return parsed

    def generate_text(
        self,
        prompt: str,
        model: Optional[str] = None,
        system_instruction: Optional[str] = None,
        temperature: float = 0.7,
    ) -> str:
        if self._mock_mode:
            return f"[Simulated OpenAI response to: {prompt[:40]}...]"
        response = self.client.responses.create(
            model=model or self.default_model,
            input=self._input(prompt, system_instruction),
            temperature=temperature,
        )
        self._record_usage(getattr(response, "usage", None))
        return response.output_text


class AnthropicClient(_ClientBase):
    """Anthropic Messages API adapter using Pydantic structured outputs."""

    provider = LLMProvider.ANTHROPIC

    def __init__(
        self,
        api_key: Optional[str] = None,
        default_model: str = DEFAULT_MODELS[LLMProvider.ANTHROPIC],
        allow_mock: bool = False,
    ):
        super().__init__(default_model=default_model, allow_mock=allow_mock)
        self.api_key = api_key or _resolve_secret("ANTHROPIC_API_KEY")
        self._client = None
        self._mock_mode = not bool(self.api_key) and allow_mock

    @property
    def client(self):
        if self._client is None:
            if not self.api_key:
                raise MissingAPIKeyError("ANTHROPIC_API_KEY is required for provider 'anthropic'.")
            try:
                from anthropic import Anthropic
            except ImportError as exc:
                raise ProviderError("Install anthropic to use provider 'anthropic'.") from exc
            self._client = Anthropic(api_key=self.api_key, timeout=DEFAULT_REQUEST_TIMEOUT_SECONDS)
        return self._client

    @staticmethod
    def _cacheable_system(system_instruction: str) -> list[dict[str, Any]]:
        """Every system prompt in this codebase is a static, module-level constant reused
        verbatim across every call for a given stage. Anthropic (unlike OpenAI/Gemini, which
        cache repeated prefixes automatically) only discounts repeat-token cost when a cache
        breakpoint is marked explicitly -- so mark it, always. Below Anthropic's minimum
        cacheable length this is a harmless no-op.
        """
        return [{"type": "text", "text": system_instruction, "cache_control": {"type": "ephemeral"}}]

    def generate_structured(
        self,
        prompt: str,
        response_schema: Type[T],
        model: Optional[str] = None,
        system_instruction: Optional[str] = None,
        temperature: float = 0.2,
    ) -> T:
        if self._mock_mode:
            return self._mock_structured(response_schema)
        kwargs: dict[str, Any] = {
            "model": model or self.default_model,
            "max_tokens": 4096,
            "messages": [{"role": "user", "content": prompt}],
            "output_format": response_schema,
        }
        if system_instruction:
            kwargs["system"] = self._cacheable_system(system_instruction)
        response = self.client.messages.parse(**kwargs)
        self._record_usage(getattr(response, "usage", None))
        parsed = getattr(response, "parsed_output", None)
        if parsed is None:
            raise ProviderError("Anthropic returned no parsed structured output.")
        return parsed

    def generate_text(
        self,
        prompt: str,
        model: Optional[str] = None,
        system_instruction: Optional[str] = None,
        temperature: float = 0.7,
    ) -> str:
        if self._mock_mode:
            return f"[Simulated Anthropic response to: {prompt[:40]}...]"
        kwargs: dict[str, Any] = {
            "model": model or self.default_model,
            "max_tokens": 4096,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
        }
        if system_instruction:
            kwargs["system"] = self._cacheable_system(system_instruction)
        response = self.client.messages.create(**kwargs)
        self._record_usage(getattr(response, "usage", None))
        return "".join(
            block.text for block in response.content if getattr(block, "type", None) == "text"
        )


class BedrockAnthropicClient(AnthropicClient):
    """Routes Claude calls through the customer's own AWS account (Bedrock) instead of
    Anthropic's public API. Reuses AnthropicClient's request/response handling -- including
    prompt caching -- unchanged; only how the underlying SDK client authenticates differs.
    Authenticates through the standard AWS credential chain (IAM role, profile, or env vars)
    rather than a static key BuffData holds, matching how enterprises actually grant and
    audit model access.
    """

    provider = LLMProvider.BEDROCK_ANTHROPIC

    def __init__(
        self,
        default_model: str,
        aws_region: Optional[str] = None,
        allow_mock: bool = False,
    ):
        _ClientBase.__init__(self, default_model=default_model, allow_mock=allow_mock)
        self.aws_region = aws_region or os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION")
        self._client = None
        self._mock_mode = allow_mock and not bool(
            os.getenv("AWS_ACCESS_KEY_ID") or os.getenv("AWS_PROFILE")
        )

    @property
    def client(self):
        if self._client is None:
            try:
                from anthropic import AnthropicBedrock
            except ImportError as exc:
                raise ProviderError(
                    "Install boto3 (pip install buffdata[enterprise]) to use provider 'bedrock_anthropic'."
                ) from exc
            self._client = (
                AnthropicBedrock(aws_region=self.aws_region, timeout=DEFAULT_REQUEST_TIMEOUT_SECONDS)
                if self.aws_region
                else AnthropicBedrock(timeout=DEFAULT_REQUEST_TIMEOUT_SECONDS)
            )
        return self._client


_NO_DEFAULT_MODEL_PROVIDERS = {
    LLMProvider.AZURE_OPENAI,
    LLMProvider.OPENAI_COMPATIBLE,
    *_LOCAL_PROVIDER_DEFAULTS,
}

# Providers routed through OpenAICompatibleClient: a customer- or user-controlled endpoint,
# never a public cloud API by construction -- the only providers network_policy="local" can
# ever allow.
_BASE_URL_ROUTED_PROVIDERS = {LLMProvider.OPENAI_COMPATIBLE, *_LOCAL_PROVIDER_DEFAULTS}


def create_llm_client(
    provider: str | LLMProvider = LLMProvider.GEMINI,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    *,
    allow_mock: bool = False,
    base_url: Optional[str] = None,
    azure_endpoint: Optional[str] = None,
    azure_api_version: Optional[str] = None,
    aws_region: Optional[str] = None,
    network_policy: str = "unrestricted",
) -> LLMClient:
    """Create exactly the requested provider; this function never falls back.

    The three private-endpoint providers (azure_openai, bedrock_anthropic, openai_compatible)
    route the same request/response contract through infrastructure the customer controls
    instead of a public API -- every optimizer stage calls generate_structured_async the same
    way regardless of which provider is selected here. Endpoint/region/base-url details are
    read from environment variables when not passed explicitly, matching how API keys already
    work for the public providers.

    network_policy="strict" skips constructing the real provider client entirely and returns
    a NetworkForbiddenClient instead -- the provider/model are still validated first, so a
    typo still fails clearly, but no credential is ever read and no request can ever be made.

    network_policy="local" is the middle ground: calls are allowed, but only to a provider
    whose endpoint is under the caller's own control (openai_compatible, or one of the
    ollama/lmstudio/vllm/llamacpp local-server providers) AND whose resolved base_url's host
    is recognizably loopback/private (see _is_local_network_host) -- never a public cloud API,
    and never an unverifiable hostname. This is a structural guarantee that data can leave the
    process but never leave the local machine/network, checked once here before any client is
    constructed, the same fail-fast pattern "strict" uses.
    """

    try:
        selected = provider if isinstance(provider, LLMProvider) else LLMProvider(provider.lower())
    except ValueError as exc:
        choices = ", ".join(p.value for p in LLMProvider)
        raise ProviderError(f"Unknown provider '{provider}'. Choose one of: {choices}.") from exc

    if network_policy not in ("unrestricted", "local", "strict"):
        raise ProviderError("network_policy must be 'unrestricted', 'local', or 'strict'.")
    from buffdata.security.policy import current_context, check_network_url, remember_secret
    execution = current_context()
    if execution:
        execution.network = network_policy
    remember_secret(api_key)
    if network_policy == "strict":
        return NetworkForbiddenClient(would_be_provider=selected)

    resolved_base_url = base_url
    if selected in _LOCAL_PROVIDER_DEFAULTS:
        preset_url, base_url_env, _api_key_env = _LOCAL_PROVIDER_DEFAULTS[selected]
        resolved_base_url = base_url or os.getenv(base_url_env) or preset_url
    elif selected == LLMProvider.OPENAI_COMPATIBLE:
        resolved_base_url = base_url or os.getenv("OPENAI_COMPATIBLE_BASE_URL")

    if network_policy == "local":
        if selected not in _BASE_URL_ROUTED_PROVIDERS:
            raise ProviderError(
                f"network_policy='local' requires a provider whose endpoint you control "
                f"(openai_compatible, ollama, lmstudio, vllm, or llamacpp), not '{selected.value}' "
                "-- that provider always calls a public cloud API regardless of this setting."
            )
        if not resolved_base_url:
            raise ProviderError(
                "network_policy='local' requires a base URL to validate -- pass --base-url or "
                "set the provider's *_BASE_URL environment variable."
            )
        from urllib.parse import urlparse

        host = urlparse(resolved_base_url).hostname
        if not host or not _is_local_network_host(host):
            raise ProviderError(
                f"network_policy='local' blocked base_url '{resolved_base_url}': its host "
                f"({host!r}) isn't recognizably a loopback or private-network address "
                "(\"localhost\", a *.local hostname, or a literal 127.x/10.x/172.16-31.x/"
                "192.168.x IP). Point --base-url at your local server's IP or hostname "
                "directly -- this check never performs DNS resolution to decide."
            )

    target_model = model or os.getenv("BUFFDATA_DEFAULT_MODEL")
    if execution:
        endpoints = {
            LLMProvider.GEMINI: "https://generativelanguage.googleapis.com",
            LLMProvider.OPENAI: os.getenv("OPENAI_BASE_URL", "https://api.openai.com"),
            LLMProvider.ANTHROPIC: os.getenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com"),
            LLMProvider.AZURE_OPENAI: azure_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT"),
            LLMProvider.BEDROCK_ANTHROPIC: "https://bedrock-runtime." + (aws_region or os.getenv("AWS_REGION", "us-east-1")) + ".amazonaws.com",
        }
        endpoint = resolved_base_url or endpoints.get(selected)
        if not endpoint:
            raise ProviderError("Provider endpoint must be configured before execution")
        check_network_url(endpoint)
    if not target_model and selected not in _NO_DEFAULT_MODEL_PROVIDERS:
        target_model = DEFAULT_MODELS[selected]

    if selected == LLMProvider.GEMINI:
        return GeminiClient(api_key=api_key, default_model=target_model, allow_mock=allow_mock)
    if selected == LLMProvider.OPENAI:
        return OpenAIClient(api_key=api_key, default_model=target_model, allow_mock=allow_mock)
    if selected == LLMProvider.ANTHROPIC:
        return AnthropicClient(api_key=api_key, default_model=target_model, allow_mock=allow_mock)
    if selected == LLMProvider.AZURE_OPENAI:
        return AzureOpenAIClient(
            api_key=api_key,
            default_model=target_model,
            azure_endpoint=azure_endpoint,
            api_version=azure_api_version,
            allow_mock=allow_mock,
        )
    if selected == LLMProvider.BEDROCK_ANTHROPIC:
        return BedrockAnthropicClient(default_model=target_model, aws_region=aws_region, allow_mock=allow_mock)
    # openai_compatible and the four local-server providers all share this one client class --
    # the local names differ only in their preset default base_url/env-var names.
    return OpenAICompatibleClient(
        api_key=api_key, default_model=target_model, base_url=resolved_base_url, allow_mock=allow_mock,
        provider_label=selected,
    )
