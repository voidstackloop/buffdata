from enum import Enum
import os
from typing import Any, Dict, List, Literal, Optional, Tuple, Union
import uuid
from pydantic import BaseModel, Field, field_validator, model_validator


class DatasetFormat(str, Enum):
    ALPACA = "alpaca"          # instruction, input, output
    CHAT = "chat"              # messages: [{"role": ..., "content": ...}]
    RAW = "raw"                # text
    DPO = "dpo"                # prompt, chosen, rejected
    CUSTOM = "custom"          # arbitrary dictionary


class ClassificationTask(str, Enum):
    BINARY = "binary"
    MULTI_CLASS = "multi-class"
    MULTI_LABEL = "multi-label"


class ClassificationMode(str, Enum):
    AUTO = "auto"
    OFF = "off"
    BINARY = "binary"
    MULTI_CLASS = "multi-class"
    MULTI_LABEL = "multi-label"


class ChatMessage(BaseModel):
    role: str = Field(..., description="Role of the speaker: user, assistant, system")
    content: str = Field(..., description="Text content of the message")


class QualityScore(BaseModel):
    overall_score: float = Field(..., ge=0.0, le=10.0, description="Overall quality score from 0.0 to 10.0")
    clarity: float = Field(..., ge=0.0, le=10.0, description="Clarity, readability and grammatical precision")
    factual_accuracy: float = Field(..., ge=0.0, le=10.0, description="Factual correctness, logical consistency, lack of hallucinations")
    reasoning_depth: float = Field(..., ge=0.0, le=10.0, description="Depth of reasoning, step-by-step thoroughness, analytical rigor")
    instruction_following: float = Field(..., ge=0.0, le=10.0, description="Degree to which constraints and instructions are strictly fulfilled")
    is_safe: bool = Field(True, description="Whether the content is safe and free of toxic/harmful material")
    issues: List[str] = Field(default_factory=list, description="Specific defect tags or issues discovered")
    recommendations: str = Field("", description="Actionable summary of how this sample can be improved")


class QualityAuditEntry(BaseModel):
    item_id: str = Field(..., description="Exact BuffData record identifier from the audit prompt")
    score: QualityScore


class QualityAuditBatch(BaseModel):
    entries: List[QualityAuditEntry] = Field(default_factory=list)


class RefinementResult(BaseModel):
    refined_prompt: Optional[str] = Field(None, description="Polished, disambiguated prompt instruction")
    refined_response: str = Field(..., description="Upgraded high-quality response with rich reasoning and clean formatting")
    reasoning_added: bool = Field(False, description="Whether chain-of-thought or reasoning steps were added")
    formatting_fixed: bool = Field(False, description="Whether markdown, code syntax, or structure was fixed")
    artifacts_removed: List[str] = Field(default_factory=list, description="AI boilerplate or unwanted artifacts removed")
    explanation_of_changes: str = Field("", description="Summary explanation of enhancements made")


class RefinementEntry(BaseModel):
    item_id: str = Field(..., description="Exact BuffData record identifier from the refinement prompt")
    result: RefinementResult


class BatchRefinementResponse(BaseModel):
    entries: List[RefinementEntry] = Field(default_factory=list)


class EvolutionResult(BaseModel):
    evolved_prompt: str = Field(..., description="More complex, nuanced, or specialized prompt")
    evolved_response: str = Field(..., description="Comprehensive, high-quality answer to the evolved prompt")
    evolution_type: str = Field(..., description="Strategy applied: deepen_reasoning, add_constraints, concretize, in_breadth")
    complexity_score: float = Field(..., ge=1.0, le=10.0, description="Complexity difficulty rating from 1 to 10")


class EvolutionEntry(BaseModel):
    item_id: str = Field(..., description="Exact BuffData record identifier from the evolution prompt")
    result: EvolutionResult


class BatchEvolutionResponse(BaseModel):
    entries: List[EvolutionEntry] = Field(default_factory=list)


class PreferenceResult(BaseModel):
    prompt: str = Field(..., description="The user prompt / instruction")
    chosen: str = Field(..., description="The superior, polished, correct response")
    rejected: str = Field(..., description="The inferior, flawed, or unrefined response")
    rejection_reason: str = Field(..., description="Specific flaw or issue present in the rejected response")


class PreferenceEntry(BaseModel):
    item_id: str = Field(..., description="Exact BuffData record identifier from the preference prompt")
    result: PreferenceResult


class BatchPreferenceResponse(BaseModel):
    entries: List[PreferenceEntry] = Field(default_factory=list)


class DatasetItem(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    format: DatasetFormat = DatasetFormat.ALPACA
    instruction: Optional[str] = None
    input: Optional[str] = None
    output: Optional[str] = None
    messages: Optional[List[ChatMessage]] = None
    text: Optional[str] = None
    labels: Optional[Union[str, int, List[str], List[int]]] = None
    prompt: Optional[str] = None
    chosen: Optional[str] = None
    rejected: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    quality_score: Optional[QualityScore] = None
    raw_data: Dict[str, Any] = Field(default_factory=dict)

    def get_prompt_and_response(self) -> Tuple[str, str]:
        """Extract primary prompt and response regardless of the underlying format."""
        if self.format == DatasetFormat.ALPACA:
            p = self.instruction or ""
            if self.input:
                p = f"{p}\n\nContext / Input:\n{self.input}" if p else self.input
            return p, self.output or ""
        elif self.format == DatasetFormat.CHAT:
            if not self.messages:
                return "", ""
            user_parts = [m.content for m in self.messages if m.role in ("user", "human")]
            asst_parts = [m.content for m in self.messages if m.role in ("assistant", "gpt", "model")]
            p = "\n\n".join(user_parts)
            r = "\n\n".join(asst_parts)
            return p, r
        elif self.format == DatasetFormat.DPO:
            return self.prompt or "", self.chosen or ""
        elif self.format == DatasetFormat.RAW:
            return "", self.text or ""
        else:
            p = str(self.raw_data.get("prompt") or self.raw_data.get("instruction") or self.raw_data.get("input") or "")
            r = str(self.raw_data.get("response") or self.raw_data.get("output") or self.raw_data.get("text") or "")
            return p, r

    def get_classification_text(self) -> str:
        """Return the input-side text that should be classified."""
        if self.format == DatasetFormat.RAW:
            return self.text or ""
        if self.format == DatasetFormat.CHAT:
            return "\n\n".join(
                m.content for m in (self.messages or []) if m.role in ("user", "human")
            )
        if self.format == DatasetFormat.ALPACA:
            return "\n\n".join(part for part in (self.instruction, self.input) if part)
        if self.format == DatasetFormat.DPO:
            return self.prompt or ""
        for key in ("text", "content", "query", "prompt", "instruction", "input"):
            value = self.raw_data.get(key)
            if value is not None and str(value).strip():
                return str(value)
        return ""

    def update_content(self, new_prompt: Optional[str] = None, new_response: Optional[str] = None):
        """Update item prompt/response preserving format conventions."""
        if self.format == DatasetFormat.ALPACA:
            if new_prompt is not None:
                self.instruction = new_prompt
                self.input = None
            if new_response is not None:
                self.output = new_response
        elif self.format == DatasetFormat.CHAT:
            if not self.messages:
                self.messages = []
            if new_prompt is not None:
                user_found = False
                for m in self.messages:
                    if m.role in ("user", "human"):
                        m.content = new_prompt
                        user_found = True
                        break
                if not user_found:
                    self.messages.insert(0, ChatMessage(role="user", content=new_prompt))
            if new_response is not None:
                asst_found = False
                for m in reversed(self.messages):
                    if m.role in ("assistant", "gpt", "model"):
                        m.content = new_response
                        asst_found = True
                        break
                if not asst_found:
                    self.messages.append(ChatMessage(role="assistant", content=new_response))
        elif self.format == DatasetFormat.DPO:
            if new_prompt is not None:
                self.prompt = new_prompt
            if new_response is not None:
                self.chosen = new_response
        elif self.format == DatasetFormat.RAW:
            if new_response is not None:
                self.text = new_response
            elif new_prompt is not None:
                self.text = new_prompt
        else:
            if new_prompt is not None:
                self.raw_data["prompt"] = new_prompt
            if new_response is not None:
                self.raw_data["response"] = new_response

    def to_dict(self) -> Dict[str, Any]:
        """Serialize without dropping unknown input columns."""
        result: Dict[str, Any] = dict(self.raw_data)
        result["id"] = self.id
        if self.format == DatasetFormat.ALPACA:
            result["instruction"] = self.instruction or ""
            result["input"] = self.input or ""
            result["output"] = self.output or ""
        elif self.format == DatasetFormat.CHAT:
            result["messages"] = [m.model_dump() for m in (self.messages or [])]
        elif self.format == DatasetFormat.DPO:
            result["prompt"] = self.prompt or ""
            result["chosen"] = self.chosen or ""
            result["rejected"] = self.rejected or ""
        elif self.format == DatasetFormat.RAW:
            result["text"] = self.text or ""
        if self.labels is not None:
            result["labels"] = self.labels

        # Attach metadata and scores if present
        if self.metadata:
            result["_buffdata_metadata"] = self.metadata
        if self.quality_score:
            result["_buffdata_quality"] = self.quality_score.model_dump()
        return result

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DatasetItem":
        """Factory to parse dictionary into appropriate DatasetItem."""
        d_copy = dict(d)
        metadata = d_copy.pop("_buffdata_metadata", {})
        quality_data = d_copy.pop("_buffdata_quality", None)
        quality_score = QualityScore(**quality_data) if quality_data else None
        item_id = str(d_copy.pop("id", None) or d_copy.pop("_buffdata_id", None) or uuid.uuid4().hex[:8])
        labels = d_copy.get("labels", d_copy.get("label", d_copy.get("category", d_copy.get("target"))))

        if "messages" in d_copy and isinstance(d_copy["messages"], list):
            msgs = [ChatMessage(**m) if isinstance(m, dict) else ChatMessage(role=m.role, content=m.content) for m in d_copy["messages"]]
            return cls(id=item_id, format=DatasetFormat.CHAT, messages=msgs, labels=labels, metadata=metadata, quality_score=quality_score, raw_data=d_copy)
        elif "chosen" in d_copy and "rejected" in d_copy:
            return cls(
                id=item_id,
                format=DatasetFormat.DPO,
                prompt=d_copy.get("prompt"),
                chosen=d_copy.get("chosen"),
                rejected=d_copy.get("rejected"),
                labels=labels,
                metadata=metadata,
                quality_score=quality_score,
                raw_data=d_copy
            )
        elif "instruction" in d_copy or "output" in d_copy:
            return cls(
                id=item_id,
                format=DatasetFormat.ALPACA,
                instruction=d_copy.get("instruction"),
                input=d_copy.get("input"),
                output=d_copy.get("output"),
                labels=labels,
                metadata=metadata,
                quality_score=quality_score,
                raw_data=d_copy
            )
        elif "text" in d_copy:
            return cls(
                id=item_id,
                format=DatasetFormat.RAW,
                text=d_copy.get("text"),
                labels=labels,
                metadata=metadata,
                quality_score=quality_score,
                raw_data=d_copy
            )
        else:
            return cls(
                id=item_id,
                format=DatasetFormat.CUSTOM,
                labels=labels,
                metadata=metadata,
                quality_score=quality_score,
                raw_data=d_copy
            )


class DatasetProfile(BaseModel):
    format: DatasetFormat
    row_count: int = Field(ge=0)
    text_fields: List[str] = Field(default_factory=list)
    existing_label_field: Optional[str] = None
    classification_applicable: bool = False
    task_type: Optional[ClassificationTask] = None
    classes: List[str] = Field(default_factory=list)
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    reasoning: str = ""
    sampled_ids: List[str] = Field(default_factory=list)


class OptimizationRunResult(BaseModel):
    accepted: List[DatasetItem] = Field(default_factory=list)
    rejected: List[DatasetItem] = Field(default_factory=list)
    profile: DatasetProfile
    metrics: Dict[str, Any] = Field(default_factory=dict)
    output_path: Optional[str] = None
    rejected_path: Optional[str] = None
    report_path: Optional[str] = None


class PipelineConfig(BaseModel):
    provider: str = Field(
        default_factory=lambda: os.getenv("BUFFDATA_PROVIDER", "gemini"),
        description=(
            "gemini, openai, anthropic, a private endpoint (azure_openai, bedrock_anthropic, "
            "openai_compatible), or a local LLM server (ollama, lmstudio, vllm, llamacpp)"
        ),
    )
    model: Optional[str] = Field(None, description="Provider model; uses a balanced default when omitted")
    base_url: Optional[str] = Field(
        None,
        description=(
            "Endpoint URL for openai_compatible or a local-server provider (ollama, lmstudio, "
            "vllm, llamacpp) -- overrides that provider's default local port and its "
            "*_BASE_URL environment variable. Point this at another machine on the network, "
            "e.g. http://192.168.1.50:11434/v1, to use a team-shared LLM server instead of one "
            "running on this machine."
        ),
    )
    fast_model: str = Field("gemini-3.5-flash-lite", description="Gemini model for high-throughput tasks")
    embedding_provider: str = Field("local", description="Local embeddings are provider-neutral")
    embedding_model: str = Field("all-MiniLM-L6-v2", description="Local embedding model")
    concurrency: int = Field(10, description="Max concurrent async requests")
    max_rpm: int = Field(60, description="Max requests per minute")
    quality_mode: Literal["llm", "sampled", "off"] = Field(
        "llm",
        description="llm scores every row; sampled audits a representative subset; off skips LLM quality scoring",
    )
    quality_sample_size: int = Field(
        50,
        ge=1,
        le=1000,
        description="Representative rows scored when quality_mode is sampled",
    )
    quality_audit_batch_size: int = Field(
        20,
        ge=1,
        le=100,
        description="Representative records grouped into one structured LLM audit request",
    )

    # Optimizer step configs
    filter_min_score: float = Field(7.0, description="Minimum quality score threshold for filtering")
    refine_mode: str = Field("all", description="Refine mode: all, response_only, prompt_only")
    evolution_strategies: List[str] = Field(default_factory=lambda: ["deepen_reasoning", "add_constraints", "concretize"])
    evolution_depth: int = Field(1, description="Evolution iterations per item")
    dedup_threshold: float = Field(0.88, description="Cosine similarity threshold for deduplication")
    dedup_method: str = Field(
        "auto",
        description="Dedup method: auto uses exact for labeled classification and minhash otherwise",
    )
    classification: ClassificationMode = ClassificationMode.AUTO
    classes: List[str] = Field(default_factory=list)
    classification_confidence: float = Field(0.75, ge=0.0, le=1.0)
    classification_sample_size: int = Field(100, ge=1, le=1000)
    scrub_pii: bool = True
    classification_pii_mode: Literal["identifiers", "all", "off"] = Field(
        "identifiers",
        description="For labeled classification: redact direct identifiers, all Presidio entities, or nothing",
    )
    accuracy_contract: Literal["balanced", "strict"] = Field(
        "balanced",
        description="Strict preserves every valid labeled row, its classification text, and its labels",
    )
    network_policy: Literal["unrestricted", "local", "strict"] = Field(
        "unrestricted",
        description=(
            "Strict makes the LLM client a structural no-op (NetworkForbiddenClient) instead of "
            "a real provider -- any stage that would need a remote call fails immediately and "
            "clearly instead of the run silently reaching a network, so a reviewer can verify "
            "'this run cannot call out' without auditing every stage's internal logic. Local "
            "allows calls, but only to a provider whose endpoint is under your own control "
            "(openai_compatible, ollama, lmstudio, vllm, llamacpp) and only when its resolved "
            "base_url's host is recognizably loopback/private -- a structural guarantee data "
            "can leave the process but never leave the local machine/network."
        ),
    )
    refine_below_score: float = Field(8.0, ge=0.0, le=10.0)
    report_html: bool = False
    observability: bool = Field(
        False,
        description=(
            "Emit an OpenTelemetry span per stage and Prometheus metrics (stage duration, "
            "accepted/rejected records, token usage) for this run. No-op when neither "
            "library is installed. Where they're actually exported (Datadog, a Prometheus "
            "scrape endpoint, ...) is standard OTel/Prometheus configuration, not something "
            "this flag controls."
        ),
    )

    @field_validator("provider")
    @classmethod
    def validate_provider(cls, value: str) -> str:
        # Local import: buffdata.engine.client sits behind buffdata.engine's package
        # __init__, which itself imports modules that depend on this schemas module --
        # importing LLMProvider at module scope here would be a circular import.
        from buffdata.engine.client import LLMProvider

        normalized = value.lower()
        try:
            LLMProvider(normalized)
        except ValueError:
            choices = ", ".join(p.value for p in LLMProvider)
            raise ValueError(f"provider must be one of: {choices}")
        return normalized

    @model_validator(mode="after")
    def validate_classification_overrides(self):
        if self.classification == ClassificationMode.BINARY and self.classes and len(self.classes) != 2:
            raise ValueError("binary classification requires exactly two classes")
        if self.classification == ClassificationMode.MULTI_CLASS and self.classes and len(self.classes) < 3:
            raise ValueError("multi-class classification requires at least three classes")
        if self.classification == ClassificationMode.MULTI_LABEL and self.classes and len(self.classes) < 2:
            raise ValueError("multi-label classification requires at least two classes")
        return self

    @model_validator(mode="after")
    def validate_network_policy(self):
        # Fast, config-only pre-flight: these two settings need a remote call regardless of
        # what the data looks like, so reject them immediately at config-construction time
        # rather than waiting for the run to reach that stage. Profiling/classification calls
        # are data-dependent (skipped automatically for already-labeled data) and can't be
        # ruled out here -- NetworkForbiddenClient is the guaranteed backstop for those.
        if self.network_policy == "strict":
            if self.quality_mode != "off":
                raise ValueError(
                    "network_policy='strict' requires quality_mode='off' -- 'llm' and 'sampled' "
                    "always call the configured provider."
                )
            if self.dedup_method == "semantic":
                raise ValueError(
                    "network_policy='strict' requires dedup_method != 'semantic' -- provider "
                    "embeddings always call the configured provider. Use 'exact', 'minhash', or "
                    "'semantic-local' (local sentence-transformers embeddings) instead."
                )
        if self.network_policy == "local":
            # Fast, provider-only pre-flight: which endpoint the *resolved* base_url points at
            # (a preset default, its env var, or this field) isn't known until create_llm_client
            # resolves it, so that host check is the guaranteed backstop -- this only rules out
            # providers that are never eligible regardless of base_url, the same split "strict"
            # uses between this validator and NetworkForbiddenClient.
            from buffdata.engine.client import _BASE_URL_ROUTED_PROVIDERS, LLMProvider

            if LLMProvider(self.provider) not in _BASE_URL_ROUTED_PROVIDERS:
                raise ValueError(
                    "network_policy='local' requires a provider whose endpoint you control "
                    "(openai_compatible, ollama, lmstudio, vllm, llamacpp) -- "
                    f"'{self.provider}' always calls a public cloud API."
                )
        return self

class AugmentationResult(BaseModel):
    id: str = Field(description="The exact ID of the source item being augmented")
    variations: List[str] = Field(description="N synthetic text variations that perfectly preserve the original semantic label and meaning")

class BatchAugmentationResponse(BaseModel):
    results: List[AugmentationResult]
