from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
import uuid
from pydantic import BaseModel, Field


class DatasetFormat(str, Enum):
    ALPACA = "alpaca"          # instruction, input, output
    CHAT = "chat"              # messages: [{"role": ..., "content": ...}]
    RAW = "raw"                # text
    DPO = "dpo"                # prompt, chosen, rejected
    CUSTOM = "custom"          # arbitrary dictionary


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


class RefinementResult(BaseModel):
    refined_prompt: Optional[str] = Field(None, description="Polished, disambiguated prompt instruction")
    refined_response: str = Field(..., description="Upgraded high-quality response with rich reasoning and clean formatting")
    reasoning_added: bool = Field(False, description="Whether chain-of-thought or reasoning steps were added")
    formatting_fixed: bool = Field(False, description="Whether markdown, code syntax, or structure was fixed")
    artifacts_removed: List[str] = Field(default_factory=list, description="AI boilerplate or unwanted artifacts removed")
    explanation_of_changes: str = Field("", description="Summary explanation of enhancements made")


class EvolutionResult(BaseModel):
    evolved_prompt: str = Field(..., description="More complex, nuanced, or specialized prompt")
    evolved_response: str = Field(..., description="Comprehensive, high-quality answer to the evolved prompt")
    evolution_type: str = Field(..., description="Strategy applied: deepen_reasoning, add_constraints, concretize, in_breadth")
    complexity_score: float = Field(..., ge=1.0, le=10.0, description="Complexity difficulty rating from 1 to 10")


class PreferenceResult(BaseModel):
    prompt: str = Field(..., description="The user prompt / instruction")
    chosen: str = Field(..., description="The superior, polished, correct response")
    rejected: str = Field(..., description="The inferior, flawed, or unrefined response")
    rejection_reason: str = Field(..., description="Specific flaw or issue present in the rejected response")


class DatasetItem(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    format: DatasetFormat = DatasetFormat.ALPACA
    instruction: Optional[str] = None
    input: Optional[str] = None
    output: Optional[str] = None
    messages: Optional[List[ChatMessage]] = None
    text: Optional[str] = None
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
        """Serialize back to dictionary matching its format."""
        result: Dict[str, Any] = {}
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
        else:
            result = dict(self.raw_data)

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

        if "messages" in d_copy and isinstance(d_copy["messages"], list):
            msgs = [ChatMessage(**m) if isinstance(m, dict) else ChatMessage(role=m.role, content=m.content) for m in d_copy["messages"]]
            return cls(id=item_id, format=DatasetFormat.CHAT, messages=msgs, metadata=metadata, quality_score=quality_score, raw_data=d_copy)
        elif "chosen" in d_copy and "rejected" in d_copy:
            return cls(
                id=item_id,
                format=DatasetFormat.DPO,
                prompt=d_copy.get("prompt"),
                chosen=d_copy.get("chosen"),
                rejected=d_copy.get("rejected"),
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
                metadata=metadata,
                quality_score=quality_score,
                raw_data=d_copy
            )
        elif "text" in d_copy:
            return cls(
                id=item_id,
                format=DatasetFormat.RAW,
                text=d_copy.get("text"),
                metadata=metadata,
                quality_score=quality_score,
                raw_data=d_copy
            )
        else:
            return cls(
                id=item_id,
                format=DatasetFormat.CUSTOM,
                metadata=metadata,
                quality_score=quality_score,
                raw_data=d_copy
            )


class PipelineConfig(BaseModel):
    model: str = Field("gemini-3.7-flash", description="Gemini model for reasoning tasks")
    fast_model: str = Field("gemini-3.5-flash-lite", description="Gemini model for high-throughput tasks")
    embedding_model: str = Field("text-embedding-004", description="Gemini embedding model")
    concurrency: int = Field(10, description="Max concurrent async requests")
    max_rpm: int = Field(60, description="Max requests per minute")
    
    # Optimizer step configs
    filter_min_score: float = Field(7.0, description="Minimum quality score threshold for filtering")
    refine_mode: str = Field("all", description="Refine mode: all, response_only, prompt_only")
    evolution_strategies: List[str] = Field(default_factory=lambda: ["deepen_reasoning", "add_constraints", "concretize"])
    evolution_depth: int = Field(1, description="Evolution iterations per item")
    dedup_threshold: float = Field(0.88, description="Cosine similarity threshold for deduplication")
    dedup_method: str = Field("semantic", description="Dedup method: semantic, minhash, exact")
