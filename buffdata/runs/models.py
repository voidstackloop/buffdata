from enum import Enum
from typing import Any
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from buffdata.models.schemas import PipelineConfig


class RunStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    INTERRUPTED = "interrupted"


class RunSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    project_id: str = Field("local", pattern=r"^[a-zA-Z0-9_-]{1,64}$")
    dataset_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    configuration: dict[str, Any] = Field(default_factory=dict)
    output_format: str = "jsonl"
    validation_dataset_id: str | None = Field(None, pattern=r"^[a-f0-9]{32}$")
    require_positive_gain: bool = False
    accuracy_seeds: list[int] = Field(default_factory=lambda: [17, 29, 43], min_length=1, max_length=20)
    accuracy_epochs: int = Field(6, ge=1, le=100)
    accuracy_min_gain: float = Field(0, ge=0, le=1)
    accuracy_max_train_rows: int = Field(100000, ge=1)

    @field_validator("configuration")
    @classmethod
    def validate_configuration(cls, value):
        from urllib.parse import urlsplit
        endpoint = value.get("base_url")
        if endpoint:
            parsed = urlsplit(endpoint)
            if parsed.username or parsed.password or parsed.query or parsed.fragment:
                raise ValueError("Provider credentials must not be embedded in endpoint URLs")
        unknown = set(value) - PipelineConfig.model_fields.keys()
        if unknown:
            raise ValueError("Unknown pipeline configuration keys: " + ", ".join(sorted(unknown)))
        return PipelineConfig(**value).model_dump(mode="json")

    @field_validator("output_format")
    @classmethod
    def validate_format(cls, value):
        from buffdata.models.formats import _normalize_format, SUPPORTED_FORMATS
        normalized = _normalize_format(value)
        if normalized not in SUPPORTED_FORMATS:
            raise ValueError("Unsupported output format")
        return normalized

    @model_validator(mode="after")
    def validate_gate(self):
        if self.require_positive_gain and not self.validation_dataset_id:
            raise ValueError("A validation dataset is required for the accuracy gate")
        if not self.configuration:
            self.configuration = PipelineConfig().model_dump(mode="json")
        return self


class RunManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: int = 1
    run_id: str
    project_id: str
    parent_run_id: str | None = None
    spec: RunSpec
    implementation: dict[str, Any]
    inputs: dict[str, str]
    artifacts: dict[str, dict[str, Any]]
    metrics: dict[str, Any]
    completed_at: str


class RunConflict(ValueError):
    pass
