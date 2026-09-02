"""buffdata: provider-neutral AI training-data optimizer."""

__version__ = "0.3.0"

from buffdata.models.schemas import (
    DatasetItem,
    DatasetFormat,
    ChatMessage,
    QualityScore,
    RefinementResult,
    EvolutionResult,
    PreferenceResult,
    PipelineConfig,
    ClassificationMode,
    ClassificationTask,
    DatasetProfile,
    OptimizationRunResult,
)
from buffdata.engine.client import (
    AnthropicClient,
    GeminiClient,
    LLMClient,
    LLMProvider,
    OpenAIClient,
    create_llm_client,
)
from buffdata.engine.pipeline import OptimizationPipeline
from buffdata.optimizers.scorer import QualityScorer, FastRuleFilter
from buffdata.optimizers.refiner import DataRefiner
from buffdata.optimizers.evolver import DataEvolver
from buffdata.optimizers.preference import PreferenceBuilder
from buffdata.optimizers.dedup import Deduplicator
from buffdata.report.generator import ReportGenerator

__all__ = [
    "DatasetItem",
    "DatasetFormat",
    "ChatMessage",
    "QualityScore",
    "RefinementResult",
    "EvolutionResult",
    "PreferenceResult",
    "PipelineConfig",
    "ClassificationMode",
    "ClassificationTask",
    "DatasetProfile",
    "OptimizationRunResult",
    "GeminiClient",
    "OpenAIClient",
    "AnthropicClient",
    "LLMClient",
    "LLMProvider",
    "create_llm_client",
    "OptimizationPipeline",
    "QualityScorer",
    "FastRuleFilter",
    "DataRefiner",
    "DataEvolver",
    "PreferenceBuilder",
    "Deduplicator",
    "ReportGenerator",
]
