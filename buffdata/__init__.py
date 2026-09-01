"""
buffdata: AI Training Data Optimizer powered by Google Gemini API.
"""

__version__ = "0.1.0"

from buffdata.models.schemas import (
    DatasetItem,
    DatasetFormat,
    ChatMessage,
    QualityScore,
    RefinementResult,
    EvolutionResult,
    PreferenceResult,
    PipelineConfig,
)
from buffdata.engine.client import GeminiClient
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
    "GeminiClient",
    "QualityScorer",
    "FastRuleFilter",
    "DataRefiner",
    "DataEvolver",
    "PreferenceBuilder",
    "Deduplicator",
    "ReportGenerator",
]
