"""Accuracy-gated dataset generation.

Iteratively augments a labeled classification dataset and stops only once a
lightweight classifier trained on the result clears a minimum *relative*
accuracy gain over the same classifier trained on the original data -- or a
bounded iteration budget runs out, in which case the best candidate found is
returned with an honest FAIL report rather than a fabricated pass.

Each failed iteration diagnoses which classes the current candidate's proxy
classifier still confuses (buffdata.evaluation.accuracy_gate.diagnose_candidate_errors)
and feeds those weak classes, misclassified examples, and the current class
balance back into the next generation prompt, so later rounds target the
model's actual error modes instead of producing undirected paraphrases.
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, List, Optional, Sequence

from pydantic import BaseModel, Field

from buffdata.engine.client import LLMClient
from buffdata.engine.limiter import AsyncRateLimiter
from buffdata.evaluation.accuracy_gate import diagnose_candidate_errors, evaluate_accuracy_gain
from buffdata.models.schemas import DatasetItem
from buffdata.optimizers.augmenter import DataAugmenter

GATED_GENERATOR_SYSTEM_PROMPT = """You are an expert AI dataset synthesizer working inside an accuracy-gated \
generation loop for supervised classification data. A lightweight classifier is retrained after every batch \
you produce, and your batches are judged by whether they measurably raise its held-out accuracy versus a fixed \
baseline. Generic paraphrases that don't change what the classifier learns are a failure, not a success.

Rules:
1. Preserve the original semantic label and every factual claim, entity, number, date, and place in each \
   source record -- you are creating harder or more diverse examples of an existing label, never a new fact.
2. Prioritize the WEAK CLASSES and HARD EXAMPLES given to you below, when present. These are the classes the \
   current classifier confuses most; your new records for these classes must be more discriminative and less \
   ambiguous than the average existing example, not just different in wording.
3. When given misclassified exemplars, study what made them ambiguous (overlapping vocabulary with another \
   class, missing distinguishing detail, unusual phrasing) and write new records for the correct label that \
   close exactly that gap -- use clearer, more class-typical signal words and structure, without stating the \
   label itself in the text.
4. Respect the target class balance given below, when present. Do not add records to classes that are already \
   over-represented relative to their target share.
5. Vary sentence structure, length, and register across records for the same class -- a classifier must not \
   be able to shortcut on surface form.
6. Output exactly N variations per requested source ID. Never copy a source record verbatim, and never \
   invent a label that isn't in the known class list.
"""


def build_gated_generation_prompt(
    weak_labels: Sequence[str],
    hard_exemplars: Dict[str, List[str]],
    class_distribution: Dict[str, float],
    target_distribution: Dict[str, float],
) -> str:
    """Build the dynamic prefix prepended to DataAugmenter's per-chunk prompt.

    Returns an empty string when there is nothing to steer toward yet (first
    iteration, before any diagnosis has run).
    """
    if not weak_labels:
        return ""

    lines = [f"WEAK CLASSES (focus your effort here): {', '.join(weak_labels)}", ""]

    if target_distribution:
        lines.append("CURRENT vs TARGET class share (add more where current < target, less where current > target):")
        for label, target in target_distribution.items():
            current = class_distribution.get(label, 0.0)
            lines.append(f"  {label}: current {current:.1%} -> target {target:.1%}")
        lines.append("")

    if hard_exemplars:
        lines.append(
            "MISCLASSIFIED EXAMPLES the current classifier got wrong (study what made them ambiguous, "
            "then write clearer examples for the correct label -- do not reuse these texts):"
        )
        for label in weak_labels:
            examples = hard_exemplars.get(label) or []
            if not examples:
                continue
            lines.append(f"  [{label}]")
            for example in examples:
                lines.append(f"    - {example[:300]}")
        lines.append("")

    return "\n".join(lines) + "\n"


class IterationMetric(BaseModel):
    iteration: int
    pool_size: int
    accuracy: float
    relative_gain: float
    weak_labels: List[str] = Field(default_factory=list)
    accepted: bool


class AccuracyGateReport(BaseModel):
    baseline_accuracy: float
    final_accuracy: float
    relative_gain: float
    target_relative_gain: float
    iterations_used: int
    passed: bool
    per_iteration: List[IterationMetric] = Field(default_factory=list)


def _class_distribution(items: Sequence[DatasetItem]) -> Dict[str, float]:
    counts = Counter(str(item.labels) for item in items if item.labels is not None)
    total = sum(counts.values()) or 1
    return {label: count / total for label, count in counts.items()}


class AccuracyGatedGenerator:
    """Orchestrates DataAugmenter + the PyTorch accuracy gate into a closed loop."""

    def __init__(
        self,
        client: LLMClient,
        limiter: Optional[AsyncRateLimiter] = None,
        model: Optional[str] = None,
    ):
        self.client = client
        self.limiter = limiter or AsyncRateLimiter()
        self.model = model or client.default_model
        self.augmenter = DataAugmenter(
            client=client,
            limiter=self.limiter,
            model=self.model,
            system_prompt=GATED_GENERATOR_SYSTEM_PROMPT,
        )

    async def generate(
        self,
        items: List[DatasetItem],
        validation_items: List[DatasetItem],
        *,
        min_relative_gain: float = 0.10,
        max_iterations: int = 5,
        multiplier: int = 1,
        chunk_size: int = 20,
        min_label_confidence: float = 0.55,
        accuracy_seeds: Sequence[int] = (17, 29, 43),
        accuracy_epochs: int = 6,
        weak_class_count: int = 2,
        hard_exemplars_per_class: int = 5,
    ) -> tuple[List[DatasetItem], AccuracyGateReport]:
        if max_iterations < 1:
            raise ValueError("max_iterations must be at least 1")

        target_distribution = _class_distribution(items)
        pool = list(items)
        best_pool = pool
        best_accuracy: Optional[float] = None
        baseline_accuracy: Optional[float] = None
        per_iteration: List[IterationMetric] = []
        weak_labels: List[str] = []
        hard_exemplars: Dict[str, List[str]] = {}

        for iteration in range(1, max_iterations + 1):
            prompt_prefix = build_gated_generation_prompt(
                weak_labels, hard_exemplars, _class_distribution(pool), target_distribution,
            )
            candidate = await self.augmenter.augment_batch_async(
                pool,
                multiplier=multiplier,
                chunk_size=chunk_size,
                min_label_confidence=min_label_confidence,
                extra_prompt=prompt_prefix,
            )

            gate = evaluate_accuracy_gain(
                items,
                candidate,
                validation_items,
                seeds=accuracy_seeds,
                epochs=accuracy_epochs,
                minimum_relative_gain=min_relative_gain,
            )
            if baseline_accuracy is None:
                baseline_accuracy = gate["original_accuracy_mean"]
            relative_gain = (
                gate["accuracy_gain"] / gate["original_accuracy_mean"]
                if gate["original_accuracy_mean"] > 0
                else 0.0
            )
            per_iteration.append(IterationMetric(
                iteration=iteration,
                pool_size=len(candidate),
                accuracy=gate["candidate_accuracy_mean"],
                relative_gain=relative_gain,
                weak_labels=list(weak_labels),
                accepted=gate["accepted"],
            ))
            pool = candidate
            if best_accuracy is None or gate["candidate_accuracy_mean"] > best_accuracy:
                best_accuracy = gate["candidate_accuracy_mean"]
                best_pool = pool

            if gate["accepted"]:
                return pool, AccuracyGateReport(
                    baseline_accuracy=baseline_accuracy,
                    final_accuracy=gate["candidate_accuracy_mean"],
                    relative_gain=relative_gain,
                    target_relative_gain=min_relative_gain,
                    iterations_used=iteration,
                    passed=True,
                    per_iteration=per_iteration,
                )

            if iteration < max_iterations:
                diagnostics = diagnose_candidate_errors(
                    pool,
                    validation_items,
                    seed=accuracy_seeds[0],
                    epochs=accuracy_epochs,
                    max_examples_per_label=hard_exemplars_per_class,
                )
                ranked = sorted(diagnostics["per_label_accuracy"].items(), key=lambda pair: pair[1])
                weak_labels = [label for label, _ in ranked[:weak_class_count]]
                hard_exemplars = {
                    label: diagnostics["misclassified_examples"].get(label, [])
                    for label in weak_labels
                }

        final_metric = per_iteration[-1]
        return best_pool, AccuracyGateReport(
            baseline_accuracy=baseline_accuracy or 0.0,
            final_accuracy=best_accuracy if best_accuracy is not None else (baseline_accuracy or 0.0),
            relative_gain=final_metric.relative_gain,
            target_relative_gain=min_relative_gain,
            iterations_used=max_iterations,
            passed=False,
            per_iteration=per_iteration,
        )
