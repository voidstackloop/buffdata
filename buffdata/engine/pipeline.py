"""Adaptive, auditable BuffData optimization pipeline."""

from __future__ import annotations

import hashlib
import json
import orjson
import os
import asyncio
from pathlib import Path
from typing import Any, List, Optional

from buffdata.engine.client import LLMClient, create_llm_client
from buffdata.engine.limiter import AsyncRateLimiter
from buffdata.engine.profiler import DatasetProfiler, representative_sample
from buffdata.engine.validator import DatasetValidator
from buffdata.models.formats import read_dataset, write_dataset_atomic
from buffdata.observability import ObservabilityRegistry, disabled_registry
from buffdata.security.permissions import restrict_to_owner
from buffdata.models.schemas import (
    ClassificationMode,
    DatasetItem,
    OptimizationRunResult,
    PipelineConfig,
)
from buffdata.optimizers.classifier import ClassificationSchema, DatasetClassifier
from buffdata.optimizers.dedup import Deduplicator
from buffdata.optimizers.refiner import DataRefiner
from buffdata.optimizers.scorer import QualityScorer
from buffdata.optimizers.scrubber import PIIScrubber
from buffdata.report.generator import ReportGenerator


STAGES = ("validate", "pii", "profile", "dedup", "score_refine", "filter", "classify")


class OptimizationPipeline:
    def __init__(self, config: PipelineConfig, client: Optional[LLMClient] = None):
        self.config = config
        self.client = client or create_llm_client(
            provider=config.provider,
            model=config.model,
            allow_mock=False,
            base_url=config.base_url,
            network_policy=config.network_policy,
        )
        self.model = config.model or self.client.default_model
        self.limiter = AsyncRateLimiter(
            max_rpm=config.max_rpm,
            concurrency=config.concurrency,
        )
        self._observability = (
            ObservabilityRegistry(enabled=True) if config.observability else disabled_registry()
        )

    @staticmethod
    def artifact_paths(output_path: Path) -> tuple[Path, Path, Path]:
        rejected = output_path.with_name(f"{output_path.stem}.rejected.jsonl")
        report = output_path.with_name(f"{output_path.stem}.report.json")
        checkpoint = output_path.with_name(f".{output_path.stem}.checkpoint.json")
        return rejected, report, checkpoint

    def _fingerprint(self, input_path: Path) -> str:
        digest = hashlib.sha256()
        if input_path.is_dir():
            for child in sorted(path for path in input_path.rglob("*") if path.is_file()):
                digest.update(child.relative_to(input_path).as_posix().encode("utf-8"))
                with child.open("rb") as handle:
                    for block in iter(lambda: handle.read(1024 * 1024), b""):
                        digest.update(block)
        else:
            with input_path.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
        digest.update(self.config.model_dump_json(exclude_none=False).encode("utf-8"))
        return digest.hexdigest()

    @staticmethod
    def _reject(item: DatasetItem, stage: str, reason: str) -> None:
        item.metadata["rejection"] = {"stage": stage, "reason": reason}

    @staticmethod
    def _atomic_json(path: Path, data: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.tmp")
        temporary.write_bytes(orjson.dumps(data, option=orjson.OPT_INDENT_2))
        # Owner-only permissions before the rename, not after: os.replace preserves the
        # source inode's mode on POSIX, so this is the only chmod needed, and it means
        # the file is never briefly world-readable at its final path.
        restrict_to_owner(temporary)
        os.replace(temporary, path)

    async def _save_checkpoint(
        self,
        path: Optional[Path],
        fingerprint: Optional[str],
        completed: List[str],
        accepted: List[DatasetItem],
        rejected: List[DatasetItem],
        profile: Any,
        metrics: dict[str, Any],
    ) -> None:
        if path is None or fingerprint is None:
            return
        self._atomic_json(
            path,
            {
                "fingerprint": fingerprint,
                "completed_stages": completed,
                "accepted": [item.to_dict() for item in accepted],
                "rejected": [item.to_dict() for item in rejected],
                "profile": profile.model_dump(mode="json") if profile else None,
                "metrics": metrics,
            },
        )

    @staticmethod
    def _load_checkpoint(path: Path, fingerprint: str):
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("fingerprint") != fingerprint:
            raise ValueError(
                f"Checkpoint {path} belongs to a different input or configuration. "
                "Remove it or restore the matching run."
            )
        return data

    async def run_file(self, input_path: Path | str, output_path: Path | str) -> OptimizationRunResult:
        input_file = Path(input_path)
        output_file = Path(output_path)
        rejected_path, report_path, checkpoint_path = self.artifact_paths(output_file)
        fingerprint = self._fingerprint(input_file)
        checkpoint = self._load_checkpoint(checkpoint_path, fingerprint)
        items = read_dataset(input_file)
        return await self.run(
            items,
            output_path=output_file,
            rejected_path=rejected_path,
            report_path=report_path,
            checkpoint_path=checkpoint_path,
            fingerprint=fingerprint,
            resume_state=checkpoint,
        )

    async def run(
        self,
        items: List[DatasetItem],
        *,
        output_path: Optional[Path] = None,
        rejected_path: Optional[Path] = None,
        report_path: Optional[Path] = None,
        checkpoint_path: Optional[Path] = None,
        fingerprint: Optional[str] = None,
        resume_state: Optional[dict[str, Any]] = None,
    ) -> OptimizationRunResult:
        accepted = list(items)
        rejected: List[DatasetItem] = []
        completed: List[str] = []
        profile = None
        metrics: dict[str, Any] = {"input_records": len(items), "stages": {}}

        def strict_labeled(current: List[DatasetItem]) -> bool:
            return (
                self.config.accuracy_contract == "strict"
                and bool(current)
                and all(item.labels is not None for item in current)
            )

        if resume_state:
            accepted = [DatasetItem.from_dict(row) for row in resume_state.get("accepted", [])]
            rejected = [DatasetItem.from_dict(row) for row in resume_state.get("rejected", [])]
            completed = list(resume_state.get("completed_stages", []))
            metrics = resume_state.get("metrics", metrics)
            profile_data = resume_state.get("profile")
            if profile_data:
                from buffdata.models.schemas import DatasetProfile

                profile = DatasetProfile.model_validate(profile_data)

        if "validate" not in completed:
            rejected_before = len(rejected)
            with self._observability.stage_span("validate"):
                valid: List[DatasetItem] = []
                for item in accepted:
                    errors = DatasetValidator.validate_item(item)
                    if errors:
                        self._reject(item, "validate", "; ".join(errors))
                        rejected.append(item)
                    else:
                        valid.append(item)
                accepted = valid
                metrics["stages"]["validate"] = {"accepted": len(accepted), "rejected": len(rejected)}
                self._observability.record_stage_outcome(
                    "validate", accepted=len(accepted), rejected=len(rejected) - rejected_before
                )
            completed.append("validate")
            await self._save_checkpoint(checkpoint_path, fingerprint, completed, accepted, rejected, profile, metrics)

        if "pii" not in completed:
            with self._observability.stage_span("pii"):
                labeled_classification = bool(accepted) and all(item.labels is not None for item in accepted)
                classification_mode = (
                    "off"
                    if strict_labeled(accepted)
                    else self.config.classification_pii_mode if labeled_classification else "all"
                )
                pii_enabled = self.config.scrub_pii and classification_mode != "off"
                if pii_enabled:
                    entities = (
                        list(PIIScrubber.IDENTIFIER_ENTITIES)
                        if labeled_classification and classification_mode == "identifiers"
                        else None
                    )
                    scrubber = PIIScrubber(entities=entities)
                    accepted = await asyncio.to_thread(scrubber.scrub_batch, accepted)
                metrics["stages"]["pii"] = {
                    "enabled": pii_enabled,
                    "policy": classification_mode,
                    "redacted_records": sum(
                        bool(item.metadata.get("pii", {}).get("entity_counts")) for item in accepted
                    ),
                }
                self._observability.record_stage_outcome("pii", accepted=len(accepted))
            completed.append("pii")
            await self._save_checkpoint(checkpoint_path, fingerprint, completed, accepted, rejected, profile, metrics)

        if "profile" not in completed:
            with self._observability.stage_span("profile"):
                profile = await DatasetProfiler(
                    client=self.client,
                    model=self.model,
                    sample_size=self.config.classification_sample_size,
                ).profile(accepted)
                metrics["stages"]["profile"] = {
                    "classification_applicable": profile.classification_applicable,
                    "confidence": profile.confidence,
                    "task_type": profile.task_type.value if profile.task_type else None,
                }
                self._observability.record_stage_outcome("profile", accepted=len(accepted))
            completed.append("profile")
            await self._save_checkpoint(checkpoint_path, fingerprint, completed, accepted, rejected, profile, metrics)

        if "dedup" not in completed:
            with self._observability.stage_span("dedup"):
                deduper = Deduplicator(client=self.client)
                method = "off" if strict_labeled(accepted) else self.config.dedup_method
                if method == "off":
                    duplicates = []
                else:
                    if method == "auto":
                        method = (
                            "exact"
                            if accepted and all(item.labels is not None for item in accepted)
                            else "minhash"
                        )
                    if method == "exact":
                        accepted, duplicates = await asyncio.to_thread(deduper.deduplicate_exact, accepted)
                    elif method == "semantic-local":
                        accepted, duplicates = await asyncio.to_thread(
                            deduper.deduplicate_semantic_local,
                            accepted,
                            threshold=self.config.dedup_threshold,
                            model_name=self.config.embedding_model,
                        )
                    else:
                        accepted, duplicates = await asyncio.to_thread(
                            deduper.deduplicate_minhash,
                            accepted,
                            threshold=self.config.dedup_threshold,
                        )
                for item in duplicates:
                    self._reject(item, "dedup", str(item.metadata.get("dedup_reason", "duplicate")))
                rejected.extend(duplicates)
                metrics["stages"]["dedup"] = {"method": method, "rejected": len(duplicates)}
                self._observability.record_stage_outcome(
                    "dedup", accepted=len(accepted), rejected=len(duplicates)
                )
            completed.append("dedup")
            await self._save_checkpoint(checkpoint_path, fingerprint, completed, accepted, rejected, profile, metrics)

        if "score_refine" not in completed:
            with self._observability.stage_span("score_refine"):
                effective_quality_mode = "off" if strict_labeled(accepted) else self.config.quality_mode
                if effective_quality_mode == "off":
                    metrics["stages"]["score_refine"] = {
                        "mode": "off",
                        "requested_mode": self.config.quality_mode,
                        "scored": 0,
                        "refined": 0,
                    }
                elif effective_quality_mode == "sampled":
                    audit_items = [
                        item.model_copy(deep=True)
                        for item in representative_sample(accepted, self.config.quality_sample_size)
                    ]
                    scorer = QualityScorer(client=self.client, limiter=self.limiter, model=self.config.fast_model)
                    await scorer.audit_sample_batch_async(
                        audit_items,
                        batch_size=self.config.quality_audit_batch_size,
                        task_type=profile.task_type.value if profile and profile.task_type else None,
                        classes=profile.classes if profile else None,
                    )
                    scores = [
                        item.quality_score.overall_score
                        for item in audit_items
                        if item.quality_score is not None and not item.metadata.get("scoring_error")
                    ]
                    metrics["stages"]["score_refine"] = {
                        "mode": "sampled",
                        "sample_requested": min(self.config.quality_sample_size, len(accepted)),
                        "scored": len(scores),
                        "failed": len(audit_items) - len(scores),
                        "remote_batches": scorer.last_audit_metrics["remote_batches"],
                        "remote_records": scorer.last_audit_metrics["remote_records"],
                        "average_overall_score": sum(scores) / len(scores) if scores else None,
                        "minimum_overall_score": min(scores) if scores else None,
                        "refined": 0,
                    }
                else:
                    scorer = QualityScorer(client=self.client, limiter=self.limiter, model=self.config.fast_model)
                    accepted = await scorer.audit_sample_batch_async(
                        accepted,
                        batch_size=self.config.quality_audit_batch_size,
                        task_type=profile.task_type.value if profile and profile.task_type else None,
                        classes=profile.classes if profile else None,
                    )
                    refinable = [
                        item
                        for item in accepted
                        if item.quality_score
                        and item.quality_score.overall_score < self.config.refine_below_score
                        and item.format.value in {"alpaca", "chat", "dpo"}
                        and not item.metadata.get("scoring_error")
                    ]
                    if refinable:
                        refiner = DataRefiner(client=self.client, limiter=self.limiter, model=self.model)
                        await refiner.refine_batch_async(refinable, mode=self.config.refine_mode)
                        await scorer.audit_sample_batch_async(
                            refinable,
                            batch_size=self.config.quality_audit_batch_size,
                            task_type=profile.task_type.value if profile and profile.task_type else None,
                            classes=profile.classes if profile else None,
                        )
                    metrics["stages"]["score_refine"] = {
                        "mode": "llm",
                        "scored": len(accepted),
                        "refined": len(refinable),
                    }
                self._observability.record_stage_outcome("score_refine", accepted=len(accepted))
            completed.append("score_refine")
            await self._save_checkpoint(checkpoint_path, fingerprint, completed, accepted, rejected, profile, metrics)

        if "filter" not in completed:
            with self._observability.stage_span("filter"):
                kept: List[DatasetItem] = []
                filtered: List[DatasetItem] = []
                effective_quality_mode = "off" if strict_labeled(accepted) else self.config.quality_mode
                if effective_quality_mode != "llm":
                    metrics["stages"]["filter"] = {
                        "action": "skipped",
                        "reason": (
                            "accuracy_contract=strict preserves labeled rows"
                            if strict_labeled(accepted)
                            else f"quality_mode={effective_quality_mode} does not produce per-row scores"
                        ),
                        "accepted": len(accepted),
                        "rejected": 0,
                    }
                else:
                    for item in accepted:
                        score = item.quality_score
                        if item.metadata.get("scoring_error"):
                            reason = f"Provider scoring failed: {item.metadata['scoring_error']}"
                        elif score is None:
                            reason = "No final quality score was produced."
                        elif not score.is_safe:
                            reason = "Record failed the safety assessment."
                        elif score.overall_score < self.config.filter_min_score:
                            reason = f"Final quality score {score.overall_score:.2f} is below {self.config.filter_min_score:.2f}."
                        else:
                            kept.append(item)
                            continue
                        self._reject(item, "filter", reason)
                        filtered.append(item)
                    accepted = kept
                    rejected.extend(filtered)
                    metrics["stages"]["filter"] = {"accepted": len(accepted), "rejected": len(filtered)}
                self._observability.record_stage_outcome(
                    "filter", accepted=len(accepted), rejected=len(filtered)
                )
            completed.append("filter")
            await self._save_checkpoint(checkpoint_path, fingerprint, completed, accepted, rejected, profile, metrics)

        if "classify" not in completed:
            rejected_before = len(rejected)
            with self._observability.stage_span("classify"):
                classifier = DatasetClassifier(self.client, self.limiter, self.config.fast_model)
                already_labeled = bool(accepted) and all(item.labels is not None for item in accepted)
                schema: Optional[ClassificationSchema] = None
                if strict_labeled(accepted):
                    action = "strict_existing_labels"
                elif self.config.classification == ClassificationMode.OFF:
                    action = "disabled"
                elif already_labeled and self.config.classification == ClassificationMode.AUTO:
                    action = "existing_labels"
                else:
                    if self.config.classification == ClassificationMode.AUTO:
                        schema = ClassificationSchema(
                            applicable=bool(profile and profile.classification_applicable),
                            task_type=profile.task_type if profile else None,
                            classes=profile.classes if profile else [],
                            confidence=profile.confidence if profile else 0.0,
                            reasoning=profile.reasoning if profile else "No profile available.",
                        )
                    else:
                        schema = await classifier.resolve_schema(
                            accepted,
                            mode=self.config.classification,
                            classes=self.config.classes,
                            sample_size=self.config.classification_sample_size,
                        )
                    if not schema.applicable:
                        action = "not_applicable"
                    elif schema.confidence < self.config.classification_confidence:
                        action = "low_confidence_skip"
                    else:
                        accepted = await classifier.classify_batch(accepted, schema)
                        accepted, failed = classifier.partition_failures(accepted)
                        for item in failed:
                            self._reject(item, "classify", item.metadata["classification_error"])
                        rejected.extend(failed)
                        action = "classified"
                metrics["stages"]["classify"] = {
                    "action": action,
                    "task_type": schema.task_type.value if schema and schema.task_type else None,
                    "classes": schema.classes if schema else (profile.classes if profile else []),
                    "confidence": schema.confidence if schema else (profile.confidence if profile else 0.0),
                }
                self._observability.record_stage_outcome(
                    "classify", accepted=len(accepted), rejected=len(rejected) - rejected_before
                )
            completed.append("classify")
            await self._save_checkpoint(checkpoint_path, fingerprint, completed, accepted, rejected, profile, metrics)

        metrics["accepted_records"] = len(accepted)
        metrics["rejected_records"] = len(rejected)
        metrics["accuracy_contract"] = self.config.accuracy_contract
        metrics["network_policy"] = self.config.network_policy
        metrics["provider"] = self.client.provider.value
        metrics["model"] = self.model
        metrics["usage"] = dict(getattr(self.client, "usage", {}))
        usage = metrics["usage"]
        if usage.get("input_tokens") or usage.get("output_tokens"):
            self._observability.record_tokens(
                self.client.provider.value,
                self.model,
                input_tokens=usage.get("input_tokens", 0),
                output_tokens=usage.get("output_tokens", 0),
            )

        if profile is None:
            profile = await DatasetProfiler(self.client, self.model).profile(accepted)

        if output_path:
            await asyncio.to_thread(write_dataset_atomic, accepted, output_path)
        if rejected_path:
            await asyncio.to_thread(write_dataset_atomic, rejected, rejected_path)
        if report_path:
            self._atomic_json(
                report_path,
                {
                    "profile": profile.model_dump(mode="json"),
                    "metrics": metrics,
                    "rejection_reasons": self._rejection_counts(rejected),
                },
            )
            if self.config.report_html and output_path:
                ReportGenerator.generate_html_report(
                    accepted,
                    output_path.with_name(f"{output_path.stem}.report.html"),
                    provider=f"{self.client.provider.value} / {self.model}",
                )
        if output_path and self._observability.prometheus_enabled:
            # Standard Prometheus node_exporter "textfile collector" format: this is a
            # batch CLI run, not a long-lived process with a /metrics endpoint to scrape,
            # so the sidecar file is the correct hand-off point -- point node_exporter's
            # --collector.textfile.directory at the output directory and it's picked up
            # on its own schedule.
            metrics_path = output_path.with_name(f"{output_path.stem}.metrics.prom")
            metrics_path.write_text(self._observability.export_prometheus_text(), encoding="utf-8")
        if checkpoint_path and checkpoint_path.exists():
            checkpoint_path.unlink()

        return OptimizationRunResult(
            accepted=accepted,
            rejected=rejected,
            profile=profile,
            metrics=metrics,
            output_path=str(output_path) if output_path else None,
            rejected_path=str(rejected_path) if rejected_path else None,
            report_path=str(report_path) if report_path else None,
        )

    @staticmethod
    def _rejection_counts(items: List[DatasetItem]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for item in items:
            rejection = item.metadata.get("rejection", {})
            key = f"{rejection.get('stage', 'unknown')}: {rejection.get('reason', 'unknown')}"
            counts[key] = counts.get(key, 0) + 1
        return counts
