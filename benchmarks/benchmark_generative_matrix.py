#!/usr/bin/env python3
"""Large clean-vs-dirty benchmarks for non-classification dataset formats.

The benchmark uses real Hugging Face records, then compares the unmodified source
slice with BuffData's output. A second condition appends disclosed deterministic
defects (40% exact duplicates and 10% invalid rows by default) to the same source
slice. No source row is removed or changed during defect injection.

Unlike the classifier benchmarks, this suite measures structural data quality:
row/content preservation, duplicate and invalid-row recall, defect leakage,
clean-output parity, character retention, and throughput. Quality scoring and
classification are disabled, and the client below cannot make network requests.
"""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter
import copy
import json
from pathlib import Path
import random
import statistics
import time
from typing import Any, Iterable

from buffdata.engine.client import LLMProvider
from buffdata.engine.pipeline import OptimizationPipeline
from buffdata.models.schemas import DatasetFormat, DatasetItem, PipelineConfig
from buffdata.optimizers.classifier import ClassificationSchema
from buffdata.optimizers.dedup import Deduplicator


CATEGORY_CLEAN = "clean_source"
CATEGORY_DUPLICATE = "injected_exact_duplicate"
CATEGORY_INVALID = "injected_invalid"


def _nonempty(value: Any) -> str:
    return str(value or "").strip()


def _assistant_text(messages: Any) -> str:
    if not isinstance(messages, list):
        return ""
    for message in reversed(messages):
        if isinstance(message, dict) and message.get("role") in {"assistant", "gpt", "model"}:
            content = _nonempty(message.get("content"))
            if content:
                return content
    return ""


def _alpaca(row: dict[str, Any]) -> dict[str, Any] | None:
    instruction, output = _nonempty(row.get("instruction")), _nonempty(row.get("output"))
    if not instruction or not output:
        return None
    return {"instruction": instruction, "input": _nonempty(row.get("input")), "output": output}


def _chat(row: dict[str, Any]) -> dict[str, Any] | None:
    messages = []
    for message in row.get("messages") or []:
        if not isinstance(message, dict):
            continue
        role, content = _nonempty(message.get("role")), _nonempty(message.get("content"))
        if role and content:
            messages.append({"role": role, "content": content})
    if not any(message["role"] in {"user", "human"} for message in messages):
        return None
    if not any(message["role"] in {"assistant", "gpt", "model"} for message in messages):
        return None
    return {"messages": messages}


def _dpo(row: dict[str, Any]) -> dict[str, Any] | None:
    prompt = _nonempty(row.get("prompt"))
    chosen, rejected = _assistant_text(row.get("chosen")), _assistant_text(row.get("rejected"))
    if not prompt or not chosen or not rejected:
        return None
    return {"prompt": prompt, "chosen": chosen, "rejected": rejected}


def _qa(row: dict[str, Any]) -> dict[str, Any] | None:
    answers = row.get("answers") or {}
    answer_values = answers.get("text") or [] if isinstance(answers, dict) else []
    answer = _nonempty(answer_values[0]) if answer_values else ""
    question, context = _nonempty(row.get("question")), _nonempty(row.get("context"))
    if not question or not context or not answer:
        return None
    return {"instruction": question, "input": context, "output": answer}


def _summarization(row: dict[str, Any]) -> dict[str, Any] | None:
    document, summary = _nonempty(row.get("document")), _nonempty(row.get("summary"))
    if not document or not summary:
        return None
    return {
        "instruction": "Summarize the following document.",
        "input": document,
        "output": summary,
    }


def _raw(row: dict[str, Any]) -> dict[str, Any] | None:
    text = _nonempty(row.get("text"))
    return {"text": text} if text else None


DATASETS: dict[str, dict[str, Any]] = {
    "alpaca_sft": {
        "hf_id": "yahma/alpaca-cleaned",
        "split": "train",
        "family": "instruction/SFT",
        "format": "alpaca",
        "description": "Alpaca instruction-following pairs",
        "normalize": _alpaca,
    },
    "ultrachat": {
        "hf_id": "HuggingFaceH4/ultrachat_200k",
        "split": "train_sft",
        "family": "multi-turn chat",
        "format": "chat",
        "description": "UltraChat multi-turn SFT conversations",
        "normalize": _chat,
    },
    "ultrafeedback_dpo": {
        "hf_id": "HuggingFaceH4/ultrafeedback_binarized",
        "split": "train_prefs",
        "family": "preference/DPO",
        "format": "dpo",
        "description": "UltraFeedback chosen/rejected response pairs",
        "normalize": _dpo,
    },
    "squad_qa": {
        "hf_id": "rajpurkar/squad",
        "split": "train",
        "family": "extractive QA",
        "format": "alpaca",
        "description": "SQuAD question/context/answer triples",
        "normalize": _qa,
    },
    "xsum": {
        "hf_id": "EdinburghNLP/xsum",
        "split": "train",
        "family": "summarization",
        "format": "alpaca",
        "description": "XSum document-summary pairs",
        "normalize": _summarization,
    },
    "wikitext_103": {
        "hf_id": "Salesforce/wikitext",
        "config": "wikitext-103-raw-v1",
        "split": "train",
        "family": "raw language modeling",
        "format": "raw",
        "description": "WikiText-103 raw non-empty text records",
        "normalize": _raw,
    },
}


class OfflineProfilerClient:
    """A zero-network client that only answers raw-text task applicability locally."""

    provider = LLMProvider.GEMINI
    default_model = "offline-structural-benchmark"
    usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    async def generate_structured_async(self, *args: Any, **kwargs: Any) -> Any:
        response_schema = kwargs.get("response_schema")
        if response_schema is ClassificationSchema:
            return ClassificationSchema(
                applicable=False,
                confidence=1.0,
                reasoning="Raw language-modeling text is generative, not closed-set classification.",
            )
        raise AssertionError("Structural benchmark attempted an unexpected LLM call")

    def generate_structured(self, *args: Any, **kwargs: Any) -> Any:
        raise AssertionError("Structural benchmark attempted a synchronous LLM call")

    def generate_text(self, *args: Any, **kwargs: Any) -> Any:
        raise AssertionError("Structural benchmark attempted a text-generation call")

    async def generate_text_async(self, *args: Any, **kwargs: Any) -> Any:
        raise AssertionError("Structural benchmark attempted a text-generation call")

    def embed_texts(self, *args: Any, **kwargs: Any) -> Any:
        raise AssertionError("Structural benchmark attempted an embedding call")


def normalize_row(dataset_name: str, source: dict[str, Any]) -> dict[str, Any] | None:
    return DATASETS[dataset_name]["normalize"](source)


def load_source_rows(dataset_name: str, count: int, seed: int) -> list[dict[str, Any]]:
    from datasets import load_dataset

    spec = DATASETS[dataset_name]
    kwargs: dict[str, Any] = {"split": spec["split"], "streaming": True}
    if spec.get("config"):
        stream = load_dataset(spec["hf_id"], spec["config"], **kwargs)
    else:
        stream = load_dataset(spec["hf_id"], **kwargs)
    stream = stream.shuffle(seed=seed, buffer_size=10_000)

    rows: list[dict[str, Any]] = []
    for source in stream:
        normalized = normalize_row(dataset_name, source)
        if normalized is None:
            continue
        normalized["id"] = f"{dataset_name}-clean-{len(rows):06d}"
        normalized["_buffdata_metadata"] = {"defect_category": CATEGORY_CLEAN}
        rows.append(normalized)
        if len(rows) >= count:
            break
    if not rows:
        raise ValueError(f"{dataset_name} yielded no valid rows")
    return rows


def _invalid_row(dataset_name: str, index: int) -> dict[str, Any]:
    fmt = DATASETS[dataset_name]["format"]
    if fmt == "alpaca":
        row: dict[str, Any] = {"instruction": "Injected invalid record", "input": "", "output": ""}
    elif fmt == "chat":
        row = {"messages": []}
    elif fmt == "dpo":
        row = {"prompt": "", "chosen": "", "rejected": ""}
    else:
        row = {"text": ""}
    row["id"] = f"{dataset_name}-invalid-{index:06d}"
    row["_buffdata_metadata"] = {"defect_category": CATEGORY_INVALID}
    return row


def make_dirty(
    dataset_name: str,
    clean_rows: list[dict[str, Any]],
    duplicate_ratio: float,
    invalid_ratio: float,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Append defects while leaving every source record unchanged and first in order."""
    rng = random.Random(seed)
    duplicate_count = int(len(clean_rows) * duplicate_ratio)
    invalid_count = int(len(clean_rows) * invalid_ratio)
    duplicates: list[dict[str, Any]] = []
    for index in range(duplicate_count):
        duplicate = copy.deepcopy(rng.choice(clean_rows))
        source_id = duplicate["id"]
        duplicate["id"] = f"{dataset_name}-duplicate-{index:06d}"
        duplicate["_buffdata_metadata"] = {
            "defect_category": CATEGORY_DUPLICATE,
            "duplicate_of": source_id,
        }
        duplicates.append(duplicate)
    invalid = [_invalid_row(dataset_name, index) for index in range(invalid_count)]
    return copy.deepcopy(clean_rows) + duplicates + invalid, {
        "source": len(clean_rows),
        "duplicates": duplicate_count,
        "invalid": invalid_count,
    }


def _semantic_payload(item: DatasetItem) -> Any:
    if item.format == DatasetFormat.ALPACA:
        return [item.instruction or "", item.input or "", item.output or ""]
    if item.format == DatasetFormat.CHAT:
        return [[message.role, message.content] for message in item.messages or []]
    if item.format == DatasetFormat.DPO:
        return [item.prompt or "", item.chosen or "", item.rejected or ""]
    if item.format == DatasetFormat.RAW:
        return item.text or ""
    return item.raw_data


def _signature(item: DatasetItem) -> str:
    return json.dumps(_semantic_payload(item), ensure_ascii=False, sort_keys=True)


def _content_chars(item: DatasetItem) -> int:
    payload = _semantic_payload(item)
    if isinstance(payload, str):
        return len(payload)
    return len(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))


def duplicate_inventory(items: Iterable[DatasetItem]) -> dict[str, int]:
    counts = Counter(Deduplicator._content(item).strip() for item in items)
    return {
        "unique_content_rows": len(counts),
        "duplicate_groups": sum(count > 1 for count in counts.values()),
        "duplicate_rows": sum(count - 1 for count in counts.values() if count > 1),
    }


def summarize_run(
    input_rows: list[dict[str, Any]],
    accepted: list[DatasetItem],
    rejected: list[DatasetItem],
    elapsed_seconds: float,
) -> dict[str, Any]:
    input_items = [DatasetItem.from_dict(row) for row in input_rows]
    source_items = {
        item.id: item
        for item in input_items
        if item.metadata.get("defect_category") == CATEGORY_CLEAN
    }
    accepted_by_id = {item.id: item for item in accepted}
    retained_source = [accepted_by_id[item_id] for item_id in source_items if item_id in accepted_by_id]
    source_changed = sum(
        _signature(source_items[item.id]) != _signature(item) for item in retained_source
    )

    categories: dict[str, dict[str, int]] = {}
    for category in (CATEGORY_CLEAN, CATEGORY_DUPLICATE, CATEGORY_INVALID):
        entered = sum(item.metadata.get("defect_category") == category for item in input_items)
        retained = sum(item.metadata.get("defect_category") == category for item in accepted)
        categories[category] = {"input": entered, "retained": retained, "removed": entered - retained}

    rejected_by_stage = Counter(
        item.metadata.get("rejection", {}).get("stage", "unknown") for item in rejected
    )
    before_chars = sum(_content_chars(item) for item in source_items.values())
    after_chars = sum(_content_chars(item) for item in retained_source)
    duplicate_total = categories[CATEGORY_DUPLICATE]["input"]
    invalid_total = categories[CATEGORY_INVALID]["input"]
    return {
        "input_rows": len(input_items),
        "retained_rows": len(accepted),
        "deleted_rows": len(rejected),
        "deletion_rate": len(rejected) / len(input_items) if input_items else 0.0,
        "source_rows": len(source_items),
        "source_retained": len(retained_source),
        "source_lost": len(source_items) - len(retained_source),
        "source_changed": source_changed,
        "source_exact_preservation_rate": (
            (len(retained_source) - source_changed) / len(source_items) if source_items else 1.0
        ),
        "source_characters_before": before_chars,
        "source_characters_after": after_chars,
        "source_character_retention": after_chars / before_chars if before_chars else 1.0,
        "input_duplicate_inventory": duplicate_inventory(input_items),
        "categories": categories,
        "duplicate_removal_recall": (
            categories[CATEGORY_DUPLICATE]["removed"] / duplicate_total if duplicate_total else None
        ),
        "invalid_removal_recall": (
            categories[CATEGORY_INVALID]["removed"] / invalid_total if invalid_total else None
        ),
        "injected_defect_leakage": (
            categories[CATEGORY_DUPLICATE]["retained"] + categories[CATEGORY_INVALID]["retained"]
        ),
        "rejected_by_stage": dict(sorted(rejected_by_stage.items())),
        "elapsed_seconds": elapsed_seconds,
        "throughput_rows_per_second": len(input_items) / elapsed_seconds if elapsed_seconds else None,
    }


async def optimize(rows: list[dict[str, Any]]) -> tuple[list[DatasetItem], list[DatasetItem], dict[str, Any]]:
    items = [DatasetItem.from_dict(row) for row in rows]
    pipeline = OptimizationPipeline(
        PipelineConfig(
            provider="gemini",
            quality_mode="off",
            classification="off",
            scrub_pii=False,
            dedup_method="exact",
            network_policy="strict",
            report_html=False,
        ),
        client=OfflineProfilerClient(),
    )
    started = time.perf_counter()
    result = await pipeline.run(items)
    elapsed = time.perf_counter() - started
    return result.accepted, result.rejected, {"pipeline": result.metrics, "elapsed": elapsed}


def _output_map(items: list[DatasetItem]) -> dict[str, str]:
    return {item.id: _signature(item) for item in items}


def score_recovery_accuracy(
    input_rows: list[dict[str, Any]],
    accepted: list[DatasetItem],
    clean_reference: dict[str, str],
) -> dict[str, Any]:
    """Score keep/delete decisions and content against the clean optimized control.

    Deletion is the positive class. A source ID retained by the clean control should
    be kept; pre-existing source duplicates omitted by that control, injected exact
    duplicates, and injected invalid rows should be deleted. This makes the metric
    meaningful for every supported generative format without pretending these open-
    ended datasets have class labels.
    """
    actual = _output_map(accepted)
    input_ids = [str(row["id"]) for row in input_rows]
    expected_keep = set(clean_reference)
    actual_keep = set(actual)

    true_positive = sum(item_id not in expected_keep and item_id not in actual_keep for item_id in input_ids)
    false_positive = sum(item_id in expected_keep and item_id not in actual_keep for item_id in input_ids)
    false_negative = sum(item_id not in expected_keep and item_id in actual_keep for item_id in input_ids)
    true_negative = sum(item_id in expected_keep and item_id in actual_keep for item_id in input_ids)
    total = len(input_ids)
    predicted_deleted = true_positive + false_positive
    expected_deleted = true_positive + false_negative
    precision = true_positive / predicted_deleted if predicted_deleted else 1.0
    recall = true_positive / expected_deleted if expected_deleted else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    content_matches = sum(actual.get(item_id) == signature for item_id, signature in clean_reference.items())

    return {
        "definition": "keep/delete decisions against clean optimized control; deletion is positive",
        "true_positive_deleted": true_positive,
        "false_positive_deleted": false_positive,
        "false_negative_deleted": false_negative,
        "true_negative_retained": true_negative,
        "expected_deleted": expected_deleted,
        "predicted_deleted": predicted_deleted,
        "cleaning_decision_accuracy": (true_positive + true_negative) / total if total else 1.0,
        "deletion_precision": precision,
        "deletion_recall": recall,
        "deletion_f1": f1,
        "content_exact_match_rows": content_matches,
        "content_expected_rows": len(clean_reference),
        "output_content_accuracy": content_matches / len(clean_reference) if clean_reference else 1.0,
    }


async def run_combo(
    dataset_name: str,
    clean_rows: list[dict[str, Any]],
    requested_scale: int,
    duplicate_ratio: float,
    invalid_ratio: float,
    seed: int,
) -> dict[str, Any]:
    effective_rows = min(requested_scale, len(clean_rows))
    source = copy.deepcopy(clean_rows[:effective_rows])
    dirty, injected = make_dirty(
        dataset_name, source, duplicate_ratio, invalid_ratio, seed + requested_scale
    )

    clean_accepted, clean_rejected, clean_run = await optimize(copy.deepcopy(source))
    dirty_accepted, dirty_rejected, dirty_run = await optimize(dirty)
    clean_metrics = summarize_run(source, clean_accepted, clean_rejected, clean_run["elapsed"])
    dirty_metrics = summarize_run(dirty, dirty_accepted, dirty_rejected, dirty_run["elapsed"])
    clean_output, dirty_output = _output_map(clean_accepted), _output_map(dirty_accepted)
    accuracy = score_recovery_accuracy(dirty, dirty_accepted, clean_output)

    return {
        "dataset": dataset_name,
        "hf_id": DATASETS[dataset_name]["hf_id"],
        "family": DATASETS[dataset_name]["family"],
        "format": DATASETS[dataset_name]["format"],
        "description": DATASETS[dataset_name]["description"],
        "requested_source_rows": requested_scale,
        "effective_source_rows": effective_rows,
        "injected": injected,
        "clean": clean_metrics,
        "dirty": dirty_metrics,
        "accuracy": accuracy,
        "dirty_output_matches_clean_output": dirty_output == clean_output,
        "clean_pipeline_metrics": clean_run["pipeline"],
        "dirty_pipeline_metrics": dirty_run["pipeline"],
    }


def aggregate(combos: list[dict[str, Any]]) -> dict[str, Any]:
    dirty = [combo["dirty"] for combo in combos]
    clean = [combo["clean"] for combo in combos]
    duplicates = sum(entry["categories"][CATEGORY_DUPLICATE]["input"] for entry in dirty)
    duplicates_removed = sum(entry["categories"][CATEGORY_DUPLICATE]["removed"] for entry in dirty)
    invalid = sum(entry["categories"][CATEGORY_INVALID]["input"] for entry in dirty)
    invalid_removed = sum(entry["categories"][CATEGORY_INVALID]["removed"] for entry in dirty)
    true_positive = sum(combo["accuracy"]["true_positive_deleted"] for combo in combos)
    false_positive = sum(combo["accuracy"]["false_positive_deleted"] for combo in combos)
    false_negative = sum(combo["accuracy"]["false_negative_deleted"] for combo in combos)
    true_negative = sum(combo["accuracy"]["true_negative_retained"] for combo in combos)
    deletion_precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 1.0
    deletion_recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 1.0
    content_matches = sum(combo["accuracy"]["content_exact_match_rows"] for combo in combos)
    content_expected = sum(combo["accuracy"]["content_expected_rows"] for combo in combos)
    return {
        "combinations": len(combos),
        "datasets": len({combo["dataset"] for combo in combos}),
        "task_families": len({combo["family"] for combo in combos}),
        "source_rows": sum(entry["source_rows"] for entry in clean),
        "dirty_input_rows": sum(entry["input_rows"] for entry in dirty),
        "dirty_deleted_rows": sum(entry["deleted_rows"] for entry in dirty),
        "clean_source_rows_lost": sum(entry["source_lost"] for entry in clean),
        "clean_source_rows_changed": sum(entry["source_changed"] for entry in clean),
        "dirty_source_rows_lost": sum(entry["source_lost"] for entry in dirty),
        "dirty_source_rows_changed": sum(entry["source_changed"] for entry in dirty),
        "injected_duplicates": duplicates,
        "injected_duplicates_removed": duplicates_removed,
        "duplicate_removal_recall": duplicates_removed / duplicates if duplicates else None,
        "injected_invalid": invalid,
        "injected_invalid_removed": invalid_removed,
        "invalid_removal_recall": invalid_removed / invalid if invalid else None,
        "accuracy_confusion_matrix": {
            "true_positive_deleted": true_positive,
            "false_positive_deleted": false_positive,
            "false_negative_deleted": false_negative,
            "true_negative_retained": true_negative,
        },
        "cleaning_decision_accuracy": (
            (true_positive + true_negative)
            / (true_positive + false_positive + false_negative + true_negative)
        ),
        "deletion_precision": deletion_precision,
        "deletion_recall": deletion_recall,
        "deletion_f1": (
            2 * deletion_precision * deletion_recall / (deletion_precision + deletion_recall)
            if deletion_precision + deletion_recall
            else 0.0
        ),
        "output_content_accuracy": content_matches / content_expected if content_expected else 1.0,
        "output_content_exact_matches": content_matches,
        "output_content_expected_rows": content_expected,
        "injected_defect_leakage": sum(entry["injected_defect_leakage"] for entry in dirty),
        "clean_dirty_output_parity_combos": sum(
            combo["dirty_output_matches_clean_output"] for combo in combos
        ),
        "mean_clean_throughput_rows_per_second": statistics.fmean(
            entry["throughput_rows_per_second"] for entry in clean
        ),
        "mean_dirty_throughput_rows_per_second": statistics.fmean(
            entry["throughput_rows_per_second"] for entry in dirty
        ),
        "remote_input_tokens": sum(
            combo[condition]["usage"].get("input_tokens", 0)
            for combo in combos
            for condition in ("clean_pipeline_metrics", "dirty_pipeline_metrics")
        ),
        "remote_output_tokens": sum(
            combo[condition]["usage"].get("output_tokens", 0)
            for combo in combos
            for condition in ("clean_pipeline_metrics", "dirty_pipeline_metrics")
        ),
    }


def render_report(payload: dict[str, Any]) -> str:
    summary, combos = payload["summary"], payload["combinations"]
    distinct_source_rows = sum(
        max(
            combo["effective_source_rows"]
            for combo in combos
            if combo["dataset"] == dataset_name
        )
        for dataset_name in {combo["dataset"] for combo in combos}
    )
    lines = [
        "# BuffData non-classification benchmark",
        "",
        "This suite compares each unmodified Hugging Face source slice with BuffData's output,",
        "then repeats the run after appending deterministic exact duplicates and invalid rows.",
        "No source row is modified or removed during injection. Classification, PII scrubbing,",
        "and LLM quality scoring are off; recorded provider token usage is zero.",
        "",
        "## Aggregate result",
        "",
        f"- {summary['datasets']} datasets / {summary['task_families']} task families / {summary['combinations']} scale combinations",
        f"- {summary['source_rows']:,} clean source-row evaluations ({distinct_source_rows:,} distinct loaded records) and {summary['dirty_input_rows']:,} dirty-pipeline row evaluations",
        f"- {summary['dirty_deleted_rows']:,} dirty rows deleted",
        f"- {summary['injected_duplicates_removed']:,}/{summary['injected_duplicates']:,} injected duplicates removed ({summary['duplicate_removal_recall']:.2%})",
        f"- {summary['injected_invalid_removed']:,}/{summary['injected_invalid']:,} invalid rows removed ({summary['invalid_removal_recall']:.2%})",
        f"- cleaning-decision accuracy: {summary['cleaning_decision_accuracy']:.4%}",
        f"- deletion precision / recall / F1: {summary['deletion_precision']:.4%} / {summary['deletion_recall']:.4%} / {summary['deletion_f1']:.4%}",
        f"- output-content exact-match accuracy: {summary['output_content_accuracy']:.4%} ({summary['output_content_exact_matches']:,}/{summary['output_content_expected_rows']:,})",
        f"- {summary['injected_defect_leakage']:,} injected defects leaked into output",
        f"- clean control: {summary['clean_source_rows_lost']:,} source rows lost, {summary['clean_source_rows_changed']:,} source rows changed",
        f"- dirty recovery: {summary['dirty_source_rows_lost']:,} source rows lost, {summary['dirty_source_rows_changed']:,} source rows changed",
        f"- dirty output exactly matched clean output in {summary['clean_dirty_output_parity_combos']}/{summary['combinations']} combinations",
        f"- remote usage: {summary['remote_input_tokens']:,} input tokens / {summary['remote_output_tokens']:,} output tokens",
        "",
        "## Per-dataset comparison",
        "",
        "| Dataset | Task family | Source | Dirty input | Decision accuracy | Delete P/R/F1 | Content accuracy | Clean lost/changed | Dirty deleted | Defect leaks | Output parity |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for combo in combos:
        clean, dirty = combo["clean"], combo["dirty"]
        accuracy = combo["accuracy"]
        lines.append(
            f"| {combo['dataset']} @ {combo['effective_source_rows']:,} | {combo['family']} | "
            f"{combo['effective_source_rows']:,} | {dirty['input_rows']:,} | "
            f"{accuracy['cleaning_decision_accuracy']:.4%} | "
            f"{accuracy['deletion_precision']:.2%}/{accuracy['deletion_recall']:.2%}/{accuracy['deletion_f1']:.2%} | "
            f"{accuracy['output_content_accuracy']:.4%} | "
            f"{clean['source_lost']:,}/{clean['source_changed']:,} | {dirty['deleted_rows']:,} | "
            f"{dirty['injected_defect_leakage']:,} | "
            f"{'yes' if combo['dirty_output_matches_clean_output'] else 'no'} |"
        )
    lines.extend([
        "",
        "`Clean lost` includes pre-existing exact duplicates in the published source slice;",
        "`changed` compares format-aware semantic payloads (instruction/input/output, every chat",
        "message, prompt/chosen/rejected, or raw text). Output parity requires identical retained",
        "source IDs and identical semantic payloads between the clean and dirty pipeline outputs.",
        "",
        "## Accuracy definition",
        "",
        "These are open-ended generative datasets, so ordinary class-label accuracy is undefined.",
        "The benchmark therefore treats deletion as the positive class and uses the clean optimized",
        "output as ground truth: records present in that control should be kept; injected defects",
        "and pre-existing duplicates omitted by the control should be deleted. Decision accuracy",
        "scores every keep/delete decision. Deletion precision, recall, and F1 expose false clean",
        "deletions and leaked defects. Output-content accuracy additionally requires every expected",
        "retained record to match the clean control's complete format-aware semantic payload exactly.",
        "",
        "## Dataset catalog",
        "",
        "| Dataset | Hugging Face source | Native task | BuffData format |",
        "|---|---|---|---|",
    ])
    for name, spec in DATASETS.items():
        lines.append(f"| {name} | `{spec['hf_id']}` | {spec['family']} | `{spec['format']}` |")
    lines.extend([
        "",
        "## Method",
        "",
        "Rows are read from shuffled streaming splits with a fixed seed. The dirty input appends",
        "40% exact copies and 10% format-invalid records by default. Originals precede injected",
        "copies, so exact deduplication has an unambiguous record to keep. Both clean and dirty",
        "conditions execute BuffData's real validate, profile, exact-dedup, filter, and disabled",
        "classification stages. Full machine-readable stage metrics and rejection counts are in",
        "`results.json`.",
        "",
    ])
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=sorted(DATASETS), default=list(DATASETS))
    parser.add_argument("--scales", nargs="+", type=int, default=[10_000, 30_000])
    parser.add_argument("--duplicate-ratio", type=float, default=0.40)
    parser.add_argument("--invalid-ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("benchmarks/results-generative")
    )
    args = parser.parse_args()
    if any(scale <= 0 for scale in args.scales):
        parser.error("all scales must be positive")
    if not 0 <= args.duplicate_ratio <= 2:
        parser.error("--duplicate-ratio must be between 0 and 2")
    if not 0 <= args.invalid_ratio <= 1:
        parser.error("--invalid-ratio must be between 0 and 1")
    return args


async def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    combinations: list[dict[str, Any]] = []
    for offset, dataset_name in enumerate(args.datasets):
        print(f"Loading {dataset_name} (up to {max(args.scales):,} valid rows)...", flush=True)
        source_rows = load_source_rows(dataset_name, max(args.scales), args.seed + offset)
        for scale in args.scales:
            print(f"  Running clean and dirty conditions at {min(scale, len(source_rows)):,} rows...", flush=True)
            combo = await run_combo(
                dataset_name,
                source_rows,
                scale,
                args.duplicate_ratio,
                args.invalid_ratio,
                args.seed + offset * 1000,
            )
            combinations.append(combo)
            print(
                f"    deleted={combo['dirty']['deleted_rows']:,}, "
                f"leaked={combo['dirty']['injected_defect_leakage']:,}, "
                f"parity={combo['dirty_output_matches_clean_output']}",
                flush=True,
            )

    payload = {
        "benchmark": "buffdata-nonclassification-clean-vs-dirty",
        "configuration": {
            "datasets": args.datasets,
            "requested_scales": args.scales,
            "duplicate_ratio": args.duplicate_ratio,
            "invalid_ratio": args.invalid_ratio,
            "seed": args.seed,
            "quality_mode": "off",
            "classification": "off",
            "dedup_method": "exact",
            "network": "forbidden; deterministic local raw-text profiling response",
        },
        "summary": aggregate(combinations),
        "combinations": combinations,
    }
    (args.output_dir / "results.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (args.output_dir / "REPORT.md").write_text(render_report(payload), encoding="utf-8")
    print(f"Wrote {args.output_dir / 'REPORT.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
