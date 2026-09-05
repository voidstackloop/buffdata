#!/usr/bin/env python3
"""Claude-first end-to-end benchmark for BuffData.

Runs the same proven methodology as benchmark_buffdata.py (deterministic
dirty-data injection: 40% conflicting-label duplicates, 40% class-skew
duplicates, 10% empty rows, then BuffData's local validate + exact-dedup)
but with Anthropic Claude as the LLM provider for every real API call:

1. Local recovery (no LLM): dirty baseline vs. BuffData-cleaned output,
   trained with identical PyTorch classifiers across fixed seeds.
2. Claude sampled audit: optimize(..., provider="anthropic") with
   quality_mode="sampled" -- one structured batch request per ~10 rows,
   informational only, never deciding which rows survive validate/dedup.
3. Claude classification accuracy: DatasetClassifier directly on a small
   held-out sample, the same path `buffdata classify` uses.

Why a separate file instead of flags on the Gemini benchmarks: the existing
matrix scripts hardcode `create_llm_client("gemini", gemini_model)` and
`--gemini-model` defaults. This file defaults to
`--provider anthropic --model claude-sonnet-5` and always passes the model
explicitly -- important because BUFFDATA_DEFAULT_MODEL in .env (currently
gemini-3.7-flash) otherwise overrides the provider default inside
create_llm_client when no model is given.

Costs: real Claude calls are billed. Defaults are intentionally small
(2 datasets, 1000/500 rows, 2 epochs, 20 audit rows, 20 classify rows).
Use --smoke for a no-download synthetic check (one tiny live call), or
--mock for zero API calls (validates plumbing only).

Requires ANTHROPIC_API_KEY unless --mock is given.

External gateway (LiteLLM / company proxy serving Claude): pass
--provider openai_compatible --base-url <gateway-url> --model <gateway-model>.
--base-url is exported to OPENAI_COMPATIBLE_BASE_URL (which the pipeline's
optimize() path reads via create_llm_client's env fallback) and
ANTHROPIC_API_KEY is forwarded to OPENAI_COMPATIBLE_API_KEY when the latter
is unset, so the existing .env key is reused.

Examples:
  # cheapest live check: synthetic data, ~2 Claude calls
  python benchmarks/benchmark_claude.py --smoke

  # same via an external proxy (uses the .env ANTHROPIC_API_KEY as the
  # gateway key unless OPENAI_COMPATIBLE_API_KEY is set)
  python benchmarks/benchmark_claude.py --smoke \\
    --provider openai_compatible --base-url https://proxy.corp/v1 \\
    --model claude-sonnet-5

  # offline plumbing check: zero API calls
  python benchmarks/benchmark_claude.py --smoke --mock

  # small real run: ag_news + imdb from Hugging Face
  python benchmarks/benchmark_claude.py

  # existing full benchmark with a Claude audit instead:
  python benchmarks/benchmark_buffdata.py --audit-provider anthropic \\
    --anthropic-model claude-sonnet-5 --gemini-audit-rows 20 \\
    --datasets ag_news --train-rows 3000 --test-rows 1000

  # existing correctness benchmark with Claude instead of Gemini:
  python benchmarks/benchmark_classification_correctness.py \\
    --provider anthropic --model claude-sonnet-5 --datasets ag_news
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchmark_buffdata import (  # noqa: E402
    build_vocab,
    make_dirty,
    optimize,
    stratified_rows,
    summarize,
    train_once,
    write_jsonl,
)

DEFAULT_PROVIDER = "anthropic"
DEFAULT_MODEL = "claude-sonnet-5"
DEFAULT_DATASETS = ["ag_news", "imdb"]


def _configure_gateway(provider: str, base_url: str | None) -> None:
    """Forward gateway settings into the env vars the client layer reads.

    The optimize() helper below builds its own PipelineConfig without a
    base_url, but create_llm_client(provider="openai_compatible") falls back
    to OPENAI_COMPATIBLE_BASE_URL / OPENAI_COMPATIBLE_API_KEY env vars -- so
    exporting here makes both the direct classifier calls and the full
    pipeline audit path reach the proxy. Reuses .env ANTHROPIC_API_KEY as the
    gateway key when no gateway-specific key is set.
    """
    import os

    if base_url:
        os.environ["OPENAI_COMPATIBLE_BASE_URL"] = base_url
    if provider == "openai_compatible" and not os.getenv("OPENAI_COMPATIBLE_API_KEY"):
        fallback = os.getenv("ANTHROPIC_API_KEY")
        if fallback:
            os.environ["OPENAI_COMPATIBLE_API_KEY"] = fallback


def _synthetic_rows(count: int, classes: int, seed: int) -> list[dict[str, Any]]:
    """Deterministic synthetic classification rows for --smoke (no HF download)."""
    rng = random.Random(seed)
    topics = [
        "stock market rallies on earnings",
        "football team wins championship game",
        "new smartphone camera review",
        "election debate over tax policy",
        "movie sequel breaks box office record",
        "scientists discover new particle",
    ]
    rows = []
    for i in range(count):
        text = f"{rng.choice(topics)} #{i} {'great ' * (i % 3)}"
        rows.append({"text": text, "label": i % classes})
    return rows


class _FakeClassifyClient:
    """Offline stand-in for --mock: exercises the full DatasetClassifier code
    path with zero network calls. Always predicts the first class, so accuracy
    is deterministic but clearly labeled mock (see `mock: true` in output)."""

    def __init__(self, provider: str, model: str, classes: list[str]):
        from buffdata.engine.client import LLMProvider

        self.provider = LLMProvider(provider)
        self.default_model = model
        self.classes = classes
        self.usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    async def generate_structured_async(self, prompt: str, response_schema, model=None, **kwargs):
        from buffdata.optimizers.classifier import BatchClassificationResponse, ClassificationResult

        ids = [line.split("ID:", 1)[1].strip() for line in prompt.splitlines() if line.startswith("ID:")]
        return BatchClassificationResponse(
            results=[ClassificationResult(id=i, labels=[self.classes[0]], confidence=1.0) for i in ids]
        )


async def measure_classification(
    rows: list[dict[str, Any]],
    class_names: list[str],
    mode: str,
    provider: str,
    model: str,
    sample_size: int,
    seed: int,
    allow_mock: bool = False,
    base_url: str | None = None,
) -> dict[str, Any]:
    """Classify a stratified sample with real Claude calls via DatasetClassifier."""
    from buffdata.engine.limiter import AsyncRateLimiter
    from buffdata.models.schemas import DatasetItem
    from buffdata.optimizers.classifier import DatasetClassifier

    if allow_mock:
        client: Any = _FakeClassifyClient(provider=provider, model=model, classes=class_names)
    else:
        from buffdata.engine.client import create_llm_client

        _configure_gateway(provider, base_url)
        client = create_llm_client(provider, model=model, base_url=base_url)

    rng = random.Random(seed)
    by_class: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        by_class.setdefault(int(row["label"]), []).append(row)
    per_class = max(1, sample_size // max(1, len(class_names)))
    sampled: list[dict[str, Any]] = []
    for label_rows in by_class.values():
        take = list(label_rows)
        rng.shuffle(take)
        sampled.extend(take[:per_class])
    rng.shuffle(sampled)
    sampled = sampled[:sample_size]

    items = [DatasetItem.from_dict({"text": row["text"]}) for row in sampled]
    true_by_id = {item.id: class_names[int(row["label"])] for item, row in zip(items, sampled)}

    limiter = AsyncRateLimiter(max_rpm=60, concurrency=5)
    classifier = DatasetClassifier(client, limiter, model)

    started = time.perf_counter()
    schema = await classifier.resolve_schema(items, mode=mode, classes=class_names, sample_size=len(items))
    classified = await classifier.classify_batch(items, schema)
    elapsed = time.perf_counter() - started

    correct = scored = 0
    for item in classified:
        if item.metadata.get("classification_error"):
            continue
        true_label = true_by_id.get(item.id)
        predicted = item.labels if isinstance(item.labels, str) else None
        if true_label is None or predicted is None:
            continue
        scored += 1
        if predicted == true_label:
            correct += 1
    return {
        "provider": provider,
        "model": model,
        "mode": mode,
        "mock": allow_mock,
        "classes": class_names,
        "sampled": len(sampled),
        "scored": scored,
        "failed": len(sampled) - scored,
        "correct": correct,
        "accuracy": correct / scored if scored else None,
        "elapsed_seconds": round(elapsed, 1),
        "usage": dict(client.usage),
    }


async def run_smoke(args: argparse.Namespace) -> dict[str, Any]:
    """Synthetic end-to-end check without Hugging Face downloads."""
    print("=== smoke: synthetic binary classification (no HF download) ===", flush=True)
    _configure_gateway(args.provider, args.base_url)
    classes = 2
    class_names = ["negative", "positive"]
    clean = _synthetic_rows(60, classes, seed=7)
    test = _synthetic_rows(20, classes, seed=8)
    _, optimizer_input, defects = make_dirty(clean, classes, seed=9)

    # --mock must stay fully offline: optimize() has no allow_mock flag, so
    # force the audit off (quality_mode="off") instead of attempting a live
    # sampled audit with an absent key.
    audit_rows = 0 if args.mock else args.audit_rows
    clean_opt, clean_q = await optimize(clean, provider=args.provider, anthropic_model=args.model)
    dirty_opt, dirty_q = await optimize(
        optimizer_input,
        gemini_audit_rows=audit_rows,
        gemini_model=args.model,
        anthropic_model=args.model,
        provider=args.provider,
    )
    vocab = build_vocab(clean)
    conditions = {
        "clean_raw": clean,
        "clean_optimized": clean_opt,
        "dirty_raw": optimizer_input,
        "dirty_optimized": dirty_opt,
    }
    recovery: dict[str, Any] = {}
    for name, rows in conditions.items():
        runs = [train_once(rows, test, vocab, classes, seed, args.epochs) for seed in args.seeds]
        recovery[name] = {"rows": len(rows), "runs": runs, "summary": summarize(runs)}
        print(f"  {name}: acc={recovery[name]['summary']['accuracy_mean']:.4f}", flush=True)

    classification = None
    try:
        classification = await measure_classification(
            clean, class_names, "binary", args.provider, args.model,
            min(args.classify_sample_size, 6), seed=11, allow_mock=args.mock,
            base_url=args.base_url,
        )
        print(f"  classification: {classification['correct']}/{classification['scored']} correct", flush=True)
    except Exception as exc:  # noqa: BLE001 - keep recovery results even if classification fails
        classification = {"error": f"{type(exc).__name__}: {exc}"}
        print(f"  classification failed (recovery results kept): {exc}", flush=True)
    return {
        "mode": "smoke-synthetic",
        "defects": defects,
        "recovery": recovery,
        "dirty_audit_quality": dirty_q,
        "clean_audit_quality": clean_q,
        "classification": classification,
    }


async def run_dataset(
    name: str,
    spec: dict[str, Any],
    args: argparse.Namespace,
    dataset_offset: int,
) -> dict[str, Any]:
    from datasets import load_dataset

    print(f"=== {name} ({spec['hf_id']}) ===", flush=True)
    ds = load_dataset(spec["hf_id"], spec.get("config")) if spec.get("config") else load_dataset(spec["hf_id"])
    train_split = ds[spec.get("train_split", "train")]
    test_split = ds["test"] if "test" in ds else ds["validation"] if "validation" in ds else train_split
    label_col = spec.get("label_field", "label")
    text_fields: list[str] = spec["text_fields"]

    classes = spec["classes"]
    names = getattr(train_split.features[label_col], "names", None)
    class_names = list(names) if names else [str(i) for i in range(classes)]

    clean_train = stratified_rows(train_split, args.train_rows, classes, text_fields, 100 + dataset_offset, label_field=label_col)
    clean_test = stratified_rows(test_split, args.test_rows, classes, text_fields, 200 + dataset_offset, label_field=label_col)
    _, optimizer_input, defects = make_dirty(clean_train, classes, 300 + dataset_offset)

    clean_opt, clean_q = await optimize(clean_train, provider=args.provider, anthropic_model=args.model)
    dirty_opt, dirty_q = await optimize(
        optimizer_input,
        gemini_audit_rows=args.audit_rows,
        gemini_model=args.model,
        anthropic_model=args.model,
        provider=args.provider,
    )

    vocab = build_vocab(clean_train)
    conditions = {"clean_raw": clean_train, "clean_optimized": clean_opt, "dirty_optimized": dirty_opt}
    dirty_raw_rows, _, _ = make_dirty(clean_train, classes, 300 + dataset_offset)
    conditions["dirty_raw"] = dirty_raw_rows
    results: dict[str, Any] = {}
    for cond_name, rows in conditions.items():
        runs = [train_once(rows, clean_test, vocab, classes, seed, args.epochs) for seed in args.seeds]
        results[cond_name] = {"rows": len(rows), "runs": runs, "summary": summarize(runs)}
        print(f"  [{name}] {cond_name} acc={results[cond_name]['summary']['accuracy_mean']:.4f}", flush=True)

    classification = None
    if not args.skip_classification:
        mode = "binary" if classes == 2 else "multi-class"
        try:
            # NOTE: classify from clean_train, not the raw HF split -- it is
            # already normalized to {"text", "label"} while raw rows carry
            # dataset-specific columns (title/content, ...).
            classification = await measure_classification(
                clean_train, class_names, mode, args.provider, args.model,
                args.classify_sample_size, seed=17 + dataset_offset,
                allow_mock=False, base_url=args.base_url,
            )
            acc = f"{classification['accuracy']:.1%}" if classification["accuracy"] is not None else "n/a"
            print(f"  [{name}] {args.provider} classification: {classification['correct']}/{classification['scored']} ({acc})", flush=True)
        except Exception as exc:  # noqa: BLE001 - keep recovery results even if classification fails
            classification = {"error": f"{type(exc).__name__}: {exc}"}
            print(f"  [{name}] classification failed (recovery results kept): {exc}", flush=True)

    return {
        "dataset": name,
        "hf_id": spec["hf_id"],
        "description": spec["description"],
        "num_classes": classes,
        "class_names": class_names,
        "defects": defects,
        "conditions": results,
        "recovery_accuracy": results["dirty_optimized"]["summary"]["accuracy_mean"] - results["dirty_raw"]["summary"]["accuracy_mean"],
        "clean_delta_accuracy": results["clean_optimized"]["summary"]["accuracy_mean"] - results["clean_raw"]["summary"]["accuracy_mean"],
        "dirty_audit_quality": dirty_q,
        "classification": classification,
    }


def markdown_report(payload: dict[str, Any]) -> str:
    m = payload["method"]
    title = {
        "anthropic": "Claude benchmark (Anthropic)",
        "openai": "OpenAI benchmark",
        "openai_compatible": "Gateway benchmark (OpenAI-compatible)",
    }.get(m["provider"], f"{m['provider']} benchmark")
    lines = [
        f"# {title}",
        "",
        f"Provider: `{m['provider']}` | Model: `{m['model']}` | Base URL: `{m.get('base_url') or '-'}` | Seeds: {m['seeds']} | Epochs: {m['epochs']} |",
        f"Audit rows: {m['audit_rows']} | Classify sample: {m['classify_sample_size']}",
        "",
    ]
    if payload.get("smoke"):
        s = payload["smoke"]
        lines += ["## Smoke (synthetic, no HF download)", ""]
        for cond, res in s["recovery"].items():
            lines.append(f"- {cond}: rows={res['rows']} acc={res['summary']['accuracy_mean']:.4f}")
        c = s.get("classification") or {}
        if c.get("error"):
            lines += [f"- classification: failed ({c['error']})", ""]
        else:
            acc = f"{c['accuracy']:.1%}" if c.get("accuracy") is not None else "n/a"
            lines += [f"- classification: {c.get('correct', '-')}/{c.get('scored', '-')} ({acc})", ""]
    lines += ["## Recovery (dirty_optimized vs dirty_raw)", ""]
    lines += ["| Dataset | Classes | Dirty raw | Dirty optimized | Recovery | Clean delta |", "|---|---|---:|---:|---:|---:|"]
    for r in payload.get("results", []):
        c = r["conditions"]
        lines.append(
            f"| {r['dataset']} | {r['num_classes']} | {c['dirty_raw']['summary']['accuracy_mean']:.4f} | "
            f"{c['dirty_optimized']['summary']['accuracy_mean']:.4f} | {r['recovery_accuracy']:+.4f} | "
            f"{r['clean_delta_accuracy']:+.4f} |"
        )
    lines += ["", "## Provider usage per dataset", ""]
    lines += ["| Dataset | Audit tokens (in/out/total) | Audit score | Classify tokens |", "|---|---|---|---|"]
    for r in payload.get("results", []):
        q = r.get("dirty_audit_quality", {})
        u = q.get("usage", {})
        cl = (r.get("classification") or {}).get("usage", {})
        lines.append(
            f"| {r['dataset']} | {u.get('input_tokens', 0)}/{u.get('output_tokens', 0)}/{u.get('total_tokens', 0)} | "
            f"{q.get('audit_avg_score', 'n/a')} | {cl.get('total_tokens', cl.get('total', 0))} |"
        )
    lines.append("")
    return "\n".join(lines)


async def main(args: argparse.Namespace) -> None:
    if not args.mock:
        import os

        from dotenv import load_dotenv

        load_dotenv()
        _configure_gateway(args.provider, args.base_url)
        if args.provider == "anthropic" and not os.getenv("ANTHROPIC_API_KEY"):
            raise SystemExit("ANTHROPIC_API_KEY is not set (see .env). Re-run with --mock for an offline check.")
        if args.provider == "openai_compatible" and not args.base_url and not os.getenv("OPENAI_COMPATIBLE_BASE_URL"):
            raise SystemExit("--base-url (or OPENAI_COMPATIBLE_BASE_URL) is required with --provider openai_compatible.")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results.json"
    report_path = output_dir / "REPORT.md"

    from benchmark_buffdata import DATASETS

    payload: dict[str, Any] = {
        "method": {
            "provider": args.provider, "model": args.model, "base_url": args.base_url,
            "seeds": args.seeds,
            "epochs": args.epochs, "audit_rows": args.audit_rows,
            "classify_sample_size": args.classify_sample_size,
        },
        "results": [],
    }
    if args.smoke:
        payload["smoke"] = await run_smoke(args)
    else:
        selected = args.datasets or DEFAULT_DATASETS
        for offset, name in enumerate(selected):
            combo = await run_dataset(name, DATASETS[name], args, offset)
            payload["results"].append(combo)
            results_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            report_path.write_text(markdown_report(payload), encoding="utf-8")
            print(f"--- {name} done (checkpoint saved) ---", flush=True)

    results_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    report_path.write_text(markdown_report(payload), encoding="utf-8")
    print(f"Results: {results_path}")
    print(f"Report:  {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Claude-first BuffData benchmark (see module docstring).")
    parser.add_argument("--output-dir", default="benchmarks/results-claude")
    parser.add_argument("--provider", default=DEFAULT_PROVIDER)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--train-rows", type=int, default=1000)
    parser.add_argument("--test-rows", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 29])
    parser.add_argument("--audit-rows", type=int, default=20)
    parser.add_argument("--classify-sample-size", type=int, default=20)
    parser.add_argument("--skip-classification", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="Synthetic data, no HF download.")
    parser.add_argument("--mock", action="store_true", help="Zero API calls (offline plumbing check).")
    asyncio.run(main(parser.parse_args()))
