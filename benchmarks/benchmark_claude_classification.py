#!/usr/bin/env python3
"""Claude classification-correctness benchmark.

Answers: does BuffData's classify stage assign correct labels when powered
by Anthropic Claude instead of Gemini? For each dataset, strips labels from
a small stratified sample, runs the real DatasetClassifier path (the same
resolve_schema + classify_batch calls `buffdata classify` makes), and scores
predictions against the held-out ground truth.

This mirrors benchmarks/benchmark_classification_correctness.py but defaults
to `--provider anthropic --model claude-sonnet-5` and adds:
  --smoke  synthetic 12-row check with zero downloads (one live Claude call,
           or zero calls with --mock);
  --mock   offline plumbing check via allow_mock (no API key, no billing).

The full correctness script already accepts --provider/--model, so for large
multi-label sweeps prefer it directly:
  python benchmarks/benchmark_classification_correctness.py \\
    --provider anthropic --model claude-sonnet-5 --datasets ag_news

Requires ANTHROPIC_API_KEY unless --mock is given. Keep --sample-size small:
every ~10 rows is one structured Claude request.

External gateway (LiteLLM / company proxy serving Claude): the .env key that
401s against api.anthropic.com works here via the OpenAI-compatible path --
create_llm_client only sends AnthropicClient to api.anthropic.com, so a
third-party key must go through --provider openai_compatible instead:

  python benchmarks/benchmark_claude_classification.py --smoke \\
    --provider openai_compatible --base-url https://proxy.corp/v1 \\
    --model claude-sonnet-5

When --provider openai_compatible is used, --base-url is exported to
OPENAI_COMPATIBLE_BASE_URL and ANTHROPIC_API_KEY is forwarded to
OPENAI_COMPATIBLE_API_KEY when the latter is unset, so the existing .env key
is reused without copying it by hand. --model must exactly match whatever
model name the gateway exposes.
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

DEFAULT_PROVIDER = "anthropic"
DEFAULT_MODEL = "claude-sonnet-5"
DEFAULT_DATASETS = ["imdb", "ag_news", "emotion"]

_SMOKE_ROWS: dict[str, dict[str, Any]] = {
    "smoke_binary": {
        "kind": "binary",
        "classes": ["negative", "positive"],
        "rows": [
            ("I loved this movie, brilliant acting.", 1),
            ("Terrible film, I walked out.", 0),
            ("A wonderful, heartwarming story.", 1),
            ("Boring and predictable waste of time.", 0),
            ("Fantastic direction and great script.", 1),
            ("Awful dialogue, painful to watch.", 0),
        ],
    },
    "smoke_multi": {
        "kind": "multi-class",
        "classes": ["sports", "tech", "politics"],
        "rows": [
            ("The striker scored twice in the final.", 0),
            ("The new chip doubles battery life.", 1),
            ("Parliament passed the budget bill.", 2),
            ("Overtime win sends team to playoffs.", 0),
            ("Cloud update adds encryption by default.", 1),
            ("Senators debated the tax amendment.", 2),
        ],
    },
}


class _FakeClassifyClient:
    """Offline stand-in for --mock: exercises the full DatasetClassifier code
    path with zero network calls. Always predicts the first class, so accuracy
    is deterministic but clearly labeled mock (see `mock: true` in output).
    The repo's own allow_mock cannot be used here: _mock_structured has no
    BatchClassificationResponse case and returns an empty model_construct()."""

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


def _configure_gateway(provider: str, base_url: str | None) -> None:
    """Forward gateway settings into the env vars the client layer reads.

    create_llm_client(provider="openai_compatible") resolves its endpoint from
    --base-url or OPENAI_COMPATIBLE_BASE_URL and its key from
    OPENAI_COMPATIBLE_API_KEY. Reuse the existing .env ANTHROPIC_API_KEY when
    the gateway-specific key is unset so external-proxy Claude keys work
    without copying secrets by hand.
    """
    import os

    if base_url:
        os.environ["OPENAI_COMPATIBLE_BASE_URL"] = base_url
    if provider == "openai_compatible" and not os.getenv("OPENAI_COMPATIBLE_API_KEY"):
        fallback = os.getenv("ANTHROPIC_API_KEY")
        if fallback:
            os.environ["OPENAI_COMPATIBLE_API_KEY"] = fallback


async def classify_and_score(
    texts: list[str],
    true_names: list[str],
    class_names: list[str],
    mode: str,
    provider: str,
    model: str,
    allow_mock: bool = False,
    base_url: str | None = None,
) -> dict[str, Any]:
    from buffdata.engine.limiter import AsyncRateLimiter
    from buffdata.models.schemas import DatasetItem
    from buffdata.optimizers.classifier import DatasetClassifier

    items = [DatasetItem.from_dict({"text": text}) for text in texts]
    if allow_mock:
        client = _FakeClassifyClient(provider=provider, model=model, classes=class_names)
    else:
        from buffdata.engine.client import create_llm_client

        # Always pass model explicitly: BUFFDATA_DEFAULT_MODEL in .env would
        # otherwise override the provider default inside create_llm_client.
        _configure_gateway(provider, base_url)
        client = create_llm_client(provider, model=model, base_url=base_url)
    limiter = AsyncRateLimiter(max_rpm=60, concurrency=5)
    classifier = DatasetClassifier(client, limiter, model)

    started = time.perf_counter()
    schema = await classifier.resolve_schema(items, mode=mode, classes=class_names, sample_size=len(items))
    classified = await classifier.classify_batch(items, schema)
    elapsed = time.perf_counter() - started

    correct = scored = errored = 0
    errors: list[str] = []
    for item, true_name in zip(classified, true_names):
        err = item.metadata.get("classification_error")
        if err:
            errored += 1
            if len(errors) < 2:
                errors.append(str(err)[:200])
            continue
        predicted = item.labels if isinstance(item.labels, str) else None
        if predicted is None:
            errored += 1
            continue
        scored += 1
        if predicted == true_name:
            correct += 1
    return {
        "provider": provider, "model": model, "mode": mode, "mock": allow_mock,
        "sampled": len(texts), "scored": scored, "errored": errored,
        "error_samples": errors, "correct": correct,
        "accuracy": correct / scored if scored else None,
        "elapsed_s": round(elapsed, 1), "usage": dict(client.usage),
    }


async def run_smoke(provider: str, model: str, allow_mock: bool, base_url: str | None = None) -> list[dict[str, Any]]:
    outcomes = []
    for name, spec in _SMOKE_ROWS.items():
        texts = [text for text, _ in spec["rows"]]
        true_names = [spec["classes"][label] for _, label in spec["rows"]]
        outcome = await classify_and_score(
            texts, true_names, spec["classes"], spec["kind"], provider, model, allow_mock, base_url,
        )
        outcome["dataset"] = name
        print(f"  {name}: {outcome['correct']}/{outcome['scored']} correct -> {outcome}", flush=True)
        outcomes.append(outcome)
    return outcomes


async def run_hf_dataset(
    name: str, spec: dict[str, Any], provider: str, model: str, sample_size: int,
    base_url: str | None = None,
) -> dict[str, Any]:
    from benchmark_scale_matrix import DATASETS as SCALAR_DATASETS, load_normalized  # noqa: E402

    train_split, _ = load_normalized(spec)
    rows = list(train_split)
    from benchmark_classification_correctness import _class_names_scalar  # noqa: E402

    class_names = _class_names_scalar(spec, rows[:200])
    kind = spec["kind"]
    rng = random.Random(999)
    sample = rng.sample(rows, min(sample_size, len(rows)))
    texts = [row["text"] for row in sample]
    true_names = [class_names[int(row["label"])] for row in sample]
    mode = "binary" if kind == "binary" else "multi-class"
    outcome = await classify_and_score(texts, true_names, class_names, mode, provider, model, False, base_url)
    outcome.update({"dataset": name, "kind": kind, "num_classes": len(class_names)})
    return outcome


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

    results: list[dict[str, Any]] = []
    if args.smoke:
        print("=== smoke: synthetic binary + multi-class (no downloads) ===", flush=True)
        results.extend(await run_smoke(args.provider, args.model, args.mock, args.base_url))
    else:
        from benchmark_scale_matrix import DATASETS as SCALAR_DATASETS

        selected = args.datasets or DEFAULT_DATASETS
        for name in selected:
            if name not in SCALAR_DATASETS:
                print(f"skip {name}: not in scale-matrix catalog", flush=True)
                continue
            print(f"=== {name} ===", flush=True)
            try:
                outcome = await run_hf_dataset(name, SCALAR_DATASETS[name], args.provider, args.model, args.sample_size, args.base_url)
            except Exception as exc:  # noqa: BLE001 - record per-dataset failures, keep going
                outcome = {"dataset": name, "error": f"{type(exc).__name__}: {exc}"}
            print(f"  -> {outcome}", flush=True)
            results.append(outcome)
            results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    title = {
        "anthropic": "Claude classification-correctness",
        "openai": "OpenAI classification-correctness",
        "openai_compatible": "Gateway classification-correctness",
    }.get(args.provider, f"{args.provider} classification-correctness")
    lines = [
        f"# {title}", "",
        f"Provider: `{args.provider}` | Model: `{args.model}` | Base URL: `{args.base_url or '-'}` | Smoke: {args.smoke} | Mock: {args.mock}", "",
        "| Dataset | Mode | Scored | Correct | Accuracy |", "|---|---|---:|---:|---:|",
    ]
    for r in results:
        acc = f"{r['accuracy']:.1%}" if r.get("accuracy") is not None else "n/a"
        lines.append(f"| {r.get('dataset')} | {r.get('mode', r.get('kind', '-'))} | {r.get('scored', '-')} | {r.get('correct', '-')} | {acc} |")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Results: {results_path}")
    print(f"Report:  {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Claude classification correctness (see module docstring).")
    parser.add_argument("--output-dir", default="benchmarks/results-claude-classification")
    parser.add_argument("--provider", default=DEFAULT_PROVIDER)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--sample-size", type=int, default=20)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--mock", action="store_true")
    asyncio.run(main(parser.parse_args()))
