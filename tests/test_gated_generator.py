import json

import pytest
from typer.testing import CliRunner

from buffdata.cli.main import app
from buffdata.engine.client import LLMProvider
from buffdata.evaluation.accuracy_gate import diagnose_candidate_errors, evaluate_accuracy_gain
from buffdata.models.formats import write_dataset
from buffdata.models.schemas import AugmentationResult, BatchAugmentationResponse, DatasetItem
from buffdata.optimizers.gated_generator import AccuracyGatedGenerator, build_gated_generation_prompt


def _separable_rows(*, flipped: bool, count: int = 40):
    rows = []
    for index in range(count):
        rows.append(DatasetItem.from_dict({
            "text": f"apple orchard red fruit {index}",
            "label": 1 if flipped else 0,
        }))
        rows.append(DatasetItem.from_dict({
            "text": f"banana market yellow fruit {index}",
            "label": 0 if flipped else 1,
        }))
    return rows


# --- direct tests of the accuracy_gate.py extensions (real, cheap training) ---

def test_relative_gain_accepts_large_improvement_and_rejects_parity():
    original = _separable_rows(flipped=True)
    candidate = _separable_rows(flipped=False)
    validation = _separable_rows(flipped=False)

    accepted = evaluate_accuracy_gain(
        original, candidate, validation, seeds=[17, 29], epochs=4, minimum_relative_gain=0.10,
    )
    assert accepted["accepted"] is True
    assert accepted["minimum_relative_gain"] == 0.10

    rejected = evaluate_accuracy_gain(
        original, original, validation, seeds=[17], epochs=2, minimum_relative_gain=0.10,
    )
    assert rejected["accepted"] is False


def test_relative_and_absolute_gain_are_mutually_exclusive():
    rows = _separable_rows(flipped=False, count=4)
    with pytest.raises(ValueError):
        evaluate_accuracy_gain(rows, rows, rows, seeds=[17], minimum_gain=0.05, minimum_relative_gain=0.10)


def test_diagnose_candidate_errors_reports_every_label():
    candidate = _separable_rows(flipped=False)
    validation = _separable_rows(flipped=False)

    diagnostics = diagnose_candidate_errors(candidate, validation, seed=17, epochs=4)

    assert set(diagnostics["per_label_accuracy"]) == {"0", "1"}
    assert set(diagnostics["misclassified_examples"]) == {"0", "1"}
    for accuracy in diagnostics["per_label_accuracy"].values():
        assert 0.0 <= accuracy <= 1.0


def test_build_gated_generation_prompt_empty_before_first_diagnosis():
    assert build_gated_generation_prompt([], {}, {}, {}) == ""


def test_build_gated_generation_prompt_surfaces_weak_classes_and_examples():
    prompt = build_gated_generation_prompt(
        ["1"],
        {"1": ["a misclassified banana review"]},
        {"0": 0.7, "1": 0.3},
        {"0": 0.5, "1": 0.5},
    )
    assert "WEAK CLASSES" in prompt
    assert "1" in prompt
    assert "a misclassified banana review" in prompt
    assert "current 30.0% -> target 50.0%" in prompt


# --- orchestration tests: stub the gate so loop control is verified deterministically ---

class RecordingAugmentClient:
    provider = LLMProvider.GEMINI
    default_model = "fake"
    usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    def __init__(self):
        self.received_prompts: list[str] = []
        self.calls = 0

    async def generate_structured_async(self, prompt, response_schema, **kwargs):
        self.received_prompts.append(prompt)
        self.calls += 1
        ids = [line.removeprefix("ID: ") for line in prompt.splitlines() if line.startswith("ID: ")]
        return BatchAugmentationResponse(results=[
            AugmentationResult(
                id=item_id,
                variations=[f"generated variation {self.calls} for {item_id} apple fruit stand"],
            )
            for item_id in ids
        ])


def _stub_gate(sequence):
    calls = {"n": 0}

    def _fake_evaluate_accuracy_gain(original, candidate, validation, **kwargs):
        index = min(calls["n"], len(sequence) - 1)
        calls["n"] += 1
        accepted, accuracy, gain = sequence[index]
        return {
            "accepted": accepted,
            "original_accuracy_mean": 0.5,
            "candidate_accuracy_mean": accuracy,
            "accuracy_gain": gain,
        }

    return _fake_evaluate_accuracy_gain, calls


def _stub_diagnostics():
    def _fake_diagnose(candidate, validation, **kwargs):
        return {
            "per_label_accuracy": {"0": 0.4, "1": 0.9},
            "misclassified_examples": {"0": ["a hard example about apples"], "1": []},
        }

    return _fake_diagnose


@pytest.mark.asyncio
async def test_gated_generator_stops_as_soon_as_gate_passes(monkeypatch):
    monkeypatch.setattr(
        "buffdata.optimizers.gated_generator.diagnose_candidate_errors", _stub_diagnostics()
    )
    fake_gate, calls = _stub_gate([
        (False, 0.5, 0.0),
        (True, 0.65, 0.15),
    ])
    monkeypatch.setattr("buffdata.optimizers.gated_generator.evaluate_accuracy_gain", fake_gate)

    client = RecordingAugmentClient()
    items = _separable_rows(flipped=False, count=5)
    validation = _separable_rows(flipped=False, count=5)
    generator = AccuracyGatedGenerator(client=client)

    candidate, report = await generator.generate(
        items, validation, min_relative_gain=0.10, max_iterations=5,
    )

    assert report.passed is True
    assert report.iterations_used == 2
    assert len(report.per_iteration) == 2
    assert calls["n"] == 2
    # Round 2's prompt should carry the weak-class feedback from round 1's diagnosis.
    assert any("WEAK CLASSES" in prompt for prompt in client.received_prompts)
    assert len(candidate) >= len(items)


@pytest.mark.asyncio
async def test_gated_generator_reports_honest_failure_after_budget(monkeypatch):
    monkeypatch.setattr(
        "buffdata.optimizers.gated_generator.diagnose_candidate_errors", _stub_diagnostics()
    )
    fake_gate, calls = _stub_gate([(False, 0.5, 0.0)])
    monkeypatch.setattr("buffdata.optimizers.gated_generator.evaluate_accuracy_gain", fake_gate)

    client = RecordingAugmentClient()
    items = _separable_rows(flipped=False, count=5)
    validation = _separable_rows(flipped=False, count=5)
    generator = AccuracyGatedGenerator(client=client)

    candidate, report = await generator.generate(
        items, validation, min_relative_gain=0.10, max_iterations=3,
    )

    assert report.passed is False
    assert report.iterations_used == 3
    assert len(report.per_iteration) == 3
    assert calls["n"] == 3
    # diagnose is only useful before another round runs, so it should not fire on the last round.
    assert candidate  # best-so-far candidate is still returned, never empty


# --- CLI smoke test: real (tiny, cheap) end-to-end path through generate_cmd ---

def test_cli_generate_runs_end_to_end_and_writes_gate_report(tmp_path, monkeypatch):
    source = tmp_path / "train.jsonl"
    validation = tmp_path / "validation.jsonl"
    output = tmp_path / "generated.jsonl"
    items = _separable_rows(flipped=False, count=4)
    write_dataset(items, source)
    write_dataset(items, validation)
    monkeypatch.setattr("buffdata.cli.main.create_llm_client", lambda *a, **k: RecordingAugmentClient())

    result = CliRunner().invoke(app, [
        "generate", str(source),
        "-o", str(output),
        "--validation-file", str(validation),
        "--max-iterations", "1",
        "--accuracy-epochs", "2",
        "--accuracy-seeds", "17",
    ])

    assert result.exit_code == 0, result.output
    assert output.exists()
    gate_report = tmp_path / "generated.accuracy_gate.json"
    assert gate_report.exists()
    payload = json.loads(gate_report.read_text(encoding="utf-8"))
    assert payload["iterations_used"] == 1
    assert len(payload["per_iteration"]) == 1


def test_cli_generate_refuses_to_overwrite_existing_output(tmp_path, monkeypatch):
    source = tmp_path / "train.jsonl"
    validation = tmp_path / "validation.jsonl"
    output = tmp_path / "generated.jsonl"
    items = _separable_rows(flipped=False, count=4)
    write_dataset(items, source)
    write_dataset(items, validation)
    output.write_text("stale", encoding="utf-8")
    monkeypatch.setattr("buffdata.cli.main.create_llm_client", lambda *a, **k: RecordingAugmentClient())

    result = CliRunner().invoke(app, [
        "generate", str(source), "-o", str(output), "--validation-file", str(validation),
    ])

    assert result.exit_code != 0
    assert output.read_text(encoding="utf-8") == "stale"
