from typer.testing import CliRunner

from buffdata.cli.main import app
from buffdata.engine.client import LLMProvider
from buffdata.evaluation.accuracy_gate import evaluate_accuracy_gain
from buffdata.models.formats import read_dataset, write_dataset
from buffdata.models.schemas import DatasetItem


class OfflineClient:
    provider = LLMProvider.GEMINI
    default_model = "offline"
    usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}


def _separable_rows(*, flipped: bool):
    rows = []
    for index in range(40):
        rows.append(DatasetItem.from_dict({
            "text": f"apple orchard red fruit {index}",
            "label": 1 if flipped else 0,
        }))
        rows.append(DatasetItem.from_dict({
            "text": f"banana market yellow fruit {index}",
            "label": 0 if flipped else 1,
        }))
    return rows


def test_accuracy_gate_accepts_only_positive_candidate_across_every_seed():
    original = _separable_rows(flipped=True)
    candidate = _separable_rows(flipped=False)
    validation = _separable_rows(flipped=False)

    result = evaluate_accuracy_gain(
        original,
        candidate,
        validation,
        seeds=[17, 29],
        epochs=4,
    )

    assert result["accepted"] is True
    assert result["accuracy_gain"] > 0
    assert all(gain > 0 for gain in result["per_seed_accuracy_gain"])


def test_accuracy_gate_rejects_parity():
    original = _separable_rows(flipped=False)

    result = evaluate_accuracy_gain(
        original,
        original,
        original,
        seeds=[17],
        epochs=2,
    )

    assert result["accepted"] is False
    assert result["accuracy_gain"] == 0


def _fake_gate(accepted):
    accuracy = 0.9 if accepted else 0.8
    return {
        "accepted": accepted,
        "criterion": "test",
        "minimum_gain": 0.0,
        "seeds": [17, 29, 43],
        "epochs": 1,
        "original_rows_evaluated": 2,
        "candidate_rows_evaluated": 2,
        "validation_rows": 2,
        "original_runs": [{"accuracy": 0.8, "train_seconds": 0.0}] * 3,
        "candidate_runs": [{"accuracy": accuracy, "train_seconds": 0.0}] * 3,
        "per_seed_accuracy_gain": [accuracy - 0.8] * 3,
        "original_accuracy_mean": 0.8,
        "candidate_accuracy_mean": accuracy,
        "accuracy_gain": accuracy - 0.8,
    }


def test_cli_publishes_non_json_output_only_after_positive_gate(tmp_path, monkeypatch):
    source = tmp_path / "train.csv"
    validation = tmp_path / "validation.tsv"
    output = tmp_path / "accepted.arrow"
    rows = _separable_rows(flipped=False)[:4]
    write_dataset(rows, source)
    write_dataset(rows, validation)
    monkeypatch.setattr("buffdata.engine.pipeline.create_llm_client", lambda **kwargs: OfflineClient())
    monkeypatch.setattr(
        "buffdata.evaluation.accuracy_gate.evaluate_accuracy_gain",
        lambda *args, **kwargs: _fake_gate(True),
    )

    result = CliRunner().invoke(app, [
        "optimize", str(source), "--output", str(output),
        "--quality-mode", "off", "--classification", "off",
        "--accuracy-contract", "strict", "--require-positive-gain",
        "--validation-file", str(validation),
    ])

    assert result.exit_code == 0, result.output
    assert output.exists()
    assert len(read_dataset(output)) == len(rows)


def test_cli_failed_gate_writes_no_output(tmp_path, monkeypatch):
    source = tmp_path / "train.parquet"
    validation = tmp_path / "validation.csv"
    output = tmp_path / "rejected.hf"
    rows = _separable_rows(flipped=False)[:4]
    write_dataset(rows, source)
    write_dataset(rows, validation)
    monkeypatch.setattr("buffdata.engine.pipeline.create_llm_client", lambda **kwargs: OfflineClient())
    monkeypatch.setattr(
        "buffdata.evaluation.accuracy_gate.evaluate_accuracy_gain",
        lambda *args, **kwargs: _fake_gate(False),
    )

    result = CliRunner().invoke(app, [
        "optimize", str(source), "--output", str(output),
        "--quality-mode", "off", "--classification", "off",
        "--accuracy-contract", "strict", "--require-positive-gain",
        "--validation-file", str(validation),
    ])

    assert result.exit_code == 2
    assert not output.exists()
    assert "No optimized output was published" in result.output
