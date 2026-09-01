import pytest
from typer.testing import CliRunner
from pathlib import Path
from buffdata.cli.main import app
from buffdata.models.formats import write_dataset
from buffdata.models.schemas import DatasetItem, DatasetFormat

runner = CliRunner()

def test_cli_stats(tmp_path: Path):
    sample_file = tmp_path / "sample.jsonl"
    items = [
        DatasetItem(format=DatasetFormat.ALPACA, instruction="Test instruction", output="Test response")
    ]
    write_dataset(items, sample_file)

    result = runner.invoke(app, ["stats", str(sample_file)])
    assert result.exit_code == 0
    assert "BuffData Optimization Audit Report" in result.output

def test_cli_dedup(tmp_path: Path):
    sample_file = tmp_path / "sample.jsonl"
    out_file = tmp_path / "out.jsonl"
    items = [
        DatasetItem(id="1", format=DatasetFormat.ALPACA, instruction="Do task", output="Task done"),
        DatasetItem(id="2", format=DatasetFormat.ALPACA, instruction="Do task", output="Task done"),
    ]
    write_dataset(items, sample_file)

    result = runner.invoke(app, ["dedup", str(sample_file), "-o", str(out_file), "--method", "exact"])
    assert result.exit_code == 0
    assert out_file.exists()
