import json
import pytest
from pathlib import Path
from buffdata.models.formats import read_dataset, write_dataset, detect_format
from buffdata.models.schemas import DatasetItem, DatasetFormat

def test_detect_format():
    assert detect_format({"messages": []}) == DatasetFormat.CHAT
    assert detect_format({"chosen": "a", "rejected": "b"}) == DatasetFormat.DPO
    assert detect_format({"instruction": "do x", "output": "y"}) == DatasetFormat.ALPACA
    assert detect_format({"text": "sample"}) == DatasetFormat.RAW

def test_read_write_jsonl(tmp_path: Path):
    file = tmp_path / "test.jsonl"
    items = [
        DatasetItem(format=DatasetFormat.ALPACA, instruction=f"Task {i}", output=f"Answer {i}")
        for i in range(5)
    ]
    write_dataset(items, file)
    assert file.exists()

    loaded = read_dataset(file)
    assert len(loaded) == 5
    assert loaded[0].instruction == "Task 0"
    assert loaded[0].output == "Answer 0"
