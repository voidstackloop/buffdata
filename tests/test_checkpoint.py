import pytest
from pathlib import Path
from buffdata.engine.checkpoint import CheckpointManager
from buffdata.models.schemas import DatasetItem, DatasetFormat

def test_checkpoint_lifecycle(tmp_path: Path):
    ckpt_file = tmp_path / "ckpt.jsonl"
    mgr = CheckpointManager(ckpt_file)
    
    item1 = DatasetItem(id="item-1", format=DatasetFormat.ALPACA, instruction="p1", output="r1")
    item2 = DatasetItem(id="item-2", format=DatasetFormat.ALPACA, instruction="p2", output="r2")
    
    assert not mgr.is_processed("item-1")
    mgr.save_item(item1)
    assert mgr.is_processed("item-1")
    assert not mgr.is_processed("item-2")

    # Reload new manager instance to verify persistence
    mgr2 = CheckpointManager(ckpt_file)
    assert mgr2.is_processed("item-1")
    assert not mgr2.is_processed("item-2")
    items = mgr2.load_all()
    assert len(items) == 1
    assert items[0].id == "item-1"
