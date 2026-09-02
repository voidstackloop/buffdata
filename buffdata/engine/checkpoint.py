import json
from pathlib import Path
from typing import Dict, List, Optional, Set, Union
from buffdata.models.schemas import DatasetItem
from buffdata.security.permissions import restrict_to_owner


class CheckpointManager:
    """Tracks processed items to enable graceful resume on interrupted runs."""

    def __init__(self, checkpoint_path: Union[str, Path]):
        self.path = Path(checkpoint_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.processed_ids: Set[str] = set()
        self._load_existing()

    def _load_existing(self):
        if self.path.exists():
            with open(self.path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            data = json.loads(line)
                            item_id = data.get("id") or data.get("_buffdata_id")
                            if item_id:
                                self.processed_ids.add(str(item_id))
                        except Exception:
                            continue

    def is_processed(self, item_id: str) -> bool:
        return str(item_id) in self.processed_ids

    def save_item(self, item: DatasetItem):
        """Append processed item to checkpoint file."""
        data = item.to_dict()
        data["_buffdata_id"] = str(item.id)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")
        restrict_to_owner(self.path)
        self.processed_ids.add(str(item.id))

    def load_all(self) -> List[DatasetItem]:
        """Load all saved items from checkpoint."""
        if not self.path.exists():
            return []
        items = []
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    items.append(DatasetItem.from_dict(json.loads(line)))
        return items

    def cleanup(self):
        """Remove checkpoint file when run completes cleanly."""
        if self.path.exists():
            self.path.unlink()
