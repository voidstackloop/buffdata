import json
from pathlib import Path
from typing import Any, Dict, Generator, Iterable, List, Optional, Union
import pandas as pd
from buffdata.models.schemas import DatasetFormat, DatasetItem


def detect_format(sample_dict: Dict[str, Any]) -> DatasetFormat:
    """Infer DatasetFormat from sample dictionary keys."""
    if "messages" in sample_dict and isinstance(sample_dict["messages"], list):
        return DatasetFormat.CHAT
    if "chosen" in sample_dict and "rejected" in sample_dict:
        return DatasetFormat.DPO
    if "instruction" in sample_dict or "output" in sample_dict:
        return DatasetFormat.ALPACA
    if "text" in sample_dict:
        return DatasetFormat.RAW
    return DatasetFormat.CUSTOM


def read_dataset(file_path: Union[str, Path], max_rows: Optional[int] = None) -> List[DatasetItem]:
    """Read dataset from JSONL, JSON, Parquet, or CSV file."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    suffix = path.suffix.lower()
    items: List[DatasetItem] = []

    if suffix == ".jsonl":
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if max_rows and i >= max_rows:
                    break
                line = line.strip()
                if line:
                    data = json.loads(line)
                    items.append(DatasetItem.from_dict(data))

    elif suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                for i, row in enumerate(data):
                    if max_rows and i >= max_rows:
                        break
                    items.append(DatasetItem.from_dict(row))
            elif isinstance(data, dict):
                items.append(DatasetItem.from_dict(data))

    elif suffix in (".parquet", ".pq"):
        df = pd.read_parquet(path)
        if max_rows:
            df = df.head(max_rows)
        for _, row in df.iterrows():
            items.append(DatasetItem.from_dict(row.to_dict()))

    elif suffix == ".csv":
        df = pd.read_csv(path)
        if max_rows:
            df = df.head(max_rows)
        for _, row in df.iterrows():
            items.append(DatasetItem.from_dict(row.to_dict()))

    else:
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if max_rows and i >= max_rows:
                    break
                line = line.strip()
                if line:
                    items.append(DatasetItem.from_dict(json.loads(line)))

    return items


def iter_dataset(file_path: Union[str, Path]) -> Generator[DatasetItem, None, None]:
    """Stream dataset items lazily line-by-line."""
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix == ".jsonl":
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield DatasetItem.from_dict(json.loads(line))
    else:
        for item in read_dataset(path):
            yield item


def write_dataset(
    items: Union[List[DatasetItem], Iterable[DatasetItem]],
    file_path: Union[str, Path],
    format_override: Optional[str] = None,
):
    """Write dataset items to destination file (.jsonl, .json, .parquet, .csv)."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = (format_override or path.suffix).lower()
    if not suffix.startswith("."):
        suffix = f".{suffix}"

    raw_items = [it.to_dict() if isinstance(it, DatasetItem) else it for it in items]

    if suffix == ".jsonl":
        with open(path, "w", encoding="utf-8") as f:
            for row in raw_items:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    elif suffix == ".json":
        with open(path, "w", encoding="utf-8") as f:
            json.dump(raw_items, f, ensure_ascii=False, indent=2)

    elif suffix in (".parquet", ".pq"):
        df = pd.DataFrame(raw_items)
        df.to_parquet(path, index=False)

    elif suffix == ".csv":
        df = pd.DataFrame(raw_items)
        df.to_csv(path, index=False)

    else:
        with open(path, "w", encoding="utf-8") as f:
            for row in raw_items:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
