"""Dataset readers and writers for common AI-training data containers."""

from __future__ import annotations

import gzip
import io
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tempfile
from typing import Any, Dict, Generator, Iterable, List, Optional, Union
import uuid

import orjson
import polars as pl
import yaml

from buffdata.models.schemas import DatasetFormat, DatasetItem


SUPPORTED_FORMATS = (
    "jsonl", "ndjson", "json", "jsonl.gz", "ndjson.gz", "json.gz",
    "csv", "csv.gz", "tsv", "tsv.gz", "parquet", "pq",
    "arrow", "feather", "ipc", "yaml", "yml", "txt", "txt.gz", "hf",
)


def detect_format(sample_dict: Dict[str, Any]) -> DatasetFormat:
    if "messages" in sample_dict and isinstance(sample_dict["messages"], list):
        return DatasetFormat.CHAT
    if "chosen" in sample_dict and "rejected" in sample_dict:
        return DatasetFormat.DPO
    if "instruction" in sample_dict or "output" in sample_dict:
        return DatasetFormat.ALPACA
    if "text" in sample_dict:
        return DatasetFormat.RAW
    return DatasetFormat.CUSTOM


def _normalize_format(value: str) -> str:
    normalized = value.lower().strip()
    while normalized.startswith("."):
        normalized = normalized[1:]
    normalized = {"hfds": "hf", "huggingface": "hf"}.get(normalized, normalized)
    if normalized not in SUPPORTED_FORMATS:
        raise ValueError(
            f"Unsupported dataset format '{value}'. Supported formats: "
            + ", ".join(SUPPORTED_FORMATS)
        )
    return normalized


def _path_format(path: Path, override: Optional[str] = None) -> str:
    if override:
        return _normalize_format(override)
    if path.is_dir():
        return "hf"
    name = path.name.lower()
    for compound in ("jsonl.gz", "ndjson.gz", "json.gz", "csv.gz", "tsv.gz", "txt.gz"):
        if name.endswith(f".{compound}"):
            return compound
    return _normalize_format(path.suffix.lower().lstrip("."))


# --- Cloud storage (S3 / GCS / ADLS) ---------------------------------------------------
#
# read_dataset/write_dataset/write_dataset_atomic accept an s3://, gs://, gcs://, az://,
# abfs://, or adl:// URL exactly where they accept a local path. Every format's actual row
# parsing/serialization (_rows_from_json_container, _decode_tabular_rows, _tabular_rows,
# DatasetItem itself) is already source-agnostic, so the cloud path below only has to get
# bytes in and out via fsspec -- it deliberately never touches the local-path branches
# above, so existing local behavior can't regress from this addition. Hugging Face
# save_to_disk directories and Snowflake stages are out of scope for now: HF datasets are
# a multi-file tree that needs real directory sync, not a single-object open/write, and
# Snowflake stages are a different connector model entirely (not a generic file URL).
CLOUD_SCHEMES = ("s3", "gs", "gcs", "az", "abfs", "abfss", "adl")


def _is_cloud_url(file_path: Union[str, Path]) -> bool:
    if isinstance(file_path, Path):
        return False
    return "://" in file_path and file_path.split("://", 1)[0].lower() in CLOUD_SCHEMES


def _cloud_format(url: str, override: Optional[str] = None) -> str:
    if override:
        normalized = _normalize_format(override)
        if normalized == "hf":
            raise ValueError("Hugging Face datasets on cloud storage are not supported yet; sync locally first.")
        return normalized
    key = url.split("://", 1)[1]
    name = PurePosixPath(key).name.lower()
    if not name or url.endswith("/"):
        raise ValueError(
            "Cloud dataset URLs must point at a single object (e.g. s3://bucket/train.jsonl), "
            "not a directory -- Hugging Face save_to_disk trees on cloud storage aren't supported yet."
        )
    for compound in ("jsonl.gz", "ndjson.gz", "json.gz", "csv.gz", "tsv.gz", "txt.gz"):
        if name.endswith(f".{compound}"):
            return compound
    return _normalize_format(PurePosixPath(name).suffix.lstrip("."))


def _cloud_storage_options(url: str) -> dict[str, Any]:
    """Credentials come from each provider's own default chain (boto3, google-auth,
    DefaultAzureCredential) picked up automatically by s3fs/gcsfs/adlfs -- nothing to pass
    explicitly in the common case. The one override supported here is a custom S3-compatible
    endpoint (MinIO, on-prem object storage, etc.) via BUFFDATA_S3_ENDPOINT_URL, since that
    has no other way to reach fsspec.
    """
    scheme = url.split("://", 1)[0].lower()
    if scheme == "s3":
        endpoint = os.getenv("BUFFDATA_S3_ENDPOINT_URL")
        if endpoint:
            return {"client_kwargs": {"endpoint_url": endpoint}}
    return {}


def _open_fsspec(url: str, mode: str):
    from buffdata.security.policy import check_cloud
    check_cloud(url)
    try:
        import fsspec
    except ImportError as exc:
        raise ImportError(
            "Install fsspec plus the matching filesystem package (s3fs for s3://, gcsfs for "
            "gs://, adlfs for az:///abfs://) -- pip install buffdata[enterprise] -- to read or "
            "write cloud storage paths."
        ) from exc
    binary_mode = mode.replace("t", "").replace("b", "") + "b"
    return fsspec.open(url, binary_mode, **_cloud_storage_options(url)).open()


def _cloud_open(url: str, mode: str, compressed: bool):
    """A file-like object for a cloud object, matching open()/gzip.open()'s text/binary
    contract closely enough for the existing per-format parsing/serialization code below to
    use it exactly like a local file handle.
    """
    handle = _open_fsspec(url, mode)
    if compressed:
        handle = gzip.GzipFile(fileobj=handle, mode="rb" if "r" in mode else "wb")
    if "b" not in mode:
        return io.TextIOWrapper(handle, encoding="utf-8")
    return handle


def _limit(rows: list[dict[str, Any]], max_rows: Optional[int]) -> list[dict[str, Any]]:
    return rows if max_rows is None else rows[:max_rows]


def _decode_tabular_value(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped or stripped[0] not in "[{" or stripped[-1] not in "]}":
        return value
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return value


def _decode_tabular_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{key: _decode_tabular_value(value) for key, value in row.items()} for row in rows]


def _tabular_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            key: json.dumps(value, ensure_ascii=False, separators=(",", ":"))
            if isinstance(value, (dict, list))
            else value
            for key, value in row.items()
        }
        for row in rows
    ]


def _rows_from_json_container(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, list):
        if not all(isinstance(row, dict) for row in data):
            raise ValueError("JSON dataset arrays must contain objects")
        return list(data)
    if not isinstance(data, dict):
        raise ValueError("JSON dataset must be an object or an array of objects")
    for key in ("data", "records", "items"):
        value = data.get(key)
        if isinstance(value, list) and all(isinstance(row, dict) for row in value):
            return list(value)
    if data and all(isinstance(value, list) for value in data.values()):
        if all(all(isinstance(row, dict) for row in value) for value in data.values()):
            rows: list[dict[str, Any]] = []
            for split, split_rows in data.items():
                rows.extend({"_buffdata_split": split, **row} for row in split_rows)
            return rows
        lengths = {len(value) for value in data.values()}
        if len(lengths) == 1:
            return [dict(zip(data, values)) for values in zip(*data.values())]
    return [data]


def _read_huggingface(path: Path, max_rows: Optional[int]) -> list[dict[str, Any]]:
    from datasets import DatasetDict, load_from_disk

    loaded = load_from_disk(str(path))
    rows: list[dict[str, Any]] = []
    if isinstance(loaded, DatasetDict):
        for split, dataset in loaded.items():
            for row in dataset:
                rows.append({"_buffdata_split": split, **dict(row)})
                if max_rows is not None and len(rows) >= max_rows:
                    return rows
    else:
        for row in loaded:
            rows.append(dict(row))
            if max_rows is not None and len(rows) >= max_rows:
                break
    return rows


def _read_dataset_cloud(url: str, max_rows: Optional[int]) -> List[DatasetItem]:
    dataset_format = _cloud_format(url)

    if dataset_format in {"jsonl", "ndjson", "jsonl.gz", "ndjson.gz"}:
        rows = []
        with _cloud_open(url, "rt", compressed=dataset_format.endswith(".gz")) as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"NDJSON row {line_number} must be an object")
                rows.append(value)
                if max_rows is not None and len(rows) >= max_rows:
                    break
    elif dataset_format in {"json", "json.gz"}:
        with _cloud_open(url, "rt", compressed=dataset_format.endswith(".gz")) as handle:
            rows = _limit(_rows_from_json_container(json.load(handle)), max_rows)
    elif dataset_format in {"yaml", "yml"}:
        with _cloud_open(url, "rt", compressed=False) as handle:
            rows = _limit(_rows_from_json_container(yaml.safe_load(handle)), max_rows)
    elif dataset_format in {"parquet", "pq"}:
        with _cloud_open(url, "rb", compressed=False) as handle:
            frame = pl.read_parquet(handle)
        rows = frame.head(max_rows).to_dicts() if max_rows is not None else frame.to_dicts()
    elif dataset_format in {"arrow", "feather", "ipc"}:
        with _cloud_open(url, "rb", compressed=False) as handle:
            frame = pl.read_ipc(handle)
        rows = frame.head(max_rows).to_dicts() if max_rows is not None else frame.to_dicts()
    elif dataset_format in {"csv", "csv.gz", "tsv", "tsv.gz"}:
        separator = "\t" if dataset_format.startswith("tsv") else ","
        with _cloud_open(url, "rb", compressed=dataset_format.endswith(".gz")) as handle:
            frame = pl.read_csv(handle, separator=separator)
        tabular = frame.head(max_rows).to_dicts() if max_rows is not None else frame.to_dicts()
        rows = _decode_tabular_rows(tabular)
    elif dataset_format in {"txt", "txt.gz"}:
        rows = []
        with _cloud_open(url, "rt", compressed=dataset_format.endswith(".gz")) as handle:
            for line in handle:
                text = line.rstrip("\r\n")
                if text:
                    rows.append({"text": text})
                    if max_rows is not None and len(rows) >= max_rows:
                        break
    else:  # pragma: no cover
        raise AssertionError(dataset_format)
    return [DatasetItem.from_dict(row) for row in rows]


def read_dataset(file_path: Union[str, Path], max_rows: Optional[int] = None) -> List[DatasetItem]:
    if _is_cloud_url(file_path):
        return _read_dataset_cloud(file_path, max_rows)
    path = Path(file_path)
    from buffdata.security.policy import check_input
    check_input(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset path not found: {path}")
    dataset_format = _path_format(path)

    if dataset_format == "hf":
        rows = _read_huggingface(path, max_rows)
    elif dataset_format in {"jsonl", "ndjson", "jsonl.gz", "ndjson.gz"}:
        opener = gzip.open if dataset_format.endswith(".gz") else open
        rows = []
        with opener(path, "rt", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"NDJSON row {line_number} must be an object")
                rows.append(value)
                if max_rows is not None and len(rows) >= max_rows:
                    break
    elif dataset_format in {"json", "json.gz"}:
        opener = gzip.open if dataset_format.endswith(".gz") else open
        with opener(path, "rt", encoding="utf-8") as handle:
            rows = _limit(_rows_from_json_container(json.load(handle)), max_rows)
    elif dataset_format in {"yaml", "yml"}:
        with path.open("r", encoding="utf-8") as handle:
            rows = _limit(_rows_from_json_container(yaml.safe_load(handle)), max_rows)
    elif dataset_format in {"parquet", "pq"}:
        frame = pl.read_parquet(path)
        rows = frame.head(max_rows).to_dicts() if max_rows is not None else frame.to_dicts()
    elif dataset_format in {"arrow", "feather", "ipc"}:
        frame = pl.read_ipc(path)
        rows = frame.head(max_rows).to_dicts() if max_rows is not None else frame.to_dicts()
    elif dataset_format in {"csv", "csv.gz", "tsv", "tsv.gz"}:
        separator = "\t" if dataset_format.startswith("tsv") else ","
        frame = pl.read_csv(path, separator=separator)
        tabular = frame.head(max_rows).to_dicts() if max_rows is not None else frame.to_dicts()
        rows = _decode_tabular_rows(tabular)
    elif dataset_format in {"txt", "txt.gz"}:
        opener = gzip.open if dataset_format.endswith(".gz") else open
        rows = []
        with opener(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                text = line.rstrip("\r\n")
                if text:
                    rows.append({"text": text})
                    if max_rows is not None and len(rows) >= max_rows:
                        break
    else:  # pragma: no cover
        raise AssertionError(dataset_format)
    return [DatasetItem.from_dict(row) for row in rows]


def iter_dataset(file_path: Union[str, Path]) -> Generator[DatasetItem, None, None]:
    if _is_cloud_url(file_path):
        dataset_format = _cloud_format(file_path)
        if dataset_format in {"jsonl", "ndjson", "jsonl.gz", "ndjson.gz"}:
            with _cloud_open(file_path, "rt", compressed=dataset_format.endswith(".gz")) as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    value = json.loads(line)
                    if not isinstance(value, dict):
                        raise ValueError(f"NDJSON row {line_number} must be an object")
                    yield DatasetItem.from_dict(value)
            return
        if dataset_format in {"txt", "txt.gz"}:
            with _cloud_open(file_path, "rt", compressed=dataset_format.endswith(".gz")) as handle:
                for line in handle:
                    text = line.rstrip("\r\n")
                    if text:
                        yield DatasetItem.from_dict({"text": text})
            return
        yield from read_dataset(file_path)
        return

    path = Path(file_path)
    dataset_format = _path_format(path)
    if dataset_format in {"jsonl", "ndjson", "jsonl.gz", "ndjson.gz"}:
        opener = gzip.open if dataset_format.endswith(".gz") else open
        with opener(path, "rt", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"NDJSON row {line_number} must be an object")
                yield DatasetItem.from_dict(value)
        return
    if dataset_format in {"txt", "txt.gz"}:
        opener = gzip.open if dataset_format.endswith(".gz") else open
        with opener(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                text = line.rstrip("\r\n")
                if text:
                    yield DatasetItem.from_dict({"text": text})
        return
    yield from read_dataset(path)


def _write_dataset_cloud(
    items: Union[List[DatasetItem], Iterable[DatasetItem]],
    url: str,
    format_override: Optional[str] = None,
) -> None:
    dataset_format = _cloud_format(url, format_override)
    item_list = list(items)
    raw_items = [item.to_dict() if isinstance(item, DatasetItem) else dict(item) for item in item_list]

    if dataset_format in {"jsonl", "ndjson", "jsonl.gz", "ndjson.gz"}:
        with _cloud_open(url, "wb", compressed=dataset_format.endswith(".gz")) as handle:
            for row in raw_items:
                handle.write(orjson.dumps(row) + b"\n")
    elif dataset_format in {"json", "json.gz"}:
        payload = orjson.dumps(raw_items, option=orjson.OPT_INDENT_2)
        with _cloud_open(url, "wb", compressed=dataset_format.endswith(".gz")) as handle:
            handle.write(payload)
    elif dataset_format in {"yaml", "yml"}:
        with _cloud_open(url, "wt", compressed=False) as handle:
            yaml.safe_dump(raw_items, handle, allow_unicode=True, sort_keys=False)
    elif dataset_format in {"parquet", "pq"}:
        with _cloud_open(url, "wb", compressed=False) as handle:
            pl.DataFrame(raw_items).write_parquet(handle)
    elif dataset_format in {"arrow", "feather", "ipc"}:
        with _cloud_open(url, "wb", compressed=False) as handle:
            pl.DataFrame(raw_items).write_ipc(handle)
    elif dataset_format in {"csv", "csv.gz", "tsv", "tsv.gz"}:
        separator = "\t" if dataset_format.startswith("tsv") else ","
        frame = pl.DataFrame(_tabular_rows(raw_items))
        with _cloud_open(url, "wt", compressed=dataset_format.endswith(".gz")) as handle:
            handle.write(frame.write_csv(separator=separator))
    elif dataset_format in {"txt", "txt.gz"}:
        with _cloud_open(url, "wt", compressed=dataset_format.endswith(".gz")) as handle:
            for item in item_list:
                parsed = item if isinstance(item, DatasetItem) else DatasetItem.from_dict(dict(item))
                text = parsed.get_classification_text().replace("\r", " ").replace("\n", " ")
                handle.write(text + "\n")
    else:  # pragma: no cover
        raise AssertionError(dataset_format)


def write_dataset(
    items: Union[List[DatasetItem], Iterable[DatasetItem]],
    file_path: Union[str, Path],
    format_override: Optional[str] = None,
) -> None:
    from buffdata.security.policy import check_path
    if not _is_cloud_url(file_path):
        check_path(file_path, write=True)
    if _is_cloud_url(file_path):
        return _write_dataset_cloud(items, file_path, format_override)
    path = Path(file_path)
    dataset_format = _path_format(path, format_override)
    path.parent.mkdir(parents=True, exist_ok=True)
    item_list = list(items)
    raw_items = [item.to_dict() if isinstance(item, DatasetItem) else dict(item) for item in item_list]

    if dataset_format in {"jsonl", "ndjson", "jsonl.gz", "ndjson.gz"}:
        opener = gzip.open if dataset_format.endswith(".gz") else open
        with opener(path, "wb") as handle:
            for row in raw_items:
                handle.write(orjson.dumps(row) + b"\n")
    elif dataset_format in {"json", "json.gz"}:
        payload = orjson.dumps(raw_items, option=orjson.OPT_INDENT_2)
        if dataset_format.endswith(".gz"):
            with gzip.open(path, "wb") as handle:
                handle.write(payload)
        else:
            path.write_bytes(payload)
    elif dataset_format in {"yaml", "yml"}:
        with path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(raw_items, handle, allow_unicode=True, sort_keys=False)
    elif dataset_format in {"parquet", "pq"}:
        pl.DataFrame(raw_items).write_parquet(path)
    elif dataset_format in {"arrow", "feather", "ipc"}:
        pl.DataFrame(raw_items).write_ipc(path)
    elif dataset_format in {"csv", "csv.gz", "tsv", "tsv.gz"}:
        separator = "\t" if dataset_format.startswith("tsv") else ","
        frame = pl.DataFrame(_tabular_rows(raw_items))
        if dataset_format.endswith(".gz"):
            with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
                handle.write(frame.write_csv(separator=separator))
        else:
            frame.write_csv(path, separator=separator)
    elif dataset_format in {"txt", "txt.gz"}:
        opener = gzip.open if dataset_format.endswith(".gz") else open
        with opener(path, "wt", encoding="utf-8") as handle:
            for item in item_list:
                parsed = item if isinstance(item, DatasetItem) else DatasetItem.from_dict(dict(item))
                text = parsed.get_classification_text().replace("\r", " ").replace("\n", " ")
                handle.write(text + "\n")
    elif dataset_format == "hf":
        from datasets import Dataset

        Dataset.from_list(raw_items).save_to_disk(str(path))
    else:  # pragma: no cover
        raise AssertionError(dataset_format)


def _write_dataset_atomic_cloud(
    items: Union[List[DatasetItem], Iterable[DatasetItem]],
    url: str,
    format_override: Optional[str] = None,
) -> None:
    """Approximates atomic replace on object storage: write to a temporary key next to the
    destination, then rename it into place via the filesystem's own move (s3fs/gcsfs/adlfs
    implement this as a server-side operation, not a re-upload), so a reader never observes
    a partially written object. True atomicity guarantees vary by provider; this matches the
    strongest pattern each one supports.
    """
    dataset_format = _cloud_format(url, format_override)
    scheme, key = url.split("://", 1)
    temp_key = f"{key}.tmp-{uuid.uuid4().hex}"
    _write_dataset_cloud(items, f"{scheme}://{temp_key}", format_override=dataset_format)
    try:
        import fsspec
    except ImportError as exc:
        raise ImportError(
            "Install fsspec plus the matching filesystem package -- pip install "
            "buffdata[enterprise] -- to write cloud storage paths."
        ) from exc
    fs = fsspec.filesystem(scheme, **_cloud_storage_options(url))
    try:
        fs.mv(temp_key, key)
    except Exception:
        if fs.exists(temp_key):
            fs.rm(temp_key)
        raise


def write_dataset_atomic(
    items: Union[List[DatasetItem], Iterable[DatasetItem]],
    file_path: Union[str, Path],
    format_override: Optional[str] = None,
) -> None:
    """Write a complete file dataset and atomically replace the destination."""
    if _is_cloud_url(file_path):
        return _write_dataset_atomic_cloud(items, file_path, format_override)
    path = Path(file_path)
    dataset_format = _path_format(path, format_override)
    if dataset_format == "hf":
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = Path(tempfile.mkdtemp(prefix=f".{path.name}.", dir=path.parent))
        backup_path: Optional[Path] = None
        try:
            write_dataset(items, temporary_path, format_override="hf")
            if path.exists():
                backup_path = path.with_name(f".{path.name}.backup-{uuid.uuid4().hex}")
                os.replace(path, backup_path)
            try:
                os.replace(temporary_path, path)
            except Exception:
                if backup_path is not None and backup_path.exists():
                    os.replace(backup_path, path)
                raise
            if backup_path is not None and backup_path.exists():
                if backup_path.is_dir():
                    shutil.rmtree(backup_path)
                else:
                    backup_path.unlink()
        finally:
            if temporary_path.exists():
                shutil.rmtree(temporary_path)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=path.suffix or f".{dataset_format}", dir=path.parent
    )
    os.close(fd)
    temporary_path = Path(temporary_name)
    try:
        write_dataset(items, temporary_path, format_override=dataset_format)
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
