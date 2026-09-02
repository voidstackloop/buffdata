import io
import json

import pytest
from typer.testing import CliRunner

from buffdata.cli.main import app
from buffdata.models.formats import (
    _cloud_format,
    _cloud_storage_options,
    _is_cloud_url,
    read_dataset,
    write_dataset,
    write_dataset_atomic,
)
from buffdata.models.schemas import DatasetItem


@pytest.mark.parametrize(
    "url",
    ["s3://bucket/train.jsonl", "gs://bucket/train.jsonl", "gcs://bucket/train.jsonl",
     "az://container/train.jsonl", "abfs://container/train.jsonl", "adl://store/train.jsonl"],
)
def test_is_cloud_url_recognizes_supported_schemes(url):
    assert _is_cloud_url(url) is True


@pytest.mark.parametrize("value", ["train.jsonl", "/tmp/train.jsonl", "C:\\data\\train.jsonl", "http://example.com/x.jsonl"])
def test_is_cloud_url_rejects_local_and_unsupported_schemes(value):
    assert _is_cloud_url(value) is False


def test_is_cloud_url_rejects_path_objects():
    from pathlib import Path
    assert _is_cloud_url(Path("s3://bucket/train.jsonl")) is False


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("s3://bucket/train.jsonl", "jsonl"),
        ("s3://bucket/train.jsonl.gz", "jsonl.gz"),
        ("gs://bucket/data/train.parquet", "parquet"),
        ("az://container/train.csv.gz", "csv.gz"),
    ],
)
def test_cloud_format_infers_from_url(url, expected):
    assert _cloud_format(url) == expected


def test_cloud_format_rejects_directory_url():
    with pytest.raises(ValueError, match="single object"):
        _cloud_format("s3://bucket/some-dataset/")


def test_cloud_format_rejects_hf_override():
    with pytest.raises(ValueError, match="Hugging Face"):
        _cloud_format("s3://bucket/whatever", override="hf")


def test_cloud_storage_options_defaults_empty(monkeypatch):
    monkeypatch.delenv("BUFFDATA_S3_ENDPOINT_URL", raising=False)
    assert _cloud_storage_options("s3://bucket/x.jsonl") == {}


def test_cloud_storage_options_s3_endpoint_override(monkeypatch):
    monkeypatch.setenv("BUFFDATA_S3_ENDPOINT_URL", "https://minio.internal:9000")
    assert _cloud_storage_options("s3://bucket/x.jsonl") == {
        "client_kwargs": {"endpoint_url": "https://minio.internal:9000"}
    }
    assert _cloud_storage_options("gs://bucket/x.jsonl") == {}


# --- fakes mirroring fsspec's OpenFile / filesystem contract, monkeypatched onto the real
# fsspec module (already an installed transitive dependency) so the actual _cloud_open /
# _open_fsspec code path is exercised end to end, not bypassed. ---

class _FakeOpenFile:
    def __init__(self, handle):
        self._handle = handle

    def open(self):
        return self._handle


class _CapturingBuffer(io.BytesIO):
    """Mirrors every write into an external sink keyed by url, so the captured bytes are
    available regardless of whether close()/gzip/TextIOWrapper actually flushes into this
    buffer's own storage on exit.
    """

    def __init__(self, sink: dict, key: str):
        super().__init__()
        self._sink = sink
        self._key = key

    def write(self, data):
        result = super().write(data)
        self._sink[self._key] = self.getvalue()
        return result


def test_read_dataset_from_s3_jsonl(monkeypatch):
    payload = (
        json.dumps({"text": "first row", "label": 0}) + "\n"
        + json.dumps({"text": "second row", "label": 1}) + "\n"
    ).encode("utf-8")

    def fake_open(url, mode, **kwargs):
        assert url == "s3://bucket/train.jsonl"
        return _FakeOpenFile(io.BytesIO(payload))

    monkeypatch.setattr("fsspec.open", fake_open)
    items = read_dataset("s3://bucket/train.jsonl")

    assert len(items) == 2
    assert items[0].text == "first row"
    assert items[1].labels == 1


def test_read_dataset_from_s3_respects_max_rows(monkeypatch):
    payload = "\n".join(json.dumps({"text": f"row {i}"}) for i in range(5)).encode("utf-8")
    monkeypatch.setattr("fsspec.open", lambda url, mode, **kw: _FakeOpenFile(io.BytesIO(payload)))
    items = read_dataset("s3://bucket/train.jsonl", max_rows=2)
    assert len(items) == 2


def test_read_dataset_from_gcs_parquet_round_trips_through_real_polars(monkeypatch):
    import polars as pl

    frame = pl.DataFrame([{"text": "a", "label": 0}, {"text": "b", "label": 1}])
    buffer = io.BytesIO()
    frame.write_parquet(buffer)
    buffer.seek(0)

    monkeypatch.setattr("fsspec.open", lambda url, mode, **kw: _FakeOpenFile(buffer))
    items = read_dataset("gs://bucket/train.parquet")

    assert len(items) == 2
    assert items[0].text == "a"


def test_write_dataset_to_s3_jsonl(monkeypatch):
    captured: dict[str, bytes] = {}
    monkeypatch.setattr(
        "fsspec.open",
        lambda url, mode, **kw: _FakeOpenFile(_CapturingBuffer(captured, url)),
    )

    items = [
        DatasetItem.from_dict({"text": "alpha", "label": 0}),
        DatasetItem.from_dict({"text": "beta", "label": 1}),
    ]
    write_dataset(items, "s3://bucket/out.jsonl")

    lines = captured["s3://bucket/out.jsonl"].decode("utf-8").strip().split("\n")
    assert len(lines) == 2
    assert json.loads(lines[0])["text"] == "alpha"


def test_write_dataset_to_s3_parquet_uses_real_polars(monkeypatch):
    captured: dict[str, bytes] = {}
    monkeypatch.setattr(
        "fsspec.open",
        lambda url, mode, **kw: _FakeOpenFile(_CapturingBuffer(captured, url)),
    )

    items = [DatasetItem.from_dict({"text": "alpha", "label": 0})]
    write_dataset(items, "s3://bucket/out.parquet")

    import polars as pl
    frame = pl.read_parquet(io.BytesIO(captured["s3://bucket/out.parquet"]))
    assert frame.to_dicts()[0]["text"] == "alpha"


def test_write_dataset_atomic_to_s3_writes_temp_then_moves_into_place(monkeypatch):
    captured: dict[str, bytes] = {}
    moves: list[tuple[str, str]] = []

    monkeypatch.setattr(
        "fsspec.open",
        lambda url, mode, **kw: _FakeOpenFile(_CapturingBuffer(captured, url)),
    )

    class FakeFS:
        def mv(self, src, dst):
            moves.append((src, dst))

        def exists(self, path):
            return False

    monkeypatch.setattr("fsspec.filesystem", lambda scheme, **kw: FakeFS())

    items = [DatasetItem.from_dict({"text": "alpha", "label": 0})]
    write_dataset_atomic(items, "s3://bucket/out.jsonl")

    assert len(moves) == 1
    src, dst = moves[0]
    assert dst == "bucket/out.jsonl"
    assert src.startswith("bucket/out.jsonl.tmp-")
    # the temp object was actually written with real content before being "moved"
    temp_url = f"s3://{src}"
    assert json.loads(captured[temp_url].decode("utf-8").strip())["text"] == "alpha"


def test_dedup_cli_command_reaches_fsspec_with_the_url_intact(monkeypatch):
    # Regression test for the exact bug found while wiring this up: with a `Path`-typed
    # Typer parameter, "s3://bucket/train.jsonl" gets silently coerced and mangled to
    # "s3:/bucket/train.jsonl" (the double slash collapses) before the command body ever
    # runs. dedup_cmd's input_file/output_file are now `str`, so this must reach fsspec
    # with the URL untouched -- going through the real CLI entry point, not calling
    # read_dataset/write_dataset directly, so a regression back to `Path` typing would
    # actually be caught here.
    seen_urls = []
    captured: dict[str, bytes] = {}

    def fake_open(url, mode, **kwargs):
        seen_urls.append(url)
        if "r" in mode:
            payload = (
                json.dumps({"text": "alpha", "label": 0}) + "\n"
                + json.dumps({"text": "alpha", "label": 0}) + "\n"  # exact duplicate
                + json.dumps({"text": "beta", "label": 1}) + "\n"
            ).encode("utf-8")
            return _FakeOpenFile(io.BytesIO(payload))
        return _FakeOpenFile(_CapturingBuffer(captured, url))

    monkeypatch.setattr("fsspec.open", fake_open)

    result = CliRunner().invoke(app, [
        "dedup", "s3://bucket/train.jsonl", "-o", "s3://bucket/out.jsonl", "--method", "exact",
    ])

    assert result.exit_code == 0, result.output
    assert "s3://bucket/train.jsonl" in seen_urls
    assert "s3://bucket/out.jsonl" in seen_urls
    assert not any(url.startswith("s3:/bucket") and not url.startswith("s3://bucket") for url in seen_urls)
    kept_lines = captured["s3://bucket/out.jsonl"].decode("utf-8").strip().split("\n")
    assert len(kept_lines) == 2  # the exact duplicate was dropped


@pytest.mark.parametrize(
    ("command", "args"),
    [
        ("optimize", ["s3://bucket/train.jsonl", "-o", "s3://bucket/out.jsonl", "--classification", "off", "--quality-mode", "off"]),
        ("generate", ["s3://bucket/train.jsonl", "-o", "s3://bucket/out.jsonl", "--validation-file", "val.jsonl"]),
    ],
)
def test_companion_artifact_commands_refuse_cloud_urls_with_a_clear_message(command, args):
    # optimize/generate compute sibling artifact paths (.rejected.jsonl/.report.json/
    # .accuracy_gate.json) via Path.with_name(), which cloud URLs don't support -- they must
    # fail loudly and explain why, not silently mangle the URL and write to the wrong place.
    result = CliRunner().invoke(app, [command, *args])
    assert result.exit_code != 0
    assert "doesn't support cloud storage URLs yet" in result.output


def test_pipeline_command_refuses_cloud_urls(tmp_path):
    config_file = tmp_path / "pipeline.yaml"
    config_file.write_text("provider: gemini\n", encoding="utf-8")
    result = CliRunner().invoke(app, [
        "pipeline", str(config_file), "-i", "s3://bucket/train.jsonl", "-o", "s3://bucket/out.jsonl",
    ])
    assert result.exit_code != 0
    assert "doesn't support cloud storage URLs yet" in result.output


def test_write_dataset_atomic_to_s3_cleans_up_temp_on_move_failure(monkeypatch):
    monkeypatch.setattr(
        "fsspec.open",
        lambda url, mode, **kw: _FakeOpenFile(_CapturingBuffer({}, url)),
    )

    removed = []

    class FailingFS:
        def mv(self, src, dst):
            raise RuntimeError("simulated move failure")

        def exists(self, path):
            return True

        def rm(self, path):
            removed.append(path)

    monkeypatch.setattr("fsspec.filesystem", lambda scheme, **kw: FailingFS())

    items = [DatasetItem.from_dict({"text": "alpha", "label": 0})]
    with pytest.raises(RuntimeError, match="simulated move failure"):
        write_dataset_atomic(items, "s3://bucket/out.jsonl")

    assert len(removed) == 1
    assert removed[0].startswith("bucket/out.jsonl.tmp-")
