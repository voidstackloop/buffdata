import json

import pytest

from buffdata.engine.client import LLMProvider
from buffdata.engine.pipeline import OptimizationPipeline
from buffdata.models.formats import iter_dataset, read_dataset, write_dataset, write_dataset_atomic
from buffdata.models.schemas import DatasetItem, PipelineConfig


class OfflineClient:
    provider = LLMProvider.GEMINI
    default_model = "offline"
    usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    async def generate_structured_async(self, *args, **kwargs):
        raise AssertionError("No remote call is expected")


def _classification_items():
    return [
        DatasetItem.from_dict({"id": "a", "text": "Hello, world", "label": 0, "source": "unit"}),
        DatasetItem.from_dict({"id": "b", "text": "Markets rose today", "label": 1, "source": "unit"}),
    ]


def _projection(items):
    return [
        (item.id, item.get_classification_text(), item.labels, item.to_dict().get("source"))
        for item in items
    ]


@pytest.mark.parametrize(
    "extension",
    [
        "jsonl", "ndjson", "json", "jsonl.gz", "ndjson.gz", "json.gz",
        "csv", "csv.gz", "tsv", "tsv.gz", "parquet", "pq",
        "arrow", "feather", "ipc", "yaml", "yml",
    ],
)
def test_classification_round_trip_across_file_formats(tmp_path, extension):
    items = _classification_items()
    path = tmp_path / f"dataset.{extension}"

    write_dataset(items, path)
    restored = read_dataset(path)

    assert _projection(restored) == _projection(items)


@pytest.mark.parametrize("extension", ["txt", "txt.gz"])
def test_plain_text_round_trip_and_streaming(tmp_path, extension):
    path = tmp_path / f"corpus.{extension}"
    items = [DatasetItem.from_dict({"text": "first line"}), DatasetItem.from_dict({"text": "second line"})]

    write_dataset(items, path)

    assert [item.text for item in read_dataset(path)] == ["first line", "second line"]
    assert [item.text for item in iter_dataset(path)] == ["first line", "second line"]


def test_nested_chat_round_trip_through_tsv(tmp_path):
    path = tmp_path / "chat.tsv"
    item = DatasetItem.from_dict({
        "id": "chat-1",
        "messages": [
            {"role": "user", "content": "Question"},
            {"role": "assistant", "content": "Answer"},
        ],
        "metadata_column": {"language": "en"},
    })

    write_dataset([item], path)
    restored = read_dataset(path)[0]

    assert [(message.role, message.content) for message in restored.messages] == [
        ("user", "Question"),
        ("assistant", "Answer"),
    ]
    assert restored.to_dict()["metadata_column"] == {"language": "en"}


def test_json_wrappers_splits_and_columnar_layouts(tmp_path):
    wrappers = {
        "records": {"records": [{"text": "one", "label": 0}, {"text": "two", "label": 1}]},
        "splits": {
            "train": [{"text": "train row", "label": 0}],
            "test": [{"text": "test row", "label": 1}],
        },
        "columnar": {"text": ["left", "right"], "label": [0, 1]},
    }
    for name, payload in wrappers.items():
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        restored = read_dataset(path)
        assert len(restored) == 2
        if name == "splits":
            assert [item.to_dict()["_buffdata_split"] for item in restored] == ["train", "test"]


def test_huggingface_dataset_directory_round_trip(tmp_path):
    path = tmp_path / "classification.hf"
    items = _classification_items()

    write_dataset(items, path)
    restored = read_dataset(path)

    assert _projection(restored) == _projection(items)


def test_atomic_huggingface_output_can_be_replaced(tmp_path):
    path = tmp_path / "atomic.hf"
    write_dataset_atomic(_classification_items(), path)
    assert len(read_dataset(path)) == 2

    replacement = [DatasetItem.from_dict({"id": "new", "text": "replacement", "label": 1})]
    write_dataset_atomic(replacement, path)

    assert _projection(read_dataset(path)) == _projection(replacement)


def test_huggingface_dataset_dict_preserves_split(tmp_path):
    from datasets import Dataset, DatasetDict

    path = tmp_path / "dataset_dict"
    DatasetDict({
        "train": Dataset.from_list([{"text": "train", "label": 0}]),
        "test": Dataset.from_list([{"text": "test", "label": 1}]),
    }).save_to_disk(str(path))

    restored = read_dataset(path)

    assert [item.to_dict()["_buffdata_split"] for item in restored] == ["train", "test"]


@pytest.mark.parametrize("extension", ["jsonl.gz", "tsv", "arrow", "parquet"])
def test_atomic_output_round_trip(tmp_path, extension):
    path = tmp_path / f"atomic.{extension}"
    items = _classification_items()

    write_dataset_atomic(items, path)

    assert _projection(read_dataset(path)) == _projection(items)


def test_max_rows_is_honored_for_streamed_and_columnar_formats(tmp_path):
    for extension in ("jsonl", "csv", "parquet"):
        path = tmp_path / f"limited.{extension}"
        write_dataset(_classification_items(), path)
        assert len(read_dataset(path, max_rows=1)) == 1


def test_unknown_extension_is_rejected(tmp_path):
    path = tmp_path / "dataset.unknown"
    path.write_text('{"text":"not silently treated as JSONL"}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported dataset format"):
        read_dataset(path)


@pytest.mark.asyncio
@pytest.mark.parametrize("input_extension,output_extension", [("csv", "arrow"), ("parquet", "hf")])
async def test_pipeline_accepts_non_json_input_and_output(tmp_path, input_extension, output_extension):
    source = tmp_path / f"input.{input_extension}"
    output = tmp_path / f"output.{output_extension}"
    items = _classification_items()
    write_dataset(items, source)

    result = await OptimizationPipeline(
        PipelineConfig(accuracy_contract="strict"),
        client=OfflineClient(),
    ).run_file(source, output)

    assert result.metrics["accuracy_contract"] == "strict"
    assert _projection(read_dataset(output)) == _projection(items)
    assert output.exists()
