import json
import os
import stat
import sys

import pytest
import yaml
from typer.testing import CliRunner

from buffdata.cli.main import app
from buffdata.governance import (
    AuditRecord,
    DataContract,
    SQLiteAuditStore,
    estimate_cost,
    record_from_report,
    summarize_usage,
)


def _write_report(path, metrics=None):
    path.write_text(json.dumps({"metrics": metrics or {}}), encoding="utf-8")
    return path


def _sample_record(**overrides) -> AuditRecord:
    defaults = dict(
        command="optimize", provider="gemini", model="gemini-3.7-flash",
        accepted_records=90, rejected_records=10,
        input_tokens=1000, output_tokens=500, total_tokens=1500,
    )
    defaults.update(overrides)
    return AuditRecord(**defaults)


# --- SQLiteAuditStore ------------------------------------------------------------------

def test_record_and_query_round_trips(tmp_path):
    store = SQLiteAuditStore(tmp_path / "audit.db")
    entry = _sample_record()
    store.record(entry)

    results = store.query()
    assert len(results) == 1
    assert results[0].run_id == entry.run_id
    assert results[0].provider == "gemini"
    assert results[0].total_tokens == 1500


def test_store_persists_across_new_instances(tmp_path):
    db_path = tmp_path / "audit.db"
    SQLiteAuditStore(db_path).record(_sample_record())
    # A fresh SQLiteAuditStore pointed at the same file must see the same data --
    # this is the actual "durable" claim, not just an in-process cache.
    reopened = SQLiteAuditStore(db_path)
    assert len(reopened.query()) == 1


def test_query_filters_by_command_and_provider(tmp_path):
    store = SQLiteAuditStore(tmp_path / "audit.db")
    store.record(_sample_record(command="optimize", provider="gemini"))
    store.record(_sample_record(command="score", provider="openai"))

    assert len(store.query(command="optimize")) == 1
    assert len(store.query(provider="openai")) == 1
    assert len(store.query(command="optimize", provider="openai")) == 0
    assert len(store.query()) == 2


def test_query_filters_by_contract_name(tmp_path):
    store = SQLiteAuditStore(tmp_path / "audit.db")
    store.record(_sample_record(contract_name="support-tickets", contract_passed=True))
    store.record(_sample_record(contract_name=None, contract_passed=None))

    results = store.query(contract_name="support-tickets")
    assert len(results) == 1
    assert results[0].contract_passed is True


def test_query_respects_limit(tmp_path):
    store = SQLiteAuditStore(tmp_path / "audit.db")
    for _ in range(5):
        store.record(_sample_record())
    assert len(store.query(limit=2)) == 2


def test_record_metadata_round_trips_as_json(tmp_path):
    store = SQLiteAuditStore(tmp_path / "audit.db")
    store.record(_sample_record(metadata={"config_hash": "abc123", "tags": ["prod"]}))
    result = store.query()[0]
    assert result.metadata == {"config_hash": "abc123", "tags": ["prod"]}


# --- record_from_report ------------------------------------------------------------------

def test_record_from_report_extracts_fields(tmp_path):
    report = _write_report(tmp_path / "out.report.json", metrics={
        "provider": "anthropic", "model": "claude-sonnet-4-6",
        "accepted_records": 40, "rejected_records": 5,
        "accuracy_contract": "strict", "network_policy": "strict",
        "usage": {"input_tokens": 200, "output_tokens": 100, "total_tokens": 300},
    })
    store = SQLiteAuditStore(tmp_path / "audit.db")
    entry = record_from_report(store, report, command="optimize")

    assert entry.provider == "anthropic"
    assert entry.accepted_records == 40
    assert entry.network_policy == "strict"
    assert entry.total_tokens == 300
    assert len(store.query()) == 1


def test_record_from_report_missing_file_raises(tmp_path):
    store = SQLiteAuditStore(tmp_path / "audit.db")
    with pytest.raises(FileNotFoundError):
        record_from_report(store, tmp_path / "nope.report.json", command="optimize")


def test_record_from_report_carries_contract_result(tmp_path):
    report = _write_report(tmp_path / "out.report.json", metrics={"accepted_records": 10, "rejected_records": 0})
    store = SQLiteAuditStore(tmp_path / "audit.db")
    entry = record_from_report(store, report, command="optimize", contract_name="my-contract", contract_passed=False)
    assert entry.contract_name == "my-contract"
    assert entry.contract_passed is False


# --- usage/cost aggregation ---------------------------------------------------------------

def test_summarize_usage_aggregates_by_provider_model():
    records = [
        _sample_record(provider="gemini", model="gemini-3.7-flash", input_tokens=100, output_tokens=50, total_tokens=150),
        _sample_record(provider="gemini", model="gemini-3.7-flash", input_tokens=200, output_tokens=100, total_tokens=300),
        _sample_record(provider="openai", model="gpt-5.4-mini", input_tokens=10, output_tokens=5, total_tokens=15),
    ]
    summary = summarize_usage(records)
    assert summary["runs"] == 3
    assert summary["totals"]["total_tokens"] == 465
    assert summary["by_provider_model"]["gemini/gemini-3.7-flash"]["total_tokens"] == 450
    assert summary["by_provider_model"]["openai/gpt-5.4-mini"]["total_tokens"] == 15


def test_estimate_cost_uses_only_supplied_pricing():
    records = [
        _sample_record(provider="gemini", model="gemini-3.7-flash", input_tokens=1000, output_tokens=1000, total_tokens=2000),
        _sample_record(provider="unknown-provider", model="unknown-model", input_tokens=500, output_tokens=500, total_tokens=1000),
    ]
    pricing = {"gemini/gemini-3.7-flash": {"input_per_1k": 0.10, "output_per_1k": 0.40}}

    result = estimate_cost(records, pricing)
    assert result["total_cost_usd"] == pytest.approx(0.10 + 0.40)
    assert result["unpriced_tokens"] == 1000
    assert result["unpriced_provider_models"] == ["unknown-provider/unknown-model"]


def test_estimate_cost_with_empty_pricing_marks_everything_unpriced():
    records = [_sample_record(total_tokens=100)]
    result = estimate_cost(records, {})
    assert result["total_cost_usd"] == 0.0
    assert result["unpriced_tokens"] == 100


# --- CLI wiring -------------------------------------------------------------------------

def test_cli_audit_record_and_query_round_trip(tmp_path):
    report = _write_report(tmp_path / "out.report.json", metrics={
        "provider": "gemini", "model": "gemini-3.7-flash",
        "accepted_records": 20, "rejected_records": 0,
        "usage": {"input_tokens": 50, "output_tokens": 20, "total_tokens": 70},
    })
    db = tmp_path / "audit.db"

    record_result = CliRunner().invoke(app, [
        "audit", "record", str(report), "--command", "optimize", "--db", str(db),
    ])
    assert record_result.exit_code == 0, record_result.output

    query_result = CliRunner().invoke(app, ["audit", "query", "--db", str(db)])
    assert query_result.exit_code == 0
    # Rich truncates table cells at the test runner's narrow terminal width ("optimize" ->
    # "optimi…"), so match the guaranteed-visible prefix rather than the full word.
    assert "optimi" in query_result.output
    assert "gemini" in query_result.output


def test_cli_audit_record_with_contract_check(tmp_path):
    report = _write_report(tmp_path / "out.report.json", metrics={
        "accepted_records": 95, "rejected_records": 5,
    })
    contract_file = tmp_path / "contract.yaml"
    contract_file.write_text(yaml.safe_dump({
        "name": "demo", "requirements": {"max_rejected_fraction": 0.10},
    }), encoding="utf-8")
    db = tmp_path / "audit.db"

    result = CliRunner().invoke(app, [
        "audit", "record", str(report), "--command", "optimize",
        "--db", str(db), "--contract", str(contract_file),
    ])
    assert result.exit_code == 0, result.output

    from buffdata.governance import SQLiteAuditStore
    entries = SQLiteAuditStore(db).query()
    assert entries[0].contract_name == "demo"
    assert entries[0].contract_passed is True


def test_cli_audit_query_with_no_matches(tmp_path):
    db = tmp_path / "empty.db"
    result = CliRunner().invoke(app, ["audit", "query", "--db", str(db)])
    assert result.exit_code == 0
    assert "No matching runs" in result.output


def test_cli_audit_usage_report_without_pricing(tmp_path):
    report = _write_report(tmp_path / "out.report.json", metrics={
        "provider": "gemini", "model": "gemini-3.7-flash",
        "accepted_records": 10, "rejected_records": 0,
        "usage": {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150},
    })
    db = tmp_path / "audit.db"
    CliRunner().invoke(app, ["audit", "record", str(report), "--command", "optimize", "--db", str(db)])

    result = CliRunner().invoke(app, ["audit", "usage-report", "--db", str(db)])
    assert result.exit_code == 0, result.output
    assert "150" in result.output
    assert "$" not in result.output  # no cost shown without --pricing


def test_cli_audit_usage_report_with_pricing(tmp_path):
    report = _write_report(tmp_path / "out.report.json", metrics={
        "provider": "gemini", "model": "gemini-3.7-flash",
        "accepted_records": 10, "rejected_records": 0,
        "usage": {"input_tokens": 1000, "output_tokens": 1000, "total_tokens": 2000},
    })
    db = tmp_path / "audit.db"
    CliRunner().invoke(app, ["audit", "record", str(report), "--command", "optimize", "--db", str(db)])

    pricing_file = tmp_path / "pricing.yaml"
    pricing_file.write_text(yaml.safe_dump({
        "gemini/gemini-3.7-flash": {"input_per_1k": 0.1, "output_per_1k": 0.4},
    }), encoding="utf-8")

    result = CliRunner().invoke(app, [
        "audit", "usage-report", "--db", str(db), "--pricing", str(pricing_file),
    ])
    assert result.exit_code == 0, result.output
    assert "$0.50" in result.output


@pytest.mark.skipif(sys.platform == "win32", reason="chmod doesn't express owner-only ACLs on Windows")
def test_audit_db_file_is_created_with_owner_only_permissions(tmp_path):
    db_path = tmp_path / "audit.db"
    SQLiteAuditStore(db_path)

    mode = stat.S_IMODE(os.stat(db_path).st_mode)
    assert mode == 0o600
