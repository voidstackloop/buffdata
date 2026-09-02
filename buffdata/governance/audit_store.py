"""A durable, queryable record of every buffdata run.

Not because report.json is wrong -- it's the correct source of truth for one run -- but
because it's ephemeral: it lives wherever that run happened to execute, and there's no way
to ask "how much did team X's runs cost last month" or "which runs used network_policy
strict" across many runs without something that persists and can be queried across all of
them.

Backed by SQLite by default: a real file on disk (durable -- survives process restarts) with
real SQL querying, which is a complete, production-viable answer for a single machine or a
small team's shared audit log, not a placeholder. AuditStore is a Protocol specifically so a
Postgres/BigQuery-backed implementation can be dropped in later behind the exact same
record()/query() calls, once a deployment needs multi-writer/multi-machine access at once --
that swap is deliberately not built here, since there's no live Postgres/BigQuery in this
environment to verify one against.

No dollar pricing is hardcoded anywhere in this module. Token usage is aggregated for real;
converting it to a cost requires the caller to supply their own pricing table, because
guessing at provider prices and presenting them as fact would just be fabricating data.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Protocol, Union

from pydantic import BaseModel, Field


class AuditRecord(BaseModel):
    run_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    recorded_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    command: str
    provider: Optional[str] = None
    model: Optional[str] = None
    accepted_records: int = 0
    rejected_records: int = 0
    accuracy_contract: Optional[str] = None
    network_policy: Optional[str] = None
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    contract_name: Optional[str] = None
    contract_passed: Optional[bool] = None
    report_path: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class AuditStore(Protocol):
    def record(self, entry: AuditRecord) -> None: ...

    def query(
        self,
        *,
        command: Optional[str] = None,
        provider: Optional[str] = None,
        contract_name: Optional[str] = None,
        since: Optional[str] = None,
        limit: int = 100,
    ) -> list[AuditRecord]: ...


_COLUMNS = (
    "run_id", "recorded_at", "command", "provider", "model",
    "accepted_records", "rejected_records", "accuracy_contract", "network_policy",
    "input_tokens", "output_tokens", "total_tokens",
    "contract_name", "contract_passed", "report_path", "metadata",
)


class SQLiteAuditStore:
    """The default AuditStore. One `runs` table, one row per recorded run."""

    def __init__(self, db_path: Union[str, Path] = "buffdata_audit.db"):
        self.db_path = Path(db_path)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    recorded_at TEXT NOT NULL,
                    command TEXT NOT NULL,
                    provider TEXT,
                    model TEXT,
                    accepted_records INTEGER,
                    rejected_records INTEGER,
                    accuracy_contract TEXT,
                    network_policy TEXT,
                    input_tokens INTEGER,
                    output_tokens INTEGER,
                    total_tokens INTEGER,
                    contract_name TEXT,
                    contract_passed INTEGER,
                    report_path TEXT,
                    metadata TEXT
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_runs_command ON runs(command)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_runs_recorded_at ON runs(recorded_at)")

    def record(self, entry: AuditRecord) -> None:
        row = entry.model_dump()
        row["contract_passed"] = None if row["contract_passed"] is None else int(row["contract_passed"])
        row["metadata"] = json.dumps(row["metadata"], ensure_ascii=False)
        placeholders = ", ".join(f":{col}" for col in _COLUMNS)
        with self._connect() as conn:
            conn.execute(
                f"INSERT OR REPLACE INTO runs ({', '.join(_COLUMNS)}) VALUES ({placeholders})",
                row,
            )

    def query(
        self,
        *,
        command: Optional[str] = None,
        provider: Optional[str] = None,
        contract_name: Optional[str] = None,
        since: Optional[str] = None,
        limit: int = 100,
    ) -> list[AuditRecord]:
        clauses: list[str] = []
        params: list[Any] = []
        if command:
            clauses.append("command = ?")
            params.append(command)
        if provider:
            clauses.append("provider = ?")
            params.append(provider)
        if contract_name:
            clauses.append("contract_name = ?")
            params.append(contract_name)
        if since:
            clauses.append("recorded_at >= ?")
            params.append(since)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM runs {where} ORDER BY recorded_at DESC LIMIT ?", (*params, limit)
            ).fetchall()
        return [_row_to_record(row) for row in rows]


def _row_to_record(row: sqlite3.Row) -> AuditRecord:
    data = dict(row)
    data["metadata"] = json.loads(data["metadata"] or "{}")
    if data["contract_passed"] is not None:
        data["contract_passed"] = bool(data["contract_passed"])
    return AuditRecord(**data)


def record_from_report(
    store: AuditStore,
    report_path: Union[str, Path],
    *,
    command: str,
    contract_name: Optional[str] = None,
    contract_passed: Optional[bool] = None,
) -> AuditRecord:
    """Ingest an existing *.report.json into the audit store as one AuditRecord."""
    report_path = Path(report_path)
    if not report_path.exists():
        raise FileNotFoundError(f"Report not found: {report_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    metrics = report.get("metrics", {})
    usage = metrics.get("usage", {})
    entry = AuditRecord(
        command=command,
        provider=metrics.get("provider"),
        model=metrics.get("model"),
        accepted_records=metrics.get("accepted_records", 0),
        rejected_records=metrics.get("rejected_records", 0),
        accuracy_contract=metrics.get("accuracy_contract"),
        network_policy=metrics.get("network_policy"),
        input_tokens=usage.get("input_tokens", 0),
        output_tokens=usage.get("output_tokens", 0),
        total_tokens=usage.get("total_tokens", 0),
        contract_name=contract_name,
        contract_passed=contract_passed,
        report_path=str(report_path),
    )
    store.record(entry)
    return entry


def summarize_usage(records: list[AuditRecord]) -> dict[str, Any]:
    """Pure token-usage aggregation -- no dollar amounts, no assumptions about pricing."""
    by_provider_model: dict[str, dict[str, int]] = {}
    totals = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    for record in records:
        key = f"{record.provider or 'unknown'}/{record.model or 'unknown'}"
        bucket = by_provider_model.setdefault(key, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})
        for field in totals:
            value = getattr(record, field)
            totals[field] += value
            bucket[field] += value
    return {"runs": len(records), "totals": totals, "by_provider_model": by_provider_model}


def estimate_cost(
    records: list[AuditRecord], pricing: dict[str, dict[str, float]]
) -> dict[str, Any]:
    """Convert token usage to a dollar estimate using a pricing table the caller supplies --
    keyed "provider/model", each value {"input_per_1k": float, "output_per_1k": float}.
    Runs whose provider/model isn't in the table are counted separately as unpriced, never
    silently assumed free or guessed at.
    """
    total_cost = 0.0
    by_key: dict[str, float] = {}
    unpriced_tokens = 0
    unpriced_keys: set[str] = set()
    for record in records:
        key = f"{record.provider}/{record.model}"
        rates = pricing.get(key)
        if not rates:
            unpriced_tokens += record.total_tokens
            unpriced_keys.add(key)
            continue
        cost = (record.input_tokens / 1000) * rates.get("input_per_1k", 0.0) + (
            record.output_tokens / 1000
        ) * rates.get("output_per_1k", 0.0)
        total_cost += cost
        by_key[key] = by_key.get(key, 0.0) + cost
    return {
        "total_cost_usd": total_cost,
        "by_provider_model_usd": by_key,
        "unpriced_tokens": unpriced_tokens,
        "unpriced_provider_models": sorted(unpriced_keys),
    }
