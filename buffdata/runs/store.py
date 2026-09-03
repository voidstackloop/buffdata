"""Additive metadata tables, transactional claims and append-only events."""
from __future__ import annotations
from contextlib import contextmanager
from datetime import datetime, timezone
import base64
import hashlib
import json
from pathlib import Path
import secrets
import time
import uuid

import sqlalchemy as sa
from buffdata.runs.models import RunConflict
from buffdata.security.policy import private_directory

metadata = sa.MetaData()
projects = sa.Table("managed_projects", metadata,
    sa.Column("id", sa.String(64), primary_key=True), sa.Column("body", sa.Text, nullable=False),
    sa.Column("active_run", sa.String(32)))
datasets = sa.Table("managed_datasets", metadata,
    sa.Column("id", sa.String(32), primary_key=True), sa.Column("project", sa.String(64), nullable=False),
    sa.Column("body", sa.Text, nullable=False), sa.Column("deleted", sa.Boolean, nullable=False, default=False))
runs = sa.Table("managed_runs", metadata,
    sa.Column("id", sa.String(32), primary_key=True), sa.Column("project", sa.String(64), nullable=False),
    sa.Column("actor", sa.String(512), nullable=False), sa.Column("status", sa.String(32), nullable=False),
    sa.Column("created", sa.Float, nullable=False), sa.Column("body", sa.Text, nullable=False),
    sa.Column("idempotency", sa.String(64), nullable=False, unique=True),
    sa.Column("request_hash", sa.String(64), nullable=False),
    sa.Column("lease", sa.String(64)), sa.Column("heartbeat", sa.Float),
    sa.Column("cancel_requested", sa.Boolean, nullable=False, default=False))
events = sa.Table("managed_events", metadata,
    sa.Column("id", sa.String(32), primary_key=True), sa.Column("run_id", sa.String(32)),
    sa.Column("project", sa.String(64), nullable=False), sa.Column("created", sa.Float, nullable=False),
    sa.Column("kind", sa.String(64), nullable=False), sa.Column("body", sa.Text, nullable=False))
sessions = sa.Table("managed_sessions", metadata,
    sa.Column("id", sa.String(64), primary_key=True), sa.Column("expires", sa.Float, nullable=False),
    sa.Column("body", sa.Text, nullable=False))
webhooks = sa.Table("managed_webhooks", metadata,
    sa.Column("id", sa.String(32), primary_key=True), sa.Column("project", sa.String(64), nullable=False),
    sa.Column("body", sa.Text, nullable=False), sa.Column("deleted", sa.Boolean, nullable=False, default=False))
# Deliberately its own table, not a field in managed_projects.body: that blob is already
# read-modify-written unlocked by update_project() (PUT /api/v1/projects/{id}), which would
# risk silently clobbering a wrapped data key under a concurrent project edit. This table's
# only writer is get_or_create_data_key(), inside its own locked transaction.
project_keys = sa.Table("managed_project_keys", metadata,
    sa.Column("project", sa.String(64), primary_key=True),
    sa.Column("wrapped_data_key", sa.Text, nullable=False))


def canonical(data):
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def digest(data):
    return hashlib.sha256(canonical(data).encode()).hexdigest()


def now_iso():
    return datetime.now(timezone.utc).isoformat()


class RunStore:
    def __init__(self, url: str):
        if url.startswith("sqlite:///") and not url.endswith(":memory:"):
            path = Path(url.removeprefix("sqlite:///"))
            private_directory(path.parent)
            # Create private from the outset, including when the user's umask is permissive.
            if not path.exists():
                import os
                os.close(os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600))
        self.engine = sa.create_engine(url, pool_pre_ping=True,
            connect_args={"check_same_thread": False, "timeout": 30} if url.startswith("sqlite") else {})
        metadata.create_all(self.engine)

    @contextmanager
    def transaction(self):
        with self.engine.connect() as conn:
            if self.engine.dialect.name == "sqlite":
                conn.exec_driver_sql("BEGIN IMMEDIATE")
            else:
                conn.begin()
            try:
                yield conn
                conn.commit()
            except BaseException:
                conn.rollback()
                raise

    def event(self, conn, project, kind, body=None, run_id=None):
        conn.execute(events.insert().values(id=uuid.uuid4().hex, project=project, run_id=run_id,
            created=time.time(), kind=kind, body=canonical(body or {})))

    def ensure_project(self, project, body):
        with self.transaction() as conn:
            if not conn.execute(sa.select(projects.c.id).where(projects.c.id == project)).first():
                conn.execute(projects.insert().values(id=project, body=canonical(body)))
                self.event(conn, project, "project.created")

    def project(self, project):
        with self.engine.connect() as conn:
            row = conn.execute(sa.select(projects).where(projects.c.id == project)).mappings().first()
            if not row:
                raise KeyError("Project not found")
            return {"id": row["id"], **json.loads(row["body"])}

    def all_projects(self):
        with self.engine.connect() as conn:
            return [{"id": r["id"], **json.loads(r["body"])} for r in conn.execute(sa.select(projects)).mappings()]

    def update_project(self, project, body, actor):
        with self.transaction() as conn:
            conn.execute(projects.update().where(projects.c.id == project).values(body=canonical(body)))
            self.event(conn, project, "project.updated", {"actor": actor})

    def add_dataset(self, project, body):
        with self.transaction() as conn:
            conn.execute(datasets.insert().values(id=body["id"], project=project, body=canonical(body), deleted=False))
            self.event(conn, project, "dataset.registered", {"dataset_id": body["id"]})
        return body

    def dataset(self, project, dataset_id):
        with self.engine.connect() as conn:
            row = conn.execute(sa.select(datasets.c.body).where(
                datasets.c.project == project, datasets.c.id == dataset_id, datasets.c.deleted == False)).first()
            if not row:
                raise KeyError("Dataset not found")
            return json.loads(row[0])

    def list_datasets(self, project):
        with self.engine.connect() as conn:
            return [json.loads(r[0]) for r in conn.execute(sa.select(datasets.c.body).where(
                datasets.c.project == project, datasets.c.deleted == False))]

    def delete_dataset(self, project, dataset_id, actor):
        with self.transaction() as conn:
            conn.execute(sa.select(projects.c.id).where(projects.c.id == project).with_for_update()).one()
            for row in conn.execute(sa.select(runs.c.body).where(runs.c.project == project)):
                spec = json.loads(row[0])["spec"]
                if dataset_id in (spec["dataset_id"], spec.get("validation_dataset_id")):
                    raise RunConflict("Dataset is referenced by run history")
            conn.execute(datasets.update().where(datasets.c.project == project,
                datasets.c.id == dataset_id).values(deleted=True))
            self.event(conn, project, "dataset.deleted", {"dataset_id": dataset_id, "actor": actor})

    def submit(self, spec, actor, key, body):
        request_hash = digest(spec.model_dump(mode="json"))
        idem = digest([spec.project_id, actor, key])
        with self.transaction() as conn:
            conn.execute(sa.select(projects.c.id).where(projects.c.id == spec.project_id).with_for_update()).first()
            old = conn.execute(sa.select(runs).where(runs.c.idempotency == idem)).mappings().first()
            if old:
                if old["request_hash"] != request_hash:
                    raise RunConflict("Idempotency key was used with different content")
                return self._run(old)
            for dataset_id in (spec.dataset_id, spec.validation_dataset_id):
                if dataset_id and not conn.execute(sa.select(datasets.c.id).where(datasets.c.project == spec.project_id,
                    datasets.c.id == dataset_id, datasets.c.deleted == False)).first():
                    raise RunConflict("Dataset was deleted before submission")
            run_id = body["id"]
            conn.execute(runs.insert().values(id=run_id, project=spec.project_id, actor=actor,
                status="queued", created=time.time(), body=canonical(body), idempotency=idem,
                request_hash=request_hash, cancel_requested=False))
            self.event(conn, spec.project_id, "run.queued", {"actor": actor}, run_id)
        return self.get(spec.project_id, run_id)

    @staticmethod
    def _run(row):
        return {**json.loads(row["body"]), "status": row["status"], "actor": row["actor"],
                "cancel_requested": row["cancel_requested"], "heartbeat": row["heartbeat"]}

    def get(self, project, run_id):
        with self.engine.connect() as conn:
            row = conn.execute(sa.select(runs).where(runs.c.project == project, runs.c.id == run_id)).mappings().first()
            if not row:
                raise KeyError("Run not found")
            return self._run(row)

    def list(self, project, limit=100):
        with self.engine.connect() as conn:
            return [self._run(row) for row in conn.execute(sa.select(runs).where(runs.c.project == project)
                    .order_by(runs.c.created.desc()).limit(min(max(limit, 1), 1000))).mappings()]

    def run_events(self, project, run_id=None):
        with self.engine.connect() as conn:
            query = sa.select(events).where(events.c.project == project)
            if run_id:
                query = query.where(events.c.run_id == run_id)
            return [{"kind": r["kind"], "created": r["created"], **json.loads(r["body"])}
                    for r in conn.execute(query.order_by(events.c.created).limit(10000)).mappings()]

    def claim(self, project, run_id=None):
        with self.transaction() as conn:
            row = conn.execute(sa.select(projects).where(projects.c.id == project).with_for_update()).mappings().one()
            if row["active_run"]:
                return None
            query = sa.select(runs).where(runs.c.project == project, runs.c.status == "queued")
            if run_id:
                query = query.where(runs.c.id == run_id)
            row = conn.execute(query.order_by(runs.c.created).limit(1).with_for_update()).mappings().first()
            if not row:
                return None
            lease = secrets.token_hex(32)
            conn.execute(runs.update().where(runs.c.id == row["id"]).values(status="running", lease=lease, heartbeat=time.time()))
            conn.execute(projects.update().where(projects.c.id == project).values(active_run=row["id"]))
            self.event(conn, project, "run.started", run_id=row["id"])
            return {**self._run(row), "status": "running", "lease": lease}

    def heartbeat(self, project, run_id, lease):
        with self.transaction() as conn:
            row = conn.execute(sa.select(runs).where(runs.c.project == project, runs.c.id == run_id,
                runs.c.lease == lease, runs.c.status == "running").with_for_update()).mappings().first()
            if not row:
                raise RunConflict("Run lease is no longer valid")
            conn.execute(runs.update().where(runs.c.id == run_id).values(heartbeat=time.time()))
            return row["cancel_requested"]

    def finish(self, project, run_id, lease, status, manifest=None, error=None, *, stale_before=None):
        if status not in {"succeeded", "failed", "cancelled", "interrupted"}:
            raise RunConflict("Invalid terminal status")
        with self.transaction() as conn:
            conn.execute(sa.select(projects.c.id).where(projects.c.id == project).with_for_update()).one()
            row = conn.execute(sa.select(runs).where(runs.c.project == project, runs.c.id == run_id,
                runs.c.lease == lease, runs.c.status == "running").with_for_update()).mappings().first()
            if not row:
                raise RunConflict("Run lease is no longer valid")
            if stale_before is not None and (row["heartbeat"] or 0) >= stale_before:
                raise RunConflict("Worker heartbeat was renewed; recovery is no longer safe")
            if row["cancel_requested"]:
                status, manifest = "cancelled", None
            body = json.loads(row["body"])
            body.update(completed_at=now_iso(), manifest=manifest if status == "succeeded" else None, error=error)
            conn.execute(runs.update().where(runs.c.id == run_id).values(status=status, body=canonical(body), lease=None))
            conn.execute(projects.update().where(projects.c.id == project).values(active_run=None))
            self.event(conn, project, "run." + status, {"error": error} if error else {}, run_id)
        return self.get(project, run_id)

    def cancel(self, project, run_id, actor):
        with self.transaction() as conn:
            conn.execute(sa.select(projects.c.id).where(projects.c.id == project).with_for_update()).one()
            row = conn.execute(sa.select(runs).where(runs.c.project == project, runs.c.id == run_id)
                               .with_for_update()).mappings().first()
            if not row:
                raise KeyError("Run not found")
            if row["status"] in {"queued", "running"}:
                conn.execute(runs.update().where(runs.c.id == run_id).values(cancel_requested=True,
                    status="cancelled" if row["status"] == "queued" else "running"))
                self.event(conn, project, "run.cancel_requested", {"actor": actor}, run_id)
        return self.get(project, run_id)

    def interrupt_stale(self, project, run_id):
        """Explicit operator recovery only. Never auto-retry a possibly billable job."""
        with self.engine.connect() as conn:
            row = conn.execute(sa.select(runs).where(runs.c.project == project, runs.c.id == run_id)).mappings().one()
            if row["status"] != "running" or time.time() - (row["heartbeat"] or 0) < 120:
                raise RunConflict("Worker is not stale; cancel it before resuming")
            lease = row["lease"]
        return self.finish(project, run_id, lease, "interrupted", error="Worker heartbeat expired",
                           stale_before=time.time() - 120)

    def record_event(self, project, run_id, kind, body=None):
        with self.transaction() as conn:
            self.event(conn, project, kind, body, run_id)

    def get_or_create_data_key(self, project, key_manager):
        with self.transaction() as conn:
            # Borrows the same per-project FOR UPDATE lock claim()/finish() already use,
            # purely for serialization -- the actual key data lives in project_keys below,
            # untouched by managed_projects.body's own (separate, pre-existing) race.
            conn.execute(sa.select(projects.c.id).where(projects.c.id == project).with_for_update()).one()
            row = conn.execute(sa.select(project_keys.c.wrapped_data_key)
                .where(project_keys.c.project == project)).first()
            if row:
                return key_manager.unwrap(base64.b64decode(row[0]))
            data_key = secrets.token_bytes(32)
            conn.execute(project_keys.insert().values(project=project,
                wrapped_data_key=base64.b64encode(key_manager.wrap(data_key)).decode()))
            self.event(conn, project, "project.data_key_created")
            return data_key

    def add_webhook(self, project, body):
        with self.transaction() as conn:
            conn.execute(webhooks.insert().values(id=body["id"], project=project, body=canonical(body), deleted=False))
            self.event(conn, project, "webhook.registered", {"webhook_id": body["id"]})
        return body

    def list_webhooks(self, project):
        with self.engine.connect() as conn:
            return [json.loads(r[0]) for r in conn.execute(sa.select(webhooks.c.body).where(
                webhooks.c.project == project, webhooks.c.deleted == False))]

    def delete_webhook(self, project, webhook_id, actor):
        with self.transaction() as conn:
            if not conn.execute(sa.select(webhooks.c.id).where(webhooks.c.project == project,
                webhooks.c.id == webhook_id, webhooks.c.deleted == False)).first():
                raise KeyError("Webhook not found")
            conn.execute(webhooks.update().where(webhooks.c.project == project,
                webhooks.c.id == webhook_id).values(deleted=True))
            self.event(conn, project, "webhook.deleted", {"webhook_id": webhook_id, "actor": actor})

    def save_session(self, token, body, lifetime=3600):
        with self.transaction() as conn:
            conn.execute(sessions.insert().values(id=hashlib.sha256(token.encode()).hexdigest(),
                         expires=time.time() + lifetime, body=canonical(body)))

    def session(self, token, *, consume=False):
        key = hashlib.sha256(token.encode()).hexdigest()
        with self.transaction() as conn:
            row = conn.execute(sa.select(sessions).where(sessions.c.id == key).with_for_update()).mappings().first()
            if consume:
                conn.execute(sessions.delete().where(sessions.c.id == key))
            if not row or row["expires"] <= time.time():
                return None
            return json.loads(row["body"])
