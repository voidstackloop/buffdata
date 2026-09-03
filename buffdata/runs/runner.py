"""Supervision, deadlines and cancellation outside the optimization subprocess."""
from __future__ import annotations
import base64
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from buffdata.security.policy import private_json


def supervise(directory: Path, seconds: int, heartbeat, env_overrides: dict | None = None):
    # Deliberately do not persist stdout/stderr: SDK and dataset errors can contain secrets.
    # env_overrides is how a run's data key (see security/keys.py) reaches the executor
    # subprocess: only ever in this short-lived child's own environment, never written to a
    # file that would outlive the process needing it.
    env = {k: v for k, v in os.environ.items() if not k.startswith(("BUFFDATA_WORKER_", "BUFFDATA_DATABASE_"))
           and k not in {"BUFFDATA_SERVER_CONFIG", "BUFFDATA_SERVER_URL", "DATABASE_URL", "PGPASSWORD",
                         "BUFFDATA_ARTIFACT_MASTER_KEY_SECRET"}}
    env.update(env_overrides or {})
    process = subprocess.Popen([sys.executable, "-m", "buffdata.runs.executor", str(directory)],
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        start_new_session=os.name == "posix", env=env)
    deadline = time.monotonic() + seconds
    status = "failed"
    try:
        while process.poll() is None:
            if heartbeat():
                status = "cancelled"
                break
            if time.monotonic() >= deadline:
                break
            time.sleep(0.5)
        else:
            status = "succeeded" if process.returncode == 0 else "failed"
    finally:
        if process.poll() is None:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGTERM)
            else:
                process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                if os.name == "posix":
                    os.killpg(process.pid, signal.SIGKILL)
                else:
                    process.kill()
                process.wait()
    if status == "succeeded":
        return status, json.loads((directory / "candidate-manifest.json").read_text())
    return status, None


def run_local(service, project, run_id):
    run = service.store.claim(project, run_id)
    if run is None:
        raise ValueError("Another run is active, or this run is not queued")
    try:
        directory, data_key = service.prepare(run)
        env_overrides = {"BUFFDATA_RUN_DATA_KEY": base64.b64encode(data_key).decode()} if data_key else None
        status, manifest = supervise(directory, service.policy(project).execution_seconds,
            lambda: service.store.heartbeat(project, run_id, run["lease"]), env_overrides=env_overrides)
        if manifest:
            service.verify_manifest(project, run_id, manifest)
        result = service.store.finish(project, run_id, run["lease"], status, manifest,
            error=None if status == "succeeded" else "Execution stopped; no dataset published")
        if result["status"] == "succeeded":
            private_json(directory / "manifest.json", manifest)
        service.dispatch_webhooks(project, run_id, result["status"])
        return result
    except (Exception, KeyboardInterrupt):
        service.store.finish(project, run_id, run["lease"], "failed", error="Execution failed; no dataset published")
        service.dispatch_webhooks(project, run_id, "failed")
        raise
