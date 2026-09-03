from __future__ import annotations
import json
import os
from pathlib import Path
import sys
import uuid
import typer
import yaml

from buffdata.runs.models import RunSpec
from buffdata.runs.service import RunService
from buffdata.security.policy import remember_secret

app = typer.Typer(help="Managed runs, immutable artifacts, and recovery.")


def read_token(file: str | None):
    if not file:
        raise ValueError("A token file is required for a remote server")
    value = sys.stdin.readline(65537).strip() if file == "-" else Path(file).read_text().strip()
    if not value or len(value) > 65536:
        raise ValueError("Invalid token file")
    return remember_secret(value)


def service(root=None):
    result = RunService(root or os.getenv("BUFFDATA_RUNS_HOME", str(Path.home() / ".local/share/buffdata/runs")))
    result.store.ensure_project("local", {"members": {"local": "administrator"}, "policy": {}})
    return result


def remote(server, token_file):
    import httpx
    from urllib.parse import urlsplit
    if urlsplit(server).scheme != "https":
        raise ValueError("Remote CLI requires an HTTPS server")
    return httpx.Client(base_url=server.rstrip("/"), headers={"Authorization": "Bearer " + read_token(token_file)},
                        timeout=60, follow_redirects=False)


def output(value):
    typer.echo(json.dumps(value, indent=2, ensure_ascii=False))


@app.command("start")
def start(input_file: Path, config: Path = typer.Option(..., "--config"),
          root: Path | None = typer.Option(None, "--root"), server: str | None = typer.Option(None, "--server"),
          token_file: str | None = typer.Option(None, "--token-file"), project: str = "local",
          output_format: str = "jsonl", validation_file: Path | None = None,
          require_positive_gain: bool = False, idempotency_key: str | None = None,
          parent_run: str | None = typer.Option(None, "--parent-run", help="Record this run as derived from an existing run ID.")):
    configuration = yaml.safe_load(config.read_text()) or {}
    if server:
        with remote(server, token_file) as client:
            def upload(path):
                with path.open("rb") as handle:
                    response = client.post("/api/v1/datasets", params={"project_id": project, "filename": path.name}, content=handle)
                response.raise_for_status()
                return response.json()["id"]
            spec = RunSpec(project_id=project, dataset_id=upload(input_file), configuration=configuration,
                output_format=output_format, validation_dataset_id=upload(validation_file) if validation_file else None,
                require_positive_gain=require_positive_gain)
            response = client.post("/api/v1/runs", json=spec.model_dump(mode="json"),
                params={"parent_run_id": parent_run} if parent_run else {},
                headers={"Idempotency-Key": idempotency_key or uuid.uuid4().hex})
            response.raise_for_status()
            output(response.json())
    else:
        from buffdata.runs.runner import run_local
        manager = service(root)
        dataset = manager.register(project, input_file)
        validation = manager.register(project, validation_file) if validation_file else None
        spec = RunSpec(project_id=project, dataset_id=dataset["id"], configuration=configuration,
            output_format=output_format, validation_dataset_id=validation["id"] if validation else None,
            require_positive_gain=require_positive_gain)
        run = manager.submit(spec, key=idempotency_key, parent=parent_run)
        output(run_local(manager, project, run["id"]) if run["status"] == "queued" else run)


@app.command("list")
def list_runs(root: Path | None = None, project: str = "local", server: str | None = None, token_file: str | None = None):
    if server:
        with remote(server, token_file) as client:
            response = client.get("/api/v1/runs", params={"project_id": project})
            response.raise_for_status()
            output(response.json())
    else:
        output(service(root).store.list(project))


def action(name, run_id, root, project, server, token_file):
    if server:
        with remote(server, token_file) as client:
            suffix = "" if name == "show" else "/" + name
            response = client.request("POST" if name in {"cancel", "resume"} else "GET",
                "/api/v1/runs/" + run_id + suffix, params={"project_id": project})
            response.raise_for_status()
            return response.json()
    manager = service(root)
    if name == "show":
        return manager.store.get(project, run_id)
    if name == "cancel":
        return manager.store.cancel(project, run_id, "local")
    if name == "resume":
        from buffdata.runs.runner import run_local
        run = manager.resume(project, run_id)
        return run_local(manager, project, run["id"])
    return getattr(manager, name)(project, run_id)


def _register_action(name):
    def command(run_id: str, root: Path | None = None, project: str = "local",
                server: str | None = None, token_file: str | None = None):
        output(action(name, run_id, root, project, server, token_file))
    app.command(name)(command)


for _name in ("show", "compare", "cancel", "resume", "verify", "lineage"):
    _register_action(_name)


webhooks_app = typer.Typer(help="Project webhook registrations for run-completion events.")
app.add_typer(webhooks_app, name="webhooks")


@webhooks_app.command("add")
def webhooks_add(url: str, event: list[str] = typer.Option(..., "--event", help="Repeatable: run.succeeded, run.failed, run.cancelled, run.interrupted"),
                 root: Path | None = None, project: str = "local", server: str | None = None, token_file: str | None = None):
    if server:
        with remote(server, token_file) as client:
            response = client.post(f"/api/v1/projects/{project}/webhooks", json={"url": url, "events": event})
            response.raise_for_status()
            output(response.json())
    else:
        output(service(root).register_webhook(project, url, event, "local"))


@webhooks_app.command("list")
def webhooks_list(root: Path | None = None, project: str = "local", server: str | None = None, token_file: str | None = None):
    if server:
        with remote(server, token_file) as client:
            response = client.get(f"/api/v1/projects/{project}/webhooks")
            response.raise_for_status()
            output(response.json())
    else:
        output(service(root).list_webhooks(project))


@webhooks_app.command("remove")
def webhooks_remove(webhook_id: str, root: Path | None = None, project: str = "local", server: str | None = None, token_file: str | None = None):
    if server:
        with remote(server, token_file) as client:
            response = client.delete(f"/api/v1/projects/{project}/webhooks/{webhook_id}")
            response.raise_for_status()
            output(response.json())
    else:
        service(root).delete_webhook(project, webhook_id, "local")
        output({"deleted": webhook_id})
