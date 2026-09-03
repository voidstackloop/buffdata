"""Run bookkeeping and immutable artifacts, outside the optimizer's stage logic."""
from __future__ import annotations
from collections import Counter
import hashlib
import hmac
import importlib.metadata
import json
import os
from pathlib import Path
import secrets
import shutil
import sysconfig
import time
import uuid

from buffdata.runs.models import RunConflict, RunManifest, RunSpec
from buffdata.runs.store import RunStore, canonical, digest, now_iso
from buffdata.security.keys import create_key_manager, decrypt_bytes, is_encrypted
from buffdata.security.policy import SecurityPolicy, check_input, check_network_url, contained_path, private_directory, private_json

WEBHOOK_EVENTS = {"run.succeeded", "run.failed", "run.cancelled", "run.interrupted"}


def hash_path(path: Path) -> str:
    contained_path(path)
    h = hashlib.sha256()
    files = sorted(path.rglob("*")) if path.is_dir() else [path]
    for child in files:
        contained_path(child, path if path.is_dir() else path.parent)
        if child.is_dir():
            continue
        if not child.is_file():
            raise ValueError("Artifact is not a regular file")
        if path.is_dir():
            h.update(child.relative_to(path).as_posix().encode() + b"\0")
        with child.open("rb") as handle:
            while block := handle.read(1024**2):
                h.update(block)
    return h.hexdigest()


def runtime_identity(policy: SecurityPolicy) -> dict:
    import buffdata
    root = Path(buffdata.__file__).parent
    # Explicit path, not engine/client.py's ambient load_dotenv(): python-dotenv's default
    # find_dotenv() resolves its search root off the *calling process's own invocation mode*
    # (a bare `python -c` script searches from the CWD; a real .py file or `-m` module search
    # from wherever the first real call frame lives) -- so the same on-disk .env could appear
    # loaded or not depending on whether this ran from a CLI script, a `-m` executor
    # subprocess, or inside uvicorn, independent of whether anything about the actual runtime
    # environment changed. That made this function's provider_environment_sha256 below
    # nondeterministic across process kinds and could falsely trip "Worker implementation
    # differs from submission" for a run that never actually changed. Resolving explicitly
    # against the installed package's own location removes that dependency; a no-op (returns
    # False) exactly like today when there is no .env, e.g. every production install.
    from dotenv import load_dotenv
    load_dotenv(root.parent / ".env")
    h = hashlib.sha256()
    for source in sorted(root.rglob("*.py")):
        h.update(source.relative_to(root).as_posix().encode())
        h.update(source.read_bytes())
    # Ray and other libraries add vendored directories to sys.path at runtime. They are
    # not installed interpreter dependencies and do not exist in a fresh subprocess.
    distribution_paths = sorted({sysconfig.get_path("purelib"), sysconfig.get_path("platlib")})
    versions = {d.metadata["Name"].lower(): d.version for d in importlib.metadata.distributions(path=distribution_paths)
                if d.metadata["Name"]}
    plugins = {}
    for group in ("buffdata.validators", "buffdata.pii_recognizers"):
        for ep in importlib.metadata.entry_points(group=group):
            name = group + ":" + ep.name
            if name not in policy.approved_plugins:
                raise ValueError("Installed plugin requires approval: " + name)
            plugins[name] = {"entrypoint": ep.value, "distribution": ep.dist.name if ep.dist else None,
                             "version": ep.dist.version if ep.dist else None}
    if set(policy.approved_plugins) != set(plugins):
        raise ValueError("An approved plugin is unavailable")
    environment_names = ("BUFFDATA_DEFAULT_MODEL", "BUFFDATA_REQUEST_TIMEOUT", "OPENAI_BASE_URL",
        "ANTHROPIC_BASE_URL", "OPENAI_COMPATIBLE_BASE_URL", "OLLAMA_BASE_URL", "LMSTUDIO_BASE_URL",
        "VLLM_BASE_URL", "LLAMACPP_BASE_URL", "AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_VERSION",
        "AWS_REGION", "AWS_DEFAULT_REGION")
    return {"source_sha256": h.hexdigest(), "dependencies": dict(sorted(versions.items())),
            "provider_environment_sha256": digest({name: os.getenv(name) for name in environment_names}),
            "plugins": plugins, "policy_sha256": digest(policy.model_dump(mode="json"))}


def copy_snapshot(source: Path, target: Path):
    contained_path(source)
    private_directory(target.parent)
    if target.exists():
        raise RunConflict("Snapshot destination already exists")
    if source.is_dir():
        shutil.copytree(source, target)
        for file in target.rglob("*"):
            file.chmod(0o700 if file.is_dir() else 0o400)
    else:
        # Private permissions apply before copying the first byte.
        with source.open("rb") as src, os.fdopen(os.open(target, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600), "wb") as dst:
            shutil.copyfileobj(src, dst)
        target.chmod(0o400)


class RunService:
    def __init__(self, root: Path | str, store: RunStore | None = None):
        self.root = contained_path(root)
        private_directory(self.root)
        self.store = store or RunStore("sqlite:///" + str(self.root / "runs.db"))
        # None (the default: BUFFDATA_ARTIFACT_MASTER_KEY_SECRET unset) means encryption at
        # rest is fully off -- create_key_manager() makes zero secret-resolver calls in that
        # case, so constructing a RunService (once per API process, once per local CLI
        # invocation) stays free for every deployment that hasn't opted in.
        self.key_manager = create_key_manager()

    def project_root(self, project):
        import re
        if not re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", project):
            raise ValueError("Invalid project ID")
        return contained_path(self.root / project, self.root)

    def policy(self, project):
        return SecurityPolicy(**self.store.project(project).get("policy", {}))

    def register(self, project, source: Path):
        policy = self.policy(project)
        source = check_input(source, policy)
        from buffdata.models.formats import _path_format
        fmt = "hf" if source.is_dir() else _path_format(source)
        before = hash_path(source)
        for existing in self.store.list_datasets(project):
            if existing["sha256"] == before and existing["path"].endswith("." + fmt):
                self.dataset_path(project, existing["id"])
                return existing
        dataset_id = uuid.uuid4().hex
        relative = Path("datasets") / dataset_id / ("source.hf" if fmt == "hf" else "source." + fmt)
        target = self.project_root(project) / relative
        copy_snapshot(source, target)
        if hash_path(target) != before or hash_path(source) != before:
            raise RunConflict("Input changed during snapshot creation")
        return self.store.add_dataset(project, {"id": dataset_id, "name": source.name,
            "path": relative.as_posix(), "sha256": before, "created_at": now_iso(),
            "bytes": sum(p.stat().st_size for p in target.rglob("*") if p.is_file()) if target.is_dir() else target.stat().st_size})

    def dataset_path(self, project, dataset_id):
        record = self.store.dataset(project, dataset_id)
        path = contained_path(self.project_root(project) / record["path"], self.project_root(project))
        if hash_path(path) != record["sha256"]:
            raise RunConflict("Registered dataset integrity check failed")
        return path

    def submit(self, spec: RunSpec, actor="local", key=None, parent=None):
        policy = self.policy(spec.project_id)
        if parent is not None:
            # Raises KeyError (-> 404 at the API layer) if the parent isn't a run in this project.
            self.store.get(spec.project_id, parent)
        identity = runtime_identity(policy)
        inputs = {}
        for name, dataset_id in (("source", spec.dataset_id), ("validation", spec.validation_dataset_id)):
            if dataset_id:
                path = self.dataset_path(spec.project_id, dataset_id)
                check_input(path, policy)
                inputs[name] = hash_path(path)
        run_id = uuid.uuid4().hex
        body = {"id": run_id, "project_id": spec.project_id, "spec": spec.model_dump(mode="json"),
                "created_at": now_iso(), "implementation": identity, "inputs": inputs,
                "parent_run_id": parent, "manifest": None}
        return self.store.submit(spec, actor, key or uuid.uuid4().hex, body)

    def run_directory(self, project, run_id):
        import re
        if not re.fullmatch("[a-f0-9]{32}", run_id):
            raise ValueError("Invalid run ID")
        return contained_path(self.project_root(project) / "runs" / run_id, self.project_root(project))

    def prepare(self, run):
        spec = RunSpec(**run["spec"])
        policy = self.policy(spec.project_id)
        if runtime_identity(policy) != run["implementation"]:
            raise RunConflict("Execution environment changed since submission")
        directory = self.run_directory(spec.project_id, run["id"])
        private_directory(directory)
        paths = {}
        for kind, dataset_id in (("source", spec.dataset_id), ("validation", spec.validation_dataset_id)):
            if dataset_id:
                source = self.dataset_path(spec.project_id, dataset_id)
                target = directory / kind / source.name
                copy_snapshot(source, target)
                if hash_path(target) != run["inputs"][kind]:
                    raise RunConflict("Snapshot no longer matches submitted input")
                paths[kind] = str(target)
        if run.get("parent_run_id"):
            previous = self.run_directory(spec.project_id, run["parent_run_id"])
            checkpoint = previous / ".optimized.checkpoint.json"
            if checkpoint.exists():
                copy_snapshot(checkpoint, directory / checkpoint.name)
                (directory / checkpoint.name).chmod(0o600)
        private_json(directory / "execution.json", {"run": run, "policy": policy.model_dump(mode="json"), "paths": paths})
        # The data key is deliberately never written to execution.json (or anywhere else on
        # disk): it's returned here so the caller can hand it to the worker subprocess purely
        # via its own environment (claim response -> supervise()'s env_overrides), which
        # vanishes with that short-lived process. A file has no such lifecycle.
        data_key = self.store.get_or_create_data_key(spec.project_id, self.key_manager) if self.key_manager else None
        return directory, data_key

    def resume(self, project, run_id, actor="local"):
        previous = self.store.get(project, run_id)
        if previous["status"] == "running":
            previous = self.store.interrupt_stale(project, run_id)
        if previous["status"] not in {"failed", "cancelled", "interrupted"}:
            raise RunConflict("Only failed, cancelled, or interrupted runs can be resumed")
        if runtime_identity(self.policy(project)) != previous["implementation"]:
            raise RunConflict("Resume requires identical implementation, dependencies, plugins, and policy")
        result = self.submit(RunSpec(**previous["spec"]), actor, parent=run_id)
        if result["inputs"] != previous["inputs"]:
            self.store.cancel(project, result["id"], actor)
            raise RunConflict("Resume input differs from original run")
        return result

    def lineage(self, project, run_id):
        """``parent_run_id`` is caller-asserted provenance (an explicit ``--parent-run`` at
        submission, or the implicit link ``resume`` already records) -- never inferred from
        content, matching ``compare()``'s refusal to treat field deltas as causal lineage.

        Ancestors are walked one run at a time and are exact regardless of project history size.
        Descendant discovery scans the project's most recent 1000 runs -- the same bound
        ``RunStore.list`` and the dataset-deletion reference check already use -- so a descendant
        older than that window would not surface here.
        """
        self.store.get(project, run_id)

        def summary(run):
            return {"run_id": run["id"], "status": run["status"], "created_at": run.get("created_at"),
                    "parent_run_id": run.get("parent_run_id")}

        ancestors, seen, current_id = [], {run_id}, run_id
        while True:
            parent_id = self.store.get(project, current_id).get("parent_run_id")
            if not parent_id or parent_id in seen:
                break
            seen.add(parent_id)
            ancestors.append(summary(self.store.get(project, parent_id)))
            current_id = parent_id

        recent = self.store.list(project, limit=1000)
        by_id = {run["id"]: run for run in recent}
        children_of: dict[str, list[str]] = {}
        for run in recent:
            if run.get("parent_run_id"):
                children_of.setdefault(run["parent_run_id"], []).append(run["id"])
        descendants, seen_desc, frontier = [], {run_id}, [run_id]
        while frontier:
            next_frontier = []
            for parent_id in frontier:
                for child_id in children_of.get(parent_id, ()):
                    if child_id in seen_desc:
                        continue
                    seen_desc.add(child_id)
                    descendants.append(summary(by_id[child_id]))
                    next_frontier.append(child_id)
            frontier = next_frontier
        return {"run_id": run_id, "ancestors": ancestors, "descendants": descendants}

    def register_webhook(self, project, url, events, actor):
        """A registered webhook is itself an egress target, so it's validated the same way any
        other BuffData-initiated destination is: HTTPS, no embedded credentials, and a
        publicly-routable address (see ``check_network_url``) -- this rejects loopback, link-local,
        and RFC1918 targets such as a cloud metadata endpoint. That check runs at registration
        time only; it does not re-pin the address on every later delivery (a known gap, not a
        promise -- DNS-rebinding-resistant re-resolution at dispatch time is a follow-up)."""
        from urllib.parse import urlsplit
        if urlsplit(url).scheme != "https":
            raise ValueError("Webhook endpoints must use HTTPS")
        check_network_url(url)
        events = set(events)
        if not events or events - WEBHOOK_EVENTS:
            raise ValueError("Unknown webhook event: choose from " + ", ".join(sorted(WEBHOOK_EVENTS)))
        webhook_id = uuid.uuid4().hex
        body = {"id": webhook_id, "url": url, "events": sorted(events), "secret": secrets.token_urlsafe(32),
                "created_at": now_iso(), "created_by": actor}
        self.store.add_webhook(project, body)
        return body  # the secret is only ever returned here, at creation

    def list_webhooks(self, project):
        return [{k: v for k, v in hook.items() if k != "secret"} for hook in self.store.list_webhooks(project)]

    def delete_webhook(self, project, webhook_id, actor):
        self.store.delete_webhook(project, webhook_id, actor)

    @staticmethod
    def _send_webhook(url, body, headers):
        import httpx
        return httpx.post(url, content=body, headers=headers, timeout=10)

    def dispatch_webhooks(self, project, run_id, status):
        """Best-effort, at-least-once notification -- called after a run already reached a
        terminal state, so a slow or dead endpoint can never delay or fail the run itself.
        Every outcome (delivered or exhausted) is recorded as a run event for the same
        auditability the rest of run history already has; delivery bodies/headers are never
        logged beyond the webhook ID and a bounded error category."""
        event = "run." + status
        payload = canonical({"event": event, "run_id": run_id, "project_id": project,
                             "status": status, "occurred_at": now_iso()}).encode()
        for hook in self.store.list_webhooks(project):
            if event not in hook["events"]:
                continue
            signature = hmac.new(hook["secret"].encode(), payload, hashlib.sha256).hexdigest()
            headers = {"Content-Type": "application/json", "X-BuffData-Event": event,
                       "X-BuffData-Signature": "sha256=" + signature}
            delivered, last_error = False, "no delivery attempt"
            for attempt, delay in enumerate((0, 0.2, 0.6)):
                if delay:
                    time.sleep(delay)
                try:
                    response = self._send_webhook(hook["url"], payload, headers)
                    if 200 <= response.status_code < 300:
                        delivered = True
                        break
                    last_error = "HTTP " + str(response.status_code)
                except Exception as exc:
                    last_error = type(exc).__name__
            self.store.record_event(project, run_id,
                "webhook.delivered" if delivered else "webhook.delivery_failed",
                {"webhook_id": hook["id"]} if delivered else {"webhook_id": hook["id"], "error": last_error})

    def verify_manifest(self, project, run_id, manifest):
        model = RunManifest(**manifest)
        run = self.store.get(project, run_id)
        if model.run_id != run_id or model.project_id != project or model.spec.model_dump(mode="json") != run["spec"]:
            raise RunConflict("Manifest does not match run identity")
        if model.implementation != run["implementation"] or model.inputs != run["inputs"]:
            raise RunConflict("Manifest provenance does not match submission")
        if model.spec.require_positive_gain and not model.metrics.get("accuracy_gate", {}).get("accepted"):
            raise RunConflict("Required accuracy gate is absent or did not pass")
        directory = self.run_directory(project, run_id)
        if not {"output", "rejected", "report"} <= model.artifacts.keys():
            raise RunConflict("Manifest is missing required artifacts")
        for item in model.artifacts.values():
            path = contained_path(directory / item["path"], directory)
            if not path.exists() or hash_path(path) != item["sha256"]:
                raise RunConflict("Artifact integrity verification failed")
        return model

    def verify(self, project, run_id):
        run = self.store.get(project, run_id)
        if run["status"] != "succeeded" or not run.get("manifest"):
            raise RunConflict("Run has no published artifacts")
        self.verify_manifest(project, run_id, run["manifest"])
        return {"run_id": run_id, "verified": True, "artifacts": len(run["manifest"]["artifacts"])}

    def artifact(self, project, run_id, name):
        self.verify(project, run_id)
        manifest = self.store.get(project, run_id)["manifest"]
        if name not in manifest["artifacts"]:
            raise KeyError("Artifact not found")
        return contained_path(self.run_directory(project, run_id) / manifest["artifacts"][name]["path"],
                              self.run_directory(project, run_id))

    def artifact_bytes(self, project, run_id, name):
        """Content, decrypted if the artifact was encrypted at rest -- never writes a
        plaintext copy to disk. Manifest hashes were computed over whatever is actually on
        disk (ciphertext, when encrypted), so verify()/hash_path() need no changes."""
        raw = self.artifact(project, run_id, name).read_bytes()
        if not is_encrypted(raw):
            return raw
        data_key = self.store.get_or_create_data_key(project, self.key_manager)
        return decrypt_bytes(raw, data_key)

    def compare(self, project, run_id):
        import tempfile
        from buffdata.models.formats import read_dataset
        run = self.store.get(project, run_id)
        output_path = self.artifact(project, run_id, "output")
        # A use-scoped temp directory, cleaned up via Python's own guaranteed context-manager
        # __exit__ (runs on the way out through an exception too) -- not an ASGI post-response
        # hook, which would leave a plaintext copy behind on any interrupted/erroring request.
        with tempfile.TemporaryDirectory(dir=self.run_directory(project, run_id)) as scratch:
            decrypted_output = Path(scratch) / output_path.name
            decrypted_output.write_bytes(self.artifact_bytes(project, run_id, "output"))
            original = read_dataset(self.dataset_path(project, run["spec"]["dataset_id"]))
            candidate = read_dataset(decrypted_output)
            report = json.loads(self.artifact_bytes(project, run_id, "report"))
        def summary(items):
            labels = Counter(canonical(item.labels) for item in items)
            content = Counter(canonical([item.get_classification_text(), item.labels]) for item in items)
            return {"records": len(items), "label_distribution": dict(labels),
                    "duplicate_rows": sum(n - 1 for n in content.values()), "multiplicities": content}
        before, after = summary(original), summary(candidate)
        old, new = before.pop("multiplicities"), after.pop("multiplicities")
        source_ids, candidate_ids = Counter(x.id for x in original), Counter(x.id for x in candidate)
        source_by_id = {x.id: x for x in original if source_ids[x.id] == 1}
        candidate_by_id = {x.id: x for x in candidate if candidate_ids[x.id] == 1}
        fields = Counter()
        for item_id in source_by_id.keys() & candidate_by_id.keys():
            left, right = source_by_id[item_id].to_dict(), candidate_by_id[item_id].to_dict()
            for key in left.keys() | right.keys():
                if left.get(key) != right.get(key):
                    fields[key] += 1
        return {"original": before, "generated": after, "removed_or_changed": sum((old - new).values()),
                "added_or_changed": sum((new - old).values()),
                "field_changes_by_matching_unique_id": dict(fields),
                "matched_unique_ids": len(source_by_id.keys() & candidate_by_id.keys()),
                "accuracy": run["manifest"]["metrics"].get("accuracy_gate"),
                "rejection_reasons": report.get("rejection_reasons", {}),
                "note": "Counts preserve duplicate multiplicities; changed/added/removed are not causal row lineage. Accuracy is absent unless measured."}
