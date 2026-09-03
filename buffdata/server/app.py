from __future__ import annotations
import base64
import hashlib
import json
import os
from pathlib import Path
import random
import secrets
import time
from urllib.parse import urlsplit

from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, ConfigDict, Field, model_validator
from starlette.concurrency import run_in_threadpool

from buffdata.governance.oidc import OIDCConfig, OIDCVerificationError, verify_bearer_token
from buffdata.runs.models import RunConflict, RunSpec
from buffdata.security.keys import is_encrypted
from buffdata.runs.service import RunService
from buffdata.runs.store import RunStore
from buffdata.security.policy import SecurityError, SecurityPolicy, contained_path, private_directory, private_json


class Settings(BaseModel):
    model_config = ConfigDict(extra="forbid")
    root: Path
    database_url: str
    public_url: str
    oidc: OIDCConfig
    client_id: str
    authorization_url: str
    token_url: str
    client_secret_file: Path | None = None
    projects: dict[str, dict] = Field(default_factory=dict)

    @model_validator(mode="after")
    def secure_urls(self):
        for url in (self.public_url, self.authorization_url, self.token_url, self.oidc.issuer):
            p = urlsplit(url)
            if p.scheme != "https" or not p.hostname or p.username or p.password:
                raise ValueError("Server and OIDC URLs must use credential-free HTTPS")
        if self.oidc.jwks_url and urlsplit(self.oidc.jwks_url).scheme != "https":
            raise ValueError("Remote JWKS must use HTTPS")
        return self


class ProjectUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    members: dict[str, str]
    policy: SecurityPolicy


class WebhookCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    url: str
    events: list[str]


def create_app(settings: Settings) -> FastAPI:
    store = RunStore(settings.database_url)
    service = RunService(settings.root, store)
    by_worker_hash: dict[str, list[str]] = {}
    for project, body in settings.projects.items():
        service.project_root(project)
        SecurityPolicy(**body.get("policy", {}))
        store.ensure_project(project, body)
        token_hash = body.get("worker_token_sha256")
        if token_hash:
            by_worker_hash.setdefault(token_hash, []).append(project)
    for members in by_worker_hash.values():
        # A shared worker token is how a worker pool is authorized across projects -- require
        # an explicit, matching worker_pool declaration on every member so pooling is always a
        # deliberate, grep-able config choice, never an accidental consequence of two projects
        # ending up with the same token hash (e.g. a copy-pasted server config).
        if len(members) > 1:
            pools = {settings.projects[member].get("worker_pool") for member in members}
            if len(pools) != 1 or None in pools:
                raise ValueError("Projects sharing a worker token must declare the same worker_pool: "
                                  + ", ".join(sorted(members)))
    app = FastAPI(title="BuffData Run Management", version="1", docs_url=None, redoc_url=None)
    app.state.service = service
    app.state.settings = settings
    templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))
    app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")

    @app.exception_handler(KeyError)
    async def missing(request, exc):
        return JSONResponse({"detail": "Resource not found"}, status_code=404)

    @app.exception_handler(RunConflict)
    async def conflict(request, exc):
        return JSONResponse({"detail": str(exc)}, status_code=409)

    @app.exception_handler(SecurityError)
    async def policy_error(request, exc):
        return JSONResponse({"detail": "Execution policy rejected the request"}, status_code=403)

    @app.exception_handler(ValueError)
    async def invalid(request, exc):
        return JSONResponse({"detail": "Invalid configuration or input"}, status_code=400)

    @app.middleware("http")
    async def boundaries(request, call_next):
        if request.headers.get("origin") and request.headers["origin"].rstrip("/") != settings.public_url.rstrip("/"):
            return JSONResponse({"detail": "Origin denied"}, status_code=403)
        try:
            response = await call_next(request)
        except Exception:
            # No provider, database credentials, filesystem paths, or dataset excerpts.
            response = JSONResponse({"detail": "Internal request failure"}, status_code=500)
        response.headers.update({"X-Content-Type-Options": "nosniff", "Referrer-Policy": "no-referrer",
            "Cache-Control": "no-store", "X-Frame-Options": "DENY",
            "Content-Security-Policy": "default-src 'self'; script-src 'self'; style-src 'self'; object-src 'none'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'"})
        return response

    def identity(request):
        authorization = request.headers.get("authorization", "")
        if authorization.startswith("Bearer "):
            try:
                return verify_bearer_token(authorization[7:], config=settings.oidc).actor
            except OIDCVerificationError:
                raise HTTPException(401, "Invalid bearer token") from None
        session = store.session(request.cookies.get("__Host-buffdata", ""))
        if not session or "actor" not in session:
            raise HTTPException(401, "Authentication required")
        if request.method not in {"GET", "HEAD", "OPTIONS"}:
            if not secrets.compare_digest(request.headers.get("x-csrf-token", ""), session["csrf"]):
                raise HTTPException(403, "CSRF verification failed")
        return session["actor"]

    def authorize(request, project, minimum="viewer"):
        actor = identity(request)
        body = store.project(project)
        role = body.get("members", {}).get(actor)
        levels = {"viewer": 1, "runner": 2, "administrator": 3}
        if levels.get(role, 0) < levels[minimum]:
            raise HTTPException(403, "Project permission denied")
        return actor

    def worker(request) -> set[str]:
        token = request.headers.get("authorization", "").removeprefix("Bearer ")
        digest = hashlib.sha256(token.encode()).hexdigest()
        # Every configured project is checked unconditionally (no early return once a match is
        # found) -- both to collect every project a pooled token authorizes, and so lookup time
        # doesn't vary with which project (if any) matches.
        authorized = {project for project, config in settings.projects.items()
                      if len(config.get("worker_token_sha256", "")) == 64
                      and secrets.compare_digest(digest, config["worker_token_sha256"])}
        if not authorized:
            raise HTTPException(401, "Invalid worker identity")
        return authorized

    def worker_project(authorized: set[str], body: dict) -> str:
        # project_id is only ambiguous for a pooled token (authorized for more than one
        # project); a single-project token keeps working exactly as before without sending it.
        if "project_id" in body:
            project = body["project_id"]
            if project not in authorized:
                raise HTTPException(403, "Run does not belong to an authorized project")
            return project
        if len(authorized) == 1:
            return next(iter(authorized))
        raise HTTPException(400, "project_id is required for a pooled worker token")

    @app.get("/healthz")
    def health():
        try:
            with store.engine.connect() as connection:
                connection.exec_driver_sql("SELECT 1")
        except Exception:
            raise HTTPException(503, "Metadata service unavailable") from None
        return {"status": "ok"}

    @app.get("/login")
    async def login():
        from authlib.integrations.httpx_client import AsyncOAuth2Client
        state, verifier, nonce = secrets.token_urlsafe(32), secrets.token_urlsafe(48), secrets.token_urlsafe(32)
        async with AsyncOAuth2Client(settings.client_id, redirect_uri=settings.public_url.rstrip("/") + "/callback",
                                    scope="openid profile", code_challenge_method="S256") as client:
            url, _ = client.create_authorization_url(settings.authorization_url, state=state, nonce=nonce, code_verifier=verifier)
        store.save_session(state, {"verifier": verifier, "nonce": nonce}, 300)
        response = RedirectResponse(url, status_code=302)
        response.set_cookie("__Host-buffdata-login", state, secure=True, httponly=True, samesite="lax", max_age=300)
        return response

    @app.get("/callback")
    async def callback(request: Request):
        state = request.query_params.get("state", "")
        if not state or not secrets.compare_digest(state, request.cookies.get("__Host-buffdata-login", "")):
            raise HTTPException(401, "Invalid login state")
        pending = store.session(state, consume=True)
        if not pending or "verifier" not in pending:
            raise HTTPException(401, "Login expired or already used")
        from authlib.integrations.httpx_client import AsyncOAuth2Client
        client_secret = settings.client_secret_file.read_text().strip() if settings.client_secret_file else None
        async with AsyncOAuth2Client(settings.client_id, client_secret,
            redirect_uri=settings.public_url.rstrip("/") + "/callback", timeout=15) as client:
            token = await client.fetch_token(settings.token_url, code=request.query_params.get("code"),
                                              code_verifier=pending["verifier"])
        try:
            verified = verify_bearer_token(token["id_token"], config=settings.oidc.model_copy(update={"audience": settings.client_id}))
            if not secrets.compare_digest(str(verified.claims.get("nonce", "")), pending["nonce"]):
                raise OIDCVerificationError("Invalid nonce")
        except (KeyError, OIDCVerificationError):
            raise HTTPException(401, "OIDC verification failed") from None
        session_id = secrets.token_urlsafe(32)
        lifetime = min(3600, max(0, int(verified.claims["exp"] - time.time())))
        store.save_session(session_id, {"actor": verified.actor, "csrf": secrets.token_urlsafe(32)}, lifetime)
        response = RedirectResponse("/", status_code=302)
        response.delete_cookie("__Host-buffdata-login", secure=True, httponly=True)
        response.set_cookie("__Host-buffdata", session_id, secure=True, httponly=True, samesite="lax", max_age=lifetime)
        return response

    @app.post("/logout")
    def logout(request: Request):
        identity(request)
        store.session(request.cookies.get("__Host-buffdata", ""), consume=True)
        response = JSONResponse({"logged_out": True})
        response.delete_cookie("__Host-buffdata", secure=True, httponly=True)
        return response

    @app.get("/")
    def dashboard(request: Request):
        try:
            actor = identity(request)
        except HTTPException:
            return RedirectResponse("/login", status_code=302)
        session = store.session(request.cookies.get("__Host-buffdata", "")) or {}
        return templates.TemplateResponse(request=request, name="dashboard.html", context={"actor": actor, "csrf": session.get("csrf", "")})

    @app.get("/api/v1/projects")
    def list_projects(request: Request):
        actor = identity(request)
        return [{"id": p["id"], "role": p["members"][actor]} for p in store.all_projects() if actor in p.get("members", {})]

    @app.put("/api/v1/projects/{project_id}")
    def update_project(project_id: str, body: ProjectUpdate, request: Request):
        actor = authorize(request, project_id, "administrator")
        if not any(role == "administrator" for role in body.members.values()) or any(
            role not in {"viewer", "runner", "administrator"} for role in body.members.values()):
            raise HTTPException(400, "Project requires an administrator and valid roles")
        current = store.project(project_id)
        current.update(body.model_dump(mode="json"))
        current.pop("id", None)
        store.update_project(project_id, current, actor)
        return {"updated": True}

    @app.get("/api/v1/datasets")
    def list_datasets(project_id: str, request: Request):
        authorize(request, project_id)
        return [{k: v for k, v in d.items() if k != "path"} for d in store.list_datasets(project_id)]

    @app.post("/api/v1/datasets", status_code=201)
    async def upload(project_id: str, filename: str, request: Request):
        authorize(request, project_id, "runner")
        from buffdata.models.formats import _path_format
        if Path(filename).name != filename or "\\" in filename or len(filename) > 200:
            raise HTTPException(400, "Invalid filename")
        _path_format(Path(filename))
        policy = service.policy(project_id)
        staging = service.project_root(project_id) / "uploads" / secrets.token_hex(16)
        private_directory(staging)
        temporary = staging / filename
        try:
            with os.fdopen(os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600), "wb") as handle:
                total = 0
                async for block in request.stream():
                    total += len(block)
                    if total > policy.max_upload_bytes:
                        raise HTTPException(413, "Upload exceeds project limit")
                    handle.write(block)
            record = await run_in_threadpool(service.register, project_id, temporary)
            return {k: v for k, v in record.items() if k != "path"}
        finally:
            temporary.unlink(missing_ok=True)
            staging.rmdir()

    @app.delete("/api/v1/datasets/{dataset_id}")
    def delete_dataset(dataset_id: str, project_id: str, confirm: str, request: Request):
        actor = authorize(request, project_id, "administrator")
        if confirm != dataset_id:
            raise HTTPException(400, "Confirmation must match dataset ID")
        # Preserve source snapshots required for historical comparison/resume.
        if any(dataset_id in (r["spec"]["dataset_id"], r["spec"].get("validation_dataset_id")) for r in store.list(project_id, 1000)):
            raise RunConflict("Dataset is referenced by run history; deletion is blocked")
        path = service.dataset_path(project_id, dataset_id)
        if path.is_dir():
            raise RunConflict("Directory dataset deletion requires local administrator review")
        store.delete_dataset(project_id, dataset_id, actor)
        path.unlink()
        return {"deleted": dataset_id}

    @app.get("/api/v1/projects/{project_id}/storage")
    def storage(project_id: str, request: Request):
        authorize(request, project_id)
        root = service.project_root(project_id)
        return {"bytes": sum(contained_path(p, root).stat().st_size for p in root.rglob("*") if p.is_file()),
                "automatic_deletion": False}

    @app.post("/api/v1/runs", status_code=202)
    def submit(spec: RunSpec, request: Request, parent_run_id: str | None = None):
        actor = authorize(request, spec.project_id, "runner")
        key = request.headers.get("idempotency-key")
        if not key or len(key) > 200:
            raise HTTPException(400, "A bounded Idempotency-Key header is required")
        return service.submit(spec, actor, key, parent=parent_run_id)

    @app.get("/api/v1/runs")
    def list_runs(project_id: str, request: Request):
        authorize(request, project_id)
        return store.list(project_id)

    @app.get("/api/v1/runs/{run_id}")
    def get_run(run_id: str, project_id: str, request: Request):
        authorize(request, project_id)
        return store.get(project_id, run_id)

    @app.get("/api/v1/runs/{run_id}/events")
    def get_events(run_id: str, project_id: str, request: Request):
        authorize(request, project_id)
        store.get(project_id, run_id)
        return store.run_events(project_id, run_id)

    @app.get("/api/v1/runs/{run_id}/lineage")
    def lineage(run_id: str, project_id: str, request: Request):
        authorize(request, project_id)
        return service.lineage(project_id, run_id)

    @app.post("/api/v1/projects/{project_id}/webhooks", status_code=201)
    def create_webhook(project_id: str, body: WebhookCreate, request: Request):
        actor = authorize(request, project_id, "administrator")
        return service.register_webhook(project_id, body.url, body.events, actor)

    @app.get("/api/v1/projects/{project_id}/webhooks")
    def list_webhooks(project_id: str, request: Request):
        authorize(request, project_id)
        return service.list_webhooks(project_id)

    @app.delete("/api/v1/projects/{project_id}/webhooks/{webhook_id}")
    def delete_webhook(project_id: str, webhook_id: str, request: Request):
        actor = authorize(request, project_id, "administrator")
        service.delete_webhook(project_id, webhook_id, actor)
        return {"deleted": webhook_id}

    @app.get("/api/v1/runs/{run_id}/compare")
    def compare(run_id: str, project_id: str, request: Request):
        authorize(request, project_id)
        return service.compare(project_id, run_id)

    @app.get("/api/v1/runs/{run_id}/verify")
    def verify(run_id: str, project_id: str, request: Request):
        authorize(request, project_id)
        return service.verify(project_id, run_id)

    @app.get("/api/v1/runs/{run_id}/artifacts/{name}")
    def artifact(run_id: str, name: str, project_id: str, request: Request):
        authorize(request, project_id)
        path = service.artifact(project_id, run_id, name)
        if path.is_dir():
            raise HTTPException(400, "Use a file output format for browser downloads")
        if is_encrypted(path):
            # Decrypted fully in memory, never written to disk -- nothing to clean up or
            # leak. Bounded by the project's own max_upload_bytes policy, the same order of
            # magnitude already accepted for uploads.
            content = service.artifact_bytes(project_id, run_id, name)
            return Response(content=content, media_type="application/octet-stream",
                headers={"Content-Disposition": f'attachment; filename="{path.name}"'})
        return FileResponse(path, filename=path.name, media_type="application/octet-stream")

    @app.post("/api/v1/runs/{run_id}/cancel")
    def cancel(run_id: str, project_id: str, request: Request):
        return store.cancel(project_id, run_id, authorize(request, project_id, "runner"))

    @app.post("/api/v1/runs/{run_id}/resume", status_code=202)
    def resume(run_id: str, project_id: str, request: Request):
        return service.resume(project_id, run_id, authorize(request, project_id, "runner"))

    @app.post("/internal/worker/claim")
    def claim(request: Request):
        candidates = list(worker(request))
        random.shuffle(candidates)  # no project has a structural priority advantage
        run = next(filter(None, (store.claim(candidate) for candidate in candidates)), None)
        if not run:
            return None
        # Authoritative, from the claimed row itself -- never derived from which candidate was
        # being tried, so this can't drift if the scan above is ever refactored.
        project = run["project_id"]
        try:
            directory, data_key = service.prepare(run)
        except Exception:
            store.finish(project, run["id"], run["lease"], "failed", error="Execution preflight failed")
            service.dispatch_webhooks(project, run["id"], "failed")
            raise
        response = {"run_id": run["id"], "lease": run["lease"], "project_id": project,
                    "seconds": service.policy(project).execution_seconds}
        if data_key is not None:
            # Never persisted anywhere -- lives only in this HTTPS response body and, next,
            # the executor subprocess's own environment (see runner.py's env_overrides).
            response["data_key"] = base64.b64encode(data_key).decode()
        return response

    @app.post("/internal/worker/{run_id}/heartbeat")
    async def heartbeat(run_id: str, request: Request):
        authorized = worker(request)
        body = await request.json()
        project = worker_project(authorized, body)
        return {"cancel": store.heartbeat(project, run_id, body["lease"])}

    @app.post("/internal/worker/{run_id}/finish")
    async def finish(run_id: str, request: Request, background_tasks: BackgroundTasks):
        authorized = worker(request)
        body = await request.json()
        project = worker_project(authorized, body)
        store.heartbeat(project, run_id, body["lease"])
        manifest = None
        if body["status"] == "succeeded":
            path = contained_path(service.run_directory(project, run_id) / "candidate-manifest.json", service.project_root(project))
            if path.stat().st_size > 16 * 1024**2:
                raise RunConflict("Manifest exceeds size limit")
            manifest = json.loads(path.read_text())
            service.verify_manifest(project, run_id, manifest)
        result = store.finish(project, run_id, body["lease"], body["status"], manifest,
            error=None if manifest else "Execution stopped; no dataset published")
        if result["status"] == "succeeded":
            private_json(service.run_directory(project, run_id) / "manifest.json", manifest)
        # Off the request path: a slow/dead webhook endpoint must never hold a worker's finish call.
        background_tasks.add_task(service.dispatch_webhooks, project, run_id, result["status"])
        return {"status": result["status"]}

    return app


def from_environment():
    import yaml
    config = yaml.safe_load(Path(os.environ["BUFFDATA_SERVER_CONFIG"]).read_text())
    if os.getenv("BUFFDATA_DATABASE_URL_FILE"):
        config["database_url"] = Path(os.environ["BUFFDATA_DATABASE_URL_FILE"]).read_text().strip()
    return create_app(Settings(**config))
