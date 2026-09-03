from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import random
import time
from pathlib import Path
import jwt
import pytest
from fastapi.testclient import TestClient
from buffdata.runs.service import RunService
from buffdata.server.app import Settings, create_app
from buffdata.governance.oidc import OIDCConfig
from tests.test_oidc import keypair, ISSUER, AUDIENCE, KID


@pytest.fixture
def team(tmp_path, keypair):
    private, jwks = keypair
    settings = Settings(root=tmp_path / "artifacts", database_url="sqlite:///" + str(tmp_path / "metadata.db"),
        public_url="https://buffdata.example", oidc=OIDCConfig(issuer=ISSUER, audience=AUDIENCE, jwks=jwks),
        client_id="browser", authorization_url=ISSUER + "/authorize", token_url=ISSUER + "/token",
        projects={"alpha": {"members": {"alice": "administrator", "reader": "viewer"}, "policy": {},
                            "worker_token_sha256": hashlib.sha256(b"worker-alpha").hexdigest()},
                  "beta": {"members": {"bob": "administrator"}, "policy": {}}})
    app = create_app(settings)
    def token(actor):
        return {"Authorization": "Bearer " + jwt.encode({"sub": actor, "iss": ISSUER, "aud": AUDIENCE,
            "exp": int(time.time()) + 60}, private, algorithm="RS256", headers={"kid": KID})}
    return TestClient(app, base_url="https://buffdata.example"), token, app.state.service


def test_server_requires_identity_and_project_scope(team):
    client, token, _ = team
    assert client.get("/api/v1/projects").status_code == 401
    assert client.get("/api/v1/projects", headers=token("alice")).json() == [{"id": "alpha", "role": "administrator"}]
    assert client.get("/api/v1/runs?project_id=beta", headers=token("alice")).status_code == 403


def test_upload_submit_claim_and_cancel(team):
    client, token, manager = team
    headers = token("alice")
    data = client.post("/api/v1/datasets?project_id=alpha&filename=data.jsonl", headers=headers,
                       content=b'{"id":"1","text":"a simple valid sentence","label":0}\n')
    assert data.status_code == 201, data.text
    payload = {"project_id": "alpha", "dataset_id": data.json()["id"],
               "configuration": {"network_policy": "strict", "quality_mode": "off", "scrub_pii": False}}
    headers["Idempotency-Key"] = "same"
    run = client.post("/api/v1/runs", json=payload, headers=headers)
    assert run.status_code == 202, run.text
    assert client.post("/api/v1/runs", json=payload, headers=headers).json()["id"] == run.json()["id"]
    assert client.post("/api/v1/runs", json=payload, headers={**token("reader"), "Idempotency-Key": "reader"}).status_code == 403
    claim = client.post("/internal/worker/claim", headers={"Authorization": "Bearer worker-alpha"})
    assert claim.status_code == 200 and claim.json()["project_id"] == "alpha", claim.text
    cancelled = client.post("/api/v1/runs/" + run.json()["id"] + "/cancel?project_id=alpha", headers=headers)
    assert cancelled.json()["cancel_requested"]


def test_lineage_endpoint_tracks_explicit_parent_and_rejects_unknown_parent(team):
    client, token, _ = team
    headers = token("alice")
    data = client.post("/api/v1/datasets?project_id=alpha&filename=data.jsonl", headers=headers,
                       content=b'{"id":"1","text":"a simple valid sentence","label":0}\n')
    payload = {"project_id": "alpha", "dataset_id": data.json()["id"],
               "configuration": {"network_policy": "strict", "quality_mode": "off", "scrub_pii": False}}
    first = client.post("/api/v1/runs", json=payload, headers={**headers, "Idempotency-Key": "first"})
    first_id = first.json()["id"]
    second = client.post("/api/v1/runs", json=payload, params={"parent_run_id": first_id},
                         headers={**headers, "Idempotency-Key": "second"})
    assert second.status_code == 202, second.text
    second_id = second.json()["id"]

    forward = client.get(f"/api/v1/runs/{second_id}/lineage?project_id=alpha", headers=headers).json()
    assert [a["run_id"] for a in forward["ancestors"]] == [first_id]
    backward = client.get(f"/api/v1/runs/{first_id}/lineage?project_id=alpha", headers=headers).json()
    assert [d["run_id"] for d in backward["descendants"]] == [second_id]

    missing_parent = client.post("/api/v1/runs", json=payload, params={"parent_run_id": "a" * 32},
                                 headers={**headers, "Idempotency-Key": "third"})
    assert missing_parent.status_code == 404


def test_webhook_registration_is_administrator_only_and_dispatches_on_finish(team, monkeypatch):
    client, token, _ = team
    headers = token("alice")
    calls = []

    class FakeResponse:
        status_code = 200

    monkeypatch.setattr(RunService, "_send_webhook",
        staticmethod(lambda url, body, headers: calls.append((url, body, headers)) or FakeResponse()))

    denied = client.post("/api/v1/projects/alpha/webhooks", json={"url": "https://93.184.216.34/hook",
        "events": ["run.failed"]}, headers=token("reader"))
    assert denied.status_code == 403

    created = client.post("/api/v1/projects/alpha/webhooks", json={"url": "https://93.184.216.34/hook",
        "events": ["run.failed"]}, headers=headers)
    assert created.status_code == 201, created.text
    assert "secret" in created.json()
    listed = client.get("/api/v1/projects/alpha/webhooks", headers=headers).json()
    assert listed[0]["id"] == created.json()["id"] and "secret" not in listed[0]

    data = client.post("/api/v1/datasets?project_id=alpha&filename=data.jsonl", headers=headers,
                       content=b'{"id":"1","text":"a simple valid sentence","label":0}\n')
    payload = {"project_id": "alpha", "dataset_id": data.json()["id"],
               "configuration": {"network_policy": "strict", "quality_mode": "off", "scrub_pii": False}}
    headers["Idempotency-Key"] = "webhook-run"
    run = client.post("/api/v1/runs", json=payload, headers=headers)
    run_id = run.json()["id"]
    claim = client.post("/internal/worker/claim", headers={"Authorization": "Bearer worker-alpha"})
    assert claim.json()["run_id"] == run_id
    finish = client.post(f"/internal/worker/{run_id}/finish",
        json={"lease": claim.json()["lease"], "status": "failed"}, headers={"Authorization": "Bearer worker-alpha"})
    assert finish.status_code == 200 and finish.json()["status"] == "failed"

    assert len(calls) == 1
    sent_payload = json.loads(calls[0][1])
    assert sent_payload["event"] == "run.failed" and sent_payload["run_id"] == run_id
    events = client.get(f"/api/v1/runs/{run_id}/events?project_id=alpha", headers=headers).json()
    assert "webhook.delivered" in [e["kind"] for e in events]

    deleted = client.delete(f"/api/v1/projects/alpha/webhooks/{created.json()['id']}", headers=headers)
    assert deleted.status_code == 200
    assert client.get("/api/v1/projects/alpha/webhooks", headers=headers).json() == []


@pytest.fixture
def pooled(tmp_path, keypair):
    private, jwks = keypair
    pool_hash = hashlib.sha256(b"pool-token").hexdigest()
    settings = Settings(root=tmp_path / "artifacts", database_url="sqlite:///" + str(tmp_path / "metadata.db"),
        public_url="https://buffdata.example", oidc=OIDCConfig(issuer=ISSUER, audience=AUDIENCE, jwks=jwks),
        client_id="browser", authorization_url=ISSUER + "/authorize", token_url=ISSUER + "/token",
        projects={"alpha": {"members": {"alice": "administrator"}, "policy": {},
                            "worker_token_sha256": pool_hash, "worker_pool": "shared"},
                  "beta": {"members": {"alice": "administrator"}, "policy": {},
                           "worker_token_sha256": pool_hash, "worker_pool": "shared"}})
    app = create_app(settings)
    def token(actor):
        return {"Authorization": "Bearer " + jwt.encode({"sub": actor, "iss": ISSUER, "aud": AUDIENCE,
            "exp": int(time.time()) + 60}, private, algorithm="RS256", headers={"kid": KID})}
    return TestClient(app, base_url="https://buffdata.example"), token, app.state.service


def _submit_run(client, headers, project, key):
    data = client.post(f"/api/v1/datasets?project_id={project}&filename=data.jsonl", headers=headers,
                       content=b'{"id":"1","text":"a simple valid sentence","label":0}\n')
    assert data.status_code == 201, data.text
    payload = {"project_id": project, "dataset_id": data.json()["id"],
               "configuration": {"network_policy": "strict", "quality_mode": "off", "scrub_pii": False}}
    run = client.post("/api/v1/runs", json=payload, headers={**headers, "Idempotency-Key": key})
    assert run.status_code == 202, run.text
    return run.json()["id"]


def test_worker_pool_requires_matching_pool_declaration(tmp_path, keypair):
    private, jwks = keypair
    pool_hash = hashlib.sha256(b"pool-token").hexdigest()
    base = dict(root=tmp_path / "artifacts", database_url="sqlite:///" + str(tmp_path / "metadata.db"),
        public_url="https://buffdata.example", oidc=OIDCConfig(issuer=ISSUER, audience=AUDIENCE, jwks=jwks),
        client_id="browser", authorization_url=ISSUER + "/authorize", token_url=ISSUER + "/token")

    with pytest.raises(ValueError, match="worker_pool"):
        create_app(Settings(**base, projects={
            "alpha": {"members": {}, "policy": {}, "worker_token_sha256": pool_hash, "worker_pool": "shared"},
            "beta": {"members": {}, "policy": {}, "worker_token_sha256": pool_hash}}))  # no worker_pool at all

    with pytest.raises(ValueError, match="worker_pool"):
        create_app(Settings(**base, projects={
            "alpha": {"members": {}, "policy": {}, "worker_token_sha256": pool_hash, "worker_pool": "shared"},
            "beta": {"members": {}, "policy": {}, "worker_token_sha256": pool_hash, "worker_pool": "other"}}))

    create_app(Settings(**base, projects={
        "alpha": {"members": {}, "policy": {}, "worker_token_sha256": pool_hash, "worker_pool": "shared"},
        "beta": {"members": {}, "policy": {}, "worker_token_sha256": pool_hash, "worker_pool": "shared"}}))


def test_pooled_worker_claims_across_both_projects(pooled):
    client, token, _ = pooled
    headers = token("alice")
    alpha_run = _submit_run(client, headers, "alpha", "alpha-1")
    beta_run = _submit_run(client, headers, "beta", "beta-1")

    pool_headers = {"Authorization": "Bearer pool-token"}
    first = client.post("/internal/worker/claim", headers=pool_headers).json()
    second = client.post("/internal/worker/claim", headers=pool_headers).json()
    # alpha's slot is occupied by whichever run the first call claimed, so the second call
    # is structurally forced to the other project -- deterministic with exactly one queued
    # run per project, regardless of the random scan order.
    assert {first["run_id"], second["run_id"]} == {alpha_run, beta_run}
    assert {first["project_id"], second["project_id"]} == {"alpha", "beta"}
    assert client.post("/internal/worker/claim", headers=pool_headers).json() is None


def test_per_project_cap_holds_under_concurrent_pooled_claims(pooled):
    client, token, _ = pooled
    headers = token("alice")
    for i in range(4):
        _submit_run(client, headers, "alpha", f"alpha-{i}")
        _submit_run(client, headers, "beta", f"beta-{i}")

    pool_headers = {"Authorization": "Bearer pool-token"}
    with ThreadPoolExecutor(8) as pool:
        results = list(pool.map(lambda _: client.post("/internal/worker/claim", headers=pool_headers).json(),
                                 range(8)))
    claimed = [r for r in results if r]
    projects = [r["project_id"] for r in claimed]
    assert projects.count("alpha") == 1 and projects.count("beta") == 1
    assert len(claimed) == 2  # only one active run per project, exactly as without pooling


def test_preflight_failure_binds_the_actually_claimed_project(pooled, monkeypatch):
    client, token, manager = pooled
    headers = token("alice")
    beta_run = _submit_run(client, headers, "beta", "beta-only")  # alpha's queue stays empty

    original_prepare = manager.prepare
    def failing_prepare(run):
        assert run["project_id"] == "beta"
        raise RuntimeError("simulated preflight failure")
    monkeypatch.setattr(manager, "prepare", failing_prepare)

    response = client.post("/internal/worker/claim", headers={"Authorization": "Bearer pool-token"})
    assert response.status_code == 500  # the preflight exception propagates, as it does today
    assert manager.store.get("beta", beta_run)["status"] == "failed"


def test_pooled_claiming_avoids_deterministic_starvation(pooled):
    client, token, _ = pooled
    headers = token("alice")
    for i in range(8):
        _submit_run(client, headers, "alpha", f"alpha-{i}")
        _submit_run(client, headers, "beta", f"beta-{i}")

    random.seed(20260903)
    pool_headers = {"Authorization": "Bearer pool-token"}
    seen = set()
    for _ in range(16):
        run = client.post("/internal/worker/claim", headers=pool_headers).json()
        if not run:
            continue
        seen.add(run["project_id"])
        client.post(f"/internal/worker/{run['run_id']}/finish",
            json={"lease": run["lease"], "project_id": run["project_id"], "status": "failed"}, headers=pool_headers)
    assert seen == {"alpha", "beta"}


def test_heartbeat_and_finish_reject_project_outside_pool(pooled):
    client, token, _ = pooled
    headers = token("alice")
    alpha_run = _submit_run(client, headers, "alpha", "alpha-only")
    pool_headers = {"Authorization": "Bearer pool-token"}
    claim = client.post("/internal/worker/claim", headers=pool_headers).json()
    assert claim["run_id"] == alpha_run

    denied = client.post(f"/internal/worker/{alpha_run}/heartbeat",
        json={"lease": claim["lease"], "project_id": "not-a-real-project"}, headers=pool_headers)
    assert denied.status_code == 403

    # Authorized for this token, but not the project this particular run belongs to.
    mismatched = client.post(f"/internal/worker/{alpha_run}/heartbeat",
        json={"lease": claim["lease"], "project_id": "beta"}, headers=pool_headers)
    assert mismatched.status_code == 409

    # The real run is unaffected by the rejected attempts and still reachable correctly.
    ok = client.post(f"/internal/worker/{alpha_run}/heartbeat",
        json={"lease": claim["lease"], "project_id": "alpha"}, headers=pool_headers)
    assert ok.status_code == 200 and ok.json() == {"cancel": False}


def test_encrypted_artifact_download_decrypts_over_http(team):
    from buffdata.runs.models import RunSpec
    from buffdata.runs.runner import run_local
    from buffdata.security.keys import LocalKeyManager, is_encrypted
    import os
    client, token, manager = team
    manager.key_manager = LocalKeyManager(os.urandom(32))
    headers = token("alice")
    data = client.post("/api/v1/datasets?project_id=alpha&filename=data.jsonl", headers=headers,
                       content=b'{"id":"1","text":"a simple valid sentence","label":0}\n')
    spec = RunSpec(project_id="alpha", dataset_id=data.json()["id"],
                   configuration={"network_policy": "strict", "quality_mode": "off", "scrub_pii": False})
    run = manager.submit(spec)
    result = run_local(manager, "alpha", run["id"])
    assert result["status"] == "succeeded", result
    assert is_encrypted(manager.artifact("alpha", run["id"], "output"))

    response = client.get(f"/api/v1/runs/{run['id']}/artifacts/output?project_id=alpha", headers=headers)
    assert response.status_code == 200
    assert b'"id"' in response.content and b"BFEK1" not in response.content
    assert "attachment" in response.headers["content-disposition"]


def test_origin_csrf_and_safe_dashboard(team):
    client, token, manager = team
    assert client.get("/api/v1/projects", headers={**token("alice"), "Origin": "https://evil.example"}).status_code == 403
    manager.store.save_session("browser-session", {"actor": "alice", "csrf": "test-csrf"})
    client.cookies.set("__Host-buffdata", "browser-session")
    assert client.post("/logout").status_code == 403
    assert client.get("/").status_code == 200
    assert "script-src 'self'" in client.get("/").headers["content-security-policy"]
    assert client.post("/logout", headers={"X-CSRF-Token": "test-csrf"}).status_code == 200


def test_upload_path_and_size_limits(team):
    client, token, manager = team
    assert client.post("/api/v1/datasets?project_id=alpha&filename=../bad.json", headers=token("alice"), content=b"[]").status_code == 400
    body = manager.store.project("alpha")
    body["policy"] = {"max_upload_bytes": 1}
    manager.store.update_project("alpha", body, "alice")
    assert client.post("/api/v1/datasets?project_id=alpha&filename=data.json", headers=token("alice"), content=b"[]").status_code == 413


def test_browser_pkce_nonce_login_and_state_replay(team, keypair, monkeypatch):
    from urllib.parse import urlsplit, parse_qs
    client, _, manager = team
    login = client.get("/login", follow_redirects=False)
    assert login.status_code == 302
    query = parse_qs(urlsplit(login.headers["location"]).query)
    assert query["code_challenge_method"] == ["S256"]
    state = query["state"][0]
    pending = manager.store.session(state)
    private, _ = keypair
    async def fetch(*args, **kwargs):
        assert kwargs["code_verifier"] == pending["verifier"]
        return {"id_token": jwt.encode({"sub": "alice", "iss": ISSUER, "aud": "browser",
            "exp": int(time.time()) + 60, "nonce": pending["nonce"]}, private,
            algorithm="RS256", headers={"kid": KID})}
    monkeypatch.setattr("authlib.integrations.httpx_client.AsyncOAuth2Client.fetch_token", fetch)
    path = "/callback?code=synthetic&state=" + state
    callback = client.get(path, follow_redirects=False)
    assert callback.status_code == 302, callback.text
    assert "Secure" in callback.headers["set-cookie"] and "HttpOnly" in callback.headers["set-cookie"]
    assert client.get("/api/v1/projects").status_code == 200
    assert client.get(path, follow_redirects=False).status_code == 401


def test_readiness_detects_metadata_outage_without_leaking_credentials(team, monkeypatch):
    client, _, manager = team
    assert client.get("/healthz").status_code == 200
    def unavailable():
        raise RuntimeError("private-database-password")
    monkeypatch.setattr(manager.store.engine, "connect", unavailable)
    response = client.get("/healthz")
    assert response.status_code == 503
    assert "private-database-password" not in response.text
