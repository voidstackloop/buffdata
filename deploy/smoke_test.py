"""Opt-in live Compose regression using ONLY disposable synthetic credentials/data.

Run from Linux/WSL: .venv/bin/python deploy/smoke_test.py [--keep]
Requires the locally built buffdata-team:local image. The production Compose definition
is reused with temporary mounts, two project workers and an ephemeral loopback port.
No real provider requests are made. Evidence/backups remain in the printed temp folder.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import secrets
import shutil
import socket
import ssl
import subprocess
import tempfile
import time

import httpx
import jwt
import yaml
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import hashes, serialization
from cryptography import x509
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keep", action="store_true", help="Keep the isolated stack for investigation")
    parser.add_argument("--image", default="buffdata-team:local", help="Locally built application image")
    args = parser.parse_args()
    os.umask(0o077)
    root = Path(tempfile.mkdtemp(prefix="buffdata-compose-smoke-"))
    project = "buffdata-smoke-" + secrets.token_hex(6)
    checks = []
    def record(name):
        checks.append(name)
        print("PASS " + name, flush=True)
    def write(name, value):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value)
        return str(path)
    def command(*argv, data=None, timeout=180):
        result = subprocess.run(argv, input=data, capture_output=True, timeout=timeout)
        if result.returncode:
            # Test subprocesses contain only synthetic values, but avoid logging even those.
            raise RuntimeError("Command failed: " + " ".join(argv[:4]) + " (inspect retained test stack)")
        return result.stdout
    compose_path = root / "compose.yaml"
    def compose(*argv, **kwargs):
        return command("docker", "compose", "-p", project, "-f", str(compose_path), *argv, **kwargs)
    def inside(service, code):
        return compose("exec", "-T", service, "python", "-c", code)
    private = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    jwk = json.loads(jwt.algorithms.RSAAlgorithm.to_jwk(private.public_key()))
    jwk.update(kid="smoke", alg="RS256", use="sig")
    with socket.socket() as available:
        available.bind(("127.0.0.1", 0))
        tls_port = available.getsockname()[1]
    public_url = f"https://localhost:{tls_port}"
    issuer = "https://identity:8443"
    ca_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "Disposable BuffData smoke CA")])
    now = datetime.now(timezone.utc)
    ca = x509.CertificateBuilder().subject_name(name).issuer_name(name).public_key(ca_key.public_key()).serial_number(
        x509.random_serial_number()).not_valid_before(now - timedelta(days=1)).not_valid_after(now + timedelta(days=1)).add_extension(
        x509.BasicConstraints(ca=True, path_length=0), critical=True).add_extension(x509.KeyUsage(
        digital_signature=True, content_commitment=False, key_encipherment=False, data_encipherment=False,
        key_agreement=False, key_cert_sign=True, crl_sign=True, encipher_only=False, decipher_only=False), critical=True).sign(ca_key, hashes.SHA256())
    cert = x509.CertificateBuilder().subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "identity")])).issuer_name(
        name).public_key(private.public_key()).serial_number(x509.random_serial_number()).not_valid_before(
        now - timedelta(days=1)).not_valid_after(now + timedelta(days=1)).add_extension(x509.SubjectAlternativeName(
        [x509.DNSName("identity"), x509.DNSName("localhost")]), critical=False).add_extension(x509.BasicConstraints(
        ca=False, path_length=None), critical=True).add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]),
        critical=False).sign(ca_key, hashes.SHA256())
    write("fixture/ca.pem", ca.public_bytes(serialization.Encoding.PEM).decode())
    write("fixture/server.pem", cert.public_bytes(serialization.Encoding.PEM).decode())
    write("fixture/signing.pem", private.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption()).decode())
    def bearer(actor):
        token = jwt.encode({"sub": actor, "iss": issuer, "aud": "buffdata-smoke",
            "exp": int(time.time()) + 3600}, private, algorithm="RS256", headers={"kid": "smoke"})
        return {"Authorization": "Bearer " + token}
    tokens = {p: secrets.token_urlsafe(32) for p in ("alpha", "beta")}
    password = secrets.token_urlsafe(32)
    client_secret = secrets.token_urlsafe(32)
    write("fixture/identity.json", json.dumps({"issuer": issuer, "jwks": {"keys": [jwk]}, "client_id": "smoke-browser",
        "client_secret": client_secret, "redirect_uri": public_url + "/callback"}))
    config = {"root": "/data", "database_url": "overridden", "public_url": public_url,
        "oidc": {"issuer": issuer, "audience": "buffdata-smoke", "jwks_url": issuer + "/oidc/jwks"},
        "client_id": "smoke-browser", "authorization_url": public_url + "/oidc/authorize", "token_url": issuer + "/oidc/token",
        "client_secret_file": "/run/secrets/oidc_client_secret",
        "projects": {p: {"members": {p: "administrator", p + "-viewer": "viewer"},
            "worker_token_sha256": hashlib.sha256(tokens[p].encode()).hexdigest(),
            "policy": {"max_upload_bytes": 1024**2, "execution_seconds": 120}} for p in tokens}}
    base = Path(__file__).resolve().parent
    definition = yaml.safe_load((base / "compose.yaml").read_text())
    definition["name"] = project
    services = definition["services"]
    api = services["api"]
    api["image"] = args.image
    api.pop("build", None)
    api.pop("ports", None)
    api["volumes"] = [write("server.yaml", yaml.safe_dump(config)) + ":/config/server.yaml:ro",
                       str(root / "data") + ":/data", str(root / "fixture" / "ca.pem") + ":/config/ca.pem:ro"]
    api["user"] = f"{os.getuid()}:{os.getgid()}"
    api["environment"]["SSL_CERT_FILE"] = "/config/ca.pem"
    api["environment"]["NO_PROXY"] += ",identity"
    services["identity"] = {"image": api["image"], "command": ["python", "/smoke_identity.py"], "user": api["user"],
        "read_only": True, "cap_drop": ["ALL"], "security_opt": ["no-new-privileges:true"],
        "networks": ["control", "edge"], "ports": [f"127.0.0.1:{tls_port}:8443"],
        "volumes": [str(root / "fixture") + ":/fixture:ro", str(base / "smoke_identity.py") + ":/smoke_identity.py:ro"],
        "mem_limit": "256m", "cpus": 1}
    services["egress"]["volumes"] = [str(base / "squid.conf") + ":/etc/squid/squid.conf:ro"]
    services["ingress"]["ports"] = ["127.0.0.1::8080"]
    services["ingress"]["volumes"] = [str(base / "nginx.conf") + ":/etc/nginx/nginx.conf:ro"]
    definition["secrets"] = {
        "database_url": {"file": write("secrets/database_url", f"postgresql+psycopg://buffdata:{password}@postgres/buffdata")},
        "postgres_password": {"file": write("secrets/postgres_password", password)},
        "oidc_client_secret": {"file": write("secrets/oidc_client_secret", client_secret)},
        "provider_secrets": {"file": write("secrets/provider_secrets.json", "{}")}}
    original_worker = services.pop("worker")
    original_worker["image"] = args.image
    (root / "models").mkdir()
    for p in tokens:
        (root / "data" / p).mkdir(parents=True)
        worker = copy.deepcopy(original_worker)
        worker["user"] = api["user"]
        worker["environment"]["BUFFDATA_WORKER_PROJECT"] = p
        worker["environment"]["BUFFDATA_WORKER_TOKEN_FILE"] = "/run/secrets/worker_" + p
        worker["secrets"] = ["worker_" + p, "provider_secrets"]
        worker["volumes"] = [str(root / "data" / p) + ":/data/" + p, str(root / "models") + ":/models:ro"]
        definition["secrets"]["worker_" + p] = {"file": write("secrets/worker_" + p, tokens[p])}
        services["worker_" + p] = worker
    write("compose.yaml", yaml.safe_dump(definition))
    print(f"Isolated smoke project: {project}\nEvidence: {root}", flush=True)
    started = time.monotonic()
    try:
        compose("config", "--quiet")
        compose("up", "-d", "--wait", timeout=600)
        tls_context = ssl.create_default_context(cafile=str(root / "fixture" / "ca.pem"))
        with httpx.Client(base_url=public_url, verify=tls_context, timeout=45, trust_env=False) as client:
            deadline = time.monotonic() + 90
            while True:
                try:
                    if client.get("/healthz", timeout=3).status_code == 200:
                        break
                except httpx.TransportError:
                    pass
                if time.monotonic() >= deadline:
                    raise AssertionError("API readiness deadline exceeded")
                time.sleep(1)
            def request(method, path, *, actor="alpha", expected=200, **kwargs):
                headers = {**bearer(actor), **kwargs.pop("headers", {})}
                response = client.request(method, path, headers=headers, **kwargs)
                assert response.status_code == expected, f"{method} {path}: {response.status_code} {response.text[:300]}"
                return response
            def run_path(run, action="", p="alpha"):
                return "/api/v1/runs/" + run["id"] + action + "?project_id=" + p
            def complete(run, p="alpha"):
                until = time.monotonic() + 150
                while time.monotonic() < until:
                    current = request("GET", run_path(run, p=p), actor=p).json()
                    if current["status"] not in {"running", "queued"}:
                        assert current["status"] == "succeeded", f"Run ended {current['status']}: {current.get('error')}"
                        return current
                    time.sleep(.5)
                raise AssertionError("Worker completion deadline exceeded")
            assert client.get("/api/v1/projects").status_code == 401
            assert request("GET", "/api/v1/projects").json() == [{"id": "alpha", "role": "administrator"}]
            request("GET", "/api/v1/runs?project_id=beta", expected=403)
            request("GET", "/api/v1/datasets?project_id=beta", expected=403)
            request("GET", "/", headers={"Origin": "https://untrusted.invalid"}, expected=403)
            page = request("GET", "/")
            assert "script-src 'self'" in page.headers["content-security-policy"]
            record("Verified bearer identity, project roles, same-origin policy and dashboard delivery")
            ingress = "http://" + compose("port", "ingress", "8080").decode().strip()
            with httpx.Client(base_url=ingress, trust_env=False, timeout=10) as edge:
                assert edge.get("/healthz").status_code == 200
                assert edge.post("/internal/worker/claim").status_code == 404
                assert edge.get("/api/v1/projects", headers=bearer("alpha")).status_code == 200
            record("Production loopback ingress works and does not expose worker-control routes")
            with httpx.Client(base_url=public_url, verify=tls_context, trust_env=False, follow_redirects=True) as browser:
                logged_in = browser.get("/login")
                assert logged_in.status_code == 200 and 'name="csrf-token"' in logged_in.text
                assert browser.get("/api/v1/projects").json() == [{"id": "alpha", "role": "administrator"}]
                assert browser.post("/logout").status_code == 403
                import re
                csrf = re.search(r'name="csrf-token" content="([^"]+)"', logged_in.text).group(1)
                assert browser.post("/logout", headers={"X-CSRF-Token": csrf}).status_code == 200
                assert browser.get("/api/v1/projects").status_code == 401
            record("Live TLS ingress and synthetic OIDC authorization-code/PKCE, remote JWKS, secure session, CSRF and logout")
            data = b''.join((json.dumps({"id": str(i), "text": f"Synthetic training example number {i} with enough valid content.",
                "label": i % 2}) + "\n").encode() for i in range(2048))
            datasets = {p: request("POST", "/api/v1/datasets?project_id=" + p + "&filename=synthetic.jsonl",
                actor=p, content=data, expected=201).json() for p in tokens}
            payload = {"project_id": "alpha", "dataset_id": datasets["alpha"]["id"],
                "configuration": {"network_policy": "strict", "quality_mode": "off", "scrub_pii": False}}
            request("POST", "/api/v1/runs", json=payload, actor="alpha-viewer", expected=403)
            request("POST", "/api/v1/datasets?project_id=alpha&filename=../bad.jsonl", content=data, expected=400)
            request("POST", "/api/v1/datasets?project_id=alpha&filename=large.jsonl", content=b"x" * (1024**2 + 1), expected=413)
            with ThreadPoolExecutor(max_workers=4) as pool:
                submitted = list(pool.map(lambda _: request("POST", "/api/v1/runs", json=payload,
                    headers={"Idempotency-Key": "concurrent"}, expected=202).json(), range(4)))
            assert len({r["id"] for r in submitted}) == 1
            request("POST", "/api/v1/runs", json={**payload, "output_format": "csv"},
                headers={"Idempotency-Key": "concurrent"}, expected=409)
            finished = complete(submitted[0])
            request("GET", run_path(finished, "/verify"))
            comparison = request("GET", run_path(finished, "/compare")).json()
            assert comparison["original"]["records"] == comparison["generated"]["records"] == 2048
            assert comparison["removed_or_changed"] == comparison["added_or_changed"] == 0
            assert comparison["accuracy"] is None
            output = request("GET", run_path(finished, "/artifacts/output"))
            assert output.headers["content-type"] == "application/octet-stream"
            for suffix in ("", "/events", "/compare", "/artifacts/output"):
                request("GET", run_path(finished, suffix), actor="beta", expected=403)
            record("Real PostgreSQL concurrent idempotency, bounded uploads and 2,048-record worker parity")
            # Stop real workers, then race two authorized claims against PostgreSQL row locks.
            compose("stop", "worker_alpha", "worker_beta")
            pending = request("POST", "/api/v1/runs", json=payload,
                headers={"Idempotency-Key": "claims"}, expected=202).json()
            def claim(_):
                return client.post("/internal/worker/claim", headers={"Authorization": "Bearer " + tokens["alpha"]}).json()
            with ThreadPoolExecutor(max_workers=2) as pool:
                claims = list(pool.map(claim, range(2)))
            assert sum(c is not None for c in claims) == 1
            lease = next(c for c in claims if c)
            rejected_finish = client.post("/internal/worker/" + pending["id"] + "/finish",
                headers={"Authorization": "Bearer " + tokens["beta"]},
                json={"lease": lease["lease"], "status": "succeeded"})
            assert rejected_finish.status_code == 409
            request("POST", run_path(pending, "/cancel"))
            ended = client.post("/internal/worker/" + pending["id"] + "/finish",
                headers={"Authorization": "Bearer " + tokens["alpha"]},
                json={"lease": lease["lease"], "status": "failed"})
            assert ended.status_code == 200 and ended.json()["status"] == "cancelled"
            request("GET", run_path(pending, "/artifacts/output"), expected=409)
            resumed = request("POST", run_path(pending, "/resume"), expected=202).json()
            assert resumed["parent_run_id"] == pending["id"]
            compose("start", "worker_alpha", "worker_beta")
            complete(resumed)
            record("Single project claim, cancellation wins publication, explicit resume preserves prior attempt")
            beta_run = request("POST", "/api/v1/runs", actor="beta", json={**payload,
                "project_id": "beta", "dataset_id": datasets["beta"]["id"]},
                headers={"Idempotency-Key": "beta"}, expected=202).json()
            complete(beta_run, "beta")
            crash_run = request("POST", "/api/v1/runs", json=payload,
                headers={"Idempotency-Key": "worker-crash"}, expected=202).json()
            until = time.monotonic() + 30
            while time.monotonic() < until:
                active = request("GET", run_path(crash_run)).json()
                if active["status"] == "running":
                    break
                assert active["status"] == "queued", "Crash fixture finished before the worker could be stopped"
                time.sleep(.05)
            assert active["status"] == "running"
            compose("kill", "-s", "SIGKILL", "worker_alpha")
            compose("stop", "worker_alpha")
            assert request("GET", run_path(crash_run)).json()["status"] == "running"
            request("POST", run_path(crash_run, "/resume"), expected=409)
            stale_at = time.monotonic() + 122
            # Start an idle replacement: it must not reclaim or auto-retry the charged attempt.
            compose("start", "worker_alpha")
            inside("worker_alpha", "import os; from pathlib import Path; assert os.getuid()!=0; "
                "assert not Path('/data/beta').exists(); assert not Path('/run/secrets/database_url').exists(); "
                "assert not Path('/var/run/docker.sock').exists(); "
                "assert all(x not in os.environ for x in ('PGPASSWORD','DATABASE_URL','BUFFDATA_DATABASE_URL_FILE'))")
            for service in ("api", "worker_alpha", "worker_beta"):
                identifier = compose("ps", "-q", service).decode().strip()
                info = json.loads(command("docker", "inspect", identifier))[0]
                assert info["HostConfig"]["ReadonlyRootfs"] and not info["HostConfig"]["Privileged"]
                assert "ALL" in info["HostConfig"]["CapDrop"]
                assert "no-new-privileges:true" in info["HostConfig"]["SecurityOpt"]
                assert info["HostConfig"]["Memory"] > 0 and info["HostConfig"]["NanoCpus"] > 0
            sbom = inside("api", "import json; from buffdata.governance.sbom import generate_sbom; print(json.dumps(generate_sbom()))")
            assert json.loads(sbom)["bomFormat"] == "CycloneDX"
            (root / "sbom.json").write_bytes(sbom)
            record("Two live project workers; non-root/read-only restrictions, volume and database-secret isolation")
            network_probe = """
import socket, httpx
for host, port in [('1.1.1.1',443), ('169.254.169.254',80), ('postgres',5432)]:
    try:
        with socket.create_connection((host,port), timeout=3):
            raise AssertionError('Direct network escape')
    except OSError:
        pass
with httpx.Client(proxy='http://egress:3128', timeout=20) as c:
    for target in ('https://example.com','https://169.254.169.254','https://127.0.0.1'):
        try:
            c.get(target)
            raise AssertionError('Proxy permitted denied destination')
        except httpx.ProxyError as exc:
            assert '403' in str(exc)
    response = c.head('https://huggingface.co')
    assert response.status_code < 500
"""
            inside("worker_alpha", network_probe)
            record("Direct internet/metadata/database blocked; proxy denies unapproved hosts and permits approved HTTPS")
            print("Waiting for the real stale-worker lease window (no simulated clock).", flush=True)
            while time.monotonic() < stale_at:
                assert request("GET", run_path(crash_run)).json()["status"] == "running"
                time.sleep(2)
            recovered = request("POST", run_path(crash_run, "/resume"), expected=202).json()
            assert request("GET", run_path(crash_run)).json()["status"] == "interrupted"
            request("GET", run_path(crash_run, "/artifacts/output"), expected=409)
            complete(recovered)
            record("Actual worker SIGKILL, no automatic retry, stale-lease fencing and successful explicit recovery")
            compose("restart", "api", "postgres")
            until = time.monotonic() + 90
            while time.monotonic() < until:
                try:
                    if client.get("/healthz", timeout=3).status_code == 200:
                        break
                except httpx.TransportError:
                    pass
                time.sleep(1)
            request("GET", run_path(finished, "/verify"))
            record("API/PostgreSQL restart preserves verified run history")
            # Quiescent backup and restore into another database; originals are never overwritten.
            compose("stop", "worker_alpha", "worker_beta", "api")
            backup = compose("exec", "-T", "postgres", "pg_dump", "-U", "buffdata", "-d", "buffdata", "--no-owner")
            (root / "metadata-backup.sql").write_bytes(backup)
            shutil.copytree(root / "data", root / "restored-data")
            compose("exec", "-T", "postgres", "createdb", "-U", "buffdata", "smoke_restore")
            compose("exec", "-T", "postgres", "psql", "-U", "buffdata", "-d", "smoke_restore", "-v", "ON_ERROR_STOP=1", data=backup)
            original_counts = compose("exec", "-T", "postgres", "psql", "-U", "buffdata", "-d", "buffdata", "-Atc",
                "SELECT count(*) FROM managed_runs; SELECT count(*) FROM managed_events;")
            restored_counts = compose("exec", "-T", "postgres", "psql", "-U", "buffdata", "-d", "smoke_restore", "-Atc",
                "SELECT count(*) FROM managed_runs; SELECT count(*) FROM managed_events;")
            assert original_counts == restored_counts
            # Boot the API against the restored database AND copied artifacts and verify all completed runs.
            write("secrets/database_url", f"postgresql+psycopg://buffdata:{password}@postgres/smoke_restore")
            api["volumes"][1] = str(root / "restored-data") + ":/data"
            write("compose.yaml", yaml.safe_dump(definition))
            compose("up", "-d", "--no-deps", "--force-recreate", "api")
            until = time.monotonic() + 90
            while time.monotonic() < until:
                try:
                    if client.get("/healthz", timeout=3).status_code == 200:
                        break
                except httpx.TransportError:
                    pass
                time.sleep(1)
            for p in tokens:
                for run in request("GET", "/api/v1/runs?project_id=" + p, actor=p).json():
                    if run["status"] == "succeeded":
                        request("GET", run_path(run, "/verify", p), actor=p)
            record("Quiescent PostgreSQL/artifact backup restored separately; completed manifests verified through restored API")
        write("results.json", json.dumps({"status": "passed", "passed": checks, "seconds": round(time.monotonic() - started, 2),
            "image_id": command("docker", "image", "inspect", args.image, "--format", "{{.Id}}").decode().strip(),
            "limitations": ["Synthetic TLS/OIDC issuer; organizational identity and ingress still require target-environment validation",
                "Offline optimizer smoke: no paid provider inference or new accuracy claim",
                "Single-host internal teams; not a hostile multi-tenant sandbox"]}, indent=2))
        print("All smoke checks passed. Results: " + str(root / "results.json"), flush=True)
    except Exception as exc:
        write("results.json", json.dumps({"status": "failed", "passed": checks,
            "failure_type": type(exc).__name__, "seconds": round(time.monotonic() - started, 2)}, indent=2))
        raise
    finally:
        if not args.keep:
            # Only the random, freshly created test project is removed. Never use production's name.
            assert project.startswith("buffdata-smoke-") and compose_path.parent == root
            compose("down", "--volumes", timeout=180)
            print("Removed disposable test containers/networks/volume; retained evidence and backups at " + str(root), flush=True)
        else:
            print("Test stack retained (--keep): " + project, flush=True)


if __name__ == "__main__":
    main()
