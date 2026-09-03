"""A worker's project is decided per claimed run, not fixed at process start -- a single
worker token may authorize more than one project (a pool, declared via matching
worker_pool entries in the server config; see buffdata/server/app.py's create_app()).
No database credentials or container-engine socket."""
import os
from pathlib import Path
import time
import httpx
from buffdata.runs.runner import supervise
from buffdata.security.policy import contained_path

ALLOWED_SECRETS = {"GEMINI_API_KEY", "GOOGLE_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY",
                   "AZURE_OPENAI_API_KEY", "OPENAI_COMPATIBLE_API_KEY"}


def secrets_for_project(all_secrets: dict, project: str) -> dict:
    """The provider-secrets file is either the original flat shape (one shared set of
    credentials, for a single project or a pool that intentionally shares one provider
    account) or a per-project nested shape (``{"alpha": {...}, "beta": {...}}``), so pooled
    projects on different provider accounts don't silently share -- or leak -- credentials."""
    if not all_secrets:
        values = {}
    elif all(isinstance(v, dict) for v in all_secrets.values()):
        values = all_secrets.get(project, {})
    elif all(isinstance(v, str) for v in all_secrets.values()):
        values = all_secrets
    else:
        raise ValueError("Invalid provider-secret file")
    if set(values) - ALLOWED_SECRETS or not all(isinstance(v, str) for v in values.values()):
        raise ValueError("Invalid provider-secret file")
    return values


def main():
    # Administrator-mounted project credentials; never supplied by API run requests.
    import json
    secret_file = os.getenv("BUFFDATA_PROVIDER_SECRETS_FILE")
    all_secrets = json.loads(Path(secret_file).read_text()) if secret_file else {}
    root = Path(os.environ.get("BUFFDATA_ARTIFACT_ROOT", "/data"))
    token = Path(os.environ["BUFFDATA_WORKER_TOKEN_FILE"]).read_text().strip()
    with httpx.Client(base_url=os.environ["BUFFDATA_SERVER_URL"], timeout=30, trust_env=False,
                      follow_redirects=False, headers={"Authorization": "Bearer " + token}) as client:
        def post(path, payload=None):
            response = client.post(path, json=payload or {})
            response.raise_for_status()
            return response.json()
        while True:
            try:
                run = post("/internal/worker/claim")
                if not run:
                    time.sleep(2)
                    continue
                project = run["project_id"]
                # Clear every known secret key before applying this run's project, so a
                # previous claimed run's credentials (a different pooled project) can never
                # leak into this one's optimizer subprocess.
                for key in ALLOWED_SECRETS:
                    os.environ.pop(key, None)
                os.environ.update(secrets_for_project(all_secrets, project))
                directory = contained_path(root / project / "runs" / run["run_id"], root / project)
                prefix = "/internal/worker/" + run["run_id"]
                # The data key (encryption at rest, opt-in -- security/keys.py) never touches
                # this worker's own persistent environment or disk: it's forwarded straight
                # into the executor subprocess's environment for this one run only.
                env_overrides = {"BUFFDATA_RUN_DATA_KEY": run["data_key"]} if run.get("data_key") else None
                status, _ = supervise(directory, run["seconds"],
                    lambda: post(prefix + "/heartbeat", {"lease": run["lease"], "project_id": project})["cancel"],
                    env_overrides=env_overrides)
                post(prefix + "/finish", {"lease": run["lease"], "project_id": project, "status": status})
            except Exception:
                # Expired leases / control-plane outages stop the child in supervise's finally.
                # Explicit resume is required; no automatic retries of provider work.
                time.sleep(5)


if __name__ == "__main__":
    main()
