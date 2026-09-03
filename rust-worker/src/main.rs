//! A drop-in replacement for `buffdata/server/worker.py`: polls the control plane for a
//! claimed run, supervises the (still-Python) `buffdata.runs.executor` subprocess that does
//! the actual dataset optimization, and reports heartbeat/finish. This binary owns none of
//! the optimization logic itself -- it is only the claim loop and process supervisor,
//! rewritten for a small, fast, dependency-light process instead of a Python one.
//!
//! Same wire contract as the Python worker (see buffdata/server/app.py's
//! /internal/worker/{claim,heartbeat,finish} handlers): a claimed run's data key (encryption
//! at rest, opt-in) never touches this process's own persistent environment or disk -- it is
//! forwarded straight into the executor subprocess's environment for that one run only.

mod path_guard;
mod secrets;
mod supervise;

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::Duration;

use serde::Deserialize;
use serde_json::{json, Value};

const IDLE_POLL_INTERVAL: Duration = Duration::from_secs(2);
const ERROR_BACKOFF: Duration = Duration::from_secs(5);

const ENV_PREFIX_DENYLIST: &[&str] = &["BUFFDATA_WORKER_", "BUFFDATA_DATABASE_"];
const ENV_KEY_DENYLIST: &[&str] = &[
    "BUFFDATA_SERVER_CONFIG",
    "BUFFDATA_SERVER_URL",
    "DATABASE_URL",
    "PGPASSWORD",
    "BUFFDATA_ARTIFACT_MASTER_KEY_SECRET",
];

#[derive(Deserialize)]
struct ClaimResponse {
    run_id: String,
    lease: String,
    project_id: String,
    seconds: u64,
    #[serde(default)]
    data_key: Option<String>,
}

#[derive(Deserialize)]
struct HeartbeatResponse {
    cancel: bool,
}

struct Config {
    server_url: String,
    token: String,
    artifact_root: PathBuf,
    provider_secrets: Value,
    python: String,
}

fn load_config() -> Result<Config, String> {
    let server_url = std::env::var("BUFFDATA_SERVER_URL")
        .map_err(|_| "BUFFDATA_SERVER_URL is required".to_string())?;
    let token_file = std::env::var("BUFFDATA_WORKER_TOKEN_FILE")
        .map_err(|_| "BUFFDATA_WORKER_TOKEN_FILE is required".to_string())?;
    let token = std::fs::read_to_string(&token_file)
        .map_err(|e| format!("cannot read BUFFDATA_WORKER_TOKEN_FILE ({token_file}): {e}"))?
        .trim()
        .to_string();
    if token.is_empty() {
        return Err("worker token file is empty".to_string());
    }
    let artifact_root = PathBuf::from(
        std::env::var("BUFFDATA_ARTIFACT_ROOT").unwrap_or_else(|_| "/data".to_string()),
    );
    let provider_secrets = match std::env::var("BUFFDATA_PROVIDER_SECRETS_FILE") {
        Ok(path) => {
            let text = std::fs::read_to_string(&path)
                .map_err(|e| format!("cannot read BUFFDATA_PROVIDER_SECRETS_FILE ({path}): {e}"))?;
            serde_json::from_str(&text)
                .map_err(|e| format!("invalid JSON in provider secrets file ({path}): {e}"))?
        }
        Err(_) => Value::Object(Default::default()),
    };
    // Not present in the Python worker, which always re-execs via sys.executable (always
    // correct by construction there). A plain Rust binary has no such interpreter of its
    // own to defer to, so this points it at the same one buffdata is actually installed
    // into -- defaults to whatever `python3` resolves to on PATH.
    let python = std::env::var("BUFFDATA_PYTHON").unwrap_or_else(|_| "python3".to_string());
    Ok(Config {
        server_url,
        token,
        artifact_root,
        provider_secrets,
        python,
    })
}

fn build_child_env(overrides: &BTreeMap<String, String>) -> Vec<(String, String)> {
    let mut env: BTreeMap<String, String> = std::env::vars()
        .filter(|(k, _)| {
            !ENV_PREFIX_DENYLIST
                .iter()
                .any(|prefix| k.starts_with(prefix))
                && !ENV_KEY_DENYLIST.contains(&k.as_str())
                && !secrets::ALLOWED_SECRETS.contains(&k.as_str())
        })
        .collect();
    for (k, v) in overrides {
        env.insert(k.clone(), v.clone());
    }
    env.into_iter().collect()
}

fn post_json(
    client: &reqwest::blocking::Client,
    url: &str,
    token: &str,
    body: &Value,
) -> Result<Value, String> {
    let response = client
        .post(url)
        .bearer_auth(token)
        .json(body)
        .send()
        .map_err(|e| format!("request to {url} failed: {e}"))?;
    let response = response
        .error_for_status()
        .map_err(|e| format!("request to {url} failed: {e}"))?;
    response
        .json::<Value>()
        .map_err(|e| format!("invalid JSON response from {url}: {e}"))
}

/// One claim attempt. `Ok(true)` means a run was claimed and carried through to a finish
/// report (success or failure of the *run itself* is reported to the control plane, not
/// returned here); `Ok(false)` means nothing was queued; `Err` is a control-plane or
/// preflight failure for which the run (if any was claimed) is deliberately left for the
/// server's own stale-lease recovery, exactly like the Python worker's bare `except Exception`.
fn run_claim_iteration(client: &reqwest::blocking::Client, cfg: &Config) -> Result<bool, String> {
    let claim_url = format!("{}/internal/worker/claim", cfg.server_url);
    let body = post_json(client, &claim_url, &cfg.token, &json!({}))?;
    if body.is_null() {
        return Ok(false);
    }
    let claim: ClaimResponse =
        serde_json::from_value(body).map_err(|e| format!("invalid claim response: {e}"))?;

    if !path_guard::valid_run_id(&claim.run_id) {
        return Err("server returned a malformed run id".to_string());
    }
    let project_root = cfg.artifact_root.join(&claim.project_id);
    let candidate_directory = project_root.join("runs").join(&claim.run_id);
    let directory = path_guard::contained_path(&candidate_directory, &project_root)
        .map_err(|e| format!("run directory outside authorized project: {e}"))?;

    let mut overrides = secrets::secrets_for_project(&cfg.provider_secrets, &claim.project_id)?;
    if let Some(data_key) = &claim.data_key {
        // Never written to disk anywhere in this process -- lives only in the claim
        // response body (already received) and, next, the executor subprocess's own
        // environment, gone the moment that short-lived process exits.
        overrides.insert("BUFFDATA_RUN_DATA_KEY".to_string(), data_key.clone());
    }

    let mut command = Command::new(&cfg.python);
    command
        .arg("-m")
        .arg("buffdata.runs.executor")
        .arg(&directory)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .env_clear()
        .envs(build_child_env(&overrides));

    let heartbeat_url = format!(
        "{}/internal/worker/{}/heartbeat",
        cfg.server_url, claim.run_id
    );
    let heartbeat = || -> Result<bool, String> {
        let body = json!({"lease": claim.lease, "project_id": claim.project_id});
        let response = post_json(client, &heartbeat_url, &cfg.token, &body)?;
        let parsed: HeartbeatResponse = serde_json::from_value(response)
            .map_err(|e| format!("invalid heartbeat response: {e}"))?;
        Ok(parsed.cancel)
    };

    let status = supervise::supervise(command, claim.seconds, heartbeat)?;

    let finish_url = format!("{}/internal/worker/{}/finish", cfg.server_url, claim.run_id);
    let finish_body =
        json!({"lease": claim.lease, "project_id": claim.project_id, "status": status});
    post_json(client, &finish_url, &cfg.token, &finish_body)?;
    Ok(true)
}

fn main() {
    let cfg = load_config().unwrap_or_else(|e| {
        eprintln!("buffdata-worker: fatal: {e}");
        std::process::exit(1);
    });
    // trust_env-equivalent of false: this client's own control-plane traffic (claim /
    // heartbeat / finish, all to the internal API) must never be routed through whatever
    // egress proxy is configured for the executor subprocess's provider calls.
    let client = reqwest::blocking::Client::builder()
        .no_proxy()
        .redirect(reqwest::redirect::Policy::none())
        .timeout(Duration::from_secs(30))
        .build()
        .expect("failed to build HTTP client");

    loop {
        match run_claim_iteration(&client, &cfg) {
            Ok(true) => {}
            Ok(false) => std::thread::sleep(IDLE_POLL_INTERVAL),
            Err(e) => {
                eprintln!("buffdata-worker: {e}");
                std::thread::sleep(ERROR_BACKOFF);
            }
        }
    }
}
