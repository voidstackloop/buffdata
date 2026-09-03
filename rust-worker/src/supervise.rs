//! Mirrors `buffdata/runs/runner.py::supervise()`: spawns a child, polls it every 500ms,
//! heartbeats the control plane on the same cadence, and kills the whole process group
//! (SIGTERM, then SIGKILL after a 30s grace period) on cancellation, deadline, or an error
//! partway through -- deliberately never captures the child's stdout/stderr, since a
//! provider SDK exception or dataset excerpt can contain secrets.

use std::process::Command;
use std::time::{Duration, Instant};

pub const HEARTBEAT_INTERVAL: Duration = Duration::from_millis(500);
pub const KILL_GRACE_PERIOD: Duration = Duration::from_secs(30);

#[cfg(unix)]
pub fn prepare_process_group(command: &mut Command) {
    use std::os::unix::process::CommandExt;
    command.process_group(0);
}

#[cfg(not(unix))]
pub fn prepare_process_group(_command: &mut Command) {}

#[cfg(unix)]
fn kill_group(pid: u32, signal: i32) {
    unsafe {
        libc::kill(-(pid as i32), signal);
    }
}

#[cfg(not(unix))]
fn kill_group(_pid: u32, _signal: i32) {}

/// `heartbeat` returns `Ok(true)` to request cancellation, `Ok(false)` to continue, or
/// `Err` on a control-plane failure -- which still runs the same kill-on-the-way-out cleanup
/// as the happy paths (Python's `finally` guarantee) before the error is returned.
pub fn supervise(
    mut command: Command,
    seconds: u64,
    mut heartbeat: impl FnMut() -> Result<bool, String>,
) -> Result<String, String> {
    prepare_process_group(&mut command);
    let mut child = command
        .spawn()
        .map_err(|e| format!("failed to spawn executor subprocess: {e}"))?;
    let pid = child.id();
    let deadline = Instant::now() + Duration::from_secs(seconds);

    let mut status = "failed".to_string();
    let mut heartbeat_error: Option<String> = None;
    loop {
        match child.try_wait() {
            Ok(Some(exit)) => {
                status = if exit.success() {
                    "succeeded"
                } else {
                    "failed"
                }
                .to_string();
                break;
            }
            Ok(None) => {}
            Err(_) => {}
        }
        match heartbeat() {
            Ok(true) => {
                status = "cancelled".to_string();
                break;
            }
            Ok(false) => {}
            Err(e) => {
                heartbeat_error = Some(e);
                break;
            }
        }
        if Instant::now() >= deadline {
            break;
        }
        std::thread::sleep(HEARTBEAT_INTERVAL);
    }

    if matches!(child.try_wait(), Ok(None)) {
        kill_group(pid, libc_sigterm());
        let kill_deadline = Instant::now() + KILL_GRACE_PERIOD;
        loop {
            if matches!(child.try_wait(), Ok(Some(_))) {
                break;
            }
            if Instant::now() >= kill_deadline {
                kill_group(pid, libc_sigkill());
                let _ = child.wait();
                break;
            }
            std::thread::sleep(Duration::from_millis(200));
        }
    }

    match heartbeat_error {
        Some(e) => Err(e),
        None => Ok(status),
    }
}

#[cfg(unix)]
fn libc_sigterm() -> i32 {
    libc::SIGTERM
}
#[cfg(unix)]
fn libc_sigkill() -> i32 {
    libc::SIGKILL
}
#[cfg(not(unix))]
fn libc_sigterm() -> i32 {
    0
}
#[cfg(not(unix))]
fn libc_sigkill() -> i32 {
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sh(script: &str) -> Command {
        let mut c = Command::new("sh");
        c.arg("-c").arg(script);
        c.stdin(std::process::Stdio::null());
        c.stdout(std::process::Stdio::null());
        c.stderr(std::process::Stdio::null());
        c
    }

    #[test]
    fn a_process_that_exits_zero_is_succeeded() {
        let status = supervise(sh("exit 0"), 10, || Ok(false)).unwrap();
        assert_eq!(status, "succeeded");
    }

    #[test]
    fn a_process_that_exits_nonzero_is_failed() {
        let status = supervise(sh("exit 1"), 10, || Ok(false)).unwrap();
        assert_eq!(status, "failed");
    }

    #[test]
    fn a_cancel_signal_from_heartbeat_kills_a_long_running_process_and_reports_cancelled() {
        let start = Instant::now();
        let status = supervise(sh("sleep 30"), 60, || Ok(true)).unwrap();
        assert_eq!(status, "cancelled");
        // Should return almost immediately, not wait out the 30s sleep or the 60s deadline.
        assert!(start.elapsed() < Duration::from_secs(5));
    }

    #[test]
    fn a_deadline_kills_a_long_running_process_and_reports_failed() {
        let start = Instant::now();
        let status = supervise(sh("sleep 30"), 1, || Ok(false)).unwrap();
        assert_eq!(status, "failed");
        assert!(start.elapsed() < Duration::from_secs(5));
    }

    #[test]
    fn a_heartbeat_error_still_kills_the_process_before_propagating() {
        let start = Instant::now();
        let result = supervise(sh("sleep 30"), 60, || {
            Err("control plane unreachable".to_string())
        });
        assert!(result.is_err());
        assert!(start.elapsed() < Duration::from_secs(5));
    }

    #[cfg(unix)]
    #[test]
    fn killing_the_process_group_also_kills_a_grandchild() {
        // The child spawns its own background grandchild; if only the direct child were
        // killed (not the whole process group), the grandchild would survive as an orphan.
        let marker = std::env::temp_dir().join(format!("buffdata-sv-test-{}", std::process::id()));
        let marker_path = marker.to_string_lossy().to_string();
        std::fs::remove_file(&marker).ok();
        let script = format!("(sleep 2; touch {marker_path}) & sleep 30",);
        let status = supervise(sh(&script), 60, || Ok(true)).unwrap();
        assert_eq!(status, "cancelled");
        std::thread::sleep(Duration::from_secs(3));
        assert!(
            !marker.exists(),
            "grandchild ran after the process group should have been killed"
        );
    }
}
