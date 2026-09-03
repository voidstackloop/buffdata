//! Mirrors `buffdata/security/policy.py::contained_path()`: no symlink anywhere in the
//! path's own chain, and the resolved path must stay inside `root`. Defense in depth --
//! the server already validated `run_id` before handing it to us, but this worker treats
//! server input the same way the rest of the codebase treats any other untrusted path.

use std::path::{Path, PathBuf};

#[derive(Debug)]
pub struct PathGuardError(pub String);

impl std::fmt::Display for PathGuardError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for PathGuardError {}

/// Every run ID handed to us by the server must match this shape before it ever touches a
/// filesystem path -- same pattern `buffdata/runs/service.py::run_directory()` enforces.
pub fn valid_run_id(run_id: &str) -> bool {
    run_id.len() == 32
        && run_id
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

fn is_symlink(path: &Path) -> bool {
    std::fs::symlink_metadata(path)
        .map(|meta| meta.file_type().is_symlink())
        .unwrap_or(false)
}

pub fn contained_path(path: &Path, root: &Path) -> Result<PathBuf, PathGuardError> {
    let mut cursor = path.to_path_buf();
    loop {
        if is_symlink(&cursor) {
            return Err(PathGuardError(
                "Symbolic links are not accepted at an artifact boundary".to_string(),
            ));
        }
        match cursor.parent() {
            Some(parent) if parent != cursor => cursor = parent.to_path_buf(),
            _ => break,
        }
    }
    let resolved = path
        .canonicalize()
        .map_err(|e| PathGuardError(format!("cannot resolve path {}: {e}", path.display())))?;
    let root_resolved = root
        .canonicalize()
        .map_err(|e| PathGuardError(format!("cannot resolve root {}: {e}", root.display())))?;
    if !resolved.starts_with(&root_resolved) {
        return Err(PathGuardError(
            "Path is outside the authorized project directory".to_string(),
        ));
    }
    Ok(resolved)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_run_id_shaped_like_a_uuid4_hex() {
        assert!(valid_run_id("0123456789abcdef0123456789abcdef"));
    }

    #[test]
    fn rejects_wrong_length() {
        assert!(!valid_run_id("0123456789abcdef"));
        assert!(!valid_run_id(""));
    }

    #[test]
    fn rejects_uppercase_and_non_hex() {
        assert!(!valid_run_id("0123456789ABCDEF0123456789abcdef"));
        assert!(!valid_run_id("../../../etc/passwdxxxxxxxxxxxxxx"));
        assert!(!valid_run_id("g123456789abcdef0123456789abcdef"));
    }

    #[test]
    fn accepts_a_path_that_stays_inside_root() {
        let tmp = std::env::temp_dir().join(format!("buffdata-pg-test-{}", std::process::id()));
        let root = tmp.join("project");
        let run_dir = root.join("runs").join("0123456789abcdef0123456789abcdef");
        std::fs::create_dir_all(&run_dir).unwrap();
        let resolved = contained_path(&run_dir, &root).unwrap();
        assert!(resolved.starts_with(root.canonicalize().unwrap()));
        std::fs::remove_dir_all(&tmp).ok();
    }

    #[test]
    fn rejects_a_path_that_escapes_root_via_dotdot() {
        let tmp = std::env::temp_dir().join(format!("buffdata-pg-test-esc-{}", std::process::id()));
        let root = tmp.join("project");
        let outside = tmp.join("outside");
        std::fs::create_dir_all(&root).unwrap();
        std::fs::create_dir_all(&outside).unwrap();
        let escaping = root.join("..").join("outside");
        assert!(contained_path(&escaping, &root).is_err());
        std::fs::remove_dir_all(&tmp).ok();
    }

    #[cfg(unix)]
    #[test]
    fn rejects_a_symlink_inside_the_authorized_root() {
        let tmp = std::env::temp_dir().join(format!("buffdata-pg-test-sym-{}", std::process::id()));
        let root = tmp.join("project");
        let outside = tmp.join("outside");
        std::fs::create_dir_all(&root).unwrap();
        std::fs::create_dir_all(&outside).unwrap();
        let link = root.join("escape");
        std::os::unix::fs::symlink(&outside, &link).unwrap();
        assert!(contained_path(&link, &root).is_err());
        std::fs::remove_dir_all(&tmp).ok();
    }
}
