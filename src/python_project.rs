//! Detect Python project markers and format a compact session context block.

use std::path::{Path, PathBuf};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PythonProjectInfo {
    pub root: PathBuf,
    pub markers: Vec<&'static str>,
    pub package_manager: Option<&'static str>,
    pub has_venv: bool,
}

pub fn detect_python_project(workspace: &Path) -> Option<PythonProjectInfo> {
    let mut markers = Vec::new();

    if workspace.join("pyproject.toml").is_file() {
        markers.push("pyproject.toml");
    }
    if workspace.join("requirements.txt").is_file() {
        markers.push("requirements.txt");
    }
    if workspace.join("setup.cfg").is_file() {
        markers.push("setup.cfg");
    }
    if workspace.join("setup.py").is_file() {
        markers.push("setup.py");
    }
    if workspace.join(".python-version").is_file() {
        markers.push(".python-version");
    }

    if markers.is_empty() {
        // Also treat a directory with .py files at the top level as a loose Python project.
        if !has_top_level_py_file(workspace) {
            return None;
        }
        markers.push("*.py");
    }

    let package_manager = if workspace.join("uv.lock").is_file() {
        Some("uv")
    } else if workspace.join("poetry.lock").is_file() {
        Some("poetry")
    } else if workspace.join("Pipfile").is_file() || workspace.join("Pipfile.lock").is_file() {
        Some("pipenv")
    } else if workspace.join("requirements.txt").is_file() {
        Some("pip")
    } else if workspace.join("pyproject.toml").is_file() {
        Some("pyproject")
    } else {
        None
    };

    let has_venv = workspace.join(".venv").is_dir()
        || workspace.join("venv").is_dir()
        || workspace.join(".uvenv").is_dir();

    Some(PythonProjectInfo {
        root: workspace.to_path_buf(),
        markers,
        package_manager,
        has_venv,
    })
}

pub fn python_project_block(workspace: &Path) -> Option<String> {
    let info = detect_python_project(workspace)?;
    let mut lines = vec![
        "PYTHON PROJECT:".to_string(),
        format!("- Root: {}", info.root.display()),
        format!("- Markers: {}", info.markers.join(", ")),
    ];
    if let Some(manager) = info.package_manager {
        lines.push(format!("- Package manager hint: {manager}"));
    }
    lines.push(format!(
        "- Virtualenv: {}",
        if info.has_venv {
            "present (.venv/venv)"
        } else {
            "not detected"
        }
    ));
    lines.push(
        "- Prefer python_diagnostics after edits; use run_pytest / run_python for validation."
            .to_string(),
    );
    Some(lines.join("\n") + "\n")
}

fn has_top_level_py_file(workspace: &Path) -> bool {
    let Ok(entries) = std::fs::read_dir(workspace) else {
        return false;
    };
    entries.flatten().any(|entry| {
        let path = entry.path();
        path.is_file()
            && path
                .extension()
                .and_then(|ext| ext.to_str())
                .is_some_and(|ext| ext.eq_ignore_ascii_case("py"))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_dir(label: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("nemocode-{label}-{nanos}"));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn detects_pyproject_and_uv() {
        let dir = temp_dir("pyproj");
        fs::write(dir.join("pyproject.toml"), "[project]\nname='demo'\n").unwrap();
        fs::write(dir.join("uv.lock"), "version = 1\n").unwrap();
        let info = detect_python_project(&dir).unwrap();
        assert!(info.markers.contains(&"pyproject.toml"));
        assert_eq!(info.package_manager, Some("uv"));
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn detects_loose_py_files() {
        let dir = temp_dir("loosepy");
        fs::write(dir.join("app.py"), "print(1)\n").unwrap();
        let info = detect_python_project(&dir).unwrap();
        assert!(info.markers.contains(&"*.py"));
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn returns_none_for_non_python_dir() {
        let dir = temp_dir("empty");
        fs::write(dir.join("README.md"), "hi\n").unwrap();
        assert!(detect_python_project(&dir).is_none());
        let _ = fs::remove_dir_all(dir);
    }
}
