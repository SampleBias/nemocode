//! The few places where Windows and Unix differ: canonical paths, the home
//! directory, locating bash, and finding tools on PATH.
use std::{
    env, io,
    path::{Path, PathBuf},
    process::Command,
};

/// `fs::canonicalize` without the `\\?\` prefix Windows adds. A verbatim path
/// does not compare equal to the ordinary spelling of the same file, and many
/// tools reject it.
pub fn canonical(path: &Path) -> io::Result<PathBuf> {
    let resolved = path.canonicalize()?;
    #[cfg(windows)]
    {
        let text = resolved.to_string_lossy();
        if let Some(share) = text.strip_prefix(r"\\?\UNC\") {
            return Ok(PathBuf::from(format!(r"\\{share}")));
        }
        if let Some(plain) = text.strip_prefix(r"\\?\") {
            return Ok(PathBuf::from(plain));
        }
    }
    Ok(resolved)
}

pub fn home_dir() -> Option<PathBuf> {
    ["HOME", "USERPROFILE"]
        .into_iter()
        .filter_map(env::var_os)
        .find(|value| !value.is_empty())
        .map(PathBuf::from)
}

/// A shell that understands `bash -c`. On Windows a bare `bash` can resolve to
/// the WSL launcher, so Git for Windows is preferred. `NEMO_BASH` overrides.
pub fn bash() -> Command {
    if let Some(path) = env::var_os("NEMO_BASH").filter(|path| !path.is_empty()) {
        return Command::new(path);
    }
    #[cfg(windows)]
    {
        for (variable, relative) in [
            ("ProgramFiles", r"Git\bin\bash.exe"),
            ("ProgramFiles(x86)", r"Git\bin\bash.exe"),
            ("LocalAppData", r"Programs\Git\bin\bash.exe"),
        ] {
            if let Some(base) = env::var_os(variable) {
                let candidate = PathBuf::from(base).join(relative);
                if candidate.is_file() {
                    return Command::new(candidate);
                }
            }
        }
    }
    Command::new("bash")
}

/// Locate an executable on PATH without spawning a shell.
pub fn find_on_path(name: &str) -> Option<PathBuf> {
    let extensions: &[&str] = if cfg!(windows) {
        &["", ".exe", ".cmd", ".bat"]
    } else {
        &[""]
    };
    env::split_paths(&env::var_os("PATH")?).find_map(|directory| {
        extensions
            .iter()
            .map(|extension| directory.join(format!("{name}{extension}")))
            .find(|candidate| is_executable(candidate))
    })
}

pub fn command_on_path(name: &str) -> bool {
    find_on_path(name).is_some()
}

fn is_executable(path: &Path) -> bool {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        path.metadata()
            .is_ok_and(|metadata| metadata.is_file() && metadata.permissions().mode() & 0o111 != 0)
    }
    #[cfg(not(unix))]
    {
        path.is_file()
    }
}

/// A tool inside a virtualenv: `bin/<tool>` on Unix, `Scripts/<tool>.exe` on Windows.
pub fn venv_tool(workspace: &Path, tool: &str) -> Option<PathBuf> {
    [".venv", "venv", ".uvenv"]
        .into_iter()
        .flat_map(|venv| {
            [
                workspace.join(venv).join("bin").join(tool),
                workspace
                    .join(venv)
                    .join("Scripts")
                    .join(format!("{tool}.exe")),
            ]
        })
        .find(|candidate| candidate.is_file())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_paths_compare_equal_to_ordinary_ones() {
        let directory = crate::storage::test_dir("canonical");
        let resolved = canonical(&directory).unwrap();
        assert!(!resolved.to_string_lossy().starts_with(r"\\?\"));
        assert!(resolved.join("new.py").starts_with(&resolved));
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn finds_tools_on_path_and_in_virtualenvs() {
        assert!(!command_on_path("nemocode-no-such-tool"));
        let directory = crate::storage::test_dir("venv-tool");
        assert_eq!(venv_tool(&directory, "python"), None);
        let scripts = directory.join(".venv").join("Scripts");
        std::fs::create_dir_all(&scripts).unwrap();
        std::fs::write(scripts.join("python.exe"), "").unwrap();
        assert_eq!(
            venv_tool(&directory, "python"),
            Some(scripts.join("python.exe"))
        );
        std::fs::remove_dir_all(directory).unwrap();
    }
}
