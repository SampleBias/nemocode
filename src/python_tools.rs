//! First-class Python tooling: pytest, run script, and non-LSP diagnostics fallback.

use anyhow::{Result, anyhow, bail};
use std::{
    path::{Path, PathBuf},
    process::{Command, Stdio},
    sync::atomic::AtomicBool,
    time::Duration,
};

const SHELL_COMMAND_TIMEOUT_SECS: u64 = 120;
const MAX_DIAGNOSTIC_LINES: usize = 80;

/// Standard-library behavioral checks; exit 5 means no assertions were run.
pub fn run_unittest(
    workspace: &Path,
    start_directory: &str,
    pattern: &str,
    interrupt: Option<&AtomicBool>,
) -> Result<String> {
    let mut cmd = Command::new(resolve_python(workspace));
    cmd.current_dir(workspace).arg("-c").arg(
        "import sys,unittest\nsuite=unittest.defaultTestLoader.discover(sys.argv[1],pattern=sys.argv[2])\nresult=unittest.TextTestRunner(verbosity=2).run(suite)\nsys.exit(5 if result.testsRun == 0 or result.testsRun == len(result.skipped) else (0 if result.wasSuccessful() else 1))"
    ).arg(start_directory).arg(pattern);
    checker_output(
        run_command(cmd, "unittest", interrupt)?,
        "behavior",
        "unittest",
        &[],
    )
}

pub fn run_pytest(
    workspace: &Path,
    target: Option<&str>,
    extra_args: Option<&str>,
    interrupt: Option<&AtomicBool>,
) -> Result<String> {
    let mut cmd = Command::new(resolve_python(workspace));
    cmd.args(["-m", "pytest"]);
    cmd.current_dir(workspace)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    if let Some(target) = target.filter(|t| !t.trim().is_empty()) {
        cmd.arg(target);
    }
    if let Some(extra) = extra_args.filter(|t| !t.trim().is_empty()) {
        for arg in shell_split(extra) {
            cmd.arg(arg);
        }
    }

    checker_output(
        run_command(cmd, "pytest", interrupt)?,
        "behavior",
        "pytest",
        &[],
    )
}

pub fn run_python(
    workspace: &Path,
    script: Option<&str>,
    module: Option<&str>,
    code: Option<&str>,
    args: Option<&str>,
    interrupt: Option<&AtomicBool>,
) -> Result<String> {
    let python = resolve_python(workspace);
    let mut cmd = Command::new(&python);
    cmd.current_dir(workspace)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    match (script, module, code) {
        (Some(script), None, None) => {
            let path = resolve_under_workspace(script, workspace)?;
            cmd.arg(&path);
        }
        (None, Some(module), None) => {
            cmd.arg("-m").arg(module);
        }
        (None, None, Some(code)) => {
            cmd.arg("-c").arg(code);
        }
        _ => bail!("run_python requires exactly one of: script, module, or code"),
    }

    if let Some(extra) = args.filter(|t| !t.trim().is_empty()) {
        for arg in shell_split(extra) {
            cmd.arg(arg);
        }
    }

    run_command(cmd, &python, interrupt)
}

pub fn ruff_check(
    workspace: &Path,
    paths: &[PathBuf],
    interrupt: Option<&AtomicBool>,
) -> Result<String> {
    let ruff = resolve_ruff(workspace);
    if ruff.is_none() {
        bail!("ruff is not installed or not on PATH");
    }

    let mut cmd = Command::new(ruff.unwrap());
    cmd.arg("check")
        .arg("--output-format")
        .arg("concise")
        .current_dir(workspace)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    if paths.is_empty() {
        cmd.arg(".");
    } else {
        for path in paths {
            cmd.arg(path);
        }
    }

    let output = run_command(cmd, "ruff check", interrupt)?;
    checker_output(output, "lint", "ruff", paths)
}

/// Fallback diagnostics when LSP is unavailable: ruff, else compileall.
pub fn fallback_diagnostics(
    workspace: &Path,
    paths: &[PathBuf],
    interrupt: Option<&AtomicBool>,
) -> Result<String> {
    if resolve_ruff(workspace).is_some() {
        return ruff_check(workspace, paths, interrupt);
    }
    let targets = if paths.is_empty() {
        crate::retrieval::python_files(workspace, 500)?
    } else {
        let mut files = Vec::new();
        for path in paths {
            if path.is_dir() {
                files.extend(crate::retrieval::python_files(path, 500)?);
            } else {
                files.push(path.clone());
            }
        }
        files
    };
    if targets.len() > 500 {
        bail!("syntax check exceeds 500 files; select narrower paths");
    }
    if targets.is_empty() {
        bail!("No Python files selected; diagnostics coverage is empty");
    }
    let mut cmd = Command::new(resolve_python(workspace));
    cmd.current_dir(workspace).arg("-c").arg(
        "import ast,sys,tokenize\nerrors=0\nfor p in sys.argv[1:]:\n try:\n  with tokenize.open(p) as f: ast.parse(f.read(),filename=p)\n except (SyntaxError,OSError,UnicodeError) as e:\n  print(str(e)); errors+=1\nprint(f'Checked {len(sys.argv)-1} files (syntax only)')\nsys.exit(bool(errors))"
    );
    cmd.args(&targets);
    let result = crate::process::run(
        cmd,
        interrupt,
        Duration::from_secs(SHELL_COMMAND_TIMEOUT_SECS),
    )?;
    checker_output(
        result.json()?,
        "syntax",
        "Python AST (syntax only, not type checking)",
        &targets,
    )
}

fn checker_output(output: String, kind: &str, source: &str, paths: &[PathBuf]) -> Result<String> {
    let mut value: serde_json::Value = serde_json::from_str(&output)?;
    value["check"] = serde_json::json!({"kind":kind,"source":source,"paths":paths});
    Ok(value.to_string())
}

fn resolve_ruff(workspace: &Path) -> Option<String> {
    crate::platform::venv_tool(workspace, "ruff")
        .or_else(|| crate::platform::find_on_path("ruff"))
        .map(|path| path.display().to_string())
}

pub fn format_diagnostic_lines(lines: &[String]) -> String {
    if lines.is_empty() {
        return "No diagnostics.".to_string();
    }
    let mut out = lines.to_vec();
    if out.len() > MAX_DIAGNOSTIC_LINES {
        let omitted = out.len() - MAX_DIAGNOSTIC_LINES;
        out.truncate(MAX_DIAGNOSTIC_LINES);
        out.push(format!("… ({omitted} more diagnostics omitted)"));
    }
    out.join("\n")
}

pub fn resolve_python(workspace: &Path) -> String {
    if let Some(python) = crate::platform::venv_tool(workspace, "python") {
        return python.display().to_string();
    }
    // On Windows `python3` is usually the Microsoft Store placeholder, so the
    // names are tried in the opposite order there.
    let names: &[&str] = if cfg!(windows) {
        &["python", "py", "python3"]
    } else {
        &["python3", "python"]
    };
    names
        .iter()
        .find(|name| crate::platform::command_on_path(name))
        .unwrap_or(&names[0])
        .to_string()
}

fn resolve_under_workspace(path: &str, workspace: &Path) -> Result<PathBuf> {
    let candidate = PathBuf::from(path);
    let absolute = if candidate.is_absolute() {
        candidate
    } else {
        workspace.join(candidate)
    };
    if !absolute.exists() {
        bail!("path does not exist: {}", absolute.display());
    }
    Ok(absolute)
}

pub fn shell_split(args: &str) -> Vec<String> {
    // Lightweight split: whitespace, keeping simple quoted segments.
    let mut out = Vec::new();
    let mut current = String::new();
    let mut in_quotes: Option<char> = None;
    for ch in args.chars() {
        match (in_quotes, ch) {
            (Some(q), c) if c == q => in_quotes = None,
            (None, '"' | '\'') => in_quotes = Some(ch),
            (None, c) if c.is_whitespace() => {
                if !current.is_empty() {
                    out.push(std::mem::take(&mut current));
                }
            }
            (_, c) => current.push(c),
        }
    }
    if !current.is_empty() {
        out.push(current);
    }
    out
}

fn run_command(cmd: Command, _label: &str, interrupt: Option<&AtomicBool>) -> Result<String> {
    crate::process::run(
        cmd,
        interrupt,
        Duration::from_secs(SHELL_COMMAND_TIMEOUT_SECS),
    )?
    .json()
}

pub fn parse_optional_path_list(
    paths: Option<&[String]>,
    workspace: &Path,
) -> Result<Vec<PathBuf>> {
    let Some(paths) = paths else {
        return Ok(Vec::new());
    };
    paths
        .iter()
        .map(|p| {
            let candidate = PathBuf::from(p);
            if candidate.is_absolute() {
                Ok(candidate)
            } else {
                Ok(workspace.join(candidate))
            }
        })
        .collect()
}

pub fn required_paths_array(value: &serde_json::Value, key: &str) -> Result<Option<Vec<String>>> {
    match value.get(key) {
        None | Some(serde_json::Value::Null) => Ok(None),
        Some(serde_json::Value::Array(items)) => {
            let mut paths = Vec::with_capacity(items.len());
            for (index, item) in items.iter().enumerate() {
                let text = item
                    .as_str()
                    .ok_or_else(|| anyhow!("'{key}[{index}]' must be a string"))?;
                paths.push(text.to_string());
            }
            Ok(Some(paths))
        }
        Some(_) => bail!("field '{key}' must be an array of strings when provided"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn formats_empty_diagnostics() {
        assert_eq!(format_diagnostic_lines(&[]), "No diagnostics.");
    }

    #[test]
    fn truncates_long_diagnostic_lists() {
        let lines: Vec<String> = (0..100).map(|i| format!("err {i}")).collect();
        let formatted = format_diagnostic_lines(&lines);
        assert!(formatted.contains("omitted"));
        assert!(formatted.lines().count() <= MAX_DIAGNOSTIC_LINES + 1);
    }
}
