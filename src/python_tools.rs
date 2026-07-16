//! First-class Python tooling: pytest, run script, and non-LSP diagnostics fallback.

use anyhow::{anyhow, bail, Context, Result};
use std::{
    io::Read,
    path::{Path, PathBuf},
    process::{Command, Stdio},
    sync::atomic::{AtomicBool, Ordering},
    thread,
    time::{Duration, Instant},
};

const SHELL_COMMAND_TIMEOUT_SECS: u64 = 120;
const MAX_SHELL_OUTPUT_BYTES: usize = 96 * 1024;
const MAX_DIAGNOSTIC_LINES: usize = 80;

pub fn run_pytest(
    workspace: &Path,
    target: Option<&str>,
    extra_args: Option<&str>,
    interrupt: Option<&AtomicBool>,
) -> Result<String> {
    let mut cmd = Command::new("pytest");
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

    run_command(cmd, "pytest", interrupt)
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
    if !command_on_path("ruff") {
        bail!("ruff is not installed or not on PATH");
    }

    let mut cmd = Command::new("ruff");
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
    Ok(format!("Diagnostics source: ruff check\n{output}"))
}

/// Fallback diagnostics when LSP is unavailable: ruff, else compileall.
pub fn fallback_diagnostics(
    workspace: &Path,
    paths: &[PathBuf],
    interrupt: Option<&AtomicBool>,
) -> Result<String> {
    if command_on_path("ruff") {
        return ruff_check(workspace, paths, interrupt);
    }

    let python = resolve_python(workspace);
    let targets: Vec<PathBuf> = if paths.is_empty() {
        collect_py_files(workspace, 40)
    } else {
        paths.to_vec()
    };

    if targets.is_empty() {
        return Ok(
            "Diagnostics source: python -m compileall\nNo Python files found to check.".to_string(),
        );
    }

    let mut lines = vec!["Diagnostics source: python -m compileall".to_string()];
    for target in targets {
        let mut cmd = Command::new(&python);
        cmd.arg("-m")
            .arg("compileall")
            .arg("-q")
            .arg(&target)
            .current_dir(workspace)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        let result = run_command(cmd, "compileall", interrupt)?;
        if result.contains("Exit code:") || result.contains("Error") || result.contains("Sorry:") {
            lines.push(format!("{}:\n{result}", target.display()));
        }
    }

    if lines.len() == 1 {
        lines.push("No syntax errors reported.".to_string());
    }

    Ok(truncate_output(&lines.join("\n")))
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

fn resolve_python(workspace: &Path) -> String {
    for candidate in [
        workspace.join(".venv/bin/python"),
        workspace.join("venv/bin/python"),
        workspace.join(".uvenv/bin/python"),
    ] {
        if candidate.is_file() {
            return candidate.display().to_string();
        }
    }
    if command_on_path("python3") {
        "python3".to_string()
    } else {
        "python".to_string()
    }
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

fn collect_py_files(workspace: &Path, limit: usize) -> Vec<PathBuf> {
    let mut files = Vec::new();
    let Ok(entries) = std::fs::read_dir(workspace) else {
        return files;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_file()
            && path
                .extension()
                .and_then(|e| e.to_str())
                .is_some_and(|e| e.eq_ignore_ascii_case("py"))
        {
            files.push(path);
            if files.len() >= limit {
                break;
            }
        }
    }
    files
}

fn command_on_path(name: &str) -> bool {
    Command::new("sh")
        .arg("-c")
        .arg(format!("command -v {name} >/dev/null 2>&1"))
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

fn shell_split(args: &str) -> Vec<String> {
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

fn run_command(
    mut cmd: Command,
    label: &str,
    interrupt: Option<&AtomicBool>,
) -> Result<String> {
    let mut child = cmd
        .spawn()
        .with_context(|| format!("failed to spawn {label} (is it installed and on PATH?)"))?;

    let started = Instant::now();
    let status = loop {
        if interrupt.is_some_and(|flag| flag.load(Ordering::Relaxed)) {
            let _ = child.kill();
            let _ = child.wait();
            bail!("command interrupted by user (Ctrl+I)");
        }

        match child.try_wait() {
            Ok(Some(status)) => break status,
            Ok(None) if started.elapsed() > Duration::from_secs(SHELL_COMMAND_TIMEOUT_SECS) => {
                let _ = child.kill();
                let _ = child.wait();
                bail!("command timed out after {SHELL_COMMAND_TIMEOUT_SECS}s");
            }
            Ok(None) => thread::sleep(Duration::from_millis(50)),
            Err(error) => return Err(error).context(format!("failed while waiting for {label}")),
        }
    };

    let mut stdout = String::new();
    let mut stderr = String::new();
    if let Some(mut reader) = child.stdout.take() {
        reader
            .read_to_string(&mut stdout)
            .context("failed to read stdout")?;
    }
    if let Some(mut reader) = child.stderr.take() {
        reader
            .read_to_string(&mut stderr)
            .context("failed to read stderr")?;
    }

    let mut text = String::new();
    if !stdout.is_empty() {
        text.push_str("Output:\n");
        text.push_str(&stdout);
    }
    if !stderr.is_empty() {
        if !text.is_empty() && !text.ends_with('\n') {
            text.push('\n');
        }
        text.push_str("Stderr:\n");
        text.push_str(&stderr);
    }
    if text.trim().is_empty() {
        text = format!("(command exited with {status})");
    } else if !status.success() {
        if !text.ends_with('\n') {
            text.push('\n');
        }
        text.push_str(&format!("Exit code: {}", status.code().unwrap_or(-1)));
    }

    Ok(truncate_output(&text))
}

fn truncate_output(output: &str) -> String {
    if output.len() <= MAX_SHELL_OUTPUT_BYTES {
        return output.to_string();
    }
    let half = MAX_SHELL_OUTPUT_BYTES / 2;
    let head = &output[..half];
    let tail = &output[output.len().saturating_sub(half)..];
    format!(
        "{head}\n\n[Output truncated: {} bytes omitted]\n\n{tail}",
        output.len().saturating_sub(head.len() + tail.len())
    )
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
