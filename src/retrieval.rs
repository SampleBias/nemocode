//! Bounded lexical retrieval; no embedding model or resident indexing daemon.
use anyhow::{Context, Result, bail};
use serde_json::{Value, json};
use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
    time::Duration,
};

const MAX_SCAN: usize = 2000;
const MAX_BYTES: u64 = 5_000_000;

pub fn files(root: &Path, limit: usize) -> Result<Vec<PathBuf>> {
    if !root.is_dir() {
        bail!("not a directory: {}", root.display());
    }
    let mut cmd = Command::new("rg");
    cmd.current_dir(root).args([
        "--files",
        "--hidden",
        "-0",
        "-g",
        "!.git",
        "-g",
        "!.env*",
        "-g",
        "!.nemocode",
        "-g",
        "!target",
        "-g",
        "!.vendor",
        "-g",
        "!models",
        "-g",
        "!.venv",
        "-g",
        "!venv",
        "-g",
        "!node_modules",
    ]);
    let out = crate::process::run(cmd, None, Duration::from_secs(10));
    let names = match out {
        Ok(out) if out.exit_code == Some(0) || out.exit_code == Some(1) => {
            if out.truncated || out.timed_out {
                bail!("file index exceeds capture budget; select a narrower directory");
            }
            out.stdout
        }
        _ => {
            let mut cmd = Command::new("git");
            cmd.current_dir(root).args([
                "ls-files",
                "--cached",
                "--others",
                "--exclude-standard",
                "-z",
            ]);
            let out = crate::process::run(cmd, None, Duration::from_secs(10))?;
            if !out.success() || out.truncated {
                bail!(
                    "install ripgrep for indexing non-Git workspaces; git file enumeration failed"
                );
            }
            out.stdout
        }
    };
    let mut paths: Vec<_> = names
        .split('\0')
        .filter(|s| !s.is_empty())
        .map(|s| root.join(s))
        .filter(|p| p.is_file() && !excluded(p))
        .collect();
    paths.sort();
    paths.dedup();
    if paths.len() > limit {
        bail!(
            "{} files exceed index limit {limit}; select a narrower directory",
            paths.len()
        );
    }
    Ok(paths)
}

fn excluded(path: &Path) -> bool {
    path.components().any(|c| {
        let name = c.as_os_str().to_string_lossy();
        name.starts_with(".env")
            || matches!(
                name.as_ref(),
                ".git"
                    | ".nemocode"
                    | ".venv"
                    | ".uvenv"
                    | "venv"
                    | "target"
                    | ".vendor"
                    | "models"
                    | "node_modules"
                    | "__pycache__"
            )
    })
}

pub fn python_files(root: &Path, limit: usize) -> Result<Vec<PathBuf>> {
    let paths: Vec<_> = files(root, MAX_SCAN)?
        .into_iter()
        .filter(|p| p.extension().is_some_and(|e| e == "py"))
        .collect();
    if paths.len() > limit {
        bail!(
            "{} Python files exceed diagnostic limit {limit}; select narrower paths",
            paths.len()
        );
    }
    Ok(paths)
}

pub fn read_range(path: &Path, start: usize, count: usize) -> Result<String> {
    range_text(path, &read_text(path)?, start, count)
}

pub fn range_text(path: &Path, text: &str, start: usize, count: usize) -> Result<String> {
    if start == 0 || count == 0 || count > 400 {
        bail!("start_line must be >=1 and line_count must be 1..400");
    }
    let total = text.lines().count();
    let mut lines = Vec::new();
    let mut bytes = 0;
    for (i, s) in text.lines().enumerate().skip(start - 1).take(count) {
        let line = format!("{}: {}", i + 1, crate::truncate_middle(s, 1000, "line"));
        if bytes + line.len() > 10_000 {
            break;
        }
        bytes += line.len();
        lines.push(line);
    }
    let next_line = start + lines.len();
    Ok(serde_json::to_string(
        &json!({"path":path,"sha256":crate::storage::hash(text.as_bytes()),
        "start_line":start,"total_lines":total,"content":lines.join("\n"),
        "next_line":next_line,"more":next_line.saturating_sub(1) < total}),
    )?)
}

pub fn read_text(path: &Path) -> Result<String> {
    let metadata = fs::metadata(path)?;
    if !metadata.is_file() || metadata.len() > MAX_BYTES {
        bail!(
            "not a text file within the 5 MB read budget: {}",
            path.display()
        );
    }
    let text = fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    if text.contains('\0') {
        bail!("binary file: {}", path.display());
    }
    Ok(text)
}

pub fn search(root: &Path, query: &str, limit: usize) -> Result<String> {
    if query.is_empty() || query.len() > 500 {
        bail!("query must contain 1..500 bytes (literal text)");
    }
    let limit = limit.clamp(1, 100);
    let mut matches = Vec::new();
    let mut scanned = 0;
    let mut skipped = 0;
    let mut truncated = false;
    'files: for path in files(root, MAX_SCAN)? {
        let Ok(text) = read_text(&path) else {
            skipped += 1;
            continue;
        };
        scanned += 1;
        for (line, content) in text.lines().enumerate() {
            if content.contains(query) {
                if matches.len() == limit {
                    truncated = true;
                    break 'files;
                }
                matches.push(
                    json!({"path":path.strip_prefix(root).unwrap_or(&path), "line":line+1,
                    "text":crate::truncate_middle(content, 500, "match")}),
                );
            }
        }
    }
    Ok(serde_json::to_string(
        &json!({"matches":matches,"truncated":truncated,"files_scanned":scanned,"files_skipped":skipped}),
    )?)
}

pub fn outline(path: &Path) -> Result<String> {
    let text = read_text(path)?;
    let mut symbols = Vec::<Value>::new();
    for (i, line) in text.lines().enumerate() {
        let s = line.trim_start();
        if [
            "def ",
            "async def ",
            "class ",
            "fn ",
            "pub fn ",
            "pub async fn ",
            "async fn ",
            "struct ",
            "pub struct ",
            "enum ",
            "impl ",
            "function ",
            "export function ",
        ]
        .iter()
        .any(|prefix| s.starts_with(prefix))
        {
            symbols.push(
                json!({"line":i+1,"signature":crate::truncate_middle(line, 240, "signature")}),
            );
            if symbols.len() == 200 {
                break;
            }
        }
    }
    Ok(serde_json::to_string(
        &json!({"path":path,"kind":"lexical outline (not a full parser)","symbols":symbols,"limit":200}),
    )?)
}

pub fn index(root: &Path) -> Result<String> {
    let all = files(root, MAX_SCAN)?;
    let paths: Vec<_> = all
        .iter()
        .take(300)
        .map(|p| p.strip_prefix(root).unwrap_or(p))
        .collect();
    Ok(serde_json::to_string(
        &json!({"root":root,"files":paths,"total":all.len(),"truncated":all.len()>300,
        "hint":"Use search_code and read_file line ranges to retrieve relevant content."}),
    )?)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn retrieves_middle_of_large_file_and_reports_revision() {
        let dir = crate::storage::test_dir("range");
        let path = dir.join("code.py");
        fs::write(
            &path,
            (1..1000).map(|i| format!("line {i}\n")).collect::<String>(),
        )
        .unwrap();
        let result: Value = serde_json::from_str(&read_range(&path, 500, 2).unwrap()).unwrap();
        assert_eq!(result["content"], "500: line 500\n501: line 501");
        assert_eq!(result["sha256"].as_str().unwrap().len(), 64);
        assert!(result["more"].as_bool().unwrap());
        fs::remove_dir_all(dir).unwrap();
    }
}
