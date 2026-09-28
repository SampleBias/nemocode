//! Bounded lexical retrieval; no embedding model or resident indexing daemon.
use anyhow::{Context, Result, bail};
use serde_json::{Value, json};
use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
    sync::atomic::{AtomicBool, Ordering},
    time::{Duration, Instant},
};

const MAX_SCAN: usize = 2000;
const MAX_BYTES: u64 = 5_000_000;

pub fn files(root: &Path, limit: usize) -> Result<Vec<PathBuf>> {
    files_with_interrupt(root, limit, None)
}

fn files_with_interrupt(
    root: &Path,
    limit: usize,
    interrupt: Option<&AtomicBool>,
) -> Result<Vec<PathBuf>> {
    if !root.is_dir() {
        bail!("not a directory: {}", root.display());
    }
    let canonical_root = root.canonicalize()?;
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
    let out = crate::process::run(cmd, interrupt, Duration::from_secs(10));
    let names = match out {
        Ok(out) if out.exit_code == Some(0) || out.exit_code == Some(1) => {
            if out.truncated || out.timed_out {
                bail!("file index exceeds capture budget; select a narrower directory");
            }
            out.stdout
        }
        Ok(out) if out.cancelled => bail!("file indexing cancelled"),
        _ => {
            let mut cmd = Command::new("git");
            cmd.current_dir(root).args([
                "ls-files",
                "--cached",
                "--others",
                "--exclude-standard",
                "-z",
            ]);
            let out = crate::process::run(cmd, interrupt, Duration::from_secs(10))?;
            if out.cancelled {
                bail!("file indexing cancelled");
            }
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
        .filter(|p| {
            p.is_file()
                && !excluded(p.strip_prefix(root).unwrap_or(p))
                && p.canonicalize().is_ok_and(|resolved| {
                    resolved
                        .strip_prefix(&canonical_root)
                        .is_ok_and(|relative| !excluded(relative))
                })
        })
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
        let line = format!("{}: {}", i + 1, s);
        if bytes + line.len() > 10_000 {
            if lines.is_empty() {
                bail!(
                    "source line {} exceeds the 10 KB range allowance; no partial edit snippet returned",
                    i + 1
                );
            }
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

pub fn search(
    root: &Path,
    query: &str,
    limit: usize,
    interrupt: Option<&AtomicBool>,
) -> Result<String> {
    if query.is_empty() || query.len() > 500 {
        bail!("query must contain 1..500 bytes (literal text)");
    }
    let limit = limit.clamp(1, 100);
    let mut matches = Vec::new();
    let mut scanned = 0;
    let mut skipped = 0;
    let mut truncated = false;
    let started = Instant::now();
    let mut selected_bytes = 0u64;
    let single_file = root.is_file();
    let search_base = if single_file {
        root.parent().context("search file has no parent")?
    } else {
        root
    };
    let mut candidates = if single_file {
        if excluded(&root.canonicalize()?) {
            bail!("protected search file");
        }
        vec![root.to_path_buf()]
    } else {
        files_with_interrupt(root, MAX_SCAN, interrupt)?
    };
    candidates.sort_by_key(|path| {
        (
            !path
                .file_name()
                .is_some_and(|s| s.to_string_lossy().contains(query)),
            path.clone(),
        )
    });
    let mut selected = Vec::new();
    for path in candidates {
        let size = fs::metadata(&path)?.len();
        if size > MAX_BYTES {
            skipped += 1;
            continue;
        }
        if selected_bytes + size > 32 * 1024 * 1024 {
            truncated = true;
            break;
        }
        selected_bytes += size;
        selected.push(path);
    }
    // Explicit paths were canonicalized and protected-path filtered above.
    // Bound command size, output, runtime and aggregate input independently.
    let mut rg_available = true;
    'batches: for batch in selected.chunks(128) {
        if interrupt.is_some_and(|flag| flag.load(Ordering::Relaxed)) {
            bail!("search cancelled");
        }
        if started.elapsed() >= Duration::from_secs(10) {
            truncated = true;
            break;
        }
        if rg_available {
            let mut cmd = Command::new("rg");
            cmd.current_dir(search_base)
                .args(["--json", "--fixed-strings", "--no-messages", "--max-count"])
                .arg((limit + 1).to_string())
                .arg("--")
                .arg(query)
                .args(batch);
            match crate::process::run(
                cmd,
                interrupt,
                Duration::from_secs(10).saturating_sub(started.elapsed()),
            ) {
                Ok(out)
                    if !matches!(out.exit_code, Some(0 | 1))
                        && !out.cancelled
                        && !out.timed_out =>
                {
                    rg_available = false;
                }
                Ok(out) => {
                    if out.cancelled {
                        bail!("search cancelled");
                    }
                    if !matches!(out.exit_code, Some(0 | 1)) || out.timed_out || out.truncated {
                        truncated = true;
                    }
                    scanned += batch.len();
                    for line in out.stdout.lines() {
                        let Ok(record) = serde_json::from_str::<Value>(line) else {
                            continue;
                        };
                        if record["type"] != "match" {
                            continue;
                        }
                        let (Some(path), Some(number), Some(content)) = (
                            record["data"]["path"]["text"].as_str(),
                            record["data"]["line_number"].as_u64(),
                            record["data"]["lines"]["text"].as_str(),
                        ) else {
                            continue;
                        };
                        if matches.len() == limit {
                            truncated = true;
                            break 'batches;
                        }
                        matches.push(json!({"path":Path::new(path).strip_prefix(search_base).unwrap_or(Path::new(path)),"line":number,"text":crate::truncate_middle(content.trim_end_matches(['\r','\n']),500,"match"),"read_start_line":number.saturating_sub(2).max(1),"read_line_count":5}));
                    }
                    continue;
                }
                Err(error)
                    if error.chain().any(|e| {
                        e.downcast_ref::<std::io::Error>()
                            .is_some_and(|e| e.kind() == std::io::ErrorKind::NotFound)
                    }) =>
                {
                    rg_available = false
                }
                Err(error) => return Err(error),
            }
        }
        for path in batch {
            if interrupt.is_some_and(|flag| flag.load(Ordering::Relaxed)) {
                bail!("search cancelled");
            }
            if started.elapsed() >= Duration::from_secs(10) {
                truncated = true;
                break 'batches;
            }
            let Ok(text) = read_text(path) else {
                skipped += 1;
                continue;
            };
            scanned += 1;
            for (line, content) in text.lines().enumerate() {
                if content.contains(query) {
                    if matches.len() == limit {
                        truncated = true;
                        break 'batches;
                    }
                    matches.push(
                        json!({"path":path.strip_prefix(search_base).unwrap_or(path), "line":line+1,
                    "text":crate::truncate_middle(content, 500, "match")}),
                    );
                }
            }
        }
    }
    Ok(serde_json::to_string(
        &json!({"matches":matches,"truncated":truncated,"files_scanned":scanned,"files_skipped":skipped,
            "bytes_selected":selected_bytes,"byte_budget":32*1024*1024,"backend":if rg_available {"rg"} else {"bounded_rust"}}),
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
    fn search_is_bounded_cancellable_and_excludes_protected_files() {
        let dir = crate::storage::test_dir("search");
        fs::write(dir.join("code.py"), "needle\nneedle\n").unwrap();
        fs::write(dir.join(".env.private"), "needle_secret\n").unwrap();
        let result: Value =
            serde_json::from_str(&search(&dir, "needle", 1, None).unwrap()).unwrap();
        assert_eq!(result["matches"].as_array().unwrap().len(), 1);
        assert_eq!(result["truncated"], true);
        assert!(!result.to_string().contains("needle_secret"));
        let result: Value =
            serde_json::from_str(&search(&dir.join("code.py"), "needle", 10, None).unwrap())
                .unwrap();
        assert_eq!(result["matches"].as_array().unwrap().len(), 2);
        assert_eq!(result["matches"][0]["path"], "code.py");
        assert!(search(&dir.join(".env.private"), "needle", 10, None).is_err());
        assert!(search(&dir, "needle", 10, Some(&AtomicBool::new(true))).is_err());
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn source_ranges_never_middle_cut_a_line() {
        let text = "a".repeat(1500);
        let value: Value =
            serde_json::from_str(&range_text(Path::new("code.py"), &text, 1, 1).unwrap()).unwrap();
        assert_eq!(value["content"], format!("1: {text}"));
        assert!(range_text(Path::new("code.py"), &"a".repeat(11_000), 1, 1).is_err());
    }
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
