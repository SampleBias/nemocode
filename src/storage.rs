//! Atomic replacement and content-addressed revisions.
use anyhow::{Context, Result, bail};
use ring::digest::{SHA256, digest};
use std::{
    fs::{self, OpenOptions},
    io::Write,
    path::Path,
    sync::atomic::{AtomicU64, Ordering},
};
static SEQUENCE: AtomicU64 = AtomicU64::new(0);

pub fn hash(bytes: &[u8]) -> String {
    digest(&SHA256, bytes)
        .as_ref()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

pub fn atomic_write(path: &Path, bytes: &[u8], expected: Option<&str>) -> Result<()> {
    atomic_write_impl(path, bytes, expected, false)
}

pub fn atomic_create(path: &Path, bytes: &[u8]) -> Result<()> {
    atomic_write_impl(path, bytes, None, true)
}

fn atomic_write_impl(
    path: &Path,
    bytes: &[u8],
    expected: Option<&str>,
    create_only: bool,
) -> Result<()> {
    let parent = path.parent().context("file has no parent")?;
    fs::create_dir_all(parent)?;
    if fs::symlink_metadata(path).is_ok_and(|m| m.file_type().is_symlink()) {
        bail!("refusing to replace symlink: {}", path.display());
    }
    let existing = match fs::read(path) {
        Ok(content) => Some(content),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => None,
        Err(e) => return Err(e.into()),
    };
    if create_only && existing.is_some() {
        bail!("create_file does not overwrite existing files");
    }
    if let Some(expected) = expected {
        let actual = existing.as_ref().map(|b| hash(b)).unwrap_or_default();
        if actual != expected {
            bail!("stale edit: file revision changed; read the file again");
        }
    }
    let temp = parent.join(format!(
        ".nemo-write-{}-{}",
        std::process::id(),
        SEQUENCE.fetch_add(1, Ordering::Relaxed)
    ));
    let result = (|| -> Result<()> {
        let mut options = OpenOptions::new();
        options.write(true).create_new(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600);
        }
        let mut file = options.open(&temp)?;
        if let Ok(metadata) = fs::metadata(path) {
            file.set_permissions(metadata.permissions())?;
        }
        file.write_all(bytes)?;
        file.sync_all()?;
        // Detect edits made while preparing the replacement. This is optimistic
        // concurrency, not an OS-wide lock against arbitrary external writers.
        if fs::read(path).ok() != existing {
            bail!("file changed during write; retry after reading");
        }
        if create_only {
            // Linking a new name is atomic and cannot replace a concurrently
            // created destination. The temporary file is on the same filesystem.
            fs::hard_link(&temp, path)?;
            fs::remove_file(&temp)?;
        } else {
            fs::rename(&temp, path)?;
        }
        fs::File::open(parent)?.sync_all()?;
        Ok(())
    })();
    if result.is_err() {
        let _ = fs::remove_file(temp);
    }
    result
}

#[cfg(test)]
pub fn test_dir(label: &str) -> std::path::PathBuf {
    let path = std::env::temp_dir().join(format!(
        "nemo-{label}-{}-{}",
        std::process::id(),
        SEQUENCE.fetch_add(1, Ordering::Relaxed)
    ));
    fs::create_dir_all(&path).unwrap();
    path
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn concurrent_creates_never_overwrite_the_winner() {
        let dir = test_dir("create-race");
        let path = dir.join("new.py");
        let barrier = std::sync::Arc::new(std::sync::Barrier::new(2));
        let handles: Vec<_> = [b"first".as_slice(), b"second".as_slice()]
            .into_iter()
            .map(|content| {
                let path = path.clone();
                let barrier = barrier.clone();
                std::thread::spawn(move || {
                    barrier.wait();
                    atomic_create(&path, content).is_ok()
                })
            })
            .collect();
        let results: Vec<_> = handles
            .into_iter()
            .map(|handle| handle.join().unwrap())
            .collect();
        assert_eq!(results.into_iter().filter(|success| *success).count(), 1);
        let winner = fs::read(&path).unwrap();
        assert!(winner == b"first" || winner == b"second");
        assert!(atomic_create(&path, b"overwrite").is_err());
        assert_eq!(fs::read(&path).unwrap(), winner);
        fs::remove_dir_all(dir).unwrap();
    }
    #[test]
    fn rejects_stale_edit_and_preserves_original() {
        let dir = test_dir("atomic");
        let path = dir.join("a");
        atomic_write(&path, b"old", None).unwrap();
        assert!(atomic_write(&path, b"new", Some(&hash(b"stale"))).is_err());
        assert_eq!(fs::read(&path).unwrap(), b"old");
        atomic_write(&path, b"new", Some(&hash(b"old"))).unwrap();
        assert_eq!(fs::read(&path).unwrap(), b"new");
        fs::remove_dir_all(dir).unwrap();
    }
}
