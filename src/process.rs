//! Bounded subprocess capture. Pipes are drained while the process runs.
use anyhow::{Context, Result};
use serde::Serialize;
use std::{
    collections::VecDeque,
    io::Read,
    process::{Child, Command, Stdio},
    sync::atomic::{AtomicBool, Ordering},
    time::{Duration, Instant},
};

const STREAM_LIMIT: usize = 48 * 1024;

pub struct DeadlineGuard {
    stop: std::sync::Arc<AtomicBool>,
    worker: Option<std::thread::JoinHandle<()>>,
}
impl DeadlineGuard {
    pub fn new(cancel: std::sync::Arc<AtomicBool>, deadline: Instant) -> Self {
        let stop = std::sync::Arc::new(AtomicBool::new(false));
        let flag = stop.clone();
        let worker = std::thread::spawn(move || {
            while !flag.load(Ordering::Relaxed) {
                if Instant::now() >= deadline {
                    cancel.store(true, Ordering::Relaxed);
                    break;
                }
                std::thread::sleep(Duration::from_millis(20));
            }
        });
        Self {
            stop,
            worker: Some(worker),
        }
    }
}
impl Drop for DeadlineGuard {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

#[derive(Debug, Serialize)]
pub struct Output {
    pub exit_code: Option<i32>,
    pub stdout: String,
    pub stderr: String,
    pub truncated: bool,
    pub timed_out: bool,
    pub cancelled: bool,
    pub duration_ms: u128,
}

impl Output {
    pub fn success(&self) -> bool {
        self.exit_code == Some(0) && !self.timed_out && !self.cancelled
    }
    pub fn json(&self) -> Result<String> {
        Ok(serde_json::to_string(self)?)
    }
}

#[derive(Default)]
struct Capture {
    head: Vec<u8>,
    tail: VecDeque<u8>,
    total: usize,
}
impl Capture {
    fn push(&mut self, bytes: &[u8]) {
        self.total += bytes.len();
        let head_len = (STREAM_LIMIT / 2 - self.head.len()).min(bytes.len());
        self.head.extend_from_slice(&bytes[..head_len]);
        for &byte in &bytes[head_len..] {
            if self.tail.len() == STREAM_LIMIT / 2 {
                self.tail.pop_front();
            }
            self.tail.push_back(byte);
        }
    }
    fn text(&self) -> String {
        let mut bytes = self.head.clone();
        if self.total > STREAM_LIMIT {
            bytes.extend_from_slice(
                format!("\n[{} bytes omitted]\n", self.total - STREAM_LIMIT).as_bytes(),
            );
        }
        bytes.extend(self.tail.iter());
        String::from_utf8_lossy(&bytes).into_owned()
    }
}

struct ProcessGuard(Child);
impl Drop for ProcessGuard {
    fn drop(&mut self) {
        terminate(&mut self.0);
        let _ = self.0.wait();
    }
}

fn terminate(child: &mut Child) {
    #[cfg(unix)]
    unsafe {
        // Each command is spawned in a new process group; never signal our group.
        libc::kill(-(child.id() as i32), libc::SIGKILL);
    }
    let _ = child.kill();
}

#[cfg(unix)]
fn nonblocking(pipe: &impl std::os::fd::AsRawFd) -> Result<()> {
    let fd = pipe.as_raw_fd();
    let flags = unsafe { libc::fcntl(fd, libc::F_GETFL) };
    if flags < 0 || unsafe { libc::fcntl(fd, libc::F_SETFL, flags | libc::O_NONBLOCK) } < 0 {
        return Err(std::io::Error::last_os_error().into());
    }
    Ok(())
}

fn drain(reader: &mut impl Read, capture: &mut Capture) -> Result<()> {
    let mut bytes = [0u8; 8192];
    // Bound work per tick even when a producer writes continuously.
    for _ in 0..32 {
        match reader.read(&mut bytes) {
            Ok(0) => break,
            Ok(n) => capture.push(&bytes[..n]),
            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => break,
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(e.into()),
        }
    }
    Ok(())
}

pub fn run(
    mut command: Command,
    interrupt: Option<&AtomicBool>,
    timeout: Duration,
) -> Result<Output> {
    #[cfg(not(unix))]
    anyhow::bail!("subprocess execution currently requires Unix");
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        command.process_group(0);
    }
    command
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let mut guard = ProcessGuard(command.spawn().context("spawn command")?);
    let mut stdout = guard.0.stdout.take().context("stdout missing")?;
    let mut stderr = guard.0.stderr.take().context("stderr missing")?;
    #[cfg(unix)]
    {
        nonblocking(&stdout)?;
        nonblocking(&stderr)?;
    }
    let mut out = Capture::default();
    let mut err = Capture::default();
    let started = Instant::now();
    let mut timed_out = false;
    let mut cancelled = false;
    let status = loop {
        drain(&mut stdout, &mut out)?;
        drain(&mut stderr, &mut err)?;
        if let Some(status) = guard.0.try_wait()? {
            break status;
        }
        cancelled = interrupt.is_some_and(|flag| flag.load(Ordering::Relaxed));
        timed_out = started.elapsed() >= timeout;
        if cancelled || timed_out {
            terminate(&mut guard.0);
            break guard.0.wait()?;
        }
        std::thread::sleep(Duration::from_millis(10));
    };
    // Terminate descendants before draining: a background child may hold pipes open.
    terminate(&mut guard.0);
    drain(&mut stdout, &mut out)?;
    drain(&mut stderr, &mut err)?;
    Ok(Output {
        exit_code: status.code(),
        stdout: out.text(),
        stderr: err.text(),
        truncated: out.total > STREAM_LIMIT || err.total > STREAM_LIMIT,
        timed_out,
        cancelled,
        duration_ms: started.elapsed().as_millis(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    fn bash(code: &str) -> Command {
        let mut cmd = Command::new("bash");
        cmd.args(["-c", code]);
        cmd
    }
    #[test]
    fn drains_both_pipes_without_deadlock_and_bounds_output() {
        let out = run(
            bash("head -c 262144 /dev/zero; head -c 262144 /dev/zero >&2"),
            None,
            Duration::from_secs(3),
        )
        .unwrap();
        assert!(out.success(), "{out:?}");
        assert!(out.truncated);
        assert!(out.stdout.len() < STREAM_LIMIT + 100);
        assert!(out.stderr.len() < STREAM_LIMIT + 100);
    }
    #[test]
    fn timeout_preserves_output_and_cleans_background_children() {
        let out = run(
            bash("echo before; sleep 60 & wait"),
            None,
            Duration::from_millis(100),
        )
        .unwrap();
        assert!(out.timed_out);
        assert!(out.stdout.contains("before"));
        assert!(out.duration_ms < 2000);
    }
    #[test]
    fn preserves_failure_and_handles_invalid_utf8() {
        let out = run(bash("printf '\\377'; exit 7"), None, Duration::from_secs(1)).unwrap();
        assert_eq!(out.exit_code, Some(7));
        assert!(out.stdout.contains('\u{fffd}'));
    }
}
