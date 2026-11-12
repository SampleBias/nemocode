//! Bounded subprocess capture. Pipes are drained while the process runs.
use anyhow::{Context, Result};
use serde::Serialize;
use std::{
    collections::VecDeque,
    io::Read,
    process::{Child, ChildStderr, ChildStdout, Command, ExitStatus, Stdio},
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
    #[cfg(windows)]
    {
        // `Child::kill` stops only the direct child, so take the tree while the
        // parent still exists. Its id cannot be reused while we hold the handle.
        if matches!(child.try_wait(), Ok(None)) {
            let _ = Command::new("taskkill")
                .args(["/T", "/F", "/PID", &child.id().to_string()])
                .stdin(Stdio::null())
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .status();
        }
    }
    let _ = child.kill();
}

struct Finished {
    status: ExitStatus,
    out: Capture,
    err: Capture,
    timed_out: bool,
    cancelled: bool,
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

#[cfg(unix)]
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

/// Unix: both pipes are non-blocking and read from the polling loop.
#[cfg(unix)]
fn supervise(
    child: &mut Child,
    mut stdout: ChildStdout,
    mut stderr: ChildStderr,
    interrupt: Option<&AtomicBool>,
    timeout: Duration,
    started: Instant,
) -> Result<Finished> {
    nonblocking(&stdout)?;
    nonblocking(&stderr)?;
    let mut out = Capture::default();
    let mut err = Capture::default();
    let mut timed_out = false;
    let mut cancelled = false;
    let status = loop {
        drain(&mut stdout, &mut out)?;
        drain(&mut stderr, &mut err)?;
        if let Some(status) = child.try_wait()? {
            break status;
        }
        cancelled = interrupt.is_some_and(|flag| flag.load(Ordering::Relaxed));
        timed_out = started.elapsed() >= timeout;
        if cancelled || timed_out {
            terminate(child);
            break child.wait()?;
        }
        std::thread::sleep(Duration::from_millis(10));
    };
    // Terminate descendants before draining: a background child may hold pipes open.
    terminate(child);
    drain(&mut stdout, &mut out)?;
    drain(&mut stderr, &mut err)?;
    Ok(Finished {
        status,
        out,
        err,
        timed_out,
        cancelled,
    })
}

/// A thread that drains one pipe, for platforms without non-blocking pipes.
#[cfg(not(unix))]
struct PipeReader {
    capture: std::sync::Arc<std::sync::Mutex<Capture>>,
    thread: std::thread::JoinHandle<()>,
}

#[cfg(not(unix))]
impl PipeReader {
    fn spawn(mut pipe: impl Read + Send + 'static) -> Self {
        let capture = std::sync::Arc::new(std::sync::Mutex::new(Capture::default()));
        let sink = capture.clone();
        let thread = std::thread::spawn(move || {
            let mut bytes = [0u8; 8192];
            loop {
                match pipe.read(&mut bytes) {
                    Ok(0) => break,
                    Ok(n) => {
                        if let Ok(mut capture) = sink.lock() {
                            capture.push(&bytes[..n]);
                        }
                    }
                    Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                    Err(_) => break,
                }
            }
        });
        Self { capture, thread }
    }

    /// Wait briefly for end-of-file. A surviving descendant can keep the pipe
    /// open indefinitely; whatever it writes later is not part of this result.
    fn finish(self) -> Capture {
        let deadline = Instant::now() + Duration::from_millis(500);
        while !self.thread.is_finished() && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(5));
        }
        let mut capture = self
            .capture
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        std::mem::take(&mut *capture)
    }
}

#[cfg(not(unix))]
fn supervise(
    child: &mut Child,
    stdout: ChildStdout,
    stderr: ChildStderr,
    interrupt: Option<&AtomicBool>,
    timeout: Duration,
    started: Instant,
) -> Result<Finished> {
    let out = PipeReader::spawn(stdout);
    let err = PipeReader::spawn(stderr);
    let mut timed_out = false;
    let mut cancelled = false;
    let status = loop {
        if let Some(status) = child.try_wait()? {
            break status;
        }
        cancelled = interrupt.is_some_and(|flag| flag.load(Ordering::Relaxed));
        timed_out = started.elapsed() >= timeout;
        if cancelled || timed_out {
            terminate(child);
            break child.wait()?;
        }
        std::thread::sleep(Duration::from_millis(10));
    };
    Ok(Finished {
        status,
        out: out.finish(),
        err: err.finish(),
        timed_out,
        cancelled,
    })
}

pub fn run(
    mut command: Command,
    interrupt: Option<&AtomicBool>,
    timeout: Duration,
) -> Result<Output> {
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
    let stdout = guard.0.stdout.take().context("stdout missing")?;
    let stderr = guard.0.stderr.take().context("stderr missing")?;
    let started = Instant::now();
    let done = supervise(&mut guard.0, stdout, stderr, interrupt, timeout, started)?;
    Ok(Output {
        exit_code: done.status.code(),
        stdout: done.out.text(),
        stderr: done.err.text(),
        truncated: done.out.total > STREAM_LIMIT || done.err.total > STREAM_LIMIT,
        timed_out: done.timed_out,
        cancelled: done.cancelled,
        duration_ms: started.elapsed().as_millis(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    fn bash(code: &str) -> Command {
        let mut cmd = crate::platform::bash();
        cmd.args(["-c", code]);
        cmd
    }
    #[test]
    fn drains_both_pipes_without_deadlock_and_bounds_output() {
        let out = run(
            bash("head -c 262144 /dev/zero; head -c 262144 /dev/zero >&2"),
            None,
            Duration::from_secs(10),
        )
        .unwrap();
        assert!(out.success(), "{out:?}");
        assert!(out.truncated);
        assert!(out.stdout.len() < STREAM_LIMIT + 100);
        assert!(out.stderr.len() < STREAM_LIMIT + 100);
    }
    #[cfg(unix)]
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
    /// The same guarantee with timings that allow for a slow shell start.
    #[test]
    fn timeout_stops_a_command_and_its_children() {
        let out = run(
            bash("echo before; sleep 60 & wait"),
            None,
            Duration::from_secs(3),
        )
        .unwrap();
        assert!(out.timed_out);
        assert!(!out.success());
        assert!(out.stdout.contains("before"), "{out:?}");
        assert!(out.duration_ms < 20_000, "{out:?}");
    }
    #[test]
    fn preserves_failure_and_handles_invalid_utf8() {
        let out = run(
            bash("printf '\\377'; exit 7"),
            None,
            Duration::from_secs(10),
        )
        .unwrap();
        assert_eq!(out.exit_code, Some(7));
        assert!(out.stdout.contains('\u{fffd}'));
    }
}
