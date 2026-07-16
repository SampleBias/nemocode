//! Diagnostics-only Python LSP client (basedpyright / pyright).
//!
//! No network access is required once the language server binary is installed.
//! Completion, hover, and rename are intentionally unsupported in v1.

use anyhow::{anyhow, bail, Context, Result};
use serde_json::{json, Value};
use std::{
    collections::HashMap,
    env,
    io::{BufRead, BufReader, Read, Write},
    path::{Path, PathBuf},
    process::{Child, ChildStdin, Command, Stdio},
    sync::{
        mpsc::{self, Receiver, Sender},
        Arc, Mutex,
    },
    thread,
    time::{Duration, Instant},
};

use crate::python_tools::format_diagnostic_lines;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LspMode {
    Auto,
    Off,
    Command,
}

pub struct PythonLspClient {
    mode: LspMode,
    command_override: Option<String>,
    root: Option<PathBuf>,
    child: Option<Child>,
    stdin: Option<ChildStdin>,
    next_id: u64,
    diagnostics: Arc<Mutex<HashMap<String, Vec<String>>>>,
    reader_tx_stop: Option<Sender<()>>,
    response_rx: Option<Receiver<(u64, Result<Value, String>)>>,
    open_files: HashMap<PathBuf, i32>,
}

impl PythonLspClient {
    pub fn from_env() -> Self {
        let raw = env::var("NEMO_PYTHON_LSP").unwrap_or_else(|_| "auto".to_string());
        let (mode, command_override) = match raw.trim() {
            "off" | "0" | "false" => (LspMode::Off, None),
            "auto" | "" => (LspMode::Auto, None),
            other => (LspMode::Command, Some(other.to_string())),
        };

        let command_override = env::var("NEMO_PYTHON_LSP_COMMAND")
            .ok()
            .filter(|s| !s.trim().is_empty())
            .or(command_override);

        Self {
            mode: if command_override.is_some() && mode != LspMode::Off {
                LspMode::Command
            } else {
                mode
            },
            command_override,
            root: None,
            child: None,
            stdin: None,
            next_id: 1,
            diagnostics: Arc::new(Mutex::new(HashMap::new())),
            reader_tx_stop: None,
            response_rx: None,
            open_files: HashMap::new(),
        }
    }

    pub fn enabled(&self) -> bool {
        self.mode != LspMode::Off
    }

    pub fn ensure_for_workspace(&mut self, workspace: &Path) -> Result<()> {
        if !self.enabled() {
            bail!("Python LSP disabled (NEMO_PYTHON_LSP=off)");
        }

        if self.root.as_ref().is_some_and(|root| root == workspace) && self.child.is_some() {
            return Ok(());
        }

        self.shutdown_silent();
        self.start(workspace)
    }

    pub fn on_workspace_changed(&mut self, workspace: &Path) {
        if self.root.as_ref().is_some_and(|root| root != workspace) {
            self.shutdown_silent();
        }
    }

    pub fn notify_file(&mut self, path: &Path, text: &str) {
        if self.child.is_none() {
            return;
        }
        if path
            .extension()
            .and_then(|e| e.to_str())
            .is_none_or(|e| !e.eq_ignore_ascii_case("py"))
        {
            return;
        }

        let uri = path_to_uri(path);
        let version = {
            let entry = self.open_files.entry(path.to_path_buf()).or_insert(0);
            *entry += 1;
            *entry
        };

        if version == 1 {
            let _ = self.notify(
                "textDocument/didOpen",
                json!({
                    "textDocument": {
                        "uri": uri,
                        "languageId": "python",
                        "version": version,
                        "text": text,
                    }
                }),
            );
        } else {
            let _ = self.notify(
                "textDocument/didChange",
                json!({
                    "textDocument": {
                        "uri": uri,
                        "version": version,
                    },
                    "contentChanges": [{ "text": text }],
                }),
            );
        }
    }

    /// Collect diagnostics, optionally filtering to paths. Waits briefly for publishDiagnostics.
    pub fn collect_diagnostics(
        &mut self,
        workspace: &Path,
        paths: &[PathBuf],
    ) -> Result<String> {
        self.ensure_for_workspace(workspace)?;

        for path in paths {
            if path.is_file() {
                if let Ok(text) = std::fs::read_to_string(path) {
                    self.notify_file(path, &text);
                }
            }
        }

        // Give the server a moment to publish diagnostics after didOpen/didChange.
        thread::sleep(Duration::from_millis(400));
        self.drain_notifications(Duration::from_millis(800));

        let map = self
            .diagnostics
            .lock()
            .map_err(|_| anyhow!("diagnostics lock poisoned"))?;

        let mut lines = Vec::new();
        if paths.is_empty() {
            for (uri, diags) in map.iter() {
                for diag in diags {
                    lines.push(format!("{}: {diag}", uri_to_display(uri)));
                }
            }
        } else {
            for path in paths {
                let uri = path_to_uri(path);
                if let Some(diags) = map.get(&uri) {
                    for diag in diags {
                        lines.push(format!("{}: {diag}", path.display()));
                    }
                }
            }
        }

        lines.sort();
        Ok(format!(
            "Diagnostics source: python LSP\n{}",
            format_diagnostic_lines(&lines)
        ))
    }

    fn start(&mut self, workspace: &Path) -> Result<()> {
        let command = self.resolve_command()?;
        let mut parts = shellish_split(&command);
        if parts.is_empty() {
            bail!("empty Python LSP command");
        }
        let program = parts.remove(0);

        let mut child = Command::new(&program)
            .args(&parts)
            .current_dir(workspace)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .with_context(|| {
                format!(
                    "failed to spawn Python LSP '{command}' (install basedpyright or pyright locally)"
                )
            })?;

        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| anyhow!("LSP stdout missing"))?;
        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| anyhow!("LSP stdin missing"))?;

        let diagnostics = self.diagnostics.clone();
        let (response_tx, response_rx) = mpsc::channel();
        let (stop_tx, stop_rx) = mpsc::channel();

        thread::spawn(move || {
            reader_loop(stdout, diagnostics, response_tx, stop_rx);
        });

        self.child = Some(child);
        self.stdin = Some(stdin);
        self.response_rx = Some(response_rx);
        self.reader_tx_stop = Some(stop_tx);
        self.root = Some(workspace.to_path_buf());
        self.open_files.clear();
        if let Ok(mut map) = self.diagnostics.lock() {
            map.clear();
        }

        let root_uri = path_to_uri(workspace);
        let init_result = self.request(
            "initialize",
            json!({
                "processId": std::process::id(),
                "rootUri": root_uri,
                "rootPath": workspace,
                "capabilities": {
                    "textDocument": {
                        "publishDiagnostics": {
                            "relatedInformation": true,
                        }
                    },
                    "workspace": {
                        "workspaceFolders": true,
                    }
                },
                "workspaceFolders": [{
                    "uri": root_uri,
                    "name": workspace.file_name()
                        .and_then(|n| n.to_str())
                        .unwrap_or("workspace"),
                }],
            }),
        );

        match init_result {
            Ok(_) => {
                let _ = self.notify("initialized", json!({}));
                Ok(())
            }
            Err(error) => {
                self.shutdown_silent();
                Err(error)
            }
        }
    }

    fn resolve_command(&self) -> Result<String> {
        if let Some(cmd) = &self.command_override {
            return Ok(cmd.clone());
        }

        for candidate in [
            "basedpyright-langserver --stdio",
            "pyright-langserver --stdio",
        ] {
            let bin = candidate.split_whitespace().next().unwrap_or("");
            if command_on_path(bin) {
                return Ok(candidate.to_string());
            }
        }

        bail!(
            "no Python language server found on PATH (tried basedpyright-langserver, pyright-langserver); \
             set NEMO_PYTHON_LSP_COMMAND or install basedpyright"
        )
    }

    fn request(&mut self, method: &str, params: Value) -> Result<Value> {
        let id = self.next_id;
        self.next_id += 1;
        let message = json!({
            "jsonrpc": "2.0",
            "id": id,
            "method": method,
            "params": params,
        });
        self.write_message(&message)?;

        let rx = self
            .response_rx
            .as_ref()
            .ok_or_else(|| anyhow!("LSP response channel missing"))?;

        let deadline = Instant::now() + Duration::from_secs(20);
        while Instant::now() < deadline {
            match rx.recv_timeout(Duration::from_millis(100)) {
                Ok((resp_id, Ok(value))) if resp_id == id => return Ok(value),
                Ok((resp_id, Err(err))) if resp_id == id => bail!("LSP error for {method}: {err}"),
                Ok(_) => continue,
                Err(mpsc::RecvTimeoutError::Timeout) => continue,
                Err(mpsc::RecvTimeoutError::Disconnected) => {
                    bail!("LSP reader disconnected during {method}")
                }
            }
        }
        bail!("timed out waiting for LSP response to {method}")
    }

    fn notify(&mut self, method: &str, params: Value) -> Result<()> {
        let message = json!({
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
        });
        self.write_message(&message)
    }

    fn write_message(&mut self, message: &Value) -> Result<()> {
        let body = serde_json::to_vec(message).context("serialize LSP message")?;
        let stdin = self
            .stdin
            .as_mut()
            .ok_or_else(|| anyhow!("LSP stdin closed"))?;
        write!(stdin, "Content-Length: {}\r\n\r\n", body.len())
            .context("write LSP header")?;
        stdin.write_all(&body).context("write LSP body")?;
        stdin.flush().context("flush LSP stdin")?;
        Ok(())
    }

    fn drain_notifications(&self, wait: Duration) {
        // Reader thread owns parsing; just wait for publishDiagnostics to arrive.
        thread::sleep(wait);
    }

    fn shutdown_silent(&mut self) {
        if let Some(stop) = self.reader_tx_stop.take() {
            let _ = stop.send(());
        }
        if let Some(mut stdin) = self.stdin.take() {
            let shutdown = json!({
                "jsonrpc": "2.0",
                "id": self.next_id,
                "method": "shutdown",
                "params": null,
            });
            self.next_id += 1;
            if let Ok(body) = serde_json::to_vec(&shutdown) {
                let _ = write!(stdin, "Content-Length: {}\r\n\r\n", body.len());
                let _ = stdin.write_all(&body);
                let _ = stdin.flush();
            }
            let exit = json!({
                "jsonrpc": "2.0",
                "method": "exit",
            });
            if let Ok(body) = serde_json::to_vec(&exit) {
                let _ = write!(stdin, "Content-Length: {}\r\n\r\n", body.len());
                let _ = stdin.write_all(&body);
                let _ = stdin.flush();
            }
        }
        if let Some(mut child) = self.child.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
        self.response_rx = None;
        self.root = None;
        self.open_files.clear();
        if let Ok(mut map) = self.diagnostics.lock() {
            map.clear();
        }
    }
}

impl Drop for PythonLspClient {
    fn drop(&mut self) {
        self.shutdown_silent();
    }
}

fn reader_loop<R: Read + Send + 'static>(
    stdout: R,
    diagnostics: Arc<Mutex<HashMap<String, Vec<String>>>>,
    response_tx: Sender<(u64, Result<Value, String>)>,
    stop_rx: Receiver<()>,
) {
    let mut reader = BufReader::new(stdout);
    loop {
        if stop_rx.try_recv().is_ok() {
            break;
        }
        match read_lsp_message(&mut reader) {
            Ok(None) => break,
            Ok(Some(message)) => {
                if let Some(id) = message.get("id").and_then(Value::as_u64) {
                    if let Some(error) = message.get("error") {
                        let msg = error
                            .get("message")
                            .and_then(Value::as_str)
                            .unwrap_or("unknown LSP error")
                            .to_string();
                        let _ = response_tx.send((id, Err(msg)));
                    } else {
                        let result = message.get("result").cloned().unwrap_or(Value::Null);
                        let _ = response_tx.send((id, Ok(result)));
                    }
                } else if message.get("method").and_then(Value::as_str)
                    == Some("textDocument/publishDiagnostics")
                {
                    if let Some(params) = message.get("params") {
                        store_diagnostics(&diagnostics, params);
                    }
                }
            }
            Err(_) => break,
        }
    }
}

fn store_diagnostics(store: &Arc<Mutex<HashMap<String, Vec<String>>>>, params: &Value) {
    let Some(uri) = params.get("uri").and_then(Value::as_str) else {
        return;
    };
    let empty = Vec::new();
    let items = params
        .get("diagnostics")
        .and_then(Value::as_array)
        .unwrap_or(&empty);

    let mut lines = Vec::with_capacity(items.len());
    for item in items {
        let severity = match item.get("severity").and_then(Value::as_u64) {
            Some(1) => "error",
            Some(2) => "warning",
            Some(3) => "info",
            Some(4) => "hint",
            _ => "diag",
        };
        let message = item
            .get("message")
            .and_then(Value::as_str)
            .unwrap_or("(no message)");
        let (line, col) = item
            .get("range")
            .and_then(|r| r.get("start"))
            .map(|start| {
                (
                    start.get("line").and_then(Value::as_u64).unwrap_or(0) + 1,
                    start
                        .get("character")
                        .and_then(Value::as_u64)
                        .unwrap_or(0)
                        + 1,
                )
            })
            .unwrap_or((0, 0));
        lines.push(format!("{line}:{col}: {severity}: {message}"));
    }

    if let Ok(mut map) = store.lock() {
        map.insert(uri.to_string(), lines);
    }
}

fn read_lsp_message<R: BufRead>(reader: &mut R) -> Result<Option<Value>> {
    let mut content_length: Option<usize> = None;
    loop {
        let mut line = String::new();
        let read = reader.read_line(&mut line)?;
        if read == 0 {
            return Ok(None);
        }
        let trimmed = line.trim_end();
        if trimmed.is_empty() {
            break;
        }
        if let Some(rest) = trimmed
            .strip_prefix("Content-Length:")
            .or_else(|| trimmed.strip_prefix("content-length:"))
        {
            content_length = Some(rest.trim().parse().context("invalid Content-Length")?);
        }
    }

    let len = content_length.ok_or_else(|| anyhow!("LSP message missing Content-Length"))?;
    let mut buf = vec![0u8; len];
    reader.read_exact(&mut buf)?;
    let value = serde_json::from_slice(&buf).context("invalid LSP JSON")?;
    Ok(Some(value))
}

fn path_to_uri(path: &Path) -> String {
    let abs = if path.is_absolute() {
        path.to_path_buf()
    } else {
        env::current_dir()
            .unwrap_or_else(|_| PathBuf::from("."))
            .join(path)
    };
    format!("file://{}", abs.display())
}

fn uri_to_display(uri: &str) -> String {
    uri.strip_prefix("file://").unwrap_or(uri).to_string()
}

fn command_on_path(name: &str) -> bool {
    Command::new("sh")
        .arg("-c")
        .arg(format!("command -v {name} >/dev/null 2>&1"))
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

fn shellish_split(command: &str) -> Vec<String> {
    command.split_whitespace().map(ToOwned::to_owned).collect()
}
