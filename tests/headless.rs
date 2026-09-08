use serde_json::{Value, json};
use std::{
    fs,
    io::{Read, Write},
    net::TcpListener,
    process::Command,
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
    thread,
    time::{Duration, Instant},
};
static SEQ: AtomicUsize = AtomicUsize::new(0);

struct Mock {
    url: String,
    stop: Arc<AtomicBool>,
    thread: Option<thread::JoinHandle<Vec<Value>>>,
}
impl Mock {
    fn new(completions: Vec<Value>) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let url = format!("http://{}/v1", listener.local_addr().unwrap());
        let stop = Arc::new(AtomicBool::new(false));
        let flag = stop.clone();
        let handle = thread::spawn(move || {
            let mut requests = Vec::new();
            let mut index = 0;
            while !flag.load(Ordering::Relaxed) {
                let (mut stream, _) = match listener.accept() {
                    Ok(s) => s,
                    Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                        thread::sleep(Duration::from_millis(5));
                        continue;
                    }
                    Err(e) => panic!("{e}"),
                };
                stream
                    .set_read_timeout(Some(Duration::from_secs(3)))
                    .unwrap();
                let mut bytes = Vec::new();
                let mut buffer = [0; 4096];
                let end = loop {
                    let count = stream.read(&mut buffer).unwrap();
                    if count == 0 {
                        panic!("unexpected EOF");
                    }
                    bytes.extend_from_slice(&buffer[..count]);
                    if let Some(i) = bytes.windows(4).position(|w| w == b"\r\n\r\n") {
                        break i + 4;
                    }
                };
                let headers = String::from_utf8_lossy(&bytes[..end]).to_lowercase();
                let length = headers
                    .lines()
                    .find_map(|line| {
                        line.strip_prefix("content-length:")
                            .and_then(|s| s.trim().parse::<usize>().ok())
                    })
                    .unwrap_or(0);
                while bytes.len() - end < length {
                    let n = stream.read(&mut buffer).unwrap();
                    if n == 0 {
                        break;
                    }
                    bytes.extend_from_slice(&buffer[..n]);
                }
                let (status, kind, body) = if headers.starts_with("get /props ") {
                    (
                        200,
                        "application/json",
                        json!({"default_generation_settings":{"n_ctx":16384},"build_info":"mock"})
                            .to_string(),
                    )
                } else if headers.starts_with("post /apply-template ") {
                    (404, "application/json", "{}".to_string())
                } else if headers.starts_with("post /v1/chat/completions ") {
                    requests.push(serde_json::from_slice(&bytes[end..end + length]).unwrap());
                    let delta = completions
                        .get(index)
                        .cloned()
                        .unwrap_or(json!({"content":"finished"}));
                    index += 1;
                    let finish = if delta.get("tool_calls").is_some() {
                        "tool_calls"
                    } else {
                        "stop"
                    };
                    let chunk = json!({"choices":[{"delta":delta,"finish_reason":finish}],"usage":{"prompt_tokens":100,"completion_tokens":20}});
                    (
                        200,
                        "text/event-stream",
                        format!("data: {chunk}\n\ndata: [DONE]\n\n"),
                    )
                } else {
                    (404, "application/json", "{}".to_string())
                };
                let response = format!(
                    "HTTP/1.1 {status} OK\r\nContent-Type: {kind}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len()
                );
                let _ = stream.write_all(response.as_bytes());
            }
            requests
        });
        Self {
            url,
            stop,
            thread: Some(handle),
        }
    }
    fn finish(mut self) -> Vec<Value> {
        self.stop.store(true, Ordering::Relaxed);
        self.thread.take().unwrap().join().unwrap()
    }
}
impl Drop for Mock {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(handle) = self.thread.take() {
            let _ = handle.join();
        }
    }
}

fn temp() -> std::path::PathBuf {
    let p = std::env::temp_dir().join(format!(
        "nemo-integration-{}-{}",
        std::process::id(),
        SEQ.fetch_add(1, Ordering::Relaxed)
    ));
    fs::create_dir_all(&p).unwrap();
    p
}
fn run(mock: &Mock, root: &std::path::Path, permission: &str) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_nemocode"))
        .args([
            "run",
            "--workspace",
            root.to_str().unwrap(),
            "--session-dir",
            root.join("session").to_str().unwrap(),
            "--permission",
            permission,
            "--prompt",
            "Complete the requested change",
        ])
        .env("NEMO_BASE_URL", &mock.url)
        .env("NEMO_TASK_TIMEOUT_SECS", "5")
        .env("NEMO_CTX", "16384")
        .env("NEMO_MAX_TOKENS", "1024")
        .env("NEMO_PYTHON_LSP", "off")
        .output()
        .unwrap()
}

#[test]
fn headless_runs_outside_checkout_and_emits_one_json_result() {
    let mock = Mock::new(vec![json!({"content":"Done."})]);
    let root = temp();
    let result = run(&mock, &root, "read-only");
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let json: Value = serde_json::from_slice(&result.stdout).unwrap();
    assert_eq!(json["answer"], "Done.");
    assert!(root.join("session/checkpoint.json").is_file());
    assert_eq!(mock.finish().len(), 1);
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn executes_native_tools_and_preserves_tool_results_in_next_request() {
    let mock = Mock::new(vec![
        json!({"tool_calls":[{"index":0,"id":"write","type":"function","function":{"name":"create_file","arguments":"{\"file_path\":\"hello.py\",\"content\":\"print(1)\\n\"}"}}]}),
        json!({"content":"Created the file; checks were not run."}),
    ]);
    let root = temp();
    let result = run(&mock, &root, "workspace");
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert_eq!(
        fs::read_to_string(root.join("hello.py")).unwrap(),
        "print(1)\n"
    );
    let requests = mock.finish();
    let messages = requests[1]["messages"].as_array().unwrap();
    assert!(
        messages
            .iter()
            .any(|m| m["role"] == "tool" && m["tool_call_id"] == "write")
    );
    let journal = fs::read_to_string(root.join("session/events.jsonl")).unwrap();
    assert!(journal.contains("tool_finished"));
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn reasoning_markup_does_not_execute_and_readonly_denies_mutations() {
    let root = temp();
    let mock = Mock::new(vec![
        json!({"reasoning_content":"<function=create_file><parameter=file_path>oops</parameter><parameter=content>bad</parameter></function>","content":"No action"}),
    ]);
    assert!(run(&mock, &root, "read-only").status.success());
    assert!(!root.join("oops").exists());
    mock.finish();
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn no_header_server_is_bounded_by_task_deadline() {
    let root = temp();
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    let start = Instant::now();
    let result = Command::new(env!("CARGO_BIN_EXE_nemocode"))
        .args([
            "run",
            "--workspace",
            root.to_str().unwrap(),
            "--session-dir",
            root.join("session").to_str().unwrap(),
            "--prompt",
            "test",
        ])
        .env("NEMO_BASE_URL", format!("http://{address}/v1"))
        .env("NEMO_TASK_TIMEOUT_SECS", "1")
        .output()
        .unwrap();
    assert!(!result.status.success());
    assert!(start.elapsed() < Duration::from_secs(4));
    fs::remove_dir_all(root).unwrap();
}
