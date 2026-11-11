use serde_json::{Value, json};
use std::{
    fs,
    io::{Read, Write},
    net::TcpListener,
    process::Command,
    sync::{
        Arc, Mutex,
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
    templates: Arc<Mutex<Vec<Value>>>,
}
impl Mock {
    fn new(completions: Vec<Value>) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let url = format!("http://{}/v1", listener.local_addr().unwrap());
        let stop = Arc::new(AtomicBool::new(false));
        let flag = stop.clone();
        let templates = Arc::new(Mutex::new(Vec::new()));
        let template_requests = templates.clone();
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
                        json!({"default_generation_settings":{"n_ctx":16384},"build_info":"mock","chat_template":"<tool_call><function=read_file>...</function></tool_call>"})
                            .to_string(),
                    )
                } else if headers.starts_with("post /apply-template ") {
                    template_requests
                        .lock()
                        .unwrap()
                        .push(serde_json::from_slice(&bytes[end..end + length]).unwrap());
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
                    let chunks = delta.get("_chunks").and_then(Value::as_array).cloned()
                        .unwrap_or_else(|| vec![json!({"choices":[{"delta":delta,"finish_reason":finish}],"usage":{"prompt_tokens":100,"completion_tokens":20}})]);
                    let mut body = chunks
                        .iter()
                        .map(|chunk| format!("data: {chunk}\n\n"))
                        .collect::<String>();
                    if delta["_no_done"] != true {
                        body.push_str("data: [DONE]\n\n");
                    }
                    (200, "text/event-stream", body)
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
            templates,
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
    run_with_session(mock, root, permission, &root.join("session"))
}

fn run_with_session(
    mock: &Mock,
    root: &std::path::Path,
    permission: &str,
    session: &std::path::Path,
) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_nemocode"))
        .args([
            "run",
            "--workspace",
            root.to_str().unwrap(),
            "--session-dir",
            session.to_str().unwrap(),
            "--permission",
            permission,
            "--prompt",
            "Complete the requested change",
        ])
        .env("NEMO_BASE_URL", &mock.url)
        .env("NEMO_TASK_TIMEOUT_SECS", "5")
        .env("NEMO_CTX", "8192")
        .env("NEMO_MAX_TOKENS", "1024")
        .env("NEMO_PYTHON_LSP", "off")
        .output()
        .unwrap()
}

fn tool(name: &str, arguments: Value) -> Value {
    json!({"tool_calls":[{"index":0,"id":name,"type":"function", "function":{
        "name":name,"arguments":arguments.to_string()}}]})
}

fn response(output: &std::process::Output) -> Value {
    serde_json::from_slice(&output.stdout)
        .unwrap_or_else(|error| panic!("{error}: {}", String::from_utf8_lossy(&output.stderr)))
}

#[test]
fn prompt_history_is_append_only_between_tool_turns() {
    let root = temp();
    fs::write(root.join("code.py"), "x = 1\ny = 2\n").unwrap();
    let mock = Mock::new(vec![
        tool("read_file", json!({"file_path":"code.py","line_count":1})),
        tool(
            "read_file",
            json!({"file_path":"code.py","start_line":2,"line_count":1}),
        ),
        json!({"content":"Inspected."}),
    ]);
    let output = run(&mock, &root, "workspace");
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let requests = mock.finish();
    assert_eq!(requests.len(), 3);
    for pair in requests.windows(2) {
        let earlier = pair[0]["messages"].as_array().unwrap();
        let later = pair[1]["messages"].as_array().unwrap();
        assert_eq!(earlier, &later[..earlier.len()]);
        assert_eq!(pair[0]["tools"], pair[1]["tools"]);
    }
    assert_eq!(
        response(&output)["task"]["reads"].as_array().unwrap().len(),
        2
    );
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn batched_ranges_and_atomic_replacements_pass_real_unittest() {
    let root = temp();
    fs::write(root.join("code.py"), "x = 1\ny = 1\n").unwrap();
    fs::write(root.join("test_code.py"),"import unittest,code\nclass TestCode(unittest.TestCase):\n def test_values(self):\n  self.assertEqual((code.x,code.y),(2,3))\n").unwrap();
    let mock = Mock::new(vec![
        tool(
            "read_multiple_files",
            json!({"files":[{"path":"code.py","start_line":1,"line_count":2},{"path":"test_code.py","start_line":2,"line_count":3}]}),
        ),
        tool(
            "edit_file",
            json!({"file_path":"code.py","edits":[{"original_snippet":"x = 1","new_snippet":"x = 2"},{"original_snippet":"y = 1","new_snippet":"y = 3"}]}),
        ),
        tool("run_unittest", json!({})),
        json!({"content":"Updated and tested."}),
    ]);
    let output = run(&mock, &root, "workspace");
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(
        fs::read_to_string(root.join("code.py")).unwrap(),
        "x = 2\ny = 3\n"
    );
    assert_eq!(response(&output)["verification_status"], "passed");
    assert_eq!(response(&output)["task_outcome"], "completed");
    let requests = mock.finish();
    let result = requests[1]["messages"].as_array().unwrap().last().unwrap()["content"]
        .as_str()
        .unwrap();
    assert!(result.contains("2: class TestCode"));
    assert!(!result.contains("1: import unittest"));
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn invalid_batched_edit_writes_nothing() {
    let root = temp();
    fs::write(root.join("code.py"), "x = 1\ny = 1\n").unwrap();
    let mock = Mock::new(vec![
        tool("read_file", json!({"file_path":"code.py"})),
        tool(
            "edit_file",
            json!({"file_path":"code.py","edits":[{"original_snippet":"x = 1","new_snippet":"x = 2"},{"original_snippet":"missing","new_snippet":"y = 3"}]}),
        ),
        json!({"content":"Unable to edit."}),
    ]);
    let output = run(&mock, &root, "workspace");
    assert!(!output.status.success());
    assert_eq!(
        fs::read_to_string(root.join("code.py")).unwrap(),
        "x = 1\ny = 1\n"
    );
    mock.finish();
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn empty_unittest_suite_is_not_successful_verification() {
    let root = temp();
    fs::write(root.join("code.py"), "x = 1\n").unwrap();
    let mock = Mock::new(vec![
        tool("run_unittest", json!({})),
        json!({"content":"All good."}),
    ]);
    let output = run(&mock, &root, "workspace");
    assert!(!output.status.success());
    let journal = fs::read_to_string(root.join("session/events.jsonl")).unwrap();
    assert!(journal.contains("no_tests_collected"));
    assert_eq!(response(&output)["verification_status"], "incomplete");
    mock.finish();
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn consecutive_edits_use_committed_revisions_without_rereading() {
    let root = temp();
    fs::write(root.join("code.py"), "x = 1\ny = 1\n").unwrap();
    let mock = Mock::new(vec![
        tool("read_file", json!({"file_path":"code.py"})),
        tool(
            "edit_file",
            json!({"file_path":"code.py","original_snippet":"x = 1","new_snippet":"x = 2"}),
        ),
        tool(
            "edit_file",
            json!({"file_path":"code.py","original_snippet":"y = 1","new_snippet":"y = 2"}),
        ),
        json!({"content":"Updated; behavior checks were not run."}),
    ]);
    let output = run(&mock, &root, "workspace");
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(
        fs::read_to_string(root.join("code.py")).unwrap(),
        "x = 2\ny = 2\n"
    );
    assert_eq!(response(&output)["task_outcome"], "completed_unverified");
    assert_eq!(mock.finish().len(), 4);
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn model_navigation_retains_workspace_boundary_across_resume() {
    let root = temp();
    fs::create_dir(root.join("subdir")).unwrap();
    fs::write(root.join("code.py"), "x = 1\n").unwrap();
    let mock = Mock::new(vec![
        tool("change_directory", json!({"path":"subdir"})),
        tool("read_file", json!({"file_path":"../code.py"})),
        tool("change_directory", json!({"path":".."})),
        tool("change_directory", json!({"path":"subdir"})),
        json!({"content":"Inspected."}),
    ]);
    let output = run(&mock, &root, "workspace");
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    mock.finish();
    let mock = Mock::new(vec![
        tool("change_directory", json!({"path":".."})),
        json!({"content":"Resumed."}),
    ]);
    let output = Command::new(env!("CARGO_BIN_EXE_nemocode"))
        .args(["run", "--workspace"])
        .arg(&root)
        .arg("--resume")
        .arg(root.join("session"))
        .env("NEMO_BASE_URL", &mock.url)
        .env("NEMO_PYTHON_LSP", "off")
        .env("NEMO_TASK_TIMEOUT_SECS", "5")
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(response(&output)["task"]["cwd"], root.to_str().unwrap());
    let journal = fs::read_to_string(root.join("session/events.jsonl")).unwrap();
    assert!(!journal.contains("permission_denied"));
    mock.finish();
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn failed_syntax_check_cannot_be_reported_as_success() {
    let root = temp();
    let mock = Mock::new(vec![
        tool(
            "create_file",
            json!({"file_path":"broken.py","content":"def broken(:\n"}),
        ),
        tool("python_diagnostics", json!({"paths":["broken.py"]})),
        json!({"content":"Done; all checks passed."}),
    ]);
    let output = run(&mock, &root, "workspace");
    assert!(!output.status.success());
    let body = response(&output);
    assert_eq!(body["ok"], false);
    assert_eq!(body["verification_status"], "failed");
    assert_eq!(body["task"]["checks"][0]["status"], "failed");
    assert!(
        !body["answer"]
            .as_str()
            .unwrap()
            .contains("all checks passed")
    );
    assert_eq!(mock.finish().len(), 5);
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn malformed_tool_requires_successful_protocol_recovery() {
    let root = temp();
    let invalid = json!({"tool_calls":[{"index":0,"id":"bad","type":"function",
        "function":{"name":"create_file","arguments":"{\"file_path\":\"oops\","}}]});
    let mock = Mock::new(vec![invalid.clone()]);
    let output = run(&mock, &root, "workspace");
    assert!(!output.status.success());
    assert_eq!(response(&output)["task"]["phase"], "incomplete");
    assert!(!root.join("oops").exists());
    assert_eq!(mock.finish().len(), 3);
    let mock = Mock::new(vec![
        invalid,
        tool("read_file", json!({"file_path":"code.py"})),
        json!({"content":"Inspected code."}),
    ]);
    fs::write(root.join("code.py"), "x = 1\n").unwrap();
    let output = run_with_session(&mock, &root, "workspace", &root.join("session2"));
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    mock.finish();
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn batch_conflicts_are_detected_before_any_file_is_written() {
    let root = temp();
    fs::write(root.join("existing.py"), "old = True\n").unwrap();
    let mock = Mock::new(vec![
        tool(
            "create_multiple_files",
            json!({"files":[
                {"path":"new.py","content":"x = 1\n"}, {"path":"existing.py","content":"x = 2\n"}
            ]}),
        ),
        json!({"content":"Done."}),
    ]);
    let output = run(&mock, &root, "workspace");
    assert!(!output.status.success());
    assert!(!root.join("new.py").exists());
    assert_eq!(
        fs::read_to_string(root.join("existing.py")).unwrap(),
        "old = True\n"
    );
    assert!(
        response(&output)["task"]["changed_files"]
            .as_array()
            .unwrap()
            .is_empty()
    );
    mock.finish();
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn equivalent_failed_navigation_stops_after_bounded_recovery() {
    let root = temp();
    let mock = Mock::new(
        [
            "missing",
            "./missing",
            "missing/../missing",
            "./missing",
            "missing",
        ]
        .iter()
        .map(|path| tool("change_directory", json!({"path":path})))
        .collect(),
    );
    let output = run(&mock, &root, "workspace");
    assert!(!output.status.success());
    assert_eq!(response(&output)["task"]["phase"], "blocked");
    assert_eq!(mock.finish().len(), 4);
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn fragmented_unicode_tool_arguments_execute_once() {
    let root = temp();
    let mock = Mock::new(vec![
        json!({"_chunks":[
            {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"one","type":"function","function":{"name":"create_","arguments":"{\"file_path\":\"unicode.py\","}}]}}]},
            {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"name":"file","arguments":"\"content\":\"greeting = 'héllo 🌍'\\n\"}"}}]},"finish_reason":"tool_calls"}],"usage":{"completion_tokens":30}}
        ]}),
        json!({"content":"Created the file; unverified."}),
    ]);
    let output = run(&mock, &root, "workspace");
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(
        fs::read_to_string(root.join("unicode.py")).unwrap(),
        "greeting = 'héllo 🌍'\n"
    );
    assert_eq!(
        response(&output)["task"]["changed_files"]
            .as_array()
            .unwrap()
            .len(),
        1
    );
    mock.finish();
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn unterminated_stream_never_executes_a_partial_operation_and_records_usage() {
    let root = temp();
    let delta = tool(
        "create_file",
        json!({"file_path":"oops.py","content":"x = 1\n"}),
    );
    let mock = Mock::new(vec![
        json!({"_chunks":[{"choices":[{"delta":delta}]}],"_no_done":true}),
    ]);
    let output = run(&mock, &root, "workspace");
    assert!(!output.status.success());
    assert!(!root.join("oops.py").exists());
    let events: Vec<Value> = fs::read_to_string(root.join("session/events.jsonl"))
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    let completion = events.iter().find(|e| e["kind"] == "completion").unwrap();
    assert_eq!(completion["data"]["complete"], false);
    assert!(completion["data"]["generated_tokens"].as_u64().unwrap() > 0);
    mock.finish();
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn workspace_denies_model_shell_execution() {
    let root = temp();
    let mock = Mock::new(vec![
        json!({"tool_calls":[{"index":0,"id":"shell","type":"function","function":{"name":"execute_bash_command","arguments":"{\"command\":\"touch forbidden\"}"}}]}),
        json!({"content":"Shell was denied."}),
    ]);
    let _ = run(&mock, &root, "workspace");
    assert!(!root.join("forbidden").exists());
    let requests = mock.finish();
    assert!(
        requests[1]["messages"]
            .as_array()
            .unwrap()
            .iter()
            .any(|message| message["role"] == "tool"
                && message["content"]
                    .as_str()
                    .is_some_and(|content| content.contains("requires --permission trusted")))
    );
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn missing_nested_read_suggests_existing_file_in_current_directory() {
    let root = temp();
    fs::write(root.join("stats.py"), "value = 1\n").unwrap();
    let mock = Mock::new(vec![
        json!({"tool_calls":[{"index":0,"id":"read","type":"function","function":{"name":"read_file","arguments":"{\"file_path\":\"src/stats.py\"}"}}]}),
        json!({"content":"Unable to read the requested path."}),
    ]);
    assert!(run(&mock, &root, "workspace").status.success());
    let requests = mock.finish();
    assert!(
        requests[1]["messages"]
            .as_array()
            .unwrap()
            .iter()
            .any(|message| message["role"] == "tool"
                && message["content"]
                    .as_str()
                    .and_then(|content| serde_json::from_str::<Value>(
                        content.split("\n\n").next()?
                    )
                    .ok())
                    .is_some_and(|outcome| outcome["error_code"] == "not_found"
                        && outcome["summary"].as_str().is_some_and(|summary| summary
                            .contains("\"stats.py\" exists in the current directory"))))
    );
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn relative_session_directory_survives_workspace_navigation() {
    let root = temp();
    fs::create_dir(root.join("subdir")).unwrap();
    let mock = Mock::new(vec![
        json!({"tool_calls":[{"index":0,"id":"cd","type":"function","function":{"name":"change_directory","arguments":"{\"path\":\"subdir\"}"}}]}),
        json!({"content":"Done."}),
    ]);
    let result = run_with_session(&mock, &root, "workspace", std::path::Path::new("session"));
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let checkpoint: Value =
        serde_json::from_slice(&fs::read(root.join("session/checkpoint.json")).unwrap()).unwrap();
    assert_eq!(checkpoint["task"]["phase"], "finished");
    assert!(!root.join("subdir/session").exists());
    let response: Value = serde_json::from_slice(&result.stdout).unwrap();
    assert_eq!(response["session"], root.join("session").to_str().unwrap());
    mock.finish();
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn readonly_rejects_native_write_and_workspace_rejects_stale_edit() {
    let root = temp();
    fs::write(root.join("code.py"), "value = 1\n").unwrap();
    let mock = Mock::new(vec![
        json!({"tool_calls":[{"index":0,"id":"write","type":"function","function":{"name":"create_file","arguments":"{\"file_path\":\"oops\",\"content\":\"bad\"}"}}]}),
        json!({"content":"Write was denied."}),
    ]);
    assert!(!run(&mock, &root, "read-only").status.success());
    assert!(!root.join("oops").exists());
    assert!(
        mock.finish()[1]["messages"]
            .as_array()
            .unwrap()
            .iter()
            .any(|m| m["role"] == "tool"
                && m["content"]
                    .as_str()
                    .is_some_and(|s| s.contains("read-only permission denies")))
    );
    let mock = Mock::new(vec![
        json!({"tool_calls":[{"index":0,"id":"edit","type":"function","function":{"name":"edit_file","arguments":"{\"file_path\":\"code.py\",\"original_snippet\":\"value = 1\",\"new_snippet\":\"value = 2\",\"expected_sha256\":\"stale\"}"}}]}),
        json!({"content":"Stale edit was denied."}),
    ]);
    assert!(
        !run_with_session(&mock, &root, "workspace", &root.join("session2"))
            .status
            .success()
    );
    assert_eq!(
        fs::read_to_string(root.join("code.py")).unwrap(),
        "value = 1\n"
    );
    assert!(
        mock.finish()[1]["messages"]
            .as_array()
            .unwrap()
            .iter()
            .any(|m| m["role"] == "tool"
                && m["content"]
                    .as_str()
                    .is_some_and(|s| s.contains("stale edit")))
    );
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn retries_failed_file_change_before_accepting_a_completion_claim() {
    let root = temp();
    fs::write(root.join("code.py"), "value = 1\n").unwrap();
    let mock = Mock::new(vec![
        json!({"tool_calls":[{"index":0,"id":"read","type":"function","function":{"name":"read_file","arguments":"{\"file_path\":\"code.py\"}"}}]}),
        json!({"tool_calls":[{"index":0,"id":"create","type":"function","function":{"name":"create_file","arguments":"{\"file_path\":\"code.py\",\"content\":\"value = 2\\n\"}"}}]}),
        json!({"content":"Updated the file."}),
        json!({"tool_calls":[{"index":0,"id":"edit","type":"function","function":{"name":"edit_file","arguments":"{\"file_path\":\"code.py\",\"original_snippet\":\"value = 1\",\"new_snippet\":\"value = 2\"}"}}]}),
        json!({"content":"Updated; tests were not run."}),
    ]);
    let result = run(&mock, &root, "workspace");
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert_eq!(
        fs::read_to_string(root.join("code.py")).unwrap(),
        "value = 2\n"
    );
    let result: Value = serde_json::from_slice(&result.stdout).unwrap();
    assert_eq!(result["task"]["failed_mutations"], json!({}));
    assert_eq!(result["task"]["changed_files"].as_array().unwrap().len(), 1);
    let requests = mock.finish();
    assert_eq!(requests.len(), 5);
    assert!(
        requests[3]["messages"].as_array().unwrap().last().unwrap()["content"]
            .as_str()
            .unwrap()
            .contains("have not succeeded")
    );
    fs::remove_dir_all(root).unwrap();
}

#[cfg(unix)]
#[test]
fn git_retrieval_fallback_does_not_read_symlinks_outside_workspace() {
    use std::os::unix::{fs::PermissionsExt, fs::symlink};
    let root = temp();
    let workspace = root.join("workspace");
    fs::create_dir(&workspace).unwrap();
    assert!(
        Command::new("git")
            .args(["init", "-q"])
            .arg(&workspace)
            .status()
            .unwrap()
            .success()
    );
    fs::write(root.join("outside.py"), "needle outside secret\n").unwrap();
    fs::write(workspace.join(".env"), "needle protected secret\n").unwrap();
    fs::write(workspace.join("safe.py"), "needle allowed\n").unwrap();
    symlink(root.join("outside.py"), workspace.join("external.py")).unwrap();
    symlink(workspace.join(".env"), workspace.join("alias.py")).unwrap();
    let bin = root.join("bin");
    fs::create_dir(&bin).unwrap();
    fs::write(bin.join("rg"), "#!/bin/sh\nexit 2\n").unwrap();
    fs::set_permissions(bin.join("rg"), fs::Permissions::from_mode(0o755)).unwrap();
    let mock = Mock::new(vec![
        json!({"tool_calls":[{"index":0,"id":"search","type":"function","function":{"name":"search_code","arguments":"{\"query\":\"needle\"}"}}]}),
        json!({"content":"Searched."}),
    ]);
    let mut paths = vec![bin];
    paths.extend(std::env::split_paths(
        &std::env::var_os("PATH").unwrap_or_default(),
    ));
    let result = Command::new(env!("CARGO_BIN_EXE_nemocode"))
        .args(["run", "--workspace"])
        .arg(&workspace)
        .arg("--session-dir")
        .arg(root.join("session"))
        .args(["--permission", "read-only", "--prompt", "Search for needle"])
        .env("PATH", std::env::join_paths(paths).unwrap())
        .env("NEMO_BASE_URL", &mock.url)
        .env("NEMO_TASK_TIMEOUT_SECS", "5")
        .env("NEMO_PYTHON_LSP", "off")
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let requests = mock.finish();
    let result = requests[1]["messages"]
        .as_array()
        .unwrap()
        .iter()
        .find(|m| m["role"] == "tool")
        .unwrap()["content"]
        .as_str()
        .unwrap();
    assert!(result.contains("needle allowed"), "{result}");
    assert!(!result.contains("secret"), "{result}");
    assert!(!result.contains("external.py"), "{result}");
    assert!(!result.contains("alias.py"), "{result}");
    fs::remove_dir_all(root).unwrap();
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
    let templates = mock.templates.lock().unwrap().clone();
    let requests = mock.finish();
    assert_eq!(requests.len(), 1);
    assert_eq!(
        requests[0]["chat_template_kwargs"]["enable_thinking"],
        false
    );
    assert_eq!(requests[0]["parallel_tool_calls"], false);
    assert_eq!(requests[0]["stop"], json!(["</tool_call>"]));
    assert_eq!(
        templates[0]["chat_template_kwargs"],
        requests[0]["chat_template_kwargs"]
    );
    assert!(!String::from_utf8_lossy(&result.stderr).contains("\x1b[2K"));
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
    assert_eq!(messages.last().unwrap()["role"], "tool");
    assert!(
        messages.last().unwrap()["content"]
            .as_str()
            .unwrap()
            .contains("TASK STATE")
    );
    let offered = requests[0]["tools"].as_array().unwrap();
    assert!(!offered.iter().any(|tool| matches!(
        tool["function"]["name"].as_str(),
        Some("run_python" | "execute_bash_command")
    )));
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
fn workspace_permission_rejects_pytest_target_outside_workspace() {
    let root = temp();
    let mock = Mock::new(vec![
        json!({"tool_calls":[{"index":0,"id":"test","type":"function","function":{"name":"run_pytest","arguments":"{\"target\":\"../elsewhere/test_x.py::test_a\"}"}}]}),
        json!({"content":"The target was rejected."}),
    ]);
    run(&mock, &root, "workspace");
    assert!(
        mock.finish()[1]["messages"]
            .as_array()
            .unwrap()
            .iter()
            .any(|m| m["role"] == "tool"
                && m["content"]
                    .as_str()
                    .is_some_and(|s| s.contains("escapes workspace")))
    );
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
