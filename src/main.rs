//! NemoCode — local coding-agent harness
//!
//! Runs against a local OpenAI-compatible llama.cpp server serving the bundled
//! Nemotron-3-Nano-4B-Coding-Agent GGUF. This is the only model the harness uses.
//!
//! Environment variables:
//!   NEMO_BASE_URL         optional, default: http://127.0.0.1:8080/v1
//!   NEMO_MODEL            optional, default: Nemotron-3-Nano-4B-Coding-Agent-Q4_K_M
//!   NEMO_API_KEY          optional, default: local (llama-server ignores unless configured)
//!   NEMO_MAX_TOKENS       optional, default: 4096
//!   NEMO_TOOL_ROUNDS      optional, default: 8
//!   NEMO_CONTEXT_BUDGET   optional, default: 12000 (approx prompt tokens kept)

use anyhow::{anyhow, bail, Context, Result};
use dotenvy::dotenv;
use eventsource_stream::Eventsource;
use futures_util::{future::join_all, StreamExt};
use owo_colors::OwoColorize;
use reqwest::Client;
use rustyline::{
    completion::{Completer, FilenameCompleter, Pair},
    error::ReadlineError,
    highlight::Highlighter,
    hint::Hinter,
    history::DefaultHistory,
    validate::Validator,
    CompletionType, Config, Context as RlContext, Editor, Helper,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{
    collections::{HashMap, HashSet},
    env,
    fs,
    io::{self, Read, Write},
    path::{Component, Path, PathBuf},
    process::{Command, Stdio},
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Mutex,
    },
    thread,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};
use walkdir::{DirEntry, WalkDir};

const MAX_FILE_SIZE: u64 = 5_000_000;
const MAX_DIRECTORY_FILES: usize = 1_000;
const SHELL_COMMAND_TIMEOUT_SECS: u64 = 120;
const MAX_SHELL_OUTPUT_BYTES: usize = 96 * 1024;
const MAX_LIST_DIRECTORY_ENTRIES: usize = 500;
const MAX_TOOL_RESULT_CHARS: usize = 48 * 1024;
const MAX_FILE_TOOL_RESULT_CHARS: usize = 12 * 1024;
const DEFAULT_MAX_TOKENS: u64 = 4_096;
const DEFAULT_CONTEXT_TOKEN_BUDGET: u32 = 12_000;
const COMPACTION_MARKER: &str = "[NemoCode context summary] Earlier messages were compacted to stay within the local context budget. Prefer the visible recent tool results and file contents as authoritative.";

const DEFAULT_BASE_URL: &str = "http://127.0.0.1:8080/v1";
const DEFAULT_MODEL: &str = "Nemotron-3-Nano-4B-Coding-Agent-Q4_K_M";

const BANNER: &str = r#"┳┓┏┓┳┳┓┏┓┏┓┏┓┳┓┏┓
┃┃┣ ┃┃┃┃┃┃ ┃┃┃┃┣ 
┛┗┗┛┛ ┗┗┛┗┛┗┛┻┛┗┛"#;

const SYSTEM_PROMPT: &str = r#"You are NemoCode, a fast local coding agent with strong software engineering judgment.
Your expertise spans system design, algorithms, testing, and best practices.
You provide thoughtful, well-structured solutions while explaining your reasoning.

Core capabilities:
1. Code Analysis & Discussion
   - Analyze code with expert-level insight
   - Explain complex concepts clearly
   - Suggest optimizations and best practices
   - Debug issues with precision

2. File Operations (via function calls):
   - read_file: Read a single file's content
   - read_multiple_files: Read multiple files at once
   - create_file: Create or overwrite a single file
   - create_multiple_files: Create multiple files at once
   - edit_file: Make precise edits to existing files using snippet replacement

3. Filesystem Navigation (via function calls):
   - list_directory: List files and folders in a directory
   - change_directory: Persistently change the process working directory
   - execute_bash_command: Run bash commands (ls, find, git, cargo, etc.)
   - The user can also navigate with /cd, !cd, or interactive ! shell mode
   - When the working directory changes, the turn includes SESSION LOCATION (cwd + project root)
   - Relative paths resolve against the current working directory unless absolute
   - Prefer list_directory or change_directory before broad bash exploration
   - Prefer paths relative to the latest SESSION LOCATION / current working directory

Guidelines:
1. Provide natural, conversational responses explaining your reasoning
2. Use function calls when you need to read, modify, list, or navigate the filesystem
3. For file operations:
   - Always read files first before editing them to understand the context
   - Use precise snippet matching for edits
   - Explain what changes you're making and why
   - Consider the impact of changes on the overall codebase
4. Follow language-specific best practices
5. Suggest tests or validation steps when appropriate
6. Be thorough in your analysis and recommendations

IMPORTANT: If something requires a tool call, call the tool promptly. Prefer action over long speculation when file operations are needed.

Remember: You are a senior engineer - be thoughtful, precise, and explain your reasoning clearly."#;

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct FunctionCall {
    name: String,
    arguments: String,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct ToolCall {
    id: String,
    #[serde(rename = "type")]
    kind: String,
    function: FunctionCall,
}

#[derive(Debug, Default)]
struct StreamedAssistantMessage {
    content: String,
    reasoning_content: String,
    tool_calls: Vec<ToolCall>,
    finish_reason: Option<String>,
}

#[derive(Debug, Clone, Default)]
struct FileReadCache {
    entries: Arc<Mutex<HashMap<PathBuf, CachedFile>>>,
}

#[derive(Debug, Clone)]
struct CachedFile {
    modified: Option<SystemTime>,
    len: u64,
    content: String,
}

impl FileReadCache {
    fn get_or_load(&self, path: &Path) -> Result<String> {
        let metadata = fs::metadata(path)
            .with_context(|| format!("could not stat '{}'", path.display()))?;
        let modified = metadata.modified().ok();
        let len = metadata.len();

        if let Ok(guard) = self.entries.lock() {
            if let Some(cached) = guard.get(path) {
                if cached.modified == modified && cached.len == len {
                    return Ok(cached.content.clone());
                }
            }
        }

        let content = fs::read_to_string(path)
            .with_context(|| format!("could not read '{}' as UTF-8 text", path.display()))?;

        if let Ok(mut guard) = self.entries.lock() {
            guard.insert(
                path.to_path_buf(),
                CachedFile {
                    modified,
                    len,
                    content: content.clone(),
                },
            );
        }

        Ok(content)
    }

    fn invalidate(&self, path: &Path) {
        if let Ok(mut guard) = self.entries.lock() {
            guard.remove(path);
        }
    }

    fn clear(&self) {
        if let Ok(mut guard) = self.entries.lock() {
            guard.clear();
        }
    }
}

fn is_read_only_tool(name: &str) -> bool {
    matches!(
        name,
        "read_file" | "read_multiple_files" | "list_directory"
    )
}

fn record_tool_call_repetition(
    seen: &mut HashMap<String, u32>,
    name: &str,
    arguments: &str,
) -> Option<String> {
    if !is_read_only_tool(name) {
        seen.clear();
        return None;
    }

    let key = format!("{name}\u{1}{arguments}");
    let count = seen.entry(key).and_modify(|value| *value += 1).or_insert(1);
    if *count >= 3 {
        Some(format!(
            "\n\n[NemoCode] This exact `{name}` call has now run {count} times this turn with no mutations in between, so its output cannot differ. Do not repeat it. Use the result above and take the next concrete step; if blocked, summarize instead of calling tools."
        ))
    } else {
        None
    }
}

struct NemoCode {
    client: Client,
    api_key: String,
    endpoint: String,
    model: String,
    project_root: PathBuf,
    max_tokens: u64,
    max_tool_rounds: usize,
    context_token_budget: u32,
    /// First non-system history index included in requests. Only moves forward.
    request_floor: usize,
    last_session_cwd: Option<PathBuf>,
    file_read_cache: FileReadCache,
    tools: Value,
    conversation_history: Vec<Value>,
}

impl NemoCode {
    fn from_env() -> Result<Self> {
        dotenv().ok();

        let project_root = require_nemocode_launch_directory()?;

        let base_url = env::var("NEMO_BASE_URL").unwrap_or_else(|_| DEFAULT_BASE_URL.to_string());
        let endpoint = format!("{}/chat/completions", base_url.trim_end_matches('/'));

        let model = env::var("NEMO_MODEL").unwrap_or_else(|_| DEFAULT_MODEL.to_string());
        let api_key = env::var("NEMO_API_KEY").unwrap_or_else(|_| "local".to_string());

        let max_tokens = env::var("NEMO_MAX_TOKENS")
            .ok()
            .map(|value| value.parse::<u64>())
            .transpose()
            .context("NEMO_MAX_TOKENS must be a positive integer")?
            .unwrap_or(DEFAULT_MAX_TOKENS);

        let max_tool_rounds = env::var("NEMO_TOOL_ROUNDS")
            .ok()
            .map(|value| value.parse::<usize>())
            .transpose()
            .context("NEMO_TOOL_ROUNDS must be a positive integer")?
            .unwrap_or(8);

        if max_tool_rounds == 0 {
            bail!("NEMO_TOOL_ROUNDS must be greater than zero");
        }

        let context_token_budget = env::var("NEMO_CONTEXT_BUDGET")
            .ok()
            .map(|value| value.parse::<u32>())
            .transpose()
            .context("NEMO_CONTEXT_BUDGET must be a positive integer")?
            .unwrap_or(DEFAULT_CONTEXT_TOKEN_BUDGET);

        if context_token_budget == 0 {
            bail!("NEMO_CONTEXT_BUDGET must be greater than zero");
        }

        let client = Client::builder()
            .user_agent("nemocode/0.1")
            .build()
            .context("failed to construct HTTP client")?;

        Ok(Self {
            client,
            api_key,
            endpoint,
            model,
            project_root,
            max_tokens,
            max_tool_rounds,
            context_token_budget,
            request_floor: 1,
            last_session_cwd: None,
            file_read_cache: FileReadCache::default(),
            tools: tool_definitions(),
            conversation_history: vec![json!({
                "role": "system",
                "content": SYSTEM_PROMPT,
            })],
        })
    }

    fn maybe_locate_user_message(&mut self, user_message: String) -> String {
        let cwd = env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
        let cwd = canonicalize_existing_or_self(cwd);
        let changed = self
            .last_session_cwd
            .as_ref()
            .is_none_or(|previous| previous != &cwd);

        if changed {
            self.last_session_cwd = Some(cwd);
            format!(
                "{}\n\n{}",
                session_location_block(&self.project_root),
                user_message
            )
        } else {
            user_message
        }
    }

    async fn handle_user_message(&mut self, user_message: String) -> Result<()> {
        let located = self.maybe_locate_user_message(user_message);
        self.conversation_history.push(json!({
            "role": "user",
            "content": located,
        }));

        let mut repeated_tool_calls = HashMap::<String, u32>::new();

        for round in 1..=self.max_tool_rounds {
            let StreamedAssistantMessage {
                content,
                reasoning_content,
                tool_calls,
                finish_reason,
            } = self.stream_completion().await?;

            let mut assistant_message = serde_json::Map::new();
            assistant_message.insert("role".to_string(), json!("assistant"));
            assistant_message.insert(
                "content".to_string(),
                if content.is_empty() {
                    Value::Null
                } else {
                    json!(content)
                },
            );

            if tool_calls.is_empty() {
                self.conversation_history
                    .push(Value::Object(assistant_message));

                if matches!(finish_reason.as_deref(), Some("length")) {
                    println!(
                        "{}",
                        "Response stopped because the token limit was reached."
                            .yellow()
                            .bold()
                    );
                }
                return Ok(());
            }

            if !reasoning_content.is_empty() {
                assistant_message
                    .insert("reasoning_content".to_string(), json!(reasoning_content));
            }

            assistant_message.insert(
                "tool_calls".to_string(),
                serde_json::to_value(&tool_calls)
                    .context("failed to serialize streamed tool calls")?,
            );
            self.conversation_history
                .push(Value::Object(assistant_message));

            println!();
            println!(
                "{}",
                format!("Executing {} function call(s)...", tool_calls.len())
                    .bright_cyan()
                    .bold()
            );

            let results = self.execute_tool_round(&tool_calls).await;
            for (tool_call, result) in tool_calls.iter().zip(results) {
                let mut compacted = compact_tool_result(&tool_call.function.name, &result);
                if let Some(nudge) = record_tool_call_repetition(
                    &mut repeated_tool_calls,
                    &tool_call.function.name,
                    &tool_call.function.arguments,
                ) {
                    compacted.push_str(&nudge);
                }

                self.conversation_history.push(json!({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": compacted,
                }));
            }

            if round == self.max_tool_rounds {
                bail!(
                    "the model exceeded the configured tool-call round limit ({})",
                    self.max_tool_rounds
                );
            }

            println!();
            println!("{}", "Processing results...".bright_blue().bold());
        }

        Ok(())
    }

    async fn execute_tool_round(&mut self, tool_calls: &[ToolCall]) -> Vec<String> {
        let run_concurrently = tool_calls.len() > 1
            && tool_calls
                .iter()
                .all(|call| is_read_only_tool(&call.function.name));

        if run_concurrently {
            for tool_call in tool_calls {
                println!(
                    "{}",
                    format!("-> {} (parallel)", tool_call.function.name).bright_blue()
                );
            }

            let cache = self.file_read_cache.clone();
            let tasks = tool_calls.iter().cloned().map(|tool_call| {
                let cache = cache.clone();
                async move {
                    match tokio::task::spawn_blocking(move || {
                        execute_read_only_tool(&cache, &tool_call)
                    })
                    .await
                    {
                        Ok(Ok(output)) => output,
                        Ok(Err(error)) => format!("Error: {error:#}"),
                        Err(error) => format!("Error: tool task failed: {error}"),
                    }
                }
            });
            return join_all(tasks).await;
        }

        let mut results = Vec::with_capacity(tool_calls.len());
        for tool_call in tool_calls {
            println!("{}", format!("-> {}", tool_call.function.name).bright_blue());
            let result = match self.execute_function_call(tool_call) {
                Ok(output) => output,
                Err(error) => format!("Error: {error:#}"),
            };
            results.push(result);
        }
        results
    }

    fn messages_for_request(&mut self) -> Vec<Value> {
        self.advance_request_floor();

        if self.request_floor <= 1 {
            return self.conversation_history.clone();
        }

        let mut out = Vec::with_capacity(
            self.conversation_history
                .len()
                .saturating_sub(self.request_floor)
                + 2,
        );

        if let Some(system) = self.conversation_history.first() {
            if system.get("role").and_then(Value::as_str) == Some("system") {
                out.push(system.clone());
            }
        }

        out.push(json!({
            "role": "user",
            "content": COMPACTION_MARKER,
        }));

        let start = self.request_floor.min(self.conversation_history.len());
        out.extend(self.conversation_history[start..].iter().cloned());
        out
    }

    fn advance_request_floor(&mut self) {
        const MARKER_TOKENS: u32 = 64;
        if self.request_floor < 1 {
            self.request_floor = 1;
        }

        loop {
            let last_idx = self.conversation_history.len().saturating_sub(1);
            if self.request_floor > last_idx {
                break;
            }

            let mut used = self
                .conversation_history
                .first()
                .filter(|message| message.get("role").and_then(Value::as_str) == Some("system"))
                .map(estimate_message_tokens)
                .unwrap_or(0);

            if self.request_floor > 1 {
                used = used.saturating_add(MARKER_TOKENS);
            }

            for message in &self.conversation_history[self.request_floor..] {
                used = used.saturating_add(estimate_message_tokens(message));
            }

            if used <= self.context_token_budget || self.request_floor >= last_idx {
                break;
            }

            self.request_floor += 1;
            while self.request_floor < last_idx
                && self.conversation_history[self.request_floor]
                    .get("role")
                    .and_then(Value::as_str)
                    == Some("tool")
            {
                self.request_floor += 1;
            }
        }
    }

    async fn stream_completion(&mut self) -> Result<StreamedAssistantMessage> {
        let messages = self.messages_for_request();
        let request_body = json!({
            "model": &self.model,
            "messages": messages,
            "tools": &self.tools,
            "tool_choice": "auto",
            "max_tokens": self.max_tokens,
            "stream": true,
        });

        let mut request = self.client.post(&self.endpoint).json(&request_body);
        if !self.api_key.is_empty() {
            request = request.bearer_auth(&self.api_key);
        }

        let mut spinner = SpinnerGuard::new("nemo");

        let response = match request.send().await {
            Ok(response) => response,
            Err(error) => {
                spinner.finish().await;
                return Err(error).context("failed to reach the local model server");
            }
        };

        let status = response.status();
        if !status.is_success() {
            spinner.finish().await;
            let body = response
                .text()
                .await
                .unwrap_or_else(|_| "<unable to read error response>".to_string());
            bail!("local model server returned HTTP {status}: {body}");
        }

        let mut stream = response.bytes_stream().eventsource();
        let mut output = StreamedAssistantMessage::default();
        let mut reasoning_header_printed = false;
        let mut assistant_header_printed = false;
        let mut first_chunk = true;

        while let Some(event) = stream.next().await {
            let event = match event {
                Ok(event) => event,
                Err(error) => {
                    spinner.finish().await;
                    return Err(error).context("failed while parsing local model SSE stream");
                }
            };
            let data = event.data.trim();

            if data == "[DONE]" {
                break;
            }
            if data.is_empty() {
                continue;
            }

            let chunk: Value = match serde_json::from_str(data) {
                Ok(chunk) => chunk,
                Err(error) => {
                    spinner.finish().await;
                    return Err(error).context(format!("invalid JSON SSE chunk: {data}"));
                }
            };

            let Some(choice) = chunk
                .get("choices")
                .and_then(Value::as_array)
                .and_then(|choices| choices.first())
            else {
                continue;
            };

            if let Some(reason) = choice.get("finish_reason").and_then(Value::as_str) {
                output.finish_reason = Some(reason.to_string());
            }

            let Some(delta) = choice.get("delta") else {
                continue;
            };

            let reasoning = delta
                .get("reasoning_content")
                .and_then(Value::as_str)
                .or_else(|| delta.get("reasoning").and_then(Value::as_str));

            if let Some(reasoning) = reasoning {
                if !reasoning.is_empty() {
                    if first_chunk {
                        spinner.finish().await;
                        first_chunk = false;
                        println!();
                    }
                    if !reasoning_header_printed {
                        println!("{}", "Reasoning:".blue().bold());
                        reasoning_header_printed = true;
                    }
                    print!("{reasoning}");
                    io::stdout().flush().ok();
                    output.reasoning_content.push_str(reasoning);
                }
            }

            if let Some(content) = delta.get("content").and_then(Value::as_str) {
                if !content.is_empty() {
                    if first_chunk {
                        spinner.finish().await;
                        first_chunk = false;
                        println!();
                    }
                    if !assistant_header_printed {
                        if reasoning_header_printed {
                            println!();
                            println!();
                        }
                        print!("{} ", "Assistant>".bright_blue().bold());
                        io::stdout().flush().ok();
                        assistant_header_printed = true;
                    }
                    print!("{content}");
                    io::stdout().flush().ok();
                    output.content.push_str(content);
                }
            }

            if let Some(tool_call_deltas) = delta.get("tool_calls").and_then(Value::as_array) {
                if !tool_call_deltas.is_empty() && first_chunk {
                    spinner.finish().await;
                    first_chunk = false;
                    println!();
                }

                for tool_delta in tool_call_deltas {
                    let index = tool_delta
                        .get("index")
                        .and_then(Value::as_u64)
                        .unwrap_or(0) as usize;

                    while output.tool_calls.len() <= index {
                        output.tool_calls.push(ToolCall {
                            kind: "function".to_string(),
                            ..ToolCall::default()
                        });
                    }

                    let accumulator = &mut output.tool_calls[index];

                    if let Some(id) = tool_delta.get("id").and_then(Value::as_str) {
                        if accumulator.id.is_empty() {
                            accumulator.id = id.to_string();
                        } else if accumulator.id != id {
                            accumulator.id.push_str(id);
                        }
                    }

                    if let Some(kind) = tool_delta.get("type").and_then(Value::as_str) {
                        accumulator.kind = kind.to_string();
                    }

                    if let Some(function) = tool_delta.get("function") {
                        if let Some(name) = function.get("name").and_then(Value::as_str) {
                            accumulator.function.name.push_str(name);
                        }
                        if let Some(arguments) = function.get("arguments").and_then(Value::as_str) {
                            accumulator.function.arguments.push_str(arguments);
                        }
                    }
                }
            }
        }

        if first_chunk {
            spinner.finish().await;
        }

        if reasoning_header_printed || assistant_header_printed {
            println!();
        }

        output
            .tool_calls
            .retain(|call| !call.function.name.trim().is_empty());

        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis();
        for (index, call) in output.tool_calls.iter_mut().enumerate() {
            if call.id.is_empty() {
                call.id = format!("call_{index}_{timestamp}");
            }
            if call.kind.is_empty() {
                call.kind = "function".to_string();
            }
        }

        Ok(output)
    }

    fn execute_function_call(&mut self, tool_call: &ToolCall) -> Result<String> {
        let function_name = tool_call.function.name.as_str();
        let arguments: Value = serde_json::from_str(&tool_call.function.arguments)
            .with_context(|| {
                format!(
                    "tool '{}' returned invalid JSON arguments: {}",
                    function_name, tool_call.function.arguments
                )
            })?;

        match function_name {
            "read_file" | "read_multiple_files" | "list_directory" => {
                execute_read_only_tool(&self.file_read_cache, tool_call)
            }
            "create_file" => {
                let file_path = required_string(&arguments, "file_path")?;
                let content = required_string(&arguments, "content")?;
                let normalized_path = create_file(file_path, content)?;
                self.file_read_cache.invalidate(&normalized_path);
                Ok(format!(
                    "Successfully created file '{}'",
                    normalized_path.display()
                ))
            }
            "create_multiple_files" => {
                let files = arguments
                    .get("files")
                    .and_then(Value::as_array)
                    .ok_or_else(|| anyhow!("missing or invalid 'files' array"))?;

                let mut created_files = Vec::with_capacity(files.len());
                for file_info in files {
                    let path = required_string(file_info, "path")?;
                    let content = required_string(file_info, "content")?;
                    let normalized_path = create_file(path, content)?;
                    self.file_read_cache.invalidate(&normalized_path);
                    created_files.push(normalized_path.display().to_string());
                }

                Ok(format!(
                    "Successfully created {} files: {}",
                    created_files.len(),
                    created_files.join(", ")
                ))
            }
            "edit_file" => {
                let file_path = required_string(&arguments, "file_path")?;
                let original_snippet = required_string(&arguments, "original_snippet")?;
                let new_snippet = required_string(&arguments, "new_snippet")?;

                self.ensure_file_in_context(file_path)?;
                let normalized_path = apply_diff_edit(file_path, original_snippet, new_snippet)?;
                self.file_read_cache.invalidate(&normalized_path);
                Ok(format!(
                    "Successfully edited file '{}'",
                    normalized_path.display()
                ))
            }
            "change_directory" => {
                let path = required_string(&arguments, "path")?;
                let new_cwd =
                    change_working_directory_with_cache(path, Some(&self.file_read_cache))?;
                println!(
                    "{}",
                    format!("cwd -> {}", format_path_for_display(&new_cwd))
                        .bright_cyan()
                        .bold()
                );
                Ok(format!(
                    "Changed working directory to '{}'",
                    format_path_for_display(&new_cwd)
                ))
            }
            "execute_bash_command" => {
                let command = required_string(&arguments, "command")?;
                let description = optional_string(&arguments, "description")?;
                let working_directory = optional_string(&arguments, "working_directory")?;

                if let Some(description) = description {
                    println!(
                        "{}",
                        format!("$ {command}  ({description})").bright_blue()
                    );
                } else {
                    println!("{}", format!("$ {command}").bright_blue());
                }

                // Bash can mutate the tree; drop cached file contents after it runs.
                let output = execute_bash_command(command, working_directory)?;
                self.file_read_cache.clear();
                Ok(output)
            }
            unknown => bail!("unknown function: {unknown}"),
        }
    }

    fn try_handle_add_command(&mut self, user_input: &str) -> Result<bool> {
        const PREFIX: &str = "/add ";
        if !user_input.to_ascii_lowercase().starts_with(PREFIX) {
            return Ok(false);
        }

        let path_to_add = user_input[PREFIX.len()..].trim();
        if path_to_add.is_empty() {
            bail!("usage: /add path/to/file-or-folder");
        }

        let normalized_path = normalize_path(path_to_add)?;
        if normalized_path.is_dir() {
            self.add_directory_to_conversation(&normalized_path)?;
        } else {
            let content = read_local_file(&normalized_path, Some(&self.file_read_cache))?;
            let content = truncate_middle(&content, MAX_TOOL_RESULT_CHARS, "file content");
            self.conversation_history.push(json!({
                "role": "system",
                "content": format!(
                    "Content of file '{}':\n\n{}",
                    normalized_path.display(),
                    content
                ),
            }));
            println!(
                "{}",
                format!(
                    "Added file '{}' to conversation.",
                    normalized_path.display()
                )
                .bright_blue()
                .bold()
            );
            println!();
        }

        Ok(true)
    }

    fn add_directory_to_conversation(&mut self, directory_path: &Path) -> Result<()> {
        println!("{}", "Scanning directory...".bright_blue().bold());

        let mut skipped_files = Vec::new();
        let mut added_files = Vec::new();

        for entry in WalkDir::new(directory_path)
            .follow_links(false)
            .into_iter()
            .filter_entry(should_descend_into)
        {
            if added_files.len() >= MAX_DIRECTORY_FILES {
                println!(
                    "{}",
                    format!("Reached maximum file limit ({MAX_DIRECTORY_FILES})")
                        .yellow()
                        .bold()
                );
                break;
            }

            let entry = match entry {
                Ok(entry) => entry,
                Err(error) => {
                    skipped_files.push(format!("<walk error: {error}>"));
                    continue;
                }
            };

            if !entry.file_type().is_file() {
                continue;
            }

            let path = entry.path();
            let file_name = path
                .file_name()
                .and_then(|value| value.to_str())
                .unwrap_or_default();

            if should_skip_file_name(file_name) {
                skipped_files.push(path.display().to_string());
                continue;
            }

            match fs::metadata(path) {
                Ok(metadata) if metadata.len() > MAX_FILE_SIZE => {
                    skipped_files.push(format!(
                        "{} (exceeds {} byte size limit)",
                        path.display(),
                        MAX_FILE_SIZE
                    ));
                    continue;
                }
                Err(error) => {
                    skipped_files.push(format!("{} ({error})", path.display()));
                    continue;
                }
                _ => {}
            }

            if is_binary_file(path) {
                skipped_files.push(path.display().to_string());
                continue;
            }

            let normalized_path = match normalize_path(&path.to_string_lossy()) {
                Ok(path) => path,
                Err(error) => {
                    skipped_files.push(format!("{} ({error})", path.display()));
                    continue;
                }
            };

            match read_local_file(&normalized_path, Some(&self.file_read_cache)) {
                Ok(content) => {
                    let content = truncate_middle(&content, MAX_FILE_TOOL_RESULT_CHARS, "file content");
                    self.conversation_history.push(json!({
                        "role": "system",
                        "content": format!(
                            "Content of file '{}':\n\n{}",
                            normalized_path.display(),
                            content
                        ),
                    }));
                    added_files.push(normalized_path);
                }
                Err(error) => {
                    skipped_files.push(format!("{} ({error:#})", normalized_path.display()));
                }
            }
        }

        println!(
            "{}",
            format!(
                "Added folder '{}' to conversation.",
                directory_path.display()
            )
            .bright_blue()
            .bold()
        );

        if !added_files.is_empty() {
            println!();
            println!(
                "{}",
                format!("Added files: ({})", added_files.len())
                    .bright_blue()
                    .bold()
            );
            for file in &added_files {
                println!("  {}", format!("{}", file.display()).bright_cyan());
            }
        }

        if !skipped_files.is_empty() {
            println!();
            println!(
                "{}",
                format!("Skipped files: ({})", skipped_files.len())
                    .yellow()
                    .bold()
            );
            for file in skipped_files.iter().take(10) {
                println!("  {}", format!("{file}").yellow().dimmed());
            }
            if skipped_files.len() > 10 {
                println!(
                    "  {}",
                    format!("... and {} more", skipped_files.len() - 10).dimmed()
                );
            }
        }

        println!();
        Ok(())
    }

    fn ensure_file_in_context(&mut self, file_path: &str) -> Result<()> {
        let normalized_path = normalize_path(file_path)?;
        let marker = format!("Content of file '{}'", normalized_path.display());

        let already_present = self.conversation_history.iter().any(|message| {
            message
                .get("content")
                .and_then(Value::as_str)
                .is_some_and(|content| content.contains(&marker))
        });

        if !already_present {
            let content = read_local_file(&normalized_path, Some(&self.file_read_cache))
                .with_context(|| {
                    format!(
                        "could not read '{}' for editing context",
                        normalized_path.display()
                    )
                })?;
            let content = truncate_middle(&content, MAX_FILE_TOOL_RESULT_CHARS, "file content");
            self.conversation_history.push(json!({
                "role": "system",
                "content": format!("{marker}:\n\n{content}"),
            }));
        }

        Ok(())
    }
}

fn execute_read_only_tool(cache: &FileReadCache, tool_call: &ToolCall) -> Result<String> {
    let function_name = tool_call.function.name.as_str();
    let arguments: Value = serde_json::from_str(&tool_call.function.arguments).with_context(|| {
        format!(
            "tool '{}' returned invalid JSON arguments: {}",
            function_name, tool_call.function.arguments
        )
    })?;

    match function_name {
        "read_file" => {
            let file_path = required_string(&arguments, "file_path")?;
            let normalized_path = normalize_path(file_path)?;
            let content = read_local_file(&normalized_path, Some(cache))?;
            Ok(format!(
                "Content of file '{}':\n\n{}",
                normalized_path.display(),
                content
            ))
        }
        "read_multiple_files" => {
            let file_paths = required_string_array(&arguments, "file_paths")?;
            let mut results = Vec::with_capacity(file_paths.len());

            for file_path in file_paths {
                match normalize_path(&file_path)
                    .and_then(|path| read_local_file(&path, Some(cache)).map(|content| (path, content)))
                {
                    Ok((path, content)) => results.push(format!(
                        "Content of file '{}':\n\n{}",
                        path.display(),
                        content
                    )),
                    Err(error) => results.push(format!("Error reading '{file_path}': {error:#}")),
                }
            }

            Ok(results.join("\n\n==================================================\n\n"))
        }
        "list_directory" => {
            let path = optional_string(&arguments, "path")?.unwrap_or(".");
            list_directory(path)
        }
        other => bail!("'{other}' is not a read-only tool"),
    }
}

fn truncate_middle(content: &str, max_chars: usize, label: &str) -> String {
    let total_chars = content.chars().count();
    if total_chars <= max_chars {
        return content.to_string();
    }

    let half = max_chars / 2;
    let head_end = content
        .char_indices()
        .nth(half)
        .map(|(idx, _)| idx)
        .unwrap_or(content.len());
    let tail_start = if half == 0 {
        content.len()
    } else {
        content
            .char_indices()
            .rev()
            .nth(half - 1)
            .map(|(idx, _)| idx)
            .unwrap_or(0)
    };

    format!(
        "{}\n\n[{label} truncated: showing head and tail]\n\n{}",
        &content[..head_end],
        &content[tail_start..]
    )
}

fn compact_tool_result(tool_name: &str, result: &str) -> String {
    let limit = match tool_name {
        "read_file" | "read_multiple_files" | "list_directory" => MAX_FILE_TOOL_RESULT_CHARS,
        _ => MAX_TOOL_RESULT_CHARS,
    };
    truncate_middle(result, limit, "tool result")
}

fn estimate_message_tokens(message: &Value) -> u32 {
    let mut chars = 0usize;

    if let Some(content) = message.get("content") {
        match content {
            Value::String(text) => chars += text.len(),
            Value::Null => {}
            other => chars += other.to_string().len(),
        }
    }

    if let Some(tool_calls) = message.get("tool_calls").and_then(Value::as_array) {
        for call in tool_calls {
            if let Some(id) = call.get("id").and_then(Value::as_str) {
                chars += id.len();
            }
            if let Some(function) = call.get("function") {
                if let Some(name) = function.get("name").and_then(Value::as_str) {
                    chars += name.len();
                }
                if let Some(arguments) = function.get("arguments").and_then(Value::as_str) {
                    chars += arguments.len();
                }
            }
        }
    }

    if let Some(id) = message.get("tool_call_id").and_then(Value::as_str) {
        chars += id.len();
    }

    if let Some(reasoning) = message.get("reasoning_content").and_then(Value::as_str) {
        chars += reasoning.len();
    }

    chars.div_ceil(4).min(u32::MAX as usize) as u32
}

struct SpinnerGuard {
    stop: Arc<AtomicBool>,
    handle: Option<tokio::task::JoinHandle<()>>,
}

impl SpinnerGuard {
    fn new(label: impl Into<String>) -> Self {
        let label = label.into();
        let stop = Arc::new(AtomicBool::new(false));
        let stop_clone = stop.clone();
        let handle = tokio::spawn(async move {
            let frames = ['|', '/', '-', '\\'];
            let mut index = 0usize;
            while !stop_clone.load(Ordering::Relaxed) {
                eprint!(
                    "\r\x1b[2K{} {} {}",
                    label.dimmed(),
                    "·".dimmed(),
                    frames[index % frames.len()].cyan()
                );
                let _ = io::stderr().flush();
                index = index.wrapping_add(1);
                tokio::time::sleep(Duration::from_millis(100)).await;
            }
            eprint!("\r\x1b[2K");
            let _ = io::stderr().flush();
        });

        Self {
            stop,
            handle: Some(handle),
        }
    }

    async fn finish(&mut self) {
        if let Some(handle) = self.handle.take() {
            self.stop.store(true, Ordering::Relaxed);
            let _ = handle.await;
        }
    }
}

impl Drop for SpinnerGuard {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(handle) = self.handle.take() {
            handle.abort();
            eprint!("\r\x1b[2K");
            let _ = io::stderr().flush();
        }
    }
}

fn tool_definitions() -> Value {
    json!([
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read the content of a single file from the filesystem",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "The path to the file to read (relative or absolute)"
                        }
                    },
                    "required": ["file_path"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "read_multiple_files",
                "description": "Read the content of multiple files from the filesystem",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_paths": {
                            "type": "array",
                            "items": { "type": "string" },
                            "description": "Array of file paths to read (relative or absolute)"
                        }
                    },
                    "required": ["file_paths"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "create_file",
                "description": "Create a new file or overwrite an existing file with the provided content",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "The path where the file should be created"
                        },
                        "content": {
                            "type": "string",
                            "description": "The content to write to the file"
                        }
                    },
                    "required": ["file_path", "content"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "create_multiple_files",
                "description": "Create multiple files at once",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "files": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "path": { "type": "string" },
                                    "content": { "type": "string" }
                                },
                                "required": ["path", "content"]
                            },
                            "description": "Array of files to create with their paths and content"
                        }
                    },
                    "required": ["files"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "edit_file",
                "description": "Edit an existing file by replacing a specific snippet with new content",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "The path to the file to edit"
                        },
                        "original_snippet": {
                            "type": "string",
                            "description": "The exact text snippet to find and replace"
                        },
                        "new_snippet": {
                            "type": "string",
                            "description": "The new text to replace the original snippet with"
                        }
                    },
                    "required": ["file_path", "original_snippet", "new_snippet"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "list_directory",
                "description": "List files and subdirectories in a directory. Use this to explore the filesystem before reading or editing files.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Directory to list (relative or absolute). Defaults to the current working directory."
                        }
                    }
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "change_directory",
                "description": "Persistently change NemoCode's process working directory. Later relative paths and bash commands will use this directory.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Directory to change into (relative, absolute, ~, or ~/...)"
                        }
                    },
                    "required": ["path"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "execute_bash_command",
                "description": "Execute a bash command in the terminal. Use for ls, find, git, cargo, builds, and other shell operations. Prefer list_directory/change_directory for simple navigation.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "The bash command to execute. May include && or ; for simple chains."
                        },
                        "description": {
                            "type": "string",
                            "description": "Short human-readable description of what this command does"
                        },
                        "working_directory": {
                            "type": "string",
                            "description": "Optional directory to run the command in. Defaults to the current working directory. Does not permanently change cwd; use change_directory for that."
                        }
                    },
                    "required": ["command"]
                }
            }
        }
    ])
}

fn required_string<'a>(value: &'a Value, key: &str) -> Result<&'a str> {
    value
        .get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| anyhow!("missing or invalid string field '{key}'"))
}

fn optional_string<'a>(value: &'a Value, key: &str) -> Result<Option<&'a str>> {
    match value.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(text)) => Ok(Some(text.as_str())),
        Some(_) => bail!("field '{key}' must be a string when provided"),
    }
}

fn required_string_array(value: &Value, key: &str) -> Result<Vec<String>> {
    let items = value
        .get(key)
        .and_then(Value::as_array)
        .ok_or_else(|| anyhow!("missing or invalid array field '{key}'"))?;

    items
        .iter()
        .enumerate()
        .map(|(index, item)| {
            item.as_str()
                .map(ToOwned::to_owned)
                .ok_or_else(|| anyhow!("'{key}[{index}]' must be a string"))
        })
        .collect()
}

fn require_nemocode_launch_directory() -> Result<PathBuf> {
    let cwd = env::current_dir().context("failed to determine current directory")?;
    let cwd = canonicalize_existing_or_self(cwd);

    let cargo_toml = cwd.join("Cargo.toml");
    let start_script = cwd.join("start-nemo.sh");
    if !cargo_toml.is_file() || !start_script.is_file() {
        bail!(
            "NemoCode must be launched from the nemocode project directory.\n\
             Current directory: {}\n\
             Expected files: Cargo.toml and start-nemo.sh\n\
             cd into the nemocode repo and run ./start-nemo.sh",
            cwd.display()
        );
    }

    let cargo = fs::read_to_string(&cargo_toml)
        .with_context(|| format!("failed to read '{}'", cargo_toml.display()))?;
    if !cargo.lines().any(|line| line.trim() == "name = \"nemocode\"") {
        bail!(
            "NemoCode must be launched from the nemocode project directory.\n\
             '{}' does not declare package name \"nemocode\".",
            cargo_toml.display()
        );
    }

    Ok(cwd)
}

fn home_dir() -> Option<PathBuf> {
    env::var_os("HOME")
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
}

fn canonicalize_existing_or_self(path: PathBuf) -> PathBuf {
    path.canonicalize().unwrap_or(path)
}

fn format_path_for_display(path: &Path) -> String {
    if let Some(home) = home_dir() {
        path.strip_prefix(&home)
            .map(|rel| format!("~/{}", rel.display()))
            .unwrap_or_else(|_| path.display().to_string())
    } else {
        path.display().to_string()
    }
}

fn session_location_block(project_root: &Path) -> String {
    let cwd = env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    format!(
        "SESSION LOCATION:\n\
         - Working directory: {}\n\
         - Project root: {}\n\n\
         Relative file paths resolve against the working directory. \
         After /cd, !cd, or ! shell mode, the working directory changes for this session.",
        format_path_for_display(&cwd),
        format_path_for_display(project_root)
    )
}

fn change_working_directory_with_cache(
    path: &str,
    cache: Option<&FileReadCache>,
) -> Result<PathBuf> {
    let trimmed = path.trim();
    let target = if trimmed.is_empty() || trimmed == "~" {
        home_dir().ok_or_else(|| anyhow!("could not resolve home directory"))?
    } else if let Some(stripped) = trimmed.strip_prefix("~/") {
        home_dir()
            .ok_or_else(|| anyhow!("could not resolve home directory"))?
            .join(stripped)
    } else {
        let raw = PathBuf::from(trimmed);
        if raw.is_absolute() {
            raw
        } else {
            env::current_dir()
                .context("failed to determine current directory")?
                .join(raw)
        }
    };

    let target = canonicalize_existing_or_self(target);
    if !target.is_dir() {
        bail!("not a directory: {}", target.display());
    }

    env::set_current_dir(&target).with_context(|| {
        format!("failed to change directory to '{}'", target.display())
    })?;

    if let Some(cache) = cache {
        cache.clear();
    }

    Ok(env::current_dir().unwrap_or(target))
}

fn display_cwd() {
    match env::current_dir() {
        Ok(cwd) => println!(
            "{}",
            format!("cwd: {}", format_path_for_display(&cwd))
                .bright_cyan()
                .bold()
        ),
        Err(error) => eprintln!("{}", format!("Error: {error}").red().bold()),
    }
}

fn cwd_prompt() -> String {
    let name = env::current_dir()
        .ok()
        .and_then(|cwd| {
            cwd.file_name()
                .map(|part| part.to_string_lossy().into_owned())
        })
        .unwrap_or_else(|| "?".to_string());
    format!("You [{name}]> ")
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum EditorMode {
    /// Main NemoCode prompt: complete paths after /cd, /add, !cd, and !commands.
    Repl,
    /// Interactive shell mode: complete filesystem paths anywhere on the line.
    Shell,
}

struct NemoHelper {
    files: FilenameCompleter,
    mode: EditorMode,
}

impl NemoHelper {
    fn new(mode: EditorMode) -> Self {
        Self {
            files: FilenameCompleter::new(),
            mode,
        }
    }
}

impl Helper for NemoHelper {}
impl Validator for NemoHelper {}
impl Highlighter for NemoHelper {}

impl Hinter for NemoHelper {
    type Hint = String;
}

impl Completer for NemoHelper {
    type Candidate = Pair;

    fn complete(
        &self,
        line: &str,
        pos: usize,
        ctx: &RlContext<'_>,
    ) -> rustyline::Result<(usize, Vec<Pair>)> {
        match self.mode {
            EditorMode::Shell => self.files.complete(line, pos, ctx),
            EditorMode::Repl => complete_repl_line(&self.files, line, pos, ctx),
        }
    }
}

fn complete_repl_line(
    files: &FilenameCompleter,
    line: &str,
    pos: usize,
    ctx: &RlContext<'_>,
) -> rustyline::Result<(usize, Vec<Pair>)> {
    let before = &line[..pos.min(line.len())];
    let lowered = before.to_ascii_lowercase();

    if let Some((path_start, directories_only)) = repl_path_completion_span(&lowered, before) {
        let path_part = &before[path_start..];
        let (rel_start, candidates) = files.complete(path_part, path_part.len(), ctx)?;
        let candidates = if directories_only {
            candidates
                .into_iter()
                .filter(|pair| pair.replacement.ends_with('/') || pair.display.ends_with('/'))
                .collect()
        } else {
            candidates
        };
        return Ok((path_start + rel_start, candidates));
    }

    if before.starts_with('!') {
        return files.complete(line, pos, ctx);
    }

    Ok((pos, Vec::new()))
}

/// Returns `(byte index where the path token starts, directories_only)`.
fn repl_path_completion_span(lowered: &str, _before: &str) -> Option<(usize, bool)> {
    const PREFIXES: &[(&str, bool)] = &[
        ("/cd ", true),
        ("!cd ", true),
        ("/add ", false),
    ];

    for (prefix, directories_only) in PREFIXES {
        if lowered.starts_with(prefix) {
            return Some((prefix.len(), *directories_only));
        }
    }

    None
}

fn new_line_editor(mode: EditorMode) -> Result<Editor<NemoHelper, DefaultHistory>> {
    let config = Config::builder()
        .completion_type(CompletionType::List)
        .build();
    let mut editor =
        Editor::with_config(config).context("failed to initialize terminal editor")?;
    editor.set_helper(Some(NemoHelper::new(mode)));
    Ok(editor)
}

fn parse_bang_cd_command(cmd: &str) -> Option<&str> {
    let cmd = cmd.trim();
    if cmd == "cd" {
        return Some("");
    }
    cmd.strip_prefix("cd ").map(str::trim)
}

fn truncate_shell_output(output: &str) -> String {
    if output.len() <= MAX_SHELL_OUTPUT_BYTES {
        return output.to_string();
    }

    let half = MAX_SHELL_OUTPUT_BYTES / 2;
    let head = &output[..half];
    let tail = &output[output.len().saturating_sub(half)..];
    format!(
        "{head}\n\n[Shell output truncated: {} bytes omitted]\n\n{tail}",
        output.len().saturating_sub(head.len() + tail.len())
    )
}

fn execute_bash_command(command: &str, working_directory: Option<&str>) -> Result<String> {
    let cwd = if let Some(dir) = working_directory {
        normalize_path(dir)?
    } else {
        env::current_dir().unwrap_or_else(|_| PathBuf::from("."))
    };

    if !cwd.is_dir() {
        bail!("working directory is not a directory: {}", cwd.display());
    }

    let mut child = Command::new("bash")
        .arg("-c")
        .arg(command)
        .current_dir(&cwd)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .with_context(|| format!("failed to spawn bash for: {command}"))?;

    let started = Instant::now();
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break status,
            Ok(None) if started.elapsed() > Duration::from_secs(SHELL_COMMAND_TIMEOUT_SECS) => {
                let _ = child.kill();
                let _ = child.wait();
                bail!("command timed out after {SHELL_COMMAND_TIMEOUT_SECS}s");
            }
            Ok(None) => thread::sleep(Duration::from_millis(50)),
            Err(error) => return Err(error).context("failed while waiting for bash command"),
        }
    };

    let mut stdout = String::new();
    let mut stderr = String::new();
    if let Some(mut reader) = child.stdout.take() {
        reader
            .read_to_string(&mut stdout)
            .context("failed to read bash stdout")?;
    }
    if let Some(mut reader) = child.stderr.take() {
        reader
            .read_to_string(&mut stderr)
            .context("failed to read bash stderr")?;
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

    Ok(truncate_shell_output(&text))
}

fn list_directory(path: &str) -> Result<String> {
    let dir = normalize_path(path)?;
    if !dir.is_dir() {
        bail!("'{}' is not a directory", dir.display());
    }

    let mut entries = fs::read_dir(&dir)
        .with_context(|| format!("could not read directory '{}'", dir.display()))?
        .collect::<std::io::Result<Vec<_>>>()
        .with_context(|| format!("could not list directory '{}'", dir.display()))?;

    entries.sort_by_key(|entry| {
        let is_dir = entry.file_type().map(|kind| kind.is_dir()).unwrap_or(false);
        (!is_dir, entry.file_name())
    });

    let total = entries.len();
    let mut lines = Vec::new();
    lines.push(format!(
        "Directory: {} ({} entries)",
        format_path_for_display(&dir),
        total
    ));

    for entry in entries.into_iter().take(MAX_LIST_DIRECTORY_ENTRIES) {
        let name = entry.file_name().to_string_lossy().into_owned();
        let file_type = entry.file_type().ok();
        let suffix = if file_type.as_ref().is_some_and(|kind| kind.is_dir()) {
            "/"
        } else if file_type.as_ref().is_some_and(|kind| kind.is_symlink()) {
            "@"
        } else {
            ""
        };
        let meta = entry.metadata().ok();
        let size = meta
            .as_ref()
            .filter(|metadata| metadata.is_file())
            .map(|metadata| format!("  {}B", metadata.len()))
            .unwrap_or_default();
        lines.push(format!("  {name}{suffix}{size}"));
    }

    if total > MAX_LIST_DIRECTORY_ENTRIES {
        lines.push(format!(
            "  ... and {} more entries",
            total - MAX_LIST_DIRECTORY_ENTRIES
        ));
    }

    Ok(lines.join("\n"))
}

fn enter_shell_mode(cache: &FileReadCache) -> Result<()> {
    println!(
        "{}",
        "Entering shell mode. Type exit or quit to return to NemoCode."
            .bright_blue()
            .bold()
    );
    display_cwd();
    println!();

    let mut editor = new_line_editor(EditorMode::Shell)?;

    loop {
        let dir_name = env::current_dir()
            .ok()
            .and_then(|cwd| {
                cwd.file_name()
                    .map(|part| part.to_string_lossy().into_owned())
            })
            .unwrap_or_else(|| "?".to_string());
        let prompt = format!("shell {dir_name}> ");

        let input = match editor.readline(&prompt) {
            Ok(line) => line,
            Err(ReadlineError::Interrupted | ReadlineError::Eof) => break,
            Err(error) => return Err(error).context("shell input failed"),
        };

        let input = input.trim();
        if input.is_empty() {
            continue;
        }
        let _ = editor.add_history_entry(input);

        if matches!(input, "exit" | "quit") {
            break;
        }

        if input == "cd" || input.starts_with("cd ") {
            let path = input.strip_prefix("cd").unwrap_or("").trim();
            match change_working_directory_with_cache(path, Some(cache)) {
                Ok(new_cwd) => println!(
                    "{}",
                    format!("Changed to: {}", format_path_for_display(&new_cwd)).dimmed()
                ),
                Err(error) => eprintln!("{}", format!("Error: {error:#}").red().bold()),
            }
            continue;
        }

        if input == "pwd" {
            display_cwd();
            continue;
        }

        if input == "clear" {
            print!("\x1B[2J\x1B[1;1H");
            io::stdout().flush().ok();
            continue;
        }

        match execute_bash_command(input, None) {
            Ok(output) => {
                cache.clear();
                print!("{output}");
                if !output.ends_with('\n') {
                    println!();
                }
            }
            Err(error) => eprintln!("{}", format!("Error: {error:#}").red().bold()),
        }
    }

    println!("{}", "Returned to NemoCode.".bright_blue().bold());
    display_cwd();
    println!();
    Ok(())
}

/// Resolve a user/model path. Relative paths use the live process cwd.
fn normalize_path(path_str: &str) -> Result<PathBuf> {
    let trimmed = path_str.trim();
    if trimmed.is_empty() {
        bail!("path cannot be empty");
    }

    let absolute = if trimmed == "~" {
        home_dir().ok_or_else(|| anyhow!("could not resolve home directory"))?
    } else if let Some(stripped) = trimmed.strip_prefix("~/") {
        home_dir()
            .ok_or_else(|| anyhow!("could not resolve home directory"))?
            .join(stripped)
    } else {
        let raw = Path::new(trimmed);
        if raw.is_absolute() {
            raw.to_path_buf()
        } else {
            env::current_dir()
                .context("failed to determine current directory")?
                .join(raw)
        }
    };

    let mut normalized = PathBuf::new();
    for component in absolute.components() {
        match component {
            Component::Prefix(prefix) => normalized.push(prefix.as_os_str()),
            Component::RootDir => normalized.push(Path::new(std::path::MAIN_SEPARATOR_STR)),
            Component::CurDir => {}
            Component::ParentDir => {
                if !normalized.pop() {
                    bail!("invalid path '{path_str}': escapes past filesystem root");
                }
            }
            Component::Normal(part) => normalized.push(part),
        }
    }

    Ok(canonicalize_existing_or_self(normalized))
}

fn read_local_file(file_path: &Path, cache: Option<&FileReadCache>) -> Result<String> {
    let metadata = fs::metadata(file_path)
        .with_context(|| format!("could not stat '{}'", file_path.display()))?;
    if !metadata.is_file() {
        bail!("'{}' is not a regular file", file_path.display());
    }
    if metadata.len() > MAX_FILE_SIZE {
        bail!(
            "file '{}' exceeds the {} byte read limit",
            file_path.display(),
            MAX_FILE_SIZE
        );
    }

    if let Some(cache) = cache {
        return cache.get_or_load(file_path);
    }

    fs::read_to_string(file_path)
        .with_context(|| format!("could not read '{}' as UTF-8 text", file_path.display()))
}

fn create_file(path: &str, content: &str) -> Result<PathBuf> {
    if content.len() as u64 > MAX_FILE_SIZE {
        bail!("file content exceeds the {} byte size limit", MAX_FILE_SIZE);
    }

    let normalized_path = normalize_path(path)?;
    let parent = normalized_path
        .parent()
        .ok_or_else(|| anyhow!("path '{}' has no parent directory", normalized_path.display()))?;

    fs::create_dir_all(parent)
        .with_context(|| format!("could not create directory '{}'", parent.display()))?;
    fs::write(&normalized_path, content)
        .with_context(|| format!("could not write '{}'", normalized_path.display()))?;

    println!(
        "{}",
        format!("Created/updated file at '{}'", normalized_path.display())
            .bright_blue()
            .bold()
    );

    Ok(normalized_path)
}

fn apply_diff_edit(path: &str, original_snippet: &str, new_snippet: &str) -> Result<PathBuf> {
    if original_snippet.is_empty() {
        bail!("original_snippet cannot be empty");
    }

    let normalized_path = normalize_path(path)?;
    let content = read_local_file(&normalized_path, None)?;
    let occurrences = content.match_indices(original_snippet).count();

    match occurrences {
        0 => {
            eprintln!();
            eprintln!("{}", "Expected snippet:".bright_blue().bold());
            eprintln!("{original_snippet}");
            eprintln!();
            eprintln!("{}", "Actual file content:".yellow().bold());
            eprintln!("{content}");
            bail!(
                "original snippet not found in '{}'",
                normalized_path.display()
            );
        }
        1 => {}
        count => {
            bail!(
                "ambiguous edit: found {count} matches in '{}'; provide a more specific snippet",
                normalized_path.display()
            )
        }
    }

    let updated_content = content.replacen(original_snippet, new_snippet, 1);
    create_file(&normalized_path.to_string_lossy(), &updated_content)?;
    println!(
        "{}",
        format!("Applied diff edit to '{}'", normalized_path.display())
            .bright_blue()
            .bold()
    );

    Ok(normalized_path)
}

fn is_binary_file(file_path: &Path) -> bool {
    let mut file = match fs::File::open(file_path) {
        Ok(file) => file,
        Err(_) => return true,
    };

    let mut buffer = [0_u8; 1_024];
    let bytes_read = match file.read(&mut buffer) {
        Ok(bytes_read) => bytes_read,
        Err(_) => return true,
    };

    buffer[..bytes_read].contains(&0)
}

fn should_descend_into(entry: &DirEntry) -> bool {
    if entry.depth() == 0 {
        return true;
    }

    if !entry.file_type().is_dir() {
        return true;
    }

    let name = entry.file_name().to_string_lossy();
    !name.starts_with('.') && !excluded_directory_names().contains(name.as_ref())
}

fn should_skip_file_name(file_name: &str) -> bool {
    if file_name.starts_with('.') || excluded_file_names().contains(file_name) {
        return true;
    }

    let lower = file_name.to_ascii_lowercase();
    excluded_suffixes()
        .iter()
        .any(|suffix| lower.ends_with(*suffix))
}

fn excluded_directory_names() -> HashSet<&'static str> {
    [
        ".git",
        ".svn",
        ".hg",
        "CVS",
        ".uv",
        "uvenv",
        ".uvenv",
        ".venv",
        "venv",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        "node_modules",
        ".next",
        ".nuxt",
        "dist",
        "build",
        ".cache",
        ".parcel-cache",
        ".turbo",
        ".vercel",
        ".output",
        ".contentlayer",
        "out",
        "coverage",
        ".nyc_output",
        "storybook-static",
        ".idea",
        ".vscode",
        "target",
        "models",
        ".llama",
    ]
    .into_iter()
    .collect()
}

fn excluded_file_names() -> HashSet<&'static str> {
    [
        ".DS_Store",
        "Thumbs.db",
        ".gitignore",
        ".python-version",
        "uv.lock",
        "package-lock.json",
        "yarn.lock",
        "pnpm-lock.yaml",
        ".env",
        ".env.local",
        ".env.development",
        ".env.production",
        ".coverage",
    ]
    .into_iter()
    .collect()
}

fn excluded_suffixes() -> &'static [&'static str] {
    &[
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".ico",
        ".svg",
        ".webp",
        ".avif",
        ".mp4",
        ".webm",
        ".mov",
        ".mp3",
        ".wav",
        ".ogg",
        ".zip",
        ".tar",
        ".gz",
        ".7z",
        ".rar",
        ".exe",
        ".dll",
        ".so",
        ".dylib",
        ".bin",
        ".pdf",
        ".doc",
        ".docx",
        ".xls",
        ".xlsx",
        ".ppt",
        ".pptx",
        ".pyc",
        ".pyo",
        ".pyd",
        ".egg",
        ".whl",
        ".db",
        ".sqlite",
        ".sqlite3",
        ".log",
        ".map",
        ".chunk.js",
        ".chunk.css",
        ".min.js",
        ".min.css",
        ".bundle.js",
        ".bundle.css",
        ".tmp",
        ".temp",
        ".ttf",
        ".otf",
        ".woff",
        ".woff2",
        ".eot",
        ".gguf",
    ]
}

fn print_welcome(project_root: &Path) {
    println!();
    println!("{}", BANNER.bright_cyan().bold());
    println!();
    println!(
        "{}",
        "Local coding agent  |  Nemotron-3-Nano-4B-Coding-Agent"
            .bright_blue()
            .bold()
    );
    println!(
        "{}",
        "Streaming replies, file tools, multi-step tool loops".blue()
    );
    println!();
    println!("{}", "Filesystem navigation".bright_blue().bold());
    println!("  {}", "/pwd                 - Show working directory".bright_cyan());
    println!("  {}", "/cd path             - Change working directory".bright_cyan());
    println!("  {}", "!                    - Interactive bash shell".bright_cyan());
    println!("  {}", "!cd path             - Change directory (one shot)".bright_cyan());
    println!("  {}", "!ls -la              - Run one bash command".bright_cyan());
    println!("  {}", "Tab                  - Autocomplete filesystem paths".bright_cyan());
    println!("  The model can also list dirs, cd, and run bash via tools.");
    println!();
    println!("{}", "File operations".bright_blue().bold());
    println!("  {}", "/add path/to/file   - Include one file".bright_cyan());
    println!("  {}", "/add path/to/folder - Include a source tree".bright_cyan());
    println!("  The model can read, create, edit, list, and navigate files.");
    println!();
    println!("{}", "Commands".bright_blue().bold());
    println!("  {}", "exit or quit - End the session".bright_cyan());
    println!("  Ask naturally; tool calls are handled automatically.");
    println!();
    println!(
        "{}",
        format!("Project root: {}", format_path_for_display(project_root)).blue().dimmed()
    );
    display_cwd();
    println!();
}

fn handle_navigation_command(user_input: &str, cache: &FileReadCache) -> Result<bool> {
    let lowered = user_input.to_ascii_lowercase();
    match lowered.as_str() {
        "/pwd" | "pwd" => {
            display_cwd();
            return Ok(true);
        }
        "!" => {
            enter_shell_mode(cache)?;
            return Ok(true);
        }
        _ => {}
    }

    if let Some(path) = user_input
        .strip_prefix("/cd ")
        .or_else(|| user_input.strip_prefix("/CD "))
    {
        let new_cwd = change_working_directory_with_cache(path.trim(), Some(cache))?;
        println!(
            "{}",
            format!("Changed to: {}", format_path_for_display(&new_cwd))
                .bright_blue()
                .bold()
        );
        display_cwd();
        return Ok(true);
    }

    if lowered == "/cd" {
        bail!("usage: /cd path/to/directory");
    }

    if let Some(cmd) = user_input.strip_prefix('!') {
        let cmd = cmd.trim();
        if let Some(path) = parse_bang_cd_command(cmd) {
            let new_cwd = change_working_directory_with_cache(path, Some(cache))?;
            println!(
                "{}",
                format!("Changed to: {}", format_path_for_display(&new_cwd)).dimmed()
            );
            display_cwd();
            return Ok(true);
        }

        let output = execute_bash_command(cmd, None)?;
        cache.clear();
        print!("{output}");
        if !output.ends_with('\n') {
            println!();
        }
        return Ok(true);
    }

    Ok(false)
}

#[tokio::main]
async fn main() -> Result<()> {
    let mut agent = match NemoCode::from_env() {
        Ok(agent) => agent,
        Err(error) => {
            eprintln!("{}", format!("Error: {error:#}").red().bold());
            std::process::exit(1);
        }
    };

    print_welcome(&agent.project_root);
    println!(
        "{}",
        format!("Model: {}", agent.model).blue().dimmed()
    );
    println!(
        "{}",
        format!("Endpoint: {}", agent.endpoint).blue().dimmed()
    );
    println!();

    let mut editor = new_line_editor(EditorMode::Repl)?;

    loop {
        let user_input = match editor.readline(&cwd_prompt()) {
            Ok(line) => line.trim().to_string(),
            Err(ReadlineError::Interrupted | ReadlineError::Eof) => {
                println!();
                println!("{}", "Exiting...".yellow().bold());
                break;
            }
            Err(error) => return Err(error).context("terminal input failed"),
        };

        if user_input.is_empty() {
            continue;
        }

        let _ = editor.add_history_entry(user_input.as_str());

        if matches!(user_input.to_ascii_lowercase().as_str(), "exit" | "quit") {
            println!("{}", "Goodbye.".bright_blue().bold());
            break;
        }

        match handle_navigation_command(&user_input, &agent.file_read_cache) {
            Ok(true) => continue,
            Ok(false) => {}
            Err(error) => {
                eprintln!("{}", format!("Error: {error:#}").red().bold());
                continue;
            }
        }

        match agent.try_handle_add_command(&user_input) {
            Ok(true) => continue,
            Ok(false) => {}
            Err(error) => {
                eprintln!("{}", format!("Error: {error:#}").red().bold());
                continue;
            }
        }

        if let Err(error) = agent.handle_user_message(user_input).await {
            eprintln!();
            eprintln!("{}", format!("Error: {error:#}").red().bold());
        }
    }

    println!("{}", "Session finished.".blue().bold());
    Ok(())
}
