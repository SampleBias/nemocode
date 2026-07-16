//! NemoCode — local coding-agent harness
//!
//! Runs against a local OpenAI-compatible llama.cpp server serving the bundled
//! Nemotron-3-Nano-4B-Coding-Agent GGUF. This is the only model the harness uses.
//!
//! Environment variables:
//!   NEMO_BASE_URL      optional, default: http://127.0.0.1:8080/v1
//!   NEMO_MODEL         optional, default: Nemotron-3-Nano-4B-Coding-Agent-Q4_K_M
//!   NEMO_API_KEY       optional, default: local (llama-server ignores unless configured)
//!   NEMO_MAX_TOKENS    optional, default: 8192
//!   NEMO_TOOL_ROUNDS   optional, default: 8

use anyhow::{anyhow, bail, Context, Result};
use dotenvy::dotenv;
use eventsource_stream::Eventsource;
use futures_util::StreamExt;
use owo_colors::OwoColorize;
use reqwest::Client;
use rustyline::{error::ReadlineError, DefaultEditor};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{
    collections::HashSet,
    env,
    fs,
    io::{self, Read, Write},
    path::{Component, Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};
use walkdir::{DirEntry, WalkDir};

const MAX_FILE_SIZE: u64 = 5_000_000;
const MAX_DIRECTORY_FILES: usize = 1_000;
const HISTORY_TRIM_THRESHOLD: usize = 20;
const HISTORY_NON_SYSTEM_KEEP: usize = 15;

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

Guidelines:
1. Provide natural, conversational responses explaining your reasoning
2. Use function calls when you need to read or modify files
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

struct NemoCode {
    client: Client,
    api_key: String,
    endpoint: String,
    model: String,
    max_tokens: u64,
    max_tool_rounds: usize,
    tools: Value,
    conversation_history: Vec<Value>,
}

impl NemoCode {
    fn from_env() -> Result<Self> {
        dotenv().ok();

        let base_url = env::var("NEMO_BASE_URL").unwrap_or_else(|_| DEFAULT_BASE_URL.to_string());
        let endpoint = format!("{}/chat/completions", base_url.trim_end_matches('/'));

        let model = env::var("NEMO_MODEL").unwrap_or_else(|_| DEFAULT_MODEL.to_string());
        let api_key = env::var("NEMO_API_KEY").unwrap_or_else(|_| "local".to_string());

        let max_tokens = env::var("NEMO_MAX_TOKENS")
            .ok()
            .map(|value| value.parse::<u64>())
            .transpose()
            .context("NEMO_MAX_TOKENS must be a positive integer")?
            .unwrap_or(8_192);

        let max_tool_rounds = env::var("NEMO_TOOL_ROUNDS")
            .ok()
            .map(|value| value.parse::<usize>())
            .transpose()
            .context("NEMO_TOOL_ROUNDS must be a positive integer")?
            .unwrap_or(8);

        if max_tool_rounds == 0 {
            bail!("NEMO_TOOL_ROUNDS must be greater than zero");
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
            max_tokens,
            max_tool_rounds,
            tools: tool_definitions(),
            conversation_history: vec![json!({
                "role": "system",
                "content": SYSTEM_PROMPT,
            })],
        })
    }

    async fn handle_user_message(&mut self, user_message: String) -> Result<()> {
        self.conversation_history.push(json!({
            "role": "user",
            "content": user_message,
        }));
        self.trim_conversation_history();

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

            for tool_call in tool_calls {
                println!("{}", format!("-> {}", tool_call.function.name).bright_blue());
                let result = match self.execute_function_call(&tool_call) {
                    Ok(output) => output,
                    Err(error) => format!("Error: {error:#}"),
                };

                self.conversation_history.push(json!({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result,
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

    async fn stream_completion(&self) -> Result<StreamedAssistantMessage> {
        let request_body = json!({
            "model": &self.model,
            "messages": &self.conversation_history,
            "tools": &self.tools,
            "tool_choice": "auto",
            "max_tokens": self.max_tokens,
            "stream": true,
        });

        let mut request = self.client.post(&self.endpoint).json(&request_body);
        if !self.api_key.is_empty() {
            request = request.bearer_auth(&self.api_key);
        }

        let response = request
            .send()
            .await
            .context("failed to reach the local model server")?;

        let status = response.status();
        if !status.is_success() {
            let body = response
                .text()
                .await
                .unwrap_or_else(|_| "<unable to read error response>".to_string());
            bail!("local model server returned HTTP {status}: {body}");
        }

        println!();
        println!("{}", "Thinking...".bright_blue().bold());

        let mut stream = response.bytes_stream().eventsource();
        let mut output = StreamedAssistantMessage::default();
        let mut reasoning_header_printed = false;
        let mut assistant_header_printed = false;

        while let Some(event) = stream.next().await {
            let event = event.context("failed while parsing local model SSE stream")?;
            let data = event.data.trim();

            if data == "[DONE]" {
                break;
            }
            if data.is_empty() {
                continue;
            }

            let chunk: Value = serde_json::from_str(data)
                .with_context(|| format!("invalid JSON SSE chunk: {data}"))?;

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
                    if !reasoning_header_printed {
                        println!();
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
                    if !assistant_header_printed {
                        if reasoning_header_printed {
                            println!();
                            println!();
                        } else {
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
            "read_file" => {
                let file_path = required_string(&arguments, "file_path")?;
                let normalized_path = normalize_path(file_path)?;
                let content = read_local_file(&normalized_path)?;
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
                        .and_then(|path| read_local_file(&path).map(|content| (path, content)))
                    {
                        Ok((path, content)) => results.push(format!(
                            "Content of file '{}':\n\n{}",
                            path.display(),
                            content
                        )),
                        Err(error) => {
                            results.push(format!("Error reading '{file_path}': {error:#}"))
                        }
                    }
                }

                Ok(results.join("\n\n==================================================\n\n"))
            }
            "create_file" => {
                let file_path = required_string(&arguments, "file_path")?;
                let content = required_string(&arguments, "content")?;
                let normalized_path = create_file(file_path, content)?;
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
                Ok(format!(
                    "Successfully edited file '{}'",
                    normalized_path.display()
                ))
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
            let content = read_local_file(&normalized_path)?;
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

            match read_local_file(&normalized_path) {
                Ok(content) => {
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
            let content = read_local_file(&normalized_path).with_context(|| {
                format!(
                    "could not read '{}' for editing context",
                    normalized_path.display()
                )
            })?;
            self.conversation_history.push(json!({
                "role": "system",
                "content": format!("{marker}:\n\n{content}"),
            }));
        }

        Ok(())
    }

    fn trim_conversation_history(&mut self) {
        if self.conversation_history.len() <= HISTORY_TRIM_THRESHOLD {
            return;
        }

        let mut system_messages = Vec::new();
        let mut non_system_messages = Vec::new();

        for message in self.conversation_history.drain(..) {
            if message.get("role").and_then(Value::as_str) == Some("system") {
                system_messages.push(message);
            } else {
                non_system_messages.push(message);
            }
        }

        let keep_from = non_system_messages
            .len()
            .saturating_sub(HISTORY_NON_SYSTEM_KEEP);
        let mut retained: Vec<Value> = non_system_messages.into_iter().skip(keep_from).collect();

        // Never retain a tail that begins in the middle of an assistant/tool
        // sequence. The newest user message is always present because trimming
        // occurs immediately after it is appended.
        if let Some(first_user) = retained.iter().position(|message| {
            message.get("role").and_then(Value::as_str) == Some("user")
        }) {
            retained.drain(..first_user);
        }

        system_messages.extend(retained);
        self.conversation_history = system_messages;
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
        }
    ])
}

fn required_string<'a>(value: &'a Value, key: &str) -> Result<&'a str> {
    value
        .get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| anyhow!("missing or invalid string field '{key}'"))
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

fn normalize_path(path_str: &str) -> Result<PathBuf> {
    let raw = Path::new(path_str);

    for component in raw.components() {
        match component {
            Component::ParentDir => {
                bail!("invalid path '{path_str}': parent-directory references are not allowed")
            }
            Component::Normal(part) if part.to_string_lossy().starts_with('~') => {
                bail!("invalid path '{path_str}': home-directory references are not allowed")
            }
            _ => {}
        }
    }

    let absolute = if raw.is_absolute() {
        raw.to_path_buf()
    } else {
        env::current_dir()
            .context("failed to determine current directory")?
            .join(raw)
    };

    let mut normalized = PathBuf::new();
    for component in absolute.components() {
        match component {
            Component::Prefix(prefix) => normalized.push(prefix.as_os_str()),
            Component::RootDir => normalized.push(Path::new(std::path::MAIN_SEPARATOR_STR)),
            Component::CurDir => {}
            Component::ParentDir => {
                bail!("invalid path '{path_str}': parent-directory references are not allowed")
            }
            Component::Normal(part) => normalized.push(part),
        }
    }

    Ok(normalized)
}

fn read_local_file(file_path: &Path) -> Result<String> {
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
    let content = read_local_file(&normalized_path)?;
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

fn print_welcome() {
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
    println!("{}", "File operations".bright_blue().bold());
    println!("  {}", "/add path/to/file   - Include one file".bright_cyan());
    println!("  {}", "/add path/to/folder - Include a source tree".bright_cyan());
    println!("  The model can read, create, and precisely edit files.");
    println!();
    println!("{}", "Commands".bright_blue().bold());
    println!("  {}", "exit or quit - End the session".bright_cyan());
    println!("  Ask naturally; tool calls are handled automatically.");
    println!();
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

    print_welcome();
    println!(
        "{}",
        format!("Model: {}", agent.model).blue().dimmed()
    );
    println!(
        "{}",
        format!("Endpoint: {}", agent.endpoint).blue().dimmed()
    );
    println!();

    let mut editor = DefaultEditor::new().context("failed to initialize terminal editor")?;

    loop {
        let user_input = match editor.readline("You> ") {
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
