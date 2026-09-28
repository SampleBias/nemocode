//! CLI configuration and read-only runtime inspection/measurement.
use anyhow::{Context, Result, bail};
use serde_json::{Value, json};
use std::{
    env,
    path::PathBuf,
    process::Command,
    time::{Duration, Instant},
};

#[derive(Default)]
pub struct Cli {
    pub command: String,
    pub prompt: Option<String>,
    pub resume: Option<PathBuf>,
    pub arguments: Vec<String>,
}

pub fn env_u64(name: &str, default: u64) -> Result<u64> {
    let value = env::var(name)
        .ok()
        .map(|v| v.parse::<u64>())
        .transpose()
        .with_context(|| format!("{name} must be an integer"))?
        .unwrap_or(default);
    if value == 0 {
        bail!("{name} must be positive");
    }
    Ok(value)
}

pub fn chat_template_kwargs() -> Result<Value> {
    let enabled = match env::var("NEMO_THINKING").as_deref().unwrap_or("off") {
        "on" | "true" | "1" => true,
        "off" | "false" | "0" => false,
        _ => bail!("NEMO_THINKING must be on or off"),
    };
    Ok(json!({"enable_thinking": enabled}))
}

/// Called before constructing Tokio or starting any worker threads.
pub fn configure() -> Result<Cli> {
    let args: Vec<String> = env::args().skip(1).collect();
    let option = |name: &str| {
        args.iter()
            .position(|s| s == name)
            .and_then(|i| args.get(i + 1))
            .cloned()
    };
    if let Some(workspace) = option("--workspace").or_else(|| env::var("NEMO_WORKSPACE").ok()) {
        env::set_current_dir(workspace).context("select workspace")?;
    }
    let config = option("--config")
        .or_else(|| env::var("NEMO_CONFIG").ok())
        .map(PathBuf::from)
        .unwrap_or(env::current_dir()?.join(".env"));
    if config.is_file() {
        for (key, value) in parse_config(&std::fs::read_to_string(&config)?)? {
            if env::var_os(&key).is_none() {
                unsafe {
                    env::set_var(key, value);
                }
            }
        }
    } else if option("--config").is_some() {
        bail!("configuration file not found: {}", config.display());
    }
    let mut cli = Cli {
        command: "chat".into(),
        ..Cli::default()
    };
    let mut i = 0;
    let mut positional = Vec::new();
    while i < args.len() {
        let arg = &args[i];
        match arg.as_str() {
            "--workspace" | "--config" | "--permission" | "--session-dir" | "--resume"
            | "--prompt" | "--max-tokens" => {
                i += 1;
                let value = args
                    .get(i)
                    .with_context(|| format!("{arg} needs a value"))?;
                match arg.as_str() {
                    "--permission" => {
                        if !matches!(value.as_str(), "read-only" | "workspace" | "trusted") {
                            bail!("permission must be read-only, workspace, or trusted");
                        }
                        unsafe {
                            env::set_var("NEMO_PERMISSION", value);
                        }
                    }
                    "--session-dir" => unsafe {
                        env::set_var("NEMO_SESSION_DIR", value);
                    },
                    "--resume" => {
                        cli.resume = Some(PathBuf::from(value));
                        unsafe {
                            env::set_var("NEMO_SESSION_DIR", value);
                            env::set_var("NEMO_RESUME", "1");
                        }
                    }
                    "--prompt" => cli.prompt = Some(value.clone()),
                    "--max-tokens" => unsafe {
                        env::set_var("NEMO_MAX_TOKENS", value);
                    },
                    _ => {}
                }
            }
            "--json" => {}
            "--help" | "-h" => cli.command = "help".into(),
            "--version" | "-V" => cli.command = "version".into(),
            option if option.starts_with('-') => bail!("unknown option {option}"),
            _ => positional.push(arg.clone()),
        }
        i += 1;
    }
    if cli.command == "chat" && !positional.is_empty() {
        cli.command = positional.remove(0);
        match cli.command.as_str() {
            "run" => {
                if cli.prompt.is_none() && !positional.is_empty() {
                    cli.prompt = Some(positional.join(" "));
                }
            }
            "chat" | "doctor" | "status" | "bench" | "config" | "runtime" | "model" => {}
            other => bail!("unknown command {other}; use --help"),
        }
        cli.arguments = positional;
    }
    if cli.command == "run" {
        if cli.prompt.is_none() && cli.resume.is_none() {
            bail!("run requires --prompt or a positional prompt");
        }
        unsafe {
            env::set_var("NEMO_HEADLESS", "1");
            env::set_var("NO_COLOR", "1");
        }
    }
    Ok(cli)
}

pub fn parse_config(text: &str) -> Result<Vec<(String, String)>> {
    let mut values = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        let line = line.strip_prefix("export ").unwrap_or(line);
        if !line.starts_with("NEMO_") {
            continue;
        }
        let (key, raw) = line
            .split_once('=')
            .context("NEMO config entries must be KEY=literal-value")?;
        if !key
            .bytes()
            .all(|c| c.is_ascii_uppercase() || c.is_ascii_digit() || c == b'_')
        {
            bail!("invalid config key {key}");
        }
        let value = if raw.starts_with(['\'', '"']) {
            if raw.len() < 2 || raw.as_bytes().last() != raw.as_bytes().first() {
                bail!("unclosed config quote for {key}");
            }
            raw[1..raw.len() - 1].to_string()
        } else {
            raw.split(" #").next().unwrap_or("").trim_end().to_string()
        };
        values.push((key.to_string(), value));
    }
    Ok(values)
}

pub fn base_url() -> String {
    env::var("NEMO_BASE_URL")
        .unwrap_or_else(|_| crate::DEFAULT_BASE_URL.into())
        .trim_end_matches('/')
        .to_string()
}
pub fn native_url() -> String {
    base_url().trim_end_matches("/v1").to_string()
}
pub fn request(
    client: &reqwest::Client,
    method: reqwest::Method,
    url: &str,
) -> reqwest::RequestBuilder {
    let mut request = client.request(method, url);
    if let Ok(key) = env::var("NEMO_API_KEY") {
        request = request.bearer_auth(key);
    }
    request
}

pub async fn management(cli: &Cli) -> Result<bool> {
    match cli.command.as_str() {
        "chat" | "run" => return Ok(false),
        "help" => {
            std::println!(
                "NemoCode\n\n  nemocode [chat] --workspace PATH\n  nemocode run --workspace PATH --prompt TEXT [--permission read-only|workspace|trusted]\n  nemocode --resume SESSION_DIR [--prompt TEXT] run\n  nemocode doctor | status | config | bench\n  nemocode runtime install|start|update\n  nemocode model install|verify\n\nOptions: --config FILE --session-dir PATH --max-tokens N --json\nConfiguration precedence: CLI > environment > selected .env > defaults.\nWorkspace permission constrains native file paths; project checkers execute project code.\nUse read-only for untrusted repositories. Trusted mode enables arbitrary shell and Python."
            );
        }
        "version" => std::println!("nemocode {}", env!("CARGO_PKG_VERSION")),
        "config" => std::println!("{}", effective_config()),
        "status" => {
            let client = reqwest::Client::builder()
                .timeout(Duration::from_secs(3))
                .build()?;
            let response = request(
                &client,
                reqwest::Method::GET,
                &format!("{}/health", native_url()),
            )
            .send()
            .await;
            let ready = response.as_ref().is_ok_and(|r| r.status().is_success());
            std::println!(
                "{}",
                json!({"ready":ready,"endpoint":base_url(),"config":effective_config()})
            );
            if !ready {
                bail!("runtime is not ready");
            }
        }
        "doctor" => {
            let mut checks = serde_json::Map::new();
            for (name, args) in [
                ("rg", vec!["--version"]),
                ("git", vec!["--version"]),
                ("python3", vec!["--version"]),
            ] {
                let mut cmd = Command::new(name);
                cmd.args(args);
                checks.insert(
                    name.into(),
                    json!(
                        crate::process::run(cmd, None, Duration::from_secs(5))
                            .map(|o| o.success())
                            .unwrap_or(false)
                    ),
                );
            }
            let binary = env::var("NEMO_LLAMA_SERVER").unwrap_or_else(|_| "llama-server".into());
            let mut cmd = Command::new(&binary);
            cmd.arg("--list-devices");
            let devices = crate::process::run(cmd, None, Duration::from_secs(10))
                .map(|o| json!(o))
                .unwrap_or_else(|e| json!({"error":e.to_string()}));
            let mem = std::fs::read_to_string("/proc/meminfo")
                .unwrap_or_default()
                .lines()
                .filter(|s| s.starts_with("MemAvailable:") || s.starts_with("MemTotal:"))
                .collect::<Vec<_>>()
                .join("\n");
            std::println!(
                "{}",
                json!({"config":effective_config(),"checks":checks,"runtime_binary":binary,"devices":devices,"memory":mem,"cpu_threads":std::thread::available_parallelism().map(|n|n.get()).unwrap_or(1)})
            );
        }
        "bench" => benchmark().await?,
        "runtime" | "model" => {
            let action = cli
                .arguments
                .first()
                .map(String::as_str)
                .unwrap_or("status");
            if !matches!(
                (cli.command.as_str(), action),
                ("runtime", "install" | "start" | "update") | ("model", "install" | "verify")
            ) {
                bail!("unsupported management action; see --help");
            }
            let launcher = launcher()?;
            let status = Command::new("bash")
                .arg(launcher)
                .arg(format!("--{}-{action}", cli.command))
                .status()?;
            if !status.success() {
                bail!("management action failed: {status}");
            }
        }
        _ => bail!("unknown command"),
    }
    Ok(true)
}

fn launcher() -> Result<PathBuf> {
    if let Some(path) = env::var_os("NEMO_LAUNCHER") {
        return Ok(path.into());
    }
    let exe = env::current_exe()?;
    let packaged = exe
        .parent()
        .context("executable parent")?
        .join("../libexec/nemocode/start-nemo.sh");
    if packaged.is_file() {
        return Ok(packaged);
    }
    let local = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("start-nemo.sh");
    if local.is_file() {
        return Ok(local);
    }
    bail!("runtime launcher not installed; set NEMO_LAUNCHER or install the release bundle")
}

pub fn effective_config() -> Value {
    let mut values = serde_json::Map::new();
    for (name, default) in [
        ("NEMO_BASE_URL", crate::DEFAULT_BASE_URL),
        ("NEMO_MODEL", crate::DEFAULT_MODEL),
        ("NEMO_CTX", "16384"),
        ("NEMO_MAX_TOKENS", "4096"),
        ("NEMO_TASK_TOKENS", "16384"),
        ("NEMO_TASK_TIMEOUT_SECS", "900"),
        ("NEMO_PERMISSION", "workspace"),
        ("NEMO_GPU_LAYERS", "auto"),
        ("NEMO_UBATCH", "512"),
        ("NEMO_TEMPERATURE", "0.6"),
        ("NEMO_TOP_P", "0.95"),
        ("NEMO_SEED", "-1"),
        ("NEMO_THINKING", "off"),
        ("NEMO_THREADS", "runtime default"),
        ("NEMO_BATCH", "2048"),
        ("NEMO_PARALLEL", "1"),
        ("NEMO_CACHE_TYPE_K", "q8_0"),
        ("NEMO_CACHE_TYPE_V", "q8_0"),
        ("NEMO_FLASH_ATTN", "on"),
        ("NEMO_TOOL_ROUNDS", "32"),
        (
            "NEMO_CONTEXT_BUDGET",
            "derived from context and output reserve",
        ),
    ] {
        values.insert(
            name.into(),
            json!(env::var(name).unwrap_or_else(|_| default.into())),
        );
    }
    values.insert("workspace".into(), json!(env::current_dir().ok()));
    json!(values)
}

async fn benchmark() -> Result<()> {
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(180))
        .build()?;
    let mut measurements = Vec::new();
    for (name, prompt) in [
        (
            "cold_or_existing",
            "Write a Python function that adds two integers. Return code only.",
        ),
        (
            "repeat_prefix",
            "Write a Python function that adds two integers. Return code only.",
        ),
        (
            "reasoning",
            "Find the bug: def average(xs): return sum(xs) / len(xs). Explain the empty-list case briefly.",
        ),
    ] {
        let start = Instant::now();
        let response = request(&client,reqwest::Method::POST,&format!("{}/chat/completions",base_url())).json(&json!({"model":env::var("NEMO_MODEL").unwrap_or_else(|_|crate::DEFAULT_MODEL.into()),"messages":[{"role":"user","content":prompt}],"max_tokens":128,"temperature":0.0,"seed":42,"stream":false,"chat_template_kwargs":chat_template_kwargs()?})).send().await?;
        let status = response.status();
        let body: Value = response.json().await?;
        if !status.is_success() {
            bail!("benchmark request rejected: {body}");
        }
        measurements.push(json!({"case":name,"wall_ms":start.elapsed().as_millis(),"usage":body["usage"],"timings":body["timings"]}));
    }
    std::println!(
        "{}",
        json!({"config":effective_config(),"measurements":measurements,"note":"Latency probe, not a coding-quality score. First request may already be warm; RSS/energy require device measurement."})
    );
    Ok(())
}
