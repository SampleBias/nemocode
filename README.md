# NemoCode

Python-first local coding agent.

NemoCode is a Rust CLI harness optimized for Python development: streaming chat,
pytest/ruff helpers, diagnostics via basedpyright/pyright (with local fallbacks),
bash/filesystem navigation, and Tab path autocomplete. Other languages still work
through file tools and bash. Out of the box it uses one model only:

[S4MPL3BI4S/Nemotron-3-Nano-4B-Coding-Agent-GGUF](https://huggingface.co/S4MPL3BI4S/Nemotron-3-Nano-4B-Coding-Agent-GGUF)

```
┳┓┏┓┳┳┓┏┓┏┓┏┓┳┓┏┓
┃┃┣ ┃┃┃┃┃┃ ┃┃┃┃┣ 
┛┗┗┛┛ ┗┗┛┗┛┗┛┻┛┗┛
```

![NemoCode Python-first welcome screen](docs/assets/session.png)

## What it does

- Local-only inference through an OpenAI-compatible `llama-server`
- Python-first system prompt and `PYTHON PROJECT` context when a Python tree is detected
- First-class tools: `python_diagnostics`, `run_pytest`, `run_python`, `ruff_check`
- Diagnostics-only LSP via `basedpyright-langserver` / `pyright-langserver` when installed
  (falls back to `ruff check` or `python -m compileall`; no network needed once installed)
- Streaming replies with a TTFB spinner until the first token
- Multi-step tool loop for files, directories, and bash (other languages via bash/files)
- Vybrid-style filesystem navigation (`/cd`, `/pwd`, `!`, `!cd`, `!command`)
- Tab autocomplete for filesystem paths
- **Ctrl+I** to interrupt the agent mid-task and steer it without restarting the session
- Must launch from the nemocode project directory

No cloud API keys. No alternate model providers. The bundled GGUF is the model.

## Requirements

- Rust toolchain (`cargo`)
- `curl` and `tar`
- About 3 GB disk for the Q4_K_M GGUF, plus RAM/VRAM for inference
- Optional for best Python diagnostics: `basedpyright` or `pyright` on `PATH`
- Optional: `pytest`, `ruff`

`llama-server` is installed automatically on first launch from the official
[llama.cpp](https://github.com/ggml-org/llama.cpp) GitHub releases into
`.vendor/llama.cpp/`. If `llama-server` is already on your `PATH`, that binary
is used instead.

Optional download helpers: `hf`, `huggingface-cli`, or `wget`.

## Quick start

```bash
git clone https://github.com/SampleBias/nemocode.git
cd nemocode
chmod +x start-nemo.sh
./start-nemo.sh
```

The startup script will:

1. Print the NemoCode banner
2. Verify it is running from the nemocode project root
3. Install `llama-server` automatically if it is not already available
4. Download `Nemotron-3-Nano-4B-Coding-Agent-Q4_K_M.gguf` into `models/` if missing
5. Start `llama-server` with `--jinja` (required for tool calling)
6. Build and launch the NemoCode CLI against `http://127.0.0.1:8080/v1`

![First-run model download](docs/assets/startup-download.png)

![Local model server ready and NemoCode launching](docs/assets/startup-ready.png)

## Launch rule

NemoCode must be started from the nemocode project directory (the folder that
contains `Cargo.toml` named `nemocode` and `start-nemo.sh`).

- `./start-nemo.sh` always `cd`s to that directory first
- `cargo run` from another directory is rejected

## Usage

```text
You [nemocode]> /cd ~/my_flask_app
Python project detected (pyproject.toml)
You [my_flask_app]> Add a /health endpoint and a pytest for it
Nemo> (reads files → edits → python_diagnostics → run_pytest)
You [my_flask_app]> /add app/
You [my_flask_app]> Refactor the routes module and keep tests green
You [my_flask_app]> exit
```

Other languages still work via file tools and bash when you ask; Python gets first-class tools and prompts.

### Commands

| Input | Effect |
| --- | --- |
| `/pwd` | Show the current working directory |
| `/cd path` | Change the process working directory |
| `!` | Enter interactive bash shell mode |
| `!cd path` | Change directory without entering shell mode |
| `!command` | Run one bash command in the current directory |
| `Tab` | Autocomplete filesystem paths |
| `/add path/to/file` | Add one file to conversation context |
| `/add path/to/folder` | Add a source tree (skips junk/binary/large files) |
| `Ctrl+I` | Interrupt the agent mid-task and send guidance (see below) |
| `exit` or `quit` | End the session |

### Interrupting the agent (Ctrl+I)

While the agent is generating a reply or running tools, press **Ctrl+I** to stop it
and steer the current task without ending the session.

What happens:

1. Streaming stops immediately (including in-progress tool rounds).
2. Any incomplete tool round is rolled back so the conversation stays consistent.
3. You get a short prompt to enter guidance. Press **Enter** on an empty line to
   resume with no extra notes, or type a correction and press **Enter**.
4. The agent continues the same user turn with your guidance treated as higher
   priority than its previous plan.

You can use Ctrl+I during model output, while tools are running, or while a bash
command is executing. Long-running bash commands are killed when interrupted.

Example:

```text
You [nemocode]> Build a small Flask app in test_app/
Nemo> (generating…)
^I
── Interrupted (Ctrl+I) ──
Enter guidance for the agent. Empty line = continue without notes.
You [nemocode]> Use plain HTML templates, no Jinja macros
Guidance recorded — resuming.
Nemo> (continues with your correction)
```

### Tab autocomplete

Press `Tab` to complete paths:

- `/cd te` → `test/` (directories preferred for `/cd` and `!cd`)
- `/add sr` → files and folders under the current path
- `!cd`, `!ls`, and other `!` commands also complete path arguments
- Inside `!` shell mode, Tab completes filesystem paths anywhere on the line

### Filesystem navigation

Relative file-tool paths resolve against the live working directory. A
`SESSION LOCATION` block (cwd + project root) is injected on user turns. When the
workspace looks like a Python project (`pyproject.toml`, `requirements.txt`,
top-level `.py` files, etc.), a short `PYTHON PROJECT` block is added too.

The model can navigate via tools:

- `list_directory`
- `change_directory`
- `execute_bash_command`

Plus file tools:

- `read_file` / `read_multiple_files`
- `create_file` / `create_multiple_files`
- `edit_file`

### Python tooling

Prefer these over free-form bash for Python work:

| Tool | Effect |
| --- | --- |
| `python_diagnostics` | LSP diagnostics when available; else `ruff` / `compileall` |
| `run_pytest` | Run pytest (optional target / extra args) |
| `run_python` | Run a script, `-m` module, or `-c` snippet (uses `.venv` Python when present) |
| `ruff_check` | Run `ruff check` when ruff is installed |

Install a language server locally if you want LSP diagnostics (example):

```bash
pip install basedpyright
# provides basedpyright-langserver
```

Set `NEMO_PYTHON_LSP=off` to skip LSP entirely, or override the command with
`NEMO_PYTHON_LSP_COMMAND`. The LSP protocol itself is local-only; internet is only
needed to install the server or fetch Python packages.

## Performance

NemoCode includes several local-inference speedups:

| Feature | What it does |
| --- | --- |
| Default `max_tokens` 4096 | Smaller completion budget for faster tool turns |
| Sticky history budget | Keeps a stable prompt prefix; default ~12k tokens (`NEMO_CONTEXT_BUDGET`) |
| Tool-result compaction | Middle-truncates large tool outputs (12KB file/list, 48KB other) |
| TTFB spinner | Shows activity until the first streamed chunk |
| Tool-call stream feedback | Single-line args spinner (name × count · size) while tool JSON streams |
| Tool-call stream guards | Early-stops runaway tool streams (>8 calls or >24KB args) and dedupes |
| SSE idle timeout | Aborts if no SSE chunk for 5 minutes (`NEMO_SSE_IDLE_TIMEOUT_SECS`; `0` = forever) |
| Bash / parallel progress | Elapsed spinner for bash; spinner + done line for parallel reads |
| SESSION LOCATION + PYTHON PROJECT | Injects cwd/root (and Python markers when detected) on user turns |
| Parallel read-only tools | Runs multiple reads/lists in one round concurrently |
| Identical-loop nudge | After 3 identical read-only calls in a turn, nudges the model not to repeat |
| File-read cache | Caches by path + mtime + size; cleared on edit / `cd` / bash |

## Configuration

Copy `.env.example` to `.env` if you want persistent overrides.

| Variable | Default | Meaning |
| --- | --- | --- |
| `NEMO_BASE_URL` | `http://127.0.0.1:8080/v1` | Local OpenAI-compatible base URL |
| `NEMO_MODEL` | `Nemotron-3-Nano-4B-Coding-Agent-Q4_K_M` | Model id sent to the server |
| `NEMO_API_KEY` | `local` | Optional; unused unless the server enforces auth |
| `NEMO_MAX_TOKENS` | `4096` | Max completion tokens |
| `NEMO_MAX_CONTINUATIONS` | `16` | Auto-resume when output hits the token limit |
| `NEMO_TOOL_ROUNDS` | `8` | Max tool-call rounds per user turn |
| `NEMO_CONTEXT_BUDGET` | `NEMO_CTX - 512` | Max prompt tokens (messages + tool schemas); compaction keeps under this |
| `NEMO_SSE_IDLE_TIMEOUT_SECS` | `300` (5 min) | Abort if the SSE stream stalls; `0` waits forever |
| `NEMO_PYTHON_LSP` | `auto` | `auto` / `off` / path-or-command for the Python language server |
| `NEMO_PYTHON_LSP_COMMAND` | unset | Optional full command (for example `basedpyright-langserver --stdio`) |

Launcher overrides for `./start-nemo.sh`:

| Variable | Default | Meaning |
| --- | --- | --- |
| `NEMO_MODEL_PATH` | `models/Nemotron-3-Nano-4B-Coding-Agent-Q4_K_M.gguf` | Local GGUF path |
| `NEMO_HOST` / `NEMO_PORT` | `127.0.0.1` / `8080` | Server bind address |
| `NEMO_CTX` | `16384` | Context size |
| `NEMO_GPU_LAYERS` | `auto` | GPU offload layers (`0` for CPU; `auto` / `fit` for low-VRAM) |
| `NEMO_THREADS` | unset | Optional CPU thread count |
| `NEMO_PARALLEL` | `1` | Server slots (1 = full context for single-agent use) |
| `NEMO_FLASH_ATTN` | `on` | Flash Attention (`on` / `off` / `auto`) |
| `NEMO_BATCH` | `2048` | Logical batch size for prefill |
| `NEMO_UBATCH` | `512` | Physical micro-batch size |
| `NEMO_CACHE_TYPE_K` / `NEMO_CACHE_TYPE_V` | `q8_0` | KV cache element types |
| `NEMO_MLOCK` | unset | Set `1` to pin the model in RAM (`--mlock`) |
| `NEMO_LLAMA_SERVER` | auto-detect / auto-install | Path to `llama-server` |
| `NEMO_LLAMA_BACKEND` | auto | Force `cpu`, `vulkan`, or `rocm` for auto-install |
| `NEMO_LLAMA_RELEASE` | latest | Pin a llama.cpp release tag (for example `b10043`) |
| `NEMO_VENDOR_DIR` | `.vendor/llama.cpp` | Where auto-installed binaries are stored |

Example CPU-only launch:

```bash
NEMO_GPU_LAYERS=0 NEMO_CTX=8192 ./start-nemo.sh
```

## Manual run

If the server is already running from the nemocode directory:

```bash
export NEMO_BASE_URL=http://127.0.0.1:8080/v1
export NEMO_MODEL=Nemotron-3-Nano-4B-Coding-Agent-Q4_K_M
cargo run --release
```

Example `llama-server` command:

```bash
llama-server \
  --model models/Nemotron-3-Nano-4B-Coding-Agent-Q4_K_M.gguf \
  --host 127.0.0.1 \
  --port 8080 \
  --ctx-size 16384 \
  --n-gpu-layers 99 \
  --parallel 1 \
  --flash-attn on \
  --batch-size 2048 \
  --ubatch-size 512 \
  --cache-type-k q8_0 \
  --cache-type-v q8_0 \
  --jinja
```

## Model

- Repo: [S4MPL3BI4S/Nemotron-3-Nano-4B-Coding-Agent-GGUF](https://huggingface.co/S4MPL3BI4S/Nemotron-3-Nano-4B-Coding-Agent-GGUF)
- File: `Nemotron-3-Nano-4B-Coding-Agent-Q4_K_M.gguf`
- Quant: Q4_K_M (~2.8 GB)
- Base: Unsloth / NVIDIA Nemotron 3 Nano 4B
- Fine-tune focus: coding and pythonic function calling

Model weights are governed by the
[NVIDIA Nemotron Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-nemotron-open-model-license/).
The NemoCode harness itself is MIT-licensed.

## Project layout

```text
nemocode/
  Cargo.toml
  LICENSE
  README.md
  start-nemo.sh
  .env.example
  src/main.rs
  src/python_project.rs # Python tree detection / PYTHON PROJECT block
  src/python_tools.rs   # pytest / run_python / ruff / compileall
  src/lsp/              # diagnostics-only Python LSP client
  docs/assets/          # README screenshots
  models/               # downloaded GGUF (gitignored)
  .vendor/llama.cpp/    # auto-installed llama-server (gitignored)
```

## License

MIT. See [LICENSE](LICENSE).
