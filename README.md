# NemoCode

Fast, lightweight local coding agent.

NemoCode is a Rust CLI harness with streaming chat, multi-step tool use, and
filesystem tools. Out of the box it uses one model only:

[S4MPL3BI4S/Nemotron-3-Nano-4B-Coding-Agent-GGUF](https://huggingface.co/S4MPL3BI4S/Nemotron-3-Nano-4B-Coding-Agent-GGUF)

```
┳┓┏┓┳┳┓┏┓┏┓┏┓┳┓┏┓
┃┃┣ ┃┃┃┃┃┃ ┃┃┃┃┣ 
┛┗┗┛┛ ┗┗┛┗┛┗┛┻┛┗┛
```

![NemoCode interactive session](docs/assets/session.png)

## What it does

- Local-only inference through an OpenAI-compatible `llama-server`
- Streaming assistant replies
- Tool loop for file read / create / edit
- `/add` to inject a file or folder into context

No cloud API keys. No alternate model providers. The bundled GGUF is the model.

## Requirements

- Rust toolchain (`cargo`)
- `curl` and `tar`
- About 3 GB disk for the Q4_K_M GGUF, plus RAM/VRAM for inference

`llama-server` is installed automatically on first launch from the official
[llama.cpp](https://github.com/ggml-org/llama.cpp) GitHub releases into
`.vendor/llama.cpp/`. If `llama-server` is already on your `PATH`, that binary
is used instead.

Optional download helpers: `hf`, `huggingface-cli`, or `wget`.

## Quick start

```bash
chmod +x start-nemo.sh
./start-nemo.sh
```

The startup script will:

1. Print the NemoCode banner
2. Install `llama-server` automatically if it is not already available
3. Download `Nemotron-3-Nano-4B-Coding-Agent-Q4_K_M.gguf` into `models/` if missing
4. Start `llama-server` with `--jinja` (required for tool calling)
5. Build and launch the NemoCode CLI against `http://127.0.0.1:8080/v1`

![First-run model download](docs/assets/startup-download.png)

![Local model server ready](docs/assets/startup-ready.png)

## Usage

```text
You> /add src/main.rs
You> Explain the tool loop and suggest a small cleanup
You> exit
```

Commands:

| Input | Effect |
| --- | --- |
| `/add path/to/file` | Add one file to conversation context |
| `/add path/to/folder` | Add a source tree (skips junk/binary/large files) |
| `exit` or `quit` | End the session |

Natural-language requests can trigger tools automatically:

- `read_file`
- `read_multiple_files`
- `create_file`
- `create_multiple_files`
- `edit_file`

## Configuration

Copy `.env.example` to `.env` if you want persistent overrides.

| Variable | Default | Meaning |
| --- | --- | --- |
| `NEMO_BASE_URL` | `http://127.0.0.1:8080/v1` | Local OpenAI-compatible base URL |
| `NEMO_MODEL` | `Nemotron-3-Nano-4B-Coding-Agent-Q4_K_M` | Model id sent to the server |
| `NEMO_API_KEY` | `local` | Optional; unused unless the server enforces auth |
| `NEMO_MAX_TOKENS` | `8192` | Max completion tokens |
| `NEMO_TOOL_ROUNDS` | `8` | Max tool-call rounds per user turn |

Launcher overrides for `./start-nemo.sh`:

| Variable | Default | Meaning |
| --- | --- | --- |
| `NEMO_MODEL_PATH` | `models/Nemotron-3-Nano-4B-Coding-Agent-Q4_K_M.gguf` | Local GGUF path |
| `NEMO_HOST` / `NEMO_PORT` | `127.0.0.1` / `8080` | Server bind address |
| `NEMO_CTX` | `16384` | Context size |
| `NEMO_GPU_LAYERS` | `99` | GPU offload layers (`0` for CPU) |
| `NEMO_THREADS` | unset | Optional CPU thread count |
| `NEMO_LLAMA_SERVER` | auto-detect / auto-install | Path to `llama-server` |
| `NEMO_LLAMA_BACKEND` | auto | Force `cpu`, `vulkan`, or `rocm` for auto-install |
| `NEMO_LLAMA_RELEASE` | latest | Pin a llama.cpp release tag (for example `b10043`) |
| `NEMO_VENDOR_DIR` | `.vendor/llama.cpp` | Where auto-installed binaries are stored |

Example CPU-only launch:

```bash
NEMO_GPU_LAYERS=0 NEMO_CTX=8192 ./start-nemo.sh
```

## Manual run

If the server is already running:

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
  docs/assets/          # README screenshots
  models/               # downloaded GGUF (gitignored)
  .vendor/llama.cpp/    # auto-installed llama-server (gitignored)
```

## License

MIT. See [LICENSE](LICENSE).
