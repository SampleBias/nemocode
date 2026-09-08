#!/usr/bin/env bash
# NemoCode launcher: install llama-server if needed, download the bundled GGUF,
# start the local model server, then run the coding-agent harness.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="${NEMO_WORKSPACE:-$PWD}"
export NEMO_WORKSPACE="$WORKSPACE"

# Configuration is data, not shell code. Environment overrides file values.
load_config() {
  local config="${NEMO_CONFIG:-$WORKSPACE/.env}" line key value
  [[ -f "$config" ]] || return 0
  while IFS= read -r line || [[ -n "$line" ]]; do
    line="${line%$'\r'}"
    line="${line#"${line%%[![:space:]]*}"}"
    line="${line%"${line##*[![:space:]]}"}"
    [[ "$line" =~ ^[[:space:]]*(#|$) ]] && continue
    line="${line#export }"
    [[ "$line" == NEMO_* ]] || continue
    if [[ ! "$line" =~ ^(NEMO_[A-Z0-9_]+)=(.*)$ ]]; then
      echo "Invalid NemoCode config line; use NEMO_KEY=literal-value" >&2; exit 1
    fi
    key="${BASH_REMATCH[1]}"; value="${BASH_REMATCH[2]}"
    if [[ "$value" == \"*\" || "$value" == \'*\' ]]; then
      value="${value:1:${#value}-2}"
    else
      value="${value%% #*}"
    fi
    if ! printenv "$key" >/dev/null 2>&1; then export "$key=$value"; fi
  done < "$config"
}
load_config

if [[ -f "$ROOT/Cargo.toml" ]]; then
  DATA_ROOT="$ROOT"
else
  DATA_ROOT="${XDG_DATA_HOME:-$HOME/.local/share}/nemocode"
fi

require_nemocode_root() {
  if [[ ! -f "$ROOT/Cargo.toml" || ! -f "$ROOT/start-nemo.sh" ]]; then
    log "NemoCode must be launched from the nemocode project directory."
    log "Missing Cargo.toml or start-nemo.sh under: $ROOT"
    exit 1
  fi
  if ! grep -Eq '^name[[:space:]]*=[[:space:]]*"nemocode"' "$ROOT/Cargo.toml"; then
    log "NemoCode must be launched from the nemocode project directory."
    log "Cargo.toml under $ROOT does not declare name = \"nemocode\"."
    exit 1
  fi
  export NEMO_PROJECT_ROOT="$ROOT"
}

BANNER="$(cat <<'EOF'
┳┓┏┓┳┳┓┏┓┏┓┏┓┳┓┏┓
┃┃┣ ┃┃┃┃┃┃ ┃┃┃┃┣ 
┛┗┗┛┛ ┗┗┛┗┛┗┛┻┛┗┛
EOF
)"

MODEL_REPO="${NEMO_MODEL_REPO:-S4MPL3BI4S/Nemotron-3-Nano-4B-Coding-Agent-GGUF}"
MODEL_FILE="${NEMO_MODEL_FILE:-Nemotron-3-Nano-4B-Coding-Agent-Q4_K_M.gguf}"
MODELS_DIR="${NEMO_MODELS_DIR:-$DATA_ROOT/models}"
MODEL_PATH="${NEMO_MODEL_PATH:-$MODELS_DIR/$MODEL_FILE}"
HOST="${NEMO_HOST:-127.0.0.1}"
PORT="${NEMO_PORT:-8080}"
CTX="${NEMO_CTX:-16384}"
GPU_LAYERS="${NEMO_GPU_LAYERS:-auto}"
FIT="${NEMO_FIT:-on}"
THREADS="${NEMO_THREADS:-}"
PARALLEL="${NEMO_PARALLEL:-1}"
FLASH_ATTN="${NEMO_FLASH_ATTN:-on}"
BATCH="${NEMO_BATCH:-2048}"
UBATCH="${NEMO_UBATCH:-512}"
CACHE_TYPE_K="${NEMO_CACHE_TYPE_K:-q8_0}"
CACHE_TYPE_V="${NEMO_CACHE_TYPE_V:-q8_0}"
MLOCK="${NEMO_MLOCK:-0}"
MODEL_REVISION="${NEMO_MODEL_REVISION:-main}"
HF_URL="https://huggingface.co/${MODEL_REPO}/resolve/${MODEL_REVISION}/${MODEL_FILE}"

VENDOR_DIR="${NEMO_VENDOR_DIR:-$DATA_ROOT/.vendor/llama.cpp}"
LLAMA_REPO="${NEMO_LLAMA_REPO:-ggml-org/llama.cpp}"
# Empty means "latest GitHub release".
LLAMA_RELEASE="${NEMO_LLAMA_RELEASE:-}"
if [[ -n "$LLAMA_RELEASE" && ! "$LLAMA_RELEASE" =~ ^b[0-9]+$ ]]; then
  echo "NEMO_LLAMA_RELEASE must be an official b<number> release tag" >&2; exit 1
fi

SERVER_PID=""
SERVER_LOG="${NEMO_SERVER_LOG:-$DATA_ROOT/.nemocode-server.log}"
LLAMA_LIB_DIR=""

cleanup() {
  if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

print_banner() {
  printf '%s\n\n' "$BANNER" >&2
}

log() {
  echo "$@" >&2
}

checksum_of() {
  if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | cut -d ' ' -f 1
  else shasum -a 256 "$1" | cut -d ' ' -f 1; fi
}

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    log "Missing required command: $1"
    log "$2"
    exit 1
  fi
}

os_name() {
  uname -s | tr '[:upper:]' '[:lower:]'
}

arch_name() {
  local arch
  arch="$(uname -m)"
  case "$arch" in
    x86_64|amd64) echo "x64" ;;
    aarch64|arm64) echo "arm64" ;;
    s390x) echo "s390x" ;;
    *) echo "$arch" ;;
  esac
}

has_vulkan() {
  if [[ "${NEMO_LLAMA_BACKEND:-}" == "cpu" ]]; then
    return 1
  fi
  if [[ "${NEMO_LLAMA_BACKEND:-}" == "vulkan" ]]; then
    return 0
  fi
  if gpu_offload_disabled; then
    return 1
  fi
  command -v vulkaninfo >/dev/null 2>&1 && vulkaninfo --summary >/dev/null 2>&1 && return 0
  return 1
}

has_cuda() {
  [[ "${NEMO_LLAMA_BACKEND:-}" == "cuda" ]] && return 0
  [[ -n "${NEMO_LLAMA_BACKEND:-}" ]] && return 1
  gpu_offload_disabled && return 1
  command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1
}

has_rocm() {
  if [[ "${NEMO_LLAMA_BACKEND:-}" == "rocm" ]]; then
    return 0
  fi
  if [[ "${NEMO_LLAMA_BACKEND:-}" == "cpu" || "${NEMO_LLAMA_BACKEND:-}" == "vulkan" ]]; then
    return 1
  fi
  command -v rocminfo >/dev/null 2>&1 || [[ -d /opt/rocm ]]
}

release_asset_name() {
  local tag="$1"
  local os arch backend
  os="$(os_name)"
  arch="$(arch_name)"

  case "$os" in
    linux)
      case "$arch" in
        x64)
          if has_cuda; then
            # Asset names vary by release. Select from the actual manifest below.
            backend="cuda"
          elif has_rocm; then
            backend="ubuntu-rocm-7.2-x64"
          elif has_vulkan; then
            backend="ubuntu-vulkan-x64"
          else
            backend="ubuntu-x64"
          fi
          ;;
        arm64)
          if has_vulkan; then
            backend="ubuntu-vulkan-arm64"
          else
            backend="ubuntu-arm64"
          fi
          ;;
        s390x)
          backend="ubuntu-s390x"
          ;;
        *)
          echo "Unsupported Linux architecture for auto-install: $arch" >&2
          exit 1
          ;;
      esac
      if [[ "$backend" == "cuda" ]]; then
        local asset
        asset="$(curl -fsSL "https://api.github.com/repos/${LLAMA_REPO}/releases/tags/${tag}" | sed -n 's/.*"name":[[:space:]]*"\([^"]*bin-ubuntu-cuda[^\"]*x64\.tar\.gz\)".*/\1/p' | head -n 1)"
        [[ -n "$asset" ]] || { echo "No CUDA Linux asset for $tag; provide NEMO_LLAMA_SERVER (Jetson needs a compatible build)." >&2; return 1; }
        echo "$asset"
      else
        echo "llama-${tag}-bin-${backend}.tar.gz"
      fi
      ;;
    darwin)
      case "$arch" in
        arm64) echo "llama-${tag}-bin-macos-arm64.tar.gz" ;;
        x64) echo "llama-${tag}-bin-macos-x64.tar.gz" ;;
        *)
          echo "Unsupported macOS architecture for auto-install: $arch" >&2
          exit 1
          ;;
      esac
      ;;
    *)
      echo "Unsupported OS for auto-install of llama-server: $os" >&2
      echo "Install llama.cpp manually: https://github.com/ggml-org/llama.cpp" >&2
      exit 1
      ;;
  esac
}

resolve_release_tag() {
  if [[ -n "$LLAMA_RELEASE" ]]; then
    echo "$LLAMA_RELEASE"
    return 0
  fi

  local api_url tag
  api_url="https://api.github.com/repos/${LLAMA_REPO}/releases/latest"
  tag="$(curl -fsSL "$api_url" | sed -n 's/.*"tag_name":[[:space:]]*"\([^"]*\)".*/\1/p' | head -n 1)"
  if [[ -z "$tag" ]]; then
    echo "Failed to resolve the latest llama.cpp release tag from GitHub." >&2
    exit 1
  fi
  echo "$tag"
}

find_vendored_llama_server() {
  local candidate
  if [[ -n "$LLAMA_RELEASE" ]]; then
    if [[ -x "$VENDOR_DIR/$LLAMA_RELEASE/llama-server" ]]; then
      echo "$VENDOR_DIR/$LLAMA_RELEASE/llama-server"; return 0
    fi
    return 1
  fi
  if [[ -x "$VENDOR_DIR/current/llama-server" ]]; then
    echo "$VENDOR_DIR/current/llama-server"
    return 0
  fi

  # Older/manual extracts may leave versioned directories.
  while IFS= read -r candidate; do
    if [[ -x "$candidate" ]]; then
      echo "$candidate"
      return 0
    fi
  done < <(find "$VENDOR_DIR" -type f -name 'llama-server' 2>/dev/null | sort -r)

  return 1
}

find_system_llama_server() {
  if [[ -n "${NEMO_LLAMA_SERVER:-}" ]]; then
    if [[ -x "${NEMO_LLAMA_SERVER}" ]]; then
      echo "${NEMO_LLAMA_SERVER}"
      return 0
    fi
    echo "NEMO_LLAMA_SERVER is set but not executable: ${NEMO_LLAMA_SERVER}" >&2
    exit 1
  fi

  if command -v llama-server >/dev/null 2>&1; then
    command -v llama-server
    return 0
  fi

  if command -v llama.cpp-server >/dev/null 2>&1; then
    command -v llama.cpp-server
    return 0
  fi

  local candidate
  for candidate in \
    "$HOME/.local/bin/llama-server" \
    "/usr/local/bin/llama-server" \
    "/opt/homebrew/bin/llama-server"
  do
    if [[ -x "$candidate" ]]; then
      echo "$candidate"
      return 0
    fi
  done

  return 1
}

install_llama_server() {
  require_cmd curl "Install curl so the launcher can download llama.cpp."
  require_cmd tar "Install tar so the launcher can extract llama.cpp."

  local tag asset url archive extract_dir current_link
  tag="$(resolve_release_tag)"
  asset="$(release_asset_name "$tag")"
  url="https://github.com/${LLAMA_REPO}/releases/download/${tag}/${asset}"
  archive="$VENDOR_DIR/downloads/${asset}"
  extract_dir="$VENDOR_DIR/${tag}"
  current_link="$VENDOR_DIR/current"

  mkdir -p "$VENDOR_DIR/downloads"

  if [[ -x "$extract_dir/llama-server" ]]; then
    ln -sfn "$extract_dir" "$current_link"
    log "Using cached llama-server from $extract_dir"
    return 0
  fi

  log "llama-server not found; installing llama.cpp automatically"
  log "  release : $tag"
  log "  asset   : $asset"
  log "  dest    : $extract_dir"
  log

  if [[ ! -f "$archive" ]]; then
    log "Downloading $url"
    curl -L --fail --progress-bar -o "${archive}.partial" "$url"
    mv "${archive}.partial" "$archive"
  else
    log "Using cached archive: $archive"
  fi

  local archive_hash
  archive_hash="$(checksum_of "$archive")"
  if [[ -n "${NEMO_LLAMA_SHA256:-}" && "$archive_hash" != "$NEMO_LLAMA_SHA256" ]]; then
    log "Runtime archive checksum mismatch"; return 1
  fi

  # Never recursively remove a path derived from configuration.
  if [[ -e "$extract_dir" ]]; then
    mv "$extract_dir" "${extract_dir}.previous.$(date +%s)"
  fi
  mkdir -p "$extract_dir"
  tar -xzf "$archive" -C "$extract_dir" --strip-components=1

  if [[ ! -x "$extract_dir/llama-server" ]]; then
    log "Downloaded llama.cpp archive, but llama-server was not found inside it."
    exit 1
  fi

  chmod +x "$extract_dir/llama-server"
  printf '%s\n' "release=$tag" "asset=$asset" "archive_sha256=$archive_hash" "backend=${NEMO_LLAMA_BACKEND:-auto}" > "$extract_dir/runtime.meta"
  ln -sfn "$extract_dir" "$current_link"
  log "Installed llama-server to $extract_dir"
}

ensure_llama_server() {
  local path

  if [[ -n "$LLAMA_RELEASE" && -z "${NEMO_LLAMA_SERVER:-}" ]]; then
    if path="$(find_vendored_llama_server)"; then echo "$path"; return 0; fi
    install_llama_server
    find_vendored_llama_server
    return
  fi

  if path="$(find_system_llama_server)"; then
    echo "$path"
    return 0
  fi

  if path="$(find_vendored_llama_server)"; then
    echo "$path"
    return 0
  fi

  install_llama_server

  if path="$(find_vendored_llama_server)"; then
    echo "$path"
    return 0
  fi

  echo "Failed to install llama-server automatically." >&2
  exit 1
}

download_model() {
  mkdir -p "$MODELS_DIR"
  mkdir -p "$(dirname "$MODEL_PATH")"

  if [[ -f "$MODEL_PATH" ]]; then
    log "Model already present: $MODEL_PATH"
    verify_model
    return 0
  fi

  log "Downloading ${MODEL_FILE}"
  log "From: ${HF_URL}"
  log "To:   ${MODEL_PATH}"
  log "This is about 2.8 GB and only happens once."

  if command -v curl >/dev/null 2>&1; then
    local tmp="${MODEL_PATH}.partial"
    curl -L --fail --progress-bar -o "$tmp" "$HF_URL"
    mv "$tmp" "$MODEL_PATH"
  elif command -v wget >/dev/null 2>&1; then
    local tmp="${MODEL_PATH}.partial"
    wget -O "$tmp" "$HF_URL"
    mv "$tmp" "$MODEL_PATH"
  else
    log "Need one of: hf, huggingface-cli, curl, or wget to download the model."
    exit 1
  fi

  if [[ ! -f "$MODEL_PATH" ]]; then
    log "Model download finished but file not found at: $MODEL_PATH"
    exit 1
  fi
  verify_model
}

verify_model() {
  [[ -s "$MODEL_PATH" ]] || { log "Missing or empty model: $MODEL_PATH"; return 1; }
  local actual expected="${NEMO_MODEL_SHA256:-}"
  actual="$(checksum_of "$MODEL_PATH")"
  if [[ -n "$expected" && "$actual" != "$expected" ]]; then
    log "Model checksum mismatch: expected $expected, got $actual"; return 1
  fi
  if [[ -f "${MODEL_PATH}.sha256" && -z "$expected" ]]; then
    expected="$(cut -d ' ' -f 1 "${MODEL_PATH}.sha256")"
    [[ "$actual" == "$expected" ]] || { log "Model changed since its recorded checksum"; return 1; }
  fi
  printf '%s\n' "$actual" > "${MODEL_PATH}.sha256"
  log "Model SHA256: $actual${expected:+ (verified)}"
}

gpu_offload_disabled() {
  [[ "${GPU_LAYERS}" == "0" || "${GPU_LAYERS}" == "none" || "${GPU_LAYERS}" == "off" ]]
}

ensure_writable_log() {
  if [[ -n "${NEMO_SERVER_LOG:-}" ]]; then
    return 0
  fi
  if [[ -e "$SERVER_LOG" && ! -w "$SERVER_LOG" ]] || [[ ! -w "$ROOT" ]]; then
    SERVER_LOG="${TMPDIR:-/tmp}/nemocode-server.log"
    log "Using writable server log: $SERVER_LOG"
  fi
}

warn_if_root() {
  if [[ "${EUID:-$(id -u)}" -eq 0 ]]; then
    log "Warning: running as root breaks GPU access on many setups."
    log "Prefer: ./start-nemo.sh   (without sudo)"
  fi
}

warn_if_low_vram() {
  if gpu_offload_disabled; then
    return 0
  fi

  local llama_server="${1:-}"
  if [[ -z "$llama_server" || ! -x "$llama_server" ]]; then
    return 0
  fi

  local devices free_mib
  devices="$(
    export LD_LIBRARY_PATH="${LLAMA_LIB_DIR:-$(dirname "$llama_server")}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    "$llama_server" --list-devices 2>/dev/null || true
  )"
  if [[ -z "$devices" ]]; then
    return 0
  fi

  free_mib="$(printf '%s\n' "$devices" | sed -n 's/.*(\([0-9][0-9]*\) MiB free).*/\1/p' | sort -n | tail -n 1)"
  if [[ -z "$free_mib" ]]; then
    return 0
  fi

  if [[ "$free_mib" -lt 2048 ]]; then
    log "Warning: GPU reports only ${free_mib} MiB free VRAM."
    if command -v nvidia-smi >/dev/null 2>&1; then
      nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null \
        | sed '/^$/d' \
        | while IFS= read -r line; do
            log "  in use: $line"
          done || true
    fi
    if [[ "$GPU_LAYERS" != "auto" && "$GPU_LAYERS" != "all" ]]; then
      log "  Tip: use NEMO_GPU_LAYERS=auto (default) or free VRAM from other apps."
    fi
  fi
}

wait_for_server() {
  local url="http://${HOST}:${PORT}/health"
  local attempt
  for attempt in $(seq 1 120); do
    if curl -fsS "$url" >/dev/null 2>&1; then
      log "Local model server is ready at http://${HOST}:${PORT}"
      return 0
    fi
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
      log "llama-server exited unexpectedly. Last log lines:"
      tail -n 40 "$SERVER_LOG" >&2 || true
      exit 1
    fi
    sleep 1
  done

  log "Timed out waiting for llama-server. See: $SERVER_LOG"
  tail -n 40 "$SERVER_LOG" >&2 || true
  exit 1
}

start_server() {
  local llama_server="${1:-}"
  if [[ -z "$llama_server" ]]; then
    llama_server="$(ensure_llama_server)"
  fi
  LLAMA_LIB_DIR="$(cd "$(dirname "$llama_server")" && pwd)"

  local -a args=(
    --model "$MODEL_PATH"
    --alias "${NEMO_MODEL:-Nemotron-3-Nano-4B-Coding-Agent-Q4_K_M}"
    --host "$HOST"
    --port "$PORT"
    --ctx-size "$CTX"
    --n-gpu-layers "$GPU_LAYERS"
    --fit "$FIT"
    --parallel "$PARALLEL"
    --flash-attn "$FLASH_ATTN"
    --batch-size "$BATCH"
    --ubatch-size "$UBATCH"
    --cache-type-k "$CACHE_TYPE_K"
    --cache-type-v "$CACHE_TYPE_V"
    --jinja
  )

  if [[ "${NEMO_IDLE_TIMEOUT_SECS:-0}" != "0" ]]; then
    local help_text
    help_text="$(LD_LIBRARY_PATH="$LLAMA_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" "$llama_server" --help 2>&1)"
    if [[ "$help_text" == *--sleep-idle-seconds* ]]; then
      args+=(--sleep-idle-seconds "$NEMO_IDLE_TIMEOUT_SECS")
    else
      log "This runtime has no idle-sleep flag; upgrade or leave NEMO_IDLE_TIMEOUT_SECS=0"; return 1
    fi
  fi

  if [[ -n "$THREADS" ]]; then
    args+=(--threads "$THREADS")
  fi

  if [[ "$MLOCK" == "1" || "$MLOCK" == "true" || "$MLOCK" == "yes" ]]; then
    args+=(--mlock)
  fi

  log "Starting llama-server with:"
  log "  binary : $llama_server"
  log "  model  : $MODEL_PATH"
  log "  listen : ${HOST}:${PORT}"
  log "  ctx    : $CTX"
  log "  gpu    : $GPU_LAYERS layers (fit=$FIT)"
  log "  parallel: $PARALLEL"
  log "  flash  : $FLASH_ATTN"
  log "  batch  : $BATCH / ubatch $UBATCH"
  log "  kv     : k=$CACHE_TYPE_K v=$CACHE_TYPE_V"
  if [[ "$MLOCK" == "1" || "$MLOCK" == "true" || "$MLOCK" == "yes" ]]; then
    log "  mlock  : enabled"
  fi
  log "  log    : $SERVER_LOG"

  # Vendored release builds keep shared libraries next to the binary.
  (
    export LD_LIBRARY_PATH="${LLAMA_LIB_DIR}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    export DYLD_LIBRARY_PATH="${LLAMA_LIB_DIR}${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
    exec "$llama_server" "${args[@]}"
  ) >"$SERVER_LOG" 2>&1 &
  SERVER_PID=$!
  wait_for_server
}

run_agent() {

  export NEMO_BASE_URL="${NEMO_BASE_URL:-http://${HOST}:${PORT}/v1}"
  export NEMO_MODEL="${NEMO_MODEL:-Nemotron-3-Nano-4B-Coding-Agent-Q4_K_M}"
  export NEMO_API_KEY="${NEMO_API_KEY:-local}"
  # So the CLI budgets prompt size against the same context window as llama-server.
  export NEMO_CTX="${CTX}"

  log
  log "Launching NemoCode..."
  log "Endpoint: ${NEMO_BASE_URL}"
  log "Model:    ${NEMO_MODEL}"
  log "Context:  ${NEMO_CTX}"
  log

  if [[ -n "${NEMO_CLI:-}" ]]; then
    "$NEMO_CLI" --workspace "$WORKSPACE" "$@"
  elif [[ -x "$ROOT/target/release/nemocode" ]]; then
    "$ROOT/target/release/nemocode" --workspace "$WORKSPACE" "$@"
  elif command -v nemocode >/dev/null 2>&1; then
    nemocode --workspace "$WORKSPACE" "$@"
  elif [[ -f "$ROOT/Cargo.toml" ]]; then
    require_cmd cargo "Install a release bundle or Rust to build the CLI."
    cargo build --release --locked --manifest-path "$ROOT/Cargo.toml"
    "$ROOT/target/release/nemocode" --workspace "$WORKSPACE" "$@"
  else
    log "NemoCode binary not found. Install the release bundle or set NEMO_CLI."; return 1
  fi
}

main() {
  print_banner
  log "NemoCode startup"
  log "Bundled model: ${MODEL_REPO}"
  log

  mkdir -p "$DATA_ROOT"
  log "Workspace: $WORKSPACE"
  log

  require_cmd curl "Install curl so the launcher can download dependencies and health-check the server."
  if ! command -v sha256sum >/dev/null 2>&1 && ! command -v shasum >/dev/null 2>&1; then
    log "Install sha256sum or shasum to verify artifacts"; return 1
  fi

  case "${1:-}" in
    --model-verify) verify_model; return ;;
    --model-install) download_model; return ;;
    --runtime-install) ensure_llama_server; return ;;
    --runtime-update)
      [[ -n "$LLAMA_RELEASE" ]] || { log "Set NEMO_LLAMA_RELEASE to the explicit release to install."; return 1; }
      install_llama_server; return ;;
  esac

  if [[ "${NEMO_ATTACH:-0}" == "1" ]]; then
    run_agent "$@"; return
  fi

  local llama_server
  llama_server="$(ensure_llama_server)"
  log "Using llama-server: $llama_server"
  log

  warn_if_root
  ensure_writable_log
  download_model
  warn_if_low_vram "$llama_server"
  start_server "$llama_server"
  if [[ "${1:-}" == "--runtime-start" ]]; then
    log "Runtime running in foreground; Ctrl+C stops it."
    wait "$SERVER_PID"
  else
    run_agent "$@"
  fi
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then main "$@"; fi
