#!/usr/bin/env bash
# NemoCode launcher: install llama-server if needed, download the bundled GGUF,
# start the local model server, then run the coding-agent harness.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

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
MODELS_DIR="${NEMO_MODELS_DIR:-$ROOT/models}"
MODEL_PATH="${NEMO_MODEL_PATH:-$MODELS_DIR/$MODEL_FILE}"
HOST="${NEMO_HOST:-127.0.0.1}"
PORT="${NEMO_PORT:-8080}"
CTX="${NEMO_CTX:-16384}"
GPU_LAYERS="${NEMO_GPU_LAYERS:-99}"
THREADS="${NEMO_THREADS:-}"
HF_URL="https://huggingface.co/${MODEL_REPO}/resolve/main/${MODEL_FILE}"

VENDOR_DIR="${NEMO_VENDOR_DIR:-$ROOT/.vendor/llama.cpp}"
LLAMA_REPO="${NEMO_LLAMA_REPO:-ggml-org/llama.cpp}"
# Empty means "latest GitHub release".
LLAMA_RELEASE="${NEMO_LLAMA_RELEASE:-}"

SERVER_PID=""
SERVER_LOG="${NEMO_SERVER_LOG:-$ROOT/.nemocode-server.log}"
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
  if [[ "${GPU_LAYERS}" == "0" ]]; then
    return 1
  fi
  command -v vulkaninfo >/dev/null 2>&1 && return 0
  [[ -e /usr/lib/libvulkan.so.1 || -e /usr/lib64/libvulkan.so.1 ]] && return 0
  ldconfig -p 2>/dev/null | grep -q 'libvulkan\.so' && return 0
  return 1
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
          if has_rocm; then
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
      echo "llama-${tag}-bin-${backend}.tar.gz"
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

  rm -rf "$extract_dir"
  mkdir -p "$extract_dir"
  tar -xzf "$archive" -C "$extract_dir" --strip-components=1

  if [[ ! -x "$extract_dir/llama-server" ]]; then
    log "Downloaded llama.cpp archive, but llama-server was not found inside it."
    exit 1
  fi

  chmod +x "$extract_dir/llama-server"
  ln -sfn "$extract_dir" "$current_link"
  log "Installed llama-server to $extract_dir"
}

ensure_llama_server() {
  local path

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

  if [[ -f "$MODEL_PATH" ]]; then
    log "Model already present: $MODEL_PATH"
    return 0
  fi

  log "Downloading ${MODEL_FILE}"
  log "From: ${HF_URL}"
  log "To:   ${MODEL_PATH}"
  log "This is about 2.8 GB and only happens once."

  if command -v hf >/dev/null 2>&1; then
    hf download "$MODEL_REPO" "$MODEL_FILE" --local-dir "$MODELS_DIR"
  elif command -v huggingface-cli >/dev/null 2>&1; then
    huggingface-cli download "$MODEL_REPO" "$MODEL_FILE" --local-dir "$MODELS_DIR"
  elif command -v curl >/dev/null 2>&1; then
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
    --host "$HOST"
    --port "$PORT"
    --ctx-size "$CTX"
    --n-gpu-layers "$GPU_LAYERS"
    --jinja
  )

  if [[ -n "$THREADS" ]]; then
    args+=(--threads "$THREADS")
  fi

  log "Starting llama-server with:"
  log "  binary : $llama_server"
  log "  model  : $MODEL_PATH"
  log "  listen : ${HOST}:${PORT}"
  log "  ctx    : $CTX"
  log "  gpu    : $GPU_LAYERS layers"
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
  require_cmd cargo "Install Rust from https://rustup.rs then re-run ./start-nemo.sh"

  export NEMO_BASE_URL="${NEMO_BASE_URL:-http://${HOST}:${PORT}/v1}"
  export NEMO_MODEL="${NEMO_MODEL:-Nemotron-3-Nano-4B-Coding-Agent-Q4_K_M}"
  export NEMO_API_KEY="${NEMO_API_KEY:-local}"

  log
  log "Building and launching NemoCode..."
  log "Endpoint: ${NEMO_BASE_URL}"
  log "Model:    ${NEMO_MODEL}"
  log

  cargo run --release
}

main() {
  print_banner
  log "NemoCode startup"
  log "Bundled model: ${MODEL_REPO}"
  log

  require_nemocode_root
  log "Project root: $NEMO_PROJECT_ROOT"
  log

  require_cmd curl "Install curl so the launcher can download dependencies and health-check the server."

  local llama_server
  llama_server="$(ensure_llama_server)"
  log "Using llama-server: $llama_server"
  log

  download_model
  start_server "$llama_server"
  run_agent
}

main "$@"
