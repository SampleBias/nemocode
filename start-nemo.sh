#!/usr/bin/env bash
# NemoCode launcher: install llama-server if needed, pick and download a GGUF
# sized for this machine, start the local model server, then run the harness.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="${NEMO_WORKSPACE:-$PWD}"
export NEMO_WORKSPACE="$WORKSPACE"

# Configuration is data, not shell code. Environment overrides file values.
# Settings that choose an executable or widen what the model may do.
is_privileged_setting() {
  case "$1" in
    NEMO_PERMISSION) [[ "$2" == trusted ]] ;;
    NEMO_PYTHON_LSP) [[ ! "$2" =~ ^(auto|off|0|false)?$ ]] ;;
    NEMO_PYTHON_LSP_COMMAND|NEMO_LAUNCHER|NEMO_LLAMA_SERVER|NEMO_LLAMA_REPO|NEMO_CLI|NEMO_BASH) return 0 ;;
    *) return 1 ;;
  esac
}

load_config() {
  local config="${NEMO_CONFIG:-$WORKSPACE/.env}" line key value workspace_config=0
  [[ -f "$config" ]] || return 0
  # A .env found in some other project may not be the user's own file. The
  # checkout's .env and an explicitly selected NEMO_CONFIG are trusted.
  if [[ -z "${NEMO_CONFIG:-}" && "$WORKSPACE" != "$ROOT" ]]; then workspace_config=1; fi
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
    if [[ "$workspace_config" == 1 ]] && is_privileged_setting "$key" "$value"; then
      echo "Ignoring $key from $config: set it in the environment or use NEMO_CONFIG" >&2
      continue
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

BANNER="$(cat <<'EOF'
┳┓┏┓┳┳┓┏┓┏┓┏┓┳┓┏┓
┃┃┣ ┃┃┃┃┃┃ ┃┃┃┃┣ 
┛┗┗┛┛ ┗┗┛┗┛┗┛┻┛┗┛
EOF
)"

MODELS_DIR="${NEMO_MODELS_DIR:-$DATA_ROOT/models}"
HOST="${NEMO_HOST:-127.0.0.1}"
PORT="${NEMO_PORT:-8080}"
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
# Filled in by select_model once the helpers below are defined.
MODEL_PRESET="" MODEL_REPO="" MODEL_FILE="" MODEL_PATH="" MODEL_REVISION=""
MODEL_SHA256="" MODEL_BYTES="" MODEL_ALIAS="" HF_URL="" CTX=""

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
  local name
  name="$(uname -s | tr '[:upper:]' '[:lower:]')"
  case "$name" in
    # Git Bash, MSYS2 and Cygwin all run native Windows programs.
    mingw*|msys*|cygwin*) echo "windows" ;;
    *) echo "$name" ;;
  esac
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
  # Windows GPU drivers install the Vulkan loader; vulkaninfo is rarely present.
  if [[ "$(os_name)" == windows && -f "${SYSTEMROOT:-C:\\Windows}/System32/vulkan-1.dll" ]]; then
    return 0
  fi
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
  if gpu_offload_disabled; then
    return 1
  fi
  command -v rocminfo >/dev/null 2>&1 || [[ -d /opt/rocm ]]
}

# File name of the build for this machine, chosen from the release's own file
# list so version suffixes (cuda-12.8, rocm-10.0) are never hardcoded here.
release_asset_name() {
  local tag="$1"
  local os arch pattern asset
  os="$(os_name)"
  arch="$(arch_name)"

  case "$os" in
    linux)
      case "$arch" in
        x64)
          if has_cuda; then
            pattern='ubuntu-cuda-[0-9.]+-x64\.tar\.gz'
          elif has_rocm; then
            pattern='ubuntu-rocm-[0-9.]+-x64\.tar\.gz'
          elif has_vulkan; then
            pattern='ubuntu-vulkan-x64\.tar\.gz'
          else
            pattern='ubuntu-x64\.tar\.gz'
          fi
          ;;
        arm64)
          if has_vulkan; then
            pattern='ubuntu-vulkan-arm64\.tar\.gz'
          else
            pattern='ubuntu-arm64\.tar\.gz'
          fi
          ;;
        s390x)
          pattern='ubuntu-s390x\.tar\.gz'
          ;;
        *)
          echo "Unsupported Linux architecture for auto-install: $arch" >&2
          exit 1
          ;;
      esac
      ;;
    windows)
      case "$arch" in
        x64|arm64) ;;
        *)
          echo "Unsupported Windows architecture for auto-install: $arch" >&2
          exit 1
          ;;
      esac
      # Vulkan runs on NVIDIA, AMD and Intel GPUs from one 40 MB download. The
      # CUDA build needs a further 400 MB of runtime libraries, so it is opt-in.
      if [[ "${NEMO_LLAMA_BACKEND:-}" == "cuda" ]]; then
        pattern="win-cuda-[0-9.]+-${arch}\\.zip"
      elif has_vulkan; then
        pattern="win-vulkan-${arch}\\.zip"
      else
        pattern="win-cpu-${arch}\\.zip"
      fi
      ;;
    darwin)
      case "$arch" in
        arm64|x64) pattern="macos-${arch}\\.tar\\.gz" ;;
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

  asset="$(curl -fsSL "https://api.github.com/repos/${LLAMA_REPO}/releases/tags/${tag}" \
    | sed -n 's/.*"name":[[:space:]]*"\([^"]*\)".*/\1/p' \
    | grep -E "^llama-${tag}-bin-${pattern}\$" | sort | head -n 1)" || true
  if [[ -z "$asset" ]]; then
    echo "llama.cpp release $tag has no build matching ${pattern}." >&2
    echo "Set NEMO_LLAMA_BACKEND (cpu, vulkan, cuda, rocm) or NEMO_LLAMA_SERVER." >&2
    return 1
  fi
  echo "$asset"
}

resolve_release_tag() {
  if [[ -n "$LLAMA_RELEASE" ]]; then
    echo "$LLAMA_RELEASE"
    return 0
  fi

  local api="https://api.github.com/repos/${LLAMA_REPO}/releases" tag=""
  tag="$(curl -fsSL "$api/latest" | sed -n 's/.*"tag_name":[[:space:]]*"\([^"]*\)".*/\1/p' | head -n 1)" || true
  # llama.cpp's "latest" release is now a version tag (v0.9.0) with no binaries,
  # only a pointer to the b<number> build that carries them.
  if [[ ! "$tag" =~ ^b[0-9]+$ ]]; then
    tag="$(curl -fsSL "https://github.com/${LLAMA_REPO}/releases/latest/download/nightly-tag.txt" 2>/dev/null \
      | tr -d '[:space:]')" || true
  fi
  if [[ ! "$tag" =~ ^b[0-9]+$ ]]; then
    tag="$(curl -fsSL "$api?per_page=30" \
      | sed -n 's/.*"tag_name":[[:space:]]*"\(b[0-9][0-9]*\)".*/\1/p' | head -n 1)" || true
  fi
  if [[ ! "$tag" =~ ^b[0-9]+$ ]]; then
    echo "Failed to resolve a llama.cpp build tag from GitHub; set NEMO_LLAMA_RELEASE=b<number>." >&2
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
  done < <(find "$VENDOR_DIR" -type f \( -name 'llama-server' -o -name 'llama-server.exe' \) 2>/dev/null | sort -r)

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

extract_runtime_archive() {
  local archive="$1" destination="$2"
  case "$archive" in
    *.zip)
      if command -v unzip >/dev/null 2>&1; then
        unzip -q -o "$archive" -d "$destination"
      else
        # Windows ships bsdtar, which reads zip files.
        tar -xf "$archive" -C "$destination"
      fi
      ;;
    *) tar -xzf "$archive" -C "$destination" --strip-components=1 ;;
  esac
}

# Point `current` at the installed release. Git Bash cannot create symlinks
# without extra privileges and would copy the whole directory instead, so on
# Windows the newest versioned directory is found by search.
link_current_runtime() {
  if [[ "$(os_name)" != windows ]]; then
    ln -sfn "$1" "$2"
  fi
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
    link_current_runtime "$extract_dir" "$current_link"
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
  extract_runtime_archive "$archive" "$extract_dir"
  if [[ "$asset" == *-win-cuda-* ]]; then
    # The Windows CUDA build ships without the CUDA runtime DLLs.
    local runtime="cudart-llama-bin-${asset#*-bin-}"
    if [[ ! -f "$VENDOR_DIR/downloads/$runtime" ]]; then
      log "Downloading CUDA runtime libraries: $runtime"
      curl -L --fail --progress-bar -o "$VENDOR_DIR/downloads/$runtime.partial" \
        "https://github.com/${LLAMA_REPO}/releases/download/${tag}/${runtime}"
      mv "$VENDOR_DIR/downloads/$runtime.partial" "$VENDOR_DIR/downloads/$runtime"
    fi
    extract_runtime_archive "$VENDOR_DIR/downloads/$runtime" "$extract_dir"
  fi

  if [[ ! -x "$extract_dir/llama-server" ]]; then
    log "Downloaded llama.cpp archive, but llama-server was not found inside it."
    exit 1
  fi

  chmod +x "$extract_dir/llama-server"
  printf '%s\n' "release=$tag" "asset=$asset" "archive_sha256=$archive_hash" "backend=${NEMO_LLAMA_BACKEND:-auto}" > "$extract_dir/runtime.meta"
  link_current_runtime "$extract_dir" "$current_link"
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
    verify_model quick
    return 0
  fi

  local tmp="${MODEL_PATH}.partial" size_note="This" free_kb=""
  if [[ -n "$MODEL_BYTES" ]]; then
    size_note="This is about $(format_gb "$MODEL_BYTES") GB and"
    # Fail before a long download rather than at the end of it.
    free_kb="$(df -Pk "$(dirname "$MODEL_PATH")" 2>/dev/null | awk 'NR==2 {print $4}' || true)"
    if [[ ! -e "$tmp" && "$free_kb" =~ ^[0-9]+$ ]] && (( free_kb < MODEL_BYTES / 1024 )); then
      log "Not enough free disk space for ${MODEL_FILE}: need $(format_gb "$MODEL_BYTES") GB in $(dirname "$MODEL_PATH")."
      log "Free some space, set NEMO_MODELS_DIR, or pick a smaller NEMO_MODEL_PRESET (mini, small)."
      exit 1
    fi
  fi

  log "Downloading ${MODEL_FILE}"
  log "From: ${HF_URL}"
  log "To:   ${MODEL_PATH}"
  log "${size_note} only happens once. An interrupted download resumes."

  if command -v curl >/dev/null 2>&1; then
    if ! curl -L --fail --progress-bar -C - -o "$tmp" "$HF_URL"; then
      log "Download failed. Run again to resume, or delete $tmp to start over."
      exit 1
    fi
  elif command -v wget >/dev/null 2>&1; then
    if ! wget -c -O "$tmp" "$HF_URL"; then
      log "Download failed. Run again to resume, or delete $tmp to start over."
      exit 1
    fi
  else
    log "Need curl or wget to download the model."
    exit 1
  fi
  mv "$tmp" "$MODEL_PATH"

  if ! verify_model; then
    # Do not leave a bad file where the next launch would load it.
    mv "$MODEL_PATH" "${MODEL_PATH}.corrupt"
    log "Moved the bad download to ${MODEL_PATH}.corrupt; run again to download a fresh copy."
    exit 1
  fi
}

# verify_model quick: trust a checksum recorded after the file was last written,
# so every launch does not re-read a multi-gigabyte model from a slow disk.
verify_model() {
  [[ -s "$MODEL_PATH" ]] || { log "Missing or empty model: $MODEL_PATH"; return 1; }
  local actual expected="${NEMO_MODEL_SHA256:-$MODEL_SHA256}" recorded=""
  if [[ "${1:-}" == quick && "${MODEL_PATH}.sha256" -nt "$MODEL_PATH" ]]; then
    recorded="$(cut -d ' ' -f 1 "${MODEL_PATH}.sha256")"
    if [[ -n "$recorded" && ( -z "$expected" || "$recorded" == "$expected" ) ]]; then
      log "Model SHA256: $recorded (recorded; run with --model-verify to re-hash)"
      return 0
    fi
  fi
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

format_gb() {
  awk -v bytes="$1" 'BEGIN { printf "%.1f", bytes / 1000000000 }'
}

total_ram_mib() {
  local value=""
  if [[ -r /proc/meminfo ]]; then
    value="$(awk '/^MemTotal:/ { print int($2 / 1024) }' /proc/meminfo 2>/dev/null || true)"
  elif command -v sysctl >/dev/null 2>&1; then
    value="$(sysctl -n hw.memsize 2>/dev/null | awk '{ print int($1 / 1048576) }' || true)"
  fi
  if [[ "$value" =~ ^[0-9]+$ ]]; then echo "$value"; else echo 0; fi
}

max_vram_mib() {
  local value=""
  if gpu_offload_disabled || [[ "${NEMO_LLAMA_BACKEND:-}" == cpu ]]; then
    echo 0; return 0
  fi
  if [[ "$(os_name)" == darwin && "$(arch_name)" == arm64 ]]; then
    # Apple Silicon shares memory between CPU and GPU; leave a third for the system.
    echo $(( $(total_ram_mib) * 2 / 3 )); return 0
  fi
  if command -v nvidia-smi >/dev/null 2>&1; then
    value="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null \
      | tr -d ' \r' | sort -n | tail -n 1 || true)"
  fi
  if [[ "$value" =~ ^[0-9]+$ ]]; then echo "$value"; else echo 0; fi
}

# Pick the largest preset that should run comfortably. Without a detected GPU
# the 4B model is the ceiling: larger ones load, but are too slow for a tool loop.
detect_model_preset() {
  local ram vram
  ram="$(total_ram_mib)"
  vram="$(max_vram_mib)"
  if (( vram >= 22000 )); then echo large
  elif (( vram >= 7000 )); then echo medium
  elif (( vram >= 3500 )); then echo small
  elif (( ram > 0 && ram < 7000 )); then echo mini
  else echo small
  fi
}

# One open-weight GGUF per hardware tier. Each entry pins the repository
# revision and the publisher's SHA-256, so the download is verified.
select_model() {
  local repo file revision sha bytes ctx=16384 custom=0
  MODEL_PRESET="${NEMO_MODEL_PRESET:-auto}"
  if [[ -n "${NEMO_MODEL_REPO:-}${NEMO_MODEL_FILE:-}${NEMO_MODEL_PATH:-}" ]]; then
    custom=1
  fi
  if [[ "$MODEL_PRESET" == auto ]]; then
    if [[ "$custom" == 1 ]]; then MODEL_PRESET=small; else MODEL_PRESET="$(detect_model_preset)"; fi
  fi
  case "$MODEL_PRESET" in
    mini) # under 8 GB RAM
      repo=unsloth/Qwen3.5-2B-GGUF file=Qwen3.5-2B-Q4_K_M.gguf bytes=1280835840
      revision=f6d5376be1edb4d416d56da11e5397a961aca8ae
      sha=aaf42c8b7c3cab2bf3d69c355048d4a0ee9973d48f16c731c0520ee914699223 ;;
    small) # 8 GB RAM on CPU, or a 4-6 GB GPU
      repo=unsloth/Qwen3.5-4B-GGUF file=Qwen3.5-4B-Q4_K_M.gguf bytes=2740937888
      revision=e87f176479d0855a907a41277aca2f8ee7a09523
      sha=00fe7986ff5f6b463e62455821146049db6f9313603938a70800d1fb69ef11a4 ;;
    medium) # 8-16 GB GPU, or a 16 GB Apple Silicon Mac
      repo=unsloth/Qwen3.5-9B-GGUF file=Qwen3.5-9B-Q4_K_M.gguf bytes=5680522464
      revision=3885219b6810b007914f3a7950a8d1b469d598a5
      sha=03b74727a860a56338e042c4420bb3f04b2fec5734175f4cb9fa853daf52b7e8 ;;
    large) # 24 GB GPU
      repo=unsloth/Qwen3.8-27B-GGUF file=Qwen3.8-27B-UD-Q4_K_M.gguf bytes=16464440224 ctx=32768
      revision=43f522df59109a037f64f1e29ab1eff5c00e8e1e
      sha=322e194ff79741c7baa497c240f677f54b201b0efab44ca8e50f122b39123482 ;;
    nemotron) # the original NemoCode model
      repo=S4MPL3BI4S/Nemotron-3-Nano-4B-Coding-Agent-GGUF
      file=Nemotron-3-Nano-4B-Coding-Agent-Q4_K_M.gguf bytes=2837073024
      revision=887ce639634143e784a966dc26b4df53486ae81d
      sha=67514959734347b17cce57ab75e9e91cd469e57830acbd8ee2a4363060431741 ;;
    *)
      echo "NEMO_MODEL_PRESET must be auto, mini, small, medium, large, or nemotron" >&2
      exit 1 ;;
  esac
  if [[ "$custom" == 1 ]]; then
    # A user-chosen model: nothing about the preset's file applies to it.
    MODEL_PRESET=custom
    revision=main sha="" bytes=""
    if [[ -n "${NEMO_MODEL_PATH:-}" && -z "${NEMO_MODEL_FILE:-}" ]]; then
      file="$(basename "$NEMO_MODEL_PATH")"
    fi
  fi
  MODEL_REPO="${NEMO_MODEL_REPO:-$repo}"
  MODEL_FILE="${NEMO_MODEL_FILE:-$file}"
  MODEL_PATH="${NEMO_MODEL_PATH:-$MODELS_DIR/$MODEL_FILE}"
  MODEL_REVISION="${NEMO_MODEL_REVISION:-$revision}"
  # The pinned checksum only describes the pinned revision.
  if [[ "$MODEL_REVISION" != "$revision" ]]; then sha="" bytes=""; fi
  MODEL_SHA256="$sha"
  MODEL_BYTES="$bytes"
  MODEL_ALIAS="${MODEL_FILE%.gguf}"
  HF_URL="https://huggingface.co/${MODEL_REPO}/resolve/${MODEL_REVISION}/${MODEL_FILE}"
  CTX="${NEMO_CTX:-$ctx}"
}
select_model

ensure_writable_log() {
  if [[ -n "${NEMO_SERVER_LOG:-}" ]]; then
    return 0
  fi
  if [[ -e "$SERVER_LOG" && ! -w "$SERVER_LOG" ]] || [[ ! -w "$(dirname "$SERVER_LOG")" ]]; then
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
  # Large models on a slow disk can take minutes to load.
  for attempt in $(seq 1 "${NEMO_SERVER_START_TIMEOUT:-300}"); do
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

  # Otherwise the health check below would pass against somebody else's server
  # while ours fails to bind, and the agent would talk to the wrong model.
  if curl -fsS --max-time 2 "http://${HOST}:${PORT}/health" >/dev/null 2>&1; then
    log "A server is already answering on ${HOST}:${PORT}."
    log "Set NEMO_ATTACH=1 to use it, or choose another NEMO_PORT."
    return 1
  fi

  local -a args=(
    --model "$MODEL_PATH"
    --alias "${NEMO_MODEL:-$MODEL_ALIAS}"
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
  export NEMO_MODEL="${NEMO_MODEL:-$MODEL_ALIAS}"
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
  elif [[ -f "$ROOT/Cargo.toml" ]]; then
    # Cargo checks freshness cheaply; never launch an executable from older source.
    require_cmd cargo "Install a release bundle or Rust to build the CLI."
    cargo build --release --locked --manifest-path "$ROOT/Cargo.toml" --target-dir "$ROOT/target" || return
    "$ROOT/target/release/nemocode" --workspace "$WORKSPACE" "$@"
  elif [[ -x "$ROOT/../../bin/nemocode" ]]; then
    "$ROOT/../../bin/nemocode" --workspace "$WORKSPACE" "$@"
  elif command -v nemocode >/dev/null 2>&1; then
    nemocode --workspace "$WORKSPACE" "$@"
  else
    log "NemoCode binary not found. Install the release bundle or set NEMO_CLI."; return 1
  fi
}

main() {
  print_banner
  log "NemoCode startup"
  log "Model: ${MODEL_FILE} (preset ${MODEL_PRESET}; ${MODEL_REPO})"
  log "       Override with NEMO_MODEL_PRESET=mini|small|medium|large|nemotron"
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
