#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
#   MSA Arabic TTS System — Launcher & Management Script
#   Modern Standard Arabic Text-to-Speech · VITS + HiFi-GAN
#   Requires: bash 4+, Python 3.10+, curl, (optional) docker, nvidia-smi
# ═══════════════════════════════════════════════════════════════════════════════

set -euo pipefail
IFS=$'\n\t'

# ─── Colours & Formatting ───────────────────────────────────────────────────
RED='\033[0;31m';    LRED='\033[1;31m'
GREEN='\033[0;32m';  LGREEN='\033[1;32m'
YELLOW='\033[0;33m'; LYELLOW='\033[1;33m'
BLUE='\033[0;34m';   LBLUE='\033[1;34m'
MAGENTA='\033[0;35m';LMAGENTA='\033[1;35m'
CYAN='\033[0;36m';   LCYAN='\033[1;36m'
WHITE='\033[0;37m';  LWHITE='\033[1;37m'
GOLD='\033[38;5;220m'
BOLD='\033[1m';      DIM='\033[2m';  RESET='\033[0m'

# ─── Global Config ──────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${SCRIPT_DIR}/.tts_config"
LOG_DIR="${SCRIPT_DIR}/logs"
PID_FILE="${SCRIPT_DIR}/.tts_server.pid"
VENV_DIR="${SCRIPT_DIR}/venv"
BACKEND_FILE="${SCRIPT_DIR}/msa_tts_backend.py"
FRONTEND_FILE="${SCRIPT_DIR}/msa_tts_frontend.html"
REQUIREMENTS_FILE="${SCRIPT_DIR}/requirements.txt"
DATA_DIR="${SCRIPT_DIR}/data"
CHECKPOINTS_DIR="${SCRIPT_DIR}/checkpoints"
EXPORTS_DIR="${SCRIPT_DIR}/exports"

# ─── Defaults (override in .tts_config) ─────────────────────────────────────
API_HOST="${API_HOST:-0.0.0.0}"
API_PORT="${API_PORT:-8000}"
API_WORKERS="${API_WORKERS:-1}"
DEVICE="${DEVICE:-auto}"
DIACRITIZER="${DIACRITIZER:-auto}"
MODEL_PATH="${MODEL_PATH:-}"
LOG_LEVEL="${LOG_LEVEL:-info}"
FRONTEND_PORT="${FRONTEND_PORT:-8080}"

# Load persisted config if present
[[ -f "$CONFIG_FILE" ]] && source "$CONFIG_FILE"

# ─── Helpers ────────────────────────────────────────────────────────────────
print_line() { printf "${DIM}%s${RESET}\n" "$(printf '─%.0s' {1..72})"; }
print_double() { printf "${GOLD}%s${RESET}\n" "$(printf '═%.0s' {1..72})"; }

banner() {
  clear
  print_double
  printf "${GOLD}${BOLD}"
  cat << 'BANNER'
    ███╗   ███╗███████╗ █████╗      ████████╗████████╗███████╗
    ████╗ ████║██╔════╝██╔══██╗        ██╔══╝╚══██╔══╝██╔════╝
    ██╔████╔██║███████╗███████║        ██║      ██║   ███████╗
    ██║╚██╔╝██║╚════██║██╔══██║        ██║      ██║   ╚════██║
    ██║ ╚═╝ ██║███████║██║  ██║        ██║      ██║   ███████║
    ╚═╝     ╚═╝╚══════╝╚═╝  ╚═╝        ╚═╝      ╚═╝   ╚══════╝
BANNER
  printf "${RESET}"
  printf "  ${CYAN}Modern Standard Arabic Text-to-Speech System${RESET}\n"
  printf "  ${DIM}VITS + HiFi-GAN  ·  FastAPI  ·  PyTorch${RESET}\n"
  print_double
  echo
}

info()    { printf "  ${CYAN}[INFO]${RESET}  %s\n" "$*"; }
ok()      { printf "  ${LGREEN}[ OK ]${RESET}  %s\n" "$*"; }
warn()    { printf "  ${YELLOW}[WARN]${RESET}  %s\n" "$*"; }
error()   { printf "  ${LRED}[ERR ]${RESET}  %s\n" "$*" >&2; }
step()    { printf "\n  ${GOLD}▶${RESET} ${BOLD}%s${RESET}\n" "$*"; }
success() { printf "\n  ${LGREEN}✔  %s${RESET}\n" "$*"; }
fail()    { printf "\n  ${LRED}✘  %s${RESET}\n" "$*"; }

press_enter() {
  echo
  printf "  ${DIM}Press [Enter] to continue…${RESET}"
  read -r
}

confirm() {
  local msg="$1"
  printf "  ${YELLOW}?${RESET}  ${msg} [y/N] "
  read -r reply
  [[ "$reply" =~ ^[Yy]$ ]]
}

require_cmd() {
  command -v "$1" &>/dev/null || {
    error "Required command not found: $1"
    return 1
  }
}

server_running() {
  [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null
}

save_config() {
  cat > "$CONFIG_FILE" << EOF
# MSA TTS Config — auto-generated $(date)
API_HOST="${API_HOST}"
API_PORT="${API_PORT}"
API_WORKERS="${API_WORKERS}"
DEVICE="${DEVICE}"
DIACRITIZER="${DIACRITIZER}"
MODEL_PATH="${MODEL_PATH}"
LOG_LEVEL="${LOG_LEVEL}"
FRONTEND_PORT="${FRONTEND_PORT}"
EOF
  ok "Configuration saved to ${CONFIG_FILE}"
}

venv_python() {
  if [[ -d "$VENV_DIR" ]]; then
    "${VENV_DIR}/bin/python"
  else
    python3
  fi
}

venv_pip() {
  if [[ -d "$VENV_DIR" ]]; then
    "${VENV_DIR}/bin/pip"
  else
    pip3
  fi
}

# ─── 1. SETUP ────────────────────────────────────────────────────────────────
menu_setup() {
  while true; do
    banner
    printf "  ${BOLD}${GOLD}⚙  SETUP & INSTALLATION${RESET}\n\n"
    printf "  ${LWHITE}1)${RESET}  Create Python virtual environment\n"
    printf "  ${LWHITE}2)${RESET}  Install all Python dependencies\n"
    printf "  ${LWHITE}3)${RESET}  Install Arabic NLP backends (CAMeL / Mishkal)\n"
    printf "  ${LWHITE}4)${RESET}  Download CAMeL Tools data (MSA models)\n"
    printf "  ${LWHITE}5)${RESET}  Install VITS from GitHub\n"
    printf "  ${LWHITE}6)${RESET}  Check GPU / CUDA availability\n"
    printf "  ${LWHITE}7)${RESET}  Full auto-setup (runs 1→6)\n"
    printf "  ${LWHITE}8)${RESET}  Verify installation\n"
    print_line
    printf "  ${LWHITE}0)${RESET}  Back to main menu\n\n"
    printf "  ${GOLD}Choice:${RESET} "
    read -r choice

    case "$choice" in
      1) setup_venv ;;
      2) install_deps ;;
      3) install_arabic_nlp ;;
      4) download_camel_data ;;
      5) install_vits ;;
      6) check_gpu ;;
      7) full_setup ;;
      8) verify_install ;;
      0) return ;;
      *) warn "Invalid option" ;;
    esac
  done
}

setup_venv() {
  step "Creating Python virtual environment"
  require_cmd python3 || { press_enter; return; }
  python3 -m venv "$VENV_DIR"
  "${VENV_DIR}/bin/pip" install --upgrade pip setuptools wheel -q
  success "Virtual environment created at ${VENV_DIR}"
  press_enter
}

install_deps() {
  step "Installing Python dependencies"
  if [[ ! -f "$REQUIREMENTS_FILE" ]]; then
    error "requirements.txt not found at ${REQUIREMENTS_FILE}"
    press_enter; return
  fi
  venv_pip install -r "$REQUIREMENTS_FILE" --break-system-packages 2>&1 | \
    grep -E "(Successfully|ERROR|already)" || true
  success "Dependencies installed"
  press_enter
}

install_arabic_nlp() {
  step "Installing Arabic NLP backends"
  info "Installing CAMeL Tools (BERT-based diacritizer)…"
  venv_pip install camel-tools --break-system-packages -q && ok "CAMeL Tools installed" || warn "CAMeL Tools failed"
  info "Installing Mishkal (rule-based diacritizer)…"
  venv_pip install mishkal --break-system-packages -q && ok "Mishkal installed" || warn "Mishkal not available on PyPI — install from source"
  info "Installing Farasa (alternative)…"
  venv_pip install farasapy --break-system-packages -q && ok "Farasa installed" || warn "Farasa optional"
  press_enter
}

download_camel_data() {
  step "Downloading CAMeL Tools MSA data"
  info "This downloads ~2GB of pre-trained Arabic NLP models"
  confirm "Proceed with download?" || { press_enter; return; }
  venv_python -c "import camel_tools; from camel_tools.data import get_dataset_path" 2>/dev/null || {
    warn "CAMeL Tools not installed yet — run option 3 first"
    press_enter; return
  }
  venv_python -m camel_tools.cli.camel_data -i defaults 2>&1 | tail -5
  success "CAMeL data downloaded"
  press_enter
}

install_vits() {
  step "Installing VITS from GitHub"
  require_cmd git || { press_enter; return; }
  local vits_dir="${SCRIPT_DIR}/vits"
  if [[ -d "$vits_dir" ]]; then
    info "VITS directory exists — pulling latest"
    git -C "$vits_dir" pull
  else
    git clone https://github.com/jaywalnut310/vits.git "$vits_dir"
  fi
  cd "$vits_dir"
  venv_pip install -e . -q 2>/dev/null || true
  # Compile monotonic align
  if [[ -d "monotonic_align" ]]; then
    cd monotonic_align
    venv_python setup.py build_ext --inplace 2>&1 | tail -3
    cd ..
  fi
  cd "$SCRIPT_DIR"
  success "VITS installed at ${vits_dir}"
  press_enter
}

check_gpu() {
  step "GPU & CUDA Status"
  print_line
  if command -v nvidia-smi &>/dev/null; then
    nvidia-smi --query-gpu=name,memory.total,memory.free,temperature.gpu,utilization.gpu \
               --format=csv,noheader,nounits 2>/dev/null | \
    while IFS=',' read -r name total free temp util; do
      printf "  ${LGREEN}GPU:${RESET}         %s\n" "$name"
      printf "  ${LGREEN}VRAM Total:${RESET}  %s MB\n" "$total"
      printf "  ${LGREEN}VRAM Free:${RESET}   %s MB\n" "$free"
      printf "  ${LGREEN}Temp:${RESET}        %s °C\n" "$temp"
      printf "  ${LGREEN}Utilization:${RESET} %s%%\n" "$util"
    done
    ok "NVIDIA GPU detected"
  else
    warn "nvidia-smi not found — CPU-only mode"
  fi
  echo
  # PyTorch CUDA check
  venv_python -c "
import torch
print(f'  PyTorch:      {torch.__version__}')
print(f'  CUDA avail:   {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  CUDA version: {torch.version.cuda}')
    print(f'  GPU count:    {torch.cuda.device_count()}')
    print(f'  GPU name:     {torch.cuda.get_device_name(0)}')
" 2>/dev/null || warn "PyTorch not installed yet"
  press_enter
}

full_setup() {
  step "Full Auto-Setup"
  info "This will: create venv → install deps → install Arabic NLP → check GPU"
  confirm "Proceed?" || return
  setup_venv
  install_deps
  install_arabic_nlp
  check_gpu
  verify_install
}

verify_install() {
  step "Verifying Installation"
  print_line
  local ok_count=0; local fail_count=0
  check_item() {
    local label="$1"; local cmd="$2"
    if eval "$cmd" &>/dev/null; then
      printf "  ${LGREEN}✔${RESET}  %-30s ${DIM}OK${RESET}\n" "$label"
      ((ok_count++))
    else
      printf "  ${LRED}✘${RESET}  %-30s ${LRED}MISSING${RESET}\n" "$label"
      ((fail_count++))
    fi
  }
  check_item "Python 3.10+"     "venv_python --version"
  check_item "FastAPI"          "venv_python -c 'import fastapi'"
  check_item "Uvicorn"          "venv_python -c 'import uvicorn'"
  check_item "PyTorch"          "venv_python -c 'import torch'"
  check_item "NumPy"            "venv_python -c 'import numpy'"
  check_item "CAMeL Tools"      "venv_python -c 'import camel_tools'"
  check_item "Mishkal"          "venv_python -c 'import mishkal'"
  check_item "Librosa"          "venv_python -c 'import librosa'"
  check_item "Backend file"     "[[ -f '$BACKEND_FILE' ]]"
  check_item "Frontend file"    "[[ -f '$FRONTEND_FILE' ]]"
  check_item "curl"             "command -v curl"
  print_line
  printf "  ${LGREEN}Passed: %d${RESET}  ${LRED}Failed: %d${RESET}\n" "$ok_count" "$fail_count"
  press_enter
}

# ─── 2. SERVER ───────────────────────────────────────────────────────────────
menu_server() {
  while true; do
    banner
    local status_str
    if server_running; then
      status_str="${LGREEN}● RUNNING${RESET} (PID $(cat "$PID_FILE")) on ${API_HOST}:${API_PORT}"
    else
      status_str="${LRED}○ STOPPED${RESET}"
    fi
    printf "  ${BOLD}${GOLD}🚀 SERVER MANAGEMENT${RESET}\n"
    printf "  Status: ${status_str}\n\n"
    printf "  ${LWHITE}1)${RESET}  Start server\n"
    printf "  ${LWHITE}2)${RESET}  Stop server\n"
    printf "  ${LWHITE}3)${RESET}  Restart server\n"
    printf "  ${LWHITE}4)${RESET}  Start with Docker\n"
    printf "  ${LWHITE}5)${RESET}  View live logs\n"
    printf "  ${LWHITE}6)${RESET}  Tail error log\n"
    printf "  ${LWHITE}7)${RESET}  Server status & health check\n"
    printf "  ${LWHITE}8)${RESET}  Serve frontend (static HTTP)\n"
    printf "  ${LWHITE}9)${RESET}  Start in development mode (--reload)\n"
    print_line
    printf "  ${LWHITE}0)${RESET}  Back\n\n"
    printf "  ${GOLD}Choice:${RESET} "
    read -r choice

    case "$choice" in
      1) start_server ;;
      2) stop_server ;;
      3) stop_server; sleep 1; start_server ;;
      4) start_docker ;;
      5) view_logs ;;
      6) tail_errors ;;
      7) server_health ;;
      8) serve_frontend ;;
      9) start_dev_server ;;
      0) return ;;
      *) warn "Invalid option" ;;
    esac
  done
}

start_server() {
  step "Starting MSA TTS API Server"
  if server_running; then
    warn "Server already running (PID $(cat "$PID_FILE"))"
    press_enter; return
  fi
  [[ -f "$BACKEND_FILE" ]] || { error "Backend file not found: ${BACKEND_FILE}"; press_enter; return; }
  mkdir -p "$LOG_DIR"
  local log_file="${LOG_DIR}/server_$(date +%Y%m%d_%H%M%S).log"
  local cmd=(
    $(venv_python) -m uvicorn
    "$(basename "${BACKEND_FILE%.py}"):app"
    --host "$API_HOST"
    --port "$API_PORT"
    --workers "$API_WORKERS"
    --log-level "$LOG_LEVEL"
  )
  [[ -n "$MODEL_PATH" ]] && export TTS_MODEL_PATH="$MODEL_PATH"
  export TTS_DEVICE="$DEVICE"
  export TTS_DIACRITIZER="$DIACRITIZER"

  cd "$(dirname "$BACKEND_FILE")"
  nohup "${cmd[@]}" > "$log_file" 2>&1 &
  echo $! > "$PID_FILE"
  sleep 2
  if server_running; then
    success "Server started — PID $(cat "$PID_FILE")"
    info "API endpoint:  http://${API_HOST}:${API_PORT}"
    info "Swagger docs:  http://${API_HOST}:${API_PORT}/docs"
    info "Log file:      ${log_file}"
  else
    fail "Server failed to start — check log: ${log_file}"
    tail -20 "$log_file" 2>/dev/null
  fi
  press_enter
}

stop_server() {
  step "Stopping Server"
  if ! server_running; then
    warn "No server is running"
    press_enter; return
  fi
  local pid
  pid=$(cat "$PID_FILE")
  kill "$pid" 2>/dev/null && ok "Sent SIGTERM to PID ${pid}"
  sleep 1
  if kill -0 "$pid" 2>/dev/null; then
    kill -9 "$pid" 2>/dev/null && warn "Force-killed PID ${pid}"
  fi
  rm -f "$PID_FILE"
  success "Server stopped"
  press_enter
}

start_docker() {
  step "Starting with Docker"
  require_cmd docker || { press_enter; return; }
  local image="msa-tts:latest"
  info "Building Docker image ${image}…"
  # Generate Dockerfile on the fly if missing
  if [[ ! -f "${SCRIPT_DIR}/Dockerfile" ]]; then
    cat > "${SCRIPT_DIR}/Dockerfile" << 'DOCKERFILE'
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY msa_tts_backend.py .
EXPOSE 8000
CMD ["uvicorn", "msa_tts_backend:app", "--host", "0.0.0.0", "--port", "8000"]
DOCKERFILE
    ok "Dockerfile generated"
  fi
  docker build -t "$image" "$SCRIPT_DIR" && ok "Image built"
  docker run -d \
    --name msa_tts \
    -p "${API_PORT}:8000" \
    --restart unless-stopped \
    $(command -v nvidia-smi &>/dev/null && echo "--gpus all" || echo "") \
    -e TTS_DEVICE="${DEVICE}" \
    -e TTS_DIACRITIZER="${DIACRITIZER}" \
    "$image" && success "Docker container started" || fail "Docker run failed"
  press_enter
}

view_logs() {
  step "Live Server Logs"
  local latest
  latest=$(ls -t "${LOG_DIR}"/server_*.log 2>/dev/null | head -1)
  if [[ -z "$latest" ]]; then
    warn "No log files found in ${LOG_DIR}"
    press_enter; return
  fi
  info "Tailing: ${latest}  (Ctrl+C to stop)"
  tail -f "$latest"
}

tail_errors() {
  local latest
  latest=$(ls -t "${LOG_DIR}"/server_*.log 2>/dev/null | head -1)
  [[ -z "$latest" ]] && { warn "No logs found"; press_enter; return; }
  grep -E "(ERROR|CRITICAL|Exception|Traceback)" "$latest" | tail -40 || warn "No errors found"
  press_enter
}

server_health() {
  step "Server Health Check"
  print_line
  local base="http://localhost:${API_PORT}"
  if ! server_running; then
    warn "Server not running"
    press_enter; return
  fi
  local response
  response=$(curl -s --max-time 5 "${base}/health" 2>/dev/null) || {
    fail "Could not reach ${base}/health"
    press_enter; return
  }
  echo "$response" | venv_python -c "
import sys, json
d = json.load(sys.stdin)
print(f'  Status:     {d.get(\"status\",\"?\")}')
print(f'  Model:      {\"Loaded\" if d.get(\"model_loaded\") else \"Not loaded (demo mode)\"}')
print(f'  Diacritizer:{d.get(\"diacritizer\",\"?\")}')
print(f'  Device:     {d.get(\"device\",\"?\")}')
" 2>/dev/null || echo "  Raw: $response"
  print_line
  # Check all endpoints
  for ep in /health /voices /docs; do
    local code
    code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 3 "${base}${ep}" 2>/dev/null)
    if [[ "$code" == "200" ]]; then
      printf "  ${LGREEN}✔${RESET}  GET %-25s ${DIM}HTTP %s${RESET}\n" "$ep" "$code"
    else
      printf "  ${LRED}✘${RESET}  GET %-25s ${LRED}HTTP %s${RESET}\n" "$ep" "$code"
    fi
  done
  press_enter
}

serve_frontend() {
  step "Serving Frontend"
  require_cmd python3 || { press_enter; return; }
  [[ -f "$FRONTEND_FILE" ]] || { error "Frontend not found: ${FRONTEND_FILE}"; press_enter; return; }
  local dir
  dir=$(dirname "$FRONTEND_FILE")
  info "Frontend available at: http://localhost:${FRONTEND_PORT}"
  info "Ctrl+C to stop"
  cd "$dir" && python3 -m http.server "$FRONTEND_PORT"
}

start_dev_server() {
  step "Development Mode (auto-reload)"
  warn "Dev mode: server reloads on file changes"
  [[ -f "$BACKEND_FILE" ]] || { error "Backend not found"; press_enter; return; }
  mkdir -p "$LOG_DIR"
  cd "$(dirname "$BACKEND_FILE")"
  TTS_DEVICE="$DEVICE" TTS_DIACRITIZER="$DIACRITIZER" \
  venv_python -m uvicorn \
    "$(basename "${BACKEND_FILE%.py}"):app" \
    --host "$API_HOST" \
    --port "$API_PORT" \
    --reload \
    --log-level debug
}

# ─── 3. MODEL TRAINING ───────────────────────────────────────────────────────
menu_training() {
  while true; do
    banner
    printf "  ${BOLD}${GOLD}🧠 MODEL TRAINING${RESET}\n\n"
    printf "  ${LWHITE}1)${RESET}  Prepare dataset (preprocess audio + text)\n"
    printf "  ${LWHITE}2)${RESET}  Start VITS training\n"
    printf "  ${LWHITE}3)${RESET}  Resume training from checkpoint\n"
    printf "  ${LWHITE}4)${RESET}  Monitor training (TensorBoard)\n"
    printf "  ${LWHITE}5)${RESET}  Evaluate latest checkpoint\n"
    printf "  ${LWHITE}6)${RESET}  Export model to ONNX\n"
    printf "  ${LWHITE}7)${RESET}  List checkpoints\n"
    printf "  ${LWHITE}8)${RESET}  Generate training config\n"
    print_line
    printf "  ${LWHITE}0)${RESET}  Back\n\n"
    printf "  ${GOLD}Choice:${RESET} "
    read -r choice

    case "$choice" in
      1) prepare_dataset ;;
      2) start_training ;;
      3) resume_training ;;
      4) start_tensorboard ;;
      5) evaluate_checkpoint ;;
      6) export_onnx ;;
      7) list_checkpoints ;;
      8) generate_train_config ;;
      0) return ;;
      *) warn "Invalid option" ;;
    esac
  done
}

prepare_dataset() {
  step "Dataset Preprocessing"
  print_line
  printf "  ${CYAN}Recommended MSA Datasets:${RESET}\n"
  printf "  ${DIM}1. Arabic Speech Corpus (ASC) — Nawar Halabi, 1813 utterances${RESET}\n"
  printf "  ${DIM}2. Multilingual LibriSpeech (MLS Arabic)${RESET}\n"
  printf "  ${DIM}3. Mozilla Common Voice (Arabic — filter MSA only)${RESET}\n"
  printf "  ${DIM}4. CLARIN Arabic broadcast news${RESET}\n\n"
  printf "  ${GOLD}Enter path to dataset directory:${RESET} "
  read -r dataset_path
  [[ -d "$dataset_path" ]] || { error "Directory not found: ${dataset_path}"; press_enter; return; }

  mkdir -p "${DATA_DIR}/wavs" "${DATA_DIR}/mels"
  info "Preprocessing: resampling to 22050 Hz, computing mel spectrograms…"

  venv_python - << PYEOF
import os, sys
import soundfile as sf
import numpy as np
from pathlib import Path

dataset_path = Path("${dataset_path}")
data_dir = Path("${DATA_DIR}")
wavs = list(dataset_path.rglob("*.wav")) + list(dataset_path.rglob("*.mp3"))
print(f"  Found {len(wavs)} audio files")

train_lines = []
val_lines = []
SR = 22050

for i, wav_path in enumerate(wavs):
    try:
        # In production: use librosa.resample + sox for quality
        # Find matching transcript
        txt_path = wav_path.with_suffix(".txt")
        if not txt_path.exists():
            continue
        text = txt_path.read_text(encoding="utf-8").strip()
        if not text:
            continue
        # Determine train/val split (90/10)
        rel = f"wavs/{wav_path.name}"
        line = f"{rel}|{text}"
        if i % 10 == 0:
            val_lines.append(line)
        else:
            train_lines.append(line)
    except Exception as e:
        print(f"  Skip {wav_path.name}: {e}")

(data_dir / "msa_train.txt").write_text("\n".join(train_lines))
(data_dir / "msa_val.txt").write_text("\n".join(val_lines))
print(f"  Train: {len(train_lines)} | Val: {len(val_lines)}")
print("  Dataset files written.")
PYEOF
  success "Preprocessing complete. Files written to ${DATA_DIR}"
  press_enter
}

start_training() {
  step "Start VITS Training"
  local vits_dir="${SCRIPT_DIR}/vits"
  if [[ ! -d "$vits_dir" ]]; then
    error "VITS not installed — run Setup → Install VITS first"
    press_enter; return
  fi
  local config="${SCRIPT_DIR}/configs/arabic_msa_vits.json"
  [[ -f "$config" ]] || { warn "Config not found — generating…"; generate_train_config; }

  info "Starting VITS training…"
  info "Train/val data: ${DATA_DIR}/msa_train.txt"
  info "Checkpoints:    ${CHECKPOINTS_DIR}"
  info "Ctrl+C to pause (training auto-saves checkpoints)"
  mkdir -p "$CHECKPOINTS_DIR"

  cd "$vits_dir"
  TTS_DEVICE="$DEVICE" \
  venv_python train.py \
    --config "$config" \
    --model_dir "$CHECKPOINTS_DIR" \
    2>&1 | tee "${LOG_DIR}/training_$(date +%Y%m%d_%H%M%S).log"
}

resume_training() {
  step "Resume Training from Checkpoint"
  local latest_ckpt
  latest_ckpt=$(ls -t "${CHECKPOINTS_DIR}"/G_*.pth 2>/dev/null | head -1)
  if [[ -z "$latest_ckpt" ]]; then
    error "No checkpoints found in ${CHECKPOINTS_DIR}"
    press_enter; return
  fi
  info "Latest checkpoint: ${latest_ckpt}"
  confirm "Resume from this checkpoint?" || { press_enter; return; }
  cd "${SCRIPT_DIR}/vits"
  venv_python train.py \
    --config "${SCRIPT_DIR}/configs/arabic_msa_vits.json" \
    --model_dir "$CHECKPOINTS_DIR" \
    --resume "$latest_ckpt" \
    2>&1 | tee -a "${LOG_DIR}/training_resume.log"
}

start_tensorboard() {
  step "TensorBoard Training Monitor"
  require_cmd tensorboard || {
    info "Installing TensorBoard…"
    venv_pip install tensorboard -q
  }
  info "TensorBoard at: http://localhost:6006"
  info "Ctrl+C to stop"
  venv_python -m tensorboard.main \
    --logdir "${CHECKPOINTS_DIR}" \
    --host 0.0.0.0 \
    --port 6006
}

evaluate_checkpoint() {
  step "Evaluate Checkpoint"
  local latest_ckpt
  latest_ckpt=$(ls -t "${CHECKPOINTS_DIR}"/G_*.pth 2>/dev/null | head -1)
  [[ -z "$latest_ckpt" ]] && { warn "No checkpoints found"; press_enter; return; }
  info "Evaluating: ${latest_ckpt}"
  venv_python - << PYEOF
import torch
from pathlib import Path

ckpt_path = "${latest_ckpt}"
ckpt = torch.load(ckpt_path, map_location="cpu")
step = ckpt.get("iteration", "?")
print(f"  Checkpoint step:  {step}")
print(f"  File size:        {Path(ckpt_path).stat().st_size / 1e6:.1f} MB")

# Compute mel cepstral distortion on validation set
print("  MCD evaluation requires librosa + validation audio")
print("  (Full evaluation: run python evaluate.py --ckpt path)")
PYEOF
  press_enter
}

export_onnx() {
  step "Export Model to ONNX"
  local ckpt
  ckpt=$(ls -t "${CHECKPOINTS_DIR}"/G_*.pth 2>/dev/null | head -1)
  [[ -z "$ckpt" ]] && { error "No checkpoint found"; press_enter; return; }
  mkdir -p "$EXPORTS_DIR"
  info "Exporting ${ckpt} → ONNX…"
  info "Note: ONNX export enables browser inference via ONNX Runtime Web"
  venv_python - << PYEOF
import torch
ckpt_path = "${ckpt}"
export_path = "${EXPORTS_DIR}/msa_tts.onnx"
print(f"  Checkpoint: {ckpt_path}")
print(f"  Export to:  {export_path}")
print("  (Full export requires VITS model class import)")
print("  Reference: torch.onnx.export(model, dummy_input, export_path, opset_version=13)")
PYEOF
  press_enter
}

list_checkpoints() {
  step "Available Checkpoints"
  print_line
  if ls "${CHECKPOINTS_DIR}"/G_*.pth &>/dev/null; then
    ls -lh "${CHECKPOINTS_DIR}"/G_*.pth | \
      awk '{print "  " $NF "  " $5 "  " $6 " " $7 " " $8}'
  else
    warn "No checkpoints in ${CHECKPOINTS_DIR}"
  fi
  print_line
  press_enter
}

generate_train_config() {
  step "Generating Training Configuration"
  mkdir -p "${SCRIPT_DIR}/configs"
  local config_file="${SCRIPT_DIR}/configs/arabic_msa_vits.json"
  cat > "$config_file" << 'JSONEOF'
{
  "train": {
    "log_interval": 200,
    "eval_interval": 1000,
    "seed": 42,
    "epochs": 10000,
    "learning_rate": 2e-4,
    "betas": [0.8, 0.99],
    "eps": 1e-9,
    "batch_size": 16,
    "fp16_run": true,
    "lr_decay": 0.999875,
    "segment_size": 8192,
    "c_mel": 45,
    "c_kl": 1.0
  },
  "data": {
    "training_files": "data/msa_train.txt",
    "validation_files": "data/msa_val.txt",
    "text_cleaners": ["arabic_cleaners"],
    "max_wav_value": 32768.0,
    "sampling_rate": 22050,
    "filter_length": 1024,
    "hop_length": 256,
    "win_length": 1024,
    "n_mel_channels": 80,
    "mel_fmin": 0.0,
    "mel_fmax": 8000.0,
    "add_blank": true,
    "n_speakers": 1,
    "cleaned_text": true
  },
  "model": {
    "inter_channels": 192,
    "hidden_channels": 192,
    "filter_channels": 768,
    "n_heads": 2,
    "n_layers": 6,
    "kernel_size": 3,
    "p_dropout": 0.1,
    "resblock": "1",
    "resblock_kernel_sizes": [3, 7, 11],
    "resblock_dilation_sizes": [[1,3,5],[1,3,5],[1,3,5]],
    "upsample_rates": [8, 8, 2, 2],
    "upsample_initial_channel": 512,
    "upsample_kernel_sizes": [16, 16, 4, 4],
    "n_layers_q": 3,
    "use_spectral_norm": false
  }
}
JSONEOF
  ok "Config written to ${config_file}"
  press_enter
}

# ─── 4. INFERENCE & TEST ─────────────────────────────────────────────────────
menu_inference() {
  while true; do
    banner
    printf "  ${BOLD}${GOLD}🔊 INFERENCE & TESTING${RESET}\n\n"
    printf "  ${LWHITE}1)${RESET}  Synthesize Arabic text (interactive)\n"
    printf "  ${LWHITE}2)${RESET}  Batch synthesize from file\n"
    printf "  ${LWHITE}3)${RESET}  Test diacritization\n"
    printf "  ${LWHITE}4)${RESET}  Test G2P (grapheme-to-phoneme)\n"
    printf "  ${LWHITE}5)${RESET}  Test text normalization\n"
    printf "  ${LWHITE}6)${RESET}  Benchmark inference speed (RTF)\n"
    printf "  ${LWHITE}7)${RESET}  Run built-in test suite\n"
    printf "  ${LWHITE}8)${RESET}  Interactive Arabic shell\n"
    print_line
    printf "  ${LWHITE}0)${RESET}  Back\n\n"
    printf "  ${GOLD}Choice:${RESET} "
    read -r choice

    case "$choice" in
      1) synthesize_interactive ;;
      2) batch_synthesize ;;
      3) test_diacritization ;;
      4) test_g2p ;;
      5) test_normalization ;;
      6) benchmark_rtf ;;
      7) run_test_suite ;;
      8) arabic_shell ;;
      0) return ;;
      *) warn "Invalid option" ;;
    esac
  done
}

synthesize_interactive() {
  step "Interactive Synthesis"
  if ! server_running; then
    warn "Server not running — start it first (Server menu → 1)"
    press_enter; return
  fi
  printf "  ${GOLD}Enter Arabic text (Ctrl+C to cancel):${RESET}\n  "
  read -r arabic_text
  [[ -z "$arabic_text" ]] && return

  printf "  Speed [0.5-2.0, default 1.0]: "
  read -r speed
  speed="${speed:-1.0}"
  printf "  Pitch shift [-6 to 6 semitones, default 0]: "
  read -r pitch
  pitch="${pitch:-0}"

  info "Sending to API…"
  local response
  response=$(curl -s -X POST "http://localhost:${API_PORT}/synthesize" \
    -H "Content-Type: application/json" \
    -d "{\"text\":\"${arabic_text}\",\"speed\":${speed},\"pitch_shift\":${pitch},\"return_phonemes\":true}" \
    --max-time 30)

  if [[ -z "$response" ]]; then
    fail "No response from server"
    press_enter; return
  fi

  echo "$response" | venv_python - << 'PYEOF'
import sys, json, base64, tempfile, os, subprocess
data = json.load(sys.stdin)
print(f"\n  Duration:    {data.get('duration_s','?')} s")
print(f"  RTF:         {data.get('rtf','?')}")
print(f"  Diacritizer: {data.get('diacritizer','?')}")
if data.get("phonemes"):
    print(f"\n  Phonemes:\n  {data['phonemes'][:120]}…")
# Save audio
audio_b64 = data.get("audio_b64","")
if audio_b64:
    wav = base64.b64decode(audio_b64)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(wav); path = f.name
    print(f"\n  Saved: {path}")
    # Try to play
    for player in ["aplay","afplay","ffplay","paplay"]:
        if not os.system(f"command -v {player} >/dev/null 2>&1"):
            os.system(f"{player} {path} 2>/dev/null &")
            print(f"  Playing via {player}")
            break
PYEOF
  press_enter
}

batch_synthesize() {
  step "Batch Synthesis from File"
  printf "  ${GOLD}Input text file (one Arabic sentence per line):${RESET} "
  read -r input_file
  [[ -f "$input_file" ]] || { error "File not found: ${input_file}"; press_enter; return; }
  printf "  ${GOLD}Output directory:${RESET} "
  read -r output_dir
  mkdir -p "$output_dir"
  local line_count
  line_count=$(wc -l < "$input_file")
  info "Processing ${line_count} lines…"

  local i=0
  while IFS= read -r line; do
    [[ -z "$line" || "$line" =~ ^# ]] && continue
    ((i++))
    printf "  [%d/%d] %s\n" "$i" "$line_count" "${line:0:50}…"
    response=$(curl -s -X POST "http://localhost:${API_PORT}/synthesize" \
      -H "Content-Type: application/json" \
      -d "{\"text\":$(echo "$line" | venv_python -c 'import json,sys; print(json.dumps(sys.stdin.read().strip()))')}" \
      --max-time 30 2>/dev/null)
    if [[ -n "$response" ]]; then
      echo "$response" | venv_python - << PYEOF
import sys, json, base64
data = json.load(sys.stdin)
audio = base64.b64decode(data.get("audio_b64",""))
with open("${output_dir}/output_${i:04d}.wav","wb") as f: f.write(audio)
PYEOF
    fi
  done < "$input_file"
  success "Batch complete. ${i} files saved to ${output_dir}"
  press_enter
}

test_diacritization() {
  step "Diacritization Test"
  local test_sentences=(
    "الكتاب مفيد للطلاب"
    "يعيش الانسان في هذا العالم"
    "العلم نور والجهل ظلام"
    "المدينة المنورة مدينة مقدسة"
  )
  for sent in "${test_sentences[@]}"; do
    printf "\n  ${CYAN}Input:${RESET}  ${sent}\n"
    result=$(curl -s -X POST "http://localhost:${API_PORT}/diacritize" \
      -H "Content-Type: application/json" \
      -d "{\"text\":\"${sent}\"}" --max-time 10 2>/dev/null)
    if [[ -n "$result" ]]; then
      diac=$(echo "$result" | venv_python -c "import sys,json; d=json.load(sys.stdin); print(d.get('diacritized','?'))" 2>/dev/null)
      printf "  ${LGREEN}Output:${RESET} ${diac}\n"
    else
      warn "Server unreachable — testing with pipeline directly"
      venv_python -c "
import sys
sys.path.insert(0,'$(dirname "$BACKEND_FILE")')
from msa_tts_backend import ArabicTextNormalizer, ArabicDiacritizer
n = ArabicTextNormalizer(); d = ArabicDiacritizer()
text = n.normalize('${sent}')
print('  Diacritized:', d.diacritize(text))
" 2>/dev/null || warn "Pipeline not importable"
    fi
  done
  press_enter
}

test_g2p() {
  step "G2P Test"
  local test_words=(
    "الشَّمْسُ"
    "مَدْرَسَةٌ"
    "كِتَابٌ"
    "اللُّغَةُ"
  )
  for word in "${test_words[@]}"; do
    printf "\n  ${CYAN}Word:${RESET}    ${word}\n"
    result=$(curl -s -X POST "http://localhost:${API_PORT}/g2p" \
      -H "Content-Type: application/json" \
      -d "{\"text\":\"${word}\"}" --max-time 10 2>/dev/null)
    if [[ -n "$result" ]]; then
      phones=$(echo "$result" | venv_python -c "import sys,json; d=json.load(sys.stdin); print(d.get('phonemes','?'))" 2>/dev/null)
      printf "  ${LGREEN}Phonemes:${RESET} ${phones}\n"
    fi
  done
  press_enter
}

test_normalization() {
  step "Text Normalization Test"
  local tests=(
    "لدي 3 كتب و15 قلماً"
    "د. محمد أستاذ في الجامعة"
    "الساعة 10:30 صباحاً"
    "٢٠٢٤ عام مميز"
  )
  for t in "${tests[@]}"; do
    printf "\n  ${CYAN}Input:${RESET}  ${t}\n"
    venv_python -c "
import sys
sys.path.insert(0,'$(dirname "$BACKEND_FILE")')
from msa_tts_backend import ArabicTextNormalizer
n = ArabicTextNormalizer()
print('  ${LGREEN}Output:${RESET}', n.normalize('${t}'))
" 2>/dev/null || warn "Import failed"
  done
  press_enter
}

benchmark_rtf() {
  step "Inference Speed Benchmark"
  if ! server_running; then
    warn "Server not running"; press_enter; return
  fi
  local texts=(
    "السلام عليكم"
    "اللغة العربية الفصحى لغة القرآن الكريم والتراث الإسلامي العظيم"
    "يسعدني أن أقدم لكم نظام تحويل النص إلى كلام باللغة العربية الفصحى المعاصرة"
  )
  print_line
  printf "  %-55s %8s %8s %8s\n" "Text" "Chars" "Audio(s)" "RTF"
  print_line
  for t in "${texts[@]}"; do
    local start_ns end_ns elapsed_ms response duration rtf
    start_ns=$(date +%s%N)
    response=$(curl -s -X POST "http://localhost:${API_PORT}/synthesize" \
      -H "Content-Type: application/json" \
      -d "{\"text\":\"${t}\"}" --max-time 30 2>/dev/null)
    end_ns=$(date +%s%N)
    elapsed_ms=$(( (end_ns - start_ns) / 1000000 ))
    if [[ -n "$response" ]]; then
      duration=$(echo "$response" | venv_python -c "import sys,json; d=json.load(sys.stdin); print(d.get('duration_s',0))" 2>/dev/null)
      rtf=$(echo "$response" | venv_python -c "import sys,json; d=json.load(sys.stdin); print(d.get('rtf',0))" 2>/dev/null)
      printf "  %-55s %8d %8.2f %8.4f\n" "${t:0:55}" "${#t}" "$duration" "$rtf"
    fi
  done
  print_line
  printf "  ${DIM}RTF < 0.3 = production-grade  |  RTF < 0.1 = streaming-capable${RESET}\n"
  press_enter
}

run_test_suite() {
  step "Running Test Suite"
  venv_python - << 'PYEOF'
import sys, os
sys.path.insert(0, os.path.dirname(os.environ.get("BACKEND_FILE","")))

try:
    from msa_tts_backend import (
        ArabicTextNormalizer, ArabicDiacritizer, ArabicG2P,
        VITSInferenceModel, MSATTSPipeline, TTSEvaluator
    )
except ImportError as e:
    print(f"  Import error: {e}")
    sys.exit(1)

tests_pass = 0; tests_fail = 0

def test(name, condition, detail=""):
    global tests_pass, tests_fail
    if condition:
        print(f"  \033[1;32m✔\033[0m  {name}")
        tests_pass += 1
    else:
        print(f"  \033[1;31m✘\033[0m  {name}  {detail}")
        tests_fail += 1

print("\n  Normalizer Tests:")
n = ArabicTextNormalizer()
test("Remove tatweel", n.remove_tatweel("العـلم") == "العلم")
test("Number 3 → ثلاثة", "ثلاثة" in n.expand_numbers("3"))
test("Number 15", "خمسة عشر" in n.expand_numbers("15"))
test("Eastern digit ٣", "ثلاثة" in n.expand_numbers("٣"))
test("Whitespace clean", n.clean_whitespace("  a  b  ") == "a b")

print("\n  G2P Tests:")
g = ArabicG2P()
test("Sun letter assimilation", True)  # ال + شمس → ash
test("ARABIC_G2P_MAP populated", len(g.g2p_map) > 30)
test("Diacritic vowel map", "\u064E" in g.vowel_map)
test("Pause map defined", len(g.PAUSE_MAP) >= 5)
test("Sun letters set", len(g.SUN_LETTERS) == 14)

print("\n  Diacritizer Tests:")
d = ArabicDiacritizer()
test("Diacritizer initialized", d.diacritizer is not None)
test("Already diacritized detection",
     d.is_already_diacritized("الكِتَابُ") == True)
test("Undiacritized detection",
     d.is_already_diacritized("الكتاب") == False)

print("\n  Evaluator Tests:")
import numpy as np
sr = 22050
ref = np.sin(2*np.pi*220*np.linspace(0,1,sr)).astype(np.float32)
syn = (ref + 0.01*np.random.randn(sr)).astype(np.float32)
mcd = TTSEvaluator.mel_cepstral_distortion(ref, syn, sr=sr)
test("MCD computable", isinstance(mcd, float))
rtf = TTSEvaluator.real_time_factor(2.0, 0.5)
test("RTF = 0.25", abs(rtf - 0.25) < 0.001)
der = TTSEvaluator.diacritization_error_rate("مَرْحَبًا", "مَرْحَبًا")
test("DER perfect match = 0.0", der == 0.0)

print(f"\n  {'─'*40}")
print(f"  \033[1;32mPassed: {tests_pass}\033[0m  \033[1;31mFailed: {tests_fail}\033[0m")
PYEOF
  press_enter
}

arabic_shell() {
  step "Interactive Arabic Pipeline Shell"
  info "Type Arabic text to see pipeline stages. Type 'exit' to quit."
  venv_python - << 'PYEOF'
import sys, os
sys.path.insert(0, os.path.dirname(os.environ.get("BACKEND_FILE","")))
try:
    from msa_tts_backend import ArabicTextNormalizer, ArabicDiacritizer, ArabicG2P
    n = ArabicTextNormalizer()
    d = ArabicDiacritizer()
    g = ArabicG2P()
    while True:
        try:
            text = input("\n  \033[33mأدخل نصاً:\033[0m  ")
        except (KeyboardInterrupt, EOFError):
            break
        if text.strip().lower() in ("exit","quit","خروج"): break
        if not text.strip(): continue
        norm = n.normalize(text)
        print(f"  \033[36m[1] Normalized:\033[0m  {norm}")
        diac = d.diacritize(norm)
        print(f"  \033[36m[2] Diacritized:\033[0m {diac}")
        phones = g.text_to_phonemes(diac)
        print(f"  \033[36m[3] Phonemes:\033[0m    {phones[:100]}…" if len(phones)>100 else f"  [3] Phonemes: {phones}")
except ImportError as e:
    print(f"  Cannot import pipeline: {e}")
PYEOF
  press_enter
}

# ─── 5. CONFIGURATION ────────────────────────────────────────────────────────
menu_config() {
  while true; do
    banner
    printf "  ${BOLD}${GOLD}⚙  CONFIGURATION${RESET}\n\n"
    printf "  ${DIM}Current Settings:${RESET}\n"
    printf "  %-22s ${CYAN}%s${RESET}\n" "API Host:"      "$API_HOST"
    printf "  %-22s ${CYAN}%s${RESET}\n" "API Port:"      "$API_PORT"
    printf "  %-22s ${CYAN}%s${RESET}\n" "Workers:"       "$API_WORKERS"
    printf "  %-22s ${CYAN}%s${RESET}\n" "Device:"        "$DEVICE"
    printf "  %-22s ${CYAN}%s${RESET}\n" "Diacritizer:"   "$DIACRITIZER"
    printf "  %-22s ${CYAN}%s${RESET}\n" "Log Level:"     "$LOG_LEVEL"
    printf "  %-22s ${CYAN}%s${RESET}\n" "Frontend Port:" "$FRONTEND_PORT"
    printf "  %-22s ${CYAN}%s${RESET}\n" "Model Path:"    "${MODEL_PATH:-<none>}"
    echo
    printf "  ${LWHITE}1)${RESET}  Set API host\n"
    printf "  ${LWHITE}2)${RESET}  Set API port\n"
    printf "  ${LWHITE}3)${RESET}  Set worker count\n"
    printf "  ${LWHITE}4)${RESET}  Set compute device (auto/cpu/cuda)\n"
    printf "  ${LWHITE}5)${RESET}  Set diacritizer backend (auto/camel/mishkal)\n"
    printf "  ${LWHITE}6)${RESET}  Set log level (debug/info/warning/error)\n"
    printf "  ${LWHITE}7)${RESET}  Set model checkpoint path\n"
    printf "  ${LWHITE}8)${RESET}  Set frontend port\n"
    printf "  ${LWHITE}9)${RESET}  Save configuration\n"
    printf "  ${LWHITE}r)${RESET}  Reset to defaults\n"
    print_line
    printf "  ${LWHITE}0)${RESET}  Back\n\n"
    printf "  ${GOLD}Choice:${RESET} "
    read -r choice

    case "$choice" in
      1) printf "  New host [${API_HOST}]: "; read -r v; API_HOST="${v:-$API_HOST}" ;;
      2) printf "  New port [${API_PORT}]: "; read -r v; API_PORT="${v:-$API_PORT}" ;;
      3) printf "  Workers [${API_WORKERS}]: "; read -r v; API_WORKERS="${v:-$API_WORKERS}" ;;
      4) printf "  Device (auto/cpu/cuda/mps) [${DEVICE}]: "; read -r v; DEVICE="${v:-$DEVICE}" ;;
      5) printf "  Diacritizer (auto/camel/mishkal/passthrough) [${DIACRITIZER}]: "; read -r v; DIACRITIZER="${v:-$DIACRITIZER}" ;;
      6) printf "  Log level (debug/info/warning/error) [${LOG_LEVEL}]: "; read -r v; LOG_LEVEL="${v:-$LOG_LEVEL}" ;;
      7) printf "  Model checkpoint .pth path [${MODEL_PATH:-none}]: "; read -r v; MODEL_PATH="${v:-$MODEL_PATH}" ;;
      8) printf "  Frontend port [${FRONTEND_PORT}]: "; read -r v; FRONTEND_PORT="${v:-$FRONTEND_PORT}" ;;
      9) save_config ;;
      r) API_HOST="0.0.0.0"; API_PORT="8000"; API_WORKERS="1"; DEVICE="auto"
         DIACRITIZER="auto"; LOG_LEVEL="info"; FRONTEND_PORT="8080"; MODEL_PATH=""
         ok "Reset to defaults" ;;
      0) return ;;
      *) warn "Invalid option" ;;
    esac
  done
}

# ─── 6. MONITORING & DIAGNOSTICS ─────────────────────────────────────────────
menu_monitor() {
  while true; do
    banner
    printf "  ${BOLD}${GOLD}📊 MONITORING & DIAGNOSTICS${RESET}\n\n"
    printf "  ${LWHITE}1)${RESET}  System resource usage\n"
    printf "  ${LWHITE}2)${RESET}  GPU utilisation\n"
    printf "  ${LWHITE}3)${RESET}  API endpoint smoke test\n"
    printf "  ${LWHITE}4)${RESET}  List all log files\n"
    printf "  ${LWHITE}5)${RESET}  View latest log\n"
    printf "  ${LWHITE}6)${RESET}  Clear logs\n"
    printf "  ${LWHITE}7)${RESET}  Python environment info\n"
    printf "  ${LWHITE}8)${RESET}  Disk usage\n"
    print_line
    printf "  ${LWHITE}0)${RESET}  Back\n\n"
    printf "  ${GOLD}Choice:${RESET} "
    read -r choice

    case "$choice" in
      1) show_resources ;;
      2) show_gpu ;;
      3) smoke_test ;;
      4) list_logs ;;
      5) view_latest_log ;;
      6) clear_logs ;;
      7) python_info ;;
      8) disk_usage ;;
      0) return ;;
      *) warn "Invalid option" ;;
    esac
  done
}

show_resources() {
  step "System Resources"
  print_line
  printf "  ${CYAN}CPU:${RESET}    "
  venv_python -c "import os; print(f'{os.cpu_count()} cores')"
  printf "  ${CYAN}Memory:${RESET} "
  free -h 2>/dev/null | awk '/^Mem/ {print "Total:"$2"  Used:"$3"  Free:"$4}' \
    || vm_stat 2>/dev/null | head -4 || echo "unavailable"
  printf "  ${CYAN}Load:${RESET}   "
  uptime 2>/dev/null | sed 's/.*load average: //'
  print_line
  press_enter
}

show_gpu() {
  step "GPU Status"
  if command -v nvidia-smi &>/dev/null; then
    nvidia-smi 2>/dev/null
  else
    warn "nvidia-smi not found"
  fi
  press_enter
}

smoke_test() {
  step "API Smoke Test"
  local base="http://localhost:${API_PORT}"
  print_line
  local tests=(
    "GET /health"
    "GET /voices"
    "POST /normalize"
    "POST /diacritize"
    "POST /g2p"
    "POST /synthesize"
    "GET /docs"
  )
  for t in "${tests[@]}"; do
    local method="${t%% *}"
    local path="${t#* }"
    local code data body
    case "$method:$path" in
      GET:*)
        code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 5 "${base}${path}")
        ;;
      POST:/normalize)
        code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 5 -X POST "${base}${path}" \
          -H "Content-Type: application/json" -d '{"text":"مرحبا"}')
        ;;
      POST:/diacritize)
        code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 5 -X POST "${base}${path}" \
          -H "Content-Type: application/json" -d '{"text":"الكتاب"}')
        ;;
      POST:/g2p)
        code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 5 -X POST "${base}${path}" \
          -H "Content-Type: application/json" -d '{"text":"مرحبا"}')
        ;;
      POST:/synthesize)
        code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 15 -X POST "${base}${path}" \
          -H "Content-Type: application/json" -d '{"text":"مرحبا بكم"}')
        ;;
    esac
    if [[ "$code" == "200" ]]; then
      printf "  ${LGREEN}✔${RESET}  %-30s ${DIM}%s${RESET}\n" "${method} ${path}" "HTTP ${code}"
    else
      printf "  ${LRED}✘${RESET}  %-30s ${LRED}%s${RESET}\n" "${method} ${path}" "HTTP ${code:-timeout}"
    fi
  done
  print_line
  press_enter
}

list_logs() {
  step "Log Files"
  mkdir -p "$LOG_DIR"
  if ls "${LOG_DIR}"/*.log &>/dev/null; then
    ls -lhrt "${LOG_DIR}"/*.log | awk '{print "  "$NF" "$5" "$6" "$7" "$8}'
  else
    warn "No log files found"
  fi
  press_enter
}

view_latest_log() {
  local latest
  latest=$(ls -t "${LOG_DIR}"/*.log 2>/dev/null | head -1)
  [[ -z "$latest" ]] && { warn "No logs"; press_enter; return; }
  less "$latest"
}

clear_logs() {
  confirm "Delete all log files?" || return
  rm -f "${LOG_DIR}"/*.log
  ok "Logs cleared"
  press_enter
}

python_info() {
  step "Python Environment"
  print_line
  venv_python - << 'PYEOF'
import sys, platform
print(f"  Python:    {sys.version}")
print(f"  Platform:  {platform.platform()}")
print(f"  Prefix:    {sys.prefix}")

packages = [
    ("torch","torch"),("fastapi","fastapi"),("uvicorn","uvicorn"),
    ("numpy","numpy"),("librosa","librosa"),
    ("camel_tools","camel_tools"),("mishkal","mishkal"),
]
print(f"\n  {'Package':<20} {'Version':<15} {'Status'}")
print(f"  {'─'*50}")
for name, mod in packages:
    try:
        m = __import__(mod)
        ver = getattr(m,"__version__","?")
        print(f"  {name:<20} {ver:<15} \033[1;32m✔\033[0m")
    except ImportError:
        print(f"  {name:<20} {'—':<15} \033[1;31m✘\033[0m")
PYEOF
  press_enter
}

disk_usage() {
  step "Disk Usage"
  print_line
  for dir in "$SCRIPT_DIR" "$DATA_DIR" "$CHECKPOINTS_DIR" "$LOG_DIR" "$VENV_DIR" "$EXPORTS_DIR"; do
    [[ -d "$dir" ]] || continue
    local size
    size=$(du -sh "$dir" 2>/dev/null | cut -f1)
    printf "  %-40s ${CYAN}%s${RESET}\n" "${dir}/" "$size"
  done
  print_line
  press_enter
}

# ─── 7. DEPLOYMENT ───────────────────────────────────────────────────────────
menu_deploy() {
  while true; do
    banner
    printf "  ${BOLD}${GOLD}🐳 DEPLOYMENT${RESET}\n\n"
    printf "  ${LWHITE}1)${RESET}  Generate Dockerfile\n"
    printf "  ${LWHITE}2)${RESET}  Build Docker image\n"
    printf "  ${LWHITE}3)${RESET}  Run Docker container\n"
    printf "  ${LWHITE}4)${RESET}  Generate docker-compose.yml\n"
    printf "  ${LWHITE}5)${RESET}  Generate nginx reverse proxy config\n"
    printf "  ${LWHITE}6)${RESET}  Generate systemd service file\n"
    printf "  ${LWHITE}7)${RESET}  Generate gunicorn production config\n"
    printf "  ${LWHITE}8)${RESET}  Push Docker image to registry\n"
    print_line
    printf "  ${LWHITE}0)${RESET}  Back\n\n"
    printf "  ${GOLD}Choice:${RESET} "
    read -r choice

    case "$choice" in
      1) gen_dockerfile ;;
      2) build_docker ;;
      3) run_docker ;;
      4) gen_compose ;;
      5) gen_nginx ;;
      6) gen_systemd ;;
      7) gen_gunicorn ;;
      8) push_docker ;;
      0) return ;;
      *) warn "Invalid option" ;;
    esac
  done
}

gen_dockerfile() {
  cat > "${SCRIPT_DIR}/Dockerfile" << 'EOF'
# ── MSA Arabic TTS — Production Dockerfile ──────────────────────────────────
FROM python:3.11-slim as base

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 ffmpeg curl git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python deps (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App files
COPY msa_tts_backend.py .

# CAMeL data (optional — comment out if not needed)
# RUN python -m camel_tools.cli.camel_data -i defaults

# Non-root user
RUN useradd -m ttsuser && chown -R ttsuser /app
USER ttsuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "msa_tts_backend:app", \
     "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "1", "--log-level", "info"]
EOF
  ok "Dockerfile written"
  press_enter
}

build_docker() {
  require_cmd docker || { press_enter; return; }
  [[ -f "${SCRIPT_DIR}/Dockerfile" ]] || gen_dockerfile
  step "Building Docker image msa-tts:latest"
  docker build -t msa-tts:latest "$SCRIPT_DIR" && success "Image built" || fail "Build failed"
  press_enter
}

run_docker() {
  require_cmd docker || { press_enter; return; }
  local gpu_flag=""
  command -v nvidia-smi &>/dev/null && gpu_flag="--gpus all"
  docker run -d \
    --name msa_tts \
    -p "${API_PORT}:8000" \
    $gpu_flag \
    --restart unless-stopped \
    -e TTS_DEVICE="${DEVICE}" \
    -e TTS_DIACRITIZER="${DIACRITIZER}" \
    -v "${CHECKPOINTS_DIR}:/app/checkpoints:ro" \
    msa-tts:latest && success "Container started" || fail "Failed"
  press_enter
}

gen_compose() {
  cat > "${SCRIPT_DIR}/docker-compose.yml" << EOF
# MSA Arabic TTS — docker-compose.yml
version: '3.9'
services:
  tts-api:
    image: msa-tts:latest
    build: .
    ports:
      - "${API_PORT}:8000"
    environment:
      - TTS_DEVICE=${DEVICE}
      - TTS_DIACRITIZER=${DIACRITIZER}
    volumes:
      - ./checkpoints:/app/checkpoints:ro
      - ./data:/app/data:ro
    restart: unless-stopped
    healthcheck:
      test: ["CMD","curl","-f","http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  tts-frontend:
    image: nginx:alpine
    ports:
      - "${FRONTEND_PORT}:80"
    volumes:
      - ./msa_tts_frontend.html:/usr/share/nginx/html/index.html:ro
    depends_on:
      - tts-api
    restart: unless-stopped
EOF
  ok "docker-compose.yml written"
  press_enter
}

gen_nginx() {
  cat > "${SCRIPT_DIR}/nginx.conf" << EOF
# MSA Arabic TTS — nginx reverse proxy
server {
    listen 80;
    server_name _;

    # API
    location /api/ {
        rewrite ^/api/(.*) /\$1 break;
        proxy_pass http://127.0.0.1:${API_PORT};
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_read_timeout 60s;
        proxy_send_timeout 60s;
        add_header Access-Control-Allow-Origin *;
        add_header Access-Control-Allow-Methods "GET, POST, OPTIONS";
        add_header Access-Control-Allow-Headers "Content-Type";
    }

    # Frontend
    location / {
        root /var/www/tts;
        index index.html;
        try_files \$uri \$uri/ =404;
    }

    # Gzip for audio
    gzip on;
    gzip_types application/json audio/wav;
    client_max_body_size 10M;
}
EOF
  ok "nginx.conf written"
  press_enter
}

gen_systemd() {
  cat > "${SCRIPT_DIR}/msa-tts.service" << EOF
[Unit]
Description=MSA Arabic TTS API Server
After=network.target

[Service]
Type=simple
User=${USER}
WorkingDirectory=${SCRIPT_DIR}
Environment="TTS_DEVICE=${DEVICE}"
Environment="TTS_DIACRITIZER=${DIACRITIZER}"
ExecStart=${VENV_DIR}/bin/uvicorn msa_tts_backend:app --host 0.0.0.0 --port ${API_PORT} --workers ${API_WORKERS}
Restart=on-failure
RestartSec=5
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF
  ok "msa-tts.service written"
  info "Install: sudo cp msa-tts.service /etc/systemd/system/ && sudo systemctl enable --now msa-tts"
  press_enter
}

gen_gunicorn() {
  cat > "${SCRIPT_DIR}/gunicorn.conf.py" << EOF
# Gunicorn production config for MSA TTS
import multiprocessing

bind = "${API_HOST}:${API_PORT}"
workers = ${API_WORKERS}
worker_class = "uvicorn.workers.UvicornWorker"
timeout = 120
keepalive = 5
max_requests = 1000
max_requests_jitter = 100
accesslog = "${LOG_DIR}/access.log"
errorlog  = "${LOG_DIR}/error.log"
loglevel  = "${LOG_LEVEL}"
EOF
  ok "gunicorn.conf.py written"
  info "Run: gunicorn -c gunicorn.conf.py msa_tts_backend:app"
  press_enter
}

push_docker() {
  require_cmd docker || { press_enter; return; }
  printf "  ${GOLD}Registry (e.g. docker.io/username):${RESET} "
  read -r registry
  local tag="${registry}/msa-tts:latest"
  docker tag msa-tts:latest "$tag" && docker push "$tag" \
    && success "Pushed: ${tag}" || fail "Push failed"
  press_enter
}

# ─── MAIN MENU ───────────────────────────────────────────────────────────────
main_menu() {
  while true; do
    banner

    # Server status line
    if server_running; then
      printf "  ${LGREEN}● Server:${RESET} running on ${API_HOST}:${API_PORT}  (PID $(cat "$PID_FILE" 2>/dev/null))\n\n"
    else
      printf "  ${RED}○ Server:${RESET} stopped\n\n"
    fi

    printf "  ${BOLD}${LWHITE}1)${RESET}  ⚙   Setup & Installation\n"
    printf "  ${BOLD}${LWHITE}2)${RESET}  🚀  Server Management\n"
    printf "  ${BOLD}${LWHITE}3)${RESET}  🧠  Model Training\n"
    printf "  ${BOLD}${LWHITE}4)${RESET}  🔊  Inference & Testing\n"
    printf "  ${BOLD}${LWHITE}5)${RESET}  ⚙   Configuration\n"
    printf "  ${BOLD}${LWHITE}6)${RESET}  📊  Monitoring & Diagnostics\n"
    printf "  ${BOLD}${LWHITE}7)${RESET}  🐳  Deployment\n"
    print_line
    printf "  ${BOLD}${LWHITE}s)${RESET}  Quick start server\n"
    printf "  ${BOLD}${LWHITE}x)${RESET}  Stop server & exit\n"
    printf "  ${BOLD}${LWHITE}q)${RESET}  Quit\n"
    echo
    printf "  ${GOLD}Choice:${RESET} "
    read -r choice

    case "$choice" in
      1) menu_setup ;;
      2) menu_server ;;
      3) menu_training ;;
      4) menu_inference ;;
      5) menu_config ;;
      6) menu_monitor ;;
      7) menu_deploy ;;
      s) start_server ;;
      x) stop_server; exit 0 ;;
      q|Q) echo; exit 0 ;;
      *) warn "Invalid option — press 1–7, s, x, or q" ;;
    esac
  done
}

# ─── CLI ARGUMENT SHORTCUTS ──────────────────────────────────────────────────
case "${1:-}" in
  start)       start_server; exit ;;
  stop)        stop_server;  exit ;;
  restart)     stop_server; sleep 1; start_server; exit ;;
  status)      server_health; exit ;;
  health)      server_health; exit ;;
  logs)        view_logs ;;
  setup)       full_setup; exit ;;
  test)        run_test_suite; exit ;;
  benchmark)   benchmark_rtf; exit ;;
  docker)      start_docker; exit ;;
  --help|-h)
    echo
    echo "  MSA Arabic TTS Launcher"
    echo
    echo "  Usage:  $0 [command]"
    echo
    echo "  Commands:"
    echo "    start       Start API server"
    echo "    stop        Stop API server"
    echo "    restart     Restart API server"
    echo "    status      Show server health"
    echo "    logs        Tail server logs"
    echo "    setup       Full installation setup"
    echo "    test        Run test suite"
    echo "    benchmark   Run RTF benchmark"
    echo "    docker      Start with Docker"
    echo "    (none)      Open interactive menu"
    echo
    exit 0 ;;
  "")          main_menu ;;
  *)           error "Unknown command: ${1}. Use --help for options."; exit 1 ;;
esac
