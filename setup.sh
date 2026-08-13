#!/usr/bin/env bash
# One-time system-level setup for Linux/macOS.
# Installs: ffmpeg (media processing), ollama (local LLM runtime for the demo app),
# uv (Python package/venv manager used by this repo instead of raw pip).
#
# Usage: bash setup.sh
set -euo pipefail

os="$(uname -s)"

echo "==> [1/3] ffmpeg"
if command -v ffmpeg >/dev/null 2>&1; then
    echo "    already installed: $(ffmpeg -version | head -n1)"
elif [[ "$os" == "Linux" ]]; then
    sudo apt-get update && sudo apt-get install -y ffmpeg
elif [[ "$os" == "Darwin" ]]; then
    command -v brew >/dev/null 2>&1 || { echo "Homebrew not found; install it from https://brew.sh first." >&2; exit 1; }
    brew install ffmpeg
else
    echo "Unsupported OS '$os' for automated ffmpeg install; install manually: https://ffmpeg.org/download.html" >&2
    exit 1
fi

echo "==> [2/3] ollama (only needed for the Gradio demo app, app/)"
if command -v ollama >/dev/null 2>&1; then
    echo "    already installed: $(ollama --version)"
elif [[ "$os" == "Darwin" ]] && command -v brew >/dev/null 2>&1; then
    brew install ollama
else
    # Official installer script, works on Linux and macOS.
    curl -fsSL https://ollama.com/install.sh | sh
fi

echo "==> [3/3] uv (Python dependency manager for this repo)"
if command -v uv >/dev/null 2>&1; then
    echo "    already installed: $(uv --version)"
elif command -v pip >/dev/null 2>&1; then
    pip install uv
elif command -v pip3 >/dev/null 2>&1; then
    pip3 install uv
else
    # No pip on PATH at all: bootstrap uv directly via its official installer.
    echo "    pip not found, using uv's standalone installer instead"
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi

cat <<'EOF'

System-level setup complete. Next, install the Python dependencies with uv:
  uv sync                 # base deps (CPU-only, no GPU required)
  uv sync --extra app      # + gradio/faster-whisper, needed to run the Gradio demo app
  uv sync --extra gpu      # + cupy-cuda12x, only needed to reproduce GPU-accelerated training
(extras can be combined, e.g. `uv sync --extra app --extra gpu`)
EOF
