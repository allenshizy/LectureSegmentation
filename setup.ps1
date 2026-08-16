# One-time system-level setup for Windows.
# Installs: ffmpeg (media processing), ollama (local LLM runtime for the demo app),
# uv (Python package/venv manager used by this repo instead of raw pip).
#
# Usage: powershell -ExecutionPolicy Bypass -File setup.ps1

function Test-CommandExists {
    param([string]$Name)
    return [bool](Get-Command $Name -ErrorAction SilentlyContinue)
}

Write-Host "==> [1/3] ffmpeg"
if (Test-CommandExists ffmpeg) {
    Write-Host "    already installed"
} elseif (Test-CommandExists winget) {
    winget install --id Gyan.FFmpeg -e --source winget --accept-package-agreements --accept-source-agreements
} elseif (Test-CommandExists choco) {
    choco install ffmpeg -y
} else {
    Write-Warning "Neither winget nor choco found. Install ffmpeg manually: https://ffmpeg.org/download.html"
}

Write-Host "==> [2/3] ollama (only needed for the Gradio demo app, app/)"
if (Test-CommandExists ollama) {
    Write-Host "    already installed"
} elseif (Test-CommandExists winget) {
    winget install --id Ollama.Ollama -e --source winget --accept-package-agreements --accept-source-agreements
} else {
    Write-Warning "winget not found. Download/install manually: https://ollama.com/download"
}

Write-Host "==> [3/3] uv (Python dependency manager for this repo)"
if (Test-CommandExists uv) {
    Write-Host "    already installed"
} elseif (Test-CommandExists pip) {
    pip install uv
} elseif (Test-CommandExists python) {
    # No pip available under this interpreter yet; bootstrap it first.
    python -m ensurepip --upgrade
    python -m pip install uv
} else {
    Write-Host "    pip/python not found, using uv's standalone installer instead"
    irm https://astral.sh/uv/install.ps1 | iex
}

Write-Host ""
Write-Host "System-level setup complete. Next, install the Python dependencies with uv:"
Write-Host "  uv sync                 # base deps (CPU-only, no GPU required)"
Write-Host "  uv sync --extra app      # + gradio/faster-whisper, needed to run the Gradio demo app"
Write-Host "  uv sync --extra gpu      # + cupy-cuda12x, only needed to reproduce GPU-accelerated training"
Write-Host "(extras can be combined, e.g. ``uv sync --extra app --extra gpu``)"
