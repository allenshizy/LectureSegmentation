## MA-Zhaoyi Shi

# running guide
- The pytorch version used in this project do not support blackwell GPU.

> For rerunning the experiment or edit the model, please refer to this detailed README: [ARCHITECTURE.md](ARCHITECTURE.md).

## Setup

Most dependencies are managed by `uv` (Python). A handful of things must be installed at the
system level first, outside of Python: **ffmpeg**, **ollama** (only needed for the demo app),
and **uv** itself (bootstrapped via `pip`).

### 1. System-level dependencies

Run the setup script for your OS (safe to re-run; it skips anything already installed):

- Windows (PowerShell): `.\setup.ps1`
- Linux / macOS: `bash setup.sh`

Or install manually:

**Linux (apt-based, e.g. Ubuntu/Debian):**
```bash
sudo apt-get update && sudo apt-get install -y ffmpeg
curl -fsSL https://ollama.com/install.sh | sh
pip install uv   # if pip itself is missing: python3 -m ensurepip --upgrade
```

**macOS:**
```bash
brew install ffmpeg
brew install ollama          # or: curl -fsSL https://ollama.com/install.sh | sh
pip install uv
```

**Windows (PowerShell):**
```powershell
winget install --id Gyan.FFmpeg -e
winget install --id Ollama.Ollama -e
pip install uv               # if pip is missing: python -m ensurepip --upgrade
```

### 2. Python dependencies (uv)

This repo runs fine on CPU; GPU is optional and only needed to reproduce accelerated training.

```bash
uv sync                  # base deps (CPU-only, no GPU required)
uv sync --extra app      # + gradio/faster-whisper, needed to run the Gradio demo app
uv sync --extra gpu      # + cupy-cuda12x, only needed to reproduce GPU-accelerated training
```

Extras can be combined, e.g. `uv sync --extra app --extra gpu`.

### 3. First run of the demo app (Whisper + STB + Qwen/Ollama)

```bash
uv run python run_app.py
```

On first run this will automatically:
- Start `ollama serve` if it isn't already running.
- Pull the default Qwen model (`qwen3:4b`, ~2.6GB) if you don't have it yet.
- Download and cache the `all-MiniLM-L6-v2` sentence encoder (~90MB) and the `faster-whisper` "small" ASR model (~500MB).

All of this requires internet access **only on first run** and can take several minutes depending
on your connection; everything is cached afterwards (`~/.ollama`, `~/.cache/huggingface`). The
trained STB checkpoints (`transformer.pt`, `detector.pt`) already ship in [app/checkpoints/](app/checkpoints/),
so no extra download/training is needed for segmentation itself.

Open the printed local URL(By default it's 127.0.0.1:7860. If the cli is not giving results, sometimes it's just lagging, you can just enter the url), and either:
- Paste a **local audio/video file path** (e.g., `/path/to/lecture.mp4`)
- Paste a **YouTube URL** (e.g., `https://www.youtube.com/watch?v=VIDEO_ID`)

The app will automatically:
- For YouTube URLs: check the video license, extract subtitles if available (skipping Whisper), or download audio and run Whisper if no subtitles exist.
- For local files: run the standard Whisper → STB → Qwen pipeline.

The pipeline may take about 7-20 min to process a lecture of 30-90 min. Most of the time is used for whisper, segmentation and generation usually only takes 1-2 min.

### 4. Cleanup

**Repo-internal** (removes `.venv`, `__pycache__`, `.pytest_cache` — safe/reversible, just `uv sync` again):
- Windows: `.\clean.ps1`
- Linux/macOS: `bash clean.sh`

**System-level** (only if you want to fully remove things from your machine):
```bash
# Linux (apt)
sudo apt-get remove -y ffmpeg
pip uninstall uv

# macOS (brew)
brew uninstall ffmpeg ollama

# Windows
winget uninstall Gyan.FFmpeg
winget uninstall Ollama.Ollama
pip uninstall uv
```
Ollama's downloaded models and caches live outside the repo in your user profile
(`~/.ollama` on Linux/macOS, `%USERPROFILE%\.ollama` on Windows) — remove that directory too if
you want to reclaim the disk space from pulled models.