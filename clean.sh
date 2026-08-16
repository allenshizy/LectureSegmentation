#!/usr/bin/env bash
# Removes repo-internal build/cache artifacts (safe, reversible: just re-run `uv sync`).
# Does NOT touch system-level installs (ffmpeg/ollama/uv) or model caches in your home dir
# (~/.ollama, ~/.cache/huggingface) — see README "Cleanup" section to remove those too.
#
# Usage: bash clean.sh
set -euo pipefail

cd "$(dirname "$0")"

echo "Removing .venv/"
rm -rf .venv

echo "Removing __pycache__ directories"
find . -type d -name "__pycache__" -prune -exec rm -rf {} +

echo "Removing .pytest_cache/"
rm -rf .pytest_cache

echo "Done. Run 'uv sync' (or with --extra app / --extra gpu) to recreate the environment."
