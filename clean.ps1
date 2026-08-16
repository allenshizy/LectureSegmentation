# Removes repo-internal build/cache artifacts (safe, reversible: just re-run `uv sync`).
# Does NOT touch system-level installs (ffmpeg/ollama/uv) or model caches in your user profile
# (%USERPROFILE%\.ollama, %USERPROFILE%\.cache\huggingface) — see README "Cleanup" section.
#
# Usage: powershell -ExecutionPolicy Bypass -File clean.ps1

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $repoRoot

Write-Host "Removing .venv/"
Remove-Item -Recurse -Force -ErrorAction SilentlyContinue ".venv"

Write-Host "Removing __pycache__ directories"
Get-ChildItem -Path . -Recurse -Directory -Filter "__pycache__" -ErrorAction SilentlyContinue |
    ForEach-Object { Remove-Item -Recurse -Force $_.FullName }

Write-Host "Removing .pytest_cache/"
Remove-Item -Recurse -Force -ErrorAction SilentlyContinue ".pytest_cache"

Write-Host "Done. Run 'uv sync' (or with --extra app / --extra gpu) to recreate the environment."
