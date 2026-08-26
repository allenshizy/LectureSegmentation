@echo off
setlocal
cd /d "%~dp0"
where py >nul 2>nul
if %errorlevel%==0 (
  start "Lecture Segmentation Study" "http://127.0.0.1:8000/index.html"
  py -m http.server 8000
) else (
  echo Python was not found. Install Python 3, then run this file again.
  pause
)