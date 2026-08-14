"""Entry point: launches the Gradio segmentation app.

Run with: python run_app.py
"""

from __future__ import annotations

from kpi.utils.logconfig import setup_logging

from app.ui.gradio_app import build_app

if __name__ == "__main__":
    setup_logging("INFO")
    print("Starting Gradio. Open http://127.0.0.1:7860 in your browser; the terminal stays occupied while the app is running.")
    print("Models are loaded only after you click 'Run pipeline'. The first run may download Whisper, SBERT, or Qwen.")
    build_app().launch()
