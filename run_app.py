"""Entry point: launches the Gradio segmentation app.

Run with: python run_app.py
"""

from __future__ import annotations

from kpi.utils.logconfig import setup_logging

from app.ui.gradio_app import build_app

if __name__ == "__main__":
    setup_logging("INFO")
    build_app().launch()
