from __future__ import annotations

import logging
import shutil
import socket
import subprocess
import time

import requests

from app.config import OllamaConfig

logger = logging.getLogger(__name__)


class OllamaManager:
    """Ensures `ollama serve` is running locally, starting it if needed."""

    def __init__(self, config: OllamaConfig | None = None) -> None:
        self.config = config or OllamaConfig()
        self._proc: subprocess.Popen | None = None

    def _is_up(self) -> bool:
        try:
            with socket.create_connection((self.config.host, self.config.port), timeout=1.0):
                return True
        except OSError:
            return False

    def ensure_running(self) -> None:
        if self._is_up():
            return
        if not self.config.auto_start:
            raise RuntimeError(
                f"Ollama is not reachable at {self.config.base_url} and auto_start is disabled."
            )
        if shutil.which("ollama") is None:
            raise RuntimeError("`ollama` executable not found on PATH. Install it or start it manually.")

        logger.info("Starting `ollama serve` (not detected on %s)", self.config.base_url)
        self._proc = subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        deadline = time.monotonic() + self.config.startup_timeout_s
        while time.monotonic() < deadline:
            if self._is_up():
                logger.info("Ollama server is up at %s", self.config.base_url)
                return
            time.sleep(0.5)
        raise RuntimeError(f"Timed out waiting for ollama serve to become reachable at {self.config.base_url}")

    def stop(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            self._proc.terminate()
        self._proc = None


class OllamaClient:
    """Minimal HTTP client for the local Ollama `/api/generate` endpoint."""

    def __init__(self, config: OllamaConfig | None = None, manager: OllamaManager | None = None) -> None:
        self.config = config or OllamaConfig()
        self.manager = manager or OllamaManager(self.config)

    def generate(self, prompt: str) -> str:
        self.manager.ensure_running()
        response = requests.post(
            f"{self.config.base_url}/api/generate",
            json={"model": self.config.model, "prompt": prompt, "stream": False},
            timeout=self.config.request_timeout_s,
        )
        response.raise_for_status()
        return response.json()["response"].strip()

    def title_chapter(self, chapter_text: str) -> str:
        prompt = (
            "Give a short (<=8 words) chapter title summarizing the following lecture excerpt. "
            "Respond with only the title, no quotes or punctuation at the end.\n\n"
            f"{chapter_text}"
        )
        return self.generate(prompt)
