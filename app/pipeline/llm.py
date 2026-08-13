from __future__ import annotations

import json
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
        if not self._is_up():
            self._start_server()
        self._ensure_model_pulled()

    def _start_server(self) -> None:
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

    def _has_model(self) -> bool:
        response = requests.get(f"{self.config.base_url}/api/tags", timeout=5.0)
        response.raise_for_status()
        tags = {entry["name"] for entry in response.json().get("models", [])}
        # Ollama tags are "name:tag"; accept an exact match or a bare-name match with default "latest".
        target = self.config.model
        return target in tags or f"{target}:latest" in tags

    def _ensure_model_pulled(self) -> None:
        if not self.config.auto_pull:
            return
        if self._has_model():
            return

        logger.info("Pulling ollama model %s (this may take a while on first run)", self.config.model)
        response = requests.post(
            f"{self.config.base_url}/api/pull",
            json={"model": self.config.model, "stream": False},
            timeout=self.config.pull_timeout_s,
        )
        response.raise_for_status()
        logger.info("Finished pulling ollama model %s", self.config.model)

    def stop(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            self._proc.terminate()
        self._proc = None


class OllamaClient:
    """Minimal HTTP client for the local Ollama `/api/generate` endpoint."""

    def __init__(self, config: OllamaConfig | None = None, manager: OllamaManager | None = None) -> None:
        self.config = config or OllamaConfig()
        self.manager = manager or OllamaManager(self.config)

    def generate(self, prompt: str, *, json_mode: bool = False) -> str:
        self.manager.ensure_running()
        payload: dict = {"model": self.config.model, "prompt": prompt, "stream": False}
        if json_mode:
            payload["format"] = "json"
        response = requests.post(
            f"{self.config.base_url}/api/generate",
            json=payload,
            timeout=self.config.request_timeout_s,
        )
        response.raise_for_status()
        return response.json()["response"].strip()

    def describe_chapter(self, chapter_text: str) -> dict:
        """Return {"title": str, "keywords": list[str], "summary": str} for a chapter."""

        prompt = (
            "You are analyzing one chapter of a lecture transcript. "
            "Respond with ONLY a JSON object with exactly these keys:\n"
            '- "title": a short chapter title (<= 8 words)\n'
            '- "keywords": a list of 3-5 short keywords/phrases\n'
            '- "summary": a one-line summary (<= 25 words)\n\n'
            f"Transcript:\n{chapter_text}"
        )
        raw = self.generate(prompt, json_mode=True)
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("Failed to parse LLM JSON response: %r", raw)
            return {"title": None, "keywords": [], "summary": None}

        keywords = parsed.get("keywords") or []
        if not isinstance(keywords, list):
            keywords = [str(keywords)]
        return {
            "title": parsed.get("title"),
            "keywords": [str(k).strip() for k in keywords],
            "summary": parsed.get("summary"),
        }

