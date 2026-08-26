from __future__ import annotations

import json
import logging
import re
import shutil
import socket
import subprocess
import time

import requests

from app.config import OllamaConfig

logger = logging.getLogger(__name__)

_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_SPECIAL_THINK_BLOCK_RE = re.compile(r"<\|(?:think|thinking)\|>.*?<\|end(?:_of)?_think\|>", re.DOTALL | re.IGNORECASE)


def _strip_thinking(text: str) -> str:
    """Remove complete or malformed thinking blocks some models emit despite think:false."""

    cleaned = _THINK_BLOCK_RE.sub("", text)
    cleaned = _SPECIAL_THINK_BLOCK_RE.sub("", cleaned)
    if "</think>" in cleaned.lower():
        closing_tag = re.search(r"</think>", cleaned, re.IGNORECASE)
        assert closing_tag is not None
        cleaned = cleaned[closing_tag.end() :]
    return cleaned.strip()


class OllamaManager:
    """Ensures `ollama serve` is running locally, starting it if needed."""

    def __init__(self, config: OllamaConfig | None = None) -> None:
        self.config = config or OllamaConfig()
        self._proc: subprocess.Popen | None = None
        self._ready = False

    def _is_up(self) -> bool:
        try:
            with socket.create_connection((self.config.host, self.config.port), timeout=1.0):
                return True
        except OSError:
            return False

    def ensure_running(self) -> None:
        if self._ready:
            return
        if not self._is_up():
            self._start_server()
        self._ensure_model_pulled()
        self._ready = True

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
        try:
            response = requests.get(
                f"{self.config.base_url}/api/tags",
                timeout=self.config.tags_timeout_s,
            )
        except requests.Timeout as exc:
            raise RuntimeError(
                f"Ollama did not answer /api/tags within {self.config.tags_timeout_s:.0f}s. "
                "It may be busy loading the model or under CPU/memory pressure."
            ) from exc
        response.raise_for_status()
        tags = {entry["name"] for entry in response.json().get("models", [])}
        # Ollama tags are "name:tag"; accept an exact match or a bare-name match with default "latest".
        target = self.config.model
        return target in tags or f"{target}:latest" in tags

    def _ensure_model_pulled(self) -> None:
        if not self.config.auto_pull:
            return
        if self._has_model():
            logger.info("Ollama model %s is already available", self.config.model)
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
        self._ready = False


class OllamaClient:
    """Minimal HTTP client for the local Ollama `/api/generate` endpoint."""

    def __init__(self, config: OllamaConfig | None = None, manager: OllamaManager | None = None) -> None:
        self.config = config or OllamaConfig()
        self.manager = manager or OllamaManager(self.config)

    def generate(self, prompt: str, *, json_mode: bool = False) -> str:
        self.manager.ensure_running()
        logger.info("Sending chapter description request to Ollama model %s", self.config.model)
        started_at = time.monotonic()
        payload: dict = {
            "model": self.config.model,
            "prompt": prompt,
            "stream": False,
            "think": False,
        }
        if json_mode:
            payload["format"] = "json"
        try:
            response = requests.post(
                f"{self.config.base_url}/api/generate",
                json=payload,
                timeout=self.config.request_timeout_s,
            )
        except requests.Timeout as exc:
            raise RuntimeError(
                f"Ollama generation timed out after {self.config.request_timeout_s:.0f}s."
            ) from exc
        response.raise_for_status()
        logger.info("Ollama response received in %.1fs", time.monotonic() - started_at)
        return _strip_thinking(response.json()["response"])

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
        for attempt in range(1, 3):
            raw = self.generate(prompt, json_mode=True)
            try:
                description = self._parse_chapter_description(raw)
            except ValueError as exc:
                logger.warning("Invalid Qwen chapter description (attempt %d/2): %s", attempt, exc)
                continue
            return description
        raise RuntimeError("Qwen returned invalid title, keywords, or summary twice")

    @staticmethod
    def _parse_chapter_description(raw: str) -> dict:
        text = raw.strip()
        if text.startswith("```"):
            text = text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError("response was not valid JSON") from exc

        title = str(parsed.get("title") or "").strip()
        summary = str(parsed.get("summary") or "").strip()
        keywords = parsed.get("keywords")
        if not isinstance(keywords, list):
            raise ValueError("keywords was not a JSON list")
        cleaned_keywords = [str(keyword).strip() for keyword in keywords if str(keyword).strip()]
        if not title or not summary or not cleaned_keywords:
            raise ValueError("response omitted title, summary, or keywords")
        return {"title": title, "keywords": cleaned_keywords, "summary": summary}

    def summarize_course(self, chapter_summaries: list[str]) -> str:
        """Return a short course-level overview based on the chapter summaries."""

        prompt = (
            "You are summarizing a lecture from its chapter summaries. Respond with only a concise "
            "two-paragraph plain-text overview. First state what kind of video/course this is. Then "
            "state the main topics covered in their approximate order. Do not mention that you were "
            "given chapter summaries.\n\n"
            "Chapter summaries:\n"
            + "\n".join(f"{index}. {summary}" for index, summary in enumerate(chapter_summaries, start=1))
        )
        return self.generate(prompt)

