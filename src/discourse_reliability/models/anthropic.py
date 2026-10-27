"""Anthropic Messages API adapter."""

from __future__ import annotations

import os
import time
from typing import Sequence

from .base import Generation


class AnthropicAdapter:
    def __init__(
        self,
        model: str = "claude-sonnet-5-5",
        api_key_env: str = "ANTHROPIC_API_KEY",
        base_url: str = "https://api.anthropic.com/v1",
        timeout: int = 90,
        max_retries: int = 3,
    ) -> None:
        import requests

        self._requests = requests
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = os.environ.get(api_key_env, "")
        if not self.api_key:
            raise RuntimeError(f"missing API key: set {api_key_env}")
        self.name = f"anthropic:{model}"
        self.timeout = timeout
        self.max_retries = max_retries

    def _split(self, messages: Sequence[dict[str, str]]) -> tuple[str, list[dict[str, str]]]:
        system = "\n".join(m["content"] for m in messages if m["role"] == "system")
        turns = [m for m in messages if m["role"] != "system"]
        return system, turns

    def _one(self, messages: Sequence[dict[str, str]], temperature: float, max_tokens: int) -> Generation:
        system, turns = self._split(messages)
        payload = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "system": system,
            "messages": turns,
        }
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        last = ""
        for attempt in range(self.max_retries):
            t0 = time.perf_counter()
            try:
                r = self._requests.post(
                    f"{self.base_url}/messages", json=payload, headers=headers, timeout=self.timeout
                )
                latency = (time.perf_counter() - t0) * 1000
                if r.status_code in (429, 500, 502, 503, 504):
                    last = f"HTTP {r.status_code}"
                    time.sleep(2**attempt)
                    continue
                r.raise_for_status()
                data = r.json()
                text = "".join(b.get("text", "") for b in data.get("content", []))
                return Generation(text=text, latency_ms=latency, usage=data.get("usage", {}))
            except Exception as exc:  # noqa: BLE001
                last = f"{type(exc).__name__}: {exc}"
                time.sleep(min(2**attempt, 8))
        return Generation(text="", error=last)

    def generate(
        self,
        messages: Sequence[dict[str, str]],
        n: int = 1,
        temperature: float = 0.0,
        max_tokens: int = 256,
    ) -> list[Generation]:
        return [self._one(messages, temperature, max_tokens) for _ in range(n)]
