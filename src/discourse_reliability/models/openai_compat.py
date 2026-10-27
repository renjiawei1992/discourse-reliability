"""OpenAI-compatible chat adapter.

Works for OpenAI, DeepSeek, Qwen (DashScope compatible mode), GLM and most
local gateways -- anything that implements ``POST /chat/completions``.

Keys are read from the environment; nothing is written to disk.
"""

from __future__ import annotations

import os
import time
from typing import Sequence

from .base import Generation

PROVIDERS = {
    "openai": ("https://api.openai.com/v1", "OPENAI_API_KEY"),
    "deepseek": ("https://api.deepseek.com/v1", "DEEPSEEK_API_KEY"),
    "qwen": ("https://dashscope.aliyuncs.com/compatible-mode/v1", "DASHSCOPE_API_KEY"),
    "glm": ("https://open.bigmodel.cn/api/paas/v4", "ZHIPU_API_KEY"),
    "siliconflow": ("https://api.siliconflow.cn/v1", "SILICONFLOW_API_KEY"),
    # Multi-vendor gateway: one key, several vendors' models behind it.
    "chatanywhere": ("https://api.chatanywhere.tech", "CHATANYWHERE_API_KEY"),
}


class OpenAICompatAdapter:
    def __init__(
        self,
        model: str,
        provider: str = "openai",
        base_url: str | None = None,
        api_key_env: str | None = None,
        timeout: int = 90,
        max_retries: int = 3,
    ) -> None:
        import requests  # imported lazily so the core stays dependency-free

        self._requests = requests
        default_url, default_env = PROVIDERS.get(provider, (None, None))
        self.base_url = (base_url or default_url or "").rstrip("/")
        if not self.base_url:
            raise ValueError(f"unknown provider {provider!r}; pass base_url")
        env = api_key_env or default_env
        self.api_key = os.environ.get(env or "", "")
        if not self.api_key:
            raise RuntimeError(f"missing API key: set {env}")
        self.model = model
        self.name = f"{provider}:{model}"
        self.timeout = timeout
        self.max_retries = max_retries
        # Adjusted once per adapter instance when a model rejects the default
        # parameter names (see the 400 handler in ``_one``).
        self._token_param = "max_tokens"
        self._send_temperature = True

    def _one(self, messages: Sequence[dict[str, str]], temperature: float, max_tokens: int) -> Generation:
        url = f"{self.base_url}/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        last = ""
        attempts = 0
        # Learning a model's parameter names must not consume the network retry
        # budget: with --max-retries 1 the first call would 400, switch, and run
        # out of attempts without ever sending a valid request.
        switch_budget = 2
        while attempts < self.max_retries:
            payload = {
                "model": self.model,
                "messages": list(messages),
                self._token_param: max_tokens,
            }
            if self._send_temperature:
                payload["temperature"] = temperature
            t0 = time.perf_counter()
            try:
                r = self._requests.post(url, json=payload, headers=headers, timeout=self.timeout)
                latency = (time.perf_counter() - t0) * 1000
                if r.status_code in (429, 500, 502, 503, 504):
                    attempts += 1
                    last = f"HTTP {r.status_code}"
                    time.sleep(2**attempts)
                    continue
                if r.status_code == 400:
                    err = {}
                    try:
                        err = r.json().get("error", {}) or {}
                    except ValueError:
                        pass
                    param = str(err.get("param") or "")
                    if param == "max_tokens" and switch_budget > 0:
                        switch_budget -= 1
                        self._token_param = "max_completion_tokens"
                        continue
                    if param == "temperature" and switch_budget > 0:
                        switch_budget -= 1
                        self._send_temperature = False
                        continue
                    return Generation(
                        text="",
                        error=f"HTTP 400: {err.get('message') or r.text[:200]}",
                    )
                r.raise_for_status()
                data = r.json()
                # Gateways alias model names; record what actually answered.
                usage = dict(data.get("usage") or {})
                usage["served_model"] = data.get("model")
                choice = data["choices"][0]
                msg = choice["message"]
                text = msg.get("content") or msg.get("reasoning_content") or ""
                if choice.get("finish_reason") == "length":
                    # Budget exhausted.  Reasoning models spend it thinking
                    # first, so the content is often empty or half a JSON
                    # object -- keep whatever arrived and mark the cause,
                    # otherwise a truncation is misread as a parsing failure.
                    return Generation(
                        text=text,
                        latency_ms=latency,
                        usage=usage,
                        error="truncated: raise --max-tokens",
                    )
                return Generation(text=text, latency_ms=latency, usage=usage)
            except Exception as exc:  # noqa: BLE001 - surface any adapter failure per item
                attempts += 1
                last = f"{type(exc).__name__}: {exc}"
                time.sleep(min(2**attempts, 8))
        return Generation(text="", error=last)

    def generate(
        self,
        messages: Sequence[dict[str, str]],
        n: int = 1,
        temperature: float = 0.0,
        max_tokens: int = 256,
    ) -> list[Generation]:
        return [self._one(messages, temperature, max_tokens) for _ in range(n)]
