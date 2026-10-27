"""Adapter registry.

Model specs:
    mock
    openai:<model>  deepseek:<model>  qwen:<model>  glm:<model>
    anthropic:<model>
    compat:<base_url>|<model>   (explicit gateway)
"""

from __future__ import annotations

from .base import Adapter, Generation
from .mock import MockAdapter

__all__ = ["Adapter", "Generation", "MockAdapter", "get_adapter"]


def get_adapter(spec: str, **kwargs) -> Adapter:
    if spec == "mock":
        return MockAdapter(**kwargs)
    if spec.startswith("anthropic:"):
        from .anthropic import AnthropicAdapter

        return AnthropicAdapter(model=spec.split(":", 1)[1], **kwargs)
    if spec.startswith("compat:"):
        from .openai_compat import OpenAICompatAdapter

        base_url, model = spec.split(":", 1)[1].split("|", 1)
        return OpenAICompatAdapter(model=model, base_url=base_url, **kwargs)
    if ":" in spec:
        from .openai_compat import OpenAICompatAdapter

        provider, model = spec.split(":", 1)
        return OpenAICompatAdapter(model=model, provider=provider, **kwargs)
    raise ValueError(f"unknown model spec: {spec!r}")
