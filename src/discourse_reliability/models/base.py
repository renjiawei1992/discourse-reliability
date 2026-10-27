"""Model adapter interface."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence


@dataclass
class Generation:
    text: str
    latency_ms: float = 0.0
    error: str | None = None
    usage: dict[str, Any] = field(default_factory=dict)


class Adapter(Protocol):
    name: str

    def generate(
        self,
        messages: Sequence[dict[str, str]],
        n: int = 1,
        temperature: float = 0.0,
        max_tokens: int = 256,
    ) -> list[Generation]:
        ...
