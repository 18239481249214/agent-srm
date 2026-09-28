"""Client protocol and the Completion record written to every transcript."""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Protocol


@dataclass
class Completion:
    text: str
    reasoning_text: str | None = None      # never enters the message thread
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    model_id: str = ""
    model_snapshot: str = ""               # what the server actually resolved
    stop_reason: str = ""
    latency_ms: int = 0
    seed_honored: bool = False
    transport_retries: int = 0
    raw_response: dict[str, Any] = field(default_factory=dict)

    def record(self, *, keep_raw: bool = False) -> dict:
        d = asdict(self)
        if not keep_raw:
            d.pop("raw_response")
        return d


class LLMClient(Protocol):
    def complete(
        self,
        system: str,
        messages: list[dict],
        max_tokens: int,
        temperature: float,
        top_p: float = 1.0,
        seed: int | None = None,
        json_schema: dict | None = None,
        reasoning: dict | None = None,
    ) -> Completion: ...


class TransportError(RuntimeError):
    pass


def with_backoff(fn, max_retries: int, base_s: float, rng: random.Random | None = None):
    """Exponential backoff with jitter on 429/5xx. Distinct from JSON-repair retries."""
    rng = rng or random.Random(0)
    last: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            return fn(), attempt
        except TransportError as e:
            last = e
            if attempt == max_retries:
                break
            time.sleep(base_s * (2 ** attempt) * (0.5 + rng.random()))
    raise TransportError(f"exhausted {max_retries} retries: {last}")
