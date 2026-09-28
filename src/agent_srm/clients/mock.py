"""Deterministic mock client. Exercises the full pipeline at zero API cost.

Text is derived from a hash of the request so a dry run is reproducible and
different personas produce visibly different intros. Ratings vary by
(rater, target) so the CSV has non-degenerate variance to eyeball.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading

from .base import Completion

_OPENERS = [
    "Hey there — nice to meet you. I've been meaning to get out of my own head today.",
    "Hi! I'm not usually the one to start these, but here goes.",
    "Hello. Long week over here, so this is a welcome distraction.",
    "Hey. I'll admit I'm curious what brings other people to these chats.",
    "Hi there. I just got back from a walk, so I'm in a decent mood.",
    "Hey — hope your day's going alright. Mine's been a mixed bag.",
]

_FOLLOWUPS = [
    "That's interesting, I hadn't thought about it that way.",
    "Ha, that sounds about right. I'd probably do the same.",
    "Fair enough. I tend to overthink that sort of thing myself.",
    "Yeah, I hear you. It's been a similar stretch on my end.",
]


def _h(*parts: object) -> int:
    return int.from_bytes(
        hashlib.sha256("|".join(str(p) for p in parts).encode()).digest()[:8], "big"
    )


class MockClient:
    """Drop-in stand-in for OpenAICompatibleClient."""

    def __init__(self, model_id: str = "mock", *, fail_json_for: set[str] | None = None,
                 scale_min: int = 1, scale_max: int = 5):
        self.model_id = model_id
        self.fail_json_for = fail_json_for or set()   # pids whose ratings never parse
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.calls = 0
        self._lock = threading.Lock()

    def smoke_test(self) -> None:
        return None

    def complete(self, system, messages, max_tokens, temperature, top_p=1.0,
                 seed=None, json_schema=None, reasoning=None) -> Completion:
        with self._lock:
            self.calls += 1
            nonce = self.calls
        sysfp = _h(system[:2000])

        if json_schema is not None:
            text = self._ratings(system, messages, json_schema, sysfp)
        else:
            # Conversation turns vary per call, as a real model at temperature
            # 1.0 does. Shared-mode intros stay byte-identical because they are
            # generated once and reused, not because the mock is deterministic.
            text = self._chat(messages, sysfp, nonce)

        return Completion(
            text=text,
            input_tokens=len(system) // 4 + sum(len(m["content"]) for m in messages) // 4,
            output_tokens=max(1, len(text) // 4),
            reasoning_tokens=0,
            model_id=self.model_id,
            model_snapshot=self.model_id,
            stop_reason="stop",
            latency_ms=1,
            seed_honored=seed is not None,
        )

    # -- internals -------------------------------------------------------
    def _chat(self, messages, sysfp, nonce) -> str:
        turn = sum(1 for m in messages if m["role"] == "assistant")
        if turn == 0:
            i = _h(sysfp, nonce) % len(_OPENERS)
            return f"{_OPENERS[i]} (#{nonce})"
        return _FOLLOWUPS[_h(sysfp, turn, nonce) % len(_FOLLOWUPS)] + f" (#{nonce})"

    def _ratings(self, system, messages, schema, sysfp) -> str:
        items = list(schema.get("properties", {}))
        partner = _h(messages[-1]["content"][:500])
        if self._marked_fail(system):
            return "Sure! Here are my ratings: warm=4, kind=5. Hope that helps."
        span = self.scale_max - self.scale_min
        out = {}
        for i, name in enumerate(items):
            # mild target effect + perceiver effect + pairing wobble
            v = self.scale_min + (_h(partner, name) % (span + 1))
            bias = (sysfp + i) % 2
            out[name] = max(self.scale_min, min(self.scale_max, v - bias + 1))
        return json.dumps(out)

    def _marked_fail(self, system: str) -> bool:
        return any(re.search(rf"\b{re.escape(p)}\b", system[:4000])
                   for p in self.fail_json_for)
