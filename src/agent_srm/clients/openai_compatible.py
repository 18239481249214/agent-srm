"""The only network adapter. Covers vLLM, Ollama, llama.cpp, TGI, LM Studio,
DeepSeek, Together, Fireworks, Z.ai, Moonshot, Qwen — anything exposing Chat
Completions.

Two places these servers genuinely disagree, both handled as config rather than
assumption: constrained-JSON support (`json_strategy`) and reasoning controls
(`reasoning`, passed straight through as extra request fields).
"""

from __future__ import annotations

import json
import os
import threading
import time

import requests

from .base import Completion, TransportError, with_backoff

_RETRYABLE = {408, 409, 425, 429, 500, 502, 503, 504}


class OpenAICompatibleClient:
    def __init__(self, base_url: str, model_id: str, *, api_key_env: str = "LLM_API_KEY",
                 timeout_s: int = 120, max_retries: int = 5, backoff_base_s: float = 2.0,
                 json_strategy: str = "none"):
        self.base_url = base_url.rstrip("/")
        self.model_id = model_id
        self.api_key = os.environ.get(api_key_env, "")
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.backoff_base_s = backoff_base_s
        self.json_strategy = json_strategy
        self._local = threading.local()

    @property
    def _session(self) -> requests.Session:
        """One Session per thread. requests.Session is not thread-safe, and
        dyads are processed in parallel."""
        s = getattr(self._local, "session", None)
        if s is None:
            s = requests.Session()
            self._local.session = s
        return s

    # -- public ----------------------------------------------------------
    def smoke_test(self) -> None:
        """One-token call at startup so bad credentials or a dead endpoint fail
        fast, before a long run spends anything."""
        self.complete("You are a test.", [{"role": "user", "content": "hi"}],
                      max_tokens=1, temperature=0.0)

    def complete(self, system, messages, max_tokens, temperature, top_p=1.0,
                 seed=None, json_schema=None, reasoning=None) -> Completion:
        payload = {
            "model": self.model_id,
            "messages": [{"role": "system", "content": system}] + messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }
        if seed is not None:
            payload["seed"] = seed
        if json_schema is not None:
            self._apply_json_strategy(payload, json_schema)
        if reasoning:
            # Passed through verbatim. Vendors disagree on the key name
            # (`reasoning_effort`, `thinking`, `enable_thinking`, ...), so the
            # config carries whatever this endpoint wants. None = provider default.
            payload.update(reasoning)

        t0 = time.time()
        (data, _), retries = with_backoff(
            lambda: self._post(payload), self.max_retries, self.backoff_base_s
        )
        latency = int((time.time() - t0) * 1000)

        choice = (data.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        text = msg.get("content") or ""

        # Reasoning is captured for the record and NEVER re-inserted into the
        # message thread (design doc §6.2).
        reasoning_text = msg.get("reasoning_content") or msg.get("reasoning") or None

        usage = data.get("usage") or {}
        details = usage.get("completion_tokens_details") or {}
        reasoning_tokens = int(
            details.get("reasoning_tokens")
            or usage.get("reasoning_tokens")
            or 0
        )

        # Most OpenAI-compatible servers accept `seed` and ignore it silently.
        honored = bool(seed is not None and data.get("system_fingerprint"))

        return Completion(
            text=text,
            reasoning_text=reasoning_text,
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
            reasoning_tokens=reasoning_tokens,
            model_id=self.model_id,
            model_snapshot=data.get("model") or self.model_id,
            stop_reason=choice.get("finish_reason", ""),
            latency_ms=latency,
            seed_honored=honored,
            transport_retries=retries,
            raw_response=data,
        )

    # -- internals -------------------------------------------------------
    def _apply_json_strategy(self, payload: dict, schema: dict) -> None:
        s = self.json_strategy
        if s == "response_format":
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "ratings", "schema": schema, "strict": True},
            }
        elif s == "guided_json":                      # vLLM
            payload["guided_json"] = schema
        elif s == "ollama_format":                    # Ollama
            payload["format"] = schema
        # "none": prompt-only; rating.py's repair loop does the work

    def _post(self, payload: dict):
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        try:
            r = self._session.post(
                f"{self.base_url}/chat/completions",
                headers=headers, data=json.dumps(payload), timeout=self.timeout_s,
            )
        except requests.RequestException as e:
            raise TransportError(str(e)) from e
        if r.status_code in _RETRYABLE:
            raise TransportError(f"HTTP {r.status_code}: {r.text[:200]}")
        if r.status_code >= 400:
            raise RuntimeError(f"HTTP {r.status_code}: {r.text[:500]}")
        return r.json(), r.status_code
