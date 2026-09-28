"""Rating call, parse, validate, repair (§6.6).

On final failure the caller quarantines the whole block — a partial block is
unusable in TripleR and silently corrupts variance estimates.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone

_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.M)
_OBJ = re.compile(r"\{.*\}", re.S)


class RatingParseError(RuntimeError):
    pass


@dataclass
class RatingResult:
    rater: str
    target: str
    rating_prompt: str
    item_order: list[str]
    raw_response: str
    parsed: dict[str, int]
    json_retries: int = 0
    reasoning_tokens: int = 0
    model_snapshot: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def record(self) -> dict:
        return {
            "rater": self.rater, "target": self.target,
            "rating_prompt": self.rating_prompt, "item_order": self.item_order,
            "raw_response": self.raw_response, "parsed": self.parsed,
            "json_retries": self.json_retries, "reasoning_tokens": self.reasoning_tokens,
        }


def parse_ratings(text: str, items: list[str], lo: int, hi: int) -> dict[str, int]:
    """Strip fences, parse, then validate shape and range. Raises on any miss."""
    if not text or not text.strip():
        raise RatingParseError("empty response")
    cleaned = _FENCE.sub("", text).strip()
    try:
        obj = json.loads(cleaned)
    except json.JSONDecodeError:
        m = _OBJ.search(cleaned)
        if not m:
            raise RatingParseError(f"no JSON object found in {cleaned[:120]!r}")
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError as e:
            raise RatingParseError(f"malformed JSON: {e}") from e

    if not isinstance(obj, dict):
        raise RatingParseError(f"expected an object, got {type(obj).__name__}")

    got, want = set(obj), set(items)
    if got != want:
        missing, extra = sorted(want - got), sorted(got - want)
        raise RatingParseError(f"key mismatch — missing {missing}, unexpected {extra}")

    out: dict[str, int] = {}
    for name in items:
        v = obj[name]
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise RatingParseError(f"{name}: expected a number, got {v!r}")
        if isinstance(v, float) and not v.is_integer():
            raise RatingParseError(f"{name}: expected a whole number, got {v!r}")
        iv = int(v)
        if not lo <= iv <= hi:
            raise RatingParseError(f"{name}: {iv} outside {lo}..{hi}")
        out[name] = iv
    return out


def rate(client, prompts, items_cfg, call_cfg, reasoning, *, system: str,
         messages: list[dict], order: list[str], rendered_prompt: str,
         rater: str, target: str, max_retries: int) -> RatingResult:
    """One directed rating, with repair retries appended to the same thread."""
    lo, hi = items_cfg.scale_min, items_cfg.scale_max
    schema = items_cfg.json_schema(order)
    thread = list(messages)
    last_err: Exception | None = None
    raw = ""
    total_reasoning = 0

    for attempt in range(max_retries + 1):
        c = client.complete(
            system=system, messages=thread,
            max_tokens=call_cfg.max_tokens, temperature=call_cfg.temperature,
            top_p=call_cfg.top_p, seed=call_cfg.seed,
            json_schema=schema, reasoning=reasoning,
        )
        raw = c.text
        total_reasoning += c.reasoning_tokens
        try:
            parsed = parse_ratings(raw, order, lo, hi)
        except RatingParseError as e:
            # Carry the transport-level facts into the message. "empty response"
            # alone is undiagnosable; finish_reason=length with a large
            # reasoning_tokens count says "raise max_tokens" immediately.
            last_err = RatingParseError(
                f"{e} [finish_reason={c.stop_reason!r} "
                f"output_tokens={c.output_tokens} "
                f"reasoning_tokens={c.reasoning_tokens} "
                f"max_tokens={call_cfg.max_tokens}]"
            )
            if attempt == max_retries:
                break
            # Repair turns are part of the sent thread and are stored as such.
            thread = thread + [
                {"role": "assistant", "content": raw or "(empty)"},
                {"role": "user", "content": prompts.repair_prompt(order, lo, hi)},
            ]
            continue
        return RatingResult(
            rater=rater, target=target, rating_prompt=rendered_prompt,
            item_order=order, raw_response=raw, parsed=parsed,
            json_retries=attempt, reasoning_tokens=total_reasoning,
            model_snapshot=c.model_snapshot or c.model_id,
        )

    raise RatingParseError(
        f"rating {rater}->{target} failed after {max_retries} repair retries: {last_err}"
    )
