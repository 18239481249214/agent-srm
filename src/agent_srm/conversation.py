"""Thread construction and the dyad turn loop (§5.6).

Each persona is a fresh instance per dyad. No memory carries across dyads.
Reasoning content is captured but NEVER inserted as an assistant turn.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone

FLAG_PATTERNS = {
    "ai_self_reference": re.compile(
        r"\b(as an (?:AI|artificial)|I am an? (?:AI|language model|assistant)|"
        r"I'?m an? (?:AI|language model|assistant)|large language model)\b", re.I),
    "third_person_narration": re.compile(r"(^\s*\*[^*]+\*)|(^\s*(?:He|She|They)\s+\w+s\b)"),
    "meta_task_reference": re.compile(
        r"\b(persona|the description above|this (?:study|experiment|task)|"
        r"rate (?:them|this person)|system prompt)\b", re.I),
    "refusal": re.compile(
        r"\b(I (?:can'?t|cannot|won'?t) (?:continue|help|do that|role-?play)|"
        r"I'?m not able to)\b", re.I),
}


def flag_message(text: str, *, median_tokens: float | None = None,
                 outlier_multiple: float = 3.0) -> list[str]:
    flags: list[str] = []
    if not text or not text.strip():
        return ["empty"]
    for name, pat in FLAG_PATTERNS.items():
        if pat.search(text):
            flags.append(name)
    if median_tokens:
        approx = len(text) / 4
        if approx > median_tokens * outlier_multiple:
            flags.append("length_outlier")
    return flags


@dataclass
class Turn:
    turn: int
    sender_pid: str
    content: str
    reasoning_text: str | None = None
    reasoning_tokens: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: int = 0
    stop_reason: str = ""
    flags: list[str] = field(default_factory=list)
    timestamp: str = ""

    def record(self) -> dict:
        return {
            "turn": self.turn, "sender_pid": self.sender_pid, "content": self.content,
            "reasoning_text": self.reasoning_text, "reasoning_tokens": self.reasoning_tokens,
            "input_tokens": self.input_tokens, "output_tokens": self.output_tokens,
            "latency_ms": self.latency_ms, "stop_reason": self.stop_reason,
            "flags": self.flags, "timestamp": self.timestamp,
        }


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def generate_intro(client, system: str, prompts, call_cfg, reasoning) -> tuple[str, object]:
    """One blind opening message — used at k=2 and as turn 1 at k>=4."""
    msgs = [{"role": "user", "content": prompts.opening_turn.strip()}]
    c = client.complete(
        system=system, messages=msgs,
        max_tokens=call_cfg.max_tokens, temperature=call_cfg.temperature,
        top_p=call_cfg.top_p, seed=call_cfg.seed, json_schema=None, reasoning=reasoning,
    )
    return c.text.strip(), c


def run_sequential_dyad(client, prompts, call_cfg, reasoning, *, k: int,
                        systems: dict[str, str], initiator: str, responder: str,
                        first_message: str | None = None) -> list[Turn]:
    """k >= 4: strict alternation, initiator first, k messages total."""
    turns: list[Turn] = []
    order = [initiator, responder]
    for t in range(1, k + 1):
        sender = order[(t - 1) % 2]
        if t == 1 and first_message is not None:
            turns.append(Turn(turn=1, sender_pid=sender, content=first_message,
                              timestamp=_now(), flags=flag_message(first_message)))
            continue
        msgs = thread_for(sender, turns, prompts, opening=prompts.opening_turn.strip())
        c = client.complete(
            system=systems[sender], messages=msgs,
            max_tokens=call_cfg.max_tokens, temperature=call_cfg.temperature,
            top_p=call_cfg.top_p, seed=call_cfg.seed, json_schema=None, reasoning=reasoning,
        )
        text = c.text.strip()
        turns.append(Turn(
            turn=t, sender_pid=sender, content=text,
            reasoning_text=c.reasoning_text, reasoning_tokens=c.reasoning_tokens,
            input_tokens=c.input_tokens, output_tokens=c.output_tokens,
            latency_ms=c.latency_ms, stop_reason=c.stop_reason,
            flags=flag_message(text), timestamp=_now(),
        ))
    return turns


def thread_for(pid: str, turns: list[Turn], prompts, *, opening: str) -> list[dict]:
    """The conversation from `pid`'s own point of view.

    Own messages are assistant turns; the partner's are user turns. The opening
    instruction is the first user turn only for the persona who sent turn 1.
    Consecutive user turns are merged with the versioned join string (§5.7).
    """
    msgs: list[dict] = []
    if turns and turns[0].sender_pid == pid:
        msgs.append({"role": "user", "content": opening})
    for t in turns:
        role = "assistant" if t.sender_pid == pid else "user"
        if msgs and msgs[-1]["role"] == role == "user":
            msgs[-1]["content"] = prompts.merge_user_turns(
                [msgs[-1]["content"], t.content])
        else:
            msgs.append({"role": role, "content": t.content})
    if not msgs or msgs[0]["role"] != "user":
        msgs.insert(0, {"role": "user", "content": opening})
    return msgs


def blind_thread(prompts, *, own_intro: str, partner_intro: str,
                 rating_prompt: str) -> list[dict]:
    """k = 2 rating context (§5.6). The two trailing user turns are merged."""
    return [
        {"role": "user", "content": prompts.opening_turn.strip()},
        {"role": "assistant", "content": own_intro},
        {"role": "user", "content": prompts.merge_user_turns(
            [partner_intro, rating_prompt])},
    ]


def rating_thread(pid: str, turns: list[Turn], prompts, rating_prompt: str) -> list[dict]:
    """k >= 4 rating context: the full thread plus the rating prompt as final user turn."""
    msgs = thread_for(pid, turns, prompts, opening=prompts.opening_turn.strip())
    if msgs and msgs[-1]["role"] == "user":
        msgs[-1]["content"] = prompts.merge_user_turns(
            [msgs[-1]["content"], rating_prompt])
    else:
        msgs.append({"role": "user", "content": rating_prompt})
    return msgs
