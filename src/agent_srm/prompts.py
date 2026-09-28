"""Prompt templates: load, hash, render.

Every rendered prompt actually sent is stored by the caller. Nothing is
reconstructed from templates afterward (§13).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

TEMPLATES = [
    "persona_system.txt", "framing_anonymous.txt", "framing_named.txt",
    "opening_turn.txt", "rating.txt", "repair.txt", "user_turn_join.txt",
]


@dataclass
class Prompts:
    persona_system: str
    framing_anonymous: str
    framing_named: str
    opening_turn: str
    rating: str
    repair: str
    user_turn_join: str
    hashes: dict[str, str]

    # -- rendering -------------------------------------------------------
    def system_for(self, summary: str | None, *, self_first: str | None = None,
                   self_last: str | None = None, other_first: str | None = None) -> str:
        """summary=None renders the task framing alone — the `none` persona mode.
        No persona wrapper text either, since 'the following is a description of
        a person' with no description would be incoherent."""
        body = ("" if summary is None
                else self.persona_system.replace("{persona_summary}", summary))
        if other_first is not None:
            framing = (self.framing_named
                       .replace("{other_first_name}", other_first)
                       .replace("{self_first_name}", self_first or "")
                       .replace("{self_last_name}", self_last or ""))
        else:
            framing = self.framing_anonymous
        if not body:
            return framing.strip() + "\n"
        return body.rstrip("\n") + "\n\n" + framing.strip() + "\n"

    def rating_prompt(self, order: list[str], scale_min: int, scale_max: int) -> str:
        return (self.rating
                .replace("{item_lines}", "\n".join(order))
                .replace("{scale_min}", str(scale_min))
                .replace("{scale_max}", str(scale_max))).strip() + "\n"

    def repair_prompt(self, order: list[str], scale_min: int, scale_max: int) -> str:
        return (self.repair
                .replace("{n_items}", str(len(order)))
                .replace("{item_names}", ", ".join(order))
                .replace("{scale_min}", str(scale_min))
                .replace("{scale_max}", str(scale_max))).strip() + "\n"

    def merge_user_turns(self, chunks: list[str]) -> str:
        """A5: the join string is a versioned constant, hashed into the manifest.

        Two implementers must produce identical bytes from the same spec, and
        many open-model chat templates reject non-alternating roles outright.
        """
        return self.user_turn_join.join(c.rstrip("\n") for c in chunks)


def load_prompts(directory: str | Path) -> Prompts:
    d = Path(directory)
    texts, hashes = {}, {}
    for name in TEMPLATES:
        p = d / name
        if not p.exists():
            raise FileNotFoundError(f"missing prompt template: {p}")
        raw = p.read_text(encoding="utf-8")
        texts[name[:-4]] = raw
        hashes[name] = hashlib.sha256(raw.encode()).hexdigest()
    return Prompts(hashes=hashes, **texts)
