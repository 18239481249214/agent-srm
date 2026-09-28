"""Block assignment, dyad enumeration, initiator orientation, item order.

All randomness derives from one top-level seed through named sub-streams
(§10), so changing one does not shift the others.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field
from itertools import combinations


def substream(seed: int, name: str, *parts: object) -> random.Random:
    """Deterministic named sub-stream. `parts` scopes it further (per block, per call)."""
    key = "|".join([str(seed), name, *(str(p) for p in parts)])
    h = hashlib.sha256(key.encode()).digest()
    return random.Random(int.from_bytes(h[:8], "big"))


@dataclass
class Dyad:
    run: int
    a: str                     # pid
    b: str                     # pid
    initiator: str | None      # pid, or None at k == 2
    dyad_id: str = ""

    def __post_init__(self) -> None:
        if not self.dyad_id:
            self.dyad_id = f"{self.run}_{self.a}_{self.b}"

    def role(self, pid: str) -> str:
        if self.initiator is None:
            return "none"
        return "initiator" if pid == self.initiator else "responder"

    def other(self, pid: str) -> str:
        return self.b if pid == self.a else self.a


@dataclass
class Block:
    run: int
    pids: list[str]
    dyads: list[Dyad] = field(default_factory=list)


def assign_blocks(pids: list[str], seed: int, block_size: int, n_blocks: int
                  ) -> tuple[list[Block], list[str]]:
    """Shuffle without replacement; chunk; drop the remainder. `run` never restarts."""
    pool = list(pids)
    substream(seed, "blocks").shuffle(pool)
    full = [pool[i:i + block_size] for i in range(0, len(pool), block_size)]
    full = [c for c in full if len(c) == block_size]
    chosen = full[:n_blocks]
    blocks = [Block(run=i + 1, pids=members) for i, members in enumerate(chosen)]
    used = {p for b in blocks for p in b.pids}
    dropped = [p for p in pool if p not in used]
    return blocks, dropped


def orient(block: Block, seed: int, k: int) -> None:
    """A4: balanced orientation table over a seeded permutation.

    Personas are permuted into indices 0..n-1 (the only randomness). For dyad
    (i, j) with i < j, let d = (j - i) mod n; persona i initiates iff
    d <= (n-1)/2. Every persona's initiator count is then floor((n-1)/2) or
    ceil((n-1)/2) — the minimum achievable spread. At k == 2 there is no
    initiator at all.
    """
    perm = list(block.pids)
    substream(seed, "initiator", block.run).shuffle(perm)
    n = len(perm)
    block.dyads = []
    for i, j in combinations(range(n), 2):
        a, b = perm[i], perm[j]
        if k == 2:
            initiator = None
        else:
            d = (j - i) % n
            initiator = a if d <= (n - 1) / 2 else b
        block.dyads.append(Dyad(run=block.run, a=a, b=b, initiator=initiator))
    return None


def initiator_counts(block: Block) -> dict[str, int]:
    counts = {p: 0 for p in block.pids}
    for d in block.dyads:
        if d.initiator is not None:
            counts[d.initiator] += 1
    return counts


def item_order(items: list[str], seed: int, run: int, rater: str, target: str,
               randomize: bool) -> list[str]:
    """Per-rating-call presentation order, seeded and reproducible."""
    order = list(items)
    if randomize:
        substream(seed, "items", run, rater, target).shuffle(order)
    return order
