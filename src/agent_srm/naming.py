"""Name assignment (§4.6). Off by default.

First names filtered on the persona's Gender field only; surnames unfiltered.
Deliberately NOT matched on race, age cohort or region. `rater`/`target` are
always pid — names live in separate columns so re-randomizing never breaks a join.
"""

from __future__ import annotations

import csv
from pathlib import Path

from .design import substream

# Small vendored fallback so the pipeline runs before the real SSA/Census files
# are fetched. scripts/fetch_names.py replaces these with the full pools.
_FALLBACK_F = ["Mary", "Linda", "Patricia", "Susan", "Karen", "Nancy", "Betty", "Sandra"]
_FALLBACK_M = ["James", "Robert", "John", "Michael", "David", "William", "Richard", "Joseph"]
_FALLBACK_L = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis"]


def load_pools(first_path: str | None, last_path: str | None
               ) -> tuple[dict[str, list[str]], list[str]]:
    first = {"Female": list(_FALLBACK_F), "Male": list(_FALLBACK_M)}
    last = list(_FALLBACK_L)
    if first_path and Path(first_path).exists():
        first = {"Female": [], "Male": []}
        with open(first_path, newline="") as f:
            for row in csv.DictReader(f):
                sex = row.get("sex", "").upper()
                key = "Female" if sex.startswith("F") else "Male"
                first[key].append(row["name"])
    if last_path and Path(last_path).exists():
        with open(last_path, newline="") as f:
            last = [row["name"] for row in csv.DictReader(f)]
    return first, last


def assign_names(personas: dict, pids: list[str], seed: int,
                 first_path: str | None = None, last_path: str | None = None,
                 match_gender: bool = True) -> dict[str, dict]:
    first_pool, last_pool = load_pools(first_path, last_path)
    rng = substream(seed, "names")
    out: dict[str, dict] = {}
    for pid in sorted(pids):
        gender = personas[pid].gender if match_gender else None
        pool = first_pool.get(gender) or (first_pool["Female"] + first_pool["Male"])
        out[pid] = {"first": rng.choice(pool), "last": rng.choice(last_pool)}
    return out
