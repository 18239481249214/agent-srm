"""Output writers (§8). Blocks are written as shards, then assembled — which is
what makes resume safe: a completed block is on disk before the next starts.
"""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

LONG_COLUMNS = [
    "study_id", "arm", "run", "rater", "target", "construct", "item", "score",
    "k", "intro_mode", "rater_role", "rater_name", "target_name", "item_position",
    "provider", "model_id", "model_snapshot", "reasoning_mode", "persona_mode",
    "temperature", "seed", "timestamp",
]

WIDE_BASE = [
    "study_id", "arm", "run", "rater", "target",
]
WIDE_TAIL = [
    "k", "intro_mode", "rater_role", "rater_name", "target_name",
    "provider", "model_id", "model_snapshot", "reasoning_mode", "persona_mode",
    "reasoning_tokens", "temperature", "seed", "timestamp",
]


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def write_shard(ckpt_dir: Path, run: int, payload: dict) -> None:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    tmp = ckpt_dir / f"block_{run:05d}.json.tmp"
    tmp.write_text(json.dumps(payload), encoding="utf-8")
    tmp.rename(ckpt_dir / f"block_{run:05d}.json")


def completed_runs(ckpt_dir: Path) -> set[int]:
    if not ckpt_dir.exists():
        return set()
    return {int(p.stem.split("_")[1]) for p in ckpt_dir.glob("block_*.json")}


def load_shards(ckpt_dir: Path) -> list[dict]:
    return [json.loads(p.read_text(encoding="utf-8"))
            for p in sorted(ckpt_dir.glob("block_*.json"))]


def write_quarantine(out: Path, run: int, payload: dict) -> None:
    q = out / "quarantine"
    q.mkdir(parents=True, exist_ok=True)
    (q / f"block_{run:05d}.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8")


def assemble(out: Path, shards: list[dict], items: list[str]) -> dict[str, int]:
    """Write ratings.csv, ratings_wide.csv and transcripts.jsonl from shards."""
    out.mkdir(parents=True, exist_ok=True)
    wide_cols = WIDE_BASE + items + WIDE_TAIL

    n_long = n_wide = n_tx = 0
    with open(out / "ratings.csv", "w", newline="", encoding="utf-8") as fl, \
         open(out / "ratings_wide.csv", "w", newline="", encoding="utf-8") as fw, \
         open(out / "transcripts.jsonl", "w", encoding="utf-8") as ft:
        wl = csv.DictWriter(fl, fieldnames=LONG_COLUMNS)
        ww = csv.DictWriter(fw, fieldnames=wide_cols)
        wl.writeheader()
        ww.writeheader()
        for shard in shards:
            for row in shard["long_rows"]:
                wl.writerow(row)
                n_long += 1
            for row in shard["wide_rows"]:
                ww.writerow(row)
                n_wide += 1
            for tx in shard["transcripts"]:
                ft.write(json.dumps(tx) + "\n")
                n_tx += 1
    return {"long_rows": n_long, "wide_rows": n_wide, "transcripts": n_tx}


def write_personas_csv(out: Path, personas: dict, run_of: dict[str, int],
                       names: dict[str, dict] | None, whitelist: list[str]) -> int:
    out.mkdir(parents=True, exist_ok=True)
    pids = sorted(run_of, key=lambda p: (run_of[p], p))
    score_keys: set[str] = set()
    for p in pids:
        score_keys |= set(personas[p].scores)
    score_cols = sorted(score_keys)
    cols = (["pid", "run", "first_name", "last_name"] + whitelist
            + score_cols + [f"{s}_pct" for s in score_cols])
    with open(out / "personas.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for pid in pids:
            p = personas[pid]
            nm = (names or {}).get(pid) or {}
            row = {"pid": pid, "run": run_of[pid],
                   "first_name": nm.get("first"), "last_name": nm.get("last")}
            row.update({k: p.covariates.get(k) for k in whitelist})
            row.update({s: p.scores.get(s) for s in score_cols})
            row.update({f"{s}_pct": p.percentiles.get(s) for s in score_cols})
            w.writerow(row)
    return len(pids)


def design_digest(blocks) -> str:
    """SHA-256 over the ordered design records. Must match across all arms."""
    parts = []
    for b in blocks:
        parts.append(f"run={b.run};pids={','.join(b.pids)}")
        for d in b.dyads:
            parts.append(f"  dyad={d.dyad_id};init={d.initiator}")
    return sha256_text("\n".join(parts))


def write_manifest(out: Path, manifest: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8")
