#!/usr/bin/env python3
"""Generate synthetic persona parquet chunks + MANIFEST.json for dry runs.

This is a FIXTURE, not data. It mimics the documented shape of Twin-2K-500's
`persona_summary` — all 19 whitelist labels, `score_*` lines with percentiles,
and the free-response sections — so the loader, extractor and validator are
exercised end to end before the real 2,058-persona download exists.

    python scripts/make_synthetic_personas.py --n 64 --chunks 2
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import pandas as pd

WHITELIST = [
    "Geographic region", "Gender", "Age", "Education level", "Race",
    "Citizen of the US", "Marital status", "Religion", "Religious attendance",
    "Political affiliation", "Income", "Political views", "Household size",
    "Employment status",
]

SCORES = [
    "score_extraversion", "score_agreeableness", "wave1_score_conscientiousness",
    "score_openness", "score_neuroticism", "score_needforcognition",
    "score_agency", "score_communion", "score_BES",
]

VALUES = {
    "Geographic region": ["Northeast", "Midwest", "South", "West"],
    "Gender": ["Male", "Female"],
    "Education level": ["High school", "Some college", "Bachelor's", "Graduate degree"],
    "Race": ["Group A", "Group B", "Group C"],
    "Citizen of the US": ["Yes", "No"],
    "Marital status": ["Single", "Married", "Divorced", "Widowed"],
    "Religion": ["None", "Tradition A", "Tradition B"],
    "Religious attendance": ["Never", "Rarely", "Monthly", "Weekly"],
    "Political affiliation": ["Party A", "Party B", "Independent"],
    "Political views": ["Left", "Center", "Right"],
    "Employment status": ["Employed full time", "Part time", "Retired", "Unemployed"],
}

ESSAY = (
    "What I aspire-to-be: someone who finishes what they start and is easy to be "
    "around. What I ought-to-be: more patient with people who are slower than me. "
    "What I actually-are: somewhere in between, most days.\n\n"
    "Dictator game thought listing: I thought about what I would want if the "
    "positions were reversed, then split it closer to even than I expected to.\n\n"
    "Forward flow word chain: river, stone, bridge, iron, rust, orange, morning."
)

# Deliberate trap: lines that LOOK like fields but are essay prose. A naive
# `^Label:` regex invents columns from these; the whitelist extractor ignores them.
PHANTOM = "Resilience: I keep going. Integrity: I try to. Empathy: more than I show."


def make_summary(rng: random.Random, pid: str) -> str:
    lines = []
    for label in WHITELIST:
        if label in VALUES:
            v = rng.choice(VALUES[label])
        elif label == "Age":
            v = str(rng.randint(18, 82))
        elif label == "Household size":
            v = str(rng.randint(1, 6))
        elif label == "Income":
            v = rng.choice(["Band 1", "Band 2", "Band 3", "Band 4"])
        elif label == "Big 5 scores":
            v = "see score fields below"
        else:
            v = f"{rng.randint(0, 10)}"
        lines.append(f"{label}: {v}")

    # Real format: a prose header, the score lines, then an explanatory gloss.
    # crt2_score is deliberately included — it does not start with `score_`.
    def blk(header, names, gloss):
        lines = [f"The person's {header} are the following:"]
        lines += [f"{n} = {rng.randint(10, 90) / 10:.3f} "
                  f"({rng.randint(1, 99)}th percentile)" for n in names]
        lines.append(gloss)
        return "\n".join(lines)

    score_lines = [
        blk("Big 5 scores", SCORES[:5],
            "Openness reflects curiosity and receptiveness to new experiences. "
            "Each score ranges from 1 to 5, and a higher score indicates a "
            "greater display of the associated traits."),
        blk("need for cognition score", ["score_needforcognition"],
            "The score ranges from 1 to 5."),
        blk("agentic / communal value scores", ["score_agency", "score_communion"],
            "Each score ranges from 1 to 9."),
        blk("CRT score", ["crt2_score"],
            "The score ranges from 0 to 4."),
        blk("basic empathy scale score", ["score_BES"],
            "The score ranges from 1 to 5."),
    ]

    filler = " ".join(
        rng.choice([
            "I grew up outside a mid-sized town and never really left.",
            "Most weeks look the same, which suits me fine.",
            "I read more than I watch, though that has been slipping.",
            "People say I am blunt; I would say direct.",
            "I cook badly and enthusiastically.",
            "I keep a short list of people I would drop anything for.",
        ]) for _ in range(60)
    )

    return (f"Respondent {pid}\n\n" + "\n".join(lines) + "\n\n"
            + "\n\n".join(score_lines) + "\n\n" + ESSAY + "\n\n" + PHANTOM
            + "\n\n" + filler + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=64, help="total personas")
    ap.add_argument("--chunks", type=int, default=2)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", default="tests/_fixture_personas")
    ap.add_argument("--force", action="store_true",
                    help="allow overwriting non-synthetic parquet files")
    args = ap.parse_args()

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    # Never clobber real data. Only files this script produced are removable.
    foreign = [p for p in outdir.glob("*.parquet")
               if not p.name.startswith("synthetic-")]
    if foreign and not args.force:
        raise SystemExit(
            f"{outdir} contains {len(foreign)} non-synthetic parquet file(s) "
            f"(e.g. {foreign[0].name}). Refusing to overwrite real persona data.\n"
            f"Use --out tests/_fixture_personas for a fixture, or --force if you "
            f"really mean to replace them."
        )
    for stale in outdir.glob("synthetic-*.parquet"):
        stale.unlink()

    rng = random.Random(args.seed)
    rows = []
    for i in range(args.n):
        pid = f"p{i:04d}"
        rows.append({"pid": pid, "persona_summary": make_summary(rng, pid),
                     "persona_text": "", "persona_json": "{}"})

    per = -(-len(rows) // args.chunks)
    manifest = {"source": "SYNTHETIC FIXTURE — not Twin-2K-500",
                "revision": None, "n": len(rows), "files": {}}
    for c in range(args.chunks):
        part = rows[c * per:(c + 1) * per]
        if not part:
            continue
        path = outdir / f"synthetic-{c:03d}.parquet"
        pd.DataFrame(part).to_parquet(path, engine="pyarrow", index=False)
        manifest["files"][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()

    (outdir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote {len(rows)} synthetic personas to {outdir} "
          f"across {len(manifest['files'])} chunk(s)")


if __name__ == "__main__":
    main()
