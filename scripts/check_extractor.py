#!/usr/bin/env python3
"""Check the covariate extractor against REAL persona data before trusting it.

This is the highest-priority verification step. `personas.py` was written against
the *documented* shape of Twin-2K-500's `persona_summary`. If the real format
differs, extraction fails silently-ish — you get a personas.csv full of blanks
rather than an error, and the actor/partner regressions are built on nothing.

    python scripts/check_extractor.py --dir data/personas
    python scripts/check_extractor.py --dir data/personas --show p0007

Reports per-field coverage across every persona, sample values, the score_*
fields found, and any labels that look like fields but are NOT on the whitelist
(these are the essay lines a naive regex would turn into phantom columns).

Exit code 1 if any whitelist field is below 100% coverage.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from agent_srm.personas import (SCORE_SECTIONS, WHITELIST,  # noqa: E402
                                 extract_covariates)

LOOKS_LIKE_FIELD = re.compile(r"^\s*([A-Z][A-Za-z0-9 /&'-]{2,40})\s*:\s*\S", re.M)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data/personas")
    ap.add_argument("--field", default="persona_summary")
    ap.add_argument("--show", default=None, help="print one persona's extraction")
    ap.add_argument("--limit", type=int, default=0, help="check only the first N")
    args = ap.parse_args()

    files = sorted(Path(args.dir).expanduser().glob("*.parquet"))
    if not files:
        raise SystemExit(f"no parquet files in {args.dir}")
    df = pd.concat([pd.read_parquet(p, engine="pyarrow") for p in files],
                   ignore_index=True)
    if args.limit:
        df = df.head(args.limit)

    n = len(df)
    field_hits = Counter()
    score_names = Counter()
    phantom = Counter()
    lengths = []
    genders = Counter()
    missing_sections = Counter()
    examples: dict[str, str] = {}

    for pid, summary in zip(df["pid"].astype(str), df[args.field].astype(str)):
        covs, scores, pcts, missing = extract_covariates(summary)
        lengths.append(len(summary))
        for k, v in covs.items():
            field_hits[k] += 1
            examples.setdefault(k, v)
        for s in scores:
            score_names[s] += 1
        genders[covs.get("Gender")] += 1
        for m in missing:
            missing_sections[m] += 1
        for label in set(LOOKS_LIKE_FIELD.findall(summary)):
            if label not in WHITELIST:
                phantom[label] += 1
        if args.show and pid == args.show:
            print(f"\n=== {pid} ===")
            for k, v in covs.items():
                print(f"  {k:26s} {v[:60]}")
            for s, v in scores.items():
                print(f"  {s:26s} {v} ({pcts.get(s)}th pct)")
            print()

    print(f"personas checked: {n}")
    print(f"summary length: min {min(lengths):,}  median "
          f"{sorted(lengths)[len(lengths)//2]:,}  max {max(lengths):,} chars\n")

    print(f"DEMOGRAPHIC FIELD COVERAGE ({len(WHITELIST)} expected)")
    bad = []
    for label in WHITELIST:
        hits = field_hits[label]
        pct = 100 * hits / n
        mark = "ok " if hits == n else "MISS"
        if hits != n:
            bad.append(label)
        print(f"  [{mark}] {label:26s} {hits:5d}/{n} ({pct:5.1f}%)  "
              f"e.g. {examples.get(label, '—')[:40]}")

    print(f"\nGENDER: {dict(genders)}")

    print(f"\nSCORE FIELDS FOUND ({len(score_names)} distinct)")
    for s, c in score_names.most_common():
        print(f"  {s:36s} {c}/{n}")

    if missing_sections:
        print("\nOPTIONAL SECTIONS MISSING (warning only)")
        for s, c in missing_sections.most_common():
            print(f"  {s:26s} {c}/{n}")

    if phantom:
        print("\nPHANTOM LABELS — look like fields, correctly NOT extracted")
        print("  (this is why extraction is by whitelist, not by regex)")
        for label, c in phantom.most_common(15):
            print(f"  {label:36s} {c}/{n}")

    if bad:
        print(f"\nFAIL: {len(bad)} demographic field(s) below full coverage: {bad}")
        print("Fix the label list or the parser in src/agent_srm/personas.py "
              "before running a study.")
        raise SystemExit(1)
    print(f"\nPASS: all {len(WHITELIST)} demographic fields present for every persona.")
    print(f"Psychometric batteries ({', '.join(SCORE_SECTIONS)}) are prose sections,")
    print("not `Label: value` fields — they are captured by the score parser above.")


if __name__ == "__main__":
    main()
