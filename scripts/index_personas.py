#!/usr/bin/env python3
"""Index parquet chunks you ALREADY have into data/personas/MANIFEST.json.

Use this when the Twin-2K-500 chunks are already on disk. It does not download
anything. It records SHA-256 for each file so the runner can verify them at
load time and fail loudly if a chunk changes underneath a study.

    python scripts/index_personas.py --dir data/personas
    python scripts/index_personas.py --dir ~/data/twin2k --revision <hf-sha>

If you know the HuggingFace revision SHA the files came from, pass --revision so
the archive records provenance. If you don't, it is recorded as null and the
checksums still pin the exact bytes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

EXPECTED_COLUMNS = {"pid", "persona_summary"}


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data/personas")
    ap.add_argument("--revision", default=None,
                    help="HuggingFace revision SHA, if known")
    ap.add_argument("--field", default="persona_summary")
    args = ap.parse_args()

    d = Path(args.dir).expanduser()
    files = sorted(d.glob("*.parquet"))
    if not files:
        raise SystemExit(f"no parquet files found in {d}")

    manifest = {
        "source": "LLM-Digital-Twin/Twin-2K-500 (full_persona)",
        "revision": args.revision,
        "field": args.field,
        "files": {},
    }

    total, all_pids = 0, set()
    print(f"indexing {len(files)} chunk(s) in {d}\n")
    for p in files:
        df = pd.read_parquet(p, engine="pyarrow")
        missing = EXPECTED_COLUMNS - set(df.columns)
        if missing:
            raise SystemExit(f"{p.name}: missing required column(s) {sorted(missing)}")
        empty = int(df[args.field].isna().sum() + (df[args.field].astype(str)
                                                   .str.strip() == "").sum())
        dupes = int(df["pid"].duplicated().sum())
        manifest["files"][p.name] = sha256_file(p)
        total += len(df)
        all_pids |= set(df["pid"].astype(str))
        print(f"  {p.name:40s} {len(df):5d} rows  "
              f"empty={empty}  dupes-in-chunk={dupes}  "
              f"sha={manifest['files'][p.name][:12]}…")

    manifest["n"] = total
    cross = total - len(all_pids)
    print(f"\ntotal rows: {total}  unique pids: {len(all_pids)}  "
          f"cross-chunk duplicates: {cross}")
    if cross:
        raise SystemExit("FAIL: pid is not unique across chunks — the runner "
                         "requires a unique primary key")

    out = d / "MANIFEST.json"
    out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    if args.revision is None:
        print("note: no --revision given, provenance recorded as null "
              "(checksums still pin the bytes)")


if __name__ == "__main__":
    main()
