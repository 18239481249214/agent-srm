"""Persona loading, checksum verification, and whitelist covariate extraction.

The runner never touches HuggingFace (§3.2). It globs local parquet, verifies
SHA-256 against the manifest written by scripts/fetch_personas.py, and fails
loudly on mismatch.

NOTE FOR IMPLEMENTERS: the covariate extractor below is written against the
*documented* shape of `persona_summary` and has not been run against a real
Twin-2K-500 chunk. Verify WHITELIST coverage on one chunk before trusting
personas.csv. Nothing here alters what the model sees — the entire
persona_summary string is passed verbatim (§3.3).
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

# Demographic fields that really are rendered as `Label: value` lines.
# Verified 2058/2058 against the real Twin-2K-500 chunks.
WHITELIST = [
    "Geographic region", "Gender", "Age", "Education level", "Race",
    "Citizen of the US", "Marital status", "Religion", "Religious attendance",
    "Political affiliation", "Income", "Political views", "Household size",
    "Employment status",
]

# The psychometric batteries are NOT `Label: value` fields in the real file.
# They appear as prose headers ("The person's CRT score is the following:")
# followed by `name = value (Nth percentile)` lines and an explanatory gloss.
# The score parser below captures them, so they are not whitelist entries.
SCORE_SECTIONS = [
    "Big 5 scores", "CRT score", "minimalism score",
    "syllogism score", "numeracy score",
]

OPTIONAL_SECTIONS = [
    "forward flow",
]

# Matches `score_openness = 3.6 (41st percentile)`, `wave1_score_conscientiousness
# = 4 (53rd percentile)`, and `crt2_score = 2 (59th percentile)`. The earlier
# version anchored on a `score_` prefix and silently dropped crt2_score.
_SCORE_RE = re.compile(
    r"^\s*([A-Za-z][A-Za-z0-9_]*)\s*=\s*(-?\d+(?:\.\d+)?)"
    r"(?:\s*\((\d+)(?:st|nd|rd|th)?\s*percentile\))?\s*$",
    re.MULTILINE,
)


class PersonaError(RuntimeError):
    pass


@dataclass
class Persona:
    pid: str
    summary: str
    covariates: dict[str, str] = field(default_factory=dict)
    scores: dict[str, float] = field(default_factory=dict)
    percentiles: dict[str, int] = field(default_factory=dict)
    missing_sections: list[str] = field(default_factory=list)

    @property
    def gender(self) -> str | None:
        return self.covariates.get("Gender")


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def extract_covariates(summary: str) -> tuple[dict[str, str], dict[str, float],
                                              dict[str, int], list[str]]:
    """Whitelist extraction only — never pattern-match `^Label:` (§3.3).

    The free-response essays contain lines that look like fields; one persona's
    essay produces phantom 'Resilience:', 'Integrity:', 'Empathy:' labels. A
    naive regex creates spurious columns.
    """
    covs: dict[str, str] = {}
    for label in WHITELIST:
        m = re.search(rf"^\s*{re.escape(label)}\s*:\s*(.+?)\s*$", summary,
                      re.MULTILINE | re.IGNORECASE)
        if m:
            covs[label] = m.group(1).strip()

    scores: dict[str, float] = {}
    pcts: dict[str, int] = {}
    for name, val, pct in _SCORE_RE.findall(summary):
        if "score" not in name.lower():        # guard against stray `x = 3` lines
            continue
        try:
            scores[name] = float(val)
        except ValueError:
            continue
        if pct:
            pcts[name] = int(pct)

    low = summary.lower()
    missing = [s for s in OPTIONAL_SECTIONS if s.lower() not in low]
    return covs, scores, pcts, missing


def verify_manifest(files: list[Path], manifest_path: Path | None) -> dict:
    if manifest_path is None or not manifest_path.exists():
        raise PersonaError(
            f"persona manifest not found at {manifest_path}; run scripts/fetch_personas.py "
            "(or scripts/make_synthetic_personas.py for a dry run) first"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = manifest.get("files", {})
    for p in files:
        want = expected.get(p.name)
        if want is None:
            raise PersonaError(f"{p.name} is not listed in {manifest_path}")
        got = sha256_file(p)
        if got != want:
            raise PersonaError(
                f"checksum mismatch for {p.name}: manifest {want[:12]}… but file {got[:12]}…"
            )
    listed = set(expected) - {p.name for p in files}
    if listed:
        raise PersonaError(f"manifest lists files not present on disk: {sorted(listed)}")
    return manifest


def load_personas(directory: str | Path, field_name: str = "persona_summary",
                  manifest: str | Path | None = None, verify: bool = True,
                  naming_enabled: bool = False) -> tuple[dict[str, Persona], dict]:
    d = Path(directory)
    files = sorted(d.glob("*.parquet"))
    if not files:
        raise PersonaError(f"no parquet chunks found in {d}")

    manifest_data = {}
    if verify:
        manifest_data = verify_manifest(
            files, Path(manifest) if manifest else d / "MANIFEST.json"
        )

    frames = [pd.read_parquet(p, engine="pyarrow") for p in files]
    df = pd.concat(frames, ignore_index=True)

    if "pid" not in df.columns:
        raise PersonaError("persona chunks are missing the `pid` column")
    if field_name not in df.columns:
        raise PersonaError(f"persona chunks are missing the `{field_name}` column")
    if df["pid"].duplicated().any():
        dupes = df.loc[df["pid"].duplicated(), "pid"].unique()[:5].tolist()
        raise PersonaError(f"duplicate pid across chunks, e.g. {dupes}")

    personas: dict[str, Persona] = {}
    warnings: list[str] = []
    for pid, summary in zip(df["pid"].astype(str), df[field_name].astype(str)):
        if not summary.strip():
            raise PersonaError(f"empty {field_name} for pid {pid}")
        covs, scores, pcts, missing = extract_covariates(summary)
        absent = [w for w in WHITELIST if w not in covs]
        if absent:
            raise PersonaError(
                f"pid {pid} is missing required whitelist field(s): {absent}"
            )
        gender = covs.get("Gender")
        if gender not in {"Male", "Female"}:
            msg = f"pid {pid} has Gender={gender!r}, expected Male or Female"
            # A10: hard failure only when names actually need to be matched
            if naming_enabled:
                raise PersonaError(msg + " (required for naming.match_gender)")
            warnings.append(msg)
        if missing:
            warnings.append(f"pid {pid} missing optional section(s): {missing}")
        personas[pid] = Persona(pid, summary, covs, scores, pcts, missing)

    return personas, {"manifest": manifest_data, "exceptions": warnings,
                      "n_personas": len(personas),
                      "files": [p.name for p in files]}
