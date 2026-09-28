# Data availability

## What is in this repository

| Path | Contents |
|---|---|
| `dataset/all_ratings_wide.csv` | All directed ratings across every arm, one row per rating, labelled by collection, model, persona mode, seed and source run |
| `dataset/all_personas.csv` | Persona covariates (demographics and psychometric scores) per arm |
| `dataset/runs_manifest.csv` | One row per arm: model, resolved snapshot, provider, seed, design digest, yield, token budgets |
| `configs/` | The exact configuration used for every arm |
| `prompts/` | Prompt templates, hashed into every run manifest |

## What is archived on OSF

Full transcripts and per-arm outputs are too large for version control:

- `transcripts.jsonl` — one record per dyad, containing every prompt **as
  actually sent**, all conversational turns with token counts and stop reasons,
  and each rating with its raw model response
- per-arm `ratings.csv`, `ratings_wide.csv`, `personas.csv`, `manifest.json`

OSF project: **<add URL>**

## What is not redistributed

Persona descriptions derive from **Twin-2K-500**, a third-party dataset:

    https://huggingface.co/datasets/LLM-Digital-Twin/Twin-2K-500

Download it separately. `data/personas/MANIFEST.json` in this repository
contains SHA-256 checksums for the exact chunks used, so you can verify you
have byte-identical files. The runner checks these at load and refuses to run
on a mismatch.

## Reproducing

See `REPRODUCE.md`. In short:

```bash
pip install -e ".[dev]"
python scripts/index_personas.py  --dir data/personas --revision <hf-revision-sha>
python scripts/check_extractor.py --dir data/personas
python -m agent_srm.cli run --config configs/<arm>.yaml --workers 4
```

Design-level randomisation is exact: the same seed and the same persona
checksums reproduce identical block composition, dyad membership, initiator
assignment and item orders, verifiable through the `design_digest` recorded in
every manifest. Model generation is not reproducible — hosted inference
endpoints accept a sampling seed and generally ignore it.

## Verification

Audit scripts are included and were run on the released data:

```bash
python scripts/check_empty.py
python scripts/verify_run.py    runs/<arm>
python scripts/audit_ratings.py runs/<arm> --sample 10
python scripts/check_leakage2.py runs/<arm>
python scripts/package_dataset.py
```

`audit_ratings.py` traces every rating from the model's literal reply through
to the CSV using a parser independent of the runner's. `check_leakage2.py`
confirms no persona's prompt contained text unique to its partner.

## Licensing

Software is MIT (`LICENSE`). Collected data is CC BY 4.0 (`LICENSE-DATA`).
Twin-2K-500 and the language-model outputs are governed by their own terms.
