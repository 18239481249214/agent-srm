# Data availability

Everything for this study is hosted in this repository. Small, browsable files
are committed directly; the two large archives are attached to the **v1.0.0
release**. An OSF project links to this repository.

## In the repository

| Path | Contents |
|---|---|
| `dataset/all_ratings_wide.csv` | All 9,000 directed ratings across every arm, one row per rating, labelled by collection, model, persona mode, seed and source run |
| `dataset/all_personas.csv` | Persona covariates (demographics and psychometric scores) per arm |
| `dataset/runs_manifest.csv` | One row per arm: model, resolved snapshot, provider, seed, design digest, yield, token budgets |
| `configs/` | The exact configuration used for every arm |
| `prompts/` | Prompt templates, hashed into every run manifest |
| `src/`, `scripts/`, `tests/` | The runner, tooling, and acceptance suite |
| `data/personas/MANIFEST.json` | SHA-256 checksums for the persona chunks used |

## In the v1.0.0 release

See **Releases → v1.0.0**. GitHub publishes a SHA-256 for each asset.

| Asset | Contents |
|---|---|
| `1_code_and_dataset.zip` | Snapshot of this repository |
| `2_persona_corpus.zip` | The seven Twin-2K-500 parquet chunks used for collection, plus `MANIFEST.json` |
| `3_run_outputs.zip` | Full per-arm outputs for all nine arms |

`3_run_outputs.zip` contains, for each arm:

- `transcripts.jsonl` — one record per dyad, containing every prompt **as
  actually sent**, all conversational turns with token counts and stop reasons,
  and each rating with its raw model response
- `ratings.csv`, `ratings_wide.csv`, `personas.csv`, `manifest.json`

## Persona corpus

Persona descriptions are from **Twin-2K-500**:

    https://huggingface.co/datasets/LLM-Digital-Twin/Twin-2K-500

The dataset is licensed **CC BY 4.0**, and the exact chunks used are
redistributed here under the same license in `2_persona_corpus.zip`. Because the
files themselves are included, replication does not depend on the upstream
dataset remaining unchanged.

Cite the original:

```
@dataset{twin2k500,
  author    = {Toubia, Olivier and Gui, George Z. and Peng, Tianyi and
               Merlau, Daniel J. and Li, Ang and Chen, Haozhe},
  title     = {Twin-2K-500: A Dataset for Building Digital Twins of 2,000 People},
  year      = {2025},
  publisher = {Hugging Face},
  howpublished = {\url{https://arxiv.org/abs/2505.17479}}
}
```

`MANIFEST.json` records a SHA-256 per chunk. The runner verifies every file at
load and refuses to start on a mismatch. Its `revision` field is null — the
upstream HuggingFace revision was not captured at download time. Since the files
themselves are included here, this does not affect reproducibility.

## Reproducing

See `REPRODUCE.md`. In short:

```bash
pip install -e ".[dev]"
# unpack 2_persona_corpus.zip into data/personas/ first
python scripts/check_extractor.py --dir data/personas
python -m agent_srm.cli run --config configs/<arm>.yaml --workers 4
```

Do not run `scripts/index_personas.py` against the supplied corpus unless you
intend to replace the checksums — it overwrites `MANIFEST.json` with hashes of
whatever is present, discarding the collection-time record.

Design-level randomisation is exact: the same seed and the same persona
checksums reproduce identical block composition, dyad membership, initiator
assignment and item orders, verifiable through the `design_digest` recorded in
every manifest. Model generation is not reproducible — hosted inference
endpoints accept a sampling seed and generally ignore it.

## Verification performed

`scripts/check_empty.py` is included and reports empty and truncated messages by
run and stop reason:

```bash
python scripts/check_empty.py            # expects runs/ to be populated
```

The following additional checks were run on the collected data before analysis.
The scripts used were written ad hoc during collection and are **not included in
this archive**; the results are reported here, and the underlying data in
`3_run_outputs.zip` supports independent re-verification.

- **Message completeness.** Across all nine arms and 45,000 generated messages,
  the final dataset contains zero empty and zero truncated messages.
- **Output chain.** Every one of the 9,000 ratings was traced from
  `raw_response` — the model's literal reply, stored before parsing — through
  the runner's parse to the CSV row, using a parser sharing no code with
  `src/agent_srm/rating.py`. All three stages agreed for every rating.
- **Experimental isolation.** Across the six persona arms, 5,984 directions were
  checked using each persona's forward-flow word chain, which is unique per
  respondent. No persona's system prompt contained text unique to its partner.
- **Structural validity.** Every block is a complete round robin with no
  self-ratings, no persona appears in more than one block, long and wide CSVs
  describe identical data, and each arm reports a single model snapshot.

Anyone wishing to repeat these checks can do so against `3_run_outputs.zip`,
which contains the raw records each check was derived from.

## Licensing

Software is MIT (`LICENSE`). Collected data is CC BY 4.0 (`LICENSE-DATA`).
Twin-2K-500 is CC BY 4.0 and redistributed under that license. Language-model
outputs in the transcripts are subject to each provider's terms.
