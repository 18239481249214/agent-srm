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

## Assembling the archive

The three release assets do **not** unpack into a working layout on their own.
Unpacking them side by side will leave the persona directory empty and the
runner will fail on a checksum error. Assemble them as follows.

### 1. Unpack the code

Unpack `1_code_and_dataset.zip`. Its contents become the project root — call it
`agent-srm`:

```
agent-srm/
├── src/  scripts/  configs/  prompts/  tests/  dataset/
├── data/personas/MANIFEST.json
├── README.md  REPRODUCE.md  DATA.md
├── LICENSE  LICENSE-DATA  .gitignore  pyproject.toml
```

Equivalently, clone this repository instead — the contents are identical.

### 2. Unpack the persona corpus *into* the project

`2_persona_corpus.zip` contains seven parquet files and a copy of
`MANIFEST.json` at its root. These belong **inside** `data/personas/`, not
beside the project:

```
agent-srm/data/personas/
├── MANIFEST.json
├── persona_chunk_001.parquet
├── persona_chunk_002.parquet
├── persona_chunk_003.parquet
├── persona_chunk_004.parquet
├── persona_chunk_005.parquet
├── persona_chunk_006.parquet
└── persona_chunk_007.parquet
```

`MANIFEST.json` is identical in both archives; overwriting is harmless.

### 3. Unpack the run outputs (optional)

Only needed to inspect the original collection. `3_run_outputs.zip` contains
nine arm directories, which belong in a `runs/` folder at the project root:

```
agent-srm/runs/
├── k10_n5__deepseek-v4-pro/
├── k10_n5_s2__deepseek-v4-pro/
├── kimi26_n5_s2__kimi-k2-6/
├── nemotron_n5_s2__nemotron-3-ultra/
├── persona_k10_n5_kimi26__kimi-k2-6/
├── persona_k10_n5_kimi26_null__kimi-k2-6/
├── persona_k10_n5_nemotron__nemotron-3-ultra/
├── persona_k10_n5_nemotron_null__nemotron-3-ultra/
└── persona_k10_n5_null__deepseek-v4-pro/
```

Each arm directory contains `transcripts.jsonl`, `ratings.csv`,
`ratings_wide.csv`, `personas.csv` and `manifest.json`.

### 4. Complete layout

```
agent-srm/
├── configs/                    nine arm configurations
├── prompts/                    prompt templates
├── scripts/                    tooling
├── src/agent_srm/              the runner
├── tests/                      acceptance suite
├── dataset/                    combined ratings and covariates
├── data/personas/              MANIFEST.json + 7 parquet chunks   ← zip 2
├── runs/                       nine arm directories (optional)    ← zip 3
├── README.md  REPRODUCE.md  DATA.md
└── pyproject.toml
```

### 5. Verify before running

```bash
pip install -e ".[dev]"
python scripts/check_extractor.py --dir data/personas
```

`check_extractor.py` must report **14/14 demographic fields at 2058/2058** and
**46 score fields**, and exits non-zero otherwise. If it reports no parquet
files found, the corpus was unpacked to the wrong location — see step 2.

Do **not** run `scripts/index_personas.py` against the supplied corpus. It
overwrites `MANIFEST.json` with checksums of whatever is present, discarding
the collection-time record that lets you confirm the files are identical to
those used here.

### 6. Run

```bash
export DEEPSEEK_API_KEY=...        # or OPENROUTER_API_KEY, per arm
python -m agent_srm.cli validate --config configs/k10_n5.yaml
python -m agent_srm.cli run --config configs/k10_n5.yaml --workers 4
```

Each arm is 50 blocks and requires API access and billing on your own account.
See `REPRODUCE.md` §6 for endpoints and per-arm token budgets.

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

Assemble the archive as above, then see `REPRODUCE.md` for full setup,
endpoints and operational notes.

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
