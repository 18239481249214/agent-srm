# REPRODUCE.md

Everything needed to reproduce this collection, including operational knowledge
that is not evident from the code. **Read §8 (Limitations) before interpreting
any output.**

Companion documents: `README.md` (tool reference), `agent_srm_design_v2.md`
(design specification and rationale).

---

## 1. What was collected

Round-robin social interactions between LLM personas, producing data structured
for Social Relations Model analysis in `TripleR`.

**Nine arms**, each 50 blocks of 5 personas at k=10 — 250 personas, 500 dyads and
1,000 directed ratings per arm. 9,000 directed ratings in total.

### Collection 1 — shared draw, paired across models

All six arms share seed `20260818` and therefore identical block composition,
dyad membership, initiator orientation and item orders. They are joinable on
`run` + `rater` + `target`, so between-arm differences are attributable to the
model or to the persona manipulation rather than to sampling.

| Run directory | Model | Personas |
|---|---|---|
| `k10_n5__deepseek-v4-pro` | DeepSeek V4 Pro | full |
| `persona_k10_n5_null__deepseek-v4-pro` | DeepSeek V4 Pro | none |
| `persona_k10_n5_nemotron__nemotron-3-ultra` | Nemotron 3 Ultra | full |
| `persona_k10_n5_nemotron_null__nemotron-3-ultra` | Nemotron 3 Ultra | none |
| `persona_k10_n5_kimi26__kimi-k2-6` | Kimi K2.6 | full |
| `persona_k10_n5_kimi26_null__kimi-k2-6` | Kimi K2.6 | none |

### Collection 2 — independent draws

One persona arm per model, each from a separate random draw of the same
2,058-persona pool. These support generalisation beyond a single sample; they are
*not* paired with Collection 1 or with each other, and roughly 30 personas
overlap between any two draws by chance.

| Run directory | Model | Seed |
|---|---|---|
| `k10_n5_s2__deepseek-v4-pro` | DeepSeek V4 Pro | 20260901 |
| `nemotron_n5_s2__nemotron-3-ultra` | Nemotron 3 Ultra | 20260902 |
| `kimi26_n5_s2__kimi-k2-6` | Kimi K2.6 | 20260903 |

Exact seeds, design digests, model snapshots and token budgets for every arm are
recorded in `dataset/runs_manifest.csv` and in each arm's `manifest.json`.

---

## 2. Environment

Python 3.11+.

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
python -m pytest -q                  # expect 33 passed
```

### Windows notes

- Activation is `.\.venv\Scripts\Activate.ps1` — the leading `.\` is required.
- If script execution is disabled: `Set-ExecutionPolicy -Scope Process
  -ExecutionPolicy Bypass`, which applies to that window only.
- Application Control may block `agent_srm.exe` and `pytest.exe` — the launcher
  shims pip generates. Use `python -m agent_srm.cli …` and `python -m pytest`.
- pyarrow may warn that `arrow_acero.dll` is blocked. Harmless; parquet reading
  does not use it.
- **A virtual environment cannot be copied between machines.** It records an
  absolute path to the Python that created it and contains compiled binaries tied
  to one interpreter version. Delete `.venv` and rebuild it on each machine.
- API keys held in environment variables do not survive closing the window.

---

## 3. Persona data

Twin-2K-500 (`LLM-Digital-Twin/Twin-2K-500`, `full_persona` config), Toubia et
al. 2025. Seven parquet chunks of 294 rows, 2,058 personas.

```bash
python scripts/index_personas.py  --dir data/personas --revision <hf-revision-sha>
python scripts/check_extractor.py --dir data/personas
```

`index_personas.py` records a SHA-256 per chunk in `MANIFEST.json`; the runner
verifies these at load and fails on mismatch, so a chunk cannot change underneath
a study unnoticed.

`check_extractor.py` must report **14/14 demographic fields at 2058/2058** and
**46 score fields**, and exits non-zero otherwise. Run it before any collection:
if the summary format differs from what the parser expects, the result is blank
columns in `personas.csv` rather than an error.

Its output includes a "phantom labels" section listing lines that resemble fields
but are prose — `Resilience:`, `Integrity:`, `Empathy:`, and the section header
`The person's CRT score is the following`. These are correctly ignored;
extraction is by explicit whitelist rather than by `^Label:` matching, because a
naive regex invents columns from essay text.

Note that `crt2_score` does not follow the `score_*` naming convention used by
the other 45 measures. An earlier parser anchored on that prefix and silently
dropped it.

**Incomplete measures in the source data** (not parsing failures):
`score_forwardflow` 2057/2058, `score_discount` and `score_presentbias`
1866/2058, `score_riskaversion` 1825/2058, `score_lossaversion` **664/2058**.
Treat loss aversion as a one-third subsample if used as a moderator.

Raw scores are **not comparable across batteries** — ranges run from 0–4 to
0–100. Use the `*_pct` percentile columns for cross-construct work.

---

## 4. Running an arm

```bash
export DEEPSEEK_API_KEY=...                       # see §6 for each arm
python -m agent_srm.cli validate --config configs/k10_n5.yaml
python -m agent_srm.cli run --config configs/k10_n5.yaml --workers 4
```

`validate` runs every configuration gate without an API call. `run` performs a
one-token smoke call before real work, so bad credentials fail in seconds.

**Pilot a single block against any new endpoint.** Copy the config, set
`n_blocks: 1` and a distinct `study_id`, and inspect the transcripts and your
provider's billing before committing hours. Every model added to this study
required at least one adjustment on first contact.

**Interruption is safe.** Blocks are checkpointed on completion and both CSVs are
rebuilt after every block, so the files on disk always describe the blocks
actually collected. Re-running the same command skips completed blocks. Power
loss, Ctrl+C and dropped connections all recover identically.

**Concurrency.** `--workers N` parallelises dyads within a block; turns within a
dyad and blocks themselves stay sequential. Two tests assert that parallel and
sequential runs produce identical rows in identical order at k=2 and k=10.
Observed at k=10, n=5 with personas: 7.2 min/block sequential, 3.1 min/block at
four workers.

**Rate limits.** Use `--rpm N` (or `execution.requests_per_minute`) rather than
relying on backoff. The gate is shared across worker threads, so `--workers 4
--rpm 15` emits 15 requests per minute in total. Set it below the provider's cap;
because the first request is not delayed, a short burst measures slightly above
the configured rate.

Null-persona arms hit rate caps more readily than persona arms: without a
13,000-character summary, calls return far faster, so the same worker count
produces many more requests per minute. OpenRouter applies stricter limits to new
accounts — 20 RPM was observed for `moonshotai/kimi-k2.6`.

---

## 5. The null arms

`personas.mode: none` loads the parquet as normal — blocks, dyads, seeds and
`personas.csv` are identical to a full run — but sends no persona text. Every
system prompt is the task framing alone (~399 characters against ~13,000), and it
is byte-identical to the framing embedded in the persona arms.

This makes each null a **paired floor**: any partner variance it shows is
generation noise rather than persona signal.

Because `personas.csv` retains the real measured traits for those personas, a
regression of SRM effects on Big Five scores in a null arm provides a
falsification test — those traits never reached the model, so they should predict
nothing.

---

## 6. Arm endpoints

All arms are OpenAI-compatible Chat Completions; only `base_url`, `model_id` and
the key variable differ.

| Arm | base_url | model_id | Env var |
|---|---|---|---|
| DeepSeek V4 Pro | `https://api.deepseek.com/v1` | `deepseek-v4-pro` | `DEEPSEEK_API_KEY` |
| Nemotron 3 Ultra | `https://openrouter.ai/api/v1` | `nvidia/nemotron-3-ultra-550b-a55b` | `OPENROUTER_API_KEY` |
| Kimi K2.6 | `https://openrouter.ai/api/v1` | `moonshotai/kimi-k2.6` | `OPENROUTER_API_KEY` |

Both OpenRouter arms pin a single provider with `allow_fallbacks: false` —
Nemotron to Baseten (NVFP4), Kimi K2.6 to CoreWeave. Provider pinning is
expressed through the arm's `reasoning:` field, which is an arbitrary request
passthrough rather than a reasoning-specific setting.

### Operational knowledge

**Reasoning tokens count against `max_tokens`.** This is the most consequential
gotcha in the project and it caused failures twice.

- At `rating.max_tokens: 512`, DeepSeek's reasoning trace consumed the entire
  budget and returned empty content on every rating call. A probe showed 281
  reasoning tokens against 27 tokens of answer on a *trivial* prompt.
- At `max_tokens: 4096`, Kimi K2.6 produced empty conversation messages in 8–12%
  of blocks: reasoning of 4,600–4,900 tokens exhausted the budget before any
  content was written. Raised to 12,288 (conversation) and 16,384 (rating), the
  longest observed generation reached 8,503 tokens.

Budgets therefore differ by arm. Exact values per arm are in
`dataset/runs_manifest.csv`.

**Verify reasoning behaviour rather than trusting documentation.** Third-party
sources listed DeepSeek V4 Pro as having reasoning disabled by default; on the
first-party API it reasons by default. `scripts/probe_endpoint.py` makes one
rating-shaped call and reports `finish_reason`, content, and the full usage block
including reasoning tokens.

Median reasoning per rating call differs by roughly an order of magnitude across
models — a measured quantity, not an assumption. See §8.

**Model aliases re-point silently.** `deepseek-v4-pro` resolved to a dated
snapshot; whatever the server returns in `model` is recorded as `model_snapshot`
on every row. Prefer requesting a pinned dated identifier where one is exposed.

**Gateways route across hosts.** OpenRouter serves each model from several
providers with automatic failover, and different hosts may serve different
quantisations. Both OpenRouter arms are pinned with fallbacks disabled so that an
outage produces an error rather than a silent reroute.

---

## 7. Verification

All of the following were run on the final dataset.

```bash
python scripts/check_empty.py                              # all runs
python scripts/verify_run.py runs/<study>__<arm>
python scripts/audit_ratings.py runs/<study>__<arm> --sample 10
python scripts/check_long_wide.py
python scripts/package_dataset.py [--write]
```

**`verify_run.py`** — round robin complete, no self-ratings, no persona in more
than one block, scores in range and non-degenerate, message counts and
alternation, no construct label in any prompt, item-order randomisation, single
model snapshot, flag rates.

**`audit_ratings.py`** — the load-bearing check. Every rating passes through
three stages: `raw_response` (the model's literal reply, stored before
processing), `parsed` (the runner's interpretation), and the CSV row. This script
re-parses stage 1 with a parser sharing no code with `src/agent_srm/rating.py`
and confirms all three agree. It was validated against three deliberately
corrupted datasets — an altered CSV value, a `parsed` value not matching the raw
text, and items remapped by position rather than by name — and detects all three.

`--sample N` prints raw replies beside CSV rows for human inspection. The model's
JSON key order frequently differs from the presentation order, because items are
randomised per call; the CSV maps by **name**, never by position.

**`package_dataset.py`** verifies every arm and then, with `--write`, produces:

| File | Contents |
|---|---|
| `dataset/all_ratings_wide.csv` | all 9,000 directed ratings, labelled by collection, model, persona mode, seed and source run |
| `dataset/all_personas.csv` | persona covariates per arm |
| `dataset/runs_manifest.csv` | one row per arm: model, snapshot, provider, seed, digest, yield, token budgets |

It refuses to write if any arm fails a structural check.

### Data-quality history

Empty messages were found in the initial collection and traced to two causes.

**Truncation** — a model's reasoning exhausting `max_tokens` before any content
was written. This affected 24 of 50 blocks in one Kimi arm and 31 of 50 in the
other, plus isolated blocks elsewhere. Because exclusion would have removed half
of an arm and would have correlated with conversations long enough to trigger
truncation, affected blocks were identified with `scripts/prune_blocks.py`, their
checkpoints removed, and the blocks re-collected under raised budgets.

**Stochastic non-response** — the model finishing normally and returning nothing,
at a background rate of roughly 0.02–0.12% and more frequent in null arms.
Affected blocks were re-collected until clean.

**The final dataset contains zero empty messages and zero truncated messages
across all 45,000 generated messages.** A subset of blocks therefore originates
from a second collection pass under a larger `max_tokens` than the original.

---

## 8. Limitations

**Read before interpreting results.**

### Design

- **Reasoning depth is confounded with model family.** Each arm ran at its
  provider's default reasoning behaviour, which differs by provider, and Kimi
  K2.6 cannot disable reasoning. Median reasoning tokens per rating call differ by
  roughly an order of magnitude across arms. A between-arm difference cannot be
  attributed to the model alone. Per-arm reasoning settings exist in config for a
  deliberate follow-up.
- **Serving precision is confounded with provider.** Nemotron 3 Ultra was served
  at NVFP4 (its native training format); DeepSeek's serving precision is not
  disclosed.
- **Generation-level reproducibility is not achieved.** Hosted endpoints accept a
  sampling seed and generally ignore it; `seed_honored` is recorded per call.
  Design-level randomisation *is* exact and verifiable via `design_digest`.
- **Rating temperature 0.0 is not deterministic** for reasoning models, since
  reasoning traces vary. Residual rating noise remains and enters the error term.
- **Personas occasionally invent names for themselves** despite naming being
  disabled — observed in roughly 1 of 12 dyads in an early pilot, not measured
  systematically. If the rate varies by target, it introduces an uncontrolled
  information asymmetry.
- **Sampling was unstratified.** Personas were drawn at random without matching
  on demographics or traits; report the distribution of the sampled personas
  against the full 2,058.
- **Kimi K2.6 is an agentic coding model** where the other two are
  general-purpose. Model *purpose* is a live alternative explanation for any
  difference it shows.

### Implementation

- **A subset of blocks was re-collected** under raised token budgets (§7).
- **`probe`, `replay` and `quarantine-report`** are specified in
  `agent_srm_design_v2.md` (§6.4, §9.6, §9.3) but were not implemented;
  `scripts/probe_endpoint.py` covers part of the first.
- **`scripts/fetch_personas.py` and `scripts/fetch_names.py` do not exist.**
  `index_personas.py` covers the case where chunks are already local; naming was
  disabled throughout.
- **The persona manifest may record `revision: null`** if indexed without
  `--revision`. Checksums pin the exact bytes, so a third party can verify file
  identity, but not independently re-fetch the matching version from HuggingFace.
- **Rating-call input and output token counts are not logged** (only reasoning
  tokens), so `scripts/cost_report.py` estimates that portion.
- **Model licences require verification before transcripts are republished.**
  DeepSeek is MIT; Kimi's terms and Nemotron's OpenMDW licence should be checked
  against the intended use.
- **Independent code review is outstanding.** The runner has been verified by the
  audit scripts in §7 and by 33 automated tests, but has not been read by anyone
  other than its authors. The audit scripts confirm that recorded data is
  internally consistent and that ratings trace unchanged from model output to
  CSV; they cannot establish that the interaction was constructed exactly as
  specified.

### Scale

- The ceiling is `floor(2058 / block_size)` — 411 blocks at n=5. Personas never
  repeat within an arm; `TripleR` requires independent groups.

---

## 9. Analysis

Analysis is deliberately out of scope for the tool; separating collection from
analysis is what makes the archive auditable.

```r
library(TripleR)
d <- read.csv("runs/k10_n5__deepseek-v4-pro/ratings_wide.csv")
RR(warm/kind + capable/effective ~ rater * target | run, data = d)
```

**Confirm the header reports `n = 50` and `average group size = 5`.** A silent
subset in R once produced plausible-looking variance components from two groups.

Two indicators per construct is deliberate: `TripleR` accepts a maximum of two
per latent construct, and a single manifest indicator cannot separate
relationship variance from error variance. `TripleR` defines "error" as the sum
of unstable perceiver, target and relationship variance — broader than Kenny's
definition, which reserves the term for unstable relationship variance. State
which is meant.

Collection 1 arms join on `run` + `rater` + `target` for paired model and
persona/null comparisons. Collection 2 arms are independent samples and should
not be pooled with Collection 1 as though paired. The `persona_mode` and
`collection` columns in `dataset/all_ratings_wide.csv` distinguish them.

Actor and partner effects join to `personas.csv` by `pid`, permitting regression
of SRM effects on measured personality — and the same regression in a null arm
serves as a falsification test.
