# agent_srm

A command-line tool for running round-robin social interactions between LLM
personas and producing data structured for Social Relations Model (SRM) analysis
in `TripleR`.

Personas are drawn from **Twin-2K-500** (Toubia et al., 2025), a dataset of 2,058
US survey respondents with measured Big Five and ~45 other psychometric scales.
Each persona's full profile is supplied to the model as a system prompt; personas
converse in dyads and then rate one another on warmth and competence. The
resulting variance decomposition separates how much of a judgment is attributable
to the perceiver, to the target, and to the unique perceiver × target pairing.

Companion documents:

- **`REPRODUCE.md`** — full setup, endpoints, operational notes, known limitations
- **`agent_srm_design_v2.md`** — design specification and rationale

---

## Study design

| Parameter | Value |
|---|---|
| Block size | 5 personas |
| Blocks per arm | 50 (250 personas, 500 dyads, 1,000 directed ratings) |
| Interaction length (`k`) | 10 messages, strictly alternating |
| Items | warmth (warm, kind), competence (capable, effective) |
| Scale | 1–5, item order randomised per rating call |
| Conversation temperature | 1.0 |
| Rating temperature | 0.0 |

Personas are assigned by partitioning a single seeded permutation of the pool, so
no persona appears in more than one block — the independence assumption `TripleR`
requires. Within each block, initiator roles follow a regular tournament: every
persona initiates exactly two of its four dyads.

Arms sharing a `design_digest` used byte-identical block composition, dyad
membership and item orders, and are therefore paired and joinable on
`run` + `rater` + `target`.

**Null arms** (`personas.mode: none`) send no persona text at all — every system
prompt is the task framing alone, byte-identical to the framing embedded in the
persona arms. Blocks, dyads, seeds and `personas.csv` are unchanged, so a null arm
is a paired floor: any partner variance it shows is generation noise rather than
persona signal.

---

## Installation

Python 3.11+.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
python -m pytest -q                # 33 tests, no network access required
```

### Verify without collecting anything

```bash
python scripts/make_synthetic_personas.py --n 64 --chunks 2 --out tests/_fixture_personas
agent_srm run --config configs/dryrun_smoke.yaml --dry-run
```

The dry run exercises the entire pipeline — block assignment, thread
construction, JSON parsing, CSV assembly, quarantine logic — against a mock
client, with no API calls.

### With real persona data

```bash
python scripts/index_personas.py  --dir data/personas --revision <hf-revision-sha>
python scripts/check_extractor.py --dir data/personas
```

`index_personas.py` records a SHA-256 per chunk; the runner verifies these at load
and fails on mismatch. `check_extractor.py` confirms all 14 demographic fields and
46 score fields parse for all 2,058 personas, and exits non-zero otherwise.

---

## Commands

| Command | Purpose |
|---|---|
| `agent_srm validate --config C` | Run every configuration gate; no API calls |
| `agent_srm run --config C [--dry-run] [--arm SLUG] [--workers N] [--rpm N]` | Collect |
| `agent_srm verify-arms --config C` | Assert arms share a design digest |

Runs are checkpointed per block and both CSVs are rebuilt after each block, so an
interrupted run resumes from the same command with nothing lost or duplicated.

`--workers N` processes dyads within a block in parallel; turns within a dyad and
blocks themselves remain sequential. The test suite asserts that parallel and
sequential runs produce identical rows in identical order.

---

## Outputs

Per arm, under `runs/{study_id}__{arm_slug}/`:

| File | Contents |
|---|---|
| `ratings_wide.csv` | one row per directed rating — **the `TripleR` input** |
| `ratings.csv` | long format, one row per item; retains `item_position` |
| `transcripts.jsonl` | one record per dyad; every prompt stored as actually sent |
| `personas.csv` | covariate join table (demographics, scores, percentiles) |
| `manifest.json` | resolved config, seeds, design digest, checksums, quarantine log |
| `quarantine/` | incomplete blocks with per-persona failure attribution |
| `run.log` | one line per API call |

A block reaches disk only if complete: all *n(n−1)* directed ratings with every
item validly parsed. Partial blocks are quarantined whole, since an incomplete
block is unusable in `TripleR` and would silently distort variance estimates.

---

## Verification

The tool ships with independent audit scripts. All were run on the collected
dataset before analysis.

| Script | Checks |
|---|---|
| `verify_run.py` | round robin complete, no self-ratings, persona independence, score validity, message alternation, no construct label in any prompt, item-order randomisation, single model snapshot |
| `audit_ratings.py` | traces every rating from the model's literal reply through the runner's parse to the CSV row, using a parser independent of the runner's |
| `check_empty.py` | empty and truncated messages, by run and stop reason |
| `package_dataset.py` | verifies all runs, then combines them into a single labelled dataset |
| `check_long_wide.py` | long and wide CSVs describe identical data |

`audit_ratings.py` is the load-bearing one: it re-parses `raw_response` — the
model's literal text, stored before processing — and compares it against both the
runner's interpretation and the CSV. It was validated against three deliberately
corrupted datasets (an altered CSV value, a mismatched parse, and items remapped
by position rather than by name) and detects all three.

Both `verify_run.py` and `audit_ratings.py` exit non-zero on failure.

---

## Analysis

Analysis is deliberately out of scope for this tool; keeping collection and
analysis separate is what makes the archive auditable.

```r
library(TripleR)
d <- read.csv("runs/<study>__<arm>/ratings_wide.csv")
RR(warm/kind + capable/effective ~ rater * target | run, data = d)
```

Confirm the header reports `n = 50` and `average group size = 5`.

Two indicators per construct is deliberate: `TripleR` accepts a maximum of two
per latent construct, and a single manifest indicator cannot separate
relationship variance from error variance. Note that `TripleR` defines "error" as
the sum of unstable perceiver, target and relationship variance — broader than
Kenny's definition, which reserves the term for unstable relationship variance.

Actor and partner effects join to `personas.csv` by `pid`, permitting regression
of SRM effects on measured personality. Running that regression in a null arm,
where the same personas are present in `personas.csv` but no persona text reached
the model, provides a falsification test.

---

## Repository layout

```
agent_srm/
├── configs/           study configurations, one per arm
├── prompts/           versioned prompt templates (hashed into every manifest)
├── data/personas/     Twin-2K-500 parquet chunks + MANIFEST.json (checksums)
├── scripts/           acquisition, verification and packaging utilities
├── src/agent_srm/     the runner
├── tests/             33 acceptance tests, no network access required
└── runs/              collected data, one directory per arm
```

---

## Limitations

Stated fully in `REPRODUCE.md` §7. In brief:

- **Reasoning depth is confounded with model family.** Each arm ran at its
  provider's default reasoning behaviour, which differs by provider; one model
  cannot disable reasoning at all. A between-arm difference cannot be attributed
  to the model alone.
- **Generation is not reproducible.** Hosted endpoints accept a sampling seed and
  generally ignore it. Design-level randomisation *is* exact and verifiable via
  `design_digest`; model sampling is not.
- **Rating temperature 0.0 is not deterministic** for reasoning models, since
  reasoning traces vary between calls.
- **Serving precision is not uniform.** Where a gateway was used, the provider was
  pinned with fallbacks disabled and the quantisation recorded; not all providers
  disclose it.
- **Some blocks were re-collected.** Truncated generations produced empty messages
  where a model's reasoning exhausted the output budget. Affected blocks were
  identified, removed and re-collected at a higher budget. The final dataset
  contains no empty or truncated messages, but a subset of blocks originates from
  a second collection pass under a larger `max_tokens`.
- **Independent code review is outstanding.** The runner has been verified by the
  audit scripts above and by 33 automated tests, but has not been read by a
  reviewer other than its authors.
