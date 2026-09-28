# Persona data

This directory holds the persona corpus used in the study.

## Source and license

**Twin-2K-500** (Toubia, Gui, Peng, Merlau, Li, & Chen, 2025)
https://huggingface.co/datasets/LLM-Digital-Twin/Twin-2K-500

Licensed CC BY 4.0 and redistributed here under the same license. The seven
parquet chunks in this archive are the exact files used for data collection,
not a re-download, so replication does not depend on the upstream dataset
remaining unchanged.

Cite the original:

    @dataset{twin2k500,
      author    = {Toubia, Olivier and Gui, George Z. and Peng, Tianyi and
                   Merlau, Daniel J. and Li, Ang and Chen, Haozhe},
      title     = {Twin-2K-500: A Dataset for Building Digital Twins of
                   2,000 People},
      year      = {2025},
      publisher = {Hugging Face},
      howpublished = {\url{https://arxiv.org/abs/2505.17479}}
    }

## Contents

    persona_chunk_001.parquet .. persona_chunk_007.parquet
    MANIFEST.json

2,058 respondents across seven chunks. File names are ours; the upstream
dataset ships the same content under its own naming. Columns used:

    pid                unique respondent identifier
    persona_summary    the condensed profile supplied as the system prompt

`persona_text` and `persona_json` are present but unused by the runner.

## MANIFEST.json

Contains SHA-256 for each chunk. The runner verifies every file at load and
refuses to start on a mismatch, so the data cannot change underneath a study
unnoticed. Verify your copy:

    python scripts/index_personas.py --dir data/personas

If the checksums it reports differ from those in MANIFEST.json, the files are
not identical to ours. Note that running that script **overwrites**
MANIFEST.json with checksums of whatever is present, so copy the original
aside first if you want to compare.

`revision` is recorded as null: the upstream HuggingFace revision was not
captured at download time. Since the files themselves are included here, that
does not affect reproducibility.

## Validation

    python scripts/check_extractor.py --dir data/personas

Must report 14/14 demographic fields at 2058/2058 and 46 score fields.
