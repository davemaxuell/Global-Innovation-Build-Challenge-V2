# Code reorganization around the V2 pipeline (2026-10-01)

**Request:** the user confirmed V2 as the best checkpoint and asked that the code contain only V2's training pipeline, then approved the plan: archive rather than delete, archive V2's experiment-only code, and update `AGENTS.md`. **This is not a training phase.** No model, dataset, run or evaluation record was changed.

## What moved (80 items, 143 files; every file hash-verified after the move)
- **To `archive/v1_code/`:**
  - the V1-only packages (`scglm_choice`, `scglm_midtrain`, `scglm_phase`, `scglm_pipeline`, `scglm_post`, `scglm_rl`, `scglm_study`) and the unused `scglm/status.py`;
  - 18 V1-only scripts, including the V1 demo;
  - 11 V1-only configs;
  - 11 tests of archived code.
- **To `archive/v2_experiments/`:** `scglm_v2.pilot_gate` and `scglm_v2.branch` (the 1B pilot gate and the decay A/B branching), the branching support removed from `scglm_v2/train.py` and its three tests, and the two experiment run-control scripts. The results stay in `v2/phases/`, and the exact code stays in the run snapshots.
- **Into the main tree:** `v2/src/scglm_v2` → `src/scglm_v2`, `v2/tests` → `tests/`, `v2/supervise_main.sh` → `scripts/`, `v2/.pydeps` → `vendor/pydeps`.
- **Documents:** V1-era documents → `docs/history/`, including `V1_BEST_CHECKPOINT_TRAINING.md` and `V1_CURRENT_PIPELINE.md` with their relative links rewritten; the MiMo PDF → `docs/references/`.

Inventory with source, destination and SHA256 for every file: [moved_files.json](moved_files.json). Full pre-move snapshot: `archive/pre_reorg_2026-10-01/code_docs_snapshot.tar.gz` (SHA256 `64d47b20…6d242f7d`).

## What was kept, and why
- **The six `src/scglm` modules**, byte-identical, because V2 imports them. `prepare_data` and `prepare_extension` produced V2's tokenizer, benchmark-exclusion index and 14.0B reused tokens. Their hashes match `v2/runs/main/run.json`.
- **`scglm_v2.tokenizer_study`**, which the plan had listed for archiving. It was kept because the corpus build reads its decision file, making it the pipeline's tokenizer-selection step.
- **`scripts/run_final_evaluation.py` and `validation_history.py`**, unchanged, because their hashes appear in the evaluation records.

## Code changes
- `scglm_v2/common.py`: the project-root depth follows the new location.
- Docstrings: `PYTHONPATH` strings updated.
- `scglm_v2/train.py`: branching support removed. The file now differs from the main run's launch snapshot by exactly the two fixes in use since the 20:43 resume, confirmed by diff.
- `tests/test_v2_train.py`: fixture import and archived tests.
- `pyproject.toml`: pytest paths.
- **New:** `scripts/demo.py` (V2 text completion) and `scripts/verify_v2.py` (read-only integrity check).
- **New checkpoint copy:** `checkpoints/v2_best`, read-only and hash-verified ([inventory](../../../checkpoints/SELECTION_2026-10-01.json)).
- **Documents:** `README`, `CURRENT_PIPELINE`, `BEST_CHECKPOINT_TRAINING` (now the V2 record), `RUN_STATUS`, the checkpoints README, `AGENTS.md` ("Retained model scope"), header notes in `COMPETITION_BRIEF`, `EVALUATION_PROTOCOL` and `POST_TRAINING_ELIGIBILITY`, and the intros of `FAILED_APPROACHES` and `TRAINING_PHASES`.

## Verification (receipts in this directory)
| Check | Result |
| --- | --- |
| Imports of archived code across all active `src`/`scripts`/`tests` files | 0 of 34 files ([import_scan.txt](import_scan.txt)) |
| Every CLI entry point (`--help`), supervisor syntax | 16 of 16 OK |
| Test suite (`python -m pytest`, no manual paths) | **77 passed** ([tests.log](tests.log)) |
| `scripts/verify_v2.py --load` | **10/10 PASS** ([verify_v2.txt](verify_v2.txt)) |
| Official runner `--prepare-only` (harness commit, versions, task files, split counts, pinned `evaluate.py`) | exit 0, no scores ([protocol_preflight.log](protocol_preflight.log)) |
| 3-step GPU trainer smoke on the real V2 manifest from `src/` | warmup LR 1.2e-6 → 3.6e-6, loss 9.800 → 9.785, about 400k tokens/s ([trainer_smoke.txt](trainer_smoke.txt)) |
| Demo from `checkpoints/v2_best` | coherent text completion |
| Links in all active documents | 0 broken |
