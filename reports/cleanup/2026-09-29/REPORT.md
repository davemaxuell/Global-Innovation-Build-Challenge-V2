# Pipeline cleanup — September 29, 2026

Status: **completed**. The user's request was to retain the code that built the current model, remove other training pipelines and document unsuccessful approaches.

## Retained model

Selected full-SFT model: [best_sft](../../../checkpoints/best_sft/), SHA256 `defd0d91948ba8501a33b0c4996bf3bf37ebdf6e90338a05bab3a1773a2d7f72`. The original 13B base, source checkpoints, completed runs, prepared data and measured results remain preserved.

The active path is fixed-mixture scratch pretraining through 13B targets, followed by the selected full-parameter SFT with raw replay. [Current implementation](../../../CURRENT_PIPELINE.md) · [Successful training record](../../../BEST_CHECKPOINT_TRAINING.md).

## Removal and simplification

- Deleted **71 superseded active files**: experiment implementations, alternative trainers/launchers, their configurations, obsolete tests and runbooks.
- Removed the adaptive pretraining controller, legacy SFT orchestration, DPO, positive-only preference training, online RL, output-only adaptation and the knowledge-heavy SFT experiment from the active training tree.
- Replaced the multi-stage post-training entry point with read-only verification by default and a single explicit `sft` reproduction action.
- Preserved the selected SFT planner and weighted optimizer update. Reproduction stops at update 105 while retaining the original 209-update schedule denominator.
- Retained data-preparation, loss, model-loading, statistical evaluation and reporting dependencies needed for this model.
- Updated the README, status, phase journal and agent instructions; redirected retired links in mutable documentation. The unexecuted extra-pretraining proposal is marked historical and withdrawn from the active roadmap.

[Exact removed paths and prior hashes](removed_files.json) · [Retained active source/config/test inventory](retained_files.json) · [Pre-cleanup inventory](inventory_before.json).

## Lessons preserved

[FAILED_APPROACHES.md](../../../FAILED_APPROACHES.md) records eight unsuccessful or unpromoted recipes with their actual outcomes, evidence and instructions against unchanged repetition. It distinguishes completed trials, deliberately skipped stages and never-launched proposals. The successful training record remains separate.

Original source snapshots inside frozen run/artifact records remain historical evidence. They are outside the active implementation and were not rewritten or removed. Existing model exports, including the user's previously preserved checkpoints, remain intact.

## Verification

- **150 tests passed; one optional H100 recovery test skipped.** Tests used disposable CPU fixtures.
- The retained SFT planner and optimizer update are AST-equivalent to the original registered functions.
- The exact selected batch plan hash, 209-update schedule, 105-update endpoint and all four response/replay counters matched the recorded run.
- Working checkpoint inventories matched; the selected model and tokenizer loaded offline, inference produced finite logits, and the 46,346,752-parameter tied-weight audit passed.
- Retained Python sources compile and their project imports resolve. No deleted training module remains imported by active code.
- No production model was retrained, overwritten, submitted or published during cleanup. A full H100 reproduction was not rerun; byte-identical GPU retraining is not claimed from CPU checks alone.

[Test log](tests.log) · [Read-only verification receipt](verification.json).
