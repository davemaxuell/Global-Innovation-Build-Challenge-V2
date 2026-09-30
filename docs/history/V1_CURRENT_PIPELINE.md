# Retained model construction and reproduction

**Scope after cleanup, September 29, 2026:** retain the implementation of the selected model's scratch-pretraining and full-SFT lineage, together with its data preparation, evaluation and checkpoint verification.

The historical result is [checkpoints/best_sft](../../checkpoints/best_sft/). [BEST_CHECKPOINT_TRAINING.md](V1_BEST_CHECKPOINT_TRAINING.md) records how it was trained. The working checkpoint and its source artifacts are unchanged.

## Code map

| Component | Active files | Purpose |
| --- | --- | --- |
| Model, streams and raw pretraining | `src/scglm/model.py`, `data.py`, `train.py` | The original 46M architecture, packed-token streams, fixed-mixture optimizer updates, inherited state and recovery |
| Raw corpus and tokenizer preparation | `src/scglm/prepare_data.py`, `prepare_extension.py`; `configs/data.json`, `data_extension.json` | Initial FineWeb/Wikipedia preparation and Edu/DCLM/Wikipedia extension |
| Pretraining recipes | `configs/train_baseline.json`, `train_extension.json`, `train_continuation_13b.json` | The three completed stages that produced the 13B parent |
| Pretraining checks and phase evaluation | `scripts/run_extension.py`, `run_pretraining_next.py`, `run_phase_evaluation.py`; `src/scglm_phase/` | Original continuation checks, recovery and measured phase reports |
| Human-demonstration preparation | `src/scglm_post/prepare.py`, `overlap.py`, `audit.py`, `procedural.py`, `system_data.py`; `configs/posttraining/data.json` | Source preparation and protected task data used in the selected lineage |
| Protected instruction-data assembly | `src/scglm_pipeline/prepare.py`, `sota_data.py`, `task_registry.py`; `configs/instruction_data.json` | Human, knowledge, grounded and procedural data plus raw replay |
| Selected full-SFT recipe | `configs/current_model.json`; `src/scglm_pipeline/sota_train.py` | Balanced target-token batching, full-parameter response loss and raw replay |
| Shared masking and schedule | `src/scglm_post/sft.py`, `common.py`, `verifiers.py` | Exactly one causal shift, response/EOS masking, chat format and token-based LR |
| Instruction evaluation and selection evidence | `src/scglm_pipeline/sota_evaluate.py`, `evaluate.py`, `generation.py`, `selection.py`, `selection_v2.py`, `posttraining_lineage.py` | Literal-answer likelihood, generation diagnostics, retention and independent confirmation machinery |
| Benchmark reporting and model use | `scripts/run_final_evaluation.py`, `validation_history.py`, `release_tools.py`, `verify_sota_package.py`, `demo.py` | Pinned benchmark reporting, artifact verification and inference |

The small data/evaluation helpers shared by these files remain. Alternative trainers, search orchestration and experiment launchers are listed in the [removal inventory](../../reports/cleanup/2026-09-29/removed_files.json). Results and lessons are in [FAILED_APPROACHES.md](../../FAILED_APPROACHES.md).

## Read-only verification

From the project root:

```bash
PYTHONPATH=src:. PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 \
  /home/bufsgpu/yes/envs/sw/bin/python scripts/run_pipeline.py verify --load
```

The default command is also `verify`. Verification checks:

- preserved base and selected-model identities, including the working-copy file inventories;
- the frozen instruction manifest and its files;
- the selected recipe against the original run contract;
- the deterministic full batch plan: **209 updates / 1,715,410 response targets**;
- the chosen prefix: **105 updates / 862,660 response targets / 419,498 replay targets**;
- optional offline CPU reload, finite forward output and tied-parameter audit with `--load`.

This command neither downloads replacement data nor performs an optimizer update.

## Reproduce the selected SFT phase

The only post-training action in the retained entry point is `sft`. When a reproduction is requested, first register its objective, exact parent, data, recipe, compute ceiling, output and command in `TRAINING_PHASES.md` and a linked phase report, as required by `AGENTS.md`.

```bash
PYTHONPATH=src:. PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 \
  /home/bufsgpu/yes/envs/sw/bin/python scripts/run_pipeline.py sft \
  --output artifacts/current_model_reproductions/sft_reproduction_001
```

The entry point selects only the assigned GPU UUID and acquires the existing project GPU lock. Output must be a child of `artifacts/current_model_reproductions/`. Original checkpoints, datasets and run directories cannot be chosen as reproduction outputs.

The trainer uses the unchanged full 209-update schedule and stops after update **105**. The selected update is a prefix of that original schedule. The deterministic planner and weighted response/replay update match their frozen historical implementations.

The same command resumes an interrupted reproduction only when its code/configuration/data contract and checkpoint checksum match. It restores optimizer state, RNGs, replay cursors and counters. A completed reproduction returns without extra training. The result is saved under `endpoints/step_0000105/export/`; `reproduction.json` records whether the weight-file bytes match the preserved model. New outputs carry a distinct reproduction provenance rather than altering the original experiment's contract.

The reproduction does not repeat hyperparameter search or reopen consumed confirmation panels. The retained evaluation modules can report a separately registered check; no result automatically replaces either preserved working checkpoint.

## Reconstruct earlier stages

The exact pretraining settings, source revisions and counts are in [the training record](V1_BEST_CHECKPOINT_TRAINING.md). The three original run records contain their parent-checkpoint hashes, runtime versions and source hashes:

1. [Scratch stage](../../runs/baseline_1b/run.json).
2. [Continuation to 12B](../../runs/continuation_12b/run.json).
3. [Continuation to 13B](../../runs/continuation_13b/run.json).

Their active trainer is `python -m scglm.train --config <new-run-config>`. A fresh reconstruction must copy the recorded settings into a new run configuration, use a separate output, and register each newly produced parent path/hash for the following continuation. Completed historical directories are not fresh-run destinations. Same-run resumes require the source/runtime contract with which that run was created; original frozen snapshots are retained for that purpose.

The selected instruction training set is already prepared and checksummed. The retained preparation helpers explain and implement its construction. Regeneration belongs in a new data directory; existing manifests and processed data remain immutable. The SFT entry point deliberately verifies the recorded dataset instead of silently preparing a different one.

## Verification performed during cleanup

The retained suite covers pretraining loss/accumulation, inherited optimizer and stream state, masking, source-group isolation, weighted replay gradients, exact interrupted/resumed SFT, preserved schedule-prefix behavior, read-only defaults, output isolation and benchmark reporting. Source-level comparison confirms that the selected SFT planner and optimizer-update function match the registered snapshot.

[Test log](../../reports/cleanup/2026-09-29/tests.log) · [Selected-model verification](../../reports/cleanup/2026-09-29/verification.json) · [Cleanup report](../../reports/cleanup/2026-09-29/REPORT.md).

The subsequent [final audit](../../reports/cleanup/2026-09-29/FINAL_CHECK.md) also removed an unused controller-probe optimizer branch, checked all active imports and all three completed pretraining recipes, and repeated the **150-pass / 1-skip** test suite and checkpoint verification. Its linked receipts record the final source hashes.

The cleanup uses disposable CPU test models and loads the preserved model for verification. It does not claim to have repeated the full H100 training run.
