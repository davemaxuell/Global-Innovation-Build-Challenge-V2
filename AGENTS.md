# Training documentation

The user requested on 2026-09-29 that every training phase be documented.

- Before launching a phase, add it to `TRAINING_PHASES.md` and create a linked phase report. Record the objective, parent checkpoint and hash, data identity, training recipe, compute limits, evaluation/selection rules, output directory, and launch/resume command.
- During execution, keep machine-readable status and a dated event log. Distinguish preparation, running, paused, failed, and completed work.
- After execution, document actual updates and token counts, checkpoint paths and hashes, evaluation results, the decision, and the next step. Record skipped or failed phases too.
- Preserve existing checkpoint exports and experiment records; use separate output directories for new experiments.
- The current objective is competition benchmark performance. Keep development-based checkpoint selection separate from final benchmark reporting and disclose previously observed evaluations.

## Retained model scope

On 2026-10-01 the user confirmed V2 as the best checkpoint and asked that the code contain only the V2 training pipeline. This supersedes the 2026-09-29 scope, which retained the V1 full-SFT lineage. Later on 2026-10-01 the user confirmed that fine-tuning our own scratch-trained model is allowed and adopted a task-tuned V2 as the submission.

- The active path is V2: V1-lineage data preparation (tokenizer, exclusion index, extension text), the V2 corpus build, WSD scratch pretraining to 45B targets, and development-only selection. Then comes one fine-tuning stage on the benchmarks' train splits (`src/scglm_v2/task_tune.py`) and the pinned official evaluation. Read `BEST_CHECKPOINT_TRAINING.md` and `CURRENT_PIPELINE.md`.
- The submitted checkpoint is `checkpoints/v2_task_tuned` (= `v2/runs/task_tune_lr2e-5_r0.5/epoch_3/export`, weight SHA256 `aaf13826…`). The user adopted it after its official scores were observed, overriding the registered Wikipedia guard. Its results must always be reported with that disclosure and next to V2 base.
- Preserve `checkpoints/v2_best` (V2 base, `9aac2111…`), the V1 baselines (`checkpoints/competition_base`, `checkpoints/best_sft`) and the WiSE-FT exports. Write new runs only to separate outputs.
- Keep `src/scglm/{model,data,train,evaluate,prepare_data,prepare_extension}.py`, `scripts/run_final_evaluation.py` and `scripts/validation_history.py` byte-identical. Their hashes are recorded in V2's run and evaluation records, and the protocol pins `evaluate.py`. Check with `PYTHONPATH=src python scripts/verify_v2.py`.
- Read `FAILED_APPROACHES.md` before proposing more training. Retired code lives in `archive/` ([inventory](reports/cleanup/2026-10-01/moved_files.json)). Do not restore it into `src/` or `scripts/`, and do not repeat its unchanged recipes as automatic next phases.
- Frozen historical code, configuration and records under `artifacts/`, `runs/`, `v2/configs/`, `v2/runs/`, `v2/phases/` and `archive/` are evidence. Active implementations live under `src/` and `scripts/`; do not treat old snapshots or pre-2026-10-01 launch commands (`PYTHONPATH=src:v2/src`) as current instructions.
- Keep the successful model-training record focused on its completed lineage. Record unsuccessful approaches and their evidence separately in `FAILED_APPROACHES.md`.
