# Final retained-pipeline audit

**Checked:** September 29, 2026. **Result: passed.**

The active model-training implementation contains the completed lineage of the selected checkpoint: fixed-mixture scratch pretraining through **13,000,048,640 raw targets**, followed by balanced full-parameter SFT with raw replay, ending at selected update **105**. Data preparation, model loading, evaluation, checkpoint recovery and hardware checks needed for that lineage remain.

The selected instruction model is [checkpoints/best_sft](../../../checkpoints/best_sft/). Its completed recipe and measured results are in [BEST_CHECKPOINT_TRAINING.md](../../../BEST_CHECKPOINT_TRAINING.md). This audit verifies the retained implementation and saved artifacts; it does not make a new claim about competition ranking.

## Active training surface

| Retained component | Verified scope |
| --- | --- |
| `src/scglm/train.py` | Fixed-mixture scratch pretraining and continuation; no adaptive controller or probe optimizer-update mode |
| `configs/train_baseline.json` | Matches the completed scratch run: 999,948,288 targets |
| `configs/train_extension.json` | Matches the completed first continuation: 11,000,086,528 additional targets |
| `configs/train_continuation_13b.json` | Matches the completed final continuation: 1,000,013,824 additional targets |
| `src/scglm_pipeline/sota_train.py` | Full-parameter SFT with raw replay; alternative training stages rejected |
| `scripts/run_pipeline.py` and `configs/current_model.json` | Read-only `verify` by default; explicit `sft` reproduces the selected recipe in a separate output directory |
| `scripts/profile_gpu.py` | Disposable synthetic hardware measurement; does not export a trained language model |

The three pretraining configurations match their completed run records exactly. The recursive scratch-lineage check passed, including parent checkpoint hashes and cumulative token accounting. The original SFT planner and weighted optimizer-update function remain AST-equivalent to the frozen registered implementation.

## Final corrections

- Removed the unused controller probe-update branch and scheduler override from the raw trainer. Ordinary update arithmetic, source sampling and learning-rate schedules are preserved. The source-epoch test now checks an ordinary retained training update.
- Removed an unreachable duplicate controller check and clarified that retained ledger fields are historical checkpoint-schema compatibility.
- Restricted the demo's stage handling to pretraining and full-SFT checkpoints.
- Replaced a stale phase-report label that implied post-training was still skipped; this report covers raw pretraining and points to the completed SFT record.
- Corrected the current configuration's evaluation disclosure: its confirmation results have already been observed and are not fresh blind evidence.
- Removed stale controller/DPO wording from current status and task comments.

These changes did not modify checkpoint weights, frozen run records, processed data or historical results. The [source receipt](final_check_sources.json) lists all seven changed source/config/test files and the final hashes of all 78 active source/config/test files.

## Verification results

| Check | Result |
| --- | --- |
| Previously removed paths | All **71** remain absent; active `experiments/` directory absent |
| Active source checks | Python compiles, JSON parses, project imports resolve, literal source/config references resolve |
| Training implementations | Raw pretraining and full-SFT only; the separate synthetic GPU profiler is a hardware diagnostic |
| Retired implementations | No active DPO, positive-only preference training, online RL, head-only adaptation, knowledge-heavy SFT experiment or adaptive controller |
| Test suite | **150 passed; 1 optional GPU recovery test skipped** |
| SFT schedule | Original **209-update / 1,715,410-response-target** plan retained; selected prefix ends at **105** |
| Selected token counts | **862,660 response targets**, **419,498 replay targets**, **4,670,374 response input tokens**, **420,106 replay input tokens** |
| Working checkpoints | Base and selected-model file inventories match their recorded hashes |
| Selected model loading | Offline CPU reload and finite forward pass passed; **46,346,752** unique parameters with tied input/output weights |
| Project training processes | None running at the time of the final check |

Selected weight SHA256: `defd0d91948ba8501a33b0c4996bf3bf37ebdf6e90338a05bab3a1773a2d7f72`.

Base weight SHA256: `ffa75b07ec377167ab2e4d4263f2fa8374549f8592f9ce2e8166875cab1f38e7`.

[Test log](final_check_tests.log) · [Checkpoint and schedule verification](final_check_checkpoint.json) · [Source and pretraining-lineage receipt](final_check_sources.json) · [Combined audit receipt](final_check.json).

## Historical evidence and limits

Frozen source snapshots, previous experiments, checkpoint exports and measured results remain under `artifacts/` and `runs/` for provenance and the separately documented [unsuccessful approaches](../../../FAILED_APPROACHES.md). They are not active launch instructions. No active project import resolves into a historical source snapshot.

Reserved corpus partitions and pinned evaluation metadata still contain historical `controller_*` names, and the raw trainer retains compatible accounting fields. These describe recorded data and checkpoint schemas; they do not implement adaptive training. Generation diagnostics also retain their existing result fields without a policy-optimization consumer.

No full H100 training reproduction or new benchmark evaluation was run for this audit. CPU tests use disposable models; verification loads the saved model without changing it. The check establishes retained code scope, local correctness and artifact identity, not byte-identical GPU retraining.
