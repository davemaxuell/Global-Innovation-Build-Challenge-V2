# Training integrity audit — 2026-09-28

**Verdict: the executed training phases passed the recorded execution and artifact-integrity checks, with one minor logging ambiguity. Some proposed phases were intentionally not run.** Correct execution does not establish that the recipes are optimal or that post-training is exhaustive.

Audit timestamp: 2026-09-28 07:22 KST (2026-09-27 22:22 UTC). This was a read-only audit of saved records and tensors; it did not perform training updates or generate new model predictions.

## Phase accounting

| Phase | Outcome | Verified extent |
| --- | --- | --- |
| Initial scratch pretraining | Completed | 15,258 updates; 999,948,288 target tokens |
| Continuation to approximately 12B | Completed | 167,848 further updates; 11,000,086,528 additional target tokens |
| Continuation to approximately 13B | Completed | 15,259 further updates; 1,000,013,824 additional target tokens |
| Human-only supervised instruction tuning | Completed | 131 updates; 1,077,789 supervised target tokens; 2,801,905 processed tokens; all 14,828 available examples |
| Mixed supervised instruction tuning | Completed | 131 updates; 1,077,749 supervised target tokens; 3,246,128 processed tokens; 27,445 of 29,828 available examples |
| Development evaluation | Completed | Saved predictions and raw-text loss aggregates independently rescored; selection reproduced |
| Local benchmarks using the recorded competition protocol | Completed | Existing base results reused with verified identity; both SFT exports evaluated |
| DPO | Not run | Neither SFT candidate passed the predeclared development gates |
| RL | Not run | Deferred in the registered recipe |
| Post-training confirmation evaluation | Not opened | No development-qualified challenger advanced |

The pretraining lineage totals **13,000,048,640 target tokens**. The two SFT runs independently start from that same base. Their tokens are not part of the 13B pretraining total. The mixed SFT run used the planned approximately matched supervised-token budget; its incomplete traversal of the available examples is not evidence of an interrupted job.

## Checks that passed

- All five final exports exactly match their respective final checkpoint tensors. The 46,346,752-parameter model retains its tied embedding/output weights; checked model and optimizer tensors are finite.
- Pretraining ancestry, target-token accounting and inherited optimizer-step counts reconcile across the three runs.
- Both SFT runs match the independently reconstructed frozen batch plans for every update, including supervised and processed token counts. Their learning rates match the declared schedule, and logged losses and gradient norms are finite. All 110 exported tensors changed from the parent in each SFT run.
- The source/configuration/runtime contract is unchanged. Checks verified 27 completed-stage artifacts, 40 previously registered files, 109 prior phase-evidence files and the original base release inventory.
- Rescoring all 6,000 procedural predictions and 1,800 grounded predictions, and recomputing raw-source loss aggregates, reproduces the reported metrics. Reapplying the selection rule produces the same base-model choice.
- Production child commands exited successfully. Benchmark protocol identities, artifact hashes, task counts and selection timing match the recorded workflow.
- The existing CPU-test receipt records **166 passing tests** before launch. Those tests were not rerun for this read-only audit.

## Minor logging finding

SFT update events use `processed_tokens` for the current batch count, while status, checkpoint and validation records use the same key for the cumulative count. Each update count matches its planned batch, and their sums exactly equal the final cumulative counters. This naming ambiguity does not change gradients, token budgets, selection or compute-ledger totals.

Preserve the historical logs and frozen trainer. A future registered trainer should use distinct `batch_processed_tokens` and `cumulative_processed_tokens` fields.

## Interpretation and remaining limitations

The mixed SFT model improved strict procedural instruction accuracy from 0% to 49.10%. Both SFT candidates increased raw-text negative log-likelihood by approximately 0.06 across the checked source panels, exceeding the project's predeclared 0.02 tolerance. The human-only candidate also failed the instruction-improvement gate. Retaining the base and skipping DPO therefore followed the registered selection rule; these outcomes do not indicate failed optimizer execution or prove that instruction tuning is ineffective.

These were two short SFT pilot runs, with one budget and seed per arm. They do not establish optimal hyperparameters or comprehensive post-training. Procedural and fictional-grounding tasks are narrow, independent human evaluation remains unscored, and prior base/benchmark outcomes were already observed and disclosed. Dataset overlap checks do not prove semantic independence.

The existing base release remains unchanged. This audit did not start additional training or publish a replacement model.

## Evidence

- [Machine-readable integrity audit](2026-09-28_training_integrity.json)
- [Post-training results](../../artifacts/posttraining_13b_v1/REPORT.md)
- [Pipeline state](../../artifacts/posttraining_13b_v1/state.json)
- [Pre-launch CPU-test receipt](../../artifacts/posttraining_13b_validation/cpu_tests.json)
