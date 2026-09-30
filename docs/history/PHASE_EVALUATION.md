# Automatic phase evaluation

> Current implementation after the September 29 cleanup: [CURRENT_PIPELINE.md](V1_CURRENT_PIPELINE.md). This reference retains historical rules, plans or execution details; completed model construction is recorded in [BEST_CHECKPOINT_TRAINING.md](V1_BEST_CHECKPOINT_TRAINING.md).

This workflow observes the existing, bounded 12B → 13B scratch-pretraining run, evaluates retained checkpoints after training releases GPU 0, and produces a reproducible performance report. It adds **zero optimizer updates**. The training supervisor remains responsible for its registered pilot, token budget, checkpointing, final development gate, and conditional confirmation gate.

The evaluator is registered during training on September 26, 2026. The 12B official scores and twelve completion examples were already observed. This is a supplemental evaluation registration, not a claim that the whole experiment was blind. The [13B training contract](../../artifacts/pretraining_13b/contract.json) and [evaluation configuration](../../archive/v1_code/configs/phase_evaluation.json) define the exact inputs.

## Phase coverage

| Phase | Automatic tests and measurements | Outcome / authority |
| --- | --- | --- |
| Corpus / continuation preflight | Verify registered data-audit evidence, checksums, available unread targets, held-out panel identities and scratch lineage | Report existing preflight; reject changed evidence |
| GPU recovery | Verify recorded exact weight, optimizer, source-stream, resume-loss and export checks | Engineering pass; does not demonstrate model quality |
| 1,000-update pilot | Before/after BF16 monitor NLL and source NLL; existing limits +0.10 mixture / +0.15 per source | Stability pass or failure; no automatic claim of improvement |
| Ongoing continuation | Every 30 seconds collect steps, loss-bearing targets, source counts, training loss, gradient norm, LR, throughput and monitor curves; reject nonfinite values | Observe existing run; do not change its schedule |
| 1B, 12B, 12.5B milestone, 13B final | Same 768-document development panel, FP32 token NLL/perplexity by source and fixed mixture; 12 fixed completion prompts; EOS, repetition, generation timing; parameter count, finite weights/logits, deterministic replay and unchanged artifacts | Report all planned checkpoints, including regressions; no intermediate selection |
| Final development decision | Reproduce paired document bootstrap and existing gain/source-regression gates from the supervisor's saved document losses | At least 0.005 mixture NLL improvement, 95% bootstrap upper bound below zero, no source point-estimate regression above 0.02 |
| Final confirmation | Independently validate supervisor's second-panel decision only when development passed | Same gate; retain 12B if improvement is unconfirmed; no retrospective milestone scoring on confirmation |
| Official benchmarks | Reuse checksummed complete 12B report; run full HellaSwag, ARC-Easy, PIQA, WinoGrande and WikiText-103 on 13B using the existing frozen protocol | Report 13B regardless of its likelihood-gate outcome; official scores never revise the frozen recommendation |
| Evidence handoff | Markdown/JSON/CSV reports, plots, all generations, per-document losses, attempt logs, immutable stage receipts and final file manifest | Local reviewable evidence; model publication/submission is not performed |
| SFT / DPO / RL | Explicitly recorded as skipped under the selected competition raw-pretraining route | No post-training job is launched |

The pilot checkpoint has already been pruned by the trainer's existing retention policy. Its recorded before/after likelihood evidence is preserved; new generation or official scores for that checkpoint cannot be reconstructed. The trainer retains the 12.5B milestone separately. Its actual rounded token count, rather than exactly 12,500,000,000, is reported.

## Interpretation

- Lower held-out NLL is evidence of improved next-token prediction on that panel. It does not establish better factual recall, instruction following, general reasoning, or competition placement.
- Pilot/ongoing BF16 monitor scores use 32 documents per source. Final FP32 development scores use 256 per source. These are different panels and numerics; compare changes only within a protocol.
- Checkpoint diagnostics use the same fixed development panel only after the final raw-text decision is frozen. Confirmation remains exclusive to the registered final-versus-parent gate. The 1B model used a different training mixture, which matters when interpreting its comparison with later checkpoints.
- All twelve previously inspected completion prompts and their full outputs are retained, including failures. Loop rate detects four repetitions of a four-word sequence; duplicate four-gram fraction is descriptive. EOS rate is measured within 64 generated tokens. None is a factual-accuracy benchmark.
- The raw and character-normalized multiple-choice accuracy metrics are kept separate. WikiText-103 uses the project's fixed raw-validation, article-boundary, 1,024-context/512-stride token-perplexity convention. These are project settings, not an assertion of unpublished organizer settings.
- Reported deltas between intermediate checkpoints and official endpoint scores are descriptive point estimates. Only the supervisor's registered paired document bootstrap supports the final raw-likelihood decision. No synthetic overall grade or win probability is created.
- The initial 12B selection and submission package stay archived. A separate reporting-selection record honestly identifies the new development/conditional-confirmation decision and the prior observation of 12B official results.

## Running and recovery

From the project root, using the existing environment:

```bash
PYTHONPATH=src /home/bufsgpu/yes/envs/sw/bin/python scripts/run_phase_evaluation.py --status

# Update a CPU-only observation/report; do not launch inference.
PYTHONPATH=src /home/bufsgpu/yes/envs/sw/bin/python scripts/run_phase_evaluation.py --once

# Persistent execution (launch only one observer).
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 \
  /home/bufsgpu/yes/envs/sw/bin/python scripts/run_phase_evaluation.py --watch
```

The launch is kept in tmux `gibc-phase-eval`; [state.json](../../artifacts/phase_evaluation_v1/state.json) supplies its current PID/stage. The [live report](../../artifacts/phase_evaluation_v1/REPORT.md), [JSON](../../artifacts/phase_evaluation_v1/report.json), and [metric table](../../artifacts/phase_evaluation_v1/performance.csv) are updated during execution. Completed and queued stages are explicit.

Code, configuration, prior results, panels and dependencies are frozen with SHA256 hashes and copied into a snapshot. Changed inputs fail closed. The observer checks the current trainer's frozen files and never rewrites them. It uses an observer lock, the existing GPU lease, the assigned GPU identity, and a GPU occupancy check. No inference is launched until training **and its held-out comparison** are complete and the GPU is available.

Each checkpoint evaluation is bounded to 15 minutes and the final benchmark command to 60 minutes, with at most two recorded attempts per stage. Failed attempts/logs are retained. Restarting the same command verifies and reuses successful stages, or recovers a valid completed output after an observer interruption. It does not erase attempts, silently increase the retry budget, or start a different training run. A 15-minute stale training heartbeat, failed/paused supervisor, or 12-hour observer deadline causes a visible failure; the original training process is left untouched. Stop the observer with SIGTERM to its own PID; do not target the training PID.

The final `handoff.json` links the fixed recommendation and all stage receipts. `EVIDENCE_MANIFEST.json` hashes the completed local evaluation archive; original model weights stay in their source run directories. Resuming after an exhausted budget or changing evaluation inputs requires a new reviewed registration/output directory.
