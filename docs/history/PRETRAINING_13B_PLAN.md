# Registered next training step: 12B to 13B raw-text pretraining

> Current implementation after the September 29 cleanup: [CURRENT_PIPELINE.md](V1_CURRENT_PIPELINE.md). This reference retains historical rules, plans or execution details; completed model construction is recorded in [BEST_CHECKPOINT_TRAINING.md](V1_BEST_CHECKPOINT_TRAINING.md).

**Date:** September 26, 2026.  
**Authorization:** the user requested the next training step and explicitly selected “Continue pretraining for the competition.”  
**Scope:** continue the same randomly initialized model lineage using ordinary next-token prediction on the existing raw-text corpus.  
**Live state:** [supervisor](../../artifacts/pretraining_13b/status.json), [training](../../runs/continuation_13b/status.json).

## Objective and evidence

Test whether a bounded continuation improves held-out language-modeling loss. The 12B model's completion weaknesses are recorded separately in [the September 26 test record](../../reports/model_tests/2026-09-26_pretrained_12b.md). No broad factual, arithmetic, instruction-following, or competition-score improvement is assumed.

The original continuation's fixed monitor NLL fell from 3.34757 at its start to 2.97729 at step 167,000. It was 2.97679 at step 160,000, indicating that late gains had flattened on this small monitor panel. That evidence supports a short, measured experiment rather than an unbounded token increase. The prior full data audit found EOS at document ends, zero unknown tokens, and no exact training/held-out document overlap under its declared checks. This does not prove semantic decontamination or establish a cause for repetition.

The LR rewarm/redecay approach follows [Ibrahim et al., Simple and Scalable Strategies to Continually Pre-train Large Language Models](https://arxiv.org/abs/2403.08763). Their experiments were at different model scales and data settings. Our budget, rates, and gates are engineering choices, not published optima for a 46M model.

The [Track 01 rules](https://gibc-v2.devpost.com/rules) require scratch training and prohibit pretrained initialization, fine-tuning an existing model, and larger-model distillation; the [Resources note](https://gibc-v2.devpost.com/resources) also excludes fine-tuning. We treat continuation of this same scratch lineage and raw pretraining objective as continued pretraining. This is our reading of the published rules, not an organizer ruling on our particular checkpoint sequence. SFT/DPO remain outside the competition run.

## Frozen training contract

| Item | Setting |
|---|---|
| Parent | Completed `runs/continuation_12b`, 12,000,034,816 cumulative targets |
| Parent checkpoint | `step_0167848_1790219210928589320.pt` |
| Parent checkpoint SHA256 | `ac189c926950bbba38234a0b0a237f4d7f1791ad7560e1a81bd206e6788f2040` |
| New run | `runs/continuation_13b` |
| Added target budget | **1,000,013,824**, exactly 15,259 optimizer updates |
| Cumulative endpoint | **13,000,048,640** next-token targets |
| Architecture | Same 46,346,752 parameters, 16,384-token tokenizer, 1,024-token context |
| Data | Same audited corpus, 60% FineWeb-Edu / 30% DCLM / 10% Wikipedia |
| Inherited state | Weights, Adam moments/step counters, data cursors, and sampler/RNG state |
| Data coverage | Continue after the parent's source cursors; no source wraps or repeated targets |
| Batch | 64 × 1,024 targets; microbatch 8; 65,536 targets per update |
| Precision | BF16 autocast; FP32 model weights and optimizer |
| LR | 3e-5 → 6e-5 over 500 updates, then cosine decay to 3e-6 |
| Optimizer | AdamW, β=(0.9, 0.95), ε=1e-8, matrix weight decay 0.1, clip norm 1.0 |
| Stability pilot | First 1,000 updates / 65,536,000 targets; included in the total budget |
| Checkpoints | Every 500 updates plus cumulative 12.5B/13B milestones and final export |
| Monitoring | Every 1,000 updates, 32 existing monitor documents per source |
| Training wall ceiling | Four elapsed hours for this new run, including same-run recovery |
| Hardware | Assigned physical GPU 0, NVIDIA H100 NVL |

At the previous measured throughput of roughly 130K–140K targets/s, training projects to about two hours plus validation/checkpoint overhead. The live rate determines the actual estimate. No new GPU, cloud spending, or corpus download is planned.

Config: [training](../../archive/v1_code/configs/train_continuation_13b.json), [evaluation and execution protocol](../../archive/v1_code/configs/continuation_13b_protocol.json).

## Required checks and automatic execution

1. Verify the complete local ancestry back to random initialization, cumulative token ledger, final parent checkpoint, and export identity.
2. Verify corpus/index bytes against the original audit, simulate the exact inherited sampler for every planned update, and reject any run that exceeds unread source capacity.
3. Partition each source's previously unscored `development` panel by a fixed hash salt into 256 development and 256 confirmation documents. Freeze those identities and all code/config before model scoring.
4. On the actual H100, verify exact weight/Adam/cursor inheritance, interrupted-versus-uninterrupted updates, and export equality in a disposable run. Four discarded update executions are recorded separately and excluded from the final model's token count.
5. Measure the unchanged parent on the new development partition and existing monitoring panel.
6. Train the first 1,000 updates. Continue automatically only if all updates are finite, weighted monitor NLL rises by no more than 0.10 nats, and no source NLL rises by more than 0.15 nats. These are divergence guards, not proof of improvement.
7. Finish the registered budget, export the model, and compare the final endpoint with the unchanged parent. No intermediate checkpoint is chosen using official scores.
8. If development passes the fixed gate, evaluate one confirmation comparison. Record the recommendation and retain the original 12B entry and its frozen evidence.

The new trainer adds chained scratch ancestry and explicit data-state inheritance. Existing run snapshots and the original competition package preserve the earlier implementation. New code is frozen for this run before launch; changed code/configuration is rejected on resume.

## Evaluation and decision rules

- Raw text only: 256 documents per source per phase, up to 1,024 predictable targets per document, 1,024-token context, stride 512, FP32 inference with TF32 disabled.
- Each phase compares exactly the same parent/candidate document IDs and scored-token counts.
- Report source NLL, source perplexity, the fixed 60/30/10 weighted NLL, and paired document bootstrap uncertainty (2,000 replicates).
- Require weighted NLL improvement of at least **0.005 nats**, a paired 95% interval entirely below zero for candidate-minus-parent NLL, and no source point-estimate regression above **0.02 nats**.
- Apply those same rules to development and, only after development passes, confirmation. A failed gate recommends retaining the 12B parent; it does not automatically change the recipe or restart training.
- Raw-likelihood gains do not establish better generation quality or benchmark accuracy. Neither the twelve diagnostic prompts nor official benchmark items become training data.

The 12B official benchmark results have already been observed. This is a new, explicitly registered follow-up experiment using a separate document comparison, not a retroactive extension of the original blind selection. The workflow records this fact. It does not rerun official benchmarks or modify the existing selection/package. Any later official result must be reported as a subsequent experiment alongside the original result.

## Commands and recovery

Prepare and validate the corpus/registration without starting training:

```bash
PYTHONPATH=src OMP_NUM_THREADS=4 /home/bufsgpu/yes/envs/sw/bin/python \
  scripts/run_pretraining_next.py --prepare-only
```

Launch under a persistent tmux session with the assigned GPU exposed:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false \
  /home/bufsgpu/yes/envs/sw/bin/python -u scripts/run_pretraining_next.py
```

After interruption, rerun the same command with `--resume`. The supervisor checks its registration and cached readiness evidence; the trainer verifies the checkpoint and runtime fingerprints. Inspect a failed pilot before starting any new experiment. An intentional SIGTERM during training saves at an update boundary and returns a paused state. A tmux session survives terminal disconnection, not a host reboot.

Status, preflight, recovery proof, pilot gate, parent/candidate evaluations, comparison, and source snapshots live in `artifacts/pretraining_13b/`. Real training checkpoints, logs, status, and final export live in `runs/continuation_13b/`. Stop only the PID confirmed in those live files; never terminate unrelated GPU users.
