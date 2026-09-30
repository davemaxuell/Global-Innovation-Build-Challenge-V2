# 12B-token continuation: execution contract and runbook

> Current implementation after the September 29 cleanup: [CURRENT_PIPELINE.md](V1_CURRENT_PIPELINE.md). This reference retains historical rules, plans or execution details; completed model construction is recorded in [BEST_CHECKPOINT_TRAINING.md](V1_BEST_CHECKPOINT_TRAINING.md).

**Authorized:** 2026-09-23; user requested proceeding with the researched 10B+ training extension.  
**Goal:** continue our own scratch-trained 46,346,752-parameter model to **12,000,034,816 cumulative next-token targets**, on physical **GPU 0 only**.  
**Live state:** [workflow status](../../artifacts/extension_workflow.json), [data status](../../data/processed/extension_12b/status.json), and, after GPU launch, [training status](../../runs/continuation_12b/status.json). These files override dated observations below.

## Current execution

**Verified 2026-09-23 17:21 KST:** corpus preparation and data preflight are complete; the 1,000-update pilot passed. Full continuation is running on GPU 0 at step 26,550 / 167,848. Cumulative training: 2,739,929,088 / 12,000,034,816 targets (22.8%); new continuation targets: 1,739,980,800. Recent speed: 131,129 targets/s, giving approximately 19.6 hours remaining if sustained. This is a dated observation, not a completion claim. [Live training status](../../runs/continuation_12b/status.json) takes precedence.

Corpus preparation finished at 2026-09-23 13:12:51 KST. Data auditing and the monitored pilot passed. Supervisor `gibc-train12` resumed the full schedule and is executing real-data GPU updates.

At setup verification, **71 CPU tests passed**. The [real H100 recovery/export test](../../artifacts/extension_gpu_recovery.json) passed with exactly equal resumed gradients and parameters. It also verified unchanged parent weights and inherited Adam moments/step counters. Four disposable synthetic updates (262,144 targets) are excluded from the model's training-token total.

The first 1B run remains complete and preserved. Do not report the 12B endpoint as complete until training status says `completed`, its cumulative count matches the target, and the export exists.

## Frozen training contract

| Item | Value |
|---|---|
| Parent | Completed `runs/baseline_1b`, 999,948,288 targets; 15,258 updates |
| Parent checkpoint SHA256 | `3cf62ff7a2d19fbfe1d2ca58603542bff666d087698a2d17b1f8981087fbb558` |
| Added training | 11,000,086,528 targets; 167,848 updates |
| Cumulative endpoint | 12,000,034,816 targets |
| Architecture/tokenizer | Unchanged 46,346,752-parameter Llama-style decoder; 16,384-token byte BPE |
| Continuation mixture | 60% FineWeb-Edu, 30% DCLM, 10% English Wikipedia; seeded categorical sampling of packed sequences |
| Effective batch | 64 sequences × 1,024 targets = 65,536; microbatch 8, accumulation 8 |
| Precision/numerics | BF16 autocast, FP32 weights and optimizer; deterministic SDPA execution |
| Inherited state | Model parameters, Adam moments and optimizer step counters |
| Reset state | New data cursors, sampler RNG seeded 20260923, and declared LR schedule |
| LR schedule | Linear 6e-5 → 3e-4 over 1,000 updates; cosine decay to 3e-5 at the final update |
| Optimizer | AdamW β=(0.9,0.95), ε=1e-8, matrix weight decay 0.1, gradient norm clip 1.0 |
| Monitoring | Every 1,000 updates; 32 held-out documents per new source, up to 1,024 predictable targets/document |
| Checkpoints | Every 500 updates; latest two plus permanent milestones crossing cumulative 2B/5B/10B/12B |
| Wall ceiling | 36 elapsed hours for this continuation, including monitored training and same-run recovery |
| Config | [configs/train_extension.json](../../archive/v1_code/configs/train_extension.json) |

Milestones round upward to an actual complete update. At baseline elapsed throughput, the extension projects roughly **23 H100-hours**, excluding preparation. Actual throughput and completion time must be measured after the pilot. No additional GPU or paid service is launched.

Learning-rate rewarming/redecay is informed by [Simple and Scalable Strategies to Continually Pre-train Large Language Models](https://arxiv.org/abs/2403.08763). Exact rates, duration, and mixture are engineering choices, not published optimal settings for a 46M model. Source evidence and licenses are in [CORPUS_PLAN_10B_PLUS.md](CORPUS_PLAN_10B_PLUS.md).

## Data contract

[configs/data_extension.json](../../configs/data_extension.json) freezes repository revisions, output directory, source quotas, tokenizer inheritance, and holdout counts. Prepare at least **8B Edu + 4B DCLM + 1.5B Wikipedia stored BPE tokens**. The spare pool provides sampling headroom; it does not raise the training budget. Retain complete documents including EOS, allowing a small quota overshoot.

The preparer:

1. Seeds the duplicate index with all previously collected raw documents and held-out panels, including raw text that never reached training. This conservatively excludes more than previously trained tokens.
2. Uses normalized content SHA256, canonical URLs, and an approximate 32-component MinHash over five-word shingles. Eight four-component bands find candidates; at least 28 matching components trigger rejection. Buckets retain up to four representatives.
3. Collects Wikipedia first and rejects Wikipedia-hosted pages from the web sources. Copies on other sites are subject to the approximate filter.
4. Applies the original benchmark 13-word exact-overlap and WikiText article-title exclusions to every new source.
5. Assigns first retained representatives to train/monitor/development/selection/final via normalized-content hash. Duplicate rejection precedes split assignment. Monitor has 128 documents/source; other panels have 512/source. Surplus reserved-bucket documents never enter training.
6. Saves source URL/ID, source shard and row, content hash, raw text, packed uint16 IDs, and document offsets/lengths. Source text is the only tokenized field; quality ratings and generated annotations are excluded.
7. Commits LMDB deduplication and cursor state only after output fsync. On restart it truncates output to committed positions. Changed preparation code/configuration is rejected.

This is **approximate document-level** near-duplicate filtering. It can miss near duplicates, excerpts, or paraphrases; it does not certify semantic decontamination or factual truth. First-representative cluster assignment inherits these limits. Quality-filtered web corpora are not guaranteed entirely human-authored.

The preflight verifies quotas, vocabulary bounds, hashes, downloaded source-file checksums, continuous document indexes, EOS boundaries, held-out exact separation, panel counts, and enough source capacity for the **entire deterministic training schedule**. It does not score official benchmarks. See [data audit](../../artifacts/extension_data_preflight.json), created on completion.

## Pilot and automated handoff

Before updates, measure the new 60/30/10 monitor and original 80/20 monitor. Then run exactly **1,000 updates / 65,536,000 additional targets**, reaching the end of LR rewarming. These updates count toward the full target.

Full continuation requires finite updates, new weighted monitor NLL increase ≤0.25 nats, legacy weighted increase ≤0.35, and every individual source increase ≤0.5 relative to the inherited parent. These predeclared divergence checks do not establish statistical improvement or guarantee later stability. Failure stops the supervisor for diagnosis without silently changing the recipe.

The pilot writes [gate evidence](../../artifacts/extension_pilot_gate.json). On success, a new trainer instance verifies and resumes the saved checkpoint with unchanged settings and cursors.

## Monitoring and recovery

Run from the project root using `/home/bufsgpu/yes/envs/sw/bin/python` (the validated environment).

```bash
cat artifacts/extension_workflow.json
tail -n 3 data/logs/prepare_extension_12b.log
cat data/processed/extension_12b/status.json
cat runs/continuation_12b/status.json
tail -n 3 artifacts/extension_workflow.log
```

Training status exists only after GPU initialization. While data is preparing, there is no new real-data loss or reliable completion ETA.

To interrupt, send SIGTERM only to the PID confirmed in current workflow/training status; the trainer saves at the next update boundary. Never terminate unrelated GPU processes. Corpus preparation is separately resumable from committed transactions. Avoid editing `src/scglm/` or frozen configs while jobs are active.

If preparation exits, rerun the identical command in a persistent session:

```bash
PYTHONPATH=src OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  /home/bufsgpu/yes/envs/sw/bin/python -u -m scglm.prepare_extension \
  --config configs/data_extension.json
```

After a **passed pilot**, resume full training with:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false \
  /home/bufsgpu/yes/envs/sw/bin/python -u scripts/run_extension.py --resume
```

Before pilot completion, interruption recovery requires inspecting saved state and finishing the same pilot with the original pre-pilot monitor evidence. Do not start a fresh run over an existing run directory or bypass a failed gate. Full-run resumption verifies code, environment, data, checkpoint, and configuration fingerprints.

## What follows pretraining

Preserve the 1B parent, milestone checkpoints, and final export. A 1B-versus-12B result measures combined additional training and corpus changes; it cannot establish an adaptive-controller gain. Register a continuation endpoint-selection extension before opening official scores; the existing matched-1B selector is not appropriate unchanged.

Official task settings remain frozen in [EVALUATION_PROTOCOL.md](../../EVALUATION_PROTOCOL.md). SFT/DPO/RL remains separate under [POST_TRAINING_PLAN.md](../../artifacts/pipeline_v1/submission_package/POST_TRAINING_PLAN.md) and is not launched by this script. The submission still needs model selection, official evaluation, model/data cards, reproducibility materials, and presentation artifacts.
