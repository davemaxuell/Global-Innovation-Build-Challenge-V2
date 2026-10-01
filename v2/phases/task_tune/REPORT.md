# Task tuning on the benchmarks' train splits: `v2_task_tune_20261001`

**Status:** registered; training launched 2026-10-01 (KST) on branch `task-tuning-20261001`. [Registration](registration.json) · [Config](config.json) · [Status](status.json) · [Events](events.jsonl)

## What and why

V2 is fine-tuned on the training splits of HellaSwag, ARC-Easy (plus ARC-Challenge), PIQA and WinoGrande. The examples are written in the exact format the pinned lm-evaluation-harness scores. The official evaluation splits are never trained on. The method follows GPT-1's multiple-choice fine-tuning, which scored HellaSwag 41.7 and PIQA 69.2 with a 12-layer model fine-tuned on these training sets. Here each answer is scored by its summed continuation log-likelihood, the quantity the harness compares, so no parameters are added.

**This changes what the scores mean.** A model trained on a benchmark's own training split is not comparable with pretraining-only models at face value. Any reported result is labelled "fine-tuned on the benchmarks' training splits" and shown next to V2 base.

## Data (`v2/data/task_tune_v1`)

| Train split (pinned revision) | Trained questions | Held-out (selection panel) | Dropped: overlap with official items |
| --- | ---: | ---: | ---: |
| HellaSwag | 31,989 | 7,849 | 67 |
| WinoGrande XL | 32,324 | 8,074 | 0 |
| PIQA | 12,460 | 3,177 | 476 |
| ARC-Easy | 1,780 | 461 | 10 |
| ARC-Challenge | 872 | 246 | 1 |

- **Split:** V2's existing 80/20 hash split. Preparation fails unless the counts match V2's recorded split.
- **Overlap check:** an item is dropped if its context matches an official evaluation item after normalization, or shares any 13-word sequence with one. The official splits were read as text only.
- **Format check:** the formatter reproduces all 15,523 recorded official harness requests exactly. The scorer's log-likelihoods are within 1e-4 of the recorded V2 values.
- **File hashes:** `train.jsonl` `461765c8…`, `development.jsonl` `31ae3b16…`.

## Recipe

- **Loss:** `(1 − r) · (CE(scores / 10) + 0.5 · gold-sequence LM) + r · replay LM`.
- **Batch:** 32 questions plus 8 replay sequences of 1,024 tokens per update.
- **Schedule:** 3 epochs, 7,449 updates; warmup over 0.2% of updates, then linear decay to 0.
- **Optimizer:** AdamW (0.9, 0.95) with weight decay 0.1 on matrices and gradient clipping at 1.0; BF16 autocast.
- **Arms:** LR 2e-5 with replay 0.5; LR 2e-5 with replay 0.7; LR 1e-5 with replay 0.5. An export is written at the end of each epoch, giving 9 candidates.

**Amended before launch:** 600-update smoke runs, checked on unseen training items, showed the registered recipe hurt raw-language fit.
- The registered recipe (no temperature, LR 6.25e-5, replay 0.3) gave Wikipedia bits/byte +0.081 for +0.9 points.
- Temperature 10 gave +2.9 points, but still +0.044 bits/byte at LR 6.25e-5.
- LR 1e-5 to 2e-5 with replay 0.5 gave +1.6 to +2.0 points at +0.006 to +0.012 bits/byte.

[Evidence](prelaunch_smoke.json).

## Selection rule (registered before training)

- **Candidates:** among those whose Wikipedia selection-panel bits/byte is at most +0.01 worse than V2's, take the highest four-task mean accuracy on the held-out 20%, scored in the harness format.
- **Threshold:** select it only if it is at least +1.0 point above V2; otherwise keep V2.
- **Official evaluation:** runs once on the registered model.

## Results

Pending.
