# Task tuning on the benchmarks' train splits: `v2_task_tune_20261001`

**Status:** completed 2026-10-01 11:20 KST on branch `task-tuning-20261001`. **Outcome: not selected under the registered rule.** V2 stays the selected model, because no candidate kept Wikipedia bits/byte within +0.01. The best candidate's official four-task mean was **52.76 vs 47.12** (+5.64 points), with WikiText-103 perplexity **25.46 vs 23.85**. [Registration](registration.json) · [Config](config.json) · [Status](status.json) · [Events](events.jsonl)

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

### Training (actual)

Each completed arm ran 7,449 updates: 238,275 questions (3 epochs × 79,425), 684,387 candidate sequences, 41,183,073 task input tokens and 61,022,208 replay targets.

- **First lr1e-5_r0.5 run:** ran out of CUDA memory at update 882 while three arms shared GPU 0. It wrote no export; `v2/runs/task_tune_lr1e-5_r0.5` is kept as the failure record. The same arm reran unchanged, alone, into `v2/runs/task_tune_lr1e-5_r0.5_rerun`.
- **Training time:** 20.5 min (lr2e-5_r0.5) and 20.4 min (lr2e-5_r0.7), run concurrently; 9.6 min for the rerun.
- **Phase GPU time:** under 1.5 hours in total, including smoke runs, selection and evaluation.

### Development-only selection (frozen 11:03 KST, [selection.json](selection.json))

Four-task mean and per-task accuracy are on the held-out 20% of the training splits, in harness format. Bits/byte changes are against V2, on the selection panels (512 documents per source).

| Candidate | 4-task mean | HellaSwag | ARC-E | PIQA | WinoGrande | Wikipedia bits/byte Δ | Mixture bits/byte Δ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| V2 (parent) | 46.64 | 27.83 | 43.60 | 61.85 | 53.27 | — | — |
| lr2e-5_r0.5 / epoch 1 | 50.94 | 36.45 | 49.02 | 64.27 | 54.00 | +0.0213 | +0.0189 |
| lr2e-5_r0.5 / epoch 2 | 52.11 | 39.11 | 49.46 | 65.12 | 54.76 | +0.0266 | +0.0215 |
| lr2e-5_r0.5 / epoch 3 | **52.88** | 40.68 | 49.89 | 65.50 | 55.44 | +0.0284 | +0.0224 |
| lr2e-5_r0.7 / epoch 1 | 50.24 | 35.53 | 47.72 | 64.18 | 53.53 | +0.0180 | +0.0149 |
| lr2e-5_r0.7 / epoch 2 | 51.84 | 38.82 | 49.24 | 65.12 | 54.17 | +0.0231 | +0.0168 |
| lr2e-5_r0.7 / epoch 3 | 52.46 | 40.58 | 49.24 | 65.31 | 54.71 | +0.0242 | +0.0172 |
| lr1e-5_r0.5 (rerun) / epoch 1 | 49.65 | 33.83 | 47.72 | 63.64 | 53.39 | +0.0124 | +0.0112 |
| lr1e-5_r0.5 (rerun) / epoch 2 | 50.83 | 35.42 | 49.67 | 64.27 | 53.95 | +0.0158 | +0.0131 |
| lr1e-5_r0.5 (rerun) / epoch 3 | 51.49 | 37.32 | 50.54 | 64.31 | 53.80 | +0.0169 | +0.0140 |

**Decision:** keep V2 ("no candidate passed the Wikipedia bits/byte guard"). Every candidate exceeded the +1.0-point gain threshold; none stayed within +0.01 Wikipedia bits/byte.

### Official evaluation (reported, not selected)

As registered, the highest-proxy candidate was evaluated once with the pinned protocol: `v2/runs/task_tune_lr2e-5_r0.5/epoch_3/export`, weight SHA256 `aaf13826…99ef1897`. [Summary](official/task_tune_lr2e-5_r0.5_epoch_3/attempt_3/summary.json)

| Model | HellaSwag | ARC-Easy | PIQA | WinoGrande | 4-task mean | WikiText-103 ppl ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| V2 (selected; pretraining only) | 28.85 ± 0.45 | 47.39 ± 1.02 | 60.77 ± 1.14 | 51.46 ± 1.40 | 47.12 | **23.85** |
| V2 fine-tuned on the benchmarks' training splits | **39.15** ± 0.49 | **54.12** ± 1.02 | **65.34** ± 1.11 | 52.41 ± 1.40 | **52.76** | 25.46 |

Raw accuracy (%) ± standard error. Normalized accuracy: HellaSwag 30.95 → 36.14, ARC-Easy 42.09 → 51.81, PIQA 60.66 → 66.00.

**Reading:**
- **Accuracy:** HellaSwag (+10.3), ARC-Easy (+6.7) and PIQA (+4.6) gains are far outside the standard errors. WinoGrande (+0.95) is within noise, as expected from the WinoGrande paper's small-model results.
- **Perplexity:** WikiText-103 is 6.8% worse, close to V1's 25.52.
- **Development vs official:** the development gain (+6.2) slightly overstated the official gain (+5.6).

### Execution incidents (no effect on scores)

| Attempt | What happened | Fix |
| --- | --- | --- |
| Official attempt 1 | Failed before scoring: the launcher's Hugging Face offline mode blocked the runner from resolving the pinned PIQA revision. | The launcher now runs the official step online. |
| Official attempt 2 | Failed before scoring: the frozen runner matches non-selected endpoints on `results[].run`, which the selector did not write. | Added `selection_runner.json`, identical to `selection.json` plus that field, verified by script. `task_select.py` now writes the field. |

Both failed attempts are in `official/…/attempt_1`, `attempt_2` and in `v2/reports/validation_history`.

## Decision and next step

- **Under the registered rule, V2 remains the selected model.**
- **The trade-off belongs to the user:** about +5.6 points of benchmark accuracy against about +1.6 WikiText-103 perplexity. Submitting the task-tuned model now would be a choice made after seeing its official scores, and would have to be disclosed that way.
- **No follow-up is scheduled automatically.** One evidence-based option is weight interpolation between V2 and this model (WiSE-FT), with the mixing weight chosen on development data under the same rule.
