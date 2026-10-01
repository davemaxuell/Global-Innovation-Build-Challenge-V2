# Weight interpolation (WiSE-FT) between V2 and its benchmark-tuned descendant: `v2_wiseft_20261001`

**Status:** completed 2026-10-01 11:37 KST. **Outcome: α = 0.5 selected under the registered development-only rule.** Official four-task mean **49.60 vs 47.12** for V2; WikiText-103 perplexity **24.09 vs 23.85**. [Registration](registration.json) · [Config](config.json) · [Status](status.json) · [Events](events.jsonl)

## Why

The task-tune phase ([A14](../task_tune/REPORT.md)) raised the official four-task mean from 47.12 to 52.76, but WikiText-103 perplexity rose from 23.85 to 25.46. No candidate passed the registered Wikipedia bits/byte guard. WiSE-FT (Wortsman et al., "Robust fine-tuning of zero-shot models", CVPR 2022) averages the weights of a pretrained model and its fine-tuned descendant. In that work, the mixture kept most of the fine-tuning gain while recovering the pretrained model's broader behaviour. This phase tests whether that trade-off holds for V2.

## Method

- **Formula:** θ(α) = (1−α)·θ0 + α·θ1 over all parameters (tied embeddings once), computed in float64 and stored as float32. No training.
- **θ0:** V2, `9aac2111…`.
- **θ1:** `v2/runs/task_tune_lr2e-5_r0.5/epoch_3`, `aaf13826…`.
- **Grid:** α ∈ {0.1, …, 0.9}.

## Selection rule (registered before any interpolated model existed)

- **Candidates:** among the α values whose Wikipedia selection-panel bits/byte is at most +0.01 worse than V2's, take the highest four-task mean on the held-out 20% of the benchmark training splits, scored in harness format.
- **Threshold:** select it only if it is at least +1.0 point above V2; otherwise keep V2.
- **Official evaluation:** once, on the selected α, or on the best eligible α if V2 is kept.

## Results

### Development-only selection (frozen 11:30 KST, [selection.json](selection.json))

Accuracy is on the held-out 20% of the benchmark training splits, in harness format. Bits/byte changes are against V2, on the selection panels (512 documents per source).

| Model | 4-task mean | HellaSwag | ARC-E | PIQA | WinoGrande | Wikipedia bits/byte Δ | Mixture bits/byte Δ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| V2 (α = 0) | 46.64 | 27.83 | 43.60 | 61.85 | 53.27 | — | — |
| α = 0.1 | 46.74 | 28.08 | 43.60 | 62.17 | 53.10 | +0.0007 | +0.0003 |
| α = 0.2 | 46.93 | 28.36 | 43.60 | 62.29 | 53.48 | +0.0018 | +0.0011 |
| α = 0.3 | 47.84 | 28.72 | 45.77 | 62.83 | 54.06 | +0.0034 | +0.0023 |
| α = 0.4 | 48.47 | 29.30 | 47.29 | 63.05 | 54.22 | +0.0055 | +0.0038 |
| **α = 0.5** | **48.81** | 30.09 | 47.29 | 63.52 | 54.32 | **+0.0081** | +0.0058 |
| α = 0.6 | 49.29 | 30.70 | 47.94 | 63.80 | 54.71 | +0.0111 | +0.0083 |
| α = 0.7 | 50.22 | 32.63 | 49.02 | 64.40 | 54.84 | +0.0147 | +0.0111 |
| α = 0.8 | 51.28 | 35.19 | 49.89 | 64.53 | 55.50 | +0.0187 | +0.0144 |
| α = 0.9 | 52.21 | 38.43 | 49.67 | 65.28 | 55.46 | +0.0234 | +0.0181 |
| (α = 1, task-tune phase) | 52.88 | 40.68 | 49.89 | 65.50 | 55.44 | +0.0284 | +0.0224 |

**Decision:** α = 0.5 ("candidate: proxy gain +0.0217 >= 0.01, wiki bpb change +0.0081"). It is the highest α within the Wikipedia guard; α = 0.6 exceeded it at +0.0111.

### Official evaluation (pinned protocol, [summary](official/alpha_0.50/attempt_1/summary.json))

Export `v2/runs/wiseft/alpha_0.50/export`, weight SHA256 `7e4567ec…` (full hash in the summary).

| Model | HellaSwag | ARC-Easy | PIQA | WinoGrande | 4-task mean | WikiText-103 ppl ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| V2 (pretraining only) | 28.85 ± 0.45 | 47.39 ± 1.02 | 60.77 ± 1.14 | 51.46 ± 1.40 | 47.12 | **23.85** |
| **WiSE-FT α = 0.5 (selected)** | 30.47 ± 0.46 | 51.73 ± 1.03 | 63.38 ± 1.12 | **52.80** ± 1.40 | 49.60 | 24.09 |
| Task-tuned, α = 1 (A14; not selected) | **39.15** ± 0.49 | **54.12** ± 1.02 | **65.34** ± 1.11 | 52.41 ± 1.40 | **52.76** | 25.46 |

Raw accuracy (%) ± standard error. Normalized accuracy for α = 0.5: HellaSwag 33.02, ARC-Easy 48.61, PIQA 63.87.

**Reading:**
- **Accuracy:** α = 0.5 gains +2.48 points on the four-task mean: ARC-Easy +4.3, PIQA +2.6, HellaSwag +1.6, WinoGrande +1.3. The mean gain is about 4.7 standard errors. The WinoGrande change alone is within noise.
- **Perplexity:** WikiText-103 perplexity rises 1.0% (23.85 → 24.09) and stays better than V1's 25.52.
- **Shape of the trade-off:** the Wikipedia cost grows faster than linearly in α; α = 0.5 carries 29% of the full cost. ARC-Easy and PIQA gains arrive early. Most of the HellaSwag gain appears only at α ≥ 0.7, which the guard excludes.

### Disclosure

- **Training data:** half of this model's weight comes from a model fine-tuned on the benchmarks' training splits. Its scores are therefore not comparable with pretraining-only models at face value and should be reported as such, next to V2.
- **Prior observations:** both endpoints' official scores were observed before this phase. The interpolation grid, endpoints and rule were fixed before any interpolated model existed.
- **Selection:** α was chosen on development data only.

## Decision and next step

- **Phase outcome:** α = 0.5 is the phase's selected model.
- **Not yet adopted:** whether it replaces `checkpoints/v2_best` as the submission is the user's decision. Nothing in `checkpoints/` was changed.
