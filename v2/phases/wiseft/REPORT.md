# Weight interpolation (WiSE-FT) between V2 and its benchmark-tuned descendant: `v2_wiseft_20261001`

**Status:** registered 2026-10-01 ~11:35 KST; running. [Registration](registration.json) · [Config](config.json) · [Status](status.json) · [Events](events.jsonl)

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

Pending.
