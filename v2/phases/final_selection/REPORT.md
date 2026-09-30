# V2 final selection and official evaluation: `v2_final_selection_20260929`

**Status:** completed 2026-10-01 07:54 KST. **Outcome under the pre-registered rule: V2 replaces V1 as the submission base.** The rules were confirmed by the user on 2026-09-29 at 23:37 KST, before any V2 final result existed. [Registration](registration.json) · [selection record](selection.json) (SHA256 `596f4b5f…a5dffc33b`) · [rule output](v1_vs_v2.json) · [events](events.jsonl)

## Step 1: development-only choice between the two decays (frozen 07:41 KST)

| Candidate | Proxy mean (ARC-E/PIQA/HS/WG, held-out train items) | Wikipedia bits/byte |
| --- | ---: | ---: |
| `main` (standard decay) | 45.92% | 0.9832 |
| `anneal_quality` (quality-weighted decay) | 45.99% | 0.9786 |

Challenger gain +0.07 points, against a required +0.5. **Selected: `v2/runs/main`**, export `model.safetensors` SHA256 `9aac2111…28011bd`. Counts: all held-out items (ARC-E 461, PIQA 3,177, HellaSwag 7,849, WinoGrande 8,074, plus 4 other sets) and the manifest's unused `selection` panels.

## Step 2: official evaluation (pinned protocol `configs/evaluation.json`, GPU `cuda:0`)

Raw accuracy (%), standard errors in parentheses:

| Model | HellaSwag | ARC-Easy | PIQA | WinoGrande | 4-task mean | WikiText-103 ppl ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| V1 13B base (previously observed) | 28.48 | 46.59 | **62.08** | 50.67 | 46.96 | 25.520 |
| **V2 selected (`main`, 45B tokens)** | **28.85** (0.45) | 47.39 (1.02) | 60.77 (1.14) | **51.46** (1.40) | **47.12** | 23.849 |
| V2 other arm (`anneal_quality`, not selected) | 28.79 | **47.81** | 61.92 | 49.80 | 47.08 | **23.556** |

Normalized accuracy for V2 selected versus V1: HellaSwag 30.95 vs 30.64, ARC-Easy 42.09 vs 42.38, PIQA 60.66 vs 60.39.

## Step 3: V1 vs V2 rule (applied mechanically by `scglm_v2.v1_vs_v2`)

V2's four-task mean of 47.12% is greater than V1's 46.96%, **and** its WikiText-103 perplexity of 23.85 is at most V1's 25.52. **Outcome: V2.**

## Honest reading

- **WikiText-103:** a clear improvement, about 6.5% lower perplexity, measured over 269,569 tokens.
- **Accuracy:** the gain is small. It's +0.17 points on the four-task mean, while per-task standard errors are 0.45–1.4 points, so the rule is satisfied but the accuracy difference is not statistically established. PIQA fell 1.3 points, in line with the direction the 1B pilot signalled; HellaSwag, ARC-Easy and WinoGrande rose 0.4–0.8 points.
- **The non-selected arm:** `anneal_quality` has a slightly better official perplexity (23.56) and PIQA. It was not selected under the frozen development rule, and official scores do not change that choice.
- **Disclosure:** V1's official scores were observed before this experiment.
