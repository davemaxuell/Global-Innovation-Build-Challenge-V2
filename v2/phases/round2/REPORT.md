# Round 2 of improving V2.1: `v2_round2_20261001`

**Status:** completed 2026-10-01 14:48 KST on branch `round2-20261001` (registered 13:47 KST). **Outcome: V2.1 is kept.** No candidate beat it by +0.5 half-B points with Wikipedia bits/byte no worse than V2.1's, and the 100% retrain of V2.1's recipe failed the Wikipedia check. No new official evaluation was run. [Registration](registration.json) · [Config](config.json) · [Status](status.json) · [Events](events.jsonl)

**Goal:** beat the submitted SCG-LM V2.1 (official mean 52.76, WikiText-103 25.46) on the four benchmarks, with Wikipedia bits/byte no worse than V2.1's.

**Approaches:**
1. Greedy model soup of the 9 existing fine-tuned models.
2. Two stronger tuning arms from V2: 6 epochs at LR 2e-5, and 3 epochs at LR 4e-5.
3. Greedy soup of all 18 fine-tuned models.
4. A final retrain of the winning recipe on 100% of the training splits.

**Selection:**
- **Halves:** the held-out items are split into half A, which builds the soups, and half B, which selects.
- **Rule:** the winner must beat V2.1 on half B by at least +0.5 points, with Wikipedia bits/byte no worse than V2.1's.
- **Final model:** committed in advance to be the 100%-retrained version of the winning recipe, if it passes the Wikipedia check.
- **Official evaluation:** once, on the final model.

## Results

### Training (actual)

| Arm | Updates | Time | Status |
| --- | ---: | ---: | --- |
| lr2e-5_e6 (6 epochs) | 14,898 | ~29 min, concurrent | completed |
| lr4e-5_e3 (3 epochs) | 7,449 | 19.4 min, concurrent | completed |
| 100% retrain of V2.1's recipe (lr2e-5_r0.5, 3 epochs) | 9,294 | ~12 min | completed |

The 100% retrain saw 297,315 questions over 3 epochs: 99,105 per epoch, the 79,425 original plus 19,680 held-out items after 127 overlap drops.

### Greedy soups (built on half A)

- **Soup of existing models:** V2.1 alone scored 52.18 on half A. Only V2.1's own epoch-2 checkpoint was kept (it tied), giving 52.18.
- **Soup of all models:** members are lr4e-5_e3 epoch 3 and lr2e-5_e6 epoch 5, at 55.35. The best single model scored 55.31.

### Development-only selection (half B, frozen 14:35 KST, [selection.json](selection.json))

| Model | Half-B mean | HellaSwag | ARC-E | PIQA | WinoGrande | Wikipedia bits/byte (Δ vs V2.1) | Eligible |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| V2.1 (reference) | 53.67 | 39.86 | 52.58 | 66.94 | 55.30 | 1.0116 (+0.0000) | — |
| round2_lr2e-5_e6/epoch_1 | 51.63 | 36.10 | 51.64 | 65.48 | 53.31 | 1.0064 (-0.0053) | yes |
| round2_lr2e-5_e6/epoch_2 | 53.26 | 38.42 | 52.11 | 66.75 | 55.75 | 1.0155 (+0.0039) | no |
| round2_lr2e-5_e6/epoch_3 | 55.66 | 40.58 | 53.05 | 66.88 | 62.11 | 1.0214 (+0.0098) | no |
| round2_lr2e-5_e6/epoch_4 | 56.40 | 40.43 | 52.58 | 67.64 | 64.95 | 1.0231 (+0.0115) | no |
| round2_lr2e-5_e6/epoch_5 | 56.82 | 40.84 | 51.64 | 67.89 | 66.92 | 1.0245 (+0.0128) | no |
| round2_lr2e-5_e6/epoch_6 | 57.31 | 40.53 | 52.58 | 67.89 | 68.23 | 1.0248 (+0.0132) | no |
| round2_lr4e-5_e3/epoch_1 | 52.75 | 39.09 | 51.17 | 65.80 | 54.95 | 1.0229 (+0.0113) | no |
| round2_lr4e-5_e3/epoch_2 | 56.17 | 40.20 | 53.05 | 67.51 | 63.93 | 1.0295 (+0.0179) | no |
| round2_lr4e-5_e3/epoch_3 | 57.23 | 41.36 | 52.58 | 68.08 | 66.89 | 1.0297 (+0.0181) | no |
| round2_soup_existing | 52.85 | 39.78 | 50.70 | 66.24 | 54.65 | 1.0104 (-0.0012) | yes |
| round2_soup_all | 56.91 | 41.38 | 51.17 | 68.02 | 67.06 | 1.0256 (+0.0139) | no |

**Decision:** "reference: best eligible half-B gain -0.0082 < 0.005". The eligible candidates (the soup of existing models and lr2e-5_e6 epoch 1) scored below V2.1. Every stronger model failed the Wikipedia condition.

### 100% retrain and final check ([final.json](final.json))

V2.1's recipe was retrained on 100% of the training splits. Its Wikipedia bits/byte was 1.0153, worse than V2.1's 1.0116, so it failed the check. The registered fallback, the development-selected model, is V2.1 itself, so no official evaluation was run. Its half-B score (79.1) is not meaningful because it trained on those items.

### What this round shows

- **The trade-off is steep and consistent.** Stronger tuning reaches 57.2–57.3 on half B, +3.6 points over V2.1, but at +0.013 to +0.018 Wikipedia bits/byte beyond V2.1.
- **Rough perplexity cost, an estimate rather than a measurement:** scaling by the V2 → V2.1 change (+0.0284 bits/byte ↔ WikiText-103 23.85 → 25.46), that is about 26.2–26.5.
- **Soups did not help.** Uniform weight averages of fine-tuned models did not beat the best single model on held-out data.
- **More data has the same cost.** The 100% retrain added 25% more updates and crossed the Wikipedia limit, like the longer runs.

## Decision and next step

- **Outcome:** V2.1 remains the submission. All round-2 exports are preserved.
- **If the user accepts more perplexity:** the strongest eligible-by-accuracy models are lr2e-5_e6 epoch 6 (half B 57.31) and lr4e-5_e3 epoch 3 (57.23). Choosing one would require changing the rule after seeing these development results, which must be disclosed; one official evaluation would follow.
