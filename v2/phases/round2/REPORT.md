# Round 2 of improving V2.1: `v2_round2_20261001`

**Status:** registered 2026-10-01 ~14:05 KST on branch `round2-20261001`; running. [Registration](registration.json) · [Config](config.json) · [Status](status.json) · [Events](events.jsonl)

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

Pending.
