# V2 pilot: `v2_pilot_1b_20260929`

**Status:** completed 2026-09-29 18:32 KST. **Decision (registered rule): GO.** The margin is small and within noise; see below.

- **Run:** `v2/runs/pilot_1b/`: 15,258 updates, 999,948,288 targets, 43 min at about 400k tokens/s. Final training loss 3.284. Export: `v2/runs/pilot_1b/export/`.
- **Comparison:** V1 `runs/baseline_1b/export`, with the same architecture, seed, batch, peak LR and cosine schedule. The data differs.
- **Data:** manifest SHA256 `6749a2f1…6a68`. [Gate record](gate.json). Development data only.

## Held-out benchmark TRAIN items, zero-shot accuracy

| Set (n) | V1 1B | V2 pilot | Δ |
| --- | ---: | ---: | ---: |
| ARC-Easy (461) | 32.8% | 35.4% | +2.6 |
| PIQA (1000) | 60.1% | 57.4% | −2.7 |
| HellaSwag (1000) | 28.2% | 28.7% | +0.5 |
| WinoGrande (1000) | 50.6% | 51.8% | +1.2 |
| **Gate: mean of these four** | **42.91%** | **43.31%** | **+0.40** |
| ARC-Challenge (246) | 17.5% | 17.1% | −0.4 |
| OpenBookQA (999) | 15.8% | 16.1% | +0.3 |
| SciQ (1000) | 67.8% | 70.7% | +2.9 |
| CommonsenseQA (1000) | 22.0% | 22.6% | +0.6 |

The standard error of a single-set difference at n=1000 is about 2.2 points, so no single-task change is statistically clear. Six of eight sets point in V2's favour.

## Bits per byte on V2 development panels (lower is better)

| Model | Edu | DCLM | Targeted | Wiki | StackEx | Books |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| V1 1B | 1.146 | 1.277 | 1.235 | **1.077** | 1.440 | 1.495 |
| V2 pilot | **1.119** | **1.237** | **1.194** | 1.145 | **1.236** | **1.359** |

V2 is better on every source except Wikipedia. V1's 1B run used 20% Wikipedia against V2's 10%; V1's final 13B lineage used about 11%. Wikipedia fit matters for WikiText-103 perplexity.

## Next step

Main run registered and ready (`v2/configs/main.json`); launch awaits the user's go-ahead.
