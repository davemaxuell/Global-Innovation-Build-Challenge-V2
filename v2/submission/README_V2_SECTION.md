# README section: V2 (draft, to merge into the repository README)

## V2: rich-data scratch pretraining

V2 retrains the same 46,346,752-parameter architecture from random initialization with three changes: faster training, a richer corpus, and a warmup-stable-decay schedule. Full details: [v2/README.md](../README.md), [v2/PLAN.md](../PLAN.md) and the phase reports in [v2/phases/](../phases/).

### Reproduce

From the project root, with the recorded environment (`requirements.lock.txt`):

```bash
export PYTHONPATH=src:vendor/pydeps
# 1. Data (CPU): benchmark-train texts, targeted classifier, tokenizer study, corpus build
python -m scglm_v2.download --config v2/configs/prep.json
python -m scglm_v2.benchmarks --config v2/configs/prep.json
python -m scglm_v2.targeted --config v2/configs/prep.json
python -m scglm_v2.tokenizer_study --config v2/configs/prep.json
python -m scglm_v2.build --config v2/configs/prep.json --stage reused --workers 36
python -m scglm_v2.build --config v2/configs/prep.json --stage new --workers 20
python -m scglm_v2.build --config v2/configs/prep.json --stage manifest
# 2. Pretraining (one GPU); the decay is scheduled by v2/runs/main/decay.json
python -m scglm_v2.train --config v2/configs/main.json
# 3. Development-only selection, then the pinned official evaluation
python -m scglm_v2.select_endpoint --baseline v2/runs/main --challenger v2/runs/anneal_quality \
    --out v2/phases/final_selection/selection.json
python scripts/run_final_evaluation.py --final-evaluation --model <selected export> \
    --selection-record v2/phases/final_selection/selection.json --device cuda:0
```

Tests: `python -m pytest` (CPU).

### Results
| Model | HellaSwag | ARC-Easy | PIQA | WinoGrande | 4-task mean | WikiText-103 ppl ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| V1 13B base | 28.48 | 46.59 | **62.08** | 50.67 | 46.96 | 25.52 |
| **V2 (selected, 45B tokens)** | **28.85** ± 0.45 | **47.39** ± 1.02 | 60.77 ± 1.14 | **51.46** ± 1.40 | **47.12** | **23.85** |

Zero-shot raw accuracy (%) ± standard error, lm-evaluation-harness v0.4.12 under the pinned protocol; WikiText-103 validation perplexity at context 1,024, stride 512. The checkpoint was chosen on development data before any official V2 score existed, and the pre-registered V1-vs-V2 rule selected V2. WikiText-103 perplexity improves clearly (about 6.5%). The accuracy mean rises 0.17 points, within per-task standard errors, and PIQA is 1.3 points lower. V1's official scores were observed before this experiment.

Records: [frozen selection](../phases/final_selection/selection.json) · [official summary](../phases/final_selection/official/main/attempt_1/summary.json) · [rule outcome](../phases/final_selection/v1_vs_v2.json) · [full report](../phases/final_selection/REPORT.md)

### Figures
`v2/submission/figures/` holds the key numbers, corpus composition, 1B pilot comparison and training curve. Each figure has a `.data.json` table and is regenerated with `python -m scglm_v2.figures`.
