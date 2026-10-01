# Selected checkpoint: V2 training record

**Recorded:** October 1, 2026. **Checkpoint:** [checkpoints/v2_best](checkpoints/v2_best/). **Selected by** the pre-registered V1-vs-V2 rule ([registration](v2/phases/final_selection/registration.json), [outcome](v2/phases/final_selection/v1_vs_v2.json)).

V2 is a 46,346,752-parameter language model trained from random initialization on 44,999,966,720 next-token targets from a 22.45B-unique-token, human-written English corpus. V2 base itself is a **base model**: no SFT, DPO or RL. The submitted model adds one fine-tuning stage (section 0). How to rebuild it: [CURRENT_PIPELINE.md](CURRENT_PIPELINE.md). The V1 lineage it replaces is recorded in [docs/history/V1_BEST_CHECKPOINT_TRAINING.md](docs/history/V1_BEST_CHECKPOINT_TRAINING.md).

## 0. Submitted model: V2 task-tuned (adopted 2026-10-01)

The submission is **[checkpoints/v2_task_tuned](checkpoints/v2_task_tuned/)**, weight SHA256 `aaf138266a2689162684232ea828712c59746f7c059aa3f87d712d4599ef1897`. It is the V2 base model documented below, fine-tuned once more:

| Item | Value |
| --- | --- |
| Parent | `checkpoints/v2_best` (`9aac2111…`) |
| Data | 79,425 training-split questions in the pinned harness's scoring format: HellaSwag 31,989, WinoGrande 32,324, PIQA 12,460, ARC-Easy 1,780, ARC-Challenge 872. This is the 80% not held out for development, after dropping 554 items that overlap official evaluation items. |
| Loss | 0.5 × (choice cross-entropy over summed answer log-likelihoods at temperature 10 + 0.5 × gold-sequence LM loss) + 0.5 × replay LM loss on V2 pretraining text |
| Optimization | AdamW (0.9, 0.95), weight decay 0.1, clip 1.0; peak LR 2e-5, warmup 0.2%, linear decay to 0; 32 questions + 8 replay sequences per update; 3 epochs = 7,449 updates; 20.5 min on the H100 |
| Code | `src/scglm_v2/task_format.py`, `task_tune.py`, `task_select.py`; config `v2/phases/task_tune/config.json` (arm `lr2e-5_r0.5`) |

| Model | HellaSwag | ARC-Easy | PIQA | WinoGrande | 4-task mean | WikiText-103 ppl ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| V2 base (section 7) | 28.85 | 47.39 | 60.77 | 51.46 | 47.12 | **23.85** |
| **V2 task-tuned (submitted)** | **39.15** ± 0.49 | **54.12** ± 1.02 | **65.34** ± 1.11 | **52.41** ± 1.40 | **52.76** | 25.46 |

**How it was chosen.** The registered development-only rule did *not* select it: its held-out Wikipedia bits/byte was 0.028 worse than V2's, against a 0.01 guard. The user adopted it after the official scores of V2 base, this model and a rule-selected 50/50 weight interpolation (mean 49.60, perplexity 24.09) had all been observed. [Adoption record](v2/phases/task_tune/ADOPTION.json) · [phase report](v2/phases/task_tune/REPORT.md) · [interpolation report](v2/phases/wiseft/REPORT.md).

**Disclosure.**
- These are task-tuned scores: the model trained on the benchmarks' training splits, though never on the evaluation splits. They are not comparable at face value with pretraining-only models, so V2 base is reported alongside.
- WikiText-103 perplexity is 6.8% worse than V2 base.

Sections 1–8 document the V2 base model.

## 1. Identity

| Item | Value |
| --- | --- |
| Working copy (read-only) | `checkpoints/v2_best/` ([hash inventory](checkpoints/SELECTION_2026-10-01.json)) |
| Original export | `v2/runs/main/export/` |
| Weight SHA256 | `9aac2111b9ca6b06f4a12278f5213c17621ded643b7b54137987ce25028011bd` |
| Run | `v2/runs/main/` (science digest `b5bc1cea…1b340da4`) |
| Data manifest SHA256 | `6749a2f1e2c3fe574febf0c1d08fb99d87d6e226c995aedd0d4ae71921496a68` |
| Tokenizer SHA256 | `1d714be4561ca083ef3cb668be7cffa38caa158c0315c22907647cd4f0f9013b` (the V1 tokenizer) |
| Decay-start checkpoint SHA256 | `175d1e297a97569c0546ae491c4a0498890cd9bb303a22b52d4edab06b4bb0c6` |

## 2. Architecture (unchanged from V1)

Hugging Face `LlamaForCausalLM`, random initialization (seed 20260923): 12 blocks, width 512, 8 heads (head dimension 64, full multi-head attention), SwiGLU 1,376, RMSNorm ε 1e-5, RoPE θ 10,000, context 1,024, vocabulary 16,384, tied input/output embeddings, no biases, no dropout. The parameter count is 46,346,752, under the 50,000,000 cap. Deep-thin (24×384, grouped-query) and Muon were considered and rejected on evidence ([v2/PLAN.md](v2/PLAN.md)).

## 3. Tokenizer

A byte-level BPE with 16,384 entries, trained from scratch in V1 on 50M FineWeb and 50M Wikipedia training characters (special tokens pad 0, bos 1, eos 2, unk 3). V2's pre-registered study trained 16k and 20k candidates on the V2 mixture and would adopt one only if it compressed at least 2% better on both mixture and benchmark text. Neither did (16k: −0.6% / +3.4%; 20k: +1.6% / +6.0%), and both compressed Wikipedia worse (3.50 and 3.60 vs 3.85 bytes/token), so the V1 tokenizer was kept. [Study](v2/phases/preparation/REPORT.md).

## 4. Corpus (`v2/data/rich_v1`)

| Bucket | Unique tokens | Documents | Share of training | Passes over the data |
| --- | ---: | ---: | ---: | ---: |
| DCLM-baseline (web) | 8,286,499,743 | 5,524,265 | 35% | 1.90 |
| FineWeb-Edu | 6,023,274,513 | 5,041,791 | 30% | 2.24 |
| Benchmark-targeted (top 20% of Edu+DCLM by classifier) | 3,521,067,924 | 3,732,888 | 15% | 1.92 |
| English Wikipedia | 2,020,057,734 | 2,487,369 | 10% | 2.23 |
| StackExchange (54 English non-code sites; question score ≥ 1, answered) | 1,613,083,489 | 1,118,884 | 5% | 1.39 |
| Project Gutenberg English (20k-character chunks) | 990,115,242 | 180,773 | 5% | 2.27 |
| **Total** | **22,454,098,645** | | | |

- **Sources** (pinned revisions): FineWeb-Edu `87f09149`, DCLM `a3b142c1`, Wikipedia 20231101.en `b04c8d1c`, StackExchange 2025 `fe382c48`, Gutenberg `164853d2`. About 14.0B tokens reuse V1's already-deduplicated extension and base text; the rest is newly downloaded.
- **Contamination controls:**
  - the 13-word exclusion index over the train and eval splits of HellaSwag, ARC-Easy, PIQA and WinoGrande plus WikiText-103;
  - a whole-domain block on `wikihow.com` (the HellaSwag source) and `instructables.com` (the PIQA source), which removed 12,708 documents;
  - exact, URL and MinHash deduplication continued from V1's database.
- **Targeted classifier:** fastText, trained on 80% of benchmark **train** splits (ARC-E/C, PIQA, HellaSwag, WinoGrande, OpenBookQA, SciQ, CommonsenseQA) against length-matched pool passages; validation AUC 0.994. The other 20% of those train splits forms the development proxy panel. Official evaluation splits were never used.
- **Details:** [preparation report](v2/phases/preparation/REPORT.md).

## 5. Pretraining

| Setting | Value |
| --- | --- |
| Updates / targets | **686,645 / 44,999,966,720** (64 × 1,024 targets per update, one microbatch of 64) |
| Tokens by source | DCLM 15,747,631,104 · Edu 13,499,219,968 · targeted 6,753,447,936 · Wikipedia 4,502,865,920 · StackExchange 2,245,974,016 · books 2,250,827,776 |
| Optimizer | AdamW, β (0.9, 0.95), ε 1e-8, weight decay 0.1 on matrices (0 on vectors), gradient clip 1.0, fused |
| Learning rate (WSD) | Linear warmup to 6e-4 over 500 updates; constant to update 617,981 (40.5B tokens); linear decay to 0 over 68,664 updates (10%) |
| Numerics | BF16 autocast; FP32 weights, optimizer state and loss; `torch.compile`; non-deterministic kernels |
| Hardware / time | One NVIDIA H100 NVL; **31.9 h** accounted training time; about 395k tokens/s (V1: 137k); peak memory 28.6 GB |
| Incidents | One failure at update 10,000 (a validation labelling bug), resumed from update 8,000. 2,000 updates (131,072,000 tokens) were computed twice; 45,131,038,720 tokens were processed in total. [Run report](v2/phases/main/REPORT.md) |

**Held-out monitor NLL:** 3.457 at update 10k → about 3.17 across the constant phase (a slow plateau) → **2.971** at update 680k. The decay alone took it from 3.168 to 2.971, and every source improved by 0.17–0.24 nats.

**Code provenance:** the core modules `src/scglm/{model,data,train,evaluate,prepare_data,prepare_extension}.py` are byte-identical to the hashes recorded in `v2/runs/main/run.json`. `src/scglm_v2/train.py` was used in two versions. Updates 0–8,000 ran the launch version, snapshotted in `v2/runs/main/snapshot_v2/`. Updates 8,000–686,645 ran the resumed version, which is the retained file. It differs from the launch snapshot by exactly two fixes (validation labels by bucket, and the resume guard reading `compile` from the V2 record block); this was verified by diff. That version's bytes were not separately snapshotted when it ran.

## 6. Evidence before and after the main run

- **1B pilot** (V2 data vs V1 data, same model, seed and schedule): benchmark-proxy mean 43.31% vs 42.91% (within noise); bits/byte better on 5 of 6 sources, worse on Wikipedia. [Pilot report](v2/phases/pilot_1b/REPORT.md).
- **Decay A/B:** re-running the final 10% with a quality-weighted mix gave development proxy +0.07 points against a required +0.5, so it was not selected. [Report](v2/phases/anneal_quality/REPORT.md) · FAILED_APPROACHES A13.

## 7. Selection and measured performance

The endpoint was chosen on development data only, frozen at 07:41 KST on 2026-10-01, before any official V2 score existed ([selection record](v2/phases/final_selection/selection.json)). The official evaluation then ran under the pinned protocol ([configs/evaluation.json](configs/evaluation.json): lm-evaluation-harness v0.4.12, zero-shot; WikiText-103 validation, context 1,024, stride 512).

| Model | HellaSwag | ARC-Easy | PIQA | WinoGrande | 4-task mean | WikiText-103 ppl ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| V1 13B base | 28.48 | 46.59 | **62.08** | 50.67 | 46.96 | 25.52 |
| **V2 (this checkpoint)** | **28.85** ± 0.45 | **47.39** ± 1.02 | 60.77 ± 1.14 | **51.46** ± 1.40 | **47.12** | **23.85** |

Raw accuracy (%) ± standard error. V2 normalized accuracy: HellaSwag 30.95, ARC-Easy 42.09, PIQA 60.66. [Exact output](v2/phases/final_selection/official/main/attempt_1/summary.json).

**Reading:**
- WikiText-103 perplexity improves clearly, by about 6.5%.
- The four-task mean rises 0.17 points, which is within per-task standard errors. The rule is met, but the accuracy gain is not statistically established.
- PIQA falls 1.3 points, as the 1B pilot suggested it might.
- V1's official scores were observed before this experiment.

## 8. Loading

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained("checkpoints/v2_best", local_files_only=True)
model = AutoModelForCausalLM.from_pretrained("checkpoints/v2_best", local_files_only=True).eval()
```

Or run `PYTHONPATH=src python scripts/demo.py --prompt "..."`. The recorded environment is PyTorch 2.10.0 (CUDA 12.8), Transformers 5.5.4, Tokenizers 0.22.2, Datasets 4.8.4, NumPy 2.2.6, Safetensors 0.8.0 and lm-eval 0.4.12 ([requirements.lock.txt](requirements.lock.txt)). Data preparation also uses `fasttext-numpy2-wheel`, vendored in `vendor/pydeps`.

AI assistance: Claude (Anthropic) assisted V2's research, implementation, tests and documentation. Codex/ChatGPT assisted V1.
