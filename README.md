# SCG-LM V2: a 46M-parameter language model trained from scratch

SCG-LM is a 46,346,752-parameter Llama-style decoder trained from **random initialization** on one H100: 45.0B tokens of human-written English in 31.9 hours. The selected checkpoint is [checkpoints/v2_best](checkpoints/v2_best/).

| Model | HellaSwag | ARC-Easy | PIQA | WinoGrande | Mean | WikiText-103 ppl ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| V1 baseline (13B tokens) | 28.48 | 46.59 | **62.08** | 50.67 | 46.96 | 25.52 |
| **V2 (45B tokens)** | **28.85** | **47.39** | 60.77 | **51.46** | **47.12** | **23.85** |

Zero-shot raw accuracy (%) with lm-evaluation-harness v0.4.12; held-out WikiText-103 validation perplexity (context 1,024, stride 512). The protocol is pinned in [configs/evaluation.json](configs/evaluation.json), with details in [EVALUATION_PROTOCOL.md](EVALUATION_PROTOCOL.md). The checkpoint was chosen on development data before any official V2 score was computed. The perplexity gain is clear; the mean-accuracy gain is within per-task standard errors (0.45–1.4 points).

- **[BEST_CHECKPOINT_TRAINING.md](BEST_CHECKPOINT_TRAINING.md):** how the model was built: architecture, tokenizer, corpus, schedule, compute, provenance and results.
- **[CURRENT_PIPELINE.md](CURRENT_PIPELINE.md):** code map, verification and reproduction commands.
- **[FAILED_APPROACHES.md](FAILED_APPROACHES.md):** what did not work (V1 post-training, V2 decay-mix A/B) and why it is not repeated.
- **[TRAINING_PHASES.md](TRAINING_PHASES.md):** dated journal of every training phase; per-phase records in [v2/phases/](v2/phases/).

## Quick start

```bash
PYTHONPATH=src python scripts/demo.py --prompt "The water cycle begins when"   # text completion, CPU
PYTHONPATH=src python scripts/verify_v2.py --load                                # read-only integrity check
python -m pytest                                                                 # CPU tests
```

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained("checkpoints/v2_best", local_files_only=True)
model = AutoModelForCausalLM.from_pretrained("checkpoints/v2_best", local_files_only=True)
```

V2 is a base model. It continues text; it has not been instruction-tuned.

In a fresh clone, first follow **Setup** in [CURRENT_PIPELINE.md](CURRENT_PIPELINE.md). The model weights, prepared data and the pinned evaluation harness are not stored in the repository.

## What changed from V1

1. **2.9× throughput** (137k → about 395k tokens/s) from `torch.compile` and a single 64-sequence microbatch. On identical data and seed, the loss matched the V1 trainer over the steps compared.
2. **A richer corpus of 22.45B unique tokens:** DCLM, FineWeb-Edu, Wikipedia, StackExchange (54 everyday and science sites), Project Gutenberg, and a 15% benchmark-targeted slice chosen by a fastText classifier trained on benchmark **train** splits.
   - **Contamination controls:** 13-word n-gram exclusion over benchmark train and eval splits and WikiText-103; whole-domain blocks on wikiHow and Instructables (the sources of HellaSwag and PIQA); exact, URL and MinHash deduplication.
3. **Schedule:** warmup, constant learning rate, and a final 10% linear decay (WSD) over 45B tokens, 3.5× V1.
4. **Kept on purpose:** the architecture, AdamW and the V1 tokenizer, each after a registered check ([v2/PLAN.md](v2/PLAN.md)).

## Repository map

| Path | Contents |
| --- | --- |
| `src/scglm/` | Core library: model, token streams, trainer, evaluation, V1-lineage data preparation (byte-identical to the code that trained V2) |
| `src/scglm_v2/` | V2 pipeline: corpus build, targeted classifier, WSD trainer, selection, V1 comparison, figures |
| `scripts/` | Official evaluation runner, demo, verification, auto-resume supervisor |
| `configs/` | Pinned evaluation protocol, model, V1-lineage data configs |
| `tests/` | CPU tests for everything above |
| `v2/` | V2 run configs, corpus, runs, phase records and submission drafts |
| `checkpoints/` | `v2_best` (selected); V1 `competition_base` and `best_sft` (baselines) |
| `release/hf_v2/` | Hugging Face release: model card, configs, tokenizer, `eval_results.json` (weights not in git) |
| `runs/`, `artifacts/`, `data/` | Historical runs, measured results and prepared data (evidence; unchanged) |
| `docs/history/`, `archive/` | V1-era documents and retired code (inventory in `reports/cleanup/2026-10-01/`) |
| `vendor/` | Pinned lm-evaluation-harness checkout; fastText for data preparation |

## Competition notes

Track 01 (TECH): trained from scratch with no pretrained weights, no fine-tuning of an existing model and no distillation. The parameter count, 46,346,752 with tied embeddings, is audited on every load. Hardware: one NVIDIA H100 NVL. V1 used 26.5 GPU-hours for 13B tokens; V2 used 31.9 GPU-hours for 45B tokens. Dataset licenses are listed in [v2/submission/BUILT_WITH.md](v2/submission/BUILT_WITH.md).

## License

Code and documentation are released under the [MIT License](LICENSE). Dataset-derived records in this repository keep their original terms, for example the web-text excerpts in `v2/data/rich_v1/targeted/selected_examples.json`; see [v2/submission/BUILT_WITH.md](v2/submission/BUILT_WITH.md) for each source's license. The pinned lm-evaluation-harness is not included and keeps its own license. The model weights are distributed separately under the license stated in their [model card](release/hf_v2/README.md).

AI assistance: Claude (Anthropic) assisted V2's research, implementation, tests and documentation; Codex/ChatGPT assisted V1. All code is in this repository.
