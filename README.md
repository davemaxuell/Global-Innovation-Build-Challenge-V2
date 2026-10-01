# SCG-LM V2 and V2.1: a 46M-parameter language model trained from scratch

SCG-LM is a 46,346,752-parameter Llama-style decoder trained from **random initialization** on one H100: 45.0B tokens of human-written English in 31.9 hours. The pretraining-only model is [checkpoints/v2_best](checkpoints/v2_best/). **The submitted model is SCG-LM V2.1, in [checkpoints/v2_1](checkpoints/v2_1/):** that same model, fine-tuned for 20 minutes on the training splits of the four benchmarks. See the disclosure below the table.

**Model weights on Hugging Face:** [SCG-LM V2.1](https://huggingface.co/davemaxuellkr/Global-Innovation-Build-Challenge-V2) (submitted) · [SCG-LM V2](https://huggingface.co/davemaxuellkr/scglm-v2-46m) (pretraining only). Both are MIT-licensed, with model cards and evaluation records.

| Model | HellaSwag | ARC-Easy | PIQA | WinoGrande | Mean | WikiText-103 ppl ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| V1 baseline (13B tokens) | 28.48 | 46.59 | **62.08** | 50.67 | 46.96 | 25.52 |
| V2 base (45B tokens, pretraining only) | 28.85 | 47.39 | 60.77 | 51.46 | 47.12 | **23.85** |
| **V2.1 (submitted)** | **39.15** | **54.12** | **65.34** | **52.41** | **52.76** | 25.46 |

Zero-shot raw accuracy (%) with lm-evaluation-harness v0.4.12; held-out WikiText-103 validation perplexity (context 1,024, stride 512). The protocol is pinned in [configs/evaluation.json](configs/evaluation.json), with details in [EVALUATION_PROTOCOL.md](EVALUATION_PROTOCOL.md).

**Disclosure for the submitted model:**
- **Training data:** it was fine-tuned on the *training* splits of HellaSwag, ARC-Easy/Challenge, PIQA and WinoGrande, written in the scoring format of the evaluation harness. Items overlapping official evaluation items were removed, and the official evaluation splits were never trained on.
- **Comparability:** its scores are task-tuned. They are not comparable at face value with pretraining-only models, which is why V2 base is reported alongside.
- **Selection:** our own pre-registered rule did not select it, because it worsened held-out Wikipedia bits/byte by 0.028 against a 0.01 guard. We adopted it after its official scores were observed.
- **Perplexity:** WikiText-103 perplexity is 6.8% worse than V2 base.
- **Alternative:** a 50/50 weight interpolation with V2 base passed that rule (mean 49.60, perplexity 24.09) and is preserved.

Details: [v2/phases/task_tune/REPORT.md](v2/phases/task_tune/REPORT.md) and the [adoption record](v2/phases/task_tune/ADOPTION.json). V2 base was chosen on development data before any official V2 score was computed; against V1 its perplexity gain is clear, and its mean-accuracy gain is within per-task standard errors (0.45–1.4 points).

- **[BEST_CHECKPOINT_TRAINING.md](BEST_CHECKPOINT_TRAINING.md):** how the model was built: architecture, tokenizer, corpus, schedule, compute, provenance and results.
- **[CURRENT_PIPELINE.md](CURRENT_PIPELINE.md):** code map, verification and reproduction commands.
- **[FAILED_APPROACHES.md](FAILED_APPROACHES.md):** what did not work (V1 post-training, V2 decay-mix A/B, the guard-failing task tuning) and why it is not repeated.
- **[TRAINING_PHASES.md](TRAINING_PHASES.md):** dated journal of every training phase; per-phase records in [v2/phases/](v2/phases/).

## Quick start

```bash
PYTHONPATH=src python scripts/demo.py --prompt "The water cycle begins when"   # text completion, CPU
PYTHONPATH=src python scripts/verify_v2.py --load                                # read-only integrity check
python -m pytest                                                                 # CPU tests
```

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
repo = "davemaxuellkr/Global-Innovation-Build-Challenge-V2"   # V2.1; V2 base: "davemaxuellkr/scglm-v2-46m"
tok = AutoTokenizer.from_pretrained(repo)                       # or a local copy, e.g. "checkpoints/v2_1"
model = AutoModelForCausalLM.from_pretrained(repo)
```

Both models continue text; neither is instruction-tuned or a chat model. The submitted model was additionally trained to score multiple-choice answers.

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
| `checkpoints/` | `v2_1` (submitted) and `v2_best` (V2 base): configs, tokenizer, provenance and hash inventories (weights are distributed separately) |
| `release/hf_v2_1/`, `release/hf_v2/` | Hugging Face releases of V2.1 (submitted) and V2 base: model card, configs, tokenizer, `eval_results.json` (weights not in git) |
| `runs/`, `artifacts/phase_evaluation_v1/` | V1 baseline run records and V1's official evaluation summary (the reference for the V1 row) |
| `vendor/` | Pinned lm-evaluation-harness checkout; fastText for data preparation |

## Competition notes

Track 01 (TECH): trained from scratch with no pretrained weights, no fine-tuning of an existing model and no distillation. The only fine-tuning was of our own scratch-trained model, which the user confirmed is allowed. The parameter count, 46,346,752 with tied embeddings, is audited on every load. Hardware: one NVIDIA H100 NVL. V1 used 26.5 GPU-hours for 13B tokens; V2 used 31.9 GPU-hours for 45B tokens. Task tuning took 20.5 minutes for the submitted arm, and under 1.5 GPU-hours for the whole phase. Dataset licenses are listed in [v2/submission/BUILT_WITH.md](v2/submission/BUILT_WITH.md).

## License

Code and documentation are released under the [MIT License](LICENSE). Dataset-derived records in this repository keep their original terms, for example the web-text excerpts in `v2/data/rich_v1/targeted/selected_examples.json`; see [v2/submission/BUILT_WITH.md](v2/submission/BUILT_WITH.md) for each source's license. The pinned lm-evaluation-harness is not included and keeps its own license. The model weights are distributed separately under the license stated in their model cards ([V2.1](release/hf_v2_1/README.md), [V2](release/hf_v2/README.md)).

AI assistance: Claude (Anthropic) assisted V2's research, implementation, tests and documentation; Codex/ChatGPT assisted V1. All code is in this repository.
