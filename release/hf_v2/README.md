---
language:
- en
license: mit
library_name: transformers
pipeline_tag: text-generation
tags:
- llama
- causal-lm
- from-scratch
- small-language-model
- pretraining
datasets:
- HuggingFaceFW/fineweb-edu
- mlfoundations/dclm-baseline-1.0
- wikimedia/wikipedia
- HuggingFaceTB/stackexchange_2025_md
- manu/project_gutenberg
model-index:
- name: SCG-LM-V2-46M
  results:
  - task:
      type: text-generation
      name: Text Generation
    dataset:
      name: HellaSwag
      type: Rowan/hellaswag
      split: validation
    metrics:
    - type: accuracy
      name: accuracy (0-shot)
      value: 28.85
    - type: accuracy
      name: normalized accuracy (0-shot)
      value: 30.95
  - task:
      type: text-generation
      name: Text Generation
    dataset:
      name: ARC-Easy
      type: allenai/ai2_arc
      config: ARC-Easy
      split: test
    metrics:
    - type: accuracy
      name: accuracy (0-shot)
      value: 47.39
    - type: accuracy
      name: normalized accuracy (0-shot)
      value: 42.09
  - task:
      type: text-generation
      name: Text Generation
    dataset:
      name: PIQA
      type: ybisk/piqa
      split: validation
    metrics:
    - type: accuracy
      name: accuracy (0-shot)
      value: 60.77
    - type: accuracy
      name: normalized accuracy (0-shot)
      value: 60.66
  - task:
      type: text-generation
      name: Text Generation
    dataset:
      name: WinoGrande
      type: allenai/winogrande
      config: winogrande_xl
      split: validation
    metrics:
    - type: accuracy
      name: accuracy (0-shot)
      value: 51.46
  - task:
      type: text-generation
      name: Text Generation
    dataset:
      name: WikiText-103
      type: Salesforce/wikitext
      config: wikitext-103-raw-v1
      split: validation
    metrics:
    - type: perplexity
      name: perplexity (context 1024, stride 512)
      value: 23.85
---

# SCG-LM V2 (46M)

SCG-LM V2 is a **46,346,752-parameter** Llama-style language model trained **from random initialization** on **45.0 billion tokens** of human-written English. Training took 31.9 hours on a single H100. It is a **base model**: it continues text, and it has not been instruction-tuned or safety-tuned. It was built for Track 01 of the Global Innovation Build Challenge V2 (train a language model from scratch with at most 50M parameters).

> **Successor:** [SCG-LM V2.1](https://huggingface.co/davemaxuellkr/Global-Innovation-Build-Challenge-V2) is this model with one short fine-tuning stage on the four benchmarks' *training* splits. It scores higher on them (four-task mean 52.76 vs 47.12), but its scores are task-tuned and its WikiText-103 perplexity is worse (25.46 vs 23.85). This repository is the pretraining-only model.

| | |
| --- | --- |
| **Architecture** | `LlamaForCausalLM`: 12 layers, width 512, 8 heads (full multi-head attention), SwiGLU 1,376, RMSNorm, RoPE θ = 10,000, tied input/output embeddings, no biases |
| **Parameters** | 46,346,752 (embeddings included; tied embeddings counted once) |
| **Context / vocabulary** | 1,024 tokens / 16,384 (byte-level BPE trained from scratch) |
| **Training tokens** | 44,999,966,720 next-token targets (about 970 per parameter) |
| **Initialization** | Random (seed 20260923); no pretrained weights, no distillation |
| **Language** | English |
| **Weights** | FP32 `model.safetensors`, SHA256 `9aac2111b9ca6b06f4a12278f5213c17621ded643b7b54137987ce25028011bd` |
| **Code and records** | https://github.com/davemaxuell/Global-Innovation-Build-Challenge-V2 |

## How to use

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

repo = "davemaxuellkr/scglm-v2-46m"
tok = AutoTokenizer.from_pretrained(repo)
model = AutoModelForCausalLM.from_pretrained(repo).eval()

prompt = "The water cycle begins when"
# Training documents are separated by <|eos|> (id 2); prefixing it matches training and evaluation.
ids = torch.tensor([[tok.eos_token_id] + tok.encode(prompt, add_special_tokens=False)])
out = model.generate(ids, max_new_tokens=60, do_sample=False, repetition_penalty=1.1,
                     pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
print(prompt + tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True))
```

Tested with Transformers 4.46.3 and 5.5.4 (PyTorch 2.10). The special tokens are `<|pad|>` = 0, `<|bos|>` = 1, `<|eos|>` = 2 and `<|unk|>` = 3. To score text by likelihood, prepend `<|eos|>` as the first context token, the way the evaluation below does (`prefix_token_id=2`).

## Evaluation

Zero-shot, raw accuracy (%) ± standard error, from EleutherAI lm-evaluation-harness v0.4.12 at a pinned commit. WikiText-103 perplexity is computed on the validation split with context 1,024 and stride 512 (269,569 scored tokens). The comparison row is our own earlier model (V1: same architecture, 13.0B training tokens) under the identical protocol.

| Model | HellaSwag | ARC-Easy | PIQA | WinoGrande | 4-task mean | WikiText-103 ppl ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Random chance | 25.00 | ≈25 | 50.00 | 50.00 | ≈37.5 | — |
| SCG-LM V1 (13B tokens) | 28.48 | 46.59 | **62.08** | 50.67 | 46.96 | 25.52 |
| **SCG-LM V2 (45B tokens)** | **28.85** ± 0.45 | **47.39** ± 1.02 | 60.77 ± 1.14 | **51.46** ± 1.40 | **47.12** | **23.85** |

Normalized accuracy (V2): HellaSwag 30.95, ARC-Easy 42.09, PIQA 60.66.

**How to read these numbers:**
- **Perplexity:** the WikiText-103 improvement over V1 is clear, about 6.5%.
- **Accuracy:** the 0.17-point gain in the four-task mean is within per-task standard errors, and PIQA is 1.3 points lower than V1.
- **Near-chance tasks:** at this scale HellaSwag and WinoGrande sit close to chance.

**Selection and disclosure:**
- The checkpoint was chosen **on development data only** (held-out text and held-out items from benchmark *training* splits) and frozen before any official V2 score was computed.
- A pre-registered rule then compared it with V1.
- V1's official scores had been observed earlier, and a second V2 variant (a different data mix in the final learning-rate decay) was evaluated and not selected (four-task mean 47.08, perplexity 23.56).

## Training data

Human-written English only; no model-generated corpora.

| Source | Unique tokens | Share of training | Passes |
| --- | ---: | ---: | ---: |
| [DCLM-baseline](https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0) (web) | 8.29B | 35% | 1.90 |
| [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) (educational web) | 6.02B | 30% | 2.24 |
| Benchmark-targeted slice (top 20% of the FineWeb-Edu + DCLM pool, see below) | 3.52B | 15% | 1.92 |
| [English Wikipedia](https://huggingface.co/datasets/wikimedia/wikipedia) (20231101.en) | 2.02B | 10% | 2.23 |
| [StackExchange](https://huggingface.co/datasets/HuggingFaceTB/stackexchange_2025_md) (54 English, mostly non-code sites; answered questions with score ≥ 1) | 1.61B | 5% | 1.39 |
| [Project Gutenberg](https://huggingface.co/datasets/manu/project_gutenberg) (English; license boilerplate removed; 20k-character chunks) | 0.99B | 5% | 2.27 |
| **Total** | **22.45B** | | |

**Contamination controls.**
- **N-gram exclusion:** any document sharing a 13-word n-gram with the train or evaluation splits of HellaSwag, ARC-Easy, PIQA or WinoGrande, or with the WikiText-103 validation and test splits, was removed.
- **Domain blocks:** all of `wikihow.com` and `instructables.com` (the sources of HellaSwag and PIQA) were blocked, removing 12,708 documents.
- **Deduplication:** exact, URL and MinHash near-duplicate removal across all sources.
- **Limits:** this is not a guarantee against paraphrased overlap.

**Benchmark-targeted slice.** A fastText classifier was trained to recognize text resembling benchmark **training** examples: 80% of the train splits of ARC-Easy/Challenge, PIQA, HellaSwag, WinoGrande, OpenBookQA, SciQ and CommonsenseQA, against length-matched web passages. The top 20% of pool documents by score form this slice. Benchmark items were never used as training text, and official evaluation splits were never used for selection or tuning.

**Tokenizer.** A 16,384-token byte-level BPE trained from scratch on 100M characters of FineWeb and English Wikipedia (training partitions only). A pre-registered comparison against new tokenizers trained on the V2 mixture kept this one.

## Training procedure

| Setting | Value |
| --- | --- |
| Objective | Causal next-token cross-entropy on packed 1,024-token sequences (documents separated by `<|eos|>`) |
| Batch | 64 × 1,024 = 65,536 tokens per update; 686,645 updates |
| Optimizer | AdamW, β = (0.9, 0.95), ε = 1e-8, weight decay 0.1 on matrices, gradient clipping 1.0 |
| Learning rate | Warmup-stable-decay: linear warmup to 6e-4 over 500 updates, constant to update 617,981 (40.5B tokens), linear decay to 0 over the final 68,664 updates |
| Precision | BF16 autocast with FP32 master weights and optimizer state; `torch.compile` |
| Throughput | About 395,000 tokens/s on one NVIDIA H100 NVL (peak memory 28.6 GB) |
| Time | 31.9 hours of training. One crash at update 10,000 was resumed from update 8,000, so 2,000 updates were computed twice |

Held-out loss plateaued near 3.17 nats during the constant phase, then fell to 2.97 during the final decay.

## Limitations and risks

- **Small model, frequent errors.** It produces fluent but often false statements. For example, it continued "The water cycle begins when" with "…the water in the ocean is depleted". Do not use it for factual answers or advice.
- **Not an assistant.** There was no instruction tuning, preference tuning or safety tuning, so it does not follow instructions and has no content filtering.
- **Data biases.** Web, encyclopedic, Q&A and older public-domain book text carry the social biases and dated viewpoints of their sources.
- **English only**, with a 1,024-token context.
- **Benchmark scope.** Scores near chance on HellaSwag and WinoGrande mean these benchmarks say little about reasoning at this scale.

**Intended uses:** research and teaching on small-scale pretraining, data mixtures and training efficiency; a baseline for other from-scratch models; text-completion experiments.

## Environmental impact

The selected run used 31.9 H100-hours. All V2 GPU work, including a 1B-token pilot and the unselected decay variant, came to 35.8 H100-hours. Power was not measured. At the H100 NVL's 400 W maximum board power, that is at most about 12.8 kWh for the selected run and 14.3 kWh in total. Carbon emissions depend on the local grid and were not measured.

## Licenses and attribution

The model weights are released under the MIT license declared above. The training data keeps its own terms:
- FineWeb-Edu: ODC-By 1.0.
- DCLM-baseline: CC-BY-4.0.
- Wikipedia: CC-BY-SA 3.0 / GFDL.
- StackExchange content: CC-BY-SA 4.0.
- Project Gutenberg: public domain.
- Common Crawl terms also apply to the web sources.

We thank the maintainers of these datasets and of lm-evaluation-harness.

## Files and reproducibility

- `model.safetensors`, `config.json`, `generation_config.json`, `tokenizer.json`: identical to the selected training export.
- `tokenizer_config.json`: identical except `tokenizer_class`, which is set to `PreTrainedTokenizerFast` so that Transformers 4.x can load the tokenizer. Token IDs are unchanged.
- `training_provenance.json`: the trainer's record of steps, token counts and data identity.
- `eval_results.json`: the official metrics above, with the SHA256 of the original evaluation record.
- **Full records** (code, configurations, per-phase registrations, the frozen development-only selection, and the evaluation outputs): https://github.com/davemaxuell/Global-Innovation-Build-Challenge-V2.

## Citation

```bibtex
@misc{scglm_v2_2026,
  title  = {SCG-LM V2: a 46M-parameter language model trained from scratch},
  author = {davemaxuell},
  year   = {2026},
  url    = {https://huggingface.co/davemaxuellkr/scglm-v2-46m}
}
```

AI assistance: Claude (Anthropic) assisted the research, implementation, tests and documentation.
