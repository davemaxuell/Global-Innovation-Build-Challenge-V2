# Built With: draft

**Languages and frameworks:** Python 3.10 · PyTorch 2.10 (CUDA 12.8, `torch.compile`, BF16 autocast, fused AdamW) · Hugging Face Transformers 5.5.4 (`LlamaForCausalLM`, random initialization only) · Tokenizers 0.22.2 · Datasets 4.8.4 · Safetensors 0.8.0 · NumPy · PyArrow · zstandard · LMDB + xxhash (deduplication) · fastText (data-selection classifier only) · matplotlib (figures)

**Evaluation:** EleutherAI lm-evaluation-harness v0.4.12 (pinned commit `6d642546`), zero-shot, pinned dataset revisions · WikiText-103 perplexity with the project's own scorer (context 1,024, stride 512)

**Hardware:** one NVIDIA H100 NVL (96 GB). V1: 26.5 GPU-hours for 13.0B tokens. V2: 31.9 GPU-hours for 45.0B tokens (selected run), plus 0.7 h for the 1B pilot and 3.2 h for the unselected decay A/B, 35.8 GPU-hours in total. Official evaluation took about 6 minutes per model on the same GPU. CPU data preparation on 64 cores.

**Pretraining datasets (public, human-written):**

| Dataset | Revision | License / notice |
| --- | --- | --- |
| HuggingFaceFW/fineweb-edu (sample-100BT) | `87f09149` | ODC-By 1.0; Common Crawl terms apply |
| mlfoundations/dclm-baseline-1.0 | `a3b142c1` | CC-BY-4.0; underlying web content rights apply |
| wikimedia/wikipedia (20231101.en) | `b04c8d1c` | CC-BY-SA 3.0 / GFDL |
| HuggingFaceTB/stackexchange_2025_md (54 sites) | `fe382c48` | Stack Exchange content CC-BY-SA 4.0, attribution required |
| manu/project_gutenberg (English) | `164853d2` | Public domain; Gutenberg boilerplate removed |
| HuggingFaceFW/fineweb (sample-10BT; V1 only) | `9bb295dd` | ODC-By 1.0 |

**Benchmark train splits used only for data selection and development evaluation, never as training text:** ARC-Easy/Challenge, PIQA, HellaSwag, WinoGrande, OpenBookQA, SciQ, CommonsenseQA (train splits; 80% for the classifier, 20% held out). Official evaluation splits were never used for training, selection or tuning.

**Not used:** no pretrained weights, no distillation from larger models, no hosted-API substitutes, no model-generated pretraining corpora.

**AI assistance (disclosure):** Claude Code (Anthropic) assisted the V2 research, implementation, tests and documentation. Codex/ChatGPT assisted V1. All code is in the repository and was reviewed.
