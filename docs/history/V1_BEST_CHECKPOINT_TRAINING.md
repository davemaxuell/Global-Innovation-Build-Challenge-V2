# Selected checkpoint: completed training record

**Recorded:** September 29, 2026. **Checkpoint:** [checkpoints/best_sft](../../checkpoints/best_sft/).

Active construction code and verification commands: [CURRENT_PIPELINE.md](V1_CURRENT_PIPELINE.md). This training record describes the original completed model; the preserved export identity below is unchanged.

This document records the completed training lineage of our selected full-SFT instruction model: a 46,346,752-parameter language model initialized from scratch, pretrained on 13,000,048,640 next-token targets, then instruction-tuned with raw-text replay. It covers the stages that produced this checkpoint.

The checkpoint was selected using the registered development and confirmation procedure for instruction performance. Its competition-benchmark measurements are reported separately below; “selected” refers to that documented selection procedure.

## 1. Checkpoint identity

| Item | Recorded value |
| --- | --- |
| Working export | `checkpoints/best_sft/` |
| Original run | `artifacts/posttraining_sota_v1/runs/sft_full_5_seed20260928/` |
| Original selected export | `artifacts/posttraining_sota_v1/runs/sft_full_5_seed20260928/endpoints/step_0000105/export/` |
| SFT recipe / seed | Recipe 5 / `20260928` |
| Selected SFT update | **105** |
| Weight format | FP32 `model.safetensors`, with tied input/output embeddings |
| Weight SHA256 | `defd0d91948ba8501a33b0c4996bf3bf37ebdf6e90338a05bab3a1773a2d7f72` |
| Parent base weight SHA256 | `ffa75b07ec377167ab2e4d4263f2fa8374549f8592f9ce2e8166875cab1f38e7` |
| Recorded SFT contract SHA256 | `475d1b108966dae1b71d611370ba75b415bff46189412aa462d8a0e71cfee96e` |

The working export is an independent, read-only copy. Its file checksums, offline model/tokenizer loading, finite forward pass, parameter count and tied embeddings were verified. [Verification inventory](../../checkpoints/SELECTION_2026-09-29.json).

## 2. Model architecture

The implementation uses Hugging Face `LlamaForCausalLM` with a custom small configuration and newly initialized weights. No external pretrained model weights enter this lineage.

| Component | Executed configuration |
| --- | --- |
| Unique parameters | **46,346,752** |
| Decoder blocks | 12 |
| Hidden width | 512 |
| Feed-forward width | 1,376; gated SiLU/SwiGLU |
| Attention | Causal multi-head attention; 8 query heads and 8 key/value heads; head dimension 64 |
| Context length | 1,024 tokens |
| Vocabulary | 16,384 tokens |
| Normalization | RMSNorm, epsilon `1e-5` |
| Position representation | Rotary positional embeddings, theta `10000` |
| Embeddings | Input embedding and final LM-head weights tied |
| Attention/MLP bias | Disabled |
| Dropout | 0 |
| Attention implementation | PyTorch scaled-dot-product attention |
| Initialization | Random initialization, standard deviation setting `0.02`, seed `20260923` |

The architecture and vocabulary stayed fixed through pretraining and SFT. All unique model parameters participated in the selected SFT run. [Model configuration](../../checkpoints/best_sft/config.json).

## 3. Tokenizer and corpus preparation

### Tokenizer

A byte-level BPE tokenizer was trained from scratch on training-partition text from the initial FineWeb and English Wikipedia corpora. It observed **50,005,546 FineWeb characters** and **50,007,101 Wikipedia characters**. Held-out partitions were excluded from tokenizer training.

The vocabulary contains 16,384 entries, including `<|pad|>` = 0, `<|bos|>` = 1, `<|eos|>` = 2 and `<|unk|>` = 3. Byte-level pretokenization and decoding are used, with no tokenizer normalizer. The learned BPE vocabulary and merges were retained through all phases.

The original tokenizer artifact has SHA256 `1d714be4561ca083ef3cb668be7cffa38caa158c0315c22907647cd4f0f9013b`; the packaged tokenizer has SHA256 `f8de5462e2967f502b8a25f2a90a87fee12ecad1f50d980a54552805ed0ea196`. Their BPE models are identical. The serialized export adds a pass-through template postprocessor, accounting for the file-identity difference. [Tokenizer training provenance](../../data/processed/main/tokenizer/training_provenance.json).

### Pretraining sources

| Source | Pinned repository revision | Use |
| --- | --- | --- |
| `HuggingFaceFW/fineweb`, `sample-10BT` | `9bb295ddab0e05d785b879661af7260fed5140fc` | Initial scratch pretraining |
| `wikimedia/wikipedia`, `20231101.en` | `b04c8d1ceb2f5cd4588862100d08de323dccfbaa` | Initial and continued pretraining |
| `HuggingFaceFW/fineweb-edu`, `sample/100BT/` | `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9` | Continued pretraining |
| `mlfoundations/dclm-baseline-1.0` | `a3b142c183aebe5af344955ae20836eb34dcf69b` | Continued pretraining |

Preparation applied document-length/type checks, benchmark-text screening, held-out partitioning and deduplication. The initial corpus used normalized content hashes for partitioning and global exact deduplication, with seeded file order and a 10,000-document shuffle buffer. The extension added canonical-URL checks and approximate near-duplicate filtering against its own and previously collected documents: five-word shingles, 32-component MinHash, eight bands of four, and rejection at 28 or more matching components.

Benchmark screening included exact 13-word matching. These specific filters define the scope of the overlap checks; approximate filtering does not establish complete semantic decontamination. Source revisions, document indexes, data checksums and source notices are retained in the manifests.

Documents were tokenized into packed streams with EOS separators and stored as little-endian `uint16` token IDs. Training counts refer to loss-bearing next-token targets, including separators. Padding is absent from packed pretraining. Stored corpus capacity and actual training exposure are tracked separately.

| Manifest | SHA256 |
| --- | --- |
| [Initial corpus](../../data/processed/main/manifest.json) | `180e9ff1acb02c12aa677dee84f92f443aa6fb0ed8d304875677ab9266d5f9e1` |
| [Continuation corpus](../../data/processed/extension_12b/manifest.json) | `70242a0a248787d31d56c73af3733ccafadaa9816b09ca0508ec487f1fd37c6d` |

## 4. Completed pretraining lineage

The model learned causal next-token prediction over three successive training phases. Each continuation inherited our own preceding model and Adam state. The final continuation also inherited source-stream cursors, consuming the next portion of the prepared extension corpus.

| Phase | Optimizer updates in phase | Additional training targets | Cumulative pretraining targets | Requested source mixture |
| --- | ---: | ---: | ---: | --- |
| Scratch pretraining | 15,258 | 999,948,288 | 999,948,288 | 80% FineWeb / 20% Wikipedia |
| Continuation to approximately 12B | 167,848 | 11,000,086,528 | 12,000,034,816 | 60% FineWeb-Edu / 30% DCLM / 10% Wikipedia |
| Continuation to approximately 13B | 15,259 | 1,000,013,824 | **13,000,048,640** | 60% FineWeb-Edu / 30% DCLM / 10% Wikipedia |

Every update contained **64 × 1,024 = 65,536 target tokens**. Exact totals follow complete updates, which explains the small differences from the rounded 1B/12B/13B names. Source selection operated at sequence granularity, so measured source shares vary slightly from requested proportions.

### Shared optimization settings

| Setting | Executed value |
| --- | --- |
| Objective | Mean causal next-token cross-entropy |
| Optimizer | AdamW; betas `(0.9, 0.95)`, epsilon `1e-8` |
| Effective batch | 64 sequences of length 1,024 |
| Microbatch / accumulation | 8 sequences / 8 microbatches per update |
| Weight decay | `0.1` on matrices and higher-dimensional tensors; zero on one-dimensional tensors |
| Gradient clipping | Global norm `1.0` |
| Precision | BF16 autocast; FP32 model weights, optimizer state and loss computation |
| Hardware | One NVIDIA H100 NVL |
| Seed | `20260923` |
| Execution | Deterministic algorithms enabled; compilation disabled |

### Learning-rate schedules

| Phase | Warmup or rewarming | Subsequent schedule |
| --- | --- | --- |
| Scratch | Linear warmup to `6e-4` over the first 305 updates, approximately 2% of the run | Cosine decay to `6e-5` |
| 12B continuation | Rewarm from `6e-5` to `3e-4` over 1,000 updates | Cosine decay to `3e-5` |
| 13B continuation | Rewarm from `3e-5` to `6e-5` over 500 updates | Cosine decay to `3e-6` |

All three phases completed their recorded target budgets. The resulting base export is [runs/continuation_13b/export](../../runs/continuation_13b/export/).

Exact cumulative pretraining exposure by source:

| Source | Next-token targets |
| --- | ---: |
| FineWeb | 799,876,096 |
| FineWeb-Edu | 7,198,570,496 |
| DCLM | 3,601,405,952 |
| Wikipedia, across both corpus preparations | 1,400,196,096 |
| **Total** | **13,000,048,640** |

Source records: [scratch run](../../runs/baseline_1b/run.json), [12B continuation](../../runs/continuation_12b/run.json), [13B continuation](../../runs/continuation_13b/run.json), and each run's completed `status.json`.

## 5. Full-parameter supervised fine-tuning with replay

### Starting point and data

SFT loaded the completed 13B base and initialized a new AdamW optimizer for instruction adaptation. It updated the transformer blocks, normalization parameters and shared embedding/output matrix together.

The eligible balanced SFT pool contained **103,429 examples**. Separate development and confirmation files each contained **7,862 examples**. Human demonstrations came from Dolly and OpenAssistant; knowledge examples came from CommonSenseQA and QASC; procedural and grounded examples were generated by the project's task programs.

| Source | Pinned revision | Response targets used through update 105 |
| --- | --- | ---: |
| `databricks/databricks-dolly-15k` | `bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a` | 344,857 |
| `OpenAssistant/oasst1` | `fdf72ae0827c1cda404aff25b6603abec9e3399b` | 51,422 |
| `tau/commonsense_qa` | `94630fe30dad47192a8546eb75f094926d47e155` | 28,651 |
| `allenai/qasc` | `a34ba204eb9a33b919c10cc08f4f1c8dae5ec070` | 273,305 |
| Project-generated grounded, arithmetic, logic, sorting, JSON and coreference tasks | Registered preparation code and manifest | 164,425 |
| **Total** | | **862,660** |

QASC supervision included the dataset's supplied supporting facts. A larger-model teacher was not used. Data preparation retained source/group separation for held-out evaluation and recorded overlap checks in the [SFT data audit](../../data/posttraining/sota_v1/data_audit.json).

Sampling was balanced by **supervised response tokens**:

| Bucket | Requested share | Actual response targets at update 105 |
| --- | ---: | ---: |
| General instructions | 30% | 258,789 |
| Knowledge | 35% | 301,956 |
| Grounded tasks | 20% | 172,518 |
| Procedural tasks | 15% | 129,397 |
| **Total** | **100%** | **862,660** |

Formatting-only supervision was capped at 5%. Bucket and dataset source are different classifications: human demonstrations can also belong to the grounded bucket.

The deterministic batch plan exposed **55,901 examples**, representing **55,836 distinct examples**, through update 105. Each example appeared at most twice. Chat-formatted prompts accounted for 628,229 response targets, and raw question/answer prompts for 234,431.

### Supervision and loss

Each example used teacher forcing: the prompt and preceding gold answer tokens conditioned prediction of the next answer token. Loss applied only to the final response and its real EOS token. Prompt and padding labels were `-100`; earlier conversation turns supplied context. The causal shift was applied once.

The maximum combined context was 1,024 tokens, and the maximum supervised response was 128 tokens including EOS. Overlength examples were excluded. The project's chat template and raw question/answer format used the existing vocabulary.

Raw replay drew protected training documents from FineWeb-Edu, DCLM and Wikipedia, targeting a 60/30/10 source mixture. The objective was:

```text
loss = 0.70 × mean supervised-response cross-entropy
     + 0.30 × mean raw-replay next-token cross-entropy
```

Each mean used its own loss-bearing token denominator. The coefficients are loss weights. Complete-document replay batching determines actual token exposure.

### Selected optimization recipe

| Setting | Executed value |
| --- | --- |
| Trainable parameters | All 46,346,752 unique parameters |
| Optimizer | Newly initialized AdamW; betas `(0.9, 0.95)`, default epsilon `1e-8` |
| Peak learning rate | `1e-5` |
| Schedule | Warmup over 3% of the planned response-target budget, then cosine decay toward `1e-6` at the end of that budget |
| Effective update size | Approximately 8,192 response targets; complete examples retained |
| Microbatch | 8 examples with gradient accumulation |
| Weight decay | `0.1` on tensors with at least two dimensions; zero on remaining tensors |
| Gradient clipping | Global norm `1.0` |
| Precision / hardware | BF16 autocast with FP32 weights, optimizer state and loss; one H100 |
| Seed | `20260928` |
| Example exposure limit | Two passes per eligible example |
| Saved evaluation endpoints | Updates 53, 105 and 209 |

The balanced two-pass plan contained **1,715,410 response targets over 209 updates**. That run completed. Development selection chose its saved **update-105** model. The checkpoint therefore contains the first 105 updates of the original 209-update schedule; reproducing it requires retaining the full planned learning-rate denominator.

Actual SFT exposure in the selected checkpoint:

| Quantity | Count |
| --- | ---: |
| Optimizer updates | 105 |
| Supervised response targets, including EOS | 862,660 |
| Raw replay targets | 419,498 |
| Processed response-example input tokens, including prompts | 4,670,374 |
| Processed replay input tokens | 420,106 |

Prompt/input counts and supervised-target counts describe different quantities. The checkpoint's pretraining ancestry remains 13,000,048,640 targets; the SFT and replay exposures above are recorded separately.

The [selected run contract](../../artifacts/posttraining_sota_v1/runs/sft_full_5_seed20260928/run.json) records the batch-plan SHA256 `57949da72d8fbc05ff893e6aded334144142d429147d4bc822bf1b9b935a88c7`. The [SFT manifest](../../data/posttraining/sota_v1/manifest.json) has SHA256 `7bc4e6c72b6dc76819f1aa1e41611c9eeec94cf29cec76bd0a30f64ccbd666ee`.

## 6. Checkpoint selection and measured performance

Development evaluation selected the recipe and checkpoint position using replicated task performance and raw-language retention checks. The recipe was assessed with seeds `20260928`, `20260929` and `20260930`; the deployment seed was fixed in advance at `20260928`. A single frozen confirmation comparison then accepted the selected export. The [frozen choice](../../artifacts/posttraining_sota_v1/selection/frozen.json) records `official_scores_used: false`.

The registered primary confirmation score was the equally weighted mean of science, commonsense and generated grounded-question accuracy:

| Confirmation measure | Selected checkpoint |
| --- | ---: |
| Science accuracy, 614 questions | 15.47% |
| Commonsense accuracy, 962 questions | 23.18% |
| Grounded accuracy, 600 questions | 99.83% |
| **Three-domain mean** | **46.16%** |

The corresponding parent mean was 36.04%. The gain was **10.12 percentage points**, with a source/fact-group bootstrap 95% interval of **8.23–12.05 points**. This score describes the registered task panel, which includes generated grounded examples. [Confirmation measurements](../../artifacts/posttraining_sota_v1/evaluation/confirmation/defd0d91948ba850_32ac7f95/result.json) and [selection statistics](../../artifacts/posttraining_sota_v1/selection/selection.json).

Competition-oriented benchmark reporting followed the frozen checkpoint choice:

| Benchmark | Split / sample count | Selected checkpoint |
| --- | --- | ---: |
| HellaSwag, raw accuracy | Validation / 10,042 | 28.25% |
| ARC-Easy, raw accuracy | Test / 2,376 | 46.46% |
| PIQA, raw accuracy | Validation / 1,838 | 61.32% |
| WinoGrande, raw accuracy | Validation / 1,267 | 51.30% |
| WikiText-103, perplexity | Validation / 269,569 scored tokens | 25.9415 |

These are local measurements under the pinned project protocol. Multiple-choice tasks used zero-shot evaluation with `lm-evaluation-harness` v0.4.12; WikiText-103 used context 1,024 and stride 512. The protocol's exact splits, shots and perplexity convention were frozen by the project. Previously observed base evaluations were disclosed in the registered configuration, and these report scores did not change the selected checkpoint. [Exact benchmark output](../../artifacts/posttraining_sota_v1/official/12_defd0d91948ba850/attempt_001/summary.json) and [evaluation protocol](../../configs/evaluation.json).

## 7. Loading and reproducibility records

The export includes the model weights, configuration, tokenizer, chat template and training provenance. From the project root, load it locally with the recorded environment:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

checkpoint = "checkpoints/best_sft"
tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    checkpoint, local_files_only=True
).eval()
```

Recorded software versions are PyTorch `2.10.0` (`2.10.0+cu128` runtime, CUDA `12.8`), Transformers `5.5.4`, Tokenizers `0.22.2`, NumPy `2.2.6` and Safetensors `0.8.0`. The pretraining data environment also recorded Datasets `4.8.4`.

For reconstruction, use the recorded corpus manifests, original run configurations, source snapshots and deterministic SFT plan together. The model export contains inference weights; optimizer/RNG and resume states belong to the original training-run artifacts. The update-105 identity above is the endpoint to reproduce.

| Record | Location |
| --- | --- |
| Architecture | [Model configuration](../../checkpoints/best_sft/config.json) |
| Initial pretraining recipe and source hashes | [Scratch run record](../../runs/baseline_1b/run.json) |
| First continuation recipe and ancestry | [12B run record](../../runs/continuation_12b/run.json) |
| Final continuation recipe and ancestry | [13B run record](../../runs/continuation_13b/run.json) |
| SFT data identities and audit | [SFT manifest](../../data/posttraining/sota_v1/manifest.json) |
| Human-demonstration source identities | [Demonstration manifest](../../data/posttraining/pipeline_v2/manifest.json) |
| Exact SFT recipe, source hashes and deterministic plan | [Selected SFT run record](../../artifacts/posttraining_sota_v1/runs/sft_full_5_seed20260928/run.json) |
| Checkpoint token counts and parent identity | [Export provenance](../../checkpoints/best_sft/training_provenance.json) |
| Selected endpoint identity | [Endpoint record](../../checkpoints/best_sft/endpoint.json) |
| Working-copy integrity and load verification | [Handoff inventory](../../checkpoints/SELECTION_2026-09-29.json) |

AI assistance: Codex/ChatGPT assisted research, implementation, tests and documentation, as recorded in the original run provenance.
