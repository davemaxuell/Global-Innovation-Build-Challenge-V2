# Pretrained 12B model — September 26, 2026 test record

**Record type:** dated diagnostic snapshot; record future tests in a new file.  
**Reviewed:** September 26, 2026.  
**Model:** SCG-LM, 46,346,752 unique parameters, 12,000,034,816 cumulative pretraining targets.  
**Endpoint:** `runs/continuation_12b/export`.  
**Conclusion:** the saved model and packaged inference command work. The model has some language-modeling and multiple-choice capability, but these tests expose weak factual reliability, arithmetic, instruction following, and substantial repetition during greedy generation. A completed training run is not evidence of a reliable assistant.

This review verified the existing full competition benchmark results, ran twelve new completion diagnostics on the assigned H100 GPU 0, and exercised the packaged demo. It did not train, modify weights, reopen checkpoint selection, or rerun the official benchmarks. The new diagnostics are recorded separately from the frozen competition pipeline.

## 1. Existing full benchmark results

These scores were produced on September 24 by the completed pipeline and checked against their recorded artifacts on September 26. They are our results under the declared evaluation protocol, not scores independently issued by the organizers.

| Benchmark | Split | Examples | Accuracy | Normalized accuracy |
|---|---|---:|---:|---:|
| HellaSwag | Validation | 10,042 | 28.44% | 30.72% |
| ARC-Easy | Test | 2,376 | 46.09% | 42.05% |
| PIQA | Validation | 1,838 | 61.59% | 60.50% |
| WinoGrande | Validation | 1,267 | 50.28% | Not defined |

**WikiText-103 token perplexity: 26.5214**, with token NLL 3.27795 over 269,569 scored tokens from 60 reconstructed validation articles. Context is 1,024 tokens and stride is 512.

The multiple-choice evaluations use zero-shot FP32 likelihood scoring without a chat template. Here, normalized accuracy uses the pinned harness's character-length normalization; it is not token normalization. Perplexity depends on our tokenizer, article reconstruction, and scoring windows, so direct comparison with differently scored published perplexities is inappropriate.

PIQA is above its 50% uniform-random baseline. WinoGrande is near its 50% baseline. These results do not establish broad reasoning competence or a competition ranking. No 1B-versus-12B official comparison was performed in this review, so the scores do not quantify the benefit attributable to the extra training tokens.

Evidence: [full benchmark summary](../../artifacts/pipeline_v1/official/endpoint_0-1790219463589560779/summary.json), [frozen protocol](../../EVALUATION_PROTOCOL.md), [integrity audit](../../artifacts/model_tests/20260926_integrity.json).

## 2. Fresh completion diagnostics

All twelve prompts were written before this diagnostic run and retained in their original order. Generation used FP32, greedy decoding, at most 64 new tokens, no added special tokens or chat template, and the unchanged model. These are illustrative prompts, not a representative or contamination-audited benchmark. The assessment below is an assistant review of the outputs, not a completed independent human-review study.

| Check | Observation |
|---|---|
| Explain a language model | Circular definition followed by repeated sentences. |
| Complete a photosynthesis explanation | Some relevant language, followed by circular and repeated claims. |
| Continue a short story | Begins a plausible continuation, then repeats dialogue. |
| Capital of France | Starts correctly with “Paris,” then repeats additional prose. |
| Largest planet | Incorrectly answers “the sun.” |
| Water boiling at standard pressure | Incorrectly gives approximately −20°C. |
| Copy a tracking code from context | Starts with the correct `VX-734`, then invents additional parcel details. |
| Continue an arithmetic pattern | Produces `3 + 3 = 5`. |
| Add seven red and five blue marbles | Gives 1 instead of 12, then repeats the answer. |
| Return a specified JSON object | Produces prose instead of the requested object. |
| Summarize a supplied sentence | Invents bus-size details and repeats them. |
| Complete `def add(a, b)` | Produces `b.add(a, b)` rather than implementing ordinary numeric addition. |

All twelve outputs were nonempty; none emitted EOS within the 64-token allowance. This is a finding about these prompts and this decoding configuration, not proof that EOS is never produced. Repetition under greedy decoding does not by itself identify a training defect. The wrong factual and arithmetic answers are additional quality limitations.

See [every prompt and complete response](../../artifacts/model_tests/20260926_base_smoke/GENERATIONS.md), [raw response records](../../artifacts/model_tests/20260926_base_smoke/responses.jsonl), and [run settings and checks](../../artifacts/model_tests/20260926_base_smoke/run.json).

## 3. Existing instruction diagnostics

The earlier development assessment recorded **0% strict procedural accuracy** across five families with 400 generated examples each: arithmetic, extraction, JSON, sorting, and classification. The separate 600-example grounded instruction battery also recorded 0% strict response accuracy.

These evaluations use the pipeline's instruction/chat format, which this base model was not trained to follow. Their exact-answer criteria also penalize extra output. They support a limitation in this interface; they do not mean that the model knows nothing or has zero multiple-choice accuracy. The plain completion tests above provide a complementary view.

Evidence: [development results](../../artifacts/pipeline_v1/evaluations/base_development/result.json). The independent human-review packet remains uncompleted.

## 4. Artifact and inference checks

- All **152 recorded checksum comparisons passed**, including 136 files listed in the submission package manifest, current export identity, selection record, benchmark outputs, and development artifacts.
- Weight identity matches the selected and evaluated endpoint: `3e5dd318632ffc44939b8fd8f664a845acef76186abda04b5efb10dbcef6ad85`.
- The parameter count and tied-embedding checks passed; all model weights are finite.
- A forward pass at the full 1,024-token context produced finite logits. This checks numerical operation, not comprehension at that length.
- Repeating the first greedy generation produced identical token IDs in the same runtime.
- The packaged demo exited successfully and matched the corresponding diagnostic response prefix. The diagnostic enabled the KV cache; the packaged demo retained its existing generation configuration.
- Export file hashes were identical before and after the diagnostic.
- After the packaged demo, its newly created Python bytecode caches were removed and the complete package inventory again matched its original manifest. Use `PYTHONDONTWRITEBYTECODE=1` when executing the immutable package. [Post-demo inventory check](../../artifacts/model_tests/20260926_base_smoke/package_after_cli.json).

The smoke run took approximately 8.17 seconds including loading and checks; its twelve generations took 3.39 seconds after warm-up. This is a small single-process measurement, not a production serving benchmark.

Evidence: [integrity audit](../../artifacts/model_tests/20260926_integrity.json), [smoke run](../../artifacts/model_tests/20260926_base_smoke/run.json), [packaged CLI result](../../artifacts/model_tests/20260926_base_smoke/packaged_cli.json).

## 5. Implications and next decisions

1. Present this checkpoint as an experimental pretrained text-completion model, with the actual benchmark scores and representative limitations.
2. Treat instruction following, factual reliability, arithmetic, repetition, and stopping behavior as open quality problems. Do not select only successful outputs for a capability claim.
3. Before choosing another training experiment, inspect the training/validation curves and data handling, and distinguish decoding effects from model limitations using a separately recorded comparison. More tokens alone are not an established remedy from these results.
4. Keep the current selection and official results intact. Any future training experiment needs a separate registration and evaluation plan that does not tune against already-opened final results.
5. The competition post-training decision remains as recorded in [POST_TRAINING_ELIGIBILITY.md](../../POST_TRAINING_ELIGIBILITY.md). This test provides no new organizer exception for SFT/DPO.

## Reproduce the diagnostic

From the workspace root, in the existing Python environment:

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false \
  /home/bufsgpu/yes/envs/sw/bin/python scripts/smoke_test_pretrained.py --device cuda:0
```

The [diagnostic script](../../scripts/smoke_test_pretrained.py) creates a fresh timestamped directory under `artifacts/model_tests/`. The CPU option is `--device cpu`. Model weights are loaded locally; no external model or API is used.

For an individual completion through the existing demo:

```bash
CUDA_VISIBLE_DEVICES=0 /home/bufsgpu/yes/envs/sw/bin/python scripts/demo.py \
  --model runs/continuation_12b/export --device cuda:0 \
  --prompt 'The capital city of France is' --max-new-tokens 32
```
