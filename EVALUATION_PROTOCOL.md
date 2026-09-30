# Frozen Track 01 Evaluation Protocol

> **Current (2026-10-01):** V2 is the selected model and the only active pipeline; see [README.md](README.md), [CURRENT_PIPELINE.md](CURRENT_PIPELINE.md) and [BEST_CHECKPOINT_TRAINING.md](BEST_CHECKPOINT_TRAINING.md). The rest of this document is a dated record; V1-era material is in [docs/history/](docs/history/).

**Protocol:** `gibc-track01-eval-v1`  
**Frozen:** September 23, 2026, before any project model benchmark scores were observed.  
**Executable configuration:** [configs/evaluation.json](configs/evaluation.json).  
**Runner:** [scripts/run_final_evaluation.py](scripts/run_final_evaluation.py).

These are our declared reporting choices. The organizer has not specified the exact shots, splits, or perplexity convention. Any later clarification must be applied consistently to every reported arm and documented as a protocol revision.

## Selection gate

Run official evaluation only after a frozen selection record exists for the reserved `selection` development panel. V2 writes it with `python -m scglm_v2.select_endpoint` ([src/scglm_v2/select_endpoint.py](src/scglm_v2/select_endpoint.py)); V1 used [scripts/select_endpoint.py](archive/v1_code/scripts/select_endpoint.py), now archived. The final runner requires `--final-evaluation` and that record. It checks `official_scores_used: false`, the panel name, and the model weights' SHA256 against the selection record.

The selected model and other completed endpoints listed in the same record may all be evaluated. Report every completed primary arm. Official benchmark scores cannot change the already selected checkpoint. Each evaluation attempt receives its own fresh results directory, including repeated evaluations of the same model.

**Post-training selection extension:** [INSTRUCTION_TUNING.md](artifacts/pipeline_v1/submission_package/INSTRUCTION_TUNING.md) specifies the implemented `scglm-post-selection-v1` extension for a completed 10B+ base and its SFT arms. `python -m scglm_post.select` freezes a challenger using instruction development and reserved raw-LM selection panels, then applies one independent confirmation gate. Its `selection.json` records `selection_kind=posttraining-v1`, `panel=selection`, and `official_scores_used=false`, and is accepted by the same final runner. Benchmark settings below remain unchanged. This is an added-compute parent/SFT comparison; use the separate selector rather than inserting post-training runs into the original matched 1B-pretraining selector. DPO/RL remain proposals in [POST_TRAINING_PLAN.md](artifacts/pipeline_v1/submission_package/POST_TRAINING_PLAN.md).

**12B continuation:** this is an unmatched longer-training extension of B0 with a changed corpus. Do not insert it into the matched B0/C selector unchanged. Register and implement its endpoint-selection extension before official evaluation, preserving the task settings below. The active training supervisor uses monitoring panels only; [CONTINUATION_RUNBOOK.md](docs/history/CONTINUATION_RUNBOOK.md) records its divergence gates.

## Harness version and task definitions

Use the clean checkout at [vendor/lm-evaluation-harness](vendor/lm-evaluation-harness), release **v0.4.12**, immutable commit **`6d642546f4688648fced259eb3302efd36ece5af`**. The installed 0.4.12 task files and relevant helpers were compared byte-for-byte with this checkout. [Upstream release commit](https://github.com/EleutherAI/lm-evaluation-harness/tree/6d642546f4688648fced259eb3302efd36ece5af)

The vendor directory is excluded from project version control. To recreate it in a fresh checkout, run `git clone --depth 1 --branch v0.4.12 https://github.com/EleutherAI/lm-evaluation-harness.git vendor/lm-evaluation-harness`. The runner verifies that the resulting commit is exactly the hash above before doing any work.

The runner hashes the official task files and relevant helpers, checks the clean Git revision and recorded package versions, and creates task YAML overrides **outside** the vendor checkout. These overrides pin dataset revisions while inheriting the release's prompts, helper functions, choices, labels, and metrics. Their exact hashes are saved in `preparation.json`.

| Task | Dataset revision | Evaluated split | Examples | Metrics |
|---|---|---|---:|---|
| HellaSwag | `Rowan/hellaswag` at `218ec52e09a7e7462a5400043bb9a69a41d06b76` | validation | 10,042 | `acc`, `acc_norm` |
| ARC-Easy | `allenai/ai2_arc`, `ARC-Easy`, at `210d026faf9955653af8916fad021475a3f00453` | test | 2,376 | `acc`, `acc_norm` |
| PIQA | `ybisk/piqa` at `ba2f7a4920f4968f04c174c53cc157c899e93501` | validation | 1,838 | `acc`, `acc_norm` |
| WinoGrande | `allenai/winogrande`, `winogrande_xl`, at `01e74176c63542e6b0bcb004dcdea22d94fb67b5` | validation | 1,267 | `acc` |

All rows are evaluated with **zero demonstrations**, batch size **1**, no example limit, no chat template, and no instruction/system prompt added beyond the task definition. Seeds for Python, NumPy, Torch, and few-shot selection are respectively `0,1234,1234,1234`.

HellaSwag uses the release's activity/context preprocessing and candidate endings. ARC-Easy and PIQA use the release's question/answer prompt and score each answer continuation. WinoGrande compares the likelihood of the common sentence suffix under each prefix containing a candidate noun phrase; it does not score generated option letters. The continuation delimiter is one space.

`acc` selects the maximum summed conditional continuation log-likelihood. In this release, `acc_norm` divides that sum by **Python `len(choice)`**, the candidate string's Unicode character count, excluding the separately added delimiter. It is **not token-count normalization**. Metrics are arithmetic means over examples. Standard harness uncertainty summaries concern the evaluated examples, not variation across independent training runs. [Pinned metric implementation](https://github.com/EleutherAI/lm-evaluation-harness/blob/6d642546f4688648fced259eb3302efd36ece5af/lm_eval/api/task.py)

Models load from local exports with `trust_remote_code=False`, float32 weights/computation, and a maximum input context of 1,024. Automatic BOS insertion is disabled. Empty contexts, if any, receive the explicitly chosen EOS prefix token ID `2`. For overlong requests, the release retains the last 1,025 combined context/continuation tokens and removes the final token from model input; preserve any truncation warnings. Continuations longer than 1,024 tokens fail rather than silently changing answer scoring. [Pinned HF adapter](https://github.com/EleutherAI/lm-evaluation-harness/blob/6d642546f4688648fced259eb3302efd36ece5af/lm_eval/models/huggingface.py)

### PIQA compatibility choice

The release's PIQA task points to the `baber/piqa` mirror. Our override loads the primary `ybisk/piqa` Parquet conversion at the immutable revision above, preserving the release's task semantics. This also avoids the legacy dataset script unsupported by the installed `datasets` 4.8.4. The original conversion and its file identities are public. [Primary conversion commit](https://huggingface.co/datasets/ybisk/piqa/commit/ba2f7a4920f4968f04c174c53cc157c899e93501)

Files used by the task are `plain_text/train-00000-of-00001.parquet` and `plain_text/validation-00000-of-00001.parquet`. Training-split availability satisfies the task definition; zero-shot evaluation uses no demonstrations. Both downloaded files were independently checked for schema, row count, and SHA256. Their hashes are recorded in the configuration. Training-corpus exclusion additionally used the public test texts without accessing hidden test answers.

## Explicit WikiText-103 perplexity

Use `Salesforce/wikitext`, configuration **`wikitext-103-raw-v1`**, revision **`b08601e04326c79dfdd32d625aee71d232d685c3`**, full **validation** split: 3,760 raw rows. This is the same revision used for training exclusions. [Pinned dataset revision](https://huggingface.co/datasets/Salesforce/wikitext/tree/b08601e04326c79dfdd32d625aee71d232d685c3)

Do not run the generic harness task named `wikitext` as a substitute: this release uses WikiText-2 and reports word/byte metrics. Our separate [scorer](src/scglm/evaluate.py) is hashed in the frozen configuration.

The exact scoring convention is:

1. Reconstruct articles using the scorer's top-level heading matcher. Discard whitespace-only rows, preserve the remaining row strings, and join them with one newline. Retain headings; perform no additional detokenization.
2. Encode each article independently with the selected model's own frozen tokenizer. Add no BOS, EOS, or other special token.
3. Score with input windows of at most 1,024 tokens and stride 512. Reset RoPE positions to zero for each window and never cross article boundaries.
4. Leave the first token of each article unscored. Score every subsequent token exactly once, using overlapping windows only as context. Apply no document or token limit to this benchmark.
5. Compute summed token NLL using FP32 cross-entropy with FP64 accumulation. Divide total NLL by the total number of scored tokens, then exponentiate. Do not average article perplexities.

Report article count, exact scored-token count, context, stride, tokenizer hash, NLL, and token perplexity. Token perplexity depends on the tokenizer and these reconstruction/window choices; it is not automatically comparable with published word perplexities or differently tokenized models.

## Commands and evidence

Validate pinned task loading and save exact commands **without loading any model or calculating scores**:

```bash
python scripts/validate.py --prepare-only --label protocol-preflight
```

After the completed endpoints have been selected on development data, run each listed endpoint in its own directory:

```bash
python scripts/validate.py \
  --model runs/baseline_1b/export \
  --selection-record artifacts/selection/selection.json \
  --device cuda:0 \
  --final-evaluation --label baseline-1b
```

Set `CUDA_VISIBLE_DEVICES` to the one assigned GPU before actual GPU execution. The example model path identifies one arm, not a predetermined winner. Use the same protocol for the candidate and any completed average-mixture control.

`scripts/validate.py` delegates to the final runner; both entry points use the same selection checks and automatic recording. Preflight also checks the pinned WikiText-103 held-out row count. These audit additions do not alter the frozen task definitions, scorer, or evaluation configuration.

Every parsed invocation writes a unique record under `artifacts/validation/runs/<UTC-time>-<id>/` and appends lifecycle events to `artifacts/validation/history.jsonl`. Recording begins before configuration checks, dataset loading, and inference. `run.json` records status (`running`, `prepared`, `completed`, `failed`, or `interrupted`), labels, timestamps, model identity, protocol hash, stage commands/exit codes, and available scores. Atomic JSON replacement and a locked, flushed append-only journal preserve separate concurrent attempts. SIGINT/SIGTERM are recorded; SIGKILL or machine failure leaves the start record without a terminal success.

By default, artifacts go into that attempt's `results/` directory. Optional `--output` requires a new directory; even failed attempts cannot be overwritten. `--history-dir` selects another journal root. The runner saves stdout/stderr for each inference stage in `harness.log` and `wikitext103.log`, and error tracebacks in `error.log`. After checking every expected benchmark metric, sample count, and few-shot setting, it retains the harness scores immediately. A later WikiText failure preserves those partial scores with a failed status. A complete run publishes `results/summary.json` with raw and normalized accuracy where defined, token NLL, token perplexity, article/token counts, and result hashes. The model files are hashed before and after inference to detect replacement during a run. No composite competition score is invented.

Inspect all attempts with `python scripts/validation_history.py`, or add `--json` for full current records. The compact view displays normalized accuracy for HellaSwag/ARC-Easy/PIQA, raw WinoGrande accuracy, and WikiText-103 perplexity; raw metrics remain in the complete JSON records. Only `completed` records represent the full suite. This journal covers these pipeline entry points; training development-panel metrics remain in each run's existing `events.jsonl`, and ad hoc direct calls to `lm_eval` are outside the pipeline.

The runner records the protocol snapshot/hash, clean harness commit, task override hashes, task split/count checks, runtime versions, planned commands, selection-record hash, and model/config/tokenizer/provenance hashes. It then runs the pinned harness and the separate WikiText-103 scorer. Harness raw samples and final reports remain in that model's output directory; `completed.json` is written only after both stages succeed.

Preparation on September 23 successfully loaded all four pinned tasks and confirmed their splits, counts, and metric names. **No model was loaded and no model benchmark score was calculated during protocol preparation.** This validates task loading and protocol wiring; it does not replace a future complete model evaluation.

System-enabled post-training additionally records aligned/conflicting system instructions, multi-turn persistence, and paired-policy accuracy. A pair shares its user request but changes its system policy and required answer. The bootstrap resamples whole request groups within task families. Promotion permits at most 2 percentage points of regression in each system mode and complete-pair accuracy, alongside the existing instruction-gain and raw-LM gates. These auxiliary checks do not change the frozen official benchmark prompts or scoring. See `INSTRUCTION_TUNING.md`.
