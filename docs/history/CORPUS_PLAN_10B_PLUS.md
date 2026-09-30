# Corpus selection for 10B+ token pretraining

> Current implementation after the September 29 cleanup: [CURRENT_PIPELINE.md](V1_CURRENT_PIPELINE.md). This reference retains historical rules, plans or execution details; completed model construction is recorded in [BEST_CHECKPOINT_TRAINING.md](V1_BEST_CHECKPOINT_TRAINING.md).

**Research date:** 2026-09-23, Asia/Seoul.  
**User direction:** extend pretraining beyond 10B tokens; research corpus sources first.  
**Status:** source research complete; execution authorized. Corpus preparation and the automatic training supervisor are running. See [CONTINUATION_RUNBOOK.md](CONTINUATION_RUNBOOK.md) and [live workflow](../../artifacts/extension_workflow.json) for the current stage. GPU pretraining waits for the complete data audit.  
**Recommendation:** target approximately **12B cumulative loss-bearing tokens**, continuing from our own completed 1B checkpoint, with **60% FineWeb-Edu / 30% DCLM / 10% English Wikipedia for the additional tokens**.

The allocation is our initial engineering choice. Published studies support these sources, but do not establish this ratio as optimal for a 46M model. Twelve billion is a practical proposed target above the requested 10B threshold, not a claim that learning saturates there.

## 1. What we already have

The baseline completed **999,948,288 targets** in 15,258 updates and about 2.09 H100-hours. It used approximately 80% FineWeb and 20% Wikipedia. Its final [export](../../runs/baseline_1b/export/) and [status](../../runs/baseline_1b/status.json) are preserved.

Keep the same 46,346,752-parameter architecture and 16,384-token byte BPE tokenizer. Its SHA256 is `1d714be4561ca083ef3cb668be7cffa38caa158c0315c22907647cd4f0f9013b`.

With the existing batch of 65,536 targets, round **up** to 12,000,034,816 cumulative tokens. That requires **11,000,086,528 additional targets**, or 167,848 additional updates. This is a continuation of our own random-initialized model, using the causal language-modeling objective. It is separate from the proposed SFT/DPO/RL stages.

The original adaptive 1B candidate has not run. The user's new priority is the larger pretraining budget; it takes precedence over automatically launching that candidate. Retain the original research protocol as an archived proposal. A 12B continuation versus the 1B baseline measures the combined effect of additional data, compute, and changed mixture; it cannot establish the benefit of the unrun adaptive controller.

## 2. Recommended sources

| Source | Role and evidence | Available scale and access | Decision |
|---|---|---|---|
| [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) | Educational explanations and knowledge-bearing web text. The [FineWeb paper](https://arxiv.org/abs/2406.17557) reports improved knowledge/reasoning benchmark performance in its own experiments. | Published 1.3T-token educational release, with later additions; public `sample-100BT` exists. Counts use the publisher's tokenizer. | **60% of continuation targets.** Primary source. |
| [DCLM-Baseline 1.0](https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0) | Broad English web language complements educational selection. [DataComp-LM](https://arxiv.org/abs/2406.11794) evaluates data curation at 412M–7B model scales and finds filtering important. | Card describes approximately 4T tokens / 3B documents; publicly accessible compressed JSONL shards. | **30%.** Use a deterministic bounded subset. |
| [English Wikipedia, 20231101.en](https://huggingface.co/datasets/wikimedia/wikipedia) | Structured reference prose with article identity, useful for a general knowledge mixture and continuity with the baseline. This is a project rationale, not evidence that 10% is optimal. | Public snapshot; 41 Parquet files totaling 11.63 GB compressed. Eligible token yield must be measured with our tokenizer. | **10%.** Exclude previously collected articles and benchmark overlaps. |

The official [SmolLM3 report](https://huggingface.co/blog/smollm3) also uses FineWeb-Edu and DCLM in its web component, alongside separate code, math, and multilingual sources. We borrow the source choice, not its much larger model, token budget, or complete mixture.

### Other sources considered

| Source | Finding | Use in this project |
|---|---|---|
| [FineWeb](https://huggingface.co/datasets/HuggingFaceFW/fineweb) | Current card reports >18.5T tokens; established source in our baseline. | Fallback for the DCLM share if its loader/quality checks fail. Declare the substitution before training. |
| [FineMath-4+](https://huggingface.co/datasets/HuggingFaceTB/finemath) | Published 9.6B Llama-token subset of mathematical web explanations; model-assisted quality filtering. | Optional later ablation at a small share, such as 5%, taken from the web allocation. It is not in the initial committed mixture. |
| [FinePDFs-Edu](https://huggingface.co/datasets/HuggingFaceFW/finepdfs-edu) | Published 350B+ tokens across 69 languages. English files alone total about 298.67 GB compressed; extraction/OCR and document-length handling add complexity. | Reserve source. Do not equate the multilingual total with English volume or assume PDF extraction is error-free. |
| [SmolLM-Corpus: fineweb-edu-dedup](https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus) | Published 220B-token deduplicated educational subset. | Valid alternative to the FineWeb-Edu sample, not an additional independent corpus. Avoid double counting overlapping content. |
| SmolLM-Corpus: Cosmopedia v2 | Textbooks/stories generated by Mixtral, according to the same source card. | Exclude from the default recipe because larger-model-generated training targets raise the existing distillation issue. Do not import the entire multi-subset repository by default. |

FineWeb-Edu is derived from FineWeb, and DCLM also comes from Common Crawl. Repository names do not establish document independence. Cross-source deduplication is required.

## 3. Exact repository pins and loading routes

All seven inspected repositories were public and ungated at inspection. Save the selected files and their content hashes in the eventual training manifest; repository pins alone do not specify which documents were consumed.

| Source | Immutable revision | Selected configuration/path |
|---|---|---|
| FineWeb-Edu | `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9` | `sample-100BT`; `sample/100BT/*.parquet` |
| DCLM | `a3b142c183aebe5af344955ae20836eb34dcf69b` | `global-shard_*/local-shard_*/*.jsonl.zst` |
| Wikipedia | `b04c8d1ceb2f5cd4588862100d08de323dccfbaa` | `20231101.en`; `20231101.en/*.parquet` |
| FineWeb fallback | `9bb295ddab0e05d785b879661af7260fed5140fc` | `sample-100BT`; `sample/100BT/*.parquet` |
| FineMath optional | `e92b25a616738fe95dc186b64dfb19f9c8525594` | `finemath-4plus`; `finemath-4plus/*.parquet` |
| FinePDFs-Edu reserve | `9cfabe2127faca99b3d5c4dc6d1fcb397399ebde` | `eng_Latn`; `data/eng_Latn/train/*.parquet` |
| SmolLM-Corpus alternative | `3ba9d605774198c5868892d7a8deda78031a781f` | Only `fineweb-edu-dedup` |

Use the 100BT source pool for educational text to provide room for exclusions. Its **140 files total 286.39 GB**, but we only need a selected subset. The 10BT, 100BT, and 350BT samples are nested according to the card; concatenating them would duplicate documents. FineWeb's 100BT pool has 150 files totaling 302.55 GB. These are measured repository file sizes, not proposed download totals.

## 4. Provenance and licensing

FineWeb-Edu retains original web text selected by a classifier trained with Llama-3-70B-Instruct ratings. DCLM's classifier uses OpenHermes/ELI5 examples to select original web documents. FineMath similarly uses model-assisted scoring. **Selection metadata is not a replacement answer or teacher logit.** Train only on the source `text`; do not serialize rating prompts, scores, or generated annotations into our token stream.

This is the technical distinction from training on larger-model completions. Our reading of the previously checked rules permits public pretraining corpora, with disclosure of their filtering provenance; it is not a claim that organizers explicitly approved each dataset. Web corpora themselves may contain synthetic pages, and no guarantee of exclusively human authorship is made. The [live rules](https://gibc-v2.devpost.com/rules) returned a bot challenge during this inspection; the substantive rules snapshot remains the earlier September 23 review in the [competition compass](../../COMPETITION_BRIEF.md).

| Dataset | Published notice to preserve |
|---|---|
| FineWeb / FineWeb-Edu / FineMath / FinePDFs-Edu | ODC-By 1.0 database notice and Common Crawl terms; underlying pages retain their own rights |
| DCLM | CC-BY-4.0 dataset notice and attribution; card describes research-oriented intended use; underlying web-content rights remain relevant |
| Wikipedia | Snapshot card lists CC-BY-SA-3.0/GFDL; current [upstream dump notice](https://dumps.wikimedia.org/legal.html) describes CC-BY-SA-4.0/GFDL with exceptions. Preserve source/article attribution and both notices rather than silently reconciling them |

Publish corpus recipes, identities, and attribution; do not relicense the entire raw corpus as project code.

## 5. What the local sample audit actually established

The [audit script](../../archive/v1_code/scripts/audit_corpus_samples.py) used our frozen tokenizer, the existing benchmark exclusion index, and a read-only check against the previous raw-document hash database. It sampled 128 records from each of three seeded files per source. Parquet reads use byte ranges; DCLM uses short streamed prefixes. No GPU training or full dataset download occurred.

| Source | Inspected documents | Passed these limited checks | Findings |
|---|---:|---:|---|
| FineWeb-Edu | 384 | 383 | One exact match to the prior collected pool |
| DCLM | 384 | 382 | Two documents outside the existing length bounds |
| Wikipedia | 384 | 367 | Seventeen documents outside the length bounds |
| FineWeb fallback | 384 | 370 | Fourteen exact matches to the prior collected pool |
| FineMath optional | 384 | 383 | One document outside the length bounds |

No benchmark exact matches appeared in these samples. This does not establish corpus-wide decontamination, factual correctness, English purity, or a reliable population rejection rate. The prior hash database covers more raw documents than the model actually trained on; overlap counts are deliberately conservative.

FineWeb-Edu sample texts used a median **1.10×** as many tokens under our BPE as under the source token counter. Thus a publisher's “10B tokens” is not our training budget. Count all shares and milestones after our tokenization, including EOS under the established objective.

Evidence: [repository metadata](../../artifacts/corpus_research/summary.json), [sample audit](../../artifacts/corpus_research/sample_audit.json), and pinned cards/files in `artifacts/corpus_research/`. The sample is for source feasibility, not a benchmark result.

## 6. Corpus budget and preparation requirements

| Source | Approximate continuation targets | Initial pool to prepare after exclusions |
|---|---:|---:|
| FineWeb-Edu | 6.60B | 8.0B unique stored tokens |
| DCLM | 3.30B | 4.0B unique stored tokens |
| Wikipedia | 1.10B | 1.5B unique stored tokens |
| **Total** | **11.00B additional** | **13.5B eligible stored tokens** |

The extra pool is sampling/boundary headroom, not automatically extra training. These yields are targets to verify, especially for Wikipedia after exclusions; do not silently repeat data or alter ratios if a quota is short. The table applies to the **continuation**, so the cumulative 12B mixture also includes the completed baseline's web/wiki tokens.

Preparation must:

1. Use a new output directory, frozen source pins, seeded file order, and bounded streaming/download caches.
2. Import the original tokenizer unchanged. The existing preparer trains a tokenizer for a new output directory, so it cannot be reused unchanged for continuation.
3. Exclude prior training documents and every reserved evaluation document, including mirrors/near duplicates. Apply existing official benchmark exclusions to all sources; add URL/article identity and a documented fuzzy-deduplication procedure.
4. Deduplicate globally across new sources and against old content. Prefer canonical Wikipedia text for Wikipedia mirrors and retain source attribution on surviving documents. Log exact and fuzzy removal counts separately.
5. Allocate disjoint new monitor/development/selection/final groups by duplicate cluster. Keep all old benchmark and final panels protected. Do not use official scores to tune the mixture.
6. Tokenize to checksummed uint16 shards with source identities and actual counts. Retain long documents through successive causal windows; a 1,024-token context does not require dropping every longer document.
7. Publish a complete manifest only when every quota, tokenizer hash, vocabulary bound, split check, and exclusion requirement passes. A small streaming sample is not sufficient for this gate.

At two bytes per token, the 13.5B-token pool requires about **27 GB** for token IDs, before raw text, source caches, indexes, and temporary files. Reserve **250 GB** initially for bounded preparation and monitor it; this is an engineering allowance, not a measured peak. The workspace had about **3.6 TB free** at inspection. Full repository downloads are unnecessary.

## 7. Continuation training and evaluation implications

At the completed baseline's measured elapsed rate, 11B extra tokens would take approximately **23 H100-hours**. This estimate excludes new preparation, additional evaluation, and changes in runtime. Keep GPU 0 as the sole assigned GPU.

The original trainer hardcoded `web` and `wiki` and rejected changed token budgets or manifests. The implemented extension adds an explicit three-source continuation path with parent-checkpoint provenance, verified inherited model/optimizer state, fresh source cursors, and a declared new schedule. Same-run resume still rejects changed scientific settings. [Continual pretraining research](https://arxiv.org/abs/2403.08763) supports considering learning-rate rewarming/redecay, but its optimum cannot be assumed here. Validate a short pilot before committing the full run.

Record cumulative milestones near 2B, 5B, 10B, and 12B, with rounded update counts and separate continuation/cumulative cost ledgers. Use protected development evidence for progress; run official benchmarks only after final model selection under the updated evaluation protocol. Preserve the 1B artifact so additional learning and regressions remain measurable.

The frozen-tokenizer preparer, cross-corpus filters, and explicit continuation trainer are implemented and tested. The user authorized execution; follow [CONTINUATION_RUNBOOK.md](CONTINUATION_RUNBOOK.md) for the active data/pilot/full-training workflow. SFT/DPO/RL remains a separate later decision under [POST_TRAINING_PLAN.md](../../artifacts/pipeline_v1/submission_package/POST_TRAINING_PLAN.md).

## Decision record

| Date | Decision | Status |
|---|---|---|
| 2026-09-23 | Increase pretraining scope beyond 10B cumulative tokens | User-requested; supersedes treating the 1B baseline as the final training budget |
| 2026-09-23 | Propose a 12B cumulative endpoint with 60/30/10 Edu/DCLM/Wikipedia continuation | Research-backed source choice; ratios and endpoint are project choices |
| 2026-09-23 | Pin seven candidate repositories; inspect bounded samples from five | Completed; metadata and audit saved |
| 2026-09-23 | Keep FineMath/PDFs as optional later sources | Prioritize a coherent general-English corpus and measurable source effects |

| 2026-09-23 | User authorized execution; launch corpus preparation and automatic audited pilot/full continuation | Running; live status takes precedence |
