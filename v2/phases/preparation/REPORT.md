# V2 preparation phase: `v2_rich_prep_20260929`

**Status:** completed 2026-09-29 17:43 KST. **Manifest:** `v2/data/rich_v1/manifest.json`, SHA256 `6749a2f1e2c3fe574febf0c1d08fb99d87d6e226c995aedd0d4ae71921496a68`. [Registration](registration.json) · [status](status.json) · [events](events.jsonl) · [plan](../../PLAN.md)

**Objective:** build the rich V2 pretraining corpus, choose the tokenizer, and freeze the development proxy panel. CPU only; no model training. **Parent:** none (V2 starts from random initialization). **Output:** `v2/data/rich_v1/`.

## Steps

| Step | State | Result |
| --- | --- | --- |
| Benchmark train-split texts | done | 101,157 classifier positives and 25,006 held-out development items from 8 sets. **SocialIQA unavailable** (its loader needs a dataset script, which the library no longer supports); recorded as a coverage gap |
| Targeted classifier | done | fastText; 46,407 class-balanced positives / 46,408 length-matched pool negatives; validation AUC **0.994**; top-20% (character-weighted) threshold **0.0562**. [Selected examples](../../data/rich_v1/targeted/selected_examples.json) are science, nature, how-to and everyday-fact pages |
| Tokenizer study | done | **V1 tokenizer kept** by the registered rule (below) |
| Download new pinned files | done | 114 files, 22.7 GB: StackExchange (54 sites), Gutenberg (12 files), DCLM (43 shards). New FineWeb-Edu files were dropped because of measured bandwidth (~5 MB/s); the reused 8.0B Edu tokens exceed the goal |
| Corpus build (filter, dedup, route, tokenize) | done | See the tables below |
| Manifest and audit | done | One fix: Edu and Wikipedia panels were empty at first (V1 had already removed their panel hash range) and now use V1's own held-out panels. Train shards were unchanged. A pilot launch stopped at step 0 for this fix is kept in `v2/runs/pilot_1b_aborted_manifest_fix/` |

### Final corpus (unique training tokens, V1 tokenizer)

| Bucket | Tokens | Documents | Passes at 17B / 40B | Largest single host |
| --- | ---: | ---: | ---: | --- |
| DCLM | 8,286,499,743 | 5,524,265 | 0.72 / 1.69 | stackoverflow.com, 0.23% of documents |
| FineWeb-Edu | 6,023,274,513 | 5,041,791 | 0.85 / 1.99 | britannica.com, 0.26% |
| Targeted | 3,521,067,924 | 3,732,888 | 0.72 / 1.70 | reference.com, 0.32% |
| Wikipedia | 2,020,057,734 | 2,487,369 | 0.84 / 1.98 | — |
| StackExchange | 1,613,083,489 | 1,118,884 | 0.53 / 1.24 | physics site, 11% |
| Gutenberg books | 990,115,242 | 180,773 chunks | 0.86 / 2.02 | — |
| **Total** | **22,454,098,645** | | | |

### Filtering of reused and new text

| Removed | Reused Edu/DCLM/Wiki | New DCLM | StackExchange | Books |
| --- | ---: | ---: | ---: | ---: |
| Input documents | 12,562,804 | 4,435,593 | 1,124,494 (Score ≥1, answered) | 293,786 chunks |
| Domain blocklist (wikiHow, Instructables) | 8,566 | 4,142 | — | — |
| 13-word benchmark match | (done in V1) | 1,124 | 786 | 92 |
| Exact duplicate / same URL | (done in V1) | 59,256 / 72,281 | — | 112,049 (same book in several encodings) |
| MinHash near-duplicate | (done in V1) | 36,296 | 430 | 141 |
| Length | (done in V1) | 13,462 | 13 | 5 |

### Tokenizer study (UTF-8 bytes per token; higher is better)

| Tokenizer | DCLM | Edu | Wiki | StackEx | Books | Mixture | Benchmark text |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| V1 16k | 3.955 | 4.165 | **3.845** | 3.993 | 3.567 | 4.006 | 4.047 |
| New 16k | 3.979 | 4.133 | 3.503 | 4.146 | 3.653 | 3.981 (−0.6%) | 4.184 (+3.4%) |
| New 20k | 4.063 | 4.232 | 3.602 | 4.206 | 3.728 | 4.071 (+1.6%) | 4.289 (+6.0%) |

No candidate gained at least 2% on both measures, so the V1 tokenizer is kept (SHA256 `1d714be4…f0f9013b`). The new tokenizers lose substantially on Wikipedia, which matters for WikiText-103 perplexity. Candidates are kept in `data/rich_v1/tokenizer_study/`.

### Trainer check (GPU; throwaway run, not retained)

The V2 trainer (`torch.compile`, microbatch 64, non-deterministic kernels) ran 150 steps on V1's own 1B corpus and schedule. Losses at steps 50/100/150 were **8.231 / 7.136 / 6.751**, against V1 `baseline_1b`'s 8.231 / 7.136 / 6.752. Throughput was **~395k tokens/s** (V1: 137k) while the CPU was busy building the corpus. CPU tests: 12 passed (`v2/tests`).

Final counts, hashes and the decision are added when the phase completes.
