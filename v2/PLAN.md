# V2 plan (approved by the user 2026-09-29, KST)

## Objective

Competition benchmark performance (HellaSwag, ARC-Easy, PIQA, WinoGrande; WikiText-103 perplexity) of a **pretrained base**. There is no SFT, because the Resources page excludes fine-tuning for Track 01. V2 replaces V1 only if it wins under the frozen selection rule below. V1's official results were observed previously; this is disclosed.

## Evidence checked before approval

| Question | Finding | Decision |
| --- | --- | --- |
| Can throughput rise? | Synthetic probe on the assigned H100: 193k tok/s (V1 settings), 329k (microbatch 64), **404k** (microbatch 64 + compile) | Adopt microbatch 64 + compile |
| Deep-thin architecture? | MobileLLM's same-size ablation: 30 layers vs 12 layers = 44.8 vs 43.9 avg (+0.9); our 24×384 GQA variant measured 25% slower | Keep V1's 12×512 architecture |
| Muon? | A 2025 benchmark against a tuned AdamW: ≤1.4× at 130M, and behind SOAP/Kron above 8× Chinchilla (V2 is ~40×) | Keep AdamW |
| Data mixture | SmolLM2: FineWeb-Edu leads on ARC/OBQA, DCLM on HellaSwag/CSQA; 60/40 mix works well | DCLM + Edu core |
| Targeted selection | BETR (2025): selecting documents similar to benchmark **train** examples gives ~2.1× compute over DCLM; small models benefit most | 15% targeted slice |
| Diversity | V1's A12: stricter heuristic filtering worsened held-out NLL; 2025 mixture work shows diversity helps HellaSwag | Cap targeted slice; add StackExchange and books |

## Data (all human-written; no model-generated corpora)

| Planned share | Source | Unique-token goal |
| ---: | --- | --- |
| 35% | DCLM-baseline (V1 extension text + 43 new pinned shards) | ≥7B |
| 30% | FineWeb-Edu (V1 extension text + 6 new pinned files) | ≥6B |
| 15% | Targeted: top 20% of the Edu+DCLM pool by benchmark-similarity classifier | ≥3B |
| 10% | English Wikipedia (V1 main + extension) | ~2.5B |
| 5% | StackExchange 2025: 54 English non-code sites | ≥1B |
| 5% | Project Gutenberg English (12 pinned files, 20k-char chunks, boilerplate removed) | ≥1B |

**Contamination controls:**
- The existing 13-word exclusion index (train and eval splits of all four benchmarks, plus WikiText-103) applies to all new text.
- Domain blocklist: `wikihow.com` (HellaSwag source) and `instructables.com` (PIQA source), for all sources including reused text.
- Exact, URL and MinHash near-duplicate removal continues from a copy of V1's deduplication database.
- The classifier uses 80% of each benchmark **train** split. The other 20% is the V2 development proxy panel. Official eval splits are never used.

## Phases

1. **Preparation** (this phase): download, tokenizer study, targeted classifier, build corpus and manifest. CPU only.
2. **Trainer + pilot:** V2 trainer (compile, microbatch 64, warmup/constant/decay schedule, bits-per-byte metric). One 1B-token pilot with V1 `baseline_1b`'s schedule, compared with that checkpoint. **Go/no-go gate.**
3. **Main run:** 12×512, AdamW, peak LR 6e-4, batch 64×1024, warmup/constant/decay. Tokens are set by the deadline: about 17B if stopping for Sep 30 20:00 KST, about 40B if stopping for the Oct 2 00:45 KST cutoff.
4. **Selection and reporting:** development-only frozen selection, then the pinned official protocol (`configs/evaluation.json`) against V1.
