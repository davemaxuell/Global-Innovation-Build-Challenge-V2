# Historical proposal: checkpoint handoff and further pretraining

**Withdrawn from the active roadmap on September 29, 2026 at the user’s cleanup request. This proposal was never launched. Current instructions: [retained pipeline](../../CURRENT_PIPELINE.md).**

**Date:** September 29, 2026. **Status:** checkpoint handoff completed; next training experiment proposed, not launched. The user asked to bring forward the best checkpoint and identify the next training step. This record answers that question; it does not describe an executed optimizer update.

## Ready-to-load checkpoints

The selected full-SFT instruction model is now available as [checkpoints/best_sft](../../checkpoints/best_sft/). It is an independent, read-only copy of recipe 5 / seed 20260928 / update 105, with weight SHA256 `defd0d91948ba8501a33b0c4996bf3bf37ebdf6e90338a05bab3a1773a2d7f72`. Its ancestry contains 13,000,048,640 raw pretraining targets, then 862,660 response and 419,498 replay targets. All 46,346,752 parameters participated in that SFT run.

The retained competition reference is available as [checkpoints/competition_base](../../checkpoints/competition_base/), weight SHA256 `ffa75b07ec377167ab2e4d4263f2fa8374549f8592f9ce2e8166875cab1f38e7`. The original runs, preserved snapshots and release packages remain intact. Both copies passed checksum verification, offline model/tokenizer reload, finite forward inference, and the original parameter/tied-weight audit. [Verification inventory](../../checkpoints/SELECTION_2026-09-29.json).

There is no demonstrated single winner across all objectives. The full-SFT model is our established instruction model; the base remains the competition reference. The head-only study's new knowledge panel gave full SFT 17.46%, base 15.47%, and adapter 15.36%, with uncertainty too wide to establish all pairwise differences. The much stronger full-SFT result on its constrained generation tasks does not imply an overall competition-score advantage.

## Recommended next experiment: bounded continued pretraining

Start independently from the **13B raw base**, and test whether a small additional raw-text budget and a higher educational-data share improve capability. Keep the full-SFT checkpoint available for instruction use. If a stronger base emerges, train a separate short full-SFT descendant and compare it with the current instruction checkpoint.

Why this follows the evidence:

- The earlier DPO and RL experiments did not improve their SFT parents. Repeating those stages without changing the signal is not supported by the results. [Completed post-training study](../posttraining/2026-09-29_upgraded_pipeline_completion.md).
- The head-only adapter did not show reliable knowledge improvement and learned little task-format behavior. [Head-only results](../../artifacts/lm_head_lora_v2/REPORT.md).
- The later knowledge-heavy full-SFT pilot improved its development knowledge score but failed raw-loss retention and scored below base on all four raw benchmark point estimates. [Latest SFT pilot](../../artifacts/sft_benchmark_20260929/REPORT.md).

These findings motivate testing base capability and data mixture. They do not prove that insufficient pretraining is the sole bottleneck or that more tokens will help.

### Research basis

[FineWeb / FineWeb-Edu](https://arxiv.org/abs/2406.17557) reports gains on knowledge/reasoning benchmarks from educational-text filtering in its experiments. [DataComp-LM](https://arxiv.org/abs/2406.11794) shows that data curation/filtering materially affects model quality in controlled pretraining comparisons. These support a controlled mixture experiment; neither establishes an optimal mix for this 46M model.

[Ibrahim et al., Simple and Scalable Strategies to Continually Pre-train Large Language Models](https://arxiv.org/abs/2403.08763) studies learning-rate rewarming/redecay and replay under data changes at substantially larger scales. This supports evaluating a conservative continuation schedule, not a guaranteed gain or a published optimum for our model.

### Proposed bounded recipe

These are proposed settings. Implementation must freeze the final source/data/evaluation contract before any candidate scoring or training begins.

| Item | Proposal |
| --- | --- |
| Phase ID / future output | `cpt_data_mix_20260929_v1`; separate `artifacts/cpt_data_mix_20260929_v1/` and `runs/cpt_data_mix_20260929_v1/` |
| Parent | Same 13B raw base for both arms; verified final resume-state SHA256 `86ef06ba045ebfd613233e9e112817e5351e6e50db2e7a312718061662d1eef8` |
| Architecture | Unchanged 46,346,752-parameter model, scratch tokenizer, 1,024-token context; full-parameter next-token training |
| Control data mix | Existing 60% FineWeb-Edu / 30% DCLM / 10% Wikipedia, by predicted target tokens |
| Candidate data mix | 80% FineWeb-Edu / 10% DCLM / 10% Wikipedia, by predicted target tokens |
| Per-arm budget | 3,815 updates × 65,536 targets = **250,019,840 additional targets** |
| Total training comparison | **500,039,680 targets** across two independent arms; each endpoint has **13,250,068,480 cumulative targets**, not 13.5B |
| Batch and precision | Existing 64×1,024 effective targets, microbatch 8, BF16 autocast with FP32 weights/Adam/loss |
| Optimizer | Inherit verified Adam moments and counters; betas (0.9, 0.95), epsilon 1e-8, matrix decay 0.1, gradient norm cap 1 |
| Proposed LR | Rewarm from the final 3e-6 to 3e-5 over 128 updates, then cosine decay to 3e-6 by update 3,815; same schedule in both arms |
| Data state | Start at the verified parent cursors; no wrap or silent restart. Remove any newly reserved held-out documents from both arms before their final manifests are frozen. |
| Hardware and ceiling | Assigned H100 only; provisional four-hour total experiment ceiling, including validation/reporting. Expected training-only time is roughly one hour using the prior measured rate; preparation/evaluation add time. |
| Checkpoints and early checks | Recovery checkpoints every 500 updates and at completion. Finite-loss/gradient checks throughout; warmup-aware divergence checks on separate monitor data. |
| Automatic continuation | No larger token extension, DPO or RL triggered by this proposal. Replication or scaling needs a separately documented follow-up. |

The corpus manifest remains [the pinned extension manifest](../../data/processed/extension_12b/manifest.json), SHA256 `70242a0a248787d31d56c73af3733ccafadaa9816b09ca0508ec487f1fd37c6d`. The final parent resume checkpoint was checksum-verified and its saved stream states inspected. All source cycle counts are zero:

| Source | Stored tokens | Parent cursor | Remaining stored tokens |
| --- | ---: | ---: | ---: |
| FineWeb-Edu | 8,000,001,725 | 7,198,570,496 | 801,431,229 |
| DCLM | 4,000,001,649 | 3,601,405,952 | 398,595,697 |
| Wikipedia | 1,500,000,029 | 1,200,123,904 | 299,876,125 |

These are physical stream capacities, not a completed audit of a new training subset. The preparation stage still needs to simulate both arms, reserve fresh held-out material, check benchmark/prior-evaluation overlap, and freeze exact readable spans. Shared source prefixes across independent arms are acceptable for a controlled comparison; they must not be counted twice as the ancestry of one model.

### Evaluation and decision

1. Prepare fresh, disjoint raw development and confirmation documents, plus a new held-out knowledge/reasoning panel. Existing consumed confirmation questions cannot become fresh evidence by renaming them. Freeze identities, metrics, source-grouping and thresholds before model scoring. The exact new knowledge-panel sources are a preparation deliverable, not yet selected.
2. Use a fixed 60/30/10 raw evaluation mixture for all models, irrespective of each arm's training mix. Measure per-source NLL, paired uncertainty, knowledge accuracy and generation diagnostics.
3. Proposed raw gate: at least 0.005-nat weighted improvement over the parent, paired 95% NLL interval below zero, and no source point-estimate regression above 0.02 nats. Require no greater than two-point knowledge-domain regression. These preservation thresholds are engineering guards, not statistical non-inferiority guarantees.
4. Rank only eligible **completed endpoints** on development. Require the educational arm to show useful benefit over the matched control before attributing benefit to its changed mixture. Freeze one winner before a single independent confirmation comparison; otherwise retain the base.
5. After selection, report the established competition benchmarks and disclose all previously observed scores. Benchmark results do not retune the chosen recipe or select another checkpoint. No automatic package replacement.
6. If the new base passes, the next post-training phase is a short full-SFT experiment with raw replay, using the same data/target budget as its current-SFT comparator where possible. DPO/RL remain deferred until a changed training signal is justified and tested.

### Preparation, launch and resume status

No new training was launched for this handoff. The next executable work is preparation of the two-arm continuation and its fresh evaluation contract. Launch/resume commands will be added here after the new runner and configuration exist and pass recovery checks; existing completed-run commands are not reused as a new experiment. Record actual tokens, seed, time, stops, checkpoint hashes and decisions in the phase journal when execution begins.
