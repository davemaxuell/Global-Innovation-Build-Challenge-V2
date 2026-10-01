# Retired approaches and lessons from completed experiments

**Recorded:** September 29, 2026. **Purpose:** prevent repetition of unsuccessful recipes while preserving the evidence behind each decision.

**Since 2026-10-01 the retained model is V2** ([BEST_CHECKPOINT_TRAINING.md](BEST_CHECKPOINT_TRAINING.md); `checkpoints/v2_best`, weight SHA256 `9aac2111b9ca6b06f4a12278f5213c17621ded643b7b54137987ce25028011bd`). Entries A01–A12 were run on V1, whose full-SFT checkpoint (weight SHA256 `defd0d91948ba8501a33b0c4996bf3bf37ebdf6e90338a05bab3a1773a2d7f72`, [V1 record](docs/history/V1_BEST_CHECKPOINT_TRAINING.md)) and 13B-token base remain preserved as baselines. A13 was run on V2. Code for these experiments is archived under `archive/` ([inventory](reports/cleanup/2026-10-01/moved_files.json)). This record describes bounded experiments on our 46M model; it does not establish that the underlying methods can never work.

## Decisions to carry forward

- Use the retained scratch-pretraining → balanced full-SFT with 30% replay-loss recipe as the reference implementation.
- Do not rerun the retired recipes below or restore their automatic launchers as routine “next phases.”
- Compare a descendant with its immediate parent as well as the base. A descendant can beat the base and still undo gains made by SFT.
- Keep instruction-format learning, knowledge accuracy, raw-language retention and competition benchmarks distinct. Improvement on one does not demonstrate improvement on all of them.
- A materially revised future experiment needs a changed, testable hypothesis, a bounded budget and fresh evaluation evidence. Reusing consumed confirmation data is not a new confirmation.

## A01 — Early human-only SFT without raw replay

**Executed:** September 28, 2026; parent = original 13B base. Full-parameter SFT, peak LR `3e-5`, one pass, approximately 8,192 response targets/update, microbatch 8. Completed **131 updates / 1,077,789 supervised targets** using the human-demonstration split.

**Observed:** instruction exact accuracy was **0.25%** versus **0.00%** for the base on the narrow procedural panel. Response NLL improved from **2.634261 to 2.255836**, while fixed-mixture raw NLL rose from **2.908931 to 2.970133**. The instruction-improvement guard and all five raw-source retention guards were exceeded.

**Decision:** retain the base. Do not repeat this high-LR, no-replay recipe expecting response-loss improvement alone to establish instruction or benchmark gains.

**Interpretation:** lower teacher-forced response loss did not translate into the measured task improvement. These results do not isolate which data or optimizer choice caused the outcome.

Evidence: [phase report](artifacts/posttraining_13b_v1/REPORT.md), [actual training counts](artifacts/posttraining_13b_v1/training/sft_human/status.json).

## A02 — Early human/procedural SFT without raw replay

**Executed:** same 13B parent, peak LR `3e-5`, one pass, full-parameter updates and no raw replay. Completed **131 updates / 1,077,749 supervised targets** on the mixed instruction split.

**Observed:** procedural instruction exact accuracy reached **49.10%**, but fixed-mixture raw NLL rose from **2.908931 to 2.970512**; all five raw-source retention guards were exceeded. WikiText-103 perplexity was **27.6288**, compared with **25.5198** for the base.

**Decision:** retain the base. Do not promote a model solely because it learns the procedural answer format while losing raw-language performance under the registered criterion.

**Interpretation:** this recipe produced measurable task adaptation. It did not satisfy the combined objective. The later retained full-SFT recipe used a different mixture, a lower peak LR and raw replay.

Evidence: [phase report](artifacts/posttraining_13b_v1/REPORT.md), [actual training counts](artifacts/posttraining_13b_v1/training/sft_mixed/status.json), [comparison metrics](reports/posttraining/2026-09-28_dpo_followup.md).

## A03 — Small procedural DPO follow-up

**Executed:** parent = A02 mixed-SFT checkpoint. Generated **60,000 own-model answers** from 7,500 prompts × 8 samples; retained **1,359 correct/incorrect preference pairs**. Standard DPO with frozen SFT reference, LR `1e-6`, beta `0.1`, one pass; **85 updates / 12,648 response/EOS targets**. Pair composition: classification 600, sorting 600, arithmetic 126, extraction 11, JSON 22.

**Observed:** procedural macro accuracy changed from **49.10% to 49.05%**: **−0.05 percentage points**, paired 95% interval **[−1.12, +0.98]**. Classification improved, extraction declined, and WikiText-103 perplexity remained **27.6009**. The model did not satisfy the parent-relative improvement and base-retention criteria.

**Decision:** no promotion. Do not repeat the same small, imbalanced procedural preference set or assume DPO repairs a parent's raw-language regression.

**Interpretation:** the experiment did not demonstrate aggregate improvement; it does not prove that preference optimization is ineffective with better data or another parent.

Evidence: [detailed report](reports/posttraining/2026-09-28_dpo_followup.md), [preference integrity](artifacts/posttraining_dpo_13b_v1/preference_integrity.json), [training integrity](artifacts/posttraining_dpo_13b_v1/training_integrity.json).

## A04 — Larger self-generated DPO after the retained SFT recipe

**Executed:** four pilots covering LR `{5e-7, 1e-6}` and beta `{0.05, 0.1}`, followed by three-seed runs of the shortlisted **LR `5e-7`, beta `0.1`** recipe with **19,204 preference pairs per seed**.

**Observed:** the SFT parents' three-domain development accuracies were **48.10%, 47.93%, 48.00%**. The full DPO endpoints scored **38.18%, 41.46%, 41.50%**. Parent-relative point-estimate changes were **−9.91, −6.46 and −6.50 points**. Even the earliest reported DPO endpoints had negative parent-relative gains. The full seed-20260928 endpoint also violated grounded-task retention checks.

**Decision:** retain SFT; no DPO descendant selected. Do not repeat this preference pool and recipe simply because its descendants still exceed the original base on a development proxy.

**Interpretation:** regression relative to the immediate parent is the relevant observation. The records do not identify a single causal explanation such as beta, preference difficulty or answer-length bias.

Evidence: [shortlisted recipe](artifacts/posttraining_sota_v1/dpo_shortlist.json), [seed and parent-relative comparisons](artifacts/posttraining_sota_v1/development_groups.json), [completed study](reports/posttraining/2026-09-29_upgraded_pipeline_completion.md).

## A05 — Positive-only SFT on the preference winners

**Executed:** three controls trained on the chosen answers from the same own-model preference pool used for A04.

**Observed:** full endpoint development accuracies were **47.57%, 47.37%, 47.70%** versus their SFT parents' **48.10%, 47.93%, 48.00%**. All three parent-relative changes were negative: approximately **−0.53, −0.56 and −0.29 points**. Earlier reported checkpoints also had negative point-estimate parent gains.

**Decision:** retain the SFT parents. Do not recycle this same winning-answer pool as an assumed quality improvement.

**Interpretation:** these small changes are not presented as proof of statistically significant harm. They did not provide the required positive evidence for advancement.

Evidence: [replicated comparisons](artifacts/posttraining_sota_v1/development_groups.json), [training and compute ledger](artifacts/posttraining_sota_v1/COMPUTE_LEDGER.json).

## A06 — Direct online RL pilots from SFT

**Executed:** four pilots crossing LR `{5e-7, 1e-6}` with KL coefficient `{0.02, 0.005}`; eight samples/prompt and 16 prompts/attempt. Each pilot completed **100 attempts**, with **43, 36, 32 and 36 actual optimizer updates** respectively. Attempts without informative relative reward did not update the model.

**Observed:** no pilot provided a retained parent-relative improvement. The registered workflow therefore did not advance to extended RL.

**Decision:** retain SFT. Do not equate attempted RL steps with optimizer updates, or extend the same reward/task distribution just because a pilot completed.

**Interpretation:** many attempts lacked usable update signal. This is evidence about this model/task/reward combination, not a general comparison of RL algorithms.

Evidence: [recorded decision](artifacts/posttraining_sota_v1/skipped_rl_sft.json), [actual update/token ledger](artifacts/posttraining_sota_v1/COMPUTE_LEDGER.json), [completed study](reports/posttraining/2026-09-29_upgraded_pipeline_completion.md).

## A07 — Output-only LM-head adaptation

**Executed:** original 13B base frozen; rank-8, alpha-8 output adapter with **135,168 trainable parameters**, **46,481,920 total parameters**. Matched the retained SFT's first 105 updates and data order. Tried LR `{1e-4, 3e-4, 1e-3}`, then two additional seeds of the retained `1e-4` recipe. The larger learning rates exceeded raw-language retention limits.

**Observed:** fresh science/commonsense confirmation mean was **15.36%**, compared with **15.47%** for the base and **17.46%** for full SFT. Adapter-minus-base difference: **−0.11 points**, 95% interval **[−1.39, +1.30]**. Adapter-minus-full-SFT difference: **−2.10 points**, interval **[−5.25, +0.82]**. Neither interval establishes a reliable knowledge improvement. Exact-answer-plus-EOS generation was **0% in each of eight diagnostic families** for the adapter. Required benchmark changes were small and mixed.

**Decision:** keep full SFT for instruction use; no adapter replacement. Do not repeat the same rank-8 output-only recipe as a presumed substitute for full instruction tuning.

**Interpretation:** strict answer-plus-termination accuracy is a formatting/task metric, not proof that the model has no knowledge. The intervals are wide, and the earlier full-SFT recipe had more tuning opportunities.

Evidence: [full report](artifacts/lm_head_lora_v2/REPORT.md), [fresh confirmation and intervals](artifacts/lm_head_lora_v2/confirmation_summary.json), [fixed selection](artifacts/lm_head_lora_v2/frozen_selection.json).

## A08 — Knowledge-heavy SFT with 50% replay loss

**Executed:** original 13B base; 60% knowledge / 20% general / 15% grounded / 5% procedural response-target mix; 50% response loss plus 50% raw replay loss; peak LR `1e-5`, seed `20260928`. Planned 124 updates / 1,013,826 response targets. The registered retention stop occurred at **31 updates / 254,759 response targets / 272,468 replay targets**.

**Observed:** two-domain development knowledge accuracy rose from **16.07% to 19.54%**, but worst-source raw NLL increased **0.013292 nats**, exceeding this experiment's **0.01-nat** threshold. The diagnostic checkpoint's raw benchmark accuracies were HellaSwag **28.27%**, ARC-Easy **45.92%**, PIQA **61.92%**, and WinoGrande **50.51%**; all four point estimates were below the base. WikiText-103 perplexity was **25.7293** versus **25.5198**. Normalized ARC-Easy accuracy improved, so the result should not be described as worse on every reported metric.

**Decision:** retain the base reference and existing full-SFT checkpoint. Do not repeat this exact recipe or relax its threshold retrospectively to turn the same result into a success.

**Interpretation:** development knowledge gains did not satisfy the combined retention/benchmark objective. Its 0.01-nat guard was stricter than the retained SFT study's 0.02-nat guard; those decisions should not be compared as if the thresholds were identical.

Evidence: [phase report](artifacts/sft_benchmark_20260929/REPORT.md), [comparisons](artifacts/sft_benchmark_20260929/comparisons.json), [diagnostic benchmarks](artifacts/sft_benchmark_20260929/official/attempt_003/summary.json).

## What was not an unsuccessful executed method

| Item | Actual status |
| --- | --- |
| Retained balanced full SFT with 30% replay loss | Selected and confirmed for the registered instruction objective. Its benchmark changes were mixed; preserve both the SFT and base references. |
| Extended RL and RL after DPO | Not run after their prerequisite selection gates; no measured result for those extended phases. |
| Earlier 12B post-training route in `pipeline_v1` | SFT/DPO were disabled under the authorization record then in use. That is a historical execution decision, not a model-performance finding. Later own-model post-training authorization was recorded. |
| Proposed two-arm 250M-token continuation from the 13B base | Proposed only, never launched. Removed from the active roadmap during this cleanup; no performance result exists. |
| Adaptive pretraining controller | Not used in the selected model's lineage. Removing its code does not establish an experimental failure. |
| Fully untied output matrix | Not the head-adapter experiment. Untying would raise the existing architecture to 54,735,360 parameters, beyond the 50M cap. |

## Engineering mistakes to avoid repeating

1. **Tied-head confusion:** directly training the tied `lm_head.weight` also changes input embeddings. The output-only study correctly used an independent residual adapter to keep the original model frozen.
2. **Checkpoint-prefix schedule changes:** update 105 used the full 209-update schedule denominator. Shortening the schedule to 105 updates would train a different model.
3. **Proxy-score overstatement:** generated grounded/format tasks can dominate a task-panel improvement. Report knowledge domains and competition benchmarks separately.
4. **Metadata integration gaps:** the output-adapter harness needed correct `tie_weights` handling; the later benchmark reporter needed its required metadata fields. Both were repaired and their evaluations completed. Smoke-test the actual loader/report path before a larger run.
5. **Mutable or reused evidence:** keep model/data hashes, actual update/token counts and consumed confirmation identities. A renamed panel or a changed promotion rule does not create independent evidence.

The active source tree now contains the retained model construction, necessary data preparation and evaluation code. Original checkpoint exports, dataset manifests, measured results and frozen source snapshots remain historical evidence. They are not an instruction to relaunch retired studies. [Cleanup inventory](reports/cleanup/2026-09-29/removed_files.json).

## A09 — Choice-discrimination study (choice_discrimination_20260929_v1)

Outcome: **failed**. Stop: `execution_or_retention_stop`. [Full evidence and actual counts](artifacts/choice_discrimination_20260929_v1/REPORT.md). No selected checkpoint was replaced and no further recipe is scheduled. An incomplete or unlaunched arm is not evidence that the choice objective failed.

A09 measured detail: the matched CE/replay control stopped at **update 8**, with **65,746 response targets / 32,760 replay targets**. Historical science instruction generation (correct answer plus EOS) declined **9/100 → 4/100**, exceeding the registered two-point guard. Its fresh natural primary score was **43.55%** versus **44.15%** for the parent and **41.90%** for the base. All raw NLL guards passed. **The choice-loss candidate was not trained**, so this outcome provides no completed paired test of the added loss. Confirmation and new official scores remain unopened. [Final verified report](artifacts/choice_discrimination_20260929_v1/FINAL_REPORT.md).

## A11 — Revised small RL study (rl_revised_20260929_v1)

Outcome: **no_difference_established**. Stop: `matched_update_comparison`. [Evidence and actual counts](artifacts/rl_revised_20260929_v1/REPORT.md). No selected checkpoint was replaced and no further recipe is scheduled automatically.

## A10 — Data-quality mid-training (midtrain_quality_20260929_v2): implementation stop, not a result

Outcome: **failed before its first optimizer update**. The retained trainer rejected the parent because the study's run seed (20260929) differed from the continuation lineage seed (20260923), which the trainer stores in the model configuration. No arm trained; this is not evidence about data-quality selection. Superseded by the v3 run with seed 20260923 (see A12). [Evidence](artifacts/midtrain_quality_20260929_v2/REPORT.md).

## A12 — Data-quality mid-training (midtrain_quality_20260929_v3)

**Executed:** from the 13B base with its AdamW state, two arms of 977 updates (64,028,672 targets each, 60/30/10 edu/dclm/wiki) from the same unused corpus span: an ordinary seeded sample versus documents passing Gopher-style quality/repetition filters and ranked by a fixed coherence score (length mix matched, per-host cap on web sources).

**Observed:** curated minus random science/commonsense continuation accuracy **−0.20 points** (one-sided 95% bounds −0.74 / +0.31). Curated held-out NLL was worse on every source: **+0.0015 edu, +0.0047 dclm, +0.0102 wiki** (all bounds above zero). Neither arm changed accuracy against the base beyond noise (random +0.15, curated −0.05 points). Retention guards against the base passed for both.

**Decision:** `not_established`; no checkpoint replaced, no SFT descendant.

**Interpretation:** at this scale and budget, stricter heuristic document selection within already-filtered sources narrowed the distribution (worse general held-out NLL) without improving knowledge-panel accuracy. It does not rule out other selection signals or larger budgets. Evidence: [report](artifacts/midtrain_quality_20260929_v3/REPORT.md), [decision](artifacts/midtrain_quality_20260929_v3/decision.json).

## A13: Quality-weighted decay mix for V2 (v2_anneal_quality_20260929)

**Executed:** 2026-10-01. The V2 main run's final 10% decay (68,664 updates, 4.5B tokens) was re-run from its decay-start milestone with more targeted/Edu/Wikipedia text (25/28/12% vs 15/30/10%) and less DCLM. Everything else was inherited bit-exactly (model, optimizer, data cursors, RNG, LR schedule).

**Observed:** the development proxy was +0.07 points (the registered threshold was +0.5); Wikipedia bits/byte −0.0046. Official, reported after the frozen choice: four-task mean 47.08% vs 47.12%; WikiText-103 perplexity 23.56 vs 23.85.

**Decision:** not selected; the standard-decay model stands. Do not repeat this mix shift expecting benchmark gains at this scale.

**Interpretation:** a controlled A/B with no established effect. It does not rule out other annealing data (for example, sources absent from the corpus) or larger models. Evidence: [report](v2/phases/anneal_quality/REPORT.md), [selection record](v2/phases/final_selection/selection.json).

## A14: Fine-tuning V2 on the benchmarks' train splits (v2_task_tune_20261001)

**Executed:** 2026-10-01, on branch `task-tuning-20261001`.
- **Data:** V2 was fine-tuned on 79,425 training-split questions from HellaSwag, ARC-Easy/Challenge, PIQA and WinoGrande. These are the 80% not held out by V2's existing hash split, after dropping 554 that overlap official evaluation items. Each question was in the exact pinned-harness scoring format.
- **Loss:** multiple-choice cross-entropy over summed continuation log-likelihoods with temperature 10, plus 0.5 × gold-sequence LM loss, plus replay of V2 pretraining text.
- **Arms:** LR 2e-5 with replay 0.5 or 0.7, and LR 1e-5 with replay 0.5. Each ran 3 epochs (7,449 updates).
- **Before launch:** the registered recipe was amended after smoke runs showed it hurt raw-language fit (no temperature, LR 6.25e-5, replay 0.3: Wikipedia bits/byte +0.081).

**Observed:**
- **Development panel:** every one of the 9 candidates gained 2.6–6.2 points on the held-out 20% of the training splits. Every one also worsened Wikipedia selection bits/byte by 0.012–0.028, beyond the registered +0.01 guard.
- **Official (reported after the frozen choice):** the best candidate scored HellaSwag 39.15 (+10.30), ARC-Easy 54.12 (+6.73), PIQA 65.34 (+4.57) and WinoGrande 52.41 (+0.95), for a mean of **52.76 vs 47.12**. WikiText-103 perplexity was **25.46 vs 23.85**.

**Decision:** not selected under the registered rule; V2 stays. Do not rerun these recipes expecting them to pass the same perplexity guard. More replay (0.7) and a lower LR (1e-5) reduced the bits/byte cost but did not bring it within the guard.

**Interpretation:** this is not a failure to improve the benchmarks; the accuracy gains are large and well outside the standard errors. The approach failed the registered combined objective because it costs raw-language fit. A materially different follow-up would need a new registration and the user's decision. One example is interpolating between V2 and the fine-tuned weights (WiSE-FT), with the mixing weight chosen on development data. Scores from this model are not comparable with pretraining-only models at face value and must be labelled as fine-tuned on the benchmarks' training splits. Evidence: [report](v2/phases/task_tune/REPORT.md), [selection](v2/phases/task_tune/selection.json), [official summary](v2/phases/task_tune/official/task_tune_lr2e-5_r0.5_epoch_3/attempt_3/summary.json).
