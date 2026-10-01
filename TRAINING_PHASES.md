# Training phase journal

Documentation policy established at the user's request on **2026-09-29 (KST)**. Each actual training phase has an objective, ancestry, recipe, status and measured result. A launch is not a completed training result.

**Current scope (2026-10-01):** V2 is the selected model ([checkpoints/v2_best](checkpoints/v2_best/)) and the only active pipeline: [V2 training record](BEST_CHECKPOINT_TRAINING.md), [current pipeline](CURRENT_PIPELINE.md), [unsuccessful approaches](FAILED_APPROACHES.md). V1 checkpoints ([best_sft](checkpoints/best_sft/), [competition_base](checkpoints/competition_base/)) are preserved as baselines.

**Scope on 2026-09-29 (superseded):** retain the completed V1 scratch-pretraining and selected full-SFT model construction ([V1 record](docs/history/V1_BEST_CHECKPOINT_TRAINING.md), [V1 pipeline](docs/history/V1_CURRENT_PIPELINE.md)).

**Cleanup is not a new training phase.** Superseded active code/configuration was removed; original exports and measured evidence remain intact. [Removal inventory](reports/cleanup/2026-09-29/removed_files.json). The proposed two-arm 250M-token continuation was not launched and has been withdrawn from the active roadmap.

The entries below document historical experiments. Launch commands and experiment-source paths in historical artifacts may refer to retired code; current execution instructions are in [CURRENT_PIPELINE.md](CURRENT_PIPELINE.md) (V2).

## Existing lineage

These entries summarize already completed work from its original evidence; they do not represent new runs.

| Phase | Starting point and completed work | Outcome and evidence |
| --- | --- | --- |
| Scratch pretraining | Random initialization; 999,948,288 targets | [Completed 1B run](runs/baseline_1b/status.json) |
| Continuation to 12B | 1B base; 11,000,086,528 additional targets | [Completed 12B run](runs/continuation_12b/status.json) |
| Continuation to 13B | 12B base; 1,000,013,824 additional targets | [13B phase assessment](artifacts/phase_evaluation_v1/REPORT.md); strongest evaluated pretraining endpoint |
| Initial 13B SFT | Two full-parameter SFT arms from the 13B base | Raw-language retention failed; [base retained](artifacts/posttraining_13b_v1/REPORT.md) |
| DPO follow-up | 1,359 preference pairs; 85 updates | Mixed results; [base retained](reports/posttraining/2026-09-28_dpo_followup.md) |
| Upgraded SFT study | Nine pilots, six full comparisons, six controls | [Selected update 105](reports/posttraining/2026-09-29_sft_method_and_preservation.md); task-panel gain, mixed required benchmarks |
| Upgraded DPO / RL | Four DPO pilots, three full runs, three positive-only controls, four RL pilots | [No improvement over selected SFT parents](reports/posttraining/2026-09-29_upgraded_pipeline_completion.md); extended RL skipped |

Both the [13B-token base](artifacts/preserved_checkpoints/2026-09-29_base_and_sft/base_13b/) and the [selected SFT](artifacts/preserved_checkpoints/2026-09-29_base_and_sft/sft_selected/) are preserved. Both have 46,346,752 parameters.

## Completed phase: benchmark-oriented SFT from the base

**Phase ID:** `sft_benchmark_20260929`. **Objective:** competition benchmark performance, as clarified by the user. **Launched:** 2026-09-29 01:13:56 KST in persistent session `gibc-sft-benchmark-20260929`; real SFT updates began after GPU allocation at 01:14:36 KST. Input checks passed: 124 planned updates / 1,013,826 response targets. [Launch verification](artifacts/sft_benchmark_20260929/launch_verified.json) records completed updates, finite training metrics, and the original base identity. Consult the linked live status for the latest observation. [Validation](artifacts/sft_benchmark_20260929/validation.json): 44 tests passed, one optional GPU test skipped.

- [Phase design](FAILED_APPROACHES.md#a08--knowledge-heavy-sft-with-50-replay-loss)
- [Configuration](artifacts/sft_benchmark_20260929/registration.json)
- [Live phase report](artifacts/sft_benchmark_20260929/REPORT.md)
- [Live status](artifacts/sft_benchmark_20260929/status.json)
- [Dated event history](artifacts/sft_benchmark_20260929/events.jsonl)

The sequence is input verification → bounded SFT with retention checks → frozen development choice → required benchmark reporting → documented decision. Later preference training or RL depends on this phase's evidence. The earlier confirmation panel is already consumed; this experiment makes no new independent-confirmation claim and does not automatically replace either preserved checkpoint.

**01:18 KST observation:** the update-31 checkpoint improved the science/commonsense development mean from 16.07% to 19.54%, but exceeded the registered 0.01-nat raw-source retention limit (maximum +0.013292 nats). Remaining updates stopped automatically after 254,759 response targets and 272,468 replay targets. The frozen decision retains the base. Required benchmarks are running on this checkpoint as a diagnostic; [comparisons](artifacts/sft_benchmark_20260929/comparisons.json) and [selection](artifacts/sft_benchmark_20260929/selection.json) record the evidence.

**01:21 KST reporting recovery:** attempt 001 failed before scoring because the new reporting wrapper omitted a metadata field required by the existing evaluator. A separate compatibility record supplies that field without changing training or the frozen decision. Six phase/compatibility tests passed. [Recovery details and resume command](FAILED_APPROACHES.md#engineering-mistakes-to-avoid-repeating). The failed attempt and correction are retained.

**Training phase completed:** [SFT completion record](artifacts/sft_benchmark_20260929/TRAINING_COMPLETED.md) and [independent verification](artifacts/sft_benchmark_20260929/training_phase_verified.json) confirm the early stop at update 31, candidate weight identity, the retain-base decision, and unchanged checksums for all 15 preserved files. Diagnostic benchmark reporting is queued behind the separate adapter experiment in persistent session `gibc-sft-benchmark-report`; the live report will record its outcome. This phase will perform no more training updates.

**Final outcome, 01:50:09 KST:** diagnostic reporting completed after the recorded metadata repair. Raw accuracies were HellaSwag 28.27%, ARC-Easy 45.92%, PIQA 61.92%, and WinoGrande 50.51%; all were below the corresponding base point estimates. WikiText-103 perplexity was 25.7293 versus 25.5198 for the base. Normalized metrics are in the [final phase report](artifacts/sft_benchmark_20260929/REPORT.md). The decision remains **retain base**. [Completion receipt](artifacts/sft_benchmark_20260929/completed.json) confirms offline reload and preserved-file verification. No additional SFT, DPO, or RL follows this rejected pilot automatically.

## Completed phase: output-adapter SFT from the preserved base

**Phase ID:** `lm_head_lora_v2`. The user's request to run training after preparation was checked against the live workspace on **2026-09-29 at 01:25:56 KST**. This separately registered experiment was already training on the H100, so no duplicate was launched.

Preparation passed: 60 source hashes, all training/fresh-panel checksums, nine CPU behavior tests, exact H100 interrupted/resumed recovery, and preservation of all 15 base/SFT files. Three learning-rate trials had completed 105 updates each; the first of two additional seed repetitions was running at update 22/105. Each trial starts from the original frozen base and trains only the rank-8 output adapter. This is separate from the stopped full-parameter SFT pilot above.

- [Phase record, recipe, limits, and resume command](artifacts/lm_head_lora_v2/TRAINING_PHASE.md)
- [Readiness and active-training verification](artifacts/lm_head_lora_v2/training_readiness_verified.json)
- [Live workflow status](artifacts/lm_head_lora_v2/status.json)
- [Persistent launcher and log location](artifacts/lm_head_lora_v2/queue_status.json)
- [Registered original design](FAILED_APPROACHES.md#a07--output-only-lm-head-adaptation)

The existing runner continues through seed repetitions, one frozen fresh-confirmation comparison, required benchmark reporting, and its final report. These launch/progress observations do not claim improvement. Both previously preserved checkpoints remain intact.

**Final outcome, 01:40:29 KST:** all five runs completed 105 updates each (three learning-rate trials and two seed repetitions), followed by fresh confirmation and required benchmarks. The fixed adapter candidate scored 15.36% on the fresh science/commonsense panel versus 15.47% for the base: -0.11 percentage points, paired 95% interval [-1.39, +1.30]. Retention passed, but improved knowledge accuracy was not demonstrated. Required benchmark changes were small and mixed: ARC-Easy 46.72% versus 46.59% for the base; PIQA tied at 62.08%; HellaSwag and WinoGrande were slightly lower; WikiText-103 perplexity was 25.5846 versus 25.5198. No checkpoint or release was replaced.

- [Completed adapter report](artifacts/lm_head_lora_v2/REPORT.md)
- [Confirmation evidence](artifacts/lm_head_lora_v2/confirmation_summary.json)
- [Benchmark results](artifacts/lm_head_lora_v2/official/summary.json)
- [Compute ledger](artifacts/lm_head_lora_v2/COMPUTE_LEDGER.json)
- [Post-run preservation verification](artifacts/lm_head_lora_v2/preservation_verified_after.json)

Next decision: retain the base for competition use. Any further training should be a new, documented experiment informed by these negative/mixed results; neither result supports automatic escalation to DPO/RL.

## Choice-discrimination phase: choice_discrimination_20260929_v1/preparation

Registered 2026-09-29T02:39:28.761825+00:00 before execution. Freeze audited data, evaluation memberships and three matched batch plans; no model scoring.

[Phase report](artifacts/choice_discrimination_20260929_v1/phases/preparation/REPORT.md) · [Registration](artifacts/choice_discrimination_20260929_v1/phases/preparation/registration.json)

## Choice-discrimination phase: choice_discrimination_20260929_v1/preflight

Registered 2026-09-29T02:56:15.033108+00:00 before execution. Verify CPU/GPU mathematics, recoverability, actual export loader and unchanged official reporting preparation.

[Phase report](artifacts/choice_discrimination_20260929_v1/phases/preflight/REPORT.md) · [Registration](artifacts/choice_discrimination_20260929_v1/phases/preflight/registration.json)

## Choice-discrimination phase: choice_discrimination_20260929_v1/reference_development

Registered 2026-09-29T03:03:52.937505+00:00 before execution. Score the frozen parent and scratch base on development only.

[Phase report](artifacts/choice_discrimination_20260929_v1/phases/reference_development/REPORT.md) · [Registration](artifacts/choice_discrimination_20260929_v1/phases/reference_development/registration.json)

## Choice-discrimination phase: choice_discrimination_20260929_v1/ce_control_seed20260929

Registered 2026-09-29T03:28:58.831682+00:00 before execution. Train ce_control at seed 20260929 for exactly 32 eligible updates, with development guards at 8/16/24/32.

[Phase report](artifacts/choice_discrimination_20260929_v1/phases/ce_control_seed20260929/REPORT.md) · [Registration](artifacts/choice_discrimination_20260929_v1/phases/ce_control_seed20260929/registration.json)

## Choice-discrimination phase: choice_discrimination_20260929_v1/choice_candidate_seed20260929

Registered 2026-09-29T03:32:38.649334+00:00 before execution. Conditional registered study phase; prerequisite gate did not pass.

[Phase report](artifacts/choice_discrimination_20260929_v1/phases/choice_candidate_seed20260929/REPORT.md) · [Registration](artifacts/choice_discrimination_20260929_v1/phases/choice_candidate_seed20260929/registration.json)

## Choice-discrimination phase: choice_discrimination_20260929_v1/ce_control_seed20260930

Registered 2026-09-29T03:32:38.653350+00:00 before execution. Conditional registered study phase; prerequisite gate did not pass.

[Phase report](artifacts/choice_discrimination_20260929_v1/phases/ce_control_seed20260930/REPORT.md) · [Registration](artifacts/choice_discrimination_20260929_v1/phases/ce_control_seed20260930/registration.json)

## Choice-discrimination phase: choice_discrimination_20260929_v1/choice_candidate_seed20260930

Registered 2026-09-29T03:32:38.657324+00:00 before execution. Conditional registered study phase; prerequisite gate did not pass.

[Phase report](artifacts/choice_discrimination_20260929_v1/phases/choice_candidate_seed20260930/REPORT.md) · [Registration](artifacts/choice_discrimination_20260929_v1/phases/choice_candidate_seed20260930/registration.json)

## Choice-discrimination phase: choice_discrimination_20260929_v1/ce_control_seed20261001

Registered 2026-09-29T03:32:38.661252+00:00 before execution. Conditional registered study phase; prerequisite gate did not pass.

[Phase report](artifacts/choice_discrimination_20260929_v1/phases/ce_control_seed20261001/REPORT.md) · [Registration](artifacts/choice_discrimination_20260929_v1/phases/ce_control_seed20261001/registration.json)

## Choice-discrimination phase: choice_discrimination_20260929_v1/choice_candidate_seed20261001

Registered 2026-09-29T03:32:38.665143+00:00 before execution. Conditional registered study phase; prerequisite gate did not pass.

[Phase report](artifacts/choice_discrimination_20260929_v1/phases/choice_candidate_seed20261001/REPORT.md) · [Registration](artifacts/choice_discrimination_20260929_v1/phases/choice_candidate_seed20261001/registration.json)

## Choice-discrimination phase: choice_discrimination_20260929_v1/confirmation

Registered 2026-09-29T03:32:38.669324+00:00 before execution. Conditional registered study phase; prerequisite gate did not pass.

[Phase report](artifacts/choice_discrimination_20260929_v1/phases/confirmation/REPORT.md) · [Registration](artifacts/choice_discrimination_20260929_v1/phases/confirmation/registration.json)

## Choice-discrimination phase: choice_discrimination_20260929_v1/official_reporting

Registered 2026-09-29T03:32:38.673313+00:00 before execution. Conditional registered study phase; prerequisite gate did not pass.

[Phase report](artifacts/choice_discrimination_20260929_v1/phases/official_reporting/REPORT.md) · [Registration](artifacts/choice_discrimination_20260929_v1/phases/official_reporting/registration.json)

**Choice-study final outcome, 2026-09-29T03:39:20.081224+00:00:** stopped at pilot control update 8 after 65,746 response targets and 32,760 replay targets. Historical science instruction accuracy declined 9% → 4%, beyond its two-point guard. Candidate, replications, confirmation and official reporting were skipped. Total accounted GPU time: 31.14 minutes. [Final verified report](artifacts/choice_discrimination_20260929_v1/FINAL_REPORT.md). Existing checkpoints and the original selected-SFT recipe remain unchanged; no automatic follow-up.

## Choice-discrimination phase: choice_discrimination_20260929_v1/post_study_verification

Registered 2026-09-29T03:41:27.622255+00:00 before execution. CPU-only final audit and defensive recovery checks; no study training or evaluation resumes.

[Phase report](artifacts/choice_discrimination_20260929_v1/phases/post_study_verification/REPORT.md) · [Registration](artifacts/choice_discrimination_20260929_v1/phases/post_study_verification/registration.json)

## Data-quality mid-training phase: midtrain_quality_20260929_v1/preparation

Registered 2026-09-29T04:46:59.788342+00:00 before execution. Scan the unused continuation span, freeze the held-out panel and write both arms' shards at fixed 60/30/10 shares. CPU only; no model scores.

[Phase report](artifacts/midtrain_quality_20260929_v1/phases/preparation/REPORT.md) · [Registration](artifacts/midtrain_quality_20260929_v1/phases/preparation/registration.json)

## Revised small RL study phase: rl_revised_20260929_v1/preflight

Registered 2026-09-29T04:52:17.147008+00:00 before execution. Run the study tests, then sample a few training prompts from the parent and check behavior/training log-probability agreement with a no-update objective pass.

[Phase report](artifacts/rl_revised_20260929_v1/phases/preflight/REPORT.md) · [Registration](artifacts/rl_revised_20260929_v1/phases/preflight/registration.json)

## Revised small RL study phase: rl_revised_20260929_v1/audit

Registered 2026-09-29T04:52:38.584872+00:00 before execution. No-update rollout audit: 384 training prompts x 8 samples x at most 64 tokens.

[Phase report](artifacts/rl_revised_20260929_v1/phases/audit/REPORT.md) · [Registration](artifacts/rl_revised_20260929_v1/phases/audit/registration.json)

## Revised small RL study phase: rl_revised_20260929_v1/reference_development

Registered 2026-09-29T04:53:13.816292+00:00 before execution. Score the selected SFT parent and the 13B base on development panels.

[Phase report](artifacts/rl_revised_20260929_v1/phases/reference_development/REPORT.md) · [Registration](artifacts/rl_revised_20260929_v1/phases/reference_development/registration.json)

**Data-quality mid-training v1 outcome, 2026-09-29:** preparation failed on CPU before any GPU work or training. The curated arm's per-host document cap also applied to Wikipedia, where every article shares one host, so the arm could not reach its token target. Superseded by `midtrain_quality_20260929_v2`, which applies the cap only to edu and dclm and is otherwise identical. [v1 preparation report](artifacts/midtrain_quality_20260929_v1/phases/preparation/REPORT.md)

## Revised small RL study phase: rl_revised_20260929_v1/train_token_mean

Registered 2026-09-29T05:19:29.084773+00:00 before execution. Train the token_mean arm from the selected SFT for at most 50 actual updates with deficit-based collection; development checks at [25, 50].

[Phase report](artifacts/rl_revised_20260929_v1/phases/train_token_mean/REPORT.md) · [Registration](artifacts/rl_revised_20260929_v1/phases/train_token_mean/registration.json)

## Revised small RL study phase: rl_revised_20260929_v1/train_prompt_mean

Registered 2026-09-29T05:28:11.856704+00:00 before execution. Train the prompt_mean arm from the selected SFT for at most 50 actual updates with deficit-based collection; development checks at [25, 50].

[Phase report](artifacts/rl_revised_20260929_v1/phases/train_prompt_mean/REPORT.md) · [Registration](artifacts/rl_revised_20260929_v1/phases/train_prompt_mean/registration.json)

## Data-quality mid-training phase: midtrain_quality_20260929_v2/preparation

Registered 2026-09-29T05:36:58.520647+00:00 before execution. Scan the unused continuation span, freeze the held-out panel and write both arms' shards at fixed 60/30/10 shares. CPU only; no model scores.

[Phase report](artifacts/midtrain_quality_20260929_v2/phases/preparation/REPORT.md) · [Registration](artifacts/midtrain_quality_20260929_v2/phases/preparation/registration.json)

## Data-quality mid-training phase: midtrain_quality_20260929_v2/preflight

Registered 2026-09-29T05:42:13.691810+00:00 before execution. Run the study tests, verify both arms' shards and the inherited checkpoint, and take one no-update forward/backward on each arm's first batch.

[Phase report](artifacts/midtrain_quality_20260929_v2/phases/preflight/REPORT.md) · [Registration](artifacts/midtrain_quality_20260929_v2/phases/preflight/registration.json)

## Data-quality mid-training phase: midtrain_quality_20260929_v2/reference_development

Registered 2026-09-29T05:42:22.555463+00:00 before execution. Score the preserved 13B base on the development panels.

[Phase report](artifacts/midtrain_quality_20260929_v2/phases/reference_development/REPORT.md) · [Registration](artifacts/midtrain_quality_20260929_v2/phases/reference_development/registration.json)

## Data-quality mid-training phase: midtrain_quality_20260929_v2/train_random

Registered 2026-09-29T05:43:41.793361+00:00 before execution. Continue the 13B base on the random arm for 977 updates (64,028,672 targets) at fixed source shares, then score development.

[Phase report](artifacts/midtrain_quality_20260929_v2/phases/train_random/REPORT.md) · [Registration](artifacts/midtrain_quality_20260929_v2/phases/train_random/registration.json)

## Data-quality mid-training phase: midtrain_quality_20260929_v2/train_curated

Registered 2026-09-29T05:43:42.392772+00:00 before execution. Registered arm; an earlier stop prevented it.

[Phase report](artifacts/midtrain_quality_20260929_v2/phases/train_curated/REPORT.md) · [Registration](artifacts/midtrain_quality_20260929_v2/phases/train_curated/registration.json)

**Data-quality mid-training v2 outcome, 2026-09-29:** preparation, preflight and base scoring completed. The random arm then stopped at trainer construction, before any optimizer update: the retained trainer requires a continuation to keep the lineage seed (20260923), and v2 used 20260929. The curated arm was skipped. Superseded by `midtrain_quality_20260929_v3` with seed 20260923 and otherwise identical. [v2 random-arm report](artifacts/midtrain_quality_20260929_v2/phases/train_random/REPORT.md)

## Data-quality mid-training phase: midtrain_quality_20260929_v3/preparation

Registered 2026-09-29T05:45:23.270434+00:00 before execution. Scan the unused continuation span, freeze the held-out panel and write both arms' shards at fixed 60/30/10 shares. CPU only; no model scores.

[Phase report](artifacts/midtrain_quality_20260929_v3/phases/preparation/REPORT.md) · [Registration](artifacts/midtrain_quality_20260929_v3/phases/preparation/registration.json)

## Data-quality mid-training phase: midtrain_quality_20260929_v3/preflight

Registered 2026-09-29T05:50:44.923535+00:00 before execution. Run the study tests, verify both arms' shards and the inherited checkpoint, and take one no-update forward/backward on each arm's first batch.

[Phase report](artifacts/midtrain_quality_20260929_v3/phases/preflight/REPORT.md) · [Registration](artifacts/midtrain_quality_20260929_v3/phases/preflight/registration.json)

## Data-quality mid-training phase: midtrain_quality_20260929_v3/reference_development

Registered 2026-09-29T05:50:53.709922+00:00 before execution. Score the preserved 13B base on the development panels.

[Phase report](artifacts/midtrain_quality_20260929_v3/phases/reference_development/REPORT.md) · [Registration](artifacts/midtrain_quality_20260929_v3/phases/reference_development/registration.json)

## Data-quality mid-training phase: midtrain_quality_20260929_v3/train_random

Registered 2026-09-29T05:52:11.323553+00:00 before execution. Continue the 13B base on the random arm for 977 updates (64,028,672 targets) at fixed source shares, then score development.

[Phase report](artifacts/midtrain_quality_20260929_v3/phases/train_random/REPORT.md) · [Registration](artifacts/midtrain_quality_20260929_v3/phases/train_random/registration.json)

## Data-quality mid-training phase: midtrain_quality_20260929_v3/train_curated

Registered 2026-09-29T06:01:31.265735+00:00 before execution. Continue the 13B base on the curated arm for 977 updates (64,028,672 targets) at fixed source shares, then score development.

[Phase report](artifacts/midtrain_quality_20260929_v3/phases/train_curated/REPORT.md) · [Registration](artifacts/midtrain_quality_20260929_v3/phases/train_curated/registration.json)

**Data-quality mid-training v3 final outcome, 2026-09-29:** both arms completed 977 updates (64,028,672 targets each, 60/30/10). Decision `not_established`. Curated minus random science/commonsense accuracy was −0.20 points (one-sided 95% bounds −0.74 to +0.31), and curated held-out NLL was worse on every source (+0.0015 edu, +0.0047 dclm, +0.0102 wiki; wiki exceeds the +0.01 limit). Neither arm changed accuracy against the base beyond noise. No checkpoint replaced; no SFT descendant. [Report](artifacts/midtrain_quality_20260929_v3/REPORT.md)

## V2 phase: v2_rich_prep_20260929 (preparation)

Registered 2026-09-29 (KST) before execution. This is a new, separate from-scratch approach approved by the user; it lives in [v2/](v2/README.md), and V1 checkpoints stay unchanged. **Objective:** build the V2 rich corpus (DCLM, FineWeb-Edu, benchmark-targeted slice, Wikipedia, StackExchange, Gutenberg) with a domain blocklist plus the existing 13-word exclusion and deduplication. Also run the tokenizer study (16k/20k versus V1; adopt at ≥2% compression gain) and freeze a development proxy panel from held-out benchmark **train** splits. CPU only; no model training; no parent checkpoint.

[Plan](v2/PLAN.md) · [Phase report](v2/phases/preparation/REPORT.md) · [Registration](v2/phases/preparation/registration.json) · [Status](v2/phases/preparation/status.json) · [Events](v2/phases/preparation/events.jsonl)

## V2 phase: v2_pilot_1b_20260929 (1B-token go/no-go pilot)

Registered 2026-09-29 (KST) before execution; it launches after the preparation manifest is complete. The run is a scratch 1B-token pilot on the V2 rich corpus using V1 `baseline_1b`'s exact schedule, compared with that V1 checkpoint on development data only: held-out benchmark **train** items and V2 development panels. **Gate:** the mean ARC-Easy/PIQA/HellaSwag/WinoGrande proxy accuracy must be at least V1 1B's. NO-GO stops V2 before the main run.

[Registration](v2/phases/pilot_1b/registration.json) · [Config](v2/configs/pilot_1b.json)

**V2 pilot outcome, 2026-09-29 18:32 KST:** the pilot completed 15,258 updates (999,948,288 targets) in 43 minutes. The gate decision is **GO** on the registered rule: mean ARC-E/PIQA/HellaSwag/WinoGrande proxy accuracy was 43.31% against 42.91% for V1 1B. The margin is within noise. PIQA was −2.7 and ARC-Easy +2.6 points. Bits/byte improved on every source except Wikipedia (1.145 vs 1.077). The V2 preparation phase completed at 17:43 KST with a 22.45B-token corpus (manifest `6749a2f1…6a68`). [Pilot report](v2/phases/pilot_1b/REPORT.md) · [Preparation report](v2/phases/preparation/REPORT.md)

## V2 phase: v2_main_20260929 (main scratch run), launched

The user approved launch at 20:13 KST on 2026-09-29, keeping Wikipedia at 10%. The run is scratch pretraining on the V2 rich corpus (manifest `6749a2f1…6a68`) with a warmup/constant/decay schedule. The decay start is still to be chosen: about 28B tokens for the Sep 30 internal target, up to the 45B ceiling for the Devpost cutoff. Development-only selection is frozen before the official evaluation, and V1 stays the fallback. [Report](v2/phases/main/REPORT.md) · [Registration](v2/phases/main/registration.json) · [Status](v2/phases/main/status.json)

**V2 main run, 2026-09-29 20:55 KST:** at the user's instruction the run continues to the 45B-token ceiling regardless of deadlines. The decay starts at step 617,981 and ends at step 686,645; expected completion is about Oct 1, 05:00 KST. One validation-labelling crash at step 10,000 was fixed and resumed from step 8,000. An auto-resume supervisor is active. [Report](v2/phases/main/REPORT.md)

## V2 phases registered: anneal A/B and final selection

Registered 2026-09-29 (KST) before execution.
- **`v2_anneal_quality_20260929`:** re-runs the main run's final 10% decay from its decay-start milestone with a quality-weighted mix (DCLM 25 / Edu 28 / targeted 25 / Wiki 12 / StackExchange 5 / books 5). It starts after the main run completes. [Registration](v2/phases/anneal_quality/registration.json)
- **`v2_final_selection_20260929`:** a development-only endpoint choice, frozen before official scores. It is followed by the pinned official evaluation and a pre-registered V1-vs-V2 rule: V2 wins if its four-task mean is > 46.955% and its WikiText-103 perplexity is ≤ 25.520; if the two disagree, the user decides. The rules await the user's confirmation. [Registration](v2/phases/final_selection/registration.json)

**V2 post-pretraining chain armed, 2026-09-29 ~23:37 KST:** the user confirmed both decision rules before any V2 final-model result existed and asked for the chain to be armed. `v2/run_after_main.sh` (tmux `gibc-v2-after-main`) waits for the main run to complete, then runs the decay A/B branch, the frozen development-only selection, official evaluation of both arms, the V1-vs-V2 rule and the figures. The evaluation path was rehearsed end to end on V1's 1B model. No V2 official score exists yet. [CPU-prep report](v2/reports/CPU_PREP_2026-09-29.md)

**V2 main run completed, 2026-10-01 04:11 KST:** 686,645 updates, 44,999,966,720 tokens, 31.9 training hours. Export `model.safetensors` SHA256 `9aac2111…28011bd`. Held-out monitor NLL fell 3.168 → 2.971 during the 10% decay. The decay A/B branch started automatically. [Report](v2/phases/main/REPORT.md)

**V2 outcome, 2026-10-01 07:54 KST:** the decay A/B branch completed (3.19 h) and was not selected (development proxy +0.07 pt < +0.5). The official pinned-protocol results for the selected V2 (`v2/runs/main/export`, SHA256 `9aac2111…`) were HellaSwag 28.85, ARC-Easy 47.39, PIQA 60.77, WinoGrande 51.46 (mean 47.12 vs V1 46.96) and WikiText-103 perplexity 23.85 vs 25.52. **The pre-registered rule selects V2 as the submission base.** The accuracy gain is within per-task standard errors; the perplexity gain is clear. [Final selection report](v2/phases/final_selection/REPORT.md) · [Decay A/B report](v2/phases/anneal_quality/REPORT.md) · FAILED_APPROACHES A13

**Code reorganization, 2026-10-01 (not a training phase):** at the user's request the active code now contains only the V2 pipeline. V1-only and V2 experiment-only code moved to `archive/` with a per-file hash inventory. The core modules are unchanged and still match V2's run record. `checkpoints/v2_best` was added and `AGENTS.md` updated to the V2 scope. Verification: 77 tests passed, `verify_v2.py` 10/10, the official-protocol preflight passed, and the GPU trainer smoke test succeeded. [Report](reports/cleanup/2026-10-01/REPORT.md)

## V2 phase: v2_task_tune_20261001 (fine-tuning on the benchmarks' train splits), registered

Registered 2026-10-01 ~10:20 KST on branch `task-tuning-20261001`, after the user confirmed that fine-tuning our own scratch-trained model is allowed (the Track 01 ban covers fine-tuning previously existing models) and approved the phase.
- **Parent:** V2 (`v2/runs/main/export`, SHA256 `9aac2111…`).
- **Data:** the 80% of the HellaSwag, ARC-Easy, ARC-Challenge, PIQA and WinoGrande train splits not held out by V2's existing hash split. That is 79,425 questions after dropping 554 that overlap official evaluation items. The data is formatted exactly as the pinned harness scores it: the formatter matches all 15,523 recorded official requests.
- **Recipe:** a GPT-1-style multiple-choice loss on the summed continuation log-likelihoods (no added parameters), plus a gold-sequence language-modelling loss and replay of V2 pretraining text. Three epochs; three arms run concurrently on GPU 0.
- **Selection:** development data only. Highest four-task mean on the held-out 20%, scored in the harness format, if it is at least +1.0 point over V2 and Wikipedia bits/byte is no more than 0.01 worse. Otherwise V2 is kept.
- **Amendment before launch:** smoke runs showed the registered loss damaged raw-language fit. The launch recipe uses temperature 10 in the choice softmax, learning rate 1e-5 or 2e-5, and replay weight 0.5 or 0.7.

[Report](v2/phases/task_tune/REPORT.md) · [Registration](v2/phases/task_tune/registration.json) · [Config](v2/phases/task_tune/config.json) · [Pre-launch smoke evidence](v2/phases/task_tune/prelaunch_smoke.json) · [Status](v2/phases/task_tune/status.json)
