**Proposed experiment: supervised choice discrimination with retained language modeling**

**Date:** September 29, 2026. **State:** design complete; preparation, implementation and training have not started. This document formalizes the first option in the [MiMo investigation](2026-09-29_mimo_v2_6_training_options.md). The [machine-readable design](2026-09-29_choice_discrimination_protocol.json) is a research specification, not a runnable training configuration.

The experiment asks one question: **does adding supervised discrimination among gold-labeled answer choices improve unseen continuation accuracy beyond otherwise identical SFT with raw replay, without losing language-model quality?** The intervention is a choice-discrimination loss. Data preparation, ordinary supervised targets, raw replay, optimizer, initialization and schedule are shared by the control and candidate.

This is a different hypothesis from the recipes in [FAILED_APPROACHES.md](../../FAILED_APPROACHES.md). It does not establish that discrimination will work, or that the earlier failures had a single cause. A failed or inconclusive result ends this experiment under its registered budget.

**Why this question fits the project.** The competition protocol scores the probability of literal continuations; it does not ask our model to generate an answer letter or a reasoning transcript. Our earlier selected SFT teaches gold answers, but its objective does not directly compare the probability assigned to all annotated distractors. Historical DPO does compare sampled responses, through reference-relative preferences, and historical RL optimizes sampled reward. The proposed loss instead uses the dataset's known correct answer and complete provided choice set. It supplies a supervised discrimination signal even when the model cannot sample a correct answer.

MiMo motivates matching training to the task/interface and improving feedback quality. This exact loss is our adaptation, not a claimed MiMo recipe. The mechanism is also standard supervised classification over sequence scores; no algorithmic novelty claim is intended. The project's [frozen benchmark configuration](../../configs/evaluation.json) remains the definition of official evaluation.

| Recorded approach | Design constraint carried into this experiment |
| --- | --- |
| A01: human-only SFT, no replay | Retain raw replay and require task-accuracy evidence; lower response loss alone is insufficient. |
| A02: procedural SFT, no replay | Natural science/commonsense accuracy is primary; generated formatting or grounded-task gains cannot determine promotion. |
| A03: small, imbalanced procedural DPO | Use supplied gold labels and all annotated choices, with equal science/commonsense contribution to the added loss. |
| A04: larger self-generated DPO | Use an additive supervised choice loss without a DPO reference ratio or self-generated preference pool; compare against the immediate SFT parent. |
| A05: SFT on sampled winners | Use the original audited gold annotations, including examples the model gets wrong; no selection by model-generated winners. |
| A06: online RL pilots | No rollout, acceptance sampler or reward threshold is needed for the training signal. Optimizer updates are counted directly. |
| A07: output-only adapter | Update the complete existing 46,346,752-parameter model; preserve the tied embedding/head and add no parameters. |
| A08: knowledge-heavy SFT with 50% replay | Keep the successful response-bucket mixture and 30% replay coefficient. Test one added objective against a matched control, with prospective retention limits. |

Neither the withdrawn 250M-token continuation proposal nor retired trainers are reinstated. The successful model's training record stays unchanged. Reuse of successful training ingredients is intentional; reuse of consumed evaluation panels is disclosed separately below.

**Starting points and arms.** Both trained arms start from independent loads of `checkpoints/best_sft/model.safetensors`, SHA256 `defd0d91948ba8501a33b0c4996bf3bf37ebdf6e90338a05bab3a1773a2d7f72`. The preserved base, `checkpoints/competition_base/model.safetensors`, SHA256 `ffa75b07ec377167ab2e4d4263f2fa8374549f8592f9ce2e8166875cab1f38e7`, is an additional evaluation reference. Both hashes were checked while designing this experiment.

| Arm | Training objective | Role |
| --- | --- | --- |
| Parent | No update | Measures whether a descendant actually improves our current instruction checkpoint |
| Base | No update | Exposes cumulative changes relative to the scratch-pretrained reference |
| C: matched control | `0.70 L_response + 0.30 L_raw` | Measures the effects of shared data preparation and additional supervised exposure |
| T: choice candidate | `0.70 L_response + 0.30 L_raw + 0.05 L_choice` | Isolates adding choice discrimination |

The control and candidate retain exactly the same response/replay coefficients. The added coefficient is fixed at 0.05 for this study; there is no automatic coefficient or learning-rate search. This additive specification replaces the provisional coefficient allocation for the first experiment in the earlier research note.

For a context `x` with candidate continuations `a_1,...,a_K` and gold index `y`, define:

```text
s_j(theta) = sum over continuation tokens t of log p_theta(a_j,t | x, a_j,<t)
P_theta(j | x, choices) = exp(s_j) / sum_k exp(s_k)
ell_choice = -log P_theta(y | x, choices)

L_choice = 0.5 * mean_science(ell_choice)
         + 0.5 * mean_commonsense(ell_choice)
```

Use a numerically stable log-sum-exp implementation and temperature 1. Gold-response and raw-text losses are each means over their own loss-bearing tokens; choice loss is a mean over items within each domain. Coefficients therefore are not token fractions. Log each term and its gradient norm during training-only preflight; the fixed coefficients must not be reinterpreted as equal gradient contributions.

The derivative with respect to each candidate score is `P_theta(j) - 1[j=y]`. It raises the gold score relative to plausible alternatives while ordinary CE and raw replay continue to train the language distribution. A ranking gain can still arise mainly by lowering distractor scores, so record gold-answer NLL as well as accuracy. This is a measured risk, not a reason to assume the auxiliary loss preserves all capabilities.

**Training data identity and construction.** Use the existing protected training side as the ingredient pool:

| Ingredient | SHA256 |
| --- | --- |
| `data/posttraining/sota_v1/manifest.json` | `7bc4e6c72b6dc76819f1aa1e41611c9eeec94cf29cec76bd0a30f64ccbd666ee` |
| `data/posttraining/sota_v1/train.jsonl` | `1158c7bd4807638b00dc2c6b905162f99dd9435f48dd0941b3ff22479b6c957e` |
| `data/posttraining/sota_v1/replay.jsonl` | `8c6cac8b2d1bb94c609700431536259ce1c118cd016698907533a3f3554b5f25` |

A read-only count found **7,695 CommonsenseQA choice items across 1,193 source groups**, and **5,827 QASC choice items across 531 groups**, after excluding control/unbalanced/non-training rows. They carry original source IDs and the pinned source revisions already recorded in [the current configuration](../../configs/current_model.json). This establishes ingredient availability; it does not yet establish the final encoded data or batch-plan identity.

Prepare at most 1,024 distinct source items from each domain per seed: 32 science and 32 commonsense items per update for 32 updates. Use seeded sampling across source groups with a maximum of four selected choice items per group. Verify the quota after all length/overlap checks; insufficient yield stops preparation rather than silently duplicating items or changing the mixture. The same per-seed item list is used by C and T.

Recover each original question from its pinned source record using its source ID. Render it as a plain question followed by a fixed answer delimiter, without a choice menu, chat markers, an instruction to emit an answer letter, or supplied supporting facts. Score each actual answer string as a continuation. The gold continuation is also an ordinary supervised example in both arms, so candidate T does not receive extra gold answers hidden from C. The planner must include these anchor examples first, then fill the shared supervised batch to the registered response-target mixture:

```text
general instructions 30%; knowledge 35%; grounded 20%; procedural 15%
formatting-only supervision <= 5%
```

These shares are by gold response targets, using whole examples and a deterministic deficit-balancing plan. Science and commonsense each receive half of the auxiliary choice loss; that is a separate denominator. Record achieved shares and planned rounding before launch. Ordinary source-supplied explanation examples can remain in the shared response mixture, but closed-book choice contexts must not contain their answers or supporting facts.

An individual rendered gold-response record may appear once in a new run; each selected original choice item appears once in its auxiliary branch. Count shared source IDs across prompt variants and report previous exposure in the parent lineage. These ingredients are known training data, not newly discovered facts or independent confirmation. No self-generated winning-answer pool, external teacher output, or examples from the five competition datasets are introduced.

The scorer must reproduce the pinned context/continuation tokenization conventions, including delimiter handling. Exclude prompt and padding labels. Apply the causal shift once. Choice scores contain no appended EOS, answer label, or chat terminator. Actual response CE may supervise its real EOS; raw replay retains normal document boundaries. Reject incomplete overlength examples; maximum combined context remains 1,024 tokens and supervised response remains at most 128 tokens. WinoGrande-style suffix scoring is an evaluation diagnostic here, not an assertion that question-answer ranking implements its objective automatically.

Raw replay retains the existing 60% Edu / 30% DCLM / 10% Wikipedia target mixture. Its batches and cursors are identical between matched arms. Use the existing complete-document batching convention, log actual target counts, and never interpret a 30% loss coefficient as exactly 30% of processed tokens.

**Optimization and limits.** Initialize a fresh AdamW optimizer for each new arm, updating all existing unique parameters. Keep betas `(0.9, 0.95)`, epsilon `1e-8`, matrix weight decay 0.1, other weight decay zero, gradient clipping at 1.0, BF16 autocast, and FP32 weights/optimizer/loss reductions. Use the assigned H100 UUID from `configs/current_model.json` and the project's lock; four CPU threads.

| Setting | Fixed study value |
| --- | --- |
| Peak LR | `3e-6` |
| Schedule | 3% response-target warmup; cosine to `3e-7` over the entire newly prepared 32-update plan |
| Update budget | 32 actual optimizer updates per arm |
| Nominal response targets | Approximately 8,192/update; approximately 262,144/arm, with exact whole-example counts frozen in the plan |
| Choice groups | 64/update, 32/domain; 2,048 distinct item IDs/arm |
| Microbatch | Up to 8 response examples; choice candidates may be chunked with exactly equivalent gradient accumulation |
| Per-arm ceilings | 3,600 seconds including its development checks; 10M processed input-token positions across training/scoring forwards |
| Seed order | Pilot/deployment seed `20260929`; conditional replications `20260930`, `20261001` |
| Recovery checkpoints | Every 8 actual updates and on clean interruption |
| Development checks | Updates 8, 16, 24 and 32; only update 32 is eligible for advancement |
| Maximum trained arms | Six: control/candidate for three fixed seeds |
| Overall ceiling | Six GPU-hours for arms plus two GPU-hours for shared reference/confirmation/final scoring; eight assigned-GPU hours total |

For the nominal pilot comparison, the two arms together expose approximately 524,288 gold response targets. Do not add that total to either individual checkpoint's ancestry. Report actual response targets, replay targets, candidate-score token positions, processed/padded inputs, elapsed time and approximate FLOPs separately. C can score the same distractors without gradients for diagnostics, but T's additional backward computation still makes this a matched-exposure comparison, not an equal-FLOP claim.

The original selected model's 209-update schedule remains intact. This study starts a new optimizer and new 32-update schedule from its saved update-105 weights; it is not a reproduction with a shortened original schedule. Stops preserve the planned denominator. Resource exhaustion before update 32 produces an incomplete/non-qualifying arm; an earlier checkpoint cannot be substituted after inspecting its score.

**Fresh evaluation is a launch prerequisite.** The original SFT development/confirmation panels, head-adapter confirmation, later benchmark-SFT evaluation and competition results have already been observed. Their metrics can be disclosed as history and old instruction panels can supply labeled retention diagnostics. They cannot become the new confirmation set by renaming or resplitting.

Reserve source/fact-disjoint natural science and commonsense panels before candidate scoring. The earlier investigation identifies SciQ and Social IQa as candidate sources for review; use them only as evaluation ingredients in this study. Pin revisions, record license/provenance, exclude browsed public preview examples, check questions/supporting facts against historical training/evaluation identities, and retain the limits of semantic overlap detection. Do not claim either source is fresh before this audit. Target at least 500 items and 50 independent source/fact groups **per domain per panel** after exclusions; insufficient yield keeps the study unlaunchable. Additional natural narrative and coreference transfer panels are reported separately; generated tasks cannot substitute for the natural primary panels.

Prepare fresh raw-language panels for the five historical source strata: original web, original Wikipedia, continuation Edu, DCLM and continuation Wikipedia. Keep whole documents/groups out of the new training/replay data. Target at least 131,072 scored tokens and 50 documents per stratum per development/confirmation panel. Audit historical exposure before claiming they are unseen. A new file drawn from already trained stream positions is not a fresh held-out panel.

The primary score is `S = 0.5 * science raw continuation accuracy + 0.5 * commonsense raw continuation accuracy`. Report character-normalized accuracy alongside it, gold-answer NLL, and template/length strata. Development and confirmation use two fixed raw presentations with equal weighting per underlying item; prompt variants stay inside their source/fact group. Hold at least one presentation out of training. Treat the mean as an internal development metric, not an invented official competition composite.

For the pilot to advance to the two replication seeds, its update-32 candidate must achieve **at least +1 percentage point in S over C and over the immediate parent**, and at least match the base. It must also satisfy the development guards below. A smaller improvement is inconclusive under this study, and does not trigger a bigger budget or a different LR. Replications run the same frozen recipe at update 32; their paired mean improvement must reach the same thresholds, with no seed showing a negative primary change against its control or parent. Deployment seed remains `20260929`; no best-seed selection.

Use these prospective retention limits:

- Per raw-language stratum: candidate NLL increase at most **0.01 nats over the SFT parent**, and at most **0.02 nats over the base**. Development uses point estimates; confirmation requires each one-sided 95% bootstrap upper bound to meet the corresponding limit.
- Per natural primary domain and declared protected instruction/transfer domain: candidate accuracy decline at most **2 percentage points versus its SFT parent**. Development uses point estimates; fresh confirmation domains require their one-sided 95% lower bound to remain at or above −0.02. Previously consumed instruction diagnostics are explicitly descriptive point-estimate guards.
- Nonfinite loss/gradient stops immediately. A development guard breach stops that arm and makes it non-qualifying. If candidate T fails, do not automatically train the remaining seeds. A candidate whose paired control cannot reach a valid endpoint cannot support a choice-loss efficacy claim.

These rules apply prospectively. They do not relabel A08 or alter the successful SFT experiment's historical limits. A parent that already exceeds a fresh base-relative guard does not receive an exemption; the descendant must repair that gap to qualify under this design.

After development qualification, freeze the pilot candidate/control hashes and open confirmation once. Compute paired source/fact-group bootstrap differences, stratified by domain, with 10,000 replicates and seed `20260929`. Use the same resampled groups for all compared models and keep every variant of an item together. The fixed candidate must have at least +1 point versus C and parent, and positive differences versus C, parent and base with **one-sided Bonferroni-adjusted bootstrap lower bounds at quantile `0.05/3` above zero**. Also require all confirmation retention guards. These are approximate bootstrap inference rules, not a guarantee of exact coverage. Report ordinary two-sided 95% intervals as descriptive estimates. If precision is insufficient, record an inconclusive result; do not consume more confirmation examples adaptively.

Only then run the unchanged competition protocol for the frozen candidate and any predeclared comparison outputs. Report all required raw/normalized accuracies and WikiText-103 perplexity, with their historical exposure disclosure. Official scores do not reopen endpoint or seed selection. An official benchmark regression must be reported plainly; it cannot be hidden by a positive internal proxy. Any subsequent research choice informed by those scores is a new disclosed iteration.

**Interpretation of possible results.** If T beats C and the parent while preserving retention, the experiment supports the choice-loss hypothesis on the registered panels. If C and T improve similarly, extra data exposure or shared formatting may help, but the added loss is not established as the cause. If only ranking loss improves, task quality is not demonstrated. If knowledge accuracy improves while raw-language retention fails, retain the current references, as in the lesson from A08. None of these outcomes automatically schedules DPO, RL, more epochs, or an optimizer switch.

**Implementation and phase registration.** Use new data under `data/posttraining/choice_discrimination_20260929_v1/` and outputs under `artifacts/choice_discrimination_20260929_v1/`, with separate control/candidate seed directories. Those paths are reserved names in this design, not existing runs. Implement any chosen trainer under active `src/`/`scripts/` and reuse verified utilities where appropriate; frozen snapshots remain read-only evidence.

Before launch, freeze the derived data manifests, each per-seed full batch plan, evaluation memberships and the source/runtime contract. Verify that the gold anchor examples fit every planned batch and that group quotas survive filtering. Complete a no-update loss/gradient preflight plus disposable correctness checks for candidate token boundaries, causal shift, no EOS in choice scores, domain weighting, candidate-order invariance, gradient accumulation and exact resume. Check the actual exported model loader and reporting metadata. These tests verify the new mathematical implementation; they are not performance evaluations.

Then register each actual training phase in [TRAINING_PHASES.md](../../TRAINING_PHASES.md) and its linked report, including objective, parent/hash, final data/hash, full recipe, ceilings, selection rules, isolated output and tested launch/resume commands. Commands are intentionally unset until an implementation exists; copying the retained reproduction command would execute the wrong experiment. Maintain `status.json` and dated `events.jsonl`, with preparation/running/paused/failed/completed states and actual counts. Record negative or skipped outcomes separately from the successful lineage. This design document creates no training process and changes no checkpoint.
