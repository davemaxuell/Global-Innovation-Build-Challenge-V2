# MiMo follow-up pipelines: mid-training and revised RL

Experiments 2 and 3 from the [MiMo investigation](../../reports/research/2026-09-29_mimo_v2_6_training_options.md), implemented September 29, 2026. **Neither has been launched.** No phase is registered yet; each runner adds its phases to [TRAINING_PHASES.md](../../TRAINING_PHASES.md) with a linked report immediately before running them, as `AGENTS.md` requires. Neither pipeline can replace `checkpoints/best_sft` or `checkpoints/competition_base`, and neither schedules any follow-up automatically.

| | Experiment 2: data-quality mid-training | Experiment 3: revised small RL |
|---|---|---|
| Config | [configs/midtrain_quality.json](../../archive/v1_code/configs/midtrain_quality.json) | [configs/rl_revised.json](../../archive/v1_code/configs/rl_revised.json) |
| Code | [src/scglm_midtrain/](../../archive/v1_code/src/scglm_midtrain/) | [src/scglm_rl/](../../archive/v1_code/src/scglm_rl/) |
| Entry point | `scripts/run_midtrain_pipeline.py` | `scripts/run_rl_pipeline.py` |
| Parent | 13B base plus its AdamW state (`runs/continuation_13b` final checkpoint) | Selected SFT `checkpoints/best_sft`; frozen copy as KL reference |
| Arms | `random` vs `curated` documents from the same unused span | `token_mean` vs `prompt_mean` loss normalization |
| Per-arm ceiling | 977 updates × 65,536 = 64,028,672 targets; 2 GPU-hours | 50 actual updates; 500 collection attempts; 2M generated tokens; 2 GPU-hours |
| Outputs | `data/midtrain_quality_20260929_v3/`, `artifacts/midtrain_quality_20260929_v3/` | `artifacts/rl_revised_20260929_v1/` |

Both pipelines share [src/scglm_study/](../../archive/v1_code/src/scglm_study/), which provides phase registration, a write-ahead GPU budget with configurable ceilings, assigned-GPU selection and a resumable development evaluator. The evaluator reuses the choice study's scorers and paired grouped bootstrap.

## Commands

From the project root, using the recorded environment. Replace `midtrain` with `rl` for experiment 3.

```bash
PYTHONPATH=src:. PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 \
  /home/bufsgpu/yes/envs/sw/bin/python scripts/run_midtrain_pipeline.py verify --load   # read-only (default)
  ... scripts/run_midtrain_pipeline.py prepare     # experiment 2 only: CPU data preparation
  ... scripts/run_midtrain_pipeline.py preflight   # tests + GPU no-update checks
  ... scripts/run_midtrain_pipeline.py run         # all gated phases
  ... scripts/run_midtrain_pipeline.py resume      # same contracts and cumulative budget
  ... scripts/run_midtrain_pipeline.py status | report
```

GPU commands use only `GPU-78815613-815b-f56b-27c0-6bc239e093fc` and hold `artifacts/pipeline_gpu0.lock`. Changing code or configuration after preflight blocks resume, because the source hashes are part of the run contract.

## Experiment 2: what happens

1. **Preparation (CPU).** Finds the first document after each source's final consumed cursor, using a binary search of the index. Skips documents without their real EOS, and excludes the choice-study raw panels, historical replay, earlier scored evaluation documents and their URL groups. Splits a new held-out raw panel (256 documents per source, by URL group) before either arm is selected.
2. **Arm selection.** `random` is a seeded sample of the eligible pool. `curated` must pass the Gopher quality and repetition filters, then is ranked by a documented coherence score. It matches the random arm's length mix bin by bin and caps documents per host. Model loss is never used for selection.
3. **Training.** The retained `scglm.train.Trainer` runs with the same seed, sampling sequence, schedule and inherited optimizer. The schedule rewarms from 3e-6 to 1e-5 over 32 updates, then decays by cosine to 3e-6.
4. **Decision.** `curated_preferred` requires three things: at least a +1 point gain over `random` in science/commonsense continuation accuracy with a one-sided 95% lower bound above zero, a held-out NLL upper bound of at most +0.01 nats per source, and passing the base-relative point guards. Otherwise the outcome is `not_established`.

**Dry run on real data (read-only, 1M tokens per source):** the hard filters pass 95.5% of FineWeb-Edu, 96.1% of DCLM and 71.4% of Wikipedia documents. Most Wikipedia failures are articles under 50 words. Edu and DCLM were already filtered upstream, so the curated arm differs from the random arm mainly through the coherence ranking. The pool is 2.5× each arm's target, so curated keeps roughly the top 40% of each length bin.

## Experiment 3: what happens

1. **Audit (no updates).** Runs 384 training prompts × 8 samples × at most 64 tokens from the parent. For each family it records correctness, termination, truncation, all-correct/all-wrong/mixed groups and verifier disagreements. A family qualifies if at least 15% of its groups are mixed and its reference answers fit the 64-token limit. Training starts only if at least three families qualify, including science or commonsense. The qualifying families, their shares and their informative rates are frozen for both arms. All seven registered families' reference answers fit, with the longest at 14 tokens.
2. **Collection.** Keeps informative groups from families that have filled their quota and resamples only the families still short (at most 4 rounds under one frozen policy). Groups are never reused across updates, and a retry never counts as an update.
3. **Loss.** Clipped policy gradient plus exact categorical KL to the frozen parent, in FP32. Both arms weight families by their declared shares. They differ only inside a family: `token_mean` normalizes by the family's token count, `prompt_mean` averages per-prompt token means. Every row's training log-probabilities must match the sampling log-probabilities within 0.002 before any update.
4. **Guards and decision.** Development checks at updates 25 and 50. Raw NLL must stay within +0.01 nats of the parent per stratum, and science/commonsense accuracy within −2 points. A breach stops that arm. The arms are compared at 50 updates on generated-task accuracy for the qualifying families, and a preference needs at least 2 points with a one-sided 95% bound excluding zero.

## Verification done

- `verify --load` passes for both. The 13B checkpoint's weights match `checkpoints/competition_base` tensor for tensor, and it contains AdamW state.
- New tests: [test_study_common.py](../../archive/v1_code/tests/test_study_common.py), [test_midtrain_pipeline.py](../../archive/v1_code/tests/test_midtrain_pipeline.py), [test_rl_pipeline.py](../../archive/v1_code/tests/test_rl_pipeline.py). They cover exact unused-span boundaries, held-out isolation, whole-document shards, sampling-versus-training log-probabilities, gradients of both normalizations against a per-token oracle, deficit refill, and bit-exact RL resume after a clean interruption. Full suite: **193 passed, 2 GPU tests skipped**.
- Not yet exercised: GPU preflight and any training or scoring on the real models.

## Disclosure

The development panels come from the choice study and were already scored on the base, the selected SFT and one control endpoint. The choice study's reserved confirmation panels and the competition benchmarks are not used by either pipeline. A result that qualifies would still need a separately registered confirmation step.
