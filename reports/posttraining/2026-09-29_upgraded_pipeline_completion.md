# Completed upgraded post-training study

Verified on **2026-09-29 shortly after 00:09 KST**. The registered workflow reached `completed / package` on **2026-09-28 at 23:48 KST**. The compute ledger finished writing in the same minute. The runner exited, and the assigned H100 has no active training or evaluation process.

## Execution and selection

- Completed nine SFT pilots, six full SFT comparisons, and six SFT controls.
- Completed preference collection for all three seeds, four DPO pilots, three full DPO runs, and three positive-only controls.
- Completed four direct-SFT RL pilots. No pilot improved over its SFT parent, so extended direct-SFT RL was skipped. RL from DPO was skipped because no DPO recipe passed parent-relative selection.
- Evaluated all 21 registered post-training reporting checkpoints on the frozen required benchmark protocol; reused the verified base report separately.
- Selected SFT recipe `sft/5/checkpoint_1`, deployment seed `20260928`, update 105. Its prefix contains 862,660 response targets and 419,498 raw replay targets.

Selected weight SHA256: `defd0d91948ba8501a33b0c4996bf3bf37ebdf6e90338a05bab3a1773a2d7f72`.

The three-seed mean development gain was **11.12 percentage points**. On the protected confirmation panel, the selected candidate scored **46.16%**, versus **36.04%** for the base: **+10.12 percentage points**, with a paired group-bootstrap 95% interval of **+8.23 to +12.05 points**. All registered confirmation retention checks passed. This panel is separate from the required competition benchmarks.

## Required benchmark results

The table uses raw accuracy (`acc`) for the four answer-selection tasks and token perplexity for WikiText-103. The original normalization, splits, tokenizer, context, and scoring protocol were preserved.

| Metric | Scratch base | Selected SFT |
| --- | ---: | ---: |
| HellaSwag accuracy | 28.48% | 28.25% |
| ARC-Easy accuracy | 46.59% | 46.46% |
| PIQA accuracy | 62.08% | 61.32% |
| WinoGrande accuracy | 50.67% | 51.30% |
| WikiText-103 perplexity, lower is better | 25.5198 | 25.9415 |

These are mixed point-estimate changes, and do not establish an overall competition-score improvement. The protected task-panel gain should not be represented as a gain on the competition benchmarks. The frozen selection decision was made before these benchmark reports and was not revised using them. Length-normalized metrics remain available in the original summaries.

## Package verification and handoff

The [candidate package](../../artifacts/posttraining_sota_v1/candidate_package/README.md) is complete. An independent invocation of `scripts/verify_sota_package.py` passed on September 29: inventory and file checksums matched, selected weights matched, the model and tokenizer loaded offline, and a CPU forward pass produced finite logits. The verifier returned `verified: true, offline_reload: true` for the weight hash above.

The package has not been uploaded or submitted. The existing base release remains preserved. No further training was launched during this status check.

- [Final workflow status](../../artifacts/posttraining_sota_v1/status.json)
- [Full experiment report](../../artifacts/posttraining_sota_v1/REPORT.md)
- [Frozen selection and confirmation](../../artifacts/posttraining_sota_v1/selection/selection.json)
- [Selected candidate benchmark summary](../../artifacts/posttraining_sota_v1/official/12_defd0d91948ba850/attempt_001/summary.json)
- [Base benchmark summary](../../artifacts/phase_evaluation_v1/attempts/official_13b/01/output/summary.json)
- [Compute ledger](../../artifacts/posttraining_sota_v1/COMPUTE_LEDGER.json)
