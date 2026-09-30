# DPO follow-up results — September 28, 2026

**Completed 2026-09-28 08:00 KST.** Preference generation, DPO, held-out development evaluation and the full recorded competition benchmark protocol have finished. The pipeline retained the 13B base recommendation; no release was replaced or published.

## Executed experiment

- Parent: the completed mixed SFT model, itself derived from our own 13B-token scratch model.
- Explicit user request authorized this exploratory DPO run despite the earlier SFT promotion-gate failure. Historical decisions and promotion thresholds stayed unchanged.
- Sampled 7,500 protected training prompts × 8 answers: 60,000 own-model answers and 371,736 generated tokens in 152.86 seconds.
- Accepted 1,359 verifier-correct/incorrect pairs: classification 600, sorting 600, arithmetic 126, extraction 11, JSON 22. All selected answers ended with EOS and met the 2× length-ratio guard.
- Completed 85 optimizer updates in 61.10 seconds, including validation/checkpoint overhead: 12,648 response/EOS target tokens, 116,040 policy-processed tokens and 116,040 reference-forward tokens.
- Standard DPO with a frozen exact SFT reference, learning rate 1e-6, beta 0.1, one pass. The released model still has 46,346,752 unique parameters.
- Development and benchmark results for the base/SFT parent were reused with verified identities and disclosed as previously observed. The DPO endpoint was newly evaluated.

## Competition-protocol results

The accuracy rows below use raw accuracy. The same fixed evaluation settings were used for all three checkpoints; these are our local protocol results, not scores supplied by the organizers. Differences are descriptive point estimates.

| Metric | 13B base | Mixed SFT | DPO |
| --- | ---: | ---: | ---: |
| HellaSwag accuracy | 28.48% | 28.14% | 28.21% |
| ARC-Easy accuracy | 46.59% | 46.21% | 46.30% |
| PIQA accuracy | 62.08% | 61.92% | 61.92% |
| WinoGrande accuracy | 50.67% | 50.12% | 49.72% |
| WikiText-103 perplexity, lower is better | 25.5198 | 27.6288 | 27.6009 |

Character-normalized accuracy is retained separately:

| Metric | 13B base | Mixed SFT | DPO |
| --- | ---: | ---: | ---: |
| HellaSwag | 30.64% | 30.67% | 30.64% |
| ARC-Easy | 42.38% | 41.67% | 41.58% |
| PIQA | 60.39% | 60.61% | 60.55% |

HellaSwag: 10,042 examples; ARC-Easy: 2,376; PIQA: 1,838; WinoGrande: 1,267. WikiText-103: 269,569 scored tokens in 60 validation articles, context 1,024 and stride 512.

## Held-out instruction development results

| Task | Mixed SFT | DPO |
| --- | ---: | ---: |
| arithmetic | 0.25% | 0.50% |
| classification | 48.50% | 52.00% |
| extraction | 97.50% | 93.75% |
| json | 99.25% | 99.00% |
| sorting | 0.00% | 0.00% |
| Macro average | 49.10% | 49.05% |

The paired macro difference was -0.05 percentage points, with a bootstrap 95% interval from -1.12 to 0.98 points. This does not demonstrate aggregate instruction improvement. Extraction declined, while classification improved. These are narrow procedural tasks; their scores do not establish broad factual knowledge.

## Decision and remaining scope

DPO did not meet the development improvement/regression guards relative to SFT and remained above the allowed raw-source loss regression relative to the base. The pipeline retained the base and left the confirmation panel unopened. Benchmark changes were small and mixed, and were not used to retroactively change the registered decision.
The 0.02 raw-loss cutoff is an internal conservative promotion policy, not the competition scoring formula. Completing DPO does not itself earn a measured performance gain.
RL was not run and remains deferred under the existing roadmap. This run completed preference preparation, DPO and evaluation. It was one short DPO pilot; no claim of an optimal recipe or improved winning probability is made.

## Verification and recovered setup issue

- 173 CPU tests passed before launch. Every DPO update was independently reconciled with the pair plan and token counters.
- Final export tensors exactly match the final checkpoint. All checked weights/optimizer tensors are finite, all 110 exported tensors changed from the SFT parent, and the parent/tokenizer stayed unchanged.
- All selected answers were traced to checksummed, EOS-ended own-model rollouts. The final audit rescored 2,000 procedural predictions and 600 grounded predictions, recomputed raw-loss aggregates and reproduced selection.
- All 27 completed-stage artifact checks passed. Prior registered sources, earlier experiments and the base release inventory remain unchanged.
- The first sampling attempt failed while saving its initial batch because the rollout directory was absent. No DPO update had occurred. The operational launcher creates that directory; the same registered workflow then completed. Original failure logs are retained. Additional discarded generation is bounded by 8,192 tokens; its exact count was not saved.
- No training process remains active.

## Records

- [Automatic pipeline report](../../artifacts/posttraining_dpo_13b_v1/REPORT.md)
- [Final audit](../../artifacts/posttraining_dpo_13b_v1/FINAL_AUDIT.json)
- [DPO training integrity](../../artifacts/posttraining_dpo_13b_v1/training_integrity.json)
- [Preference integrity](../../artifacts/posttraining_dpo_13b_v1/preference_integrity.json)
- [Compute ledger](../../artifacts/posttraining_dpo_13b_v1/COMPUTE_LEDGER.json)
- [DPO benchmark summary](../../artifacts/posttraining_dpo_13b_v1/official/endpoint_2-1790549633481212279/summary.json)
- [Recorded recipe and run commands](../../FAILED_APPROACHES.md)
