**Post-pretraining pipeline review — September 28, 2026**

**Verdict: the core training and selection design is sound, and the completed runs remain supported by the saved evidence. I cannot give an unconditional correctness sign-off: two reproducible defects need correction before the next run.** The existing 173 tests pass, but they miss both defects below.

This review covers the current 13B-token base handoff, human-only and mixed SFT, the separate DPO follow-up, development scoring, conditional confirmation, benchmark reporting, and preservation of the base release. The model has 46,346,752 parameters; “13B” refers to cumulative pretraining targets. Verification used CPU tests, isolated failure reproductions, complete instruction-data re-encoding, saved predictions, checkpoint tensors, and file checksums. It did not rerun full GPU training or benchmark inference.

1. **[P2] JSON verifiers accept arrays when the instruction requires an object.** In [procedural.py](../../src/scglm_post/procedural.py), lines 82–84, and [system_data.py](../../src/scglm_post/system_data.py), lines 134–138, `object_pairs_hook=lambda pairs: pairs` makes parsed objects indistinguishable from arrays of key/value pairs. For a request requiring `{"number": 95405}`, the current verifier accepts `[["number", 95405]]`. The same problem affects system-conditioned `number`, `value`, and arithmetic `sum` tasks. These verifiers determine both development accuracy and chosen/rejected DPO labels, so a future run could reward the wrong output type. Require an actual top-level object, preserve duplicate-key rejection, and test arrays, nested arrays, duplicate keys, booleans, and extra fields. Fresh inspection found **zero affected accepted answers among 8,000 saved development predictions and zero affected chosen answers among all 1,359 saved DPO pairs**. This defect does not change the current reported scores or selection.

2. **[P2] An early training failure cannot recover automatically.** [workflow.py](../../artifacts/posttraining_sota_v1/snapshot/src/scglm_pipeline/workflow.py), lines 255–257, rejects any existing training directory without `checkpoint_latest.json`. Both trainers create the directory and run record before their first durable checkpoint; the current recipes checkpoint every 25 updates. Injecting a first-update exception into a tiny CPU SFT run produced `status=failed`, no checkpoint, and the retry error `Training output exists without a recoverable checkpoint`. DPO has the same initialization/checkpoint structure. Write a durable step-zero checkpoint before optimization and handle interrupted initialization without discarding failed-attempt evidence. Add a failure-injection test before the first checkpoint. The completed production runs were not affected, and recovery from an existing checkpoint passes the current tests.

**Fresh verification evidence** is saved in [the accompanying JSON record](2026-09-28_post_pretraining_pipeline_review.json).

| Check | Result |
| --- | --- |
| Existing CPU suite, including DPO follow-up tests | 173 passed in 17.11 seconds |
| Registered source/config/runtime identities | Both workflows unchanged |
| Completed-stage artifact hashes | 27 checks in each workflow passed |
| Prepared instruction data | All 50,401 split entries re-encoded; masks, EOS, tokenizer/chat-template equivalence, counts, and group/prompt separation passed; includes the human-control subset duplicated in the mixed pool |
| Human-only SFT | 131 updates; 1,077,789 response/EOS targets; 2,801,905 processed tokens |
| Mixed SFT | 131 updates; 1,077,749 response/EOS targets; 3,246,128 processed tokens |
| DPO | 85 updates; 12,648 response/EOS targets; 116,040 policy tokens and 116,040 reference-forward tokens |
| Training plan and saved tensors | Every update's token counts reconcile; all three exports exactly match their final checkpoints; checked weights and optimizer tensors are finite |
| Preference data | All 1,359 pairs re-encoded and relabeled with the current verifier; prompt identity and length checks passed; chosen/rejected texts traced to EOS-ended own-model answers; all 469 rollout files hashed |
| Saved instruction predictions | All 8,000 reproduce the saved correctness flags |
| Selection | Both frozen development decisions reproduced exactly; both retain the base; confirmation scoring remains unused |
| Benchmark evidence | Endpoint identities, protocol identity, and underlying result-file hashes verified for all reported endpoints |
| Existing base release | All 233 manifest-listed files verified |

The objective implementations are appropriate. SFT supervises the final assistant response and EOS, masks prompts and earlier turns, shifts causal targets once, and weights accumulated gradients by the effective batch's supervised-token count. DPO uses summed response/EOS log probabilities, a frozen exact SFT reference, and the reference-relative log-sigmoid loss described in [Rafailov et al., equation 7](https://arxiv.org/html/2305.18290v3). Policy and reference do not share an optimizer. The two SFT arms start independently from the same base with closely matched supervised-token budgets.

Data preparation pins human-source revisions, groups conversations and duplicate prompts before splitting, protects benchmark and raw-language panels, and screens held-out examples against the completed pretraining corpus. Preference sampling uses protected training prompts. The recorded overlap methods do not certify semantic or arbitrary-substring independence; the code and reports correctly state that limitation.

Development selection applies the fixed instruction-gain and regression guards before benchmark reporting. The separate exploratory DPO run deliberately bypassed the earlier SFT eligibility decision for execution under its recorded user request, while retaining promotion thresholds and historical selections. This is disclosed in the experiment configuration and is not an accidental selection bypass.

The retained base is the correct outcome under the registered policy. Mixed SFT's procedural macro accuracy was 49.10%; DPO's was 49.05%, with a paired improvement interval spanning zero. Both remain outside the raw-language regression tolerance relative to the base. Those observations support the conservative selection decision, not a claim that the post-training recipe improved general model quality.

There are three smaller considerations for a future registered revision:

- Sampling specifies temperature and top-p but leaves top-k implicit. The pinned Transformers 5.5.4 runtime resolves it to **50**, so the completed sampling used top-k filtering as well. State the full resolved generation settings explicitly. This is reproducible with the current pinned runtime; it is not evidence of corrupted rollouts. See the [versioned GenerationConfig documentation](https://huggingface.co/docs/transformers/v5.5.4/main_classes/text_generation).
- Sorting scoring requires exactly comma-space formatting, while prompts say only “comma-separated.” For example, `5,67,83` is rejected when `5, 67, 83` is expected. Either make that spacing requirement explicit or parse the integer list. Inspection found no corresponding false negatives in the saved development outputs or mislabeled sorting rejects in the saved preference pairs.
- The quality evidence remains narrow: procedural/fictional tasks, one seed per training recipe, and no completed human-quality ratings. DPO pairs are concentrated in classification and sorting (1,200 of 1,359). A successful run of this pipeline alone does not establish broad instruction-following quality or an optimal recipe. RL is explicitly deferred, not an implemented stage that failed to run.

Only these new review records were added. Registered implementation files, training recipes, weights, decisions, and release files were preserved. Address the two defects in a newly registered revision with focused regression tests; the present evidence does not justify rerunning or replacing the completed models merely because these latent defects were found.
