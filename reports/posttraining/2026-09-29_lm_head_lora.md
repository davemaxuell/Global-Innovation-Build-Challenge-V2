# Separate output-head adaptation experiment — September 29, 2026

The user authorized this ablation while preserving all previous training results. **The complete experiment finished September 29 at 01:40:29 KST**, with successful runner exit. All five corrected training runs completed 105 updates each, and development, fresh confirmation and required benchmark reporting finished. At independent verification at 07:50 KST, the assigned H100 was idle. The head-only adapter did not establish a reliable improvement over the base, and no previous checkpoint or release choice was replaced. See [completed results](../../artifacts/lm_head_lora_v2/REPORT.md), [final runner state](../../artifacts/lm_head_lora_v2/status.json), and [successful queue exit](../../artifacts/lm_head_lora_v2/queue_status.json).

## Preserved starting points

All 15 files in the [read-only preservation snapshot](../../artifacts/preserved_checkpoints/2026-09-29_base_and_sft/MANIFEST.json) passed checksum verification. The base weight hash is `ffa75b07ec377167ab2e4d4263f2fa8374549f8592f9ce2e8166875cab1f38e7`; selected full-SFT hash is `defd0d91948ba8501a33b0c4996bf3bf37ebdf6e90338a05bab3a1773a2d7f72`. All 45 code files in the completed post-training study still matched its original registration before launch. Original release packages and checkpoint selection remain intact.

## Phase record

| Phase | Recorded outcome / next behavior |
| --- | --- |
| Architecture | Implemented a separate rank-8, alpha-8 correction on the output path. All transformer layers and the tied input/base-output embedding stay frozen. 135,168 trainable / 46,481,920 total parameters. |
| Data | Original SFT train/replay files reused with verified batch-plan identity. Prepared 1,014 fresh development and 1,014 fresh knowledge-confirmation questions; 600 generated confirmation diagnostics. Raw retention uses 64 older development documents per source. |
| Verification | Nine CPU behavioral tests passed; the real-H100 two-update continuous run exactly matched one update plus checkpoint/reload plus one update. Frozen tensors, optimizer state, adapter, counters and replay positions matched. |
| Initial infrastructure attempt | First LR completed 105 updates; second LR stopped at update 12. A benchmark-harness re-tying compatibility issue was found before official evaluation or protected confirmation. All artifacts remain in `lm_head_lora_v1`; no method or selection parameter changed. |
| Compatibility correction | Updated HF tied-weight metadata to point at the actual nested frozen head. Added an actual HFLM wrapper test that preserves logits and the output correction. Repeated CPU/H100 validation successfully. |
| Corrected sequence | Completed under a separate registration in `lm_head_lora_v2`. Fresh panel files are byte-identical to the initial design. The first corrected LR export's adapter weights are also byte-identical to the initial attempt, confirming unchanged training math. |
| Execution queue | Persistent session `gibc-head-lora-queue` successfully waited for the single assigned H100 and launched the corrected study. No other workload was interrupted. |
| Automatic study | Completed three LRs × 105 matched SFT updates, then two additional seeds at selected LR 1e-4: 525 corrected-study updates total. Completed the frozen candidate/base/full-SFT confirmation comparison and all required benchmarks. No release replacement. |
| Final artifact verification | All five adapter-export inventories, all 11 evaluation-report inventories and all 15 preserved checkpoint files passed checksum checks. Original frozen-tensor digests agree across all five exports. |

The initial attempt used 117 actual optimizer updates, recorded in [its preserved interruption ledger](../../artifacts/lm_head_lora_v1/SUPERSEDED.json). Disposable validation updates are separate and retained under [validation attempts](../../artifacts/lm_head_lora_validation/). They must not be mistaken for improvements or included in the final model's ancestry. The corrected sequence starts each model independently from the preserved base.

The new primary score equally weights raw answer-likelihood accuracy on fresh science and commonsense questions. Generated grounding does not determine selection. Grouped bootstrap intervals compare the fixed head-adapter candidate with both preexisting comparators. This two-domain score must not be compared numerically with the prior study's three-domain aggregate.

## Measured outcome

| Model | Fresh knowledge confirmation accuracy | WikiText-103 perplexity (lower is better) |
| --- | ---: | ---: |
| Preserved base | 15.47% | 25.5198 |
| Preserved full SFT | 17.46% | 25.9415 |
| Selected head-only adapter | 15.36% | 25.5846 |

The adapter-minus-base knowledge difference was **-0.11 percentage points**, with grouped-bootstrap 95% interval **[-1.39, +1.30]**. It passed the registered raw-NLL/domain-accuracy retention guards, but did not establish improvement. Full SFT's higher point estimate is not a statistically established advantage on this small two-domain confirmation panel either; its comparison intervals include zero.

The higher adapter learning rates (3e-4 and 1e-3) failed raw-language-model retention guards. The selected 1e-4 recipe passed those guards in all three seeds. Its required benchmark point estimates were near the base: HellaSwag 28.42%, ARC-Easy 46.72%, PIQA 62.08%, WinoGrande 50.59%. These small mixed changes do not establish a competition-score gain.

On the 800 generation diagnostics, the head adapter scored **0% exact-answer-plus-EOS**, ended with EOS in only 4% of cases, and had a 29.88% repetition rate. Full SFT retained much stronger task-format behavior in these tests. This result does not support replacing it with this rank-8 head-only recipe. It does not rule out all other head-only methods, budgets or ranks.

The [method and execution guide](../../FAILED_APPROACHES.md) records the research basis, exact objective, masks, data limitations, decision rules, checkpoint format and resume commands. The [completed report](../../artifacts/lm_head_lora_v2/REPORT.md) and [compute ledger](../../artifacts/lm_head_lora_v2/COMPUTE_LEDGER.json) contain the measured results and run accounting.

Evidence: [validation](../../artifacts/lm_head_lora_validation/validation.json), [fresh-data manifest](../../artifacts/lm_head_lora_v2/data/manifest.json), [configuration](../../artifacts/lm_head_lora_v2/registration.json), [queue launch](../../artifacts/lm_head_lora_v2/queue_launch.json), [original SFT recipe](2026-09-29_sft_method_and_preservation.md).
