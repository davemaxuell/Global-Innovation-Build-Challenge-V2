# Executed SFT method and checkpoint preservation

This describes the completed `posttraining-sota-v1` study, using its registered configuration, trainer, deterministic batch plan, and selected export provenance. It is an account of the executed method, not a proposal for another training run.

## Preserved checkpoints

At the user's request, separate read-only copies of both model exports were created under [preserved checkpoints](../../artifacts/preserved_checkpoints/2026-09-29_base_and_sft/README.md). All 15 inventoried files were verified against their recorded checksums. Original run exports and release packages remain intact.

| Copy | Weight SHA256 |
| --- | --- |
| `base_13b/` | `ffa75b07ec377167ab2e4d4263f2fa8374549f8592f9ce2e8166875cab1f38e7` |
| `sft_selected/` | `defd0d91948ba8501a33b0c4996bf3bf37ebdf6e90338a05bab3a1773a2d7f72` |

Each export contains weights, architecture configuration, tokenizer and provenance. Historical optimizer/resume states remain in the original run directories. The preserved copies are local and read-only, not an off-machine backup. Future experiments should write to separate outputs.

## Starting point and supervision

Every main SFT comparison began independently from the same completed 46,346,752-parameter scratch base with 13,000,048,640 pretraining targets. SFT updated all model parameters using the unchanged architecture and tokenizer vocabulary. It used full-parameter optimization, without LoRA adapters or an external pretrained model.

Examples contained a prompt followed by a supplied correct response. The model used teacher forcing: each target token was predicted with the gold preceding answer tokens as context. Cross-entropy applied only to the final response and its real EOS token; prompt and padding positions had label `-100` and contributed no loss. The causal shift occurred once. Earlier conversation turns could supply context but were not additional supervised targets in that example.

Maximum combined context was 1,024 tokens. Maximum supervised response was 128 tokens including EOS. Overlength examples were rejected rather than truncated into incomplete answers. Prompts mixed the project's textual chat format and raw question/answer format. No new vocabulary tokens were added.

## Data inventory and actual exposure

The prepared train file contains 118,429 examples. Of these, 15,000 older procedural examples were reserved for the legacy control; the main balanced SFT pool contained **103,429 eligible examples**. Development and confirmation each contained 7,862 examples and were not training inputs.

The main data consisted of retained Databricks Dolly and OpenAssistant demonstrations; CommonSenseQA answer text; QASC answer text and its dataset-supplied supporting facts; and program-generated grounded, arithmetic, logic, sorting, JSON and coreference tasks. QASC supporting facts were supplied demonstrations, not generated teacher reasoning. No larger-model teacher was used.

The mixture was defined by supervised response tokens, not by example counts:

| Bucket | Planned share | Selected checkpoint response targets |
| --- | ---: | ---: |
| General instructions | 30% | 258,789 |
| Knowledge | 35% | 301,956 |
| Grounded tasks | 20% | 172,518 |
| Procedural tasks | 15% | 129,397 |
| Total | 100% | 862,660 |

Formatting-only supervision was capped at 5%. Grounded supervision also included relevant human demonstrations, so bucket and dataset source are not the same classification.

Reconstructing the deterministic plan and checking it against the registered plan hash established **55,901 example exposures / 55,836 distinct examples** through selected update 105, with at most two exposures per example. The selected response-token totals by source were Dolly 344,857; OpenAssistant 51,422; CommonSenseQA 28,651; QASC 273,305; program-generated tasks 164,425. Chat prompts accounted for 628,229 response targets and raw prompts for 234,431.

## Raw-text replay and objective

Raw replay used protected training documents from the existing pretraining corpus, targeting Edu/DCLM/Wikipedia proportions of 60/30/10. Raw next-token cross-entropy was computed separately from supervised response cross-entropy.

The selected recipe minimized:

`loss = 0.7 * mean_response_cross_entropy + 0.3 * mean_raw_replay_cross_entropy`

Each mean had its own target-token denominator. Thus 70/30 describes the objective weights; it is not an exact measured token-count split. Whole-document batching can overshoot the requested replay token quantity. The selected export records **419,498 replay targets**, in addition to the 862,660 response targets. It processed 4,670,374 response-example input tokens and 420,106 replay input tokens; input tokens must not be confused with supervised targets.

## Optimization and selection

| Setting | Executed selected recipe |
| --- | --- |
| Optimizer | AdamW, betas (0.9, 0.95) |
| Peak learning rate | 0.00001 |
| Schedule | 3% target-token warmup, then cosine decay to 10% of peak over the full planned budget |
| Effective update size | Approximately 8,192 response targets, using complete examples |
| Microbatch | 8 examples with accumulated gradients |
| Weight decay | 0.1 on tensors with at least two dimensions; zero on remaining tensors |
| Gradient clipping | Norm 1.0 |
| Precision / hardware | BF16 autocast with FP32 loss computation, one assigned H100 |
| Repetition limit | At most two passes through each eligible example |
| Evaluation checkpoints | Approximately 25%, 50%, and 100% of the full run |

Nine pilots compared learning rates `3e-6`, `1e-5`, `3e-5` with replay loss weights `0`, `0.1`, `0.3`. Each used 250,000 response targets. The two retained leading recipes were run independently with three seeds. Controls tested the older data and balanced data without replay.

The nominal four-million-response-target ceiling was reduced by the available mixture under the two-pass limit to **1,715,410 targets**, or **209 updates** for the selected seed. All 209 updates completed. Development selection chose the stored **update-105 export**, containing 862,660 response targets, using replicated performance and retention checks. The deployment seed was fixed in advance. The choice of update 105 is a registered selection outcome, not proof that a nearby checkpoint differs significantly.

The selected model then passed the single frozen confirmation comparison. Required competition benchmarks were reported afterward and could not change the registered selection. DPO and RL descendants were separate experiments; their weights are not included in the preserved selected SFT export.

## Interpretation limits

This was bounded, short-response task adaptation with raw replay. It did not add another large pretraining corpus or a broad long-form reasoning curriculum. Its protected task-panel gains transferred poorly to the required competition benchmarks. Response-length limits, source balance, task difficulty, and evaluation alignment are hypotheses to investigate; these records alone do not identify a single cause of the weak transfer.

Sources: [configuration](../../artifacts/posttraining_sota_v1/snapshot/configuration.json), [selected run contract](../../artifacts/posttraining_sota_v1/runs/sft_full_5_seed20260928/run.json), [selected export provenance](../../artifacts/posttraining_sota_v1/runs/sft_full_5_seed20260928/endpoints/step_0000105/export/training_provenance.json), [trainer](../../src/scglm_pipeline/sota_train.py), [response masking](../../src/scglm_pipeline/task_registry.py), [loss implementation](../../src/scglm_post/sft.py), [completed results](2026-09-29_upgraded_pipeline_completion.md).
