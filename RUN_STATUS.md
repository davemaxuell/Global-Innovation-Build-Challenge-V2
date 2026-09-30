# Current model status

**Updated:** October 1, 2026, after the V2 selection and the code reorganization.

- **Selected model:** [checkpoints/v2_best](checkpoints/v2_best/). It is V2, trained from scratch on 44,999,966,720 targets (686,645 updates, 31.9 h on one H100).
- **Weight SHA256:** `9aac2111b9ca6b06f4a12278f5213c17621ded643b7b54137987ce25028011bd`. Selected on development data only, then confirmed by the pre-registered V1-vs-V2 rule ([outcome](v2/phases/final_selection/v1_vs_v2.json)).
- **Official results:** HellaSwag 28.85, ARC-Easy 47.39, PIQA 60.77, WinoGrande 51.46 (mean 47.12), WikiText-103 perplexity 23.85. V1 13B base: 46.96 / 25.52.
- **Baselines preserved:** V1 [competition_base](checkpoints/competition_base/) (13B tokens) and V1 [best_sft](checkpoints/best_sft/).
- **Active code:** only the V2 pipeline ([CURRENT_PIPELINE.md](CURRENT_PIPELINE.md)). Retired code is in `archive/` ([report](reports/cleanup/2026-10-01/REPORT.md)).
- **Training running:** none.

[V2 training record](BEST_CHECKPOINT_TRAINING.md) · [Unsuccessful approaches](FAILED_APPROACHES.md) · [Phase journal](TRAINING_PHASES.md)
