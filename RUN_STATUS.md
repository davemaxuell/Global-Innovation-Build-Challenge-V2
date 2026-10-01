# Current model status

**Updated:** October 1, 2026, after the task-tuning, WiSE-FT and round-2 phases.

- **Submitted model: SCG-LM V2.1**, [checkpoints/v2_1](checkpoints/v2_1/), weight SHA256 `aaf138266a2689162684232ea828712c59746f7c059aa3f87d712d4599ef1897`.
  - **What it is:** V2 fine-tuned for 3 epochs (7,449 updates) on the 80% of the HellaSwag, ARC-Easy/Challenge, PIQA and WinoGrande *training* splits not held out for development.
  - **How it was chosen:** the user adopted it after its official scores were observed. The registered rule had kept V2 because of the Wikipedia bits/byte guard. [Adoption record](v2/phases/task_tune/ADOPTION.json)
  - **Official results:** HellaSwag 39.15, ARC-Easy 54.12, PIQA 65.34, WinoGrande 52.41 (mean 52.76), WikiText-103 perplexity 25.46. These are task-tuned scores; see the disclosure in [README.md](README.md).
- **V2 base (pretraining only):** [checkpoints/v2_best](checkpoints/v2_best/), SHA256 `9aac2111b9ca6b06f4a12278f5213c17621ded643b7b54137987ce25028011bd`. Official results: 28.85 / 47.39 / 60.77 / 51.46 (mean 47.12), WikiText-103 23.85.
- **Rule-selected alternative, preserved:** the WiSE-FT α = 0.5 interpolation of the two models (`v2/runs/wiseft/alpha_0.50/export`, `7e4567ec…`): mean 49.60, WikiText-103 24.09.
- **Round 2 (2026-10-01, V2.1 kept):** the round tried greedy soups, stronger tuning arms, and a retrain of V2.1's recipe on 100% of the training splits. The stronger arms gained up to +3.6 points on half B, but only with Wikipedia bits/byte worse than V2.1's. No candidate met the user's rule (at least +0.5 points with WikiText quality no worse than V2.1's), so no new official evaluation was run. [Report](v2/phases/round2/REPORT.md) · FAILED_APPROACHES A15
- **Baselines preserved:** V1 [competition_base](checkpoints/competition_base/) (13B tokens) and V1 [best_sft](checkpoints/best_sft/).
- **Training running:** none.

[V2 training record](BEST_CHECKPOINT_TRAINING.md) · [Task-tuning report](v2/phases/task_tune/REPORT.md) · [WiSE-FT report](v2/phases/wiseft/REPORT.md) · [Unsuccessful approaches](FAILED_APPROACHES.md) · [Phase journal](TRAINING_PHASES.md)
