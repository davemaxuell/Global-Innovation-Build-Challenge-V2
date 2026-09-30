# Working checkpoints

| Purpose | Ready-to-load directory | Weight SHA256 |
| --- | --- | --- |
| **Selected: V2 base model** (45.0B tokens; competition benchmarks) | [v2_best/](v2_best/) | `9aac2111b9ca6b06f4a12278f5213c17621ded643b7b54137987ce25028011bd` |
| V1 base reference (13.0B tokens) | [competition_base/](competition_base/) | `ffa75b07ec377167ab2e4d4263f2fa8374549f8592f9ce2e8166875cab1f38e7` |
| V1 full-SFT instruction model | [best_sft/](best_sft/) | `defd0d91948ba8501a33b0c4996bf3bf37ebdf6e90338a05bab3a1773a2d7f72` |

All three are independent, read-only copies of preserved exports, each a normal local Hugging Face model with tokenizer, configuration and training provenance.

- **`v2_best`** (added 2026-10-01) is a hash-verified copy of `v2/runs/main/export/`: offline load, finite forward pass and the 46,346,752-parameter tied-embedding audit all pass. [Inventory](SELECTION_2026-10-01.json) · [training record](../BEST_CHECKPOINT_TRAINING.md) · [frozen selection](../v2/phases/final_selection/selection.json).
- **The V1 checkpoints** are unchanged from their 2026-09-29 handoff ([inventory](SELECTION_2026-09-29.json), [V1 record](../docs/history/V1_BEST_CHECKPOINT_TRAINING.md)). `best_sft` is an instruction model selected for a different objective; it is not a benchmark improvement over the V1 base.

Continued training needs optimizer and RNG state, which these exports don't contain. It lives in the original run directories (for V2: `v2/runs/main/checkpoints/` and `v2/runs/main/milestones/`). New training must write to a new output directory.
