# V2 main run: `v2_main_20260929`

**Status:** running since 2026-09-29 20:13 KST (tmux `gibc-v2-main`). [Registration](registration.json) · [status](status.json) · [events](events.jsonl) · run directory `v2/runs/main/`

- **Objective:** V2 base model for the competition benchmarks. Scratch initialization, no parent.
- **Data:** `v2/data/rich_v1/manifest.json` (SHA256 `6749a2f1…6a68`, 22.45B unique tokens). Shares: DCLM 35 / Edu 30 / targeted 15 / Wikipedia 10 / StackExchange 5 / books 5, as approved by the user.
- **Recipe:** `v2/configs/main.json`: 12×512 (46,346,752 parameters), AdamW, peak LR 6e-4, 500 warmup steps, then constant. A linear decay to 0 over about 10% of completed steps starts when `v2/runs/main/decay.json` is written. The ceiling is 45B tokens, at most 4 passes per source.
- **Pending decision:** decay start (latest about Sep 30 13:30 KST for the internal target, about 28B tokens; about Oct 1 00:00 KST for the Devpost cutoff, 45B ceiling).
- **Resume:** `CUDA_VISIBLE_DEVICES=GPU-78815613-815b-f56b-27c0-6bc239e093fc PYTHONPATH=src:v2/src python -m scglm_v2.train --config v2/configs/main.json --resume v2/runs/main/checkpoint_latest.json`

Progress, milestones and results are added here.

## Event log

- **20:13 KST:** launched. Warmup completed at step 500 (loss 5.305); about 395k tokens/s.
- **20:41 KST: failed at step 10,000** in the first periodic validation. `ValueError: No scored tokens for targeted`: targeted panel records keep their original `dclm`/`edu` source label, which V1's reader preserves. Training itself was healthy.
- **Fix:** `V2Trainer.documents` now labels held-out documents by manifest bucket. A second latent bug was found before resuming: the resume guard read `compile` from the base record (always `false`) and would have rejected this resume; it now reads the `v2` block. Two regression tests were added (14 CPU tests pass). The data, manifest and scientific configuration are unchanged.
- **20:43 KST:** resumed from the step-8,000 checkpoint (`d332506e…`). 2,000 updates (about 131M tokens, about 6 minutes) were repeated. The replayed losses matched the original to within non-deterministic-kernel noise, and the step-10,000 validation passed (mixture NLL 3.457).
- **20:55 KST:** the user said to run the training to completion without regard to the deadline. The registered 10% linear decay was scheduled to end at the 45B-token ceiling: `decay.json` sets start step 617,981 (40.5B tokens) and 68,664 decay steps, ending at step 686,645 (44,999,966,720 tokens). The file was written atomically and the trainer accepted it at step 11,938. Expected completion is about Oct 1, 05:00 KST.
- **20:57 KST:** the auto-resume supervisor (`v2/supervise_main.sh`, tmux `gibc-v2-supervisor`) started. It resumes from `checkpoint_latest.json` on failure or a killed process, at most 3 times, and never resumes a deliberate pause. It was tested on 5 fake-trainer scenarios.

## Completion

- **2026-10-01 04:11 KST: completed.** 686,645 updates, 44,999,966,720 tokens, 31.9 h of accounted training time. One failure (step 10,000, fixed and resumed); no auto-resumes were needed after that. Final training loss 3.085.
- **Export:** `v2/runs/main/export/` (`model.safetensors` SHA256 `9aac2111b9ca6b06f4a12278f5213c17621ded643b7b54137987ce25028011bd`). Decay-start milestone SHA256 `175d1e29…b4bb0c6`.
- **Held-out monitor NLL (small panels, about ±0.01 noise):** 3.457 (10k) → about 3.17 by the end of the constant phase (plateau) → **2.971** at step 680k. That is −0.197 nats during the decay, with every source improving by 0.17–0.24 nats.
- **Next:** the decay A/B branch started automatically at 04:11:45 KST (`v2/phases/anneal_quality/`); its initial LR exactly matches the main schedule at the branch point. Selection and official evaluation follow in `v2/phases/final_selection/`.
