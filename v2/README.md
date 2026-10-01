# V2: rich-data scratch pretraining

**Started:** 2026-09-29 (KST). **Status:** completed; V2 is the selected model ([../BEST_CHECKPOINT_TRAINING.md](../BEST_CHECKPOINT_TRAINING.md)). Since 2026-10-01 the V2 code lives in the main tree (`src/scglm_v2/`, `scripts/`, `tests/`); this directory holds V2's configs, data, runs, phase records and submission drafts. V1's checkpoints are preserved as baselines: checkpoints/competition_base (13B base) and checkpoints/best_sft.

V2 trains a new 46M-parameter model from random initialization. It keeps the V1 architecture and the AdamW recipe, and changes three things:

1. **Throughput.** The measured synthetic rate is 404k tokens/s (microbatch 64, `torch.compile`), versus 137k for V1 in practice. At a fixed wall-clock time this gives roughly 2.5× the tokens.
2. **Richer, human-written data.** DCLM, FineWeb-Edu, a benchmark-targeted slice, Wikipedia, StackExchange and Project Gutenberg, with stronger contamination controls.
3. **An English tokenizer trained on the actual mixture.** It is adopted only if it compresses text at least 2% better than the V1 tokenizer.

This directory holds V2's configs, data, runs and records. V1 files under `data/`, `runs/`, `checkpoints/` and `artifacts/` were only read.

| Path | Contents |
| --- | --- |
| [PLAN.md](PLAN.md) | Approved plan, evidence and decision rules |
| [configs/](configs/) | Preparation and training configurations |
| [../src/scglm_v2/](../src/scglm_v2/) | V2 code (moved from `v2/src/` on 2026-10-01; module names unchanged) |
| [phases/](phases/) | Per-phase registration, report, status and dated events |
| `data/` | V2 corpora (large; not for the repository) |
| `runs/` | V2 training runs: `pilot_1b`, `main` (selected), `anneal_quality` |
| [../scripts/supervise_main.sh](../scripts/supervise_main.sh) | Auto-resume supervisor for the main run (moved 2026-10-01) |
| `run_after_main.sh` (retired; kept on the training server) | Post-pretraining chain that ran on 2026-10-01 (archived with the decay A/B code) |
| [submission/](submission/) | Devpost and README drafts, Built With list, and figures (regenerate with `python -m scglm_v2.figures`) |
| `reports/rehearsal/` | End-to-end rehearsal of selection and official reporting on V1's 1B model (not a V2 result) |

Current commands are in [../CURRENT_PIPELINE.md](../CURRENT_PIPELINE.md). Records written before 2026-10-01 show `PYTHONPATH=src:v2/src`, which reflects the old code location.
