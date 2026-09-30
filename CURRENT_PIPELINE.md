# V2 pipeline: code map and reproduction

**Scope since 2026-10-01:** the active code builds, trains, selects and evaluates the V2 model, and nothing else. What it produced: [BEST_CHECKPOINT_TRAINING.md](BEST_CHECKPOINT_TRAINING.md). Retired code is under `archive/` ([inventory](reports/cleanup/2026-10-01/moved_files.json)); V1's pipeline guide is in [docs/history/V1_CURRENT_PIPELINE.md](docs/history/V1_CURRENT_PIPELINE.md).

Run everything from the project root with the recorded environment `/home/bufsgpu/yes/envs/sw/bin/python` ([requirements.lock.txt](requirements.lock.txt)). `PYTHONPATH=src` is enough for training and evaluation; data preparation also needs `vendor/pydeps` (fastText). `pytest` sets both.

## Setup (fresh clone)

The repository holds code, configurations, tests, documentation and small run and evaluation records. Model weights, prepared data (about 200 GB), full per-step logs and V1's `artifacts/` stay local, so some links in V1-era documents point to files that are not in the repository.

```bash
pip install -r requirements.lock.txt                       # recorded environment (PyTorch 2.10, Transformers 5.5.4, ...)
pip install --target vendor/pydeps fasttext-numpy2-wheel==0.9.2          # data preparation only
git clone https://github.com/EleutherAI/lm-evaluation-harness vendor/lm-evaluation-harness
git -C vendor/lm-evaluation-harness checkout 6d642546f4688648fced259eb3302efd36ece5af   # pinned by configs/evaluation.json
```

The weights (`checkpoints/v2_best/model.safetensors`, SHA256 `9aac2111…28011bd`) are distributed separately; place them in `checkpoints/v2_best/`, then run `scripts/verify_v2.py`.

## Code map

| Stage | Code | Configuration |
| --- | --- | --- |
| 1. Base corpus, byte-level BPE tokenizer, benchmark-exclusion index | `src/scglm/prepare_data.py` | [configs/data.json](configs/data.json) |
| 2. Extension corpus (URL/exact/MinHash dedup, filtering) | `src/scglm/prepare_extension.py` | [configs/data_extension.json](configs/data_extension.json) |
| 3a. Download new pinned sources | `src/scglm_v2/download.py` | [v2/configs/prep.json](v2/configs/prep.json) |
| 3b. Benchmark-train texts (classifier positives + held-out development items) | `src/scglm_v2/benchmarks.py` | same |
| 3c. Targeted-slice classifier | `src/scglm_v2/targeted.py` | same |
| 3d. Tokenizer decision (study; kept V1's) | `src/scglm_v2/tokenizer_study.py` | same |
| 3e. Corpus build: filter, dedup, route, tokenize, manifest | `src/scglm_v2/build.py`, `sources.py`, `common.py` | same |
| 4. Pretraining (WSD, compile) + auto-resume | `src/scglm/{model,data,train}.py`, `src/scglm_v2/train.py`, [scripts/supervise_main.sh](scripts/supervise_main.sh) | [v2/configs/main.json](v2/configs/main.json), [configs/model.json](configs/model.json) |
| 5a. Development-only endpoint selection | `src/scglm_v2/devevaluate.py`, `select_endpoint.py` | rule in [v2/phases/final_selection/registration.json](v2/phases/final_selection/registration.json) |
| 5b. Official evaluation (pinned protocol) | [scripts/run_final_evaluation.py](scripts/run_final_evaluation.py), `scripts/validation_history.py`, `src/scglm/evaluate.py`, `vendor/lm-evaluation-harness` | [configs/evaluation.json](configs/evaluation.json) |
| 5c. Comparison with the V1 baseline | `src/scglm_v2/v1_vs_v2.py` | same registration |
| 6. Demo, figures, verification | [scripts/demo.py](scripts/demo.py), `src/scglm_v2/figures.py`, [scripts/verify_v2.py](scripts/verify_v2.py) | — |

**Frozen files:**
- **The six `src/scglm` modules** are kept byte-identical to the hashes in `v2/runs/main/run.json`. The official protocol also pins `src/scglm/evaluate.py`.
- **`scripts/run_final_evaluation.py` and `validation_history.py`** are the runner whose hash is in every evaluation record. Its `--selection-record` help text still names V1's `scripts/select_endpoint.py`; V2 produces that record with `scglm_v2.select_endpoint`.
- **V1's CLI default:** `scglm.train` keeps `configs/train_baseline.json` as its default `--config`. V2 always calls `scglm_v2.train` with an explicit config.

## Verify (read-only)

```bash
PYTHONPATH=src python scripts/verify_v2.py --load   # hashes, selection, outcome, offline reload
python -m pytest                                     # CPU tests (77)
```

## Reproduce

Existing outputs (`data/processed/*`, `v2/data/rich_v1`, `v2/runs/*`, `v2/phases/*`) are evidence and must not be overwritten. For a fresh reproduction, copy each config and point `output_dir` or `run_dir` at a new directory.

```bash
export PYTHONPATH=src:vendor/pydeps HF_HUB_DISABLE_XET=1   # plain HTTP was faster than xet on this server
# Stages 1-2: V1 data lineage (tokenizer, exclusion index, extension text reused by V2)
python -m scglm.prepare_data --config configs/data.json
python -m scglm.prepare_extension --config configs/data_extension.json
# Stage 3: V2 corpus
python -m scglm_v2.download --config v2/configs/prep.json
python -m scglm_v2.benchmarks --config v2/configs/prep.json
python -m scglm_v2.targeted --config v2/configs/prep.json
python -m scglm_v2.tokenizer_study --config v2/configs/prep.json
python -m scglm_v2.build --config v2/configs/prep.json --stage reused --workers 36
python -m scglm_v2.build --config v2/configs/prep.json --stage new --workers 20
python -m scglm_v2.build --config v2/configs/prep.json --stage manifest
# Stage 4: pretraining on one GPU (about 32 h at ~395k tokens/s)
CUDA_VISIBLE_DEVICES=<gpu> python -m scglm_v2.train --config <copy of v2/configs/main.json with a new run_dir>
#   decay: write <run_dir>/decay.json atomically, e.g. {"decay_start_step": 617981, "decay_steps": 68664}
#   resume: add --resume <run_dir>/checkpoint_latest.json   (scripts/supervise_main.sh automates this for v2/runs/main)
# Stage 5: selection (development data only), official evaluation, V1 comparison
python -m scglm_v2.select_endpoint --baseline <run_dir> --out <new>/selection.json --device cuda
python scripts/run_final_evaluation.py --final-evaluation --model <run_dir>/export \
    --selection-record <new>/selection.json --device cuda:0 --output <fresh dir> --history-dir <history dir>
python -m scglm_v2.v1_vs_v2 --v2-summary <fresh dir>/summary.json --out <new>/v1_vs_v2.json
# Stage 6
python -m scglm_v2.figures
python scripts/demo.py --prompt "The water cycle begins when"
```

Actual execution of the selected model: [v2/phases/](v2/phases/) (registration, report, status and dated events per phase) and [TRAINING_PHASES.md](TRAINING_PHASES.md). In the frozen records, launch commands from before 2026-10-01 used `PYTHONPATH=src:v2/src` (the code was in `v2/src/scglm_v2/` then); module names are unchanged.
