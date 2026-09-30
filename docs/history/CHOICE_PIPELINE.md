# Choice-discrimination experiment

This isolated study implements the [registered protocol](../../reports/research/2026-09-29_choice_discrimination_protocol.md) and the fixed scope approved on September 29, 2026. It tests the added supervised choice loss against identical additional response SFT and raw replay. It does not change the retained selected-model trainer, its lineage validator, the update-105/209 reproduction recipe, or either checkpoint alias.

The [configuration](../../archive/v1_code/configs/choice_discrimination.json) contains the original protocol plus the two presentations, pinned evaluation sources, conservative preview exclusions and diagnostic-only transfer decision. The implementation lives in [src/scglm_choice](../../archive/v1_code/src/scglm_choice/); the entrypoint is [scripts/run_choice_pipeline.py](../../archive/v1_code/scripts/run_choice_pipeline.py).

Original protocol metadata such as `designed_not_launched` and pending-plan fields remains a historical snapshot in this configuration. The prepared manifest and phase `status.json` files record current readiness and execution; the configuration is never rewritten to update live status.

Run from the project root with the recorded environment:

```bash
PYTHONPATH=src:. PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 \
  /home/bufsgpu/yes/envs/sw/bin/python scripts/run_choice_pipeline.py verify --load
```

Omitting the subcommand performs read-only verification. `status` also reads without modifying experiment records. `prepare` freezes data and three complete paired plans; `preflight` runs mathematical/regression checks, a training-only no-update gradient test, disposable GPU recovery and export loading, and the unchanged official scorer's preparation path. `run` executes the gates in order. `resume` resumes those same contracts, memberships, optimizer/RNG state and cumulative resource ceilings. `report` refreshes the report from existing evidence.

```bash
PYTHONPATH=src:. PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 \
  /home/bufsgpu/yes/envs/sw/bin/python scripts/run_choice_pipeline.py run

PYTHONPATH=src:. PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 \
  /home/bufsgpu/yes/envs/sw/bin/python scripts/run_choice_pipeline.py resume
```

Execution selects only `GPU-78815613-815b-f56b-27c0-6bc239e093fc`, uses four CPU threads and acquires the existing `artifacts/pipeline_gpu0.lock`. All new data and outputs remain under the registered `choice_discrimination_20260929_v1` directories. No CLI flag can select an alternative recipe, endpoint, output root or seed.

The source/runtime and data contracts are immutable after successful preflight. A code or runtime change requires resolving the mismatch; it cannot silently resume a differently implemented experiment. A terminal failed or inconclusive study returns its recorded decision without retraining. A clean interruption at an update boundary saves state and remains resumable. A hard exit can roll back to the last checkpoint; all previously charged forwards remain spent. GPU wall time between the active-session timestamp and recovery is conservatively charged, including uncertain downtime, up to the reserved remaining ceiling. That conservative charge can make an interrupted arm ineligible. Budget exhaustion never authorizes an earlier endpoint.

Every actual/conditional phase is registered in [TRAINING_PHASES.md](../../TRAINING_PHASES.md) before execution. Linked phase reports contain exact parent and data hashes, recipe, ceilings, launch/resume commands, dated events and machine-readable status. Recovery checkpoints preserve model and AdamW state, RNG state, batch/replay position, committed counters and a resource snapshot. The independent write-ahead budget ledger is authoritative across rollback; attempt records distinguish actual work from committed checkpoint ancestry.

The preparation audit uses existing corpus overlap protections and a conservative union of historical prepared posttraining pools, replay and raw-evaluation identities. Original controller holdouts require evidence of no controller use. Continuation documents must begin strictly beyond the final consumed cursor and contain their real EOS; whole document identities are excluded from replay, even though scoring uses at most the first 1,023 targets. Exact text, 13-word overlap and the existing corpus MinHash protections do not establish complete semantic decontamination.

SciQ supporting passages supply grouping/overlap evidence but never model inputs. Social IQa includes the original context. The first 100 rows of every original split and matching known preview strings are excluded conservatively because the earlier browsing record did not retain exact preview IDs. Both panels keep whole connected source/fact groups. COPA keeps its official development/test assignments and removes bridging premises. Winograd groups all schemas sharing a candidate pair, conservatively keeping twins together. Both transfer tasks are diagnostic only. Historical instruction retention remains explicitly previously observed.

Only update 32 can advance. The fixed pilot seed runs control before candidate, followed conditionally by two replication seeds, one four-model confirmation claim, and unchanged official scoring for all six endpoints. Confirmation uses 10,000 paired source-group resamples with shared draws across models, Bonferroni-adjusted primary lower bounds, domain lower bounds and per-stratum raw-NLL upper bounds. Failed/inconclusive outcomes do not schedule another recipe.

The [live/final report](../../artifacts/choice_discrimination_20260929_v1/REPORT.md) and [machine-readable report](../../artifacts/choice_discrimination_20260929_v1/report.json) state the actual outcome. They must not be read as evidence of efficacy until the required gates have completed. Historical base/parent benchmark results retain their previously-observed disclosure; official scores cannot reopen selection.
