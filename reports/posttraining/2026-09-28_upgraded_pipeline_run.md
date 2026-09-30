# Upgraded post-training run verification

Observed **2026-09-28 11:35:40 KST**. This is a dated run observation; the linked live files take precedence.

The workflow described in [POSTTRAINING.md](../../artifacts/posttraining_sota_v1/candidate_package/POSTTRAINING.md) was already running when reviewed. Its persistent `gibc-post-sota` session started at **11:26:18 KST**, with Python PID `1484362`, and is configured to continue through local candidate packaging. No duplicate training process was launched during this review.

## Execution verified

- The completed scratch parent has **46,346,752 parameters** and **13,000,048,640 pretraining targets**. Parent weight SHA256: `ffa75b07ec377167ab2e4d4263f2fa8374549f8592f9ce2e8166875cab1f38e7`.
- Training is on the assigned H100 NVL, UUID `GPU-78815613-815b-f56b-27c0-6bc239e093fc`, exposed as `cuda:0`.
- The base development evaluation finished. The first SFT pilot reached **24 of 31 optimizer updates**, consuming **197,072 of 250,000 response targets**. Its recorded optimizer metrics were finite. These training observations do not establish a quality improvement.
- The launch command uses `scripts/run_pipeline.py run --config configs/posttraining_sota/default.json --through package --device cuda:0` with the existing `/home/bufsgpu/yes/envs/sw/bin/python` environment.

## Registration and validation

Independent read-only verification passed for all **45 registered source files**, the configuration, parent identity, source snapshots, and **10 checksummed data files**. Current source hashes and the data manifest match the [implementation validation record](../../artifacts/posttraining_sota_verification/IMPLEMENTATION_VALIDATION.json).

That matching validation record reports **212 CPU tests passed**, with the allocated-GPU test skipped in the CPU suite and run separately. The disposable GPU kernel smoke check and exact checkpoint-resume check passed. The latter verified identical weights, replay cursors, and counters after recovery. These tests were not rerun against the occupied GPU during this review.

The dataset contains **118,429 training examples**, with **7,862 development** and **7,862 confirmation examples**. Checksum verification does not certify semantic decontamination. No confirmation outcomes were scored during this review.

## Automated route

1. Compare nine SFT pilots across three learning rates and three raw-replay weights.
2. Train eligible shortlisted recipes across three seeds, with development evaluations and matched controls.
3. Collect verified own-policy preferences; run DPO and its positive-only control when their data and parent checks pass.
4. Attempt online RL only where all-seed readiness and retention checks pass; record explicit skips otherwise.
5. Select using development evidence, perform the registered single-candidate confirmation when eligible, report the unchanged required benchmarks, and create a separate local candidate package.

The [frozen experiment matrix](../../artifacts/posttraining_sota_v1/experiment_matrix.json) records a realized full SFT capacity of **1,715,410 response targets** under the mixture and two-epoch limit. Four million is the configured ceiling. The existing base remains the competition checkpoint unless a candidate passes selection.

## Live records

- [Study status](../../artifacts/posttraining_sota_v1/status.json)
- [First SFT pilot status](../../artifacts/posttraining_sota_v1/runs/sft_pilot_0/status.json)
- [Runner log](../../artifacts/posttraining_sota_v1/runner.log)
- [Launch record](../../artifacts/posttraining_sota_v1/launch.json)
- [Registered configuration](../../artifacts/posttraining_sota_v1/snapshot/configuration.json)

The tmux process continues independently of this conversation. An interruption requires resuming the identical registered command after checking process ownership; this workflow does not provide an automatic crash-restart supervisor.
