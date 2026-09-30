# Decay A/B branch: `v2_anneal_quality_20260929`

**Status:** completed 2026-10-01 07:23 KST. **Outcome: not selected.** No development difference was established. [Registration](registration.json) · [frozen config](config.json)

The branch re-ran the main run's final 10% decay from its decay-start milestone (SHA256 `175d1e29…b4bb0c6`) with DCLM 25 / Edu 28 / targeted 25 / Wiki 12 / StackExchange 5 / books 5. It ran 68,664 updates, 4,499,963,904 tokens (DCLM 1.123B, Edu 1.261B, targeted 1.126B, Wiki 0.540B, StackExchange 0.224B, books 0.226B), in 3.19 h with no errors. The initial LR matched the main schedule at the branch point exactly. Export SHA256 `cbb7d03c…f6183cc9`.

**Held-out monitor NLL at matched positions** (branch vs main): 3.132 vs 3.143, 3.112 vs 3.118, 3.088 vs 3.085, 3.050 vs 3.049, 3.014 vs 3.016, 2.968 vs 2.971. **Frozen development selection:** proxy +0.07 points against a required +0.5, and Wikipedia bits/byte −0.0046. Official (reported, not used for selection): four-task mean 47.08% vs 47.12% for the selected model; WikiText-103 perplexity 23.56 vs 23.85.

**Interpretation:** at 46M parameters, shifting the decay-phase mix toward targeted, Edu and Wikipedia text made no established difference to the benchmark proxies. See `FAILED_APPROACHES.md` A13.
