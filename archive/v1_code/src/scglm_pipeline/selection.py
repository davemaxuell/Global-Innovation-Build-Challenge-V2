"""Paired raw-document uncertainty for retained model evaluation."""
import numpy as np


def paired_raw_intervals(base, candidate, seed, replicates):
    a = {(r["source"], r["id"]): r for r in base}
    b = {(r["source"], r["id"]): r for r in candidate}
    if len(a) != len(base) or len(b) != len(candidate) or a.keys() != b.keys():
        raise ValueError("Raw document identities differ")
    rng = np.random.default_rng(seed)
    result = {}
    for source in sorted({key[0] for key in a}):
        keys = sorted(key for key in a if key[0] == source)
        if any(a[key]["token_count"] != b[key]["token_count"] for key in keys):
            raise ValueError("Raw document target counts differ")
        counts = np.array([a[k]["token_count"] for k in keys])
        diffs = np.array([b[k]["nll_sum"]-a[k]["nll_sum"] for k in keys])
        if counts.sum() <= 0:
            raise ValueError("No raw targets")
        indexes = rng.integers(0, len(keys), (replicates, len(keys)))
        boot = diffs[indexes].sum(1)/counts[indexes].sum(1)
        result[source] = {"nll_change": float(diffs.sum()/counts.sum()),
                          "lower_95": float(np.quantile(boot, .025)), "upper_95": float(np.quantile(boot, .975))}
    return result
