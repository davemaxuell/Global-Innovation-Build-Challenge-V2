"""Apply the pre-registered V1-vs-V2 rule to official summaries (no discretion).

Rule (v2/phases/final_selection/registration.json, step_3): the selected V2 export
replaces V1 if its official four-task mean raw accuracy is > V1's AND its
WikiText-103 perplexity is <= V1's. If exactly one condition holds, the user decides.

Run: PYTHONPATH=src python -m scglm_v2.v1_vs_v2 --v2-summary <summary.json> --out <decision.json>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .common import ROOT, sha256_file, write_json

TASKS = ("hellaswag", "arc_easy", "piqa", "winogrande")
V1_SUMMARY = ROOT / "artifacts/phase_evaluation_v1/attempts/official_13b/01/output/summary.json"


def headline(summary: dict) -> dict:
    b = summary["metrics"]["benchmarks"]
    return {"four_task_mean_acc": sum(b[t]["acc"] for t in TASKS) / len(TASKS),
            "per_task_acc": {t: b[t]["acc"] for t in TASKS},
            "wikitext103_perplexity": summary["metrics"]["wikitext103"]["perplexity"]}


def decide(v1: dict, v2: dict) -> str:
    better_acc = v2["four_task_mean_acc"] > v1["four_task_mean_acc"]
    not_worse_ppl = v2["wikitext103_perplexity"] <= v1["wikitext103_perplexity"]
    if better_acc and not_worse_ppl:
        return "V2"
    if not better_acc and not not_worse_ppl:
        return "V1"
    return "USER_DECIDES"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--v2-summary", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    v1 = headline(json.loads(V1_SUMMARY.read_text()))
    v2 = headline(json.loads(args.v2_summary.read_text()))
    outcome = decide(v1, v2)
    write_json(args.out, {"outcome": outcome, "v1_13b_base": v1, "v2_selected": v2,
                          "v1_summary": str(V1_SUMMARY), "v1_summary_sha256": sha256_file(V1_SUMMARY),
                          "v2_summary": str(args.v2_summary), "v2_summary_sha256": sha256_file(args.v2_summary),
                          "rule": "V2 if four-task mean acc > V1 AND WikiText-103 perplexity <= V1; V1 if neither; otherwise user decides"})
    print(json.dumps({"outcome": outcome, "v1": v1["four_task_mean_acc"], "v2": v2["four_task_mean_acc"],
                      "v1_ppl": v1["wikitext103_perplexity"], "v2_ppl": v2["wikitext103_perplexity"]}))


if __name__ == "__main__":
    main()
