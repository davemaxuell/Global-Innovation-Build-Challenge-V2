"""Score multiple-choice answers the way lm-evaluation-harness does, for V2 and V2.1 side by side.

Each answer is scored by its summed continuation log-likelihood after the question, in the
harness's own prompt format (src/scglm_v2/task_format.py); the highest score is the model's
choice. The three questions were written for this demo before it was run and are not taken
from any evaluation set. They illustrate the scoring; they are not evidence of accuracy.

Run: PYTHONPATH=src python scripts/demo_choices.py [--out v2/submission/demo_choices.txt]
"""
from __future__ import annotations

import argparse
from pathlib import Path

from transformers import PreTrainedTokenizerFast

from scglm.model import load_model
from scglm_v2.task_format import encode_pair, sequence_logprobs

ROOT = Path(__file__).resolve().parents[1]
ITEMS = [  # (context in harness format, answer choices, index of the sensible answer)
    ("Question: How do you keep ice cream from melting on a hot day?\nAnswer:",
     ["Put it in a cooler packed with ice.", "Put it in a sunny window."], 0),
    ("Question: Which gas do plants take in from the air to make their food?\nAnswer:",
     ["carbon dioxide", "oxygen", "helium", "neon"], 0),
    ("Making tea: A woman fills a kettle with water and puts it on the stove. She",
     ["waits for the water to boil and pours it over a tea bag.", "throws the kettle into the garden and starts painting.",
      "reads a newspaper about football in the snow.", "takes the stove apart with a hammer."], 0),
]
MODELS = (("V2 (pretraining only)", "checkpoints/v2_best"), ("V2.1 (submitted)", "checkpoints/v2_1"))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, help="also write the printed table to this file")
    args = p.parse_args()
    lines = []
    for name, path in MODELS:
        model = load_model(ROOT / path).eval()
        tok = PreTrainedTokenizerFast.from_pretrained(str(ROOT / path), local_files_only=True)
        lines.append(f"== {name}")
        right = 0
        for ctx, choices, gold in ITEMS:
            scores = sequence_logprobs(model, [encode_pair(tok, ctx, " " + c) for c in choices], "cpu")
            best = max(range(len(scores)), key=scores.__getitem__)
            right += best == gold
            lines.append("  " + ctx.split("\n")[0].replace("Question: ", ""))
            for i, (c, s) in enumerate(zip(choices, scores)):
                lines.append(f"    {s:8.2f}  {c:58s}{'  <- chosen' if i == best else ''}")
        lines.append(f"  sensible answer chosen: {right} of {len(ITEMS)}")
    print("\n".join(lines))
    if args.out:
        args.out.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
