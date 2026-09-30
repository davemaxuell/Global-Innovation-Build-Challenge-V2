"""Record illustrative completion checks on a local model, without training.

These hand-written prompts are diagnostics, not a held-out benchmark. Every
response is retained. Official evaluation and model selection are not rerun.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

CASES = [
    ("explanation", "A language model is a computer program that"),
    ("science_completion", "Plants use photosynthesis to"),
    ("story_completion", "Mira opened the old notebook and found"),
    ("geography_fact", "The capital city of France is"),
    ("astronomy_fact", "The largest planet in the Solar System is"),
    ("science_fact", "At standard atmospheric pressure, water boils at"),
    ("grounded_extraction", "The parcel belongs to Nira. Its tracking code is VX-734. The parcel's tracking code is"),
    ("arithmetic_pattern", "1 + 1 = 2\n2 + 2 = 4\n3 + 3 ="),
    ("word_problem", "A box contains 7 red marbles and 5 blue marbles. The total number of marbles is"),
    ("json_instruction", 'Return only a JSON object with the key color and value blue.\nAnswer:'),
    ("summary_instruction", "Text: Asha missed the bus, so she walked to school.\nSummary in one sentence:"),
    ("code_completion", "def add(a, b):\n    return"),
]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(2 ** 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=ROOT / "runs/continuation_12b/export")
    parser.add_argument("--device", choices=["cpu", "cuda:0"], default="cpu")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.device == "cuda:0" and os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        parser.error("Set CUDA_VISIBLE_DEVICES=0 to use the assigned physical GPU only")
    output = args.output or ROOT / "artifacts/model_tests" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    record = {
        "status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
        "model": str(args.model.resolve()), "device": args.device,
        "script_sha256": sha256(__file__), "argv": sys.argv,
        "decode": {"do_sample": False, "max_new_tokens": 64, "use_cache": True,
                   "add_special_tokens": False, "chat_template": False, "dtype": "float32"},
        "limitations": ["Twelve illustrative prompts; no contamination audit or representative accuracy claim.",
                        "Timing is a single-process smoke measurement, not a serving benchmark.",
                        "The base checkpoint has not been instruction tuned."],
    }
    write_json(output / "run.json", record)
    write_json(output / "prompts.json", [{"id": key, "prompt": prompt} for key, prompt in CASES])
    gpu_lock = None
    try:
        if args.device.startswith("cuda"):
            gpu_lock = (ROOT / "artifacts/pipeline_gpu0.lock").open("a")
            fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

        import torch
        import transformers
        from transformers import AutoTokenizer
        from scglm.model import load_model, audit_parameters

        torch.set_num_threads(4)
        torch.manual_seed(20260926)
        torch.backends.cuda.matmul.allow_tf32 = False
        before = {p.name: sha256(p) for p in sorted(args.model.iterdir()) if p.is_file()}
        model = load_model(args.model).to(args.device).eval()
        tokenizer = AutoTokenizer.from_pretrained(str(args.model), local_files_only=True)
        record["runtime"] = {"torch": torch.__version__, "transformers": transformers.__version__}
        record["parameter_audit"] = audit_parameters(model)
        record["artifact_sha256"] = before
        record["model_source_sha256"] = sha256(ROOT / "src/scglm/model.py")
        record["all_weights_finite"] = all(bool(torch.isfinite(p).all()) for p in model.parameters())
        if not record["all_weights_finite"]:
            raise ValueError("Nonfinite model weights")
        if args.device.startswith("cuda"):
            record["gpu_name"] = torch.cuda.get_device_name(0)
            torch.cuda.reset_peak_memory_stats()

        def sync():
            if args.device.startswith("cuda"):
                torch.cuda.synchronize()

        @torch.inference_mode()
        def generate(prompt, max_new_tokens=64):
            values = tokenizer.encode(prompt, add_special_tokens=False)
            if not values or len(values) + max_new_tokens > model.config.max_position_embeddings:
                raise ValueError("Prompt is empty or exceeds the context budget")
            ids = torch.tensor([values], device=args.device)
            sync()
            start = time.perf_counter()
            result = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                                    do_sample=False, max_new_tokens=max_new_tokens, use_cache=True,
                                    pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
            sync()
            elapsed = time.perf_counter() - start
            new_ids = result[0, len(values):].tolist()
            return {"prompt_tokens": len(values), "generated_ids": new_ids,
                    "generated_tokens": len(new_ids),
                    "response": tokenizer.decode(new_ids, skip_special_tokens=True),
                    "ended_eos": bool(new_ids and new_ids[-1] == tokenizer.eos_token_id),
                    "elapsed_seconds": elapsed, "tokens_per_second": len(new_ids) / elapsed}

        generate("The", max_new_tokens=4)  # Warm-up; not a scored prompt.
        rows = []
        with (output / "responses.jsonl").open("w") as handle:
            for key, prompt in CASES:
                row = {"id": key, "prompt": prompt, **generate(prompt)}
                rows.append(row)
                handle.write(json.dumps(row) + "\n")
                handle.flush()
                print(json.dumps({"id": key, "response": row["response"]}), flush=True)
        record["deterministic_repeat_matches"] = generate(CASES[0][1])["generated_ids"] == rows[0]["generated_ids"]

        # Full configured context: check numerical operation, not long-context comprehension.
        values = tokenizer.encode("The quick brown fox jumps over the lazy dog. ", add_special_tokens=False)
        limit = model.config.max_position_embeddings
        ids = torch.tensor([(values * (limit // len(values) + 1))[:limit]], device=args.device)
        with torch.inference_mode():
            logits = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).logits
            record["full_context_check"] = {"tokens": limit, "logits_shape": list(logits.shape),
                                             "all_logits_finite": bool(torch.isfinite(logits).all())}
        if not record["full_context_check"]["all_logits_finite"]:
            raise ValueError("Nonfinite full-context logits")
        if args.device.startswith("cuda"):
            record["peak_allocated_gpu_mib"] = torch.cuda.max_memory_allocated() / 2 ** 20
        record["nonempty_responses"] = sum(bool(r["response"].strip()) for r in rows)
        record["eos_responses"] = sum(r["ended_eos"] for r in rows)
        record["generated_tokens"] = sum(r["generated_tokens"] for r in rows)
        record["generation_seconds"] = sum(r["elapsed_seconds"] for r in rows)
        after = {p.name: sha256(p) for p in sorted(args.model.iterdir()) if p.is_file()}
        record["model_files_unchanged"] = before == after
        if not record["model_files_unchanged"] or not record["deterministic_repeat_matches"]:
            raise ValueError("Identity or deterministic replay check failed")
        record["status"] = "completed"
    except BaseException:
        record["status"] = "failed"
        record["error"] = traceback.format_exc()
        raise
    finally:
        record["finished_utc"] = datetime.now(timezone.utc).isoformat()
        record["elapsed_seconds"] = time.perf_counter() - started
        write_json(output / "run.json", record)
        if gpu_lock is not None:
            gpu_lock.close()
        print(f"Recorded diagnostic: {output}", flush=True)


if __name__ == "__main__":
    main()
