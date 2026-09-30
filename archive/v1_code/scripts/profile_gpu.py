"""Bounded synthetic hardware check; its weights are never used for pretraining."""
import argparse
import json
import os
from pathlib import Path
import time

import torch

from scglm.model import build_model, audit_parameters, forward_loss
from scglm.train import configure_numerics, runtime_fingerprint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=int, default=120)
    parser.add_argument("--output", default="artifacts/gpu_profile.json")
    args = parser.parse_args()
    configure_numerics({"deterministic": True}, "cuda")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Expose exactly one GPU with CUDA_VISIBLE_DEVICES")
    torch.set_num_threads(8)
    torch.manual_seed(20260923)
    torch.backends.cuda.matmul.allow_tf32 = True
    model = build_model(json.loads(Path("configs/model.json").read_text())).cuda().train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=6e-4, fused=True)
    batch, microbatch, context = 64, 8, 1024
    tokens = torch.randint(model.config.vocab_size, (batch, context + 1), device="cuda")
    durations, losses = [], []
    started = time.monotonic()
    while len(durations) < 5 or time.monotonic() - started < args.seconds:
        torch.cuda.synchronize()
        before = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.0
        for i in range(0, batch, microbatch):
            block = tokens[i:i+microbatch]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = forward_loss(model, block[:, :-1], block[:, 1:])
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite loss in GPU profile")
            (loss / (batch // microbatch)).backward()
            loss_sum += loss.detach().item() / (batch // microbatch)
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        torch.cuda.synchronize()
        durations.append(time.monotonic() - before)
        losses.append(loss_sum)
        if len(durations) % 10 == 0 or len(durations) == 1:
            print(json.dumps({"step": len(durations), "seconds": durations[-1], "loss": loss_sum,
                              "gradient_norm": float(grad)}), flush=True)
    steady = durations[3:]
    throughput = len(steady) * batch * context / sum(steady)
    result = {
        "purpose": "Synthetic throughput/memory smoke test only. Discarded weights; not a language-model result.",
        "device": torch.cuda.get_device_name(), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        **audit_parameters(model), "torch": torch.__version__, "cuda": torch.version.cuda,
        "microbatch_sequences": microbatch, "batch_sequences": batch, "sequence_length": context,
        "numerics": runtime_fingerprint()["numerics"],
        "steps": len(durations), "synthetic_target_tokens": len(durations)*batch*context,
        "elapsed_seconds": time.monotonic()-started,
        "steady_target_tokens_per_second": throughput,
        "peak_allocated_gib": torch.cuda.max_memory_allocated()/1024**3,
        "peak_reserved_gib": torch.cuda.max_memory_reserved()/1024**3,
        "one_billion_train_only_hours_estimate": 1e9/throughput/3600,
        "estimate_excludes": "Data preparation, evaluation, checkpoints, probes and any hardware contention.",
        "first_loss": losses[0], "last_loss": losses[-1],
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
