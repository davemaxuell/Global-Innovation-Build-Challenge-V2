"""Select only the registered H100 and fix deterministic numerics before CUDA starts."""
import os
import subprocess


def select_assigned_gpu(uuid):
    listing = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,name", "--format=csv,noheader"], text=True)
    rows = [line.split(", ", 2) for line in listing.splitlines()]
    if not any(u == uuid and "H100" in name for _, u, name in rows):
        raise ValueError("Registered H100 UUID is unavailable")
    mapping = {index: u for index, u, _ in rows}
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", uuid).split(",")
    if uuid not in [mapping.get(v.strip(), v.strip()) for v in visible]:
        raise ValueError("Assigned H100 excluded from current allocation")
    os.environ["CUDA_VISIBLE_DEVICES"] = uuid
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def process_environment(threads=4):
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = str(threads)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
