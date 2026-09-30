"""Materialize a branch config from a template once the parent milestone exists.

Fills ``parent_checkpoint_sha256`` from the parent's milestone record after
re-hashing the file, resolves ``inherits``, and writes the frozen config that the
launch command uses. Refuses to overwrite an existing materialized config.

Run: PYTHONPATH=src:v2/src python -m scglm_v2.branch --template v2/configs/anneal_quality.json \
       --out v2/phases/anneal_quality/config.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from scglm import train as v1


def materialize(template: Path, out: Path) -> dict:
    config = v1.read_config(template)
    checkpoint = Path(config["parent_checkpoint"])
    record = json.loads(checkpoint.with_suffix(".json").read_text())
    actual = v1.sha256_file(checkpoint)
    if actual != record["sha256"]:
        raise ValueError("Milestone file hash differs from its record")
    config["parent_checkpoint_sha256"] = actual
    config["materialized_from"] = str(template)
    if out.exists():
        raise FileExistsError(f"{out} exists; a branch config is frozen once written")
    v1.atomic_json(out, config)
    return config


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--template", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    config = materialize(args.template, args.out)
    print(json.dumps({"out": str(args.out), "parent_checkpoint": config["parent_checkpoint"],
                      "parent_checkpoint_sha256": config["parent_checkpoint_sha256"]}))


if __name__ == "__main__":
    main()
