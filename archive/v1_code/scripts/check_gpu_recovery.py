"""Exercise real GPU checkpoint/reload/export plumbing on disposable artificial IDs."""
import argparse
import json
import os
from pathlib import Path
import tempfile
import time

import numpy as np
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
import torch
from transformers import AutoTokenizer

from scglm.model import load_model
from scglm.train import Trainer, read_config, sha256_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--extension", action="store_true", help="Inherit the real scratch parent, with disposable three-source data")
    args = parser.parse_args()
    if args.deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True)
    start = time.monotonic()
    output = Path("artifacts/extension_gpu_recovery.json" if args.extension else "artifacts/gpu_recovery.json")
    output.parent.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="gpu_recovery_", dir=output.parent) as directory:
        root = Path(directory).resolve()
        vocabulary = {f"<|{name}|>": i for i, name in enumerate(["pad", "bos", "eos", "unk"])}
        vocabulary.update({f"synthetic{i}": i for i in range(4, 16384)})
        tok = root / "tokenizer.json"
        Tokenizer(WordLevel(vocabulary, unk_token="<|unk|>")).save(str(tok))
        if args.extension:
            parent_manifest = json.loads(Path("data/processed/main/manifest.json").read_text())
            tok = Path(parent_manifest["tokenizer"]["path"])
        manifest = {"status": "complete", "tokenizer": {"path": str(tok), "sha256": sha256_file(tok)}, "sources": {}}
        rng = np.random.default_rng(909)
        for source in (("edu", "dclm", "wiki") if args.extension else ("web", "wiki")):
            path = root / f"{source}.bin"
            rng.integers(4, 16384, 1000000, dtype=np.uint16).tofile(path)
            manifest["sources"][source] = {"splits": {"train": {"path": str(path), "sha256": sha256_file(path)}}}
        manifest_path = root / "manifest.json"
        manifest_path.write_text(json.dumps(manifest))
        config = {**read_config("configs/train_extension.json" if args.extension else "configs/train_baseline.json"), "data_manifest": str(manifest_path),
                  "run_dir": str(root / "run"), "validation_every": 0}
        original = Trainer(config)
        parent_weight_difference = None
        if args.extension:
            parent_payload = torch.load(config["parent_checkpoint"], map_location="cpu", weights_only=False)
            parent_weight_difference = max(float((v.detach().cpu()-parent_payload["model"][k]).abs().max()) for k,v in original.model.state_dict().items())
            assert parent_weight_difference == 0
            for key, values in parent_payload["optimizer"]["state"].items():
                for name, value in values.items():
                    actual = original.optimizer.state_dict()["state"][key][name]
                    if torch.is_tensor(value):
                        assert torch.equal(value.cpu(), actual.cpu()), "Parent Adam moments/step changed"
            del parent_payload
        for _ in range(2):
            original.update(0.8)
        original.save_checkpoint("disposable_gpu_recovery_check")
        parent_state = original.state()
        expected_loss, _, _ = original.update(0.8)
        expected_gradients = {k: v.grad.detach().cpu().clone() for k, v in original.model.named_parameters()}
        expected = {k: v.detach().cpu().clone() for k, v in original.model.state_dict().items()}
        expected_counts = dict(original.main_source_tokens)
        del original
        torch.cuda.empty_cache()
        restored = Trainer(config, resume=root / "run" / "checkpoint_latest.json")
        def equal_state(left, right):
            if torch.is_tensor(left):
                assert torch.equal(left.cpu(), right.cpu()), "Restored tensor differs before the next update"
            elif isinstance(left, dict):
                assert left.keys() == right.keys()
                for key in left:
                    equal_state(left[key], right[key])
            elif isinstance(left, (tuple, list)):
                assert len(left) == len(right)
                for x, y in zip(left, right):
                    equal_state(x, y)
            elif isinstance(left, np.ndarray):
                assert np.array_equal(left, right)
            else:
                assert left == right
        equal_state(parent_state, restored.state())
        print("Pre-update model, optimizer, RNG and stream state are exactly restored.", flush=True)
        actual_loss, _, _ = restored.update(0.8)
        differences = [float((restored.model.state_dict()[k].detach().cpu()-v).abs().max()) for k, v in expected.items()]
        maximum_difference = max(differences)
        maximum_gradient_difference = max(float((v.grad.detach().cpu()-expected_gradients[k]).abs().max()) for k,v in restored.model.named_parameters())
        print(json.dumps({"expected_loss": expected_loss, "actual_loss": actual_loss,
                          "maximum_parameter_difference": maximum_difference,
                          "maximum_gradient_difference": maximum_gradient_difference,
                          "parameters_with_difference": sum(v > 0 for v in differences)}), flush=True)
        assert expected_counts == restored.main_source_tokens
        assert actual_loss == expected_loss and maximum_difference == 0 and maximum_gradient_difference == 0
        restored.export()
        tokenizer = AutoTokenizer.from_pretrained(root / "run" / "export", local_files_only=True)
        assert len(tokenizer) == 16384
        assert [tokenizer.pad_token_id, tokenizer.bos_token_id, tokenizer.eos_token_id] == [0, 1, 2]
        exported = load_model(root / "run" / "export")
        export_difference = max(float((exported.state_dict()[k]-restored.model.state_dict()[k].detach().cpu()).abs().max()) for k in expected)
        assert export_difference == 0
        result = {"status": "passed", "purpose": "Disposable synthetic GPU checkpoint/resume/export check; weights discarded.",
                  "gpu": torch.cuda.get_device_name(), "parameter_count": restored.parameter_count,
                  "expected_loss": expected_loss, "resumed_loss": actual_loss,
                  "maximum_parameter_difference": maximum_difference,
                  "maximum_gradient_difference": maximum_gradient_difference,
                  "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                  "export_parameter_difference": export_difference, "exported_tokenizer_size": len(tokenizer),
                  "synthetic_optimizer_updates_executed": 4, "synthetic_target_tokens_executed": 4*65536,
                  "parent_weight_difference": parent_weight_difference,
                  "checked_three_source_continuation": args.extension,
                  "elapsed_seconds": time.monotonic()-start}
        output.write_text(json.dumps(result, indent=2)+"\n")
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
