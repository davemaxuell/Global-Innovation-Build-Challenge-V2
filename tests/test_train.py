"""CPU integration tests for accumulation and exact checkpoint continuation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel

from scglm.train import Trainer


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def training_config(tmp_path):
    vocabulary = {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3}
    vocabulary.update({f"word{i}": i for i in range(4, 64)})
    tokenizer_path = tmp_path / "tokenizer.json"
    Tokenizer(WordLevel(vocabulary, unk_token="<unk>")).save(str(tokenizer_path))
    manifest = {
        "tokenizer": {"path": str(tokenizer_path), "sha256": _sha(tokenizer_path)},
        "sources": {},
    }
    for source, offset in (("web", 0), ("wiki", 13)):
        path = tmp_path / f"{source}.bin"
        tokens = ((np.arange(1024, dtype=np.uint16) + offset) % 60 + 4).astype(np.uint16)
        tokens.tofile(path)
        manifest["sources"][source] = {"splits": {"train": {
            "path": str(path), "sha256": _sha(path), "tokens": len(tokens),
            "num_tokens": len(tokens), "token_count": len(tokens),
        }}}
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    model_path = tmp_path / "model.json"
    model_path.write_text(json.dumps({
        "vocab_size": 64, "hidden_size": 32, "intermediate_size": 80,
        "num_hidden_layers": 2, "num_attention_heads": 4,
        "max_position_embeddings": 8,
    }))
    return {
        "seed": 1234,
        "model_config": str(model_path),
        "data_manifest": str(manifest_path),
        "run_dir": str(tmp_path / "run"),
        "arm": "B0",
        "target_tokens": 128,
        "sequence_length": 8,
        "batch_sequences": 4,
        "microbatch_sequences": 2,
        "web_share": 0.8,
        "learning_rate": 6e-4,
        "min_lr_ratio": 0.1,
        "warmup_fraction": 0.02,
        "betas": [0.9, 0.95],
        "adam_epsilon": 1e-8,
        "weight_decay": 0.1,
        "gradient_clip": 1.0,
        "checkpoint_every": 10,
        "log_every": 10,
        "validation_every": 0,
        "max_wall_hours": 1,
        "compile": False,
        "controller": False,
        "max_source_epochs": 1.0,
    }


def test_accumulation_matches_one_effective_batch(training_config, tmp_path):
    accumulated = Trainer(training_config, device="cpu")
    full_batch_config = {
        **training_config, "run_dir": str(tmp_path / "full_batch"),
        "microbatch_sequences": training_config["batch_sequences"],
    }
    full_batch = Trainer(full_batch_config, device="cpu")
    loss_a, norm_a, lr_a = accumulated.update(0.8)
    loss_b, norm_b, lr_b = full_batch.update(0.8)
    assert loss_a == pytest.approx(loss_b, rel=1e-6)
    assert norm_a == pytest.approx(norm_b, rel=1e-5)
    assert lr_a == lr_b
    assert accumulated.main_source_tokens == full_batch.main_source_tokens
    assert accumulated.main_tokens == full_batch.main_tokens == 32
    for left, right in zip(accumulated.model.parameters(), full_batch.model.parameters()):
        torch.testing.assert_close(left.grad, right.grad, atol=5e-8, rtol=1e-4)
        # FP32 reduction order differs between batch sizes; Adam can amplify a
        # tiny gradient difference when a coordinate is close to zero.
        torch.testing.assert_close(left, right, atol=1e-6, rtol=5e-5)


def test_checkpoint_resume_reproduces_uninterrupted_updates(training_config):
    original = Trainer(training_config, device="cpu")
    for _ in range(2):
        original.update(0.8)
    original.save_checkpoint(reason="test_interruption")
    expected_losses = [original.update(0.8)[0] for _ in range(2)]
    expected_state = original.state()
    restored = Trainer(
        training_config, device="cpu",
        resume=original.run_dir / "checkpoint_latest.json",
    )
    actual_losses = [restored.update(0.8)[0] for _ in range(2)]
    assert actual_losses == expected_losses
    assert restored.step == original.step == 4
    assert restored.main_tokens == original.main_tokens == 128
    assert restored.main_source_tokens == original.main_source_tokens
    assert sum(restored.main_source_tokens.values()) == restored.main_tokens
    for name, expected in expected_state["model"].items():
        torch.testing.assert_close(restored.model.state_dict()[name], expected, atol=0, rtol=0)
    # The next source selections and source cursors must also agree after resume.
    expected_x, expected_y, expected_counts = original.batch(0.8)
    actual_x, actual_y, actual_counts = restored.batch(0.8)
    np.testing.assert_array_equal(actual_x, expected_x)
    np.testing.assert_array_equal(actual_y, expected_y)
    assert actual_counts == expected_counts


@pytest.mark.parametrize("change", [{"learning_rate": 1e-3}, {"microbatch_sequences": 4}])
def test_resume_rejects_changed_scientific_settings(training_config, tmp_path, change):
    original = Trainer(training_config, device="cpu")
    original.update(0.8)
    checkpoint = original.save_checkpoint()
    changed = {
        **training_config, "run_dir": str(tmp_path / "changed"),
        **change,
    }
    with pytest.raises(ValueError, match="different scientific configuration"):
        Trainer(changed, device="cpu", resume=checkpoint)


def test_resume_rejects_corrupted_checkpoint_before_loading(training_config):
    original = Trainer(training_config, device="cpu")
    original.update(0.8)
    checkpoint = original.save_checkpoint()
    with checkpoint.open("ab") as stream:
        stream.write(b"corrupt")
    with pytest.raises(ValueError, match="integrity check failed"):
        Trainer(
            training_config, device="cpu",
            resume=original.run_dir / "checkpoint_latest.json",
        )


def test_source_epoch_budget_blocks_an_update(training_config):
    trainer = Trainer({**training_config, "max_source_epochs": 0.02}, device="cpu")
    # All four eight-token examples request web tokens: 32 > 2% of 1023.
    with pytest.raises(RuntimeError, match="epoch budget exceeded"):
        trainer.update(1.0)
    assert trainer.step == trainer.main_tokens == 0


def test_epoch_accounting_uses_stream_cursor_after_update(training_config):
    trainer = Trainer({**training_config, "max_source_epochs": 0.05}, device="cpu")
    trainer.update(1.0)
    assert trainer.main_source_tokens["web"] == 32
    assert trainer.streams["web"].tokens_consumed == 32
    with pytest.raises(RuntimeError, match="epoch budget exceeded"):
        trainer.update(1.0)
    assert trainer.streams["web"].tokens_consumed == 32
    assert trainer.main_tokens == 32
    assert trainer.step == 1


@pytest.mark.parametrize("share", [float("nan"), float("inf"), -0.1, 1.1])
def test_invalid_mixture_rejected_before_sampling(training_config, share):
    trainer = Trainer(training_config, device="cpu")
    with pytest.raises(ValueError, match="finite"):
        trainer.batch(share)
    assert all(s.tokens_consumed == 0 for s in trainer.streams.values())


def test_canonical_token_export_preserves_vocabulary(training_config):
    from transformers import PreTrainedTokenizerFast
    manifest = json.loads(Path(training_config["data_manifest"]).read_text())
    path = Path(manifest["tokenizer"]["path"])
    vocabulary = {"<|pad|>": 0, "<|bos|>": 1, "<|eos|>": 2, "<|unk|>": 3}
    vocabulary.update({f"word{i}": i for i in range(4, 64)})
    Tokenizer(WordLevel(vocabulary, unk_token="<|unk|>")).save(str(path))
    manifest["tokenizer"]["sha256"] = _sha(path)
    Path(training_config["data_manifest"]).write_text(json.dumps(manifest))
    trainer = Trainer(training_config, device="cpu")
    trainer.update(0.8)
    trainer.export()
    exported = PreTrainedTokenizerFast.from_pretrained(trainer.run_dir / "export", local_files_only=True)
    assert len(exported) == trainer.model.config.vocab_size == 64
    assert (exported.pad_token_id, exported.bos_token_id, exported.eos_token_id, exported.unk_token_id) == (0, 1, 2, 3)
    assert trainer.model.config.scglm_initialization_seed == training_config["seed"]


def test_resume_requires_original_record_and_runtime(training_config, tmp_path, monkeypatch):
    import scglm.train as training
    trainer = Trainer(training_config, device="cpu")
    checkpoint = trainer.save_checkpoint()
    with pytest.raises(ValueError, match="original run.json"):
        Trainer({**training_config, "run_dir": str(tmp_path / "unrelated")}, device="cpu", resume=checkpoint)
    monkeypatch.setattr(training, "runtime_fingerprint", lambda: {"modified": True})
    with pytest.raises(ValueError, match="fingerprint changed"):
        Trainer(training_config, device="cpu", resume=checkpoint)





