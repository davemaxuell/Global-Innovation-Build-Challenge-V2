"""Continuation must inherit weights/moments and exactly recover all new streams."""
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from test_train import training_config
from scglm.train import Trainer, sha256_file


@pytest.fixture
def continuation(training_config, tmp_path):
    parent = Trainer(training_config, device="cpu")
    parent.run()
    pointer = json.loads((parent.run_dir / "checkpoint_latest.json").read_text())
    manifest = json.loads(Path(training_config["data_manifest"]).read_text())
    manifest["sources"]["edu"] = manifest["sources"].pop("web")
    path = tmp_path / "dclm.bin"
    np.arange(1024, dtype=np.uint16).__mod__(60).__add__(4).tofile(path)
    manifest["sources"]["dclm"] = {"splits": {"train": {"path": str(path), "sha256": sha256_file(path)}}}
    manifest_path = tmp_path / "extension.json"
    manifest_path.write_text(json.dumps(manifest))
    config = {**training_config, "run_dir": str(tmp_path / "extension"),
              "data_manifest": str(manifest_path), "arm": "CPT12",
              "parent_run": str(parent.run_dir), "parent_checkpoint": pointer["path"],
              "parent_checkpoint_sha256": pointer["sha256"],
              "source_weights": {"edu": .6, "dclm": .3, "wiki": .1},
              "rewarm_steps": 2, "initial_learning_rate": 6e-5,
              "milestone_cumulative_tokens": [160]}
    return parent, config


def assert_tree_equal(left, right):
    if torch.is_tensor(left):
        torch.testing.assert_close(left, right, atol=0, rtol=0)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_tree_equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for x, y in zip(left, right):
            assert_tree_equal(x, y)
    else:
        assert left == right


def test_inherits_parent_weights_and_moments_resets_streams(continuation):
    parent, config = continuation
    trainer = Trainer(config, device="cpu")
    assert_tree_equal(parent.model.state_dict(), trainer.model.state_dict())
    assert_tree_equal(parent.optimizer.state_dict(), trainer.optimizer.state_dict())
    assert trainer.step == trainer.main_tokens == 0
    assert all(s.tokens_consumed == 0 for s in trainer.streams.values())
    trainer.write_status("ready")
    status = json.loads((trainer.run_dir / "status.json").read_text())
    assert status["parent_tokens"] == status["cumulative_tokens"] == 128
    assert status["cumulative_target_tokens"] == 256


def test_three_source_continuation_recovers_exactly(continuation):
    _, config = continuation
    trainer = Trainer(config, device="cpu")
    assert trainer.update(None)[2] == pytest.approx(6e-5)
    checkpoint = trainer.save_checkpoint("milestone_160")
    expected_loss = trainer.update(None)
    assert expected_loss[2] == config["learning_rate"]
    expected = trainer.state()
    restored = Trainer(config, device="cpu", resume=checkpoint)
    assert restored.update(None) == expected_loss
    assert_tree_equal(expected["model"], restored.state()["model"])
    assert_tree_equal(expected["optimizer"], restored.state()["optimizer"])
    assert restored.main_source_tokens == trainer.main_source_tokens
    assert sum(restored.main_source_tokens.values()) == restored.main_tokens == 64
    expected_x, expected_y, expected_counts = trainer.batch(None)
    x, y, counts = restored.batch(None)
    np.testing.assert_array_equal(x, expected_x)
    np.testing.assert_array_equal(y, expected_y)
    assert counts == expected_counts
    for _ in range(3):
        restored.save_checkpoint()
    assert (restored.run_dir / "milestones/milestone_160.pt").is_file()


def test_anneal_reaches_declared_floor(continuation):
    _, config = continuation
    trainer = Trainer(config, device="cpu")
    lrs = [trainer.update(None)[2] for _ in range(trainer.total_steps)]
    assert lrs[-1] == pytest.approx(config["learning_rate"] * config["min_lr_ratio"])


@pytest.fixture
def second_continuation(continuation, tmp_path):
    _, config = continuation
    parent = Trainer(config, device="cpu")
    parent.run()
    pointer = json.loads((parent.run_dir / "checkpoint_latest.json").read_text())
    child = {**config, "run_dir": str(tmp_path / "second_continuation"),
             "parent_run": str(parent.run_dir), "parent_checkpoint": pointer["path"],
             "parent_checkpoint_sha256": pointer["sha256"], "inherit_data_state": True,
             "milestone_cumulative_tokens": [384]}
    return parent, child


def test_chained_continuation_inherits_unread_streams_and_all_ancestor_tokens(second_continuation):
    parent, config = second_continuation
    expected_streams = {s: stream.state_dict() for s, stream in parent.streams.items()}
    expected_rng = parent.rng.bit_generator.state
    child = Trainer(config, device="cpu")
    assert child.parent_tokens == 256
    assert child.main_tokens == child.step == 0
    assert child.main_source_tokens == dict.fromkeys(child.sources, 0)
    assert {s: stream.state_dict() for s, stream in child.streams.items()} == expected_streams
    assert child.rng.bit_generator.state == expected_rng
    assert_tree_equal(parent.model.state_dict(), child.model.state_dict())
    assert_tree_equal(parent.optimizer.state_dict(), child.optimizer.state_dict())
    expected = parent.batch(None)
    actual = child.batch(None)
    np.testing.assert_array_equal(actual[0], expected[0])
    np.testing.assert_array_equal(actual[1], expected[1])
    assert actual[2] == expected[2]


def test_chained_continuation_resume_export_and_milestone_accounting(second_continuation):
    _, config = second_continuation
    child = Trainer(config, device="cpu")
    child.update(None)
    checkpoint = child.save_checkpoint()
    expected_loss = child.update(None)
    expected = child.state()
    restored = Trainer(config, device="cpu", resume=checkpoint)
    assert restored.update(None) == expected_loss
    assert_tree_equal(expected["model"], restored.state()["model"])
    assert_tree_equal(expected["optimizer"], restored.state()["optimizer"])
    assert expected["streams"] == restored.state()["streams"]
    restored.run()
    status = json.loads((restored.run_dir / "status.json").read_text())
    export = json.loads((restored.run_dir / "export/training_provenance.json").read_text())
    record = json.loads((restored.run_dir / "run.json").read_text())
    milestone = json.loads((restored.run_dir / "milestones/milestone_384.json").read_text())
    assert status["main_tokens"] == 128
    assert status["cumulative_tokens"] == export["cumulative_tokens"] == milestone["cumulative_tokens"] == 384
    assert record["cumulative_target_tokens"] == 384
    assert len(export["parent"]["scratch_lineage"]) == 2


def test_chained_continuation_rejects_corrupt_ancestor_ledger(second_continuation):
    parent, config = second_continuation
    status_path = parent.run_dir / "status.json"
    status = json.loads(status_path.read_text())
    status["cumulative_tokens"] += 1
    status_path.write_text(json.dumps(status))
    with pytest.raises(ValueError, match="cumulative token ledger"):
        Trainer(config, device="cpu")


def test_original_random_run_can_predate_cumulative_fields(continuation):
    parent, config = continuation
    for filename, key in (("status.json", "cumulative_tokens"), ("run.json", "cumulative_target_tokens")):
        path = parent.run_dir / filename
        record = json.loads(path.read_text())
        record.pop(key)
        path.write_text(json.dumps(record))
    child = Trainer(config, device="cpu")
    assert child.parent_tokens == 128


@pytest.mark.parametrize("change", [
    {"source_weights": {"edu": .5, "dclm": .4, "wiki": .1}},
    {"source_weights": {"wiki": .1, "dclm": .3, "edu": .6}},
    {"batch_sequences": 2},
])
def test_inherited_sampler_must_match_parent(second_continuation, change):
    _, config = second_continuation
    with pytest.raises(ValueError, match="Inherited"):
        Trainer({**config, **change}, device="cpu")


def test_inherited_stream_capacity_includes_prior_consumption(second_continuation):
    _, config = second_continuation
    child = Trainer({**config, "max_source_epochs": .001}, device="cpu")
    before = {s: stream.state_dict() for s, stream in child.streams.items()}
    with pytest.raises(RuntimeError, match="epoch budget exceeded"):
        child.update(None)
    assert before == {s: stream.state_dict() for s, stream in child.streams.items()}


def test_chained_continuation_can_explicitly_start_fresh_streams(second_continuation):
    _, config = second_continuation
    child = Trainer({**config, "inherit_data_state": False}, device="cpu")
    assert child.parent_tokens == 256
    assert all(s.tokens_consumed == 0 for s in child.streams.values())


@pytest.mark.parametrize("field,value,match", [
    ("parent_checkpoint_sha256", "bad", "integrity"),
    ("source_weights", {"edu": .8, "dclm": .3}, "sum to one"),
    ("controller", True, "fixed-mixture"),
])
def test_invalid_continuation_rejected(continuation, field, value, match):
    _, config = continuation
    with pytest.raises(ValueError, match=match):
        Trainer({**config, field: value}, device="cpu")
