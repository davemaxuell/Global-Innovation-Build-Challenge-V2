"""Archived 2026-10-01 with the decay A/B branching code (not part of the selected model's pipeline).
These tests passed against src/scglm_v2/train.py as snapshotted in v2/runs/anneal_quality/snapshot_v2/."""
import json
from pathlib import Path

import pytest

from scglm_v2.train import V2Trainer
from test_train import training_config  # noqa: F401
from test_v2_train import _wsd


def _branch_setup(training_config, tmp_path, monkeypatch):
    """Parent: WSD run with a config decay 6->10 that saves a decay-start milestone."""
    import torch
    monkeypatch.setattr(V2Trainer, "export", lambda self: self.log("export"))
    parent_cfg = {**_wsd(training_config, tmp_path, "parent"), "source_weights": {"web": 0.5, "wiki": 0.5},
                  "decay_start_step": 6, "decay_steps": 4}
    parent = V2Trainer(parent_cfg, device="cpu")
    parent.run()
    assert parent.step == 10
    milestone = parent.run_dir / "milestones" / "milestone_decay_start_6.pt"
    sha = json.loads(milestone.with_suffix(".json").read_text())["sha256"]
    final = {k: v.clone() for k, v in parent.model.state_dict().items()}
    branch_cfg = {**parent_cfg, "run_dir": str(tmp_path / "branch"), "warmup_steps": 0, "decay_start_step": 0,
                  "decay_steps": 4, "target_tokens": 32 * 4, "parent_run": str(parent.run_dir),
                  "parent_checkpoint": str(milestone), "parent_checkpoint_sha256": sha, "inherit_data_state": True}
    return parent, final, branch_cfg


def test_branch_with_same_mix_reproduces_parent_decay_exactly(training_config, tmp_path, monkeypatch):
    import torch
    parent, final, branch_cfg = _branch_setup(training_config, tmp_path, monkeypatch)
    branch = V2Trainer(branch_cfg, device="cpu")
    assert branch.parent_tokens == 6 * 32
    assert branch.lr_at(0) == parent.lr_at(6)
    branch.run()
    assert branch.step == 4
    for name, value in branch.model.state_dict().items():
        torch.testing.assert_close(value, final[name], atol=0, rtol=0)
    status = json.loads((branch.run_dir / "status.json").read_text())
    assert status["status"] == "completed" and status["cumulative_tokens"] == 10 * 32
    record = json.loads((branch.run_dir / "run.json").read_text())
    assert record["initialization"] == "continuation of own documented scratch checkpoint"
    assert record["parent"]["step"] == 6 and record["cumulative_target_tokens"] == 10 * 32
    assert not (branch.run_dir / "milestones" / "milestone_decay_start_0.pt").exists()


def test_branch_with_new_mix_diverges_and_bad_parents_are_rejected(training_config, tmp_path, monkeypatch):
    import torch
    parent, final, branch_cfg = _branch_setup(training_config, tmp_path, monkeypatch)
    other = V2Trainer({**branch_cfg, "source_weights": {"web": 0.9, "wiki": 0.1}}, device="cpu")
    other.run()
    assert any(not torch.equal(v, final[k]) for k, v in other.model.state_dict().items())
    with pytest.raises(ValueError, match="integrity"):
        V2Trainer({**branch_cfg, "run_dir": str(tmp_path / "bad_sha"), "parent_checkpoint_sha256": "0" * 64}, device="cpu")
    outside = tmp_path / "elsewhere.pt"
    outside.write_bytes(Path(branch_cfg["parent_checkpoint"]).read_bytes())
    with pytest.raises(ValueError, match="inside a separate parent run"):
        V2Trainer({**branch_cfg, "run_dir": str(tmp_path / "bad_path"), "parent_checkpoint": str(outside)}, device="cpu")


def test_branch_materializer_freezes_hash_and_trains(training_config, tmp_path, monkeypatch):
    """Template (inherits parent config) -> frozen config with the milestone hash -> branch run."""
    import torch
    from scglm_v2.branch import materialize
    parent, final, branch_cfg = _branch_setup(training_config, tmp_path, monkeypatch)
    parent_file = tmp_path / "parent_config.json"
    parent_file.write_text(json.dumps({k: v for k, v in branch_cfg.items() if not k.startswith("parent_")}))
    template = tmp_path / "template.json"
    template.write_text(json.dumps({"inherits": str(parent_file), "parent_run": branch_cfg["parent_run"],
                                    "parent_checkpoint": branch_cfg["parent_checkpoint"],
                                    "parent_checkpoint_sha256": "FILLED_AT_LAUNCH"}))
    out = tmp_path / "frozen.json"
    config = materialize(template, out)
    assert config["parent_checkpoint_sha256"] == branch_cfg["parent_checkpoint_sha256"]
    with pytest.raises(FileExistsError):
        materialize(template, out)
    branch = V2Trainer(json.loads(out.read_text()), device="cpu")
    branch.run()
    for name, value in branch.model.state_dict().items():
        torch.testing.assert_close(value, final[name], atol=0, rtol=0)
