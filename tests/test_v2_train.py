"""CPU tests for the V2 trainer's WSD schedule, movable decay end, and resume."""
import json
from pathlib import Path

import pytest

from scglm_v2.train import V2Trainer, wsd_learning_rate
from test_train import training_config  # noqa: F401  (V1 fixture, tests/test_train.py)


def _wsd(config, tmp_path, name="run"):
    return {**config, "run_dir": str(tmp_path / name), "schedule": "wsd", "warmup_steps": 1,
            "target_tokens": 32 * 40, "max_source_epochs": 1000.0}


def test_wsd_shape():
    lrs = [wsd_learning_rate(s, 1.0, 2, 0.1, None, None) for s in range(5)]
    assert lrs == [0.5, 1.0, 1.0, 1.0, 1.0]
    decay = [wsd_learning_rate(s, 1.0, 2, 0.1, 3, 3) for s in range(3, 7)]
    assert decay[0] == pytest.approx(0.7) and decay[2] == pytest.approx(0.1) and decay[3] == pytest.approx(0.1)


def test_decay_file_moves_the_end_and_exports(training_config, tmp_path, monkeypatch):
    monkeypatch.setattr(V2Trainer, "export", lambda self: self.log("export"))
    trainer = V2Trainer(_wsd(training_config, tmp_path), device="cpu")
    trainer.run(max_steps=5)
    assert trainer.step == 5 and json.loads((trainer.run_dir / "status.json").read_text())["status"] == "paused"
    (trainer.run_dir / "decay.json").write_text(json.dumps({"decay_start_step": 7, "decay_steps": 3}))
    resumed = V2Trainer(_wsd(training_config, tmp_path), device="cpu",
                        resume=trainer.run_dir / "checkpoint_latest.json")
    resumed.run()
    assert resumed.step == 10
    status = json.loads((resumed.run_dir / "status.json").read_text())
    assert status["status"] == "completed"
    assert (resumed.run_dir / "milestones" / "milestone_decay_start_7.pt").exists()
    lrs = [json.loads(l)["learning_rate"] for l in (resumed.run_dir / "events.jsonl").open()
           if json.loads(l)["event"] == "train"]
    assert lrs[-1] == pytest.approx(training_config["learning_rate"] * training_config["min_lr_ratio"])


def test_resume_is_exact_and_rejects_schedule_change(training_config, tmp_path):
    config = _wsd(training_config, tmp_path)
    original = V2Trainer(config, device="cpu")
    for _ in range(2):
        original.update(0.8)
    original.save_checkpoint(reason="test")
    expected = [original.update(0.8)[0] for _ in range(2)]
    restored = V2Trainer(config, device="cpu", resume=original.run_dir / "checkpoint_latest.json")
    assert [restored.update(0.8)[0] for _ in range(2)] == expected
    with pytest.raises(ValueError, match="warmup_steps"):
        V2Trainer({**config, "warmup_steps": 3}, device="cpu", resume=original.run_dir / "checkpoint_latest.json")


def test_resume_guard_reads_compile_from_v2_block(training_config, tmp_path):
    """Regression: run.json config stores compile=False; the true value is in the v2 block."""
    config = _wsd(training_config, tmp_path)
    trainer = V2Trainer(config, device="cpu")
    trainer.save_checkpoint(reason="test")
    record = json.loads((trainer.run_dir / "run.json").read_text())
    assert record["config"]["compile"] is False and record["v2"]["compile"] is False
    record["v2"]["compile"] = True  # simulate a compiled run's record
    (trainer.run_dir / "run.json").write_text(json.dumps(record))
    with pytest.raises(ValueError, match="compile"):
        V2Trainer(config, device="cpu", resume=trainer.run_dir / "checkpoint_latest.json")
    V2Trainer({**config, "compile": True}, device="cpu", resume=trainer.run_dir / "checkpoint_latest.json")


def test_validation_documents_are_labelled_by_bucket(tmp_path):
    """Regression: targeted-bucket panel records carry source 'dclm' (main run crash at step 10000)."""
    from types import SimpleNamespace
    panel = tmp_path / "targeted_monitor.jsonl"
    panel.write_text(json.dumps({"id": "a", "source": "dclm", "tokens": [5, 6, 2]}) + "\n")
    fake = SimpleNamespace(sources=("targeted",), manifest_path=tmp_path / "manifest.json",
                           manifest={"sources": {"targeted": {"splits": {"monitor": {"jsonl_path": str(panel)}}}}})
    rows = list(V2Trainer.documents(fake, "monitor"))
    assert rows[0]["source"] == "targeted" and rows[0]["origin_source"] == "dclm"

