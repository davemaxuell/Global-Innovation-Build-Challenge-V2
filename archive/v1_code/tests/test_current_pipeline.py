"""Retained-recipe equivalence, endpoint scheduling and output protection."""
import ast
import copy
import json
from pathlib import Path
import sys

import pytest

from test_posttraining import tokenizer
from test_sota_posttraining import sota_fixture
from scglm_pipeline.common import ROOT, read
from scglm_pipeline.generation import weights_digest
from scglm_pipeline.sota_train import Trainer
from scripts import run_pipeline


def test_selected_prefix_preserves_full_schedule_and_exact_updates(sota_fixture, tmp_path):
    config, recipe, manifest, base = sota_fixture
    reference = Trainer(config, recipe, parent=base, manifest=manifest, output=tmp_path / "reference")
    assert len(reference.plan) > 2
    reference.run(max_updates=2)
    selected = Trainer(config, {**recipe, "stop_after_updates": 2}, parent=base,
                       manifest=manifest, output=tmp_path / "selected")
    selected.run()
    assert selected.planned == reference.planned
    assert selected.step == selected.updates == 2
    assert selected.counters == reference.counters
    assert weights_digest(selected.model) == weights_digest(reference.model)
    assert read(tmp_path / "selected/status.json")["status"] == "completed"
    assert selected.endpoints[-1]["step"] == 2
    resumed = Trainer(config, {**recipe, "stop_after_updates": 2}, parent=base,
                      manifest=manifest, output=tmp_path / "selected")
    resumed.run()
    assert resumed.step == 2
    assert weights_digest(resumed.model) == weights_digest(selected.model)


def test_selected_optimization_and_batch_planner_match_frozen_source():
    historical = ROOT / "artifacts/posttraining_sota_v1/snapshot/src/scglm_pipeline/sota_train.py"
    current = ROOT / "src/scglm_pipeline/sota_train.py"
    def functions(path):
        return {node.name: ast.dump(node, include_attributes=False)
                for node in ast.parse(path.read_text()).body if isinstance(node, ast.FunctionDef)}
    before, after = functions(historical), functions(current)
    for name in ("balanced_plan", "sft_update"):
        assert before[name] == after[name]


@pytest.mark.parametrize("stage", ["dpo", "rl", "positive_only"])
def test_retired_training_stages_are_unavailable(stage, tmp_path):
    with pytest.raises(ValueError, match="retained full-parameter SFT"):
        Trainer({}, {}, parent=tmp_path / "base", manifest=tmp_path / "data",
                output=tmp_path / "work", stage=stage)


@pytest.mark.parametrize("relative", ["checkpoints/best_sft", "runs/continuation_13b", "data/processed/main",
                                     "artifacts/posttraining_sota_v1", "artifacts/current_model_reproductions"])
def test_reproduction_cannot_write_to_preserved_paths(relative):
    with pytest.raises(ValueError, match="child of"):
        run_pipeline.checked_output(relative)


def test_output_symlink_cannot_escape_reproduction_root(tmp_path, monkeypatch):
    monkeypatch.setattr(run_pipeline, "ROOT", tmp_path)
    allowed = tmp_path / "artifacts/current_model_reproductions"
    allowed.mkdir(parents=True)
    (allowed / "escape").symlink_to(tmp_path)
    with pytest.raises(ValueError, match="child of"):
        run_pipeline.checked_output(allowed / "escape/checkpoints")


def test_default_command_only_verifies(monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(run_pipeline, "verify", lambda *a, **k: calls.append("verify") or {"verified": True})
    monkeypatch.setattr(run_pipeline, "gpu_environment", lambda *a: pytest.fail("Default command touched GPU"))
    monkeypatch.setattr(sys, "argv", ["run_pipeline.py"])
    run_pipeline.main()
    assert calls == ["verify"]
    assert json.loads(capsys.readouterr().out)["verified"] is True


def test_gpu_visibility_cannot_select_unallocated_device(monkeypatch):
    monkeypatch.setattr(run_pipeline.subprocess, "check_output", lambda *a, **k: "0, GPU-assigned\n1, GPU-other\n")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    with pytest.raises(ValueError, match="assigned H100"):
        run_pipeline.gpu_environment({"allowed_gpu_uuids": ["GPU-assigned"]}, "cuda:0")
