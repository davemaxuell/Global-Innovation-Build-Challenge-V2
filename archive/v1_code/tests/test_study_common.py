"""Shared study layer: budgets, registration, resumable evaluation, guards and paired inference."""
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from test_posttraining import tokenizer, tiny_model
from scglm.model import save_model
from scglm_study import common
from scglm_study.budget import Budget, BudgetExceeded
from scglm_study.common import read, sha256, write_json
from scglm_study.evaluate import evaluate, guards, paired_difference, summarize, task_bootstrap, load_result

LIMITS = {"gpu_seconds": {"shared": 100, "arm": 50}, "overall_gpu_seconds": 120, "max_positions": {"arm": 1000}}


def test_budget_category_overall_position_and_crash_limits(tmp_path):
    clock = [0.]
    budget = Budget(tmp_path, LIMITS, clock=lambda: clock[0])
    with budget.session("arm", timer=False):
        budget.charge(900, "rollout")
        with pytest.raises(BudgetExceeded):
            budget.charge(101, "rollout")
        clock[0] = 40
    assert budget.remaining("arm") == 10
    # An unclean exit is charged up to the reservation and marked uncertain.
    budget.value["active"] = {"category": "shared", "started": clock[0], "reserved_seconds": budget.remaining("shared")}
    budget._save()
    clock[0] += 70
    restored = Budget(tmp_path, LIMITS, clock=lambda: clock[0])
    assert restored.value["categories"]["shared"]["seconds"] == 70
    assert restored.value["categories"]["shared"]["uncertain_seconds"] == 70
    # Overall cap binds before the shared category cap.
    assert restored.remaining("shared") == 10
    with pytest.raises(ValueError):
        restored.remaining("unregistered")
    with pytest.raises(ValueError):
        restored.charge(1, "outside_session")


def test_budget_refuses_to_reset_after_preflight(tmp_path):
    (tmp_path / "preflight.json").write_text("{}")
    with pytest.raises(ValueError):
        Budget(tmp_path, LIMITS)


def test_register_writes_journal_once_and_freezes_record(tmp_path, monkeypatch):
    monkeypatch.setattr(common, "ROOT", tmp_path)
    (tmp_path / "TRAINING_PHASES.md").write_text("# Phases\n")
    (tmp_path / "artifacts").mkdir()
    cfg = {"experiment_id": "x_v1", "title": "X", "script": "scripts/x.py", "_config_path": "configs/x.json",
           "parent": {"path": "p", "weight_sha256": "0" * 64}, "budget": {}, "evaluation": {},
           "isolation": {"data_root": "data/x", "output_root": "artifacts/x"}}
    first = common.register(cfg, "train", "Objective.", data={"a": 1})
    common.register(cfg, "train", "Objective.", data={"a": 1})
    journal = (tmp_path / "TRAINING_PHASES.md").read_text()
    assert journal.count("x_v1/train") == 1
    assert read(first / "status.json")["status"] == "preparation"
    assert "resume" in read(first / "registration.json")["resume_command"]
    with pytest.raises(ValueError):
        common.register(cfg, "train", "Changed objective.", data={"a": 1})


def _export(tmp_path, tok):
    path = tmp_path / "export"
    save_model(tiny_model(tok), path)
    tok.save_pretrained(str(path))
    return path


def _panels(tok):
    choices = [{"id": f"c{i}", "group_id": f"g{i // 2}", "domain": d, "question": "Hello", "choices": ["yes", "no"],
                "answer_index": i % 2} for i, d in enumerate(["science", "science", "commonsense", "commonsense"])]
    raw = [{"id": f"r{i}", "group_id": f"rg{i}", "source": "legacy_web", "tokens": tok.encode("Hello World 1234")[:6] + [i + 4]}
           for i in range(3)]
    generate = [{"id": f"t{i}", "group_id": f"tg{i}", "family": "arithmetic", "prompt": "Hello", "answer": "1234",
                 "verifier": "exact", "prompt_style": "raw", "verifier_version": "scglm-tasks-v1"} for i in range(2)]
    return {"primary": {"kind": "choice", "rows": choices}, "raw": {"kind": "raw", "rows": raw},
            "instructions": {"kind": "generate", "rows": generate}}


def test_evaluation_resumes_by_item_and_binds_model_hash(tmp_path, tokenizer):
    path = _export(tmp_path, tokenizer)
    charges = []
    charge = lambda n, kind: charges.append((n, kind))
    templates = ["{question}\nAnswer:", "Question: {question}\nThe answer is"]
    first = evaluate(path, tmp_path / "a", _panels(tokenizer), charge, presentations=templates, device="cpu")
    assert set(first["metrics"]) == {"primary", "raw", "instructions"}
    assert "primary" in first["metrics"]["primary"]
    assert first["metrics"]["raw"]["legacy_web"]["documents"] == 3
    # Interrupt after one completed item plus a torn partial line, then resume.
    partial = tmp_path / "b"
    evaluate(path, partial, _panels(tokenizer), charge, presentations=templates, device="cpu")
    lines = (partial / "primary.jsonl").read_text().splitlines()
    (partial / "primary.jsonl").write_text(lines[0] + "\n" + lines[1][:10])
    (partial / "result.json").unlink()
    resumed = evaluate(path, partial, _panels(tokenizer), charge, presentations=templates, device="cpu")
    assert resumed["metrics"] == first["metrics"]
    assert load_result(partial)["files"] == resumed["files"]
    assert charges and all(n > 0 for n, _ in charges)
    # The same directory cannot silently hold another model's evidence.
    other = tmp_path / "other"
    torch.manual_seed(1)
    model = tiny_model(tokenizer)
    with torch.no_grad():
        model.model.norm.weight.mul_(2)
    save_model(model, other)
    tokenizer.save_pretrained(str(other))
    with pytest.raises(ValueError):
        evaluate(other, partial, _panels(tokenizer), charge, presentations=templates, device="cpu")


def test_guards_report_breaches_and_reject_unpaired_coverage():
    reference = {"raw": {"a": {"nll": 2.0, "token_count": 10}},
                 "primary": {"groups": {"science": {"accuracy": .5}, "commonsense": {"accuracy": .5}}}}
    candidate = {"raw": {"a": {"nll": 2.011, "token_count": 10}},
                 "primary": {"groups": {"science": {"accuracy": .47}, "commonsense": {"accuracy": .49}}}}
    rules = {"raw_panels": ["raw"], "raw_nll_delta": .01, "choice_panels": ["primary"], "accuracy_delta_min": -.02}
    assert guards(candidate, reference, rules, "parent") == ["raw:raw:a:vs_parent", "accuracy:primary:science:vs_parent"]
    candidate["raw"]["a"]["token_count"] = 9
    with pytest.raises(ValueError):
        guards(candidate, reference, rules, "parent")


def _result(tmp_path, name, primary, tasks):
    directory = tmp_path / name
    directory.mkdir()
    for panel, rows in (("primary", primary), ("instructions", tasks)):
        (directory / f"{panel}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return {"directory": str(directory), "files": {f"{p}.jsonl": sha256(directory / f"{p}.jsonl") for p in ("primary", "instructions")}}


def test_paired_inference_detects_a_consistent_gain(tmp_path):
    rng = np.random.default_rng(0)
    base_primary, better_primary, base_tasks, better_tasks = [], [], [], []
    for i in range(400):
        domain = "science" if i % 2 else "commonsense"
        truth = float(rng.random() < .4)
        base_primary.append({"id": str(i), "group": str(i // 4), "domain": domain, "correct": truth})
        better_primary.append({"id": str(i), "group": str(i // 4), "domain": domain, "correct": max(truth, float(i % 5 == 0))})
        family = "a" if i % 2 else "b"
        base_tasks.append({"id": str(i), "group": str(i), "family": family, "correct": truth})
        better_tasks.append({"id": str(i), "group": str(i), "family": family, "correct": max(truth, float(i % 4 == 0))})
    results = {"new": _result(tmp_path, "new", better_primary, better_tasks), "old": _result(tmp_path, "old", base_primary, base_tasks)}
    primary = paired_difference(results, ("new", "old"), "primary", "primary", replicates=2000, seed=1)
    assert primary["point"] > 0 and primary["lower_95_one_sided"] > 0
    task = task_bootstrap(results, ("new", "old"), "instructions", replicates=2000, seed=1, families=["a", "b"])
    assert task["point"] > 0 and task["lower_95_one_sided"] > 0 and task["families"] == ["a", "b"]
    reverse = task_bootstrap(results, ("old", "new"), "instructions", replicates=2000, seed=1)
    assert reverse["upper_95_one_sided"] < 0
    with pytest.raises(ValueError):
        task_bootstrap(results, ("new", "old"), "instructions", replicates=10, seed=1, families=["missing"])


def test_summaries_match_hand_counts():
    records = {"g": [{"family": "x", "correct": 1., "answer_correct": True, "ended_eos": True},
                     {"family": "x", "correct": 0., "answer_correct": True, "ended_eos": False}],
               "r": [{"source": "s", "nll_sum": 3., "token_count": 2}, {"source": "s", "nll_sum": 1., "token_count": 2}]}
    result = summarize(records, {"g": "generate", "r": "raw"})
    assert result["g"]["groups"]["x"] == {"accuracy": .5, "items": 2, "answer_correct": 1., "ended_eos": .5}
    assert result["r"]["s"] == {"nll": 1., "token_count": 4, "documents": 2}
