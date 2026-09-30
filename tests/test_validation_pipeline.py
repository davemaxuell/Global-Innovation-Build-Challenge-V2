"""Exercise audit durability and result acceptance without opening model scores."""
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
from types import SimpleNamespace

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import run_final_evaluation as runner
from validation_history import ValidationRun, harness_metrics, sha256, wikitext_metrics


def read_runs(history):
    return [json.loads(p.read_text()) for p in sorted((history / "runs").glob("*/run.json"))]


@pytest.fixture
def pipeline(tmp_path, monkeypatch):
    """Real runner and child processes, synthetic model and benchmark outputs."""
    config = json.loads((runner.ROOT / "configs/evaluation.json").read_text())
    for definition in config["harness"]["task_definitions"].values():
        definition["expected_examples"] = 2
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    model = tmp_path / "model with spaces"
    model.mkdir()
    for name, payload in {
        "config.json": {"vocab_size": 8, "max_position_embeddings": 1024},
        "tokenizer.json": {}, "training_provenance.json": {"fixture": True},
    }.items():
        (model / name).write_text(json.dumps(payload))
    (model / "model.safetensors").write_bytes(b"synthetic test fixture; never loaded")
    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps({
        "panel": "selection", "official_scores_used": False, "selected_model": str(model),
        "selected_model_sha256": sha256(model / "model.safetensors"),
    }))
    history = tmp_path / "history"

    def prepare(config_path, output):
        manifest = {"protocol_sha256": sha256(config_path), "fixture": True}
        return config, runner.ROOT / "vendor/lm-evaluation-harness", output / "overrides", manifest

    monkeypatch.setattr(runner, "prepare", prepare)
    import transformers

    class FakeTokenizer:
        eos_token_id = 2

        def __len__(self):
            return 8

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: FakeTokenizer())
    benchmark = {"results": {}, "n-samples": {}, "n-shot": {}}
    for name, definition in config["harness"]["task_definitions"].items():
        benchmark["results"][name] = {f"{metric},none": 0.5 for metric in definition["metrics"]}
        benchmark["n-samples"][name] = {"original": 2, "effective": 2}
        benchmark["n-shot"][name] = 0
    wiki = {
        "metadata": {**{k: config["wikitext103"][k] for k in
                        ("dataset", "configuration", "revision", "split", "context_length", "stride")}},
        "metrics": {"sources": {"wikitext103": {
            "nll_sum": 20.0, "token_count": 10, "documents": 2, "nll": 2.0, "perplexity": math.exp(2.0),
        }}},
    }
    fixture = SimpleNamespace(config=config, config_path=config_path, model=model, history=history,
                              benchmark=benchmark, wiki=wiki, failure=None, mutate_model=False)

    def commands(config, repo, overlays, model, output, device):
        specifications = []
        for name, path, payload in (
            ("harness", output / "harness/model/results_fixture.json", benchmark),
            ("wikitext103", output / "wikitext103.json", wiki),
        ):
            code = "from pathlib import Path; import sys; print('stage fixture', flush=True); "
            if fixture.failure == name:
                code += "print('deliberate failure', file=sys.stderr); sys.exit(7)"
            else:
                code += f"p=Path({str(path)!r}); p.parent.mkdir(parents=True, exist_ok=True); p.write_text({json.dumps(payload)!r})"
                if fixture.mutate_model and name == "wikitext103":
                    code += f"; Path({str(model / 'model.safetensors')!r}).write_bytes(b'changed')"
            specifications.append({"cwd": str(tmp_path), "command": [sys.executable, "-c", code]})
        return specifications

    monkeypatch.setattr(runner, "commands", commands)
    fixture.arguments = ["--config", str(config_path), "--history-dir", str(history),
                         "--model", str(model), "--selection-record", str(selection), "--final-evaluation"]
    return fixture


def test_repeated_preflight_creates_separate_durable_records(pipeline, monkeypatch):
    def no_inference(*args, **kwargs):
        pytest.fail("Preflight must never run inference")

    monkeypatch.setattr(ValidationRun, "run_stage", no_inference)
    for _ in range(2):
        runner.main(pipeline.arguments + ["--prepare-only", "--label", "repeat"])
    runs = read_runs(pipeline.history)
    assert len(runs) == 2
    assert len({r["run_id"] for r in runs}) == 2
    assert all(r["status"] == "prepared" and r["metrics"] == {} for r in runs)
    assert all(Path(r["output"], "commands.json").exists() for r in runs)
    events = [json.loads(line) for line in (pipeline.history / "history.jsonl").read_text().splitlines()]
    assert [e["event"] for e in events] == ["started", "finished", "started", "finished"]


def test_success_saves_all_five_results_and_logs(pipeline):
    runner.main(pipeline.arguments)
    run, = read_runs(pipeline.history)
    assert run["status"] == "completed"
    assert set(run["metrics"]["benchmarks"]) == {"hellaswag", "arc_easy", "piqa", "winogrande"}
    assert run["metrics"]["wikitext103"]["perplexity"] == math.exp(2)
    assert run["execution"]["artifact_sha256"]["model.safetensors"] == sha256(pipeline.model / "model.safetensors")
    assert all("stage fixture" in Path(stage["log"]).read_text() for stage in run["stages"])
    assert all(stage["returncode"] == 0 for stage in run["stages"])
    assert Path(run["output"], "completed.json").exists()
    assert json.loads(Path(run["output"], "summary.json").read_text())["metrics"] == run["metrics"]


def test_later_failure_keeps_first_stage_scores(pipeline):
    pipeline.failure = "wikitext103"
    with pytest.raises(subprocess.CalledProcessError):
        runner.main(pipeline.arguments)
    run, = read_runs(pipeline.history)
    assert run["status"] == "failed"
    assert len(run["metrics"]["benchmarks"]) == 4
    assert "wikitext103" not in run["metrics"]
    assert run["stages"][-1]["returncode"] == 7
    assert "deliberate failure" in Path(run["stages"][-1]["log"]).read_text()
    assert not Path(run["output"], "completed.json").exists()


def test_failed_preflight_and_existing_output_are_recorded(pipeline, monkeypatch, tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    sentinel = output / "results.json"
    sentinel.write_text("keep me")
    with pytest.raises(FileExistsError):
        runner.main(pipeline.arguments + ["--output", str(output)])
    assert sentinel.read_text() == "keep me"

    def broken(*args):
        raise ValueError("frozen runtime mismatch")

    monkeypatch.setattr(runner, "prepare", broken)
    with pytest.raises(ValueError, match="runtime mismatch"):
        runner.main(pipeline.arguments + ["--prepare-only"])
    runs = read_runs(pipeline.history)
    assert [r["status"] for r in runs] == ["failed", "failed"]
    assert Path(runs[-1]["output"], "protocol_requested.json").exists()


def test_selection_gate_failure_is_recorded(tmp_path):
    with pytest.raises(ValueError, match="selection-record"):
        runner.main(["--history-dir", str(tmp_path)])
    run, = read_runs(tmp_path)
    assert run["status"] == "failed"
    assert not run["stages"]


def test_changed_weights_fail_before_inference(pipeline, monkeypatch):
    (pipeline.model / "model.safetensors").write_bytes(b"wrong endpoint")
    monkeypatch.setattr(ValidationRun, "run_stage", lambda *args: pytest.fail("must not run"))
    with pytest.raises(ValueError, match="changed after"):
        runner.main(pipeline.arguments)
    run, = read_runs(pipeline.history)
    assert run["status"] == "failed"


def test_export_replaced_during_inference_cannot_complete(pipeline):
    pipeline.mutate_model = True
    with pytest.raises(ValueError, match="changed during"):
        runner.main(pipeline.arguments)
    run, = read_runs(pipeline.history)
    assert run["status"] == "failed"
    assert not Path(run["output"], "completed.json").exists()


@pytest.mark.parametrize("problem", ["missing_task", "partial_count", "wrong_shots", "nan", "missing_metric"])
def test_incomplete_harness_output_is_not_success(pipeline, problem):
    result = pipeline.benchmark
    if problem == "missing_task":
        del result["results"]["piqa"]
    elif problem == "partial_count":
        result["n-samples"]["piqa"]["effective"] = 1
    elif problem == "wrong_shots":
        result["n-shot"]["piqa"] = 5
    elif problem == "nan":
        result["results"]["piqa"]["acc,none"] = float("nan")
    else:
        del result["results"]["piqa"]["acc,none"]
    with pytest.raises((ValueError, KeyError)):
        runner.main(pipeline.arguments)
    run, = read_runs(pipeline.history)
    assert run["status"] == "failed"
    assert not Path(run["output"], "completed.json").exists()


@pytest.mark.parametrize("problem", ["wrong_dataset", "inconsistent_perplexity", "zero_tokens"])
def test_invalid_wikitext_output_cannot_complete(pipeline, problem):
    if problem == "wrong_dataset":
        pipeline.wiki["metadata"]["configuration"] = "wikitext-2-raw-v1"
    elif problem == "zero_tokens":
        pipeline.wiki["metrics"]["sources"]["wikitext103"]["token_count"] = 0
    else:
        pipeline.wiki["metrics"]["sources"]["wikitext103"]["perplexity"] = 3
    with pytest.raises(ValueError):
        runner.main(pipeline.arguments)
    assert read_runs(pipeline.history)[0]["status"] == "failed"


def test_sigterm_records_interruption_and_reaps_child(tmp_path):
    previous = signal.getsignal(signal.SIGTERM)
    with pytest.raises(KeyboardInterrupt):
        with ValidationRun(tmp_path, output=None, metadata={"mode": "official"}) as run:
            run.run_stage("interrupt", {"cwd": str(tmp_path), "command": [sys.executable, "-c",
                "import os,signal,time; os.kill(os.getppid(), signal.SIGTERM); time.sleep(30)"]}, dict(os.environ))
    record, = read_runs(tmp_path)
    assert record["status"] == record["stages"][0]["status"] == "interrupted"
    assert record["stages"][0]["returncode"] is not None
    assert signal.getsignal(signal.SIGTERM) == previous


def test_concurrent_runs_do_not_lose_journal_records(tmp_path):
    code = ("from pathlib import Path\nfrom validation_history import ValidationRun\n"
            f"with ValidationRun(Path({str(tmp_path)!r}), output=None, metadata={{'mode':'prepare'}}): pass\n")
    env = {**os.environ, "PYTHONPATH": str(SCRIPTS)}
    processes = [subprocess.Popen([sys.executable, "-c", code], env=env, stdout=subprocess.DEVNULL) for _ in range(6)]
    assert all(p.wait(timeout=20) == 0 for p in processes)
    events = [json.loads(line) for line in (tmp_path / "history.jsonl").read_text().splitlines()]
    assert len(events) == 12
    assert len({e["run_id"] for e in events}) == 6
    assert len(read_runs(tmp_path)) == 6


def test_real_commands_cover_exact_protocol_and_paths_with_spaces(tmp_path):
    config = json.loads((runner.ROOT / "configs/evaluation.json").read_text())
    planned = runner.commands(config, tmp_path / "harness repo", tmp_path / "overrides",
                              tmp_path / "model export", tmp_path / "output", "cpu")
    harness, wiki = [spec["command"] for spec in planned]
    assert harness[harness.index("--tasks") + 1:harness.index("--include_path")] == config["harness"]["tasks"]
    assert "--limit" not in harness and "--log_samples" in harness
    assert f"pretrained={tmp_path / 'model export'}," in harness[harness.index("--model_args") + 1]
    assert "wikitext103" in wiki and "wikitext" not in harness
    assert wiki[wiki.index("--dataset-revision") + 1] == config["wikitext103"]["revision"]
