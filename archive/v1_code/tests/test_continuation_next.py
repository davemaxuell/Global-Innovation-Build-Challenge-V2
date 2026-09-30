"""Scientific decision and registration checks for the bounded next continuation."""
import copy
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("run_pretraining_next", Path(__file__).resolve().parents[1] / "scripts/run_pretraining_next.py")
next_run = importlib.util.module_from_spec(spec)
spec.loader.exec_module(next_run)


@pytest.fixture
def comparison():
    weights = {"edu": .6, "dclm": .3, "wiki": .1}
    rows = [{"source": s, "id": str(i), "nll_sum": 3.0 * (i+1), "token_count": i+1}
            for s in weights for i in range(8)]
    policy = {"bootstrap_seed": 1, "bootstrap_replicates": 100,
              "min_mixture_nll_improvement": .005, "max_source_nll_regression": .02}
    return {"documents": rows}, weights, policy


def test_paired_gate_accepts_consistent_improvement(comparison):
    base, weights, policy = comparison
    candidate = copy.deepcopy(base)
    for row in candidate["documents"]:
        row["nll_sum"] -= .01 * row["token_count"]
    result = next_run.compare(base, candidate, weights, policy)
    assert result["passed"]
    assert result["candidate_minus_parent_nll"] == pytest.approx(-.01)
    assert result["upper_95"] < 0


def test_paired_gate_rejects_no_gain_and_source_regression(comparison):
    base, weights, policy = comparison
    assert not next_run.compare(base, base, weights, policy)["passed"]
    candidate = copy.deepcopy(base)
    for row in candidate["documents"]:
        row["nll_sum"] += (.03 if row["source"] == "wiki" else -.10) * row["token_count"]
    assert not next_run.compare(base, candidate, weights, policy)["passed"]


@pytest.mark.parametrize("field,value", [("id", "foreign"), ("token_count", 123)])
def test_paired_gate_rejects_mismatched_documents(comparison, field, value):
    base, weights, policy = comparison
    candidate = copy.deepcopy(base)
    candidate["documents"][0][field] = value
    with pytest.raises(ValueError, match="differ"):
        next_run.compare(base, candidate, weights, policy)
