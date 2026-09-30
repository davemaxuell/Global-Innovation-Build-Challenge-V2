import json
import numpy as np
import pytest
from scripts.preflight_extension import check_index
from scripts.run_extension import pilot_failures
from scglm.evaluate import fixed_mixture_metrics


def test_index_audit_rejects_reserved_overlap_and_broken_eos(tmp_path):
    index = tmp_path / "index.jsonl"
    index.write_text(json.dumps({"content_sha256": "abc", "offset": 0, "length": 3})+"\n")
    assert check_index(index, np.array([4, 5, 2]), set()) == 1
    with pytest.raises(ValueError, match="reserved"):
        check_index(index, np.array([4, 5, 2]), {"abc"})
    with pytest.raises(ValueError, match="EOS"):
        check_index(index, np.array([4, 5, 6]), set())


def test_pilot_gate_uses_real_metric_schema_and_detects_source_regression():
    def result(loss):
        return {group: fixed_mixture_metrics([
            {"source": s, "nll_sum": 100*loss[s], "token_count": 100} for s in weights
        ], weights=weights) for group, weights in {
            "new": {"edu": .6, "dclm": .3, "wiki": .1},
            "legacy": {"web": .8, "wiki": .2}}.items()}
    before = result(dict.fromkeys(("edu", "dclm", "wiki", "web"), 3.0))
    after = result(dict.fromkeys(("edu", "dclm", "wiki", "web"), 2.9))
    assert pilot_failures(before, after) == []
    after = result({"edu": 3., "dclm": 3., "wiki": 3.6, "web": 3.})
    assert len(pilot_failures(before, after)) == 2
