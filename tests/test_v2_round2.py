"""Round 2: half split, selection rule against the V2.1 reference, uniform averaging."""
import torch

from scglm.model import build_model
from scglm_v2.round2 import average_exports, decide, half_of

TINY = {"vocab_size": 64, "hidden_size": 32, "intermediate_size": 80, "num_hidden_layers": 2,
        "num_attention_heads": 4, "max_position_embeddings": 32}


def test_half_split_is_fixed_and_balanced():
    items = [{"task": "piqa", "label": i % 2, "encoded": [[[i], [i + 1]], [[i], [i + 2]]]} for i in range(2000)]
    first = [half_of(i) for i in items]
    assert first == [half_of(i) for i in items]
    assert 900 < sum(first) < 1100


def test_rule_needs_gain_and_wikipedia_no_worse():
    r = lambda name, acc, wiki: {"export": name, "proxy_mean_acc": acc, "bits_per_byte_selection_panels": {"wiki": wiki}}
    ref = r("v2.1", 0.50, 1.01)
    assert decide(ref, [r("a", 0.505, 1.01)], 0.005)[0]["export"] == "a"          # inclusive edges
    assert decide(ref, [r("a", 0.5049, 1.0)], 0.005)[0] is ref                     # gain too small
    assert decide(ref, [r("a", 0.60, 1.0101)], 0.005)[0] is ref                    # Wikipedia worse
    assert decide(ref, [r("a", 0.60, 1.02), r("b", 0.51, 1.0)], 0.005)[0]["export"] == "b"


def test_average_exports(tmp_path):
    names, models = None, []
    for seed in (1, 2):
        m = build_model({**TINY, "seed": seed})
        m.save_pretrained(tmp_path / str(seed), safe_serialization=True)
        models.append(m)
    names = [n for n, _ in models[0].named_parameters()]
    avg = average_exports([tmp_path / "1", tmp_path / "2"], names)
    p1, p2 = dict(models[0].named_parameters()), dict(models[1].named_parameters())
    assert all(torch.allclose(avg[n].float(), (p1[n] + p2[n]) / 2, atol=1e-6) for n in names)
