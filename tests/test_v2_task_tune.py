"""Task-tuning phase: harness formatting, scoring/training agreement, schedule and selection rule."""
import math

import torch

from scglm.model import build_model
from scglm_v2.task_format import encode_pair, harness_item, score_items, sequence_logprobs
from scglm_v2.task_select import decide
from scglm_v2.task_tune import _lr, task_losses

TINY = {"vocab_size": 64, "hidden_size": 32, "intermediate_size": 80, "num_hidden_layers": 2,
        "num_attention_heads": 4, "max_position_embeddings": 32, "seed": 123}


class CharTokenizer:
    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return [ord(c) % 60 + 4 for c in text]


def test_harness_templates():
    hs = harness_item("hellaswag", {"activity_label": "Roof shingle removal", "ctx_a": "A man is on a roof.",
                                    "ctx_b": "he", "endings": ["a [title] b", "c", "d", "e"], "label": "2"})
    assert hs["pairs"][0] == ("Roof shingle removal: A man is on a roof. He", " a. b")
    assert hs["label"] == 2
    arc = harness_item("arc_easy", {"question": "Q?", "answerKey": "B",
                                    "choices": {"label": ["A", "B"], "text": ["x", "y"]}})
    assert arc["pairs"] == [("Question: Q?\nAnswer:", " x"), ("Question: Q?\nAnswer:", " y")] and arc["label"] == 1
    piqa = harness_item("piqa", {"goal": "G", "sol1": "s1", "sol2": "s2", "label": 0})
    assert piqa["pairs"][1] == ("Question: G\nAnswer:", " s2")
    wg = harness_item("winogrande", {"sentence": "A beat B because _ was fast.", "option1": "A",
                                     "option2": "B", "answer": "2"})
    assert wg["pairs"] == [("A beat B because A", " was fast."), ("A beat B because B", " was fast.")]
    assert wg["label"] == 1


def test_encode_pair_moves_trailing_space():
    tok = CharTokenizer()
    ctx, cont = encode_pair(tok, "ab ", "c")
    assert ctx == tok.encode("ab") and cont == tok.encode(" c")


def _items(tok):
    raw = [("arc_easy", {"question": "q", "answerKey": "A", "choices": {"label": ["A", "B", "C"], "text": ["x", "yy", "z"]}}),
           ("winogrande", {"sentence": "a _ b.", "option1": "c", "option2": "d", "answer": "1"}),
           ("piqa", {"goal": "g", "sol1": "s", "sol2": "tt", "label": 1})]
    out = []
    for task, doc in raw:
        item = harness_item(task, doc)
        out.append({"task": task, "label": item["label"], "choices": item["choices"],
                    "encoded": [list(map(list, encode_pair(tok, c, k))) for c, k in item["pairs"]]})
    return out


def test_training_scores_match_scorer():
    torch.manual_seed(0)
    model = build_model({**TINY}).eval()
    items = _items(CharTokenizer())
    flat = [tuple(p) for item in items for p in item["encoded"]]
    batched = sequence_logprobs(model, flat, "cpu", max_rows=2)
    single = [sequence_logprobs(model, [p], "cpu")[0] for p in flat]
    assert all(abs(a - b) < 1e-4 for a, b in zip(batched, single))   # padding does not change scores
    choice_ce, gold_lm, _ = task_losses(model, items, "cpu")
    expected, k = [], 0
    for item in items:
        s = torch.tensor(single[k:k + len(item["encoded"])]); k += len(item["encoded"])
        expected.append(-(s[item["label"]] - torch.logsumexp(s, 0)))
    assert abs(float(choice_ce) - float(torch.stack(expected).mean())) < 1e-4
    assert math.isfinite(float(gold_lm)) and float(gold_lm) > 0
    result = score_items(model, items, "cpu")
    assert set(result) >= {"arc_easy", "winogrande", "piqa"} and result["_proxy_mean_acc"] is None


def test_lr_schedule():
    assert _lr(0, 100, 1.0, 10) == 0.1
    assert _lr(9, 100, 1.0, 10) == 1.0
    assert abs(_lr(99, 100, 1.0, 10)) < 1e-12


def test_selection_rule():
    rule = {"min_proxy_gain": 0.01, "max_wiki_bpb_regression": 0.01}
    r = lambda name, proxy, wiki: {"export": name, "proxy_mean_acc": proxy, "bits_per_byte_selection_panels": {"wiki": wiki}}
    parent = r("v2", 0.50, 1.00)
    assert decide(parent, [r("a", 0.51, 1.01)], rule)[0]["export"] == "a"              # inclusive edges
    assert decide(parent, [r("a", 0.5099, 0.9)], rule)[0] is parent                    # gain too small
    assert decide(parent, [r("a", 0.60, 1.0101)], rule)[0] is parent                    # Wikipedia guard
    assert decide(parent, [r("a", 0.60, 1.02), r("b", 0.52, 1.0)], rule)[0]["export"] == "b"
