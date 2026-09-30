"""CPU checks for the retained instruction preparation and likelihood evaluation."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from test_posttraining import tokenizer, tiny_model, example
from scglm_post.common import (encode_example, write_json, write_jsonl, sha256, render_prompt,
                               tokenizer_identity, publish_export)
from scglm_post.procedural import generate as procedural
from scglm_pipeline.battery import generate, correct, FAMILIES
from scglm_pipeline.evaluate import response_logps, generate_many, choice_scores, raw_metrics
from scglm_pipeline.common import read, digest, raw_panels
from scglm_pipeline.prepare import rebalance
from scglm_pipeline.selection import paired_raw_intervals


def test_battery_disjoint_balanced_variable_facts_and_adversarial_answers():
    development, confirmation = generate("development"), generate("confirmation")
    assert not {r["id"] for r in development} & {r["id"] for r in confirmation}
    assert not {r["group"] for r in development} & {r["group"] for r in confirmation}
    assert len({r["answer"] for r in development if r["family"] == "grounded_choice"}) == 7
    for rows in (development, confirmation):
        assert {r["family"] for r in rows} == set(FAMILIES)
        for row in rows:
            assert correct(row, row["answer"])
            assert not correct(row, row["answer"]+" unrelated claim")
            if row["choices"]:
                assert len(set(row["choices"])) == 4
                assert row["choices"][row["answer_index"]] == row["answer"]
                assert all(not correct(row, c) for c in row["choices"] if c != row["answer"])
        assert {len({r["group"] for r in rows if r["family"] == f}) for f in FAMILIES} == {18}


def test_response_logps_match_manual_sum_and_mask_padding(tokenizer):
    torch.set_num_threads(1)
    model = tiny_model(tokenizer).eval()
    rows = [example(tokenizer, "yes"), example(tokenizer, "a longer answer")]
    values, counts = response_logps(model, rows)
    for i, row in enumerate(rows):
        inputs = torch.tensor([row["input_ids"]])
        logits = model(inputs).logits[0].log_softmax(-1)
        expected = sum(logits[j-1, row["input_ids"][j]] for j in range(row["prompt_length"], len(row["input_ids"])))
        torch.testing.assert_close(values[i], expected)
        assert counts[i] == row["target_tokens"]


def test_left_padded_greedy_generation_agrees_with_individual_calls(tokenizer):
    torch.set_num_threads(1)
    model = tiny_model(tokenizer).eval()
    prompts = ["Hello", "A longer input with words"]
    together = generate_many(model, tokenizer, prompts, max_new_tokens=4, batch_size=2)
    alone = generate_many(model, tokenizer, prompts, max_new_tokens=4, batch_size=1)
    assert together == alone


def test_choice_likelihood_scores_answers_without_eos_or_letters(tokenizer):
    model = tiny_model(tokenizer).eval()
    row = {"prompt": "Choose an answer", "choices": ["yes", "a longer answer"], "answer_index": 0}
    scores = choice_scores(model, tokenizer, row)
    for i, answer in enumerate(row["choices"]):
        r = encode_example([{"role": "user", "content": row["prompt"]}, {"role": "assistant", "content": answer}], tokenizer)
        inputs = torch.tensor([r["input_ids"][:-1]])
        logits = model(inputs).logits[0].log_softmax(-1)
        expected = sum(logits[j-1, r["input_ids"][j]] for j in range(r["prompt_length"], len(r["input_ids"])-1))
        assert scores["log_likelihoods"][i] == pytest.approx(float(expected.detach()), abs=1e-5)
        assert scores["target_lengths"][i] == r["target_tokens"]-1


def test_legacy_and_continuation_wiki_are_distinct(tmp_path):
    config = {"corpora": {}}
    raw = []
    for namespace, weights in (("legacy", {"web": .8, "wiki": .2}), ("continuation", {"edu": .6, "dclm": .3, "wiki": .1})):
        sources = {}
        for source in weights:
            path = tmp_path/f"{namespace}-{source}.jsonl"
            write_jsonl(path, [{"id": "same-id", "tokens": [1, 2, 3]}])
            sources[source] = {"splits": {"selection": {"jsonl_path": str(path), "jsonl_sha256": sha256(path)}}}
            raw.append({"id": "same-id", "source": namespace+"/"+source, "nll_sum": 10 if namespace == "legacy" else 30, "token_count": 10})
        manifest = tmp_path/(namespace+".json")
        write_json(manifest, {"status": "complete", "sources": sources})
        config["corpora"][namespace] = {"weights": weights, "manifest": str(manifest), "sha256": sha256(manifest)}
    rows = raw_panels(config, "development")
    assert len({(r["source"], r["id"]) for r in rows}) == 5
    metrics = raw_metrics(raw, config)
    assert metrics["mixtures"]["legacy"]["fixed_q_nll"] == 1
    assert metrics["mixtures"]["continuation"]["fixed_q_nll"] == pytest.approx(3)
    assert metrics["sources"]["legacy/wiki"]["nll"] != metrics["sources"]["continuation/wiki"]["nll"]


def test_paired_raw_bootstrap_checks_documents_and_counts():
    rows = [{"id": str(i), "source": "continuation/wiki", "nll_sum": 30, "token_count": 10} for i in range(5)]
    other = [{**r, "nll_sum": 31} for r in rows]
    result = paired_raw_intervals(rows, other, 42, 100)["continuation/wiki"]
    assert result == pytest.approx({"nll_change": .1, "lower_95": .1, "upper_95": .1})
    with pytest.raises(ValueError, match="identities"):
        paired_raw_intervals(rows, other[:-1], 42, 100)
    with pytest.raises(ValueError, match="counts"):
        paired_raw_intervals(rows, [{**r, "token_count": 11} for r in other], 42, 100)










def test_procedural_cap_preserves_groups_and_human_control():
    rows = [{"id": "h", "source": "human", "group": "h", "target_tokens": 100}]
    rows += [{"id": f"{g}{i}", "source": "procedural", "group": str(g), "target_tokens": 10} for g in range(3) for i in range(2)]
    mixed, human = rebalance(rows, .2)
    assert len(human) == 1 and len(mixed) == 3
    assert sum(r["source"] == "procedural" for r in mixed) == 2
    assert len({r["group"] for r in mixed if r["source"] == "procedural"}) == 1












def test_tokenizer_identity_allows_only_explicit_noop_processor(tokenizer, tmp_path):
    original, exported = tmp_path/"original", tmp_path/"exported"
    original.mkdir(); exported.mkdir()
    tokenizer.save_pretrained(original); tokenizer.save_pretrained(exported)
    raw = read(original/"tokenizer.json"); raw["post_processor"] = None; write_json(original/"tokenizer.json", raw)
    expected = {"config": {"tokenizer": str(original)}, "tokenizer_sha256": sha256(original/"tokenizer.json")}
    assert tokenizer_identity(exported, expected)["identity_template_only"]
    bad = read(exported/"tokenizer.json"); bad["model"]["vocab"]["<|unk|>"] = 99
    write_json(exported/"tokenizer.json", bad)
    with pytest.raises(ValueError, match="vocabulary"):
        tokenizer_identity(exported, expected)











