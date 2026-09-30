"""Mid-training study: quality rules, unused-span search, isolation of held-out groups and arm shards."""
import json
from pathlib import Path

import numpy as np
import pytest

from test_posttraining import tokenizer
from scglm.data import SourceStream
from scglm_study.common import ROOT, load_config, read, sha256, sources
from scglm_midtrain.quality import coherence, features, hard_failures
from scglm_midtrain.prepare import (first_entry_after, heldout_rows, scan, select_curated, select_random,
                                    split_heldout, write_shard, length_bin)
from scglm_midtrain.runner import trainer_config

CFG = load_config(ROOT / "configs/midtrain_quality.json")
RULES, WEIGHTS = CFG["selection"]["hard_filters"], CFG["selection"]["coherence_weights"]
PROSE = ("The river carries water from the mountains to the sea. Along the way, it shapes the valley and "
         "feeds the fields that farmers have worked for generations. In spring the snow melts and the flow "
         "rises quickly, so the people who live near the banks watch the water with care.\n\n"
         "Scientists measure the flow every day. They record how the level changes with the weather and "
         "use those records to warn the towns downstream when a flood is likely to come.")


def test_quality_rules_pass_prose_and_reject_repetition_and_boilerplate():
    f = features(PROSE)
    assert hard_failures(f, RULES) == []
    assert 0.8 < coherence(f, WEIGHTS) <= 1
    spam = "\n".join(["Click here to subscribe now"] * 40)
    assert {"duplicate_lines", "boilerplate", "stopwords"} <= set(hard_failures(features(spam), RULES))
    listing = "\n".join(f"- item {i} for sale" for i in range(60))
    assert "bullet_lines" in hard_failures(features(listing), RULES)
    with pytest.raises(ValueError):
        coherence(f, {**WEIGHTS, "terminal_lines": 0.5})


def test_unused_span_search_is_exact_at_every_boundary(tmp_path):
    index = tmp_path / "train.idx.jsonl"
    offsets = [0, 7, 19, 20, 55, 100]
    lines = [json.dumps({"id": str(i), "offset": o, "length": 5}) + "\n" for i, o in enumerate(offsets)]
    index.write_text("".join(lines))
    starts = np.cumsum([0] + [len(l.encode()) for l in lines]).tolist()
    for cursor in (-1, 0, 6, 7, 19, 20, 54, 99, 100, 500):
        expected = next((starts[i] for i, o in enumerate(offsets) if o > cursor), starts[-1])
        assert first_entry_after(index, cursor) == expected


def _docs(n, seed=0):
    rng = np.random.default_rng(seed)
    docs = []
    for i in range(n):
        length = int(rng.integers(40, 3000))
        score = None if i % 3 == 0 else float(rng.random())
        docs.append({"id": f"d{i}", "length": length, "score": score, "host": f"h{i % 50}", "group_id": f"g{i // 2}",
                     "offset": 0, "content_sha256": str(i), "url": None})
    return docs


def test_heldout_groups_never_enter_training_and_arms_meet_targets():
    pool = _docs(600)
    cfg = {"selection": {"heldout_documents_per_source": 20, "heldout_min_scored_tokens_per_source": 5000}}
    held, remaining = split_heldout(cfg, pool, "edu", 7)
    assert len(held) >= 20 and not {d["group_id"] for d in held} & {d["group_id"] for d in remaining}
    target = 150000
    random_docs = select_random(remaining, target, 7)
    assert select_random(remaining, target, 7) == random_docs
    assert sum(d["length"] for d in random_docs) >= target
    spec = {"length_bin_edges": [128, 256, 512, 1024, 2048], "max_host_document_fraction": 0.1}
    curated, audit = select_curated(remaining, random_docs, target, spec)
    assert sum(d["length"] for d in curated) >= target
    assert all(d["score"] is not None for d in curated)
    assert len({d["id"] for d in curated}) == len(curated)
    hosts = {}
    for d in curated:
        hosts[d["host"]] = hosts.get(d["host"], 0) + 1
    assert max(hosts.values()) <= audit["host_cap"]
    # Length mix follows the random arm unless a bin's shortfall is recorded.
    for b in range(6):
        share_r = sum(d["length"] for d in random_docs if length_bin(d["length"], spec["length_bin_edges"]) == b) / target
        share_c = sum(d["length"] for d in curated if length_bin(d["length"], spec["length_bin_edges"]) == b) / target
        assert share_c >= share_r - 0.05 or str(b) in audit["length_bin_shortfall_tokens"]
    with pytest.raises(ValueError):
        select_random(remaining, 10 ** 9, 7)


def _corpus(tmp_path, tok, texts, eos=2):
    tokens, index, offset = [], [], 0
    for i, text in enumerate(texts):
        ids = tok.encode(text, add_special_tokens=False) + [eos]
        if i == 3:
            ids = ids[:-1] + [5]  # stored without its real EOS
        tokens.extend(ids)
        index.append({"id": f"edu:{i}", "content_sha256": f"{i:064d}", "url": f"https://site{i}.org/a",
                      "source_file": "f", "offset": offset, "length": len(ids)})
        offset += len(ids)
    (tmp_path / "train.bin").write_bytes(np.asarray(tokens, dtype="<u2").tobytes())
    (tmp_path / "train.idx.jsonl").write_text("".join(json.dumps(r) + "\n" for r in index))
    manifest = {"sources": {"edu": {"splits": {"train": {"path": str(tmp_path / "train.bin"), "sha256": "x",
                                                         "index_path": str(tmp_path / "train.idx.jsonl"), "index_sha256": "y"}}}}}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    return index


def test_scan_skips_consumed_exposed_truncated_documents(tmp_path, tokenizer, monkeypatch):
    import scglm_midtrain.prepare as prepare
    texts = [PROSE] * 8
    index = _corpus(tmp_path, tokenizer, texts)
    cfg = {"corpus": {"manifest": str(tmp_path / "manifest.json")}, "tokenizer": {"eos_id": 2},
           "selection": {**CFG["selection"], "min_document_tokens": 4}}
    monkeypatch.setattr(prepare, "ROOT", Path("/"))
    excluded = {"contents": {f"{5:064d}"}, "groups": set()}
    pool, counts, _ = scan(cfg, "edu", index[1]["offset"], excluded, tokenizer.backend_tokenizer, 1, tmp_path)
    ids = [d["id"] for d in pool]
    # Doc 0/1 start at or before the cursor; 3 lacks EOS; 5 is historically exposed.
    assert ids == ["edu:2", "edu:4", "edu:6", "edu:7"]
    assert counts["historical_exposure"] == 1 and counts["short_or_truncated"] == 1
    assert all(d["score"] is not None for d in pool)
    with pytest.raises(ValueError):
        scan(cfg, "edu", index[1]["offset"], excluded, tokenizer.backend_tokenizer, 10 ** 9, tmp_path)


def test_shards_preserve_whole_documents_and_stream_exactly(tmp_path, tokenizer):
    index = _corpus(tmp_path, tokenizer, [PROSE[:80 + 10 * i] for i in range(6)])
    docs = [{**e, "host": "h", "group_id": e["id"], "score": 1.} for e in index if e["id"] != "edu:3"]
    spec = {"path": str(tmp_path / "train.bin")}
    shard = write_shard(docs, spec, tmp_path / "arm" / "edu", seed=3)
    source = np.memmap(tmp_path / "train.bin", dtype="<u2", mode="r")
    written = np.memmap(shard["path"], dtype="<u2", mode="r")
    entries = [json.loads(l) for l in Path(shard["index_path"]).read_text().splitlines()]
    assert shard["tokens"] == sum(d["length"] for d in docs) == len(written)
    for e in entries:
        np.testing.assert_array_equal(written[e["offset"]:e["offset"] + e["length"]],
                                      source[e["source_offset"]:e["source_offset"] + e["length"]])
        assert written[e["offset"] + e["length"] - 1] == 2
    stream = SourceStream(shard["path"], seq_len=16)
    x, y = stream.next_batch(2)
    np.testing.assert_array_equal(x.reshape(-1)[1:], y.reshape(-1)[:-1])
    rows = heldout_rows(docs[:2], spec, "heldout_edu")
    assert rows[0]["tokens"] == source[docs[0]["offset"]:docs[0]["offset"] + docs[0]["length"]].tolist()


def test_registered_config_matches_the_parent_run_and_trainer_contract():
    parent = read(ROOT / CFG["parent"]["run"] / "run.json")["config"]
    config = trainer_config(CFG, "curated")
    for key in ("betas", "adam_epsilon", "weight_decay"):
        assert config[key] == parent[key]
    assert config["target_tokens"] == CFG["training"]["targets_per_arm"] == 977 * 64 * 1024
    assert config["source_weights"] == parent["source_weights"]
    assert config["inherit_data_state"] is False and config["validation_every"] == 0
    assert config["initial_learning_rate"] * 1 == pytest.approx(config["learning_rate"] * config["min_lr_ratio"])
    assert trainer_config(CFG, "random")["seed"] == config["seed"]
    assert Path(config["run_dir"]).is_relative_to(ROOT / CFG["isolation"]["output_root"])
    assert sha256(ROOT / CFG["corpus"]["manifest"]) == CFG["corpus"]["sha256"]
    assert set(sources(CFG)) >= {"src/scglm/train.py", "src/scglm_midtrain/prepare.py", "configs/midtrain_quality.json"}


def test_single_host_source_is_not_capped_by_host():
    pool = [{**d, "host": "en.wikipedia.org"} for d in _docs(600)]
    reference = select_random(pool, 150000, 7)
    spec = {"length_bin_edges": [128, 256, 512, 1024, 2048], "max_host_document_fraction": 0.002}
    with pytest.raises(ValueError):
        select_curated(pool, reference, 150000, spec)
    curated, audit = select_curated(pool, reference, 150000, spec, host_cap=False)
    assert sum(d["length"] for d in curated) >= 150000
    assert CFG["selection"]["host_cap_sources"] == ["edu", "dclm"]


def test_trainer_model_config_matches_the_parent_lineage_exactly():
    """The retained trainer compares its derived model config (including seed) with the parent's."""
    record = read(ROOT / CFG["parent"]["run"] / "run.json")
    config = trainer_config(CFG, "random")
    derived = read(ROOT / config["model_config"])
    derived["seed"] = config["seed"]
    for kind, token in CFG_SPECIAL.items():
        derived[f"{kind}_token_id"] = token
    assert derived == record["model_config"]
    assert config["seed"] == record["config"]["seed"]
    assert sha256(ROOT / CFG["tokenizer"]["path"]) == record["tokenizer_sha256"]


CFG_SPECIAL = {"pad": 0, "bos": 1, "eos": 2}
