import json
import random

import lmdb
import numpy as np

from scglm.prepare_extension import canonical_url, sketch, matching_reason, remember, read_progress
from scglm.prepare_data import fingerprint


def make_text(seed=1, words=1000):
    rng = random.Random(seed)
    return " ".join(f"term{rng.randrange(10000)}" for _ in range(words))


def row(text, url=""):
    return {"digest":fingerprint(text),"sketch":sketch(text),"url":canonical_url(url)}


def test_cross_source_near_duplicate_and_distinct_documents(tmp_path):
    original = make_text()
    edited = original + " Added a brief unrelated concluding sentence."
    assert np.count_nonzero(np.frombuffer(sketch(original),dtype="<u4") == np.frombuffer(sketch(edited),dtype="<u4")) >= 28
    env = lmdb.open(str(tmp_path/"db"), map_size=10_000_000)
    with env.begin(write=True) as txn:
        remember(txn,row(original,"http://example.org/article#section"))
    with env.begin() as txn:
        assert matching_reason(txn,row(original.upper())) == "exact_document"
        assert matching_reason(txn,row(edited)) == "approximate_near_document"
        assert matching_reason(txn,row(make_text(5),"https://www.example.org/article")) == "same_url"
        assert matching_reason(txn,row(make_text(2))) is None
    env.close()


def test_dedup_state_and_progress_commit_together(tmp_path):
    env=lmdb.open(str(tmp_path/"db"),map_size=10_000_000)
    a,b=row(make_text(3)),row(make_text(4))
    with env.begin(write=True) as txn:
        remember(txn,a)
        txn.put(b"!progress",json.dumps({"output_bytes":10}).encode())
    try:
        with env.begin(write=True) as txn:
            remember(txn,b)
            txn.put(b"!progress",json.dumps({"output_bytes":20}).encode())
            raise RuntimeError("simulated interrupted preparation")
    except RuntimeError:
        pass
    assert read_progress(env)=={"output_bytes":10}
    with env.begin() as txn:
        assert matching_reason(txn,a)=="exact_document"
        assert matching_reason(txn,b) is None
    env.close()


def test_canonical_urls_preserve_semantic_queries():
    assert canonical_url("http://www.example.com/a/#heading")=="https://example.com/a"
    assert canonical_url("https://example.com/?id=1") != canonical_url("https://example.com/?id=2")
