import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from scglm_v2 import build
from scglm_v2.benchmarks import format_example, is_development
from scglm_v2.sources import blocked, chunk_text, clean_gutenberg, clean_stackexchange


class _Identity:
    def encode_batch(self, texts, add_special_tokens=False):
        class E:
            def __init__(self, t):
                self.ids = [len(t) % 60000 + 4]
        return [E(t) for t in texts]


def _setup(monkeypatch):
    monkeypatch.setattr(build, "_TOK", _Identity())
    monkeypatch.setattr(build, "_CFG", {"domain_blocklist": ["wikihow.com"], "targeted": {"pool_sources": []}})


@pytest.mark.parametrize("range_bytes", [1, 7, 64, 10_000])
def test_byte_ranges_keep_every_line_exactly_once(tmp_path, monkeypatch, range_bytes):
    _setup(monkeypatch)
    rows = [{"id": f"d{i}", "url": f"https://site{i % 3}.org/p{i}", "title": "", "text": "x" * (i % 13 + 1),
             "content_sha256": f"{i:064x}"} for i in range(200)]
    path = tmp_path / "raw.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    size, seen = path.stat().st_size, Counter()
    for i, start in enumerate(range(0, size, range_bytes)):
        build._reused_range(("wiki", str(path), start, min(size, start + range_bytes) - 1, f"p{i:05d}", str(tmp_path / "out")))
    for idx in (tmp_path / "out" / "wiki" / "parts").glob("*.idx.jsonl"):
        seen.update(json.loads(l)["id"] for l in idx.open())
    for panel in (tmp_path / "out" / "wiki" / "parts").glob("*.*.jsonl"):
        if not panel.name.endswith(".idx.jsonl"):
            seen.update(json.loads(l)["id"] for l in panel.open())
    assert seen == Counter(r["id"] for r in rows)


def test_domain_blocklist_matches_subdomains_only():
    assert blocked("https://www.wikihow.com/Do-X", ["wikihow.com"])
    assert blocked("https://es.wikihow.com/x", ["wikihow.com"])
    assert not blocked("https://notwikihow.com/x", ["wikihow.com"])
    assert not blocked(None, ["wikihow.com"])


def test_blocked_documents_are_dropped(tmp_path, monkeypatch):
    _setup(monkeypatch)
    rows = [{"id": "a", "url": "https://www.wikihow.com/x", "text": "hello", "content_sha256": "f" * 64},
            {"id": "b", "url": "https://ok.org/x", "text": "hello", "content_sha256": "e" * 64}]
    path = tmp_path / "raw.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    result = build._reused_range(("wiki", str(path), 0, path.stat().st_size - 1, "p0", str(tmp_path / "out")))
    assert result["counts"]["domain_blocklist"] == 1
    tokens = np.fromfile(tmp_path / "out" / "wiki" / "parts" / "p0.bin", dtype="<u2")
    assert tokens[-1] == 2 and len(tokens) == 2


def test_gutenberg_boilerplate_and_chunks():
    body = "Header\n*** START OF THE PROJECT GUTENBERG EBOOK X ***\nPara one.\n\nPara two.\n*** END OF THE PROJECT GUTENBERG EBOOK X ***\nLicense"
    clean = clean_gutenberg(body)
    assert clean == "Para one.\n\nPara two."
    assert chunk_text("a" * 10 + "\n\n" + "b" * 10, 12) == ["a" * 10, "b" * 10]


def test_stackexchange_usernames_removed():
    text = "# T\n\n(Asked by: username_0 on 2012-12-13)\n\nQ?\n\n### Answer by username_2 on 2014-04-15. Score: 6\n\nA. **username_1**: c"
    out = clean_stackexchange(text)
    assert "username_" not in out and "### Answer" in out


def test_benchmark_formatting_and_stable_holdout():
    item = format_example("winogrande", {"sentence": "Tom gave _ the ball.", "option1": "Ann", "option2": "Bob", "answer": "2"})
    assert item["text"] == "Tom gave Bob the ball."
    assert is_development("piqa", "abc") == is_development("piqa", "abc")
