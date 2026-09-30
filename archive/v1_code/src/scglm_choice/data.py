"""Audited primary/transfer sources, historical exposure and fresh raw panels."""
from collections import Counter, defaultdict
import io
import json
from pathlib import Path
import random
import re
import tarfile
import urllib.request
import xml.etree.ElementTree as ET
import zipfile

import numpy as np

from scglm.prepare_data import ExclusionIndex, ngram_hashes
from scglm_post.common import identity, normalized
from scglm_pipeline.sota_data import Protection
from .common import ROOT, DOMAINS, read, sha256, digest, write_json, read_jsonl, write_jsonl, event
from .objective import choice_rows


def fetch(url, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with urllib.request.urlopen(url, timeout=60) as response:
            content = response.read()
        temporary = path.with_suffix(".download")
        temporary.write_bytes(content)
        temporary.replace(path)
    return path


def primary_sources(cfg, directory):
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq
    rows, inventory = [], {}
    for spec in cfg["primary_sources"]:
        repo, revision = spec["repository"], spec["revision"]
        notice = Path(hf_hub_download(repo, "README.md", revision=revision, repo_type="dataset"))
        saved = directory / (repo.replace("/", "_") + ".README.md")
        saved.write_bytes(notice.read_bytes())
        record = {**spec, "notice_sha256": sha256(saved), "files": {}}
        if spec["domain"] == "science":
            parts = {}
            for split in ("train", "validation", "test"):
                name = f"data/{split}-00000-of-00001.parquet"
                path = Path(hf_hub_download(repo, name, revision=revision, repo_type="dataset"))
                record["files"][name] = {"path": str(path), "sha256": sha256(path)}
                parts[split] = pq.read_table(path).to_pylist()
        else:
            path = fetch(spec["url"], directory / "socialiqa-train-dev.zip")
            record["files"][path.name] = {"path": str(path), "sha256": sha256(path)}
            parts = {}
            with zipfile.ZipFile(path) as archive:
                for split in ("train", "dev"):
                    lines = archive.read(f"socialiqa-train-dev/{split}.jsonl").decode().splitlines()
                    labels = archive.read(f"socialiqa-train-dev/{split}-labels.lst").decode().splitlines()
                    if len(lines) != len(labels):
                        raise ValueError("Social IQa labels do not align")
                    parts[split] = [{**json.loads(line), "label": label} for line, label in zip(lines, labels)]
        for split, values in parts.items():
            for index, r in enumerate(values):
                if spec["domain"] == "science":
                    question = r["question"].strip()
                    choices = [r["correct_answer"], r["distractor1"], r["distractor2"], r["distractor3"]]
                    gold = r["correct_answer"]
                    facts = [question, r["support"].strip()]
                    group_text = r["support"].strip() or question
                else:
                    question = r["context"].strip() + "\n" + r["question"].strip()
                    choices = [r["answerA"], r["answerB"], r["answerC"]]
                    gold = choices[int(r["label"]) - 1]
                    facts = [r["context"].strip(), question]
                    group_text = r["context"].strip()
                if not all(c.strip() for c in choices) or len(set(map(normalized, choices))) != len(choices):
                    continue
                # Independent of labels, keep a fixed source-item choice ordering.
                random.Random(f"choice:{repo}:{split}:{index}").shuffle(choices)
                rows.append({"id": f"{repo}:{split}:{index}", "source": repo, "original_split": split,
                             "original_index": index, "revision": revision, "license": spec["license"],
                             "domain": spec["domain"], "question": question, "choices": choices,
                             "answer_index": choices.index(gold), "facts": facts,
                             "group_id": repo + ":" + identity(group_text),
                             "preview": index < cfg["preview_exclusion"]["first_rows_per_original_split"] or
                             any(s.lower() in question.lower() for s in cfg["preview_exclusion"]["text_fragments"])})
        inventory[repo] = record
    return rows, inventory


def group_related(rows):
    """Union source/context groups and shared question/support 13-grams."""
    parent = list(range(len(rows)))
    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    def union(a, b):
        a, b = root(a), root(b)
        parent[max(a,b)] = min(a,b)
    seen = {}
    for i, row in enumerate(rows):
        keys = [(row["source"], "group", row["group_id"])]
        for text in row.get("facts", [row["question"]]):
            if text.strip():
                keys.append((row["source"], "exact", identity(text)))
                keys.extend((row["source"], "ngram", k) for k in ngram_hashes(text))
        for key in keys:
            if key in seen:
                union(i, seen[key])
            else:
                seen[key] = i
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[root(i)].append(row)
    result = []
    for group in groups.values():
        key = digest(sorted(r["id"] for r in group))
        result.extend({**r, "group_id": key} for r in group)
    return result


def partition(rows, *, seed, min_items=500, min_groups=50, raw=False, min_tokens=131072):
    groups = defaultdict(list)
    for row in rows:
        groups[row["group_id"]].append(row)
    order = sorted(groups)
    random.Random(seed).shuffle(order)
    result = {"development": [], "confirmation": []}
    cursor = 0
    for panel in result:
        used_groups, tokens = 0, 0
        while ((tokens < min_tokens if raw else len(result[panel]) < min_items) or used_groups < min_groups):
            if cursor == len(order):
                raise ValueError(f"Insufficient mandatory {panel} coverage: {len(result[panel])} items, {used_groups} groups, {tokens} targets")
            group = groups[order[cursor]]
            cursor += 1
            result[panel].extend({**r, "split": panel} for r in group)
            used_groups += 1
            if raw:
                tokens += sum(len(r["tokens"])-1 for r in group)
    return result


class Exposure:
    """Conservative union of historical pools, replay and raw scoring evidence.

    Even unused rows in previously prepared posttraining pools are excluded.
    Corpus exact/MinHash protection comes from the existing verified LMDB.
    """
    def __init__(self, tokenizer, directory):
        self.index, self.exact, self.doc_ids = ExclusionIndex(), set(), set()
        self.files, self.seen_files = {}, set()
        self.tokenizer = tokenizer
        self.protection = Protection(read(ROOT / "configs/current_model.json"), tokenizer)
        candidates = [p for p in (ROOT / "data/posttraining").glob("*/*.jsonl")
                      if "choice_discrimination" not in str(p)]
        candidates += [p for p in (ROOT / "artifacts").rglob("*.jsonl")
                       if p.name in ("raw.jsonl", "raw_documents.jsonl", "replay.jsonl", "documents.jsonl")
                       and "choice_discrimination" not in str(p)]
        for path in sorted(set(candidates)):
            self.add_file(path)
        event(directory, "historical_exposure_indexed", files=len(self.files), document_ids=len(self.doc_ids), exact_texts=len(self.exact))

    def add(self, text):
        if isinstance(text, str) and text.strip():
            self.index.add(text)
            self.exact.add(identity(text))

    def add_file(self, path):
        h = sha256(path)
        self.files[str(path.relative_to(ROOT))] = h
        if h in self.seen_files:
            return
        self.seen_files.add(h)
        with path.open() as stream:
            for line in stream:
                row = json.loads(line)
                if "id" in row and ("token_count" in row or "tokens" in row or "replay" in path.name):
                    self.doc_ids.add(row["id"])
                    # Strip namespacing and replay chunk offsets; content IDs are SHA256.
                    self.doc_ids.update(re.findall(r"[0-9a-f]{64}", row["id"]))
                for key in ("prompt", "question", "context"):
                    self.add(row.get(key))
                for fact in row.get("split_facts", []):
                    self.add(fact)
                for msg in row.get("messages", []):
                    self.add(msg.get("content"))
                tokens = row.get("tokens")
                if tokens is None and "replay" in path.name:
                    tokens = row.get("input_ids")
                if tokens:
                    self.add(self.tokenizer.decode(tokens, skip_special_tokens=True))

    def blocked(self, text, *, pretraining=True):
        return (identity(text) in self.exact or self.index.matches(text) or
                self.protection.blocked(text, pretraining=pretraining))

    def close(self):
        self.protection.close()


def primary_panels(cfg, rows, exposure, tokenizer):
    rows = group_related(rows)
    rejected = set()
    reasons = Counter()
    for row in rows:
        reason = None
        if row["preview"]:
            reason = "browsed_preview_or_conservative_prefix"
        elif any(exposure.blocked(t) for t in row["facts"] if t.strip()):
            reason = "historical_or_benchmark_overlap"
        else:
            try:
                for template in cfg["presentations"]:
                    choice_rows(tokenizer, row, template)
            except ValueError:
                reason = "complete_token_boundary_or_length"
        if reason:
            rejected.add(row["group_id"])
            reasons[reason] += 1
    clean = [r for r in rows if r["group_id"] not in rejected]
    panels = {s: [] for s in ("development", "confirmation")}
    for domain in DOMAINS:
        split = partition([r for r in clean if r["domain"] == domain], seed=f"20260929:{domain}")
        for s in panels:
            panels[s].extend(split[s])
    return panels, {"rejections": dict(reasons), "rejected_groups": len(rejected), "eligible_items": len(clean),
                    "semantic_decontamination_certified": False,
                    "protections": "Exact text, 13-word matches, prepared-corpus exact/MinHash index; whole-group removal"}


def raw_panels(cfg, exposure, tokenizer, directory):
    retained = read(ROOT / "configs/current_model.json")
    corpus = {name: read(ROOT / spec["manifest"]) for name, spec in retained["corpora"].items()}
    receipts, pools = {}, defaultdict(list)
    # No controller use is permitted in any completed scratch phase.
    statuses = [ROOT / "runs" / name / "status.json" for name in ("baseline_1b", "continuation_12b", "continuation_13b")]
    for path in statuses:
        st = read(path)
        if st["status"] != "completed" or st.get("controller_windows_done") or st.get("controller_seconds", 0) or st.get("probe_tokens", 0):
            raise ValueError("Controller exposure is not auditable as unused")
        receipts[str(path.relative_to(ROOT))] = sha256(path)
    consumed = Counter()
    for path in statuses[1:]:
        consumed.update(read(path)["source_tokens"])
    # Check the final persisted stream state, not just status totals.
    import torch
    pointer = read(ROOT / "runs/continuation_13b/checkpoint_latest.json")
    checkpoint = Path(pointer["path"])
    if sha256(checkpoint) != pointer["sha256"]:
        raise ValueError("Final stream checkpoint changed")
    state = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
    for source, stream in state["streams"].items():
        if stream["cycles"] or stream["tokens_consumed"] != consumed[source] or stream["cursor"] != consumed[source]:
            raise ValueError("Continuation history contains wrapping or inconsistent cursors")
    receipts[str(checkpoint)] = pointer["sha256"]
    del state
    seen = set()
    for name in ("legacy", "continuation"):
        for source, info in corpus[name]["sources"].items():
            stratum = f"{name}_{source}"
            split_names = [s for s in info["splits"] if s.startswith("controller_")] if name == "legacy" else ["train"]
            for split in split_names:
                spec = info["splits"][split]
                index, binary = Path(spec["index_path"]), Path(spec["path"])
                for p, expected in ((index, spec["index_sha256"]), (binary, spec["sha256"])):
                    if sha256(p) != expected:
                        raise ValueError("Raw corpus source changed")
                    receipts[str(p)] = expected
                mmap = np.memmap(binary, mode="r", dtype="<u2")
                with index.open() as stream:
                    for line in stream:
                        entry = json.loads(line)
                        start, length = entry["offset"], entry["length"]
                        # Last training read has one lookahead token at the cursor.
                        if name == "continuation" and start <= consumed[source]:
                            continue
                        content = entry["content_sha256"]
                        if content in exposure.doc_ids or entry["id"] in exposure.doc_ids or content in seen:
                            continue
                        # Reject truncated stored documents using their real EOS.
                        if length < 2 or int(mmap[start + length - 1]) != tokenizer.eos_token_id:
                            continue
                        full = mmap[start:start+length].astype(int).tolist()
                        text = tokenizer.decode(full, skip_special_tokens=True)
                        if exposure.blocked(text, pretraining=False):
                            continue
                        tokens = full[:1024]  # <=1,023 targets, whole document kept out of training.
                        group = identity(entry.get("url") or content)
                        pools[stratum].append({"id": entry["id"], "source": stratum, "group_id": group,
                            "tokens": tokens, "full_document_tokens": length, "full_tokens_sha256": digest(full),
                            "content_sha256": content, "offset": start, "original_split": split,
                            "index_path": str(index), "source_file": entry["source_file"],
                            "url": entry.get("url"), "exposure_cursor": consumed[source] if name == "continuation" else None})
                        seen.add(content)
                        # Freeze enough whole documents for both panels, no model scores involved.
                        if sum(len(r["tokens"])-1 for r in pools[stratum]) >= 400000 and len(pools[stratum]) >= 200:
                            break
                del mmap
                if sum(len(r["tokens"])-1 for r in pools[stratum]) >= 400000:
                    break
            event(directory, "raw_stratum_audited", stratum=stratum, eligible_documents=len(pools[stratum]),
                  scored_capacity=sum(len(r["tokens"])-1 for r in pools[stratum]))
    panels = {s: [] for s in ("development", "confirmation")}
    for stratum in cfg["evaluation"]["raw_strata"]:
        split = partition(pools[stratum], seed=f"20260929:{stratum}", raw=True, min_groups=50)
        for s in panels:
            panels[s].extend(split[s])
    return panels, {"source_receipts": receipts, "continuation_consumed_targets": dict(consumed),
                    "controller_use": "none in all three completed scratch phases; historical raw/replay IDs excluded",
                    "raw_cap": 1023, "scope": "Prepared corpus dedup plus audited source positions; not semantic decontamination"}


def transfer_sources(cfg, directory, exposure, tokenizer):
    """Official COPA split assignments and XML schema twins, diagnostic only."""
    result = {s: [] for s in ("development", "confirmation")}
    report = {}
    for name in ("copa", "winograd"):
        try:
            if name == "copa":
                url = "https://asgordon.github.io/downloads/COPA-resources.tgz"
                path = fetch(url, directory / "COPA-resources.tgz")
                rows_by_split = {}
                with tarfile.open(path) as archive:
                    for original, split in (("dev", "development"), ("test", "confirmation")):
                        member = next(m for m in archive.getmembers() if m.name.endswith(f"datasets/copa-{original}.xml"))
                        xml = ET.fromstring(archive.extractfile(member).read())
                        rows_by_split[split] = []
                        for item in xml.findall("item"):
                            premise = item.findtext("p").strip()
                            choices = [item.findtext("a1").strip(), item.findtext("a2").strip()]
                            choices = [x[0].lower()+x[1:] for x in choices]
                            prompt = premise.rstrip(".") + (" because" if item.attrib["asks-for"] == "cause" else " therefore")
                            row = {"id": "copa:" + item.attrib["id"], "group_id": identity(premise),
                                   "question": premise, "causal_prompt": prompt, "choices": choices,
                                   "answer_index": int(item.attrib["most-plausible-alternative"])-1,
                                   "domain": "narrative", "source": "copa", "license": "BSD-2-Clause"}
                            if any(p in premise for p in ("broke his toe", "tipped the bottle", "neighbor's door")):
                                continue
                            rows_by_split[split].append(row)
            else:
                url = "https://cs.nyu.edu/~davise/papers/WinogradSchemas/WSCollection.xml"
                path = fetch(url, directory / "WSCollection.xml")
                xml = ET.fromstring(path.read_bytes())
                rows = []
                for index, schema in enumerate(xml.findall("schema")):
                    def txt(tag):
                        node = schema.find(tag)
                        return " ".join("".join(node.itertext()).split()) if node is not None else ""
                    prefix, pronoun, suffix = txt("text/txt1"), txt("text/pron"), txt("text/txt2")
                    choices = [" ".join("".join(a.itertext()).split()) for a in schema.findall("answers/answer")]
                    if len(choices) != 2 or not prefix or not suffix:
                        continue
                    answer = txt("correctAnswer").strip(". ")
                    group = txt("source") or str(index // 2)
                    rows.append({"id": f"winograd:{index}", "group_id": "winograd:candidates:" + digest(sorted(map(normalized, choices))),
                                 "question": prefix + " " + pronoun + " " + suffix,
                                 "choices": choices, "answer_index": {"A":0,"B":1}[answer],
                                 "suffix_probe": {"prefix": prefix + " ", "suffix": " " + suffix},
                                 "source": "winograd", "source_note": group, "domain": "coreference_suffix", "license": "CC-BY-4.0"})
                rows_by_split = {"development": [], "confirmation": []}
                # Candidate-set groups conservatively keep twins together even
                # around unpaired schemas where index//2 would split a pair.
                groups = sorted({r["group_id"] for r in rows})
                random.Random(20260929).shuffle(groups)
                membership = {g: ("development" if i%2 == 0 else "confirmation") for i,g in enumerate(groups)}
                for r in rows:
                    rows_by_split[membership[r["group_id"]]].append(r)
            rejected = set()
            memberships = defaultdict(set)
            for split, rows in rows_by_split.items():
                for row in rows:
                    memberships[row["group_id"]].add(split)
            rejected.update(g for g, splits in memberships.items() if len(splits) > 1)
            for rows in rows_by_split.values():
                for row in rows:
                    try:
                        choice_rows(tokenizer, row)
                        if exposure.blocked(row["question"]):
                            rejected.add(row["group_id"])
                    except ValueError:
                        rejected.add(row["group_id"])
            coverage = {}
            for split, rows in rows_by_split.items():
                clean = [r for r in rows if r["group_id"] not in rejected]
                available = len(clean) >= cfg["transfer_min_items"] and len({r["group_id"] for r in clean}) >= cfg["transfer_min_groups"]
                coverage[split] = {"available": available, "clean_items": len(clean), "clean_groups": len({r["group_id"] for r in clean})}
                if available:
                    result[split].extend({**r, "split": split} for r in clean)
            report[name] = {"url": url, "sha256": sha256(path), "coverage": coverage, "diagnostic_only": True}
        except (OSError, ValueError, KeyError, ET.ParseError, StopIteration) as error:
            report[name] = {"available": False, "reason": repr(error), "diagnostic_only": True}
    return result, report
