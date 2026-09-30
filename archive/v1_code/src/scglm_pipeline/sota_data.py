"""Pinned, protected training tasks and raw replay for the replacement pipeline."""
from __future__ import annotations
from collections import Counter, defaultdict
import json
from pathlib import Path
import random
import shutil
import tempfile
import numpy as np
from .common import ROOT, read, resolve, sha256, digest, write_json, write_jsonl, read_jsonl, check_file, raw_panels
from .task_registry import task, task_identity, audit_tasks, encode_response, prompt_text, raw_row, PRIMARY_DOMAINS, VERSION
from scglm_post.common import identity, normalized, load_manifest as old_manifest, tokenizer_identity, render_prompt


def group_splits(records, seed):
    """Assign intact source groups, then remove secondary facts bridging splits."""
    by_group = defaultdict(list)
    for row in records:
        by_group[row["group_id"]].append(row)
    groups = sorted(by_group)
    random.Random(seed).shuffle(groups)
    targets = {"train": .8*len(records), "development": .1*len(records), "confirmation": .1*len(records)}
    counts = Counter(); assigned = []
    # Assign large groups first with deterministic randomized tie breaking.
    groups.sort(key=lambda g: -len(by_group[g]))
    for group in groups:
        split = max(targets, key=lambda s: targets[s]-counts[s])
        for row in by_group[group]:
            updated={**row,"split":split}
            if row.get("schema")==VERSION:updated["id"]=task_identity(updated)
            assigned.append(updated)
        counts[split] += len(by_group[group])
    fact_splits = defaultdict(set)
    for row in assigned:
        for fact in row.get("split_facts", []):
            fact_splits[identity(fact)].add(row["split"])
    bridges = {k for k, splits in fact_splits.items() if len(splits)>1}
    good = [r for r in assigned if not any(identity(f) in bridges for f in r.get("split_facts", []))]
    return good, {"secondary_fact_bridges": len(bridges), "removed_rows": len(assigned)-len(good)}


def public_tasks(spec, cache, seed):
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq
    source = spec["repository"]
    pin = spec["revision"]
    if len(pin) != 40:
        raise ValueError("Data requires an immutable source commit")
    path = Path(hf_hub_download(source, spec["file"], revision=pin, repo_type="dataset"))
    notice = Path(hf_hub_download(source, "README.md", revision=pin, repo_type="dataset"))
    cache.mkdir(parents=True, exist_ok=True)
    shutil.copy2(notice, cache / (source.replace("/", "_")+".README.md"))
    records=[]; source_hash=sha256(path)
    for row in pq.read_table(path).to_pylist():
        labels, choices = row["choices"]["label"], row["choices"]["text"]
        if row["answerKey"] not in labels or len(set(map(normalized,choices))) != len(choices):
            continue
        answer = choices[labels.index(row["answerKey"])]
        domain = spec["domain"]
        facts = [row[k] for k in ("fact1","fact2") if row.get(k)]
        group_key = row.get("fact1", row.get("question_concept", row["question"]))
        group = source+":"+identity(group_key)
        style = "raw" if int(digest(row["id"])[0],16)%2 == 0 else "chat"
        shuffled = list(choices); random.Random(str(seed)+row["id"]).shuffle(shuffled)
        prompt = row["question"].strip()+"\nChoices: "+" | ".join(shuffled)+"\nReturn the answer text only."
        records.append(task(source=source, group=group, split="train", family=domain,
            prompt=prompt, answer=answer, domain=domain, choices=shuffled, bucket="knowledge",
            prompt_style=style, license=spec["license"], split_facts=[row["question"],*facts],
            provenance={"source_id": row["id"], "revision": pin, "original_split": "train",
                        "source_sha256": source_hash}, reference_facts=facts, control=False))
    records, report = group_splits(records, seed)
    # The source supplies both supporting facts. Preserve them as short SFT
    # demonstrations, without claiming that prose has an automated RL oracle.
    explanations=[]
    for r in records:
        if r["split"]=="train" and r["reference_facts"]:
            response="Supporting facts:\n"+"\n".join(r["reference_facts"])+"\nAnswer: "+r["answer"]
            explanations.append(task(source=source,group=r["group_id"],split="train",family=domain+"/supporting_facts",
                prompt=r["prompt"].replace("Return the answer text only.","Give the supporting facts, followed by the answer."),
                answer=response,verifier="none",bucket="knowledge",license=spec["license"],prompt_style=r["prompt_style"],
                provenance=r["provenance"],split_facts=r["split_facts"],control=False))
    records+=explanations
    return records, {"repository": source, "revision": pin, "file": spec["file"],
        "cache_path": str(path), "sha256": sha256(path), "license": spec["license"],
        "notice_sha256": sha256(notice), "split_report": report}


def generated_tasks(split, count, seed):
    """Different wrapper/composition/operand partitions, including suffix probes."""
    phase = {"train":0,"development":1,"confirmation":2}[split]
    rng = random.Random(seed + phase*100003)
    wrappers = (("Read this record", "Use these facts", "Consider the following account"),
                ("Consult this report", "The complete evidence follows", "Refer to this account"),
                ("According to this entry", "Base your answer on this note", "Review this description"))[phase]
    rows=[]
    for family in ("grounded", "arithmetic", "logic", "sorting", "format", "coreference"):
        for i in range(count):
            difficulty = i % 3
            # Independent generated fact worlds are source groups. Template count
            # is reported separately; source groups are not claimed as templates.
            template = i % 60
            group = f"generated:{split}:{family}:world{i}"
            a = rng.randrange((phase+1)*10, (phase+2)*10) if difficulty else rng.randrange(1,10)
            b = rng.randrange(1,10); c = rng.randrange(1,10)
            style = "raw" if i % 2 == 0 else "chat"
            domain = None; choices = []; verifier="exact"; bucket="procedural"; suffix_probe=None
            entity = f"Depot-{split}-{i:06d}"
            wrapper = wrappers[template%3]
            if family == "grounded":
                values = rng.sample(range(20,999),4)
                answer = str(values[0]); choices = list(map(str,values)); rng.shuffle(choices)
                modes = (
                    f"{entity} contains {answer} blue boxes and {values[1]} red boxes. How many blue boxes are there?",
                    f"{entity} stores {answer} parcels. Every parcel has {b} stamps. How many parcels are stored?",
                    f"The manager of {entity} is Person-{split}-{i}. That person's room number is {answer}. What is the manager's room number?")
                prompt = f"{wrapper}: {modes[difficulty]} Reply with the integer only."
                domain="grounded"; bucket="grounded"; verifier="integer"
            elif family == "arithmetic":
                operands=[a,b] if difficulty<2 else [a,b,c]+[rng.randrange(1,10) for _ in range(phase)]
                answer = str(sum(operands)); expression=" + ".join(map(str,operands))
                prompt = f"{wrapper}: case {entity}. Compute {expression}. Reply with the integer only."
                verifier="integer"
            elif family == "logic":
                names=[f"Item-{split}-{i}-{j}" for j in range(3)]
                answer=names[0]; choices=list(names); rng.shuffle(choices)
                prompt=f"{wrapper}: {names[0]} is heavier than {names[1]}. {names[1]} is heavier than {names[2]}. Which item is heaviest? Return its name only."
            elif family == "sorting":
                # Hold out list lengths beyond the training distribution.
                values=rng.sample(range(0,100), 2+difficulty+phase)
                answer=", ".join(map(str,sorted(values))); verifier="integer_list"
                prompt=f"{wrapper}: case {entity}. Sort {', '.join(map(str,values))} ascending. Return comma-separated integers."
            elif family == "coreference":
                names=[f"Person-{split}-{i}-A",f"Person-{split}-{i}-B"]
                owner=names[i%2]; answer=owner; choices=list(names); rng.shuffle(choices)
                prompt=f"{wrapper}: {names[0]} met {names[1]} at {entity}. {owner} carried the green bag. Who carried the green bag? Return the person's name only."
                suffix_probe={"prefix":f"{names[0]} met {names[1]}. {owner} carried the green bag. The person with the green bag, ",
                              "suffix":", carried it home.","candidates":choices,"answer_index":choices.index(owner)}
            else:
                answer=json.dumps({"number": a}); verifier="json_integer"
                prompt=f"{wrapper}: {entity} has number {a}. Return a JSON object with exactly the integer field number."
            rows.append(task(source="program", group=group, split=split, family=family, prompt=prompt,
                answer=answer, verifier=verifier, difficulty=difficulty, bucket=bucket, domain=domain,
                prompt_style=style, choices=choices, provenance={"generator": VERSION,"seed":seed,"template":f"{phase}:{family}:{difficulty}:{template%3}","independent_unit":"generated_fact_world"},
                control=False, formatting_only=family=="format",suffix_probe=suffix_probe))
    return rows


class Protection:
    def __init__(self, config, tokenizer):
        import lmdb
        from scglm.prepare_data import ExclusionIndex
        from scglm.prepare_extension import ExclusionUnpickler
        source=check_file(config["benchmark_index"], config["benchmark_index_sha256"])
        with source.open("rb") as stream:
            self.benchmark=ExclusionUnpickler(stream).load()
        self.heldout=ExclusionIndex(); self.generated=ExclusionIndex();self.exact=set(); self.rejected=Counter()
        for phase in ("development","confirmation"):
            for row in raw_panels(config,phase):
                self.protect(tokenizer.decode(row["tokens"],skip_special_tokens=True))
        corpus_path=check_file(config["corpora"]["continuation"]["manifest"],config["corpora"]["continuation"]["sha256"])
        corpus=read(corpus_path)
        code=ROOT/"src/scglm/prepare_extension.py"
        if sha256(code)!=corpus["preparation_code_sha256"]:
            raise ValueError("Pretraining index implementation changed")
        prep=corpus_path.parent/"preparation_config.json"
        if sha256(prep)!=corpus["configuration_sha256"]:
            raise ValueError("Pretraining index configuration changed")
        import hashlib
        contract=hashlib.sha256((json.dumps(read(prep),sort_keys=True)+corpus["preparation_code_sha256"]+corpus["parent_manifest_sha256"]).encode()).hexdigest()
        self.env=lmdb.open(str(corpus_path.parent/"dedup.lmdb"),readonly=True,create=False,lock=True)
        self.txn=self.env.begin()
        if self.txn.get(b"!contract")!=contract.encode():
            self.close(); raise ValueError("Pretraining index contract mismatch")
        progress=json.loads(self.txn.get(b"!progress"))
        if not progress["old_done"] or any(not x["complete"] for x in progress["sources"].values()):
            self.close(); raise ValueError("Pretraining index incomplete")

    def protect(self,text,*,generated=False):
        (self.generated if generated else self.heldout).add(text); self.exact.add(identity(text))

    def blocked(self,text,*,training=False,pretraining=False,generated=False):
        from scglm.prepare_extension import matching_reason,sketch
        reason=None
        if self.benchmark.matches(text): reason="benchmark"
        elif training and (identity(text) in self.exact or self.heldout.matches(text)
                           or (not generated and self.generated.matches(text))): reason="heldout"
        elif pretraining and len(normalized(text).split())>=5:
            reason=matching_reason(self.txn,{"digest":identity(text),"sketch":sketch(text),"url":""})
        if reason: self.rejected[reason]+=1
        return reason

    def close(self):
        self.txn.abort(); self.env.close()


def load(path):
    path=resolve(path); manifest=read(path)
    if manifest.get("status")!="complete" or manifest.get("schema")!=VERSION:
        raise ValueError("Incomplete or incompatible SOTA data")
    for entry in manifest["files"].values():
        if sha256(path.parent/entry["file"])!=entry["sha256"]:
            raise ValueError("Prepared SOTA data changed")
    return manifest


def confirmation_identity(tasks,legacy,documents):
    # Independent of filenames, grouping metadata, task IDs and source aliases.
    items={digest(["task",prompt_text(r),r["answer"]]) for r in tasks}
    items.update(digest(["task",render_prompt(r["messages"][:-1]),r["messages"][-1]["content"]]) for r in legacy)
    items.update(digest(["raw",r["tokens"]]) for r in documents)
    ordered=sorted(items)
    return {"confirmation_content_sha256":digest(ordered),"confirmation_item_sha256":ordered}


def prepare(config, destination):
    from transformers import AutoTokenizer
    destination=resolve(destination)
    implementation={p.name:sha256(p) for p in (Path(__file__),Path(__file__).with_name("task_registry.py"),ROOT/"src/scglm_post/verifiers.py")}
    if destination.exists():
        manifest=load(destination/"manifest.json")
        if manifest["config_sha256"]!=digest(config) or manifest.get("implementation")!=implementation: raise ValueError("Data registration changed")
        return manifest
    destination.parent.mkdir(parents=True,exist_ok=True)
    tokenizer=AutoTokenizer.from_pretrained(str(resolve(config["parent"])),local_files_only=True)
    old_path=check_file(config["legacy_manifest"],config["legacy_manifest_sha256"])
    legacy=old_manifest(old_path); tokenizer_identity(resolve(config["parent"]),legacy)
    with tempfile.TemporaryDirectory(prefix=".sota-data-",dir=destination.parent) as directory:
        out=Path(directory); rows=[]; sources={}; legacy_files={}
        for split in ("train","development","confirmation"):
            old=read_jsonl(old_path.parent/legacy["splits"][split]["file"])
            path=out/f"legacy_{split}.jsonl"; write_jsonl(path,old); legacy_files[split]=path.name
            for r in old:
                procedural=r["source"]=="procedural"
                bucket="grounded" if r.get("category") in ("closed_qa","information_extraction","summarization") else "general"
                row=task(source=r["source"],group=r["group"],split=split,family="legacy/"+r["category"],
                         prompt=r["messages"][-2]["content"],answer=r["messages"][-1]["content"],
                         verifier="legacy" if procedural else "none", messages=r["messages"][:-1],
                         license=r["license"],bucket="procedural" if procedural else bucket,
                         legacy=r, provenance=r.get("provenance",{}),control=True,balanced=not procedural)
                rows.append(row)
        for spec in config["public_sources"]:
            new, record=public_tasks(spec,out/"licenses",config["seed"])
            rows+=new; sources[spec["repository"]]=record
        for split in ("train","development","confirmation"):
            rows+=generated_tasks(split,config["generated_per_family"][split],config["seed"])
        from scglm.prepare_data import ExclusionIndex
        exposed=ExclusionIndex(); exposed_exact=set()
        for r in rows:
            if r.get("control") and r["split"]=="train":
                exposed.add(r["prompt"]);exposed_exact.add(identity(r["prompt"]))
        protection=Protection(config,tokenizer)
        try:
            # Old protected panels remain protected even if an item is too long for a new trainer.
            for r in rows:
                if r["split"]!="train": protection.protect(prompt_text(r),generated=r["source"]=="program")
            rejected=set()
            for r in rows:
                texts=[r["prompt"],*r.get("split_facts",[])]
                if any(protection.blocked(t,pretraining=r["split"]!="train") for t in texts):
                    rejected.add(r["group_id"])
                if r["split"]!="train" and not r.get("control") and any(identity(t) in exposed_exact or exposed.matches(t) for t in texts):
                    rejected.add(r["group_id"]);protection.rejected["previous_posttraining"]+=1
            rows=[r for r in rows if r["group_id"] not in rejected]
            # Protect every new holdout before filtering any train group.
            for r in rows:
                if r["split"]!="train":
                    for text in [r["prompt"],*r.get("split_facts",[])]: protection.protect(text,generated=r["source"]=="program")
            for r in rows:
                if r["split"]=="train" and any(protection.blocked(t,training=True,generated=r["source"]=="program") for t in
                    [r["prompt"],*r.get("split_facts",[])]): rejected.add(r["group_id"])
            splits={s:[] for s in ("train","development","confirmation")}; filtered=Counter()
            for r in rows:
                if r["group_id"] in rejected: continue
                try:
                    encoded=encode_response(r,tokenizer,max_length=config["max_length"],max_response=config["max_response_tokens"])
                except ValueError:
                    filtered["length_or_boundary"]+=1; continue
                # The rollout engine must be able to use every admitted verified prompt.
                if r["verifier"]!="none" and encoded["prompt_length"]+config["max_response_tokens"]>config["max_length"]:
                    filtered["rollout_context"]+=1; continue
                splits[r["split"]].append({**r,**encoded})
            # Drop exact repeated examples, but reject cross-split identities in the audit.
            for s in splits: splits[s]=list({r["id"]:r for r in splits[s]}.values())
            counts=audit_tasks(splits)
            for split in ("development","confirmation"):
                for domain in PRIMARY_DOMAINS:
                    panel=[r for r in splits[split] if r.get("domain")==domain]
                    if len(panel)<config["min_panel_items"] or len({r["group_id"] for r in panel})<config["min_panel_groups"]:
                        raise ValueError(f"Insufficient protected {split}/{domain}: {len(panel)} items, {len({r['group_id'] for r in panel})} groups")
            files={}
            for split, data in splits.items():
                path=out/f"{split}.jsonl"; write_jsonl(path,data)
                files[split]={"file":path.name,"sha256":sha256(path),"examples":len(data)}
            for split,name in legacy_files.items():
                files["legacy_"+split]={"file":name,"sha256":sha256(out/name)}
            replay=[]; corpus=read(resolve(config["corpora"]["continuation"]["manifest"]))
            for source in ("edu","dclm","wiki"):
                spec=corpus["sources"][source]["splits"]["train"]
                index=check_file(spec["index_path"],spec["index_sha256"])
                binary=check_file(spec["path"],spec["sha256"])
                mmap=np.memmap(binary,mode="r",dtype="<u2"); used=0
                with index.open() as stream:
                    for line in stream:
                        entry=json.loads(line); start=entry["offset"]; end=start+entry["length"]
                        tokens=mmap[start:end].astype(int).tolist()
                        text=tokenizer.decode(tokens,skip_special_tokens=True)
                        if protection.blocked(text,training=True): continue
                        for offset in range(0,len(tokens)-1,config["max_length"]-1):
                            values=tokens[offset:offset+config["max_length"]]
                            if len(values)<2:continue
                            replay.append({"id":entry["id"]+f":{offset}","source":source,**raw_row(values)})
                            used+=len(values)-1
                            if used>=config["replay_targets_per_source"]: break
                        if used>=config["replay_targets_per_source"]: break
                del mmap
                if used<config["replay_targets_per_source"]: raise ValueError("Insufficient protected replay")
            path=out/"replay.jsonl";write_jsonl(path,replay)
            files["replay"]={"file":path.name,"sha256":sha256(path),"examples":len(replay)}
            report={"counts":counts,"rejected_groups":len(rejected),"filters":dict(filtered),
                    "overlap_rejections":dict(protection.rejected),"semantic_decontamination_certified":False,
                    "group_and_prompt_disjoint":True,"all_admitted_examples_encoded":True,
                    "program_overlap_policy":"Generated worlds require disjoint IDs, exact prompts, wrappers, and declared composition/operand partitions. Shared generator boilerplate n-grams are allowed only between program-generated worlds, not natural-source data, raw replay, prior human data, or benchmarks.",
                    "family_counts":{s:dict(Counter(r["family"] for r in v)) for s,v in splits.items()},
                    "bucket_target_tokens":dict(Counter({b:sum(r['target_tokens'] for r in splits['train'] if r['bucket']==b and r.get('balanced',True)) for b in ('general','knowledge','grounded','procedural')}))}
            write_json(out/"data_audit.json",report)
            files["audit"]={"file":"data_audit.json","sha256":sha256(out/"data_audit.json")}
            for notice in (out/"licenses").glob("*"):
                files["license/"+notice.name]={"file":str(notice.relative_to(out)),"sha256":sha256(notice)}
            manifest={"schema":VERSION,"status":"complete","config_sha256":digest(config),"config":config,
                      "implementation":implementation,
                      "tokenizer_sha256":sha256(resolve(config["parent"])/"tokenizer.json"),
                      "legacy_manifest_sha256":sha256(old_path),"sources":sources,"files":files,"audit":report,
                      **confirmation_identity(splits["confirmation"],read_jsonl(out/legacy_files["confirmation"]),raw_panels(config,"confirmation"))}
            write_json(out/"manifest.json",manifest)
            out.rename(destination)
        finally: protection.close()
    return manifest
