"""Mathematical oracles and failure-mode tests for the isolated choice study."""
import copy
import json
import os
from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from test_posttraining import tokenizer, tiny_model as make_tiny_model
from scglm.model import build_model
from scglm_post.sft import response_loss_sum, learning_rate
from scglm_pipeline.task_registry import raw_row
from scglm_choice.common import config, roots, digest, read, read_jsonl, write_json, write_jsonl, ARMS, SEEDS
from scglm_choice.objective import encode, logps, update, choice_loss, positions
from scglm_choice.train import Recovery, optimizer
from scglm_choice.budget import Budget, BudgetExceeded
from scglm_choice.data import partition, group_related
from scglm_choice.plans import select_choices
from scglm_choice.selection import (bootstrap, paired, pilot, replication, claim_confirmation,
                                    verify_claim, official_selection, retention)


@pytest.fixture
def tiny_model(tokenizer):
    return make_tiny_model(tokenizer)


def examples(tok):
    response = [encode(tok,"Question?\nAnswer:"," 42",eos=True), encode(tok,"Another?\nAnswer:"," 1",eos=True)]
    replay = [raw_row([1,4,5,6]), raw_row([1,7,8])]
    items = [{"domain":d, "answer_index":i%2,
              "encoded_choices":[encode(tok,"Question?\nAnswer:"," 42"), encode(tok,"Question?\nAnswer:"," 1")]}
             for i,d in enumerate(["science","science","commonsense"])]
    return response,replay,items


def test_continuation_boundary_matches_pinned_harness(tokenizer):
    import sys
    from scglm_choice.common import ROOT
    sys.path.insert(0,str(ROOT/"vendor/lm-evaluation-harness"))
    from lm_eval.api.model import TemplateLM
    adapter = SimpleNamespace(backend="causal",tok_encode=lambda x:tokenizer.encode(x,add_special_tokens=False))
    for prompt in ("Question?\nAnswer:","Question: Question?\nThe answer is", "Question?\nAnswer: "):
        c,t = TemplateLM._encode_pair(adapter,prompt," 42")
        row = encode(tokenizer,prompt," 42")
        assert row["input_ids"] == c+t
        assert row["labels"] == [-100]*len(c)+t
        assert tokenizer.eos_token_id not in row["input_ids"]
        assert encode(tokenizer,prompt," 42",eos=True)["input_ids"] == c+t+[tokenizer.eos_token_id]
    with pytest.raises(ValueError):
        encode(tokenizer,"Prompt:"," <|eos|>")
    with pytest.raises(ValueError):
        encode(tokenizer,"Prompt:"," 42",max_length=1)


def test_choice_logps_shift_padding_eos_oracle(tokenizer,tiny_model):
    response,_,items = examples(tokenizer)
    rows = items[0]["encoded_choices"]
    values,lengths = logps(tiny_model,rows)
    for i,row in enumerate(rows):
        ids = torch.tensor([row["input_ids"]])
        logits = tiny_model(input_ids=ids,use_cache=False).logits.float()
        expected = sum(logits[0,j-1].log_softmax(-1)[row["input_ids"][j]] for j in range(row["prompt_length"],len(row["input_ids"])))
        assert torch.allclose(values[i],expected,atol=1e-6)
        assert lengths[i] == row["target_tokens"]
    loss,count = response_loss_sum(tiny_model,response)
    scored,_ = logps(tiny_model,response)
    assert torch.allclose(loss,-scored.sum(),atol=1e-5)
    assert count == sum(r["target_tokens"] for r in response)


def test_listwise_value_gradients_order_and_domain_weight(tokenizer,tiny_model):
    _,_,items = examples(tokenizer)
    actual = choice_loss(tiny_model,items)
    terms = []
    for r in items:
        scores,_=logps(tiny_model,r["encoded_choices"])
        terms.append(-scores.log_softmax(0)[r["answer_index"]])
    expected = .5*((terms[0]+terms[1])/2+terms[2])
    assert torch.allclose(actual,expected,atol=1e-6)
    actual.backward()
    grads = [p.grad.clone() for p in tiny_model.parameters()]
    tiny_model.zero_grad()
    reversed_items = [{**r,"encoded_choices":list(reversed(r["encoded_choices"])),"answer_index":1-r["answer_index"]} for r in items]
    reverse = choice_loss(tiny_model,reversed_items)
    reverse.backward()
    assert torch.allclose(reverse,actual)
    for p,g in zip(tiny_model.parameters(),grads):
        torch.testing.assert_close(p.grad,g,atol=2e-6,rtol=2e-5)


@pytest.mark.parametrize("candidate",[False,True])
def test_accumulated_gradients_equal_full_objective(tokenizer,tiny_model,candidate):
    response,replay,items = examples(tokenizer)
    oracle = copy.deepcopy(tiny_model)
    ce,n = response_loss_sum(oracle,response)
    raw,rn = response_loss_sum(oracle,replay)
    expected = .7*ce/n+.3*raw/rn
    if candidate:
        expected = expected+.05*choice_loss(oracle,items)
    expected.backward()
    torch.nn.utils.clip_grad_norm_(oracle.parameters(),1.)
    update(tiny_model,optimizer(tiny_model),response,replay,items,candidate=candidate,microbatch=1,step=False)
    for a,b in zip(tiny_model.parameters(),oracle.parameters()):
        torch.testing.assert_close(a.grad,b.grad,atol=1e-6,rtol=3e-5)


def _recovery_test(directory,model,tok,device,charge=None):
    response,replay,items=examples(tok)
    template=copy.deepcopy(model).to(device)
    def train(state,steps):
        for _ in range(steps):
            rate=learning_rate((state.step+1)*10,40,{"learning_rate":3e-6,"warmup_fraction":.03})
            for g in state.optimizer.param_groups:g["lr"]=rate
            update(state.model,state.optimizer,response,replay,items,candidate=True,charge=charge)
            state.step+=1;state.counters["response_targets"]+=sum(r["target_tokens"] for r in response)
    torch.manual_seed(42);random.seed(42);np.random.seed(42)
    uninterrupted=Recovery(directory/"full",{"seed":42},copy.deepcopy(template),None)
    uninterrupted.optimizer=optimizer(uninterrupted.model)
    train(uninterrupted,4)
    expected_rng=(torch.rand(3),np.random.rand(3),random.random())
    torch.manual_seed(42);random.seed(42);np.random.seed(42)
    partial=Recovery(directory/"resumed",{"seed":42},copy.deepcopy(template),None)
    partial.optimizer=optimizer(partial.model)
    train(partial,2);partial.save()
    torch.rand(10);np.random.rand(10);random.random()
    model2=copy.deepcopy(template)
    resumed=Recovery(directory/"resumed",{"seed":42},model2,optimizer(model2))
    train(resumed,2)
    assert resumed.step==4 and resumed.counters==uninterrupted.counters
    for a,b in zip(uninterrupted.model.parameters(),resumed.model.parameters()):
        torch.testing.assert_close(a,b,atol=0,rtol=0)
    for a,b in zip(uninterrupted.optimizer.state.values(),resumed.optimizer.state.values()):
        for k in a: torch.testing.assert_close(a[k],b[k],atol=0,rtol=0)
    torch.testing.assert_close(torch.rand(3),expected_rng[0],atol=0,rtol=0)
    np.testing.assert_array_equal(np.random.rand(3),expected_rng[1]);assert random.random()==expected_rng[2]
    with pytest.raises(ValueError,match="Frozen record changed"):
        Recovery(directory/"resumed",{"seed":43},copy.deepcopy(template),None)


def test_exact_interrupted_resume(tokenizer,tiny_model,tmp_path):
    _recovery_test(tmp_path,tiny_model,tokenizer,"cpu")


@pytest.mark.skipif(os.environ.get("SCGLM_CHOICE_GPU_TEST") != "1",reason="Assigned-GPU test runs inside budgeted preflight only")
def test_cuda_recovery(tokenizer,tiny_model,tmp_path):
    torch.use_deterministic_algorithms(True)
    assert os.environ["CUDA_VISIBLE_DEVICES"] == config()["budget"]["assigned_gpu_uuids"][0]
    counts={"positions":0,"by_kind":{}}
    def charge(n,kind):
        counts["positions"]+=n;counts["by_kind"][kind]=counts["by_kind"].get(kind,0)+n
        if os.environ.get("SCGLM_CHOICE_TEST_COUNTS"):
            write_json(Path(os.environ["SCGLM_CHOICE_TEST_COUNTS"]),counts)
    _recovery_test(tmp_path,tiny_model,tokenizer,"cuda:0",charge)


def test_resource_rollback_and_crash_accounting(tmp_path):
    clock=[100.]
    b=Budget(tmp_path,{},clock=lambda:clock[0])
    with b.session("arm",timer=False):
        b.charge(6000000,"response");clock[0]+=20
    b=Budget(tmp_path,{},clock=lambda:clock[0])
    with b.session("arm",timer=False):
        with pytest.raises(BudgetExceeded): b.charge(4000001,"repeated_after_rollback")
    b.value["active"]={"category":"arm","started":clock[0],"reserved_seconds":b.remaining("arm")};b._save()
    clock[0]+=100
    restored=Budget(tmp_path,{},clock=lambda:clock[0])
    assert restored.value["categories"]["arm"]["seconds"]==120
    assert restored.value["categories"]["arm"]["positions"]==6000000
    assert restored.value["categories"]["arm"]["uncertain_seconds"]==100
    restored.value["categories"]["arm"]["seconds"]=3600
    with pytest.raises(BudgetExceeded):
        with restored.session("arm",timer=False):pass


def test_shared_and_overall_time_limits(tmp_path):
    clock=[0.];b=Budget(tmp_path,{},clock=lambda:clock[0])
    with b.session("shared",timer=False):
        clock[0]=7199
        assert b.remaining("shared")==1
        clock[0]=7200
        with pytest.raises(BudgetExceeded): b.charge(1,"evaluation")
    with pytest.raises(BudgetExceeded):
        with b.session("shared",timer=False):pass
    for i in range(6): b._entry(str(i))["seconds"]=3600
    with pytest.raises(BudgetExceeded):
        with b.session("another",timer=False):pass


def test_missing_budget_cannot_reset_existing_study(tmp_path):
    (tmp_path / "runs/control").mkdir(parents=True)
    write_json(tmp_path / "runs/control/run.json", {"contract": "already_started"})
    with pytest.raises(ValueError, match="Missing cumulative compute ledger"):
        Budget(tmp_path, {})


def test_group_quota_and_whole_group_split():
    items=[{"id":f"{d}:{i}","domain":d,"group_id":str(i//5)} for d in ("science","commonsense") for i in range(30)]
    a=select_choices(items,1,count=12,quota=4)
    assert a==select_choices(items,1,count=12,quota=4)
    for pool in a:
        assert len(pool)==12
        assert max(sum(r["group_id"]==g for r in pool) for g in {r["group_id"] for r in pool})<=4
    with pytest.raises(ValueError):select_choices(items,1,count=25,quota=4)
    rows=[{"id":str(i),"group_id":str(i//3)} for i in range(30)]
    panels=partition(rows,seed=4,min_items=7,min_groups=2)
    groups=[{r["group_id"] for r in panels[s]} for s in panels]
    assert not groups[0]&groups[1]
    assert all(len(p)==9 for p in panels.values())
    with pytest.raises(ValueError):partition(rows,seed=4,min_items=500)


def test_related_facts_union():
    fact="one two three four five six seven eight nine ten eleven twelve thirteen fourteen"
    rows=[{"id":str(i),"source":"science","group_id":str(i),"question":str(i),"facts":[t]}
          for i,t in enumerate([fact, fact+" more", "a separate paragraph"])]
    grouped=group_related(rows)
    assert grouped[0]["group_id"]==grouped[1]["group_id"]!=grouped[2]["group_id"]


def fake_metrics(score=.5):
    return {"primary":score,"domains":{d:{"accuracy":score,"items":500} for d in ("science","commonsense")},
            "instructions":{"math":{"accuracy":1.,"items":100}},
            "raw":{s:{"nll":2.,"token_count":131072,"documents":200} for s in config()["evaluation"]["raw_strata"]}}


def test_pilot_and_replication_thresholds_and_transfer_diagnostic_only():
    cfg=config();base=parent=control=fake_metrics();candidate=fake_metrics(.51)
    candidate["transfer_diagnostic_only"]={"bad":0.}
    assert pilot(candidate,control,parent,base,cfg)["passed"]
    assert not pilot(fake_metrics(.509),control,parent,base,cfg)["passed"]
    members=[{"seed":s,"candidate":candidate,"control":control} for s in SEEDS]
    assert replication(members,parent,base,cfg)["passed"]
    members[2]["candidate"]=fake_metrics(.499)
    assert not replication(members,parent,base,cfg)["passed"]
    with pytest.raises(ValueError):replication(members[:2],parent,base,cfg)
    bad=copy.deepcopy(candidate);bad["raw"][next(iter(bad["raw"]))]["nll"]+=.01001
    assert retention(bad,parent,base,cfg=cfg)


def test_bootstrap_paired_draws_and_membership_rejection():
    rows=[{"id":f"{d}:{i}","group":f"{d}:{i//2}","domain":d,"correct":float(i%2)} for d in ("science","commonsense") for i in range(10)]
    names,points,draws=bootstrap({"a":rows,"b":copy.deepcopy(rows)},"primary",replicates=100)
    assert np.array_equal(draws["primary"][0],draws["primary"][1])
    unequal=copy.deepcopy(rows);unequal[0]["group"]="changed"
    with pytest.raises(ValueError):bootstrap({"a":rows,"b":unequal},"primary",replicates=100)
    with pytest.raises(ValueError):bootstrap({"a":rows,"b":rows[:-1]},"primary",replicates=100)


def test_confirmation_claim_all_four_binding_and_content_reuse(tmp_path):
    data=tmp_path/"data";data.mkdir()
    row={"id":"a","question":"A?","choices":["x","y"],"answer_index":0}
    write_jsonl(data/"primary_confirmation.jsonl",[row]);write_jsonl(data/"raw_confirmation.jsonl",[{"tokens":[1,2,3]}]);write_jsonl(data/"transfer_confirmation.jsonl",[])
    manifest={"confirmation_membership_sha256":"panel"};models={n:n*8 for n in ("candidate","control","parent","base")}
    claim=claim_confirmation(manifest,models,tmp_path/"claims",data)
    assert claim_confirmation(manifest,models,tmp_path/"claims",data)==claim
    for h in models.values(): verify_claim(claim,manifest,h)
    with pytest.raises(ValueError): verify_claim(claim,manifest,"replacement")
    with pytest.raises(ValueError):claim_confirmation(manifest,{**models,"control":"changed"},tmp_path/"claims",data)
    write_jsonl(data/"primary_confirmation.jsonl",[row,{**row,"id":"b","question":"B?"}])
    with pytest.raises(ValueError,match="already claimed"):
        claim_confirmation(manifest,models,tmp_path/"claims",data)


def test_official_selection_schema():
    models={f"{a}_{s}":Path(f"/tmp/{a}_{s}/step_0000032/export") for s in SEEDS for a in ARMS}
    hashes={n:digest(n) for n in models};key=next(iter(models));selection=official_selection(models,hashes,key)
    assert selection["panel"]=="selection" and selection["official_scores_used"] is False
    for name,path in models.items():
        matches=[r for r in selection["results"] if Path(r["run"])/"export"==path]
        assert len(matches)==1 and matches[0]["export_model_sha256"]==hashes[name]


def test_output_isolation_and_default_read_only():
    from scglm_choice.common import within
    from scglm_choice.runner import verify
    cfg=config();data,out=roots(cfg)
    with pytest.raises(ValueError):within(out/"../outside",out)
    before={p.name:p.stat().st_mtime_ns for p in out.iterdir()} if out.exists() else {}
    assert verify(cfg)["read_only"] is True
    after={p.name:p.stat().st_mtime_ns for p in out.iterdir()} if out.exists() else {}
    assert before==after


def test_frozen_real_plans_and_memberships():
    from scglm_choice.prepare import load,audit_isolation
    from scglm_choice.plans import make_plan
    from transformers import AutoTokenizer
    from scglm_choice.common import ROOT
    cfg=config();data,_=roots(cfg)
    if not (data/"manifest.json").exists():pytest.skip("No frozen local study data")
    manifest=load(cfg);audit_isolation(data)
    choices={r["id"]:r for r in read_jsonl(data/"choices.jsonl")}
    responses={r["id"]:r for r in read_jsonl(data/"responses.jsonl")}
    original=read_jsonl(ROOT/"data/posttraining/sota_v1/train.jsonl")
    replay=read_jsonl(data/"replay.jsonl")
    tok=AutoTokenizer.from_pretrained(str(ROOT/cfg["parent"]["path"]),local_files_only=True)
    for seed in SEEDS:
        plan=read(data/f"plan_{seed}.json")
        regenerated,_=make_plan(original,replay,list(choices.values()),seed,cfg)
        assert digest(regenerated)==digest(plan)
        assert len(plan["batches"])==32 and plan["schedule_denominator"]==plan["response_targets"]
        seen=set();groups={d:{} for d in ("science","commonsense")}
        for batch in plan["batches"]:
            assert len(batch["choice_ids"])==64
            assert sum(choices[i]["domain"]=="science" for i in batch["choice_ids"])==32
            assert all(choices[i]["anchor"]["id"] in batch["response_ids"] for i in batch["choice_ids"])
            assert sum(responses[i]["target_tokens"] for i in batch["response_ids"])==batch["response_targets"]
            assert not seen.intersection(batch["choice_ids"]);seen.update(batch["choice_ids"])
            for i in batch["choice_ids"]:
                r=choices[i];counts=groups[r["domain"]];counts[r["group_id"]]=counts.get(r["group_id"],0)+1
        assert len(seen)==2048 and all(max(c.values())<=4 for c in groups.values())
        assert manifest["planned_positions"][str(seed)]["candidate"]<=10000000
    for split in ("development","confirmation"):
        rows=read_jsonl(data/f"primary_{split}.jsonl")
        for domain in ("science","commonsense"):
            panel=[r for r in rows if r["domain"]==domain]
            assert len(panel)>=500 and len({r["group_id"] for r in panel})>=50
            assert not any(r["preview"] for r in panel)
        rows=read_jsonl(data/f"raw_{split}.jsonl")
        for source in cfg["evaluation"]["raw_strata"]:
            panel=[r for r in rows if r["source"]==source]
            assert sum(len(r["tokens"])-1 for r in panel)>=131072
            assert len(panel)>=50 and max(len(r["tokens"])-1 for r in panel)<=1024


def test_confirmation_uses_adjusted_bounds_and_raw_upper_bounds(monkeypatch):
    from scglm_choice.selection import confirmation
    from scglm_choice import evaluate as evaluation
    cfg=config();results={}
    for model in ("candidate","control","parent","base"):
        score=.55 if model=="candidate" else .5
        primary=[{"id":f"{d}:{i}","group":f"{d}:{i}","domain":d,"correct":score}
                 for d in ("science","commonsense") for i in range(20)]
        raw=[{"id":f"{s}:{i}","group":f"{s}:{i}","source":s,"token_count":100,
              "nll_sum":200.+(.5 if model=="candidate" else 0)} for s in cfg["evaluation"]["raw_strata"] for i in range(20)]
        records={"primary":primary,"raw":raw,"instructions":[{"id":"old","group":"old","family":"math","correct":1.}]}
        results[model]={"contract":{"split":"confirmation","claim_sha256":"same"},"metrics":fake_metrics(score),"records":records}
    monkeypatch.setattr(evaluation,"evidence",lambda result,kind:result["records"][kind])
    assert confirmation(results,cfg)["passed"]
    # Same acceptable mean; variation makes the one-sided upper bound fail.
    for i,r in enumerate(results["candidate"]["records"]["raw"]):
        r["nll_sum"] += 20 if i%2 else -20
    failed=confirmation(results,cfg)
    assert not failed["passed"]
    assert any(f.startswith("confirmation_raw_retention") for f in failed["failures"])
