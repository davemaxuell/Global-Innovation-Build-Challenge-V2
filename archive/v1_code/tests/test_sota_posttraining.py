"""Executable loss oracles, isolation checks, and crash/restart coverage."""
import copy
import os
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from test_posttraining import tokenizer,tiny_model
from scglm.model import save_model,load_model
from scglm_post.common import encode_example
from scglm_post.verifiers import exact_integer_object
from scglm_post.system_data import generate_system
from scglm_post.sft import response_loss_sum
from scglm_pipeline.common import read,write_json,write_jsonl,read_jsonl,sha256,digest
from scglm_pipeline.task_registry import task,encode_response,prompt_text,raw_row,verify,audit_tasks,VERSION
from scglm_pipeline.generation import sample,weights_digest
from scglm_pipeline.sota_train import Trainer,sft_update,balanced_plan
from scglm_pipeline.sota_data import group_splits,generated_tasks
from scglm_pipeline.sota_evaluate import raw_choices,suffix_scores,evaluate
from scglm_pipeline.selection_v2 import grouped_gain,retention,claim_confirmation,verify_claim,immutable,replicated


@pytest.mark.parametrize("text",['[]','42','null','{"number":true}','{"number":3.0}',
                                  '{"number":3,"number":3}','{"number":NaN}','{"number":3,"extra":0}'])
def test_json_reward_rejects_wrong_types_duplicates_and_extra_keys(text):
    assert not exact_integer_object(text,{"number":3})




@pytest.mark.parametrize("style",["chat","raw"])
def test_response_boundary_eos_and_literal_continuation(tokenizer,style):
    row=task(source="fixture",group="1",split="train",family="science",prompt="Question?",answer="42",prompt_style=style)
    encoded=encode_response(row,tokenizer)
    boundary=encoded["prompt_length"]
    assert tokenizer.decode(encoded["input_ids"][:boundary])==prompt_text(row)
    assert encoded["labels"][:boundary]==[-100]*boundary
    assert encoded["input_ids"][-1]==tokenizer.eos_token_id
    assert encode_response(row,tokenizer,eos=False)["input_ids"]==encoded["input_ids"][:-1]
    if style=="raw":assert "Assistant:" not in prompt_text(row)














def test_replay_gradient_matches_weighted_full_batch_oracle(tokenizer):
    a=tiny_model(tokenizer);b=copy.deepcopy(a)
    rows=[encode_response(task(source="f",group=str(i),split="train",family="a",prompt="Compute?",answer=v),tokenizer)
          for i,v in enumerate(("3","12345"))]
    replay=[raw_row(tokenizer.encode("a longer raw text for replay",add_special_tokens=False))]
    opt=torch.optim.SGD(a.parameters(),lr=.01)
    sft_update(a,opt,rows,replay,{"replay_lambda":.3,"microbatch_examples":1,"gradient_clip":1000})
    other=torch.optim.SGD(b.parameters(),lr=.01);other.zero_grad()
    x,n=response_loss_sum(b,rows);y,m=response_loss_sum(b,replay)
    (.7*x/n+.3*y/m).backward();other.step()
    for x,y in zip(a.parameters(),b.parameters()):torch.testing.assert_close(x,y,atol=2e-8,rtol=1e-5)




def test_group_partition_discards_secondary_fact_bridges():
    rows=[{"group_id":str(i),"split_facts":["shared" if i%2==0 else str(i)]} for i in range(100)]
    divided,report=group_splits(rows,42)
    for fact in set(f for r in divided for f in r["split_facts"]):
        assert len({r["split"] for r in divided if fact in r["split_facts"]})==1
    assert report["removed_rows"]>0


def test_generated_worlds_are_disjoint_and_reference_answers_verify():
    rows={s:generated_tasks(s,12,42) for s in ("train","development","confirmation")}
    audit_tasks(rows)
    for values in rows.values():assert all(verify(r,r["answer"]) for r in values)
    rows["development"][0]["group_id"]=rows["train"][0]["group_id"]
    from scglm_pipeline.task_registry import task_identity
    rows["development"][0]["id"]=task_identity(rows["development"][0])
    with pytest.raises(ValueError,match="crossed"):audit_tasks(rows)












@pytest.fixture
def sota_fixture(tmp_path,tokenizer):
    torch.set_num_threads(1)
    model=tiny_model(tokenizer);model.config.max_position_embeddings=1024
    base=tmp_path/"base/export";save_model(model,base);tokenizer.save_pretrained(base)
    write_json(base/"training_provenance.json",{"cumulative_pretraining_tokens":100,"disposable_smoke_test":True})
    write_json(base.parent/"status.json",{"status":"completed"})
    c=read("configs/current_model.json")
    c.update(parent=str(base),parent_sha256=sha256(base/"model.safetensors"),expected_parameters=sum(p.numel() for p in model.parameters()),
             expected_pretraining_tokens=100,disposable_smoke_test=True,output=str(tmp_path/"work"),data_output=str(tmp_path/"data"),
             device="cpu",cpu_threads=1,seeds=[1,2,3],seed=1,deployment_seed=1)
    c["selection"]["bootstrap_replicates"]=20
    data=tmp_path/"data";data.mkdir();files={}
    for split in ("train","development","confirmation"):
        rows=[]
        for i in range(20):
            domain=("science","commonsense","grounded")[i%3]
            row=task(source="fixture",group=f"{split}/{i}",split=split,family=domain,
                     prompt=f"{split} case {i}: answer three?",answer="3",verifier="integer",choices=["3","9"],
                     domain=domain,bucket=("general","knowledge","grounded","procedural")[i%4],control=i%2==0)
            rows.append({**row,**encode_response(row,tokenizer)})
        for row in generated_tasks(split,1,42):
            if row["family"]=="coreference":rows.append({**row,**encode_response(row,tokenizer)})
        write_jsonl(data/(split+".jsonl"),rows)
        files[split]={"file":split+".jsonl","sha256":sha256(data/(split+".jsonl"))}
        old=generate_system(split,6)
        old=[{**r,**encode_example(r["messages"],tokenizer,max_length=1024)} for r in old]
        write_jsonl(data/("legacy_"+split+".jsonl"),old)
        files["legacy_"+split]={"file":"legacy_"+split+".jsonl","sha256":sha256(data/("legacy_"+split+".jsonl"))}
    replay=[{"source":s,"id":s,**raw_row([12,13,14,15,16])} for s in ("edu","dclm","wiki")]
    write_jsonl(data/"replay.jsonl",replay);files["replay"]={"file":"replay.jsonl","sha256":sha256(data/"replay.jsonl")}
    manifest={"schema":VERSION,"status":"complete","files":files,"tokenizer_sha256":sha256(base/"tokenizer.json"),
              "confirmation_content_sha256":"fixture-panel"}
    write_json(data/"manifest.json",manifest)
    recipe={"seed":1,"learning_rate":1e-4,"replay_lambda":.3,"max_targets":70,"epochs":2,"target_tokens_per_update":8,
            "microbatch_examples":2,"warmup_fraction":.03,"weight_decay":0.,"gradient_clip":1.,"checkpoint_every":1,
            "max_seconds":100,"endpoint_fractions":[.5,1.],"max_format_share":1.}
    return c,recipe,data/"manifest.json",base


def test_step_zero_checkpoint_and_exact_replay_resume(sota_fixture,tmp_path):
    c,r,manifest,base=sota_fixture
    full=Trainer(c,r,parent=base,manifest=manifest,output=tmp_path/"full");full.run()
    part=Trainer(c,r,parent=base,manifest=manifest,output=tmp_path/"resumed")
    assert read(tmp_path/"resumed/checkpoint_latest.json")["step"]==0
    part.run(max_updates=1)
    resumed=Trainer(c,r,parent=base,manifest=manifest,output=tmp_path/"resumed");resumed.run()
    assert full.replay_cursors==resumed.replay_cursors and full.counters==resumed.counters
    assert weights_digest(full.model)==weights_digest(resumed.model)
    assert full.updates>1


def test_initialization_failure_recovers_and_changed_recipe_rejects(sota_fixture,tmp_path):
    c,r,manifest,base=sota_fixture;path=tmp_path/"train"
    trainer=Trainer(c,r,parent=base,manifest=manifest,output=path);trainer.lock.close()
    (path/"checkpoint_latest.json").unlink()
    restarted=Trainer(c,r,parent=base,manifest=manifest,output=path);restarted.run(max_updates=0)
    assert list((path/"initialization_attempts").glob("*/run.json"))
    with pytest.raises(ValueError,match="contract"):
        Trainer(c,{**r,"learning_rate":1e-2},parent=base,manifest=manifest,output=path)


def test_checkpoint_and_manifest_tampering_fail_closed(sota_fixture,tmp_path):
    c,r,manifest,base=sota_fixture;path=tmp_path/"train"
    trainer=Trainer(c,r,parent=base,manifest=manifest,output=path);trainer.run(max_updates=1)
    pointer=read(path/"checkpoint_latest.json");(path/pointer["file"]).write_bytes(b"tampered")
    with pytest.raises(ValueError,match="Checkpoint identity"):
        Trainer(c,r,parent=base,manifest=manifest,output=path)
    (manifest.parent/"train.jsonl").write_text("[]\n")
    with pytest.raises(ValueError,match="data changed"):
        Trainer(c,r,parent=base,manifest=manifest,output=tmp_path/"other")


def test_raw_choice_scores_equal_explicit_likelihood_and_suffix_masks(tokenizer):
    model=tiny_model(tokenizer).eval();row=task(source="f",group="1",split="train",family="science",
                    prompt="Which?",answer="three",choices=["three","a long choice"])
    result=raw_choices(model,tokenizer,row)
    assert result["character_lengths"]==[5,13]
    with torch.no_grad():
        expected=[-response_loss_sum(model,[encode_response({**row,"prompt_style":"raw"},tokenizer,c,eos=False)])[0].item() for c in row["choices"]]
    assert result["scores"]==pytest.approx(expected,rel=1e-6)
    probe={"prefix":"Person ","candidates":["A","B"],"suffix":", went home.","answer_index":0}
    assert len(suffix_scores(model,tokenizer,probe)["scores"])==2


def test_grouped_interval_pairs_by_id_and_keeps_contexts_together():
    base=[{"id":f"{d}{i}","domain":d,"group":str(i//2),"correct":False} for d in ("science","commonsense","grounded") for i in range(6)]
    candidate=[{**r,"correct":True} for r in reversed(base)]
    result=grouped_gain(base,candidate,seed=1,replicates=100)
    assert result["gain"]==1 and result["lower_95"]==1
    candidate[0]["group"]="changed"
    with pytest.raises(ValueError,match="grouping"):grouped_gain(base,candidate,seed=1,replicates=10)


def test_confirmation_is_once_per_content_and_exact_candidate(tmp_path):
    config={"parent_sha256":"base"};manifest={"confirmation_content_sha256":"content"}
    frozen={"candidate_sha256":"one"};path=claim_confirmation(frozen,manifest,config,tmp_path)
    assert claim_confirmation(frozen,manifest,config,tmp_path)==path
    verify_claim(path,"one",manifest,config)
    with pytest.raises(ValueError,match="Frozen"):
        claim_confirmation({"candidate_sha256":"two"},manifest,{**config,"other_filename":True},tmp_path)
    with pytest.raises(ValueError):verify_claim(path,"two",manifest,config)


def test_confirmation_rejects_partial_reuse_and_ignores_item_ids(tmp_path):
    from scglm_pipeline.sota_data import confirmation_identity
    row=task(source="f",group="a",split="confirmation",family="science",prompt="A?",answer="3")
    first=confirmation_identity([row],[],[])
    renamed=confirmation_identity([{**row,"id":"renamed","group_id":"new-group"}],[],[])
    assert first==renamed
    config={"parent_sha256":"base"};claim_confirmation({"candidate_sha256":"one"},first,config,tmp_path)
    other=task(source="f",group="b",split="confirmation",family="science",prompt="B?",answer="4")
    second=confirmation_identity([row,other],[],[])
    with pytest.raises(ValueError,match="previously consumed"):
        claim_confirmation({"candidate_sha256":"two"},second,config,tmp_path)


def test_evaluation_cache_and_unclaimed_confirmation(sota_fixture,tmp_path,monkeypatch):
    c,r,manifest,base=sota_fixture
    c["evaluation"].update(max_new_tokens=2,batch_size=16,legacy_per_family=6,max_raw_tokens_per_document=8)
    c["corpora"]={"test":{"weights":{"raw":1.}}}
    monkeypatch.setattr("scglm_pipeline.sota_evaluate.raw_panels",lambda *a:[{"id":"x","source":"test/raw","tokens":[12,13,14,15]}])
    with pytest.raises(ValueError,match="one-use claim"):evaluate(base,manifest,c,"confirmation",tmp_path/"no")
    result=evaluate(base,manifest,c,"development",tmp_path/"eval")
    assert set(result["metrics"]["domains"])=={"science","commonsense","grounded"}
    assert result["metrics"]["system"]["policy_pairs"]["examples"]>0
    assert evaluate(base,manifest,c,"development",tmp_path/"eval")==result
    (tmp_path/"eval/primary.jsonl").write_text("changed")
    with pytest.raises(ValueError,match="evidence changed"):evaluate(base,manifest,c,"development",tmp_path/"eval")










def test_missing_retention_measures_fail_closed():
    config=read("configs/current_model.json")
    assert "missing_raw_sources" in retention({}, {}, config["selection"])
    assert "missing_system_modes" in retention({}, {}, config["selection"])




def test_immutable_decision_accepts_json_tuple_roundtrip(tmp_path):
    path=tmp_path/"record.json";value={"shortlist":[(1,2,3)]}
    immutable(path,value);immutable(path,value)
    with pytest.raises(ValueError):immutable(path,{"shortlist":[(1,2,4)]})




@pytest.mark.skipif(os.environ.get("SOTA_GPU_SMOKE")!="1",reason="Explicit allocated-GPU smoke only")
def test_allocated_gpu_checkpoint_resume_is_exact(sota_fixture,tmp_path):
    from scripts.run_pipeline import gpu_environment
    c,r,manifest,base=sota_fixture;gpu_environment(c,"cuda:0")
    full=Trainer(c,r,parent=base,manifest=manifest,output=tmp_path/"full",device="cuda:0");full.run()
    part=Trainer(c,r,parent=base,manifest=manifest,output=tmp_path/"part",device="cuda:0");part.run(max_updates=1)
    resumed=Trainer(c,r,parent=base,manifest=manifest,output=tmp_path/"part",device="cuda:0");resumed.run()
    assert weights_digest(full.model)==weights_digest(resumed.model)
    assert full.counters==resumed.counters and full.replay_cursors==resumed.replay_cursors
    from scglm_pipeline.common import ROOT
    path=ROOT/"reports/cleanup/2026-09-29/gpu_resume.json";path.parent.mkdir(parents=True,exist_ok=True)
    write_json(path,{"status":"passed","disposable":True,"gpu_uuid":os.environ["CUDA_VISIBLE_DEVICES"],
                     "identical_model_weights":True,"identical_replay_cursors":True,"identical_counters":True,
                     "updates":full.updates,"counters":dict(full.counters)})
