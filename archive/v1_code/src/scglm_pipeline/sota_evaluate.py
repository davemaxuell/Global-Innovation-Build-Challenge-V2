"""Protected development/confirmation measurements; no official benchmark inputs."""
from collections import defaultdict
import gc
import torch
from transformers import AutoTokenizer
from scglm.model import load_model
from scglm.evaluate import evaluate_documents
from scglm_post.common import render_prompt
from scglm_post.evaluate import evaluate_records, generation_sample, system_metrics
from scglm_post.procedural import correct as legacy_correct
from .common import read,read_jsonl,write_json,write_jsonl,sha256,digest,resolve,raw_panels
from .posttraining_lineage import identity
from .sota_data import load
from .task_registry import encode_response,prompt_text,verify,PRIMARY_DOMAINS
from .evaluate import response_logps,raw_metrics,generate_many
from .generation import sample
from .battery import repetition


@torch.inference_mode()
def raw_choices(model,tokenizer,row):
    raw={**row,"prompt_style":"raw"}
    encoded=[encode_response(raw,tokenizer,c,eos=False,max_length=model.config.max_position_embeddings,
                             max_response=model.config.max_position_embeddings) for c in row["choices"]]
    scores,lengths=response_logps(model,encoded)
    # Exactly the literal continuation; no chat marker, answer letter, or EOS.
    characters=torch.tensor([len(c) for c in row["choices"]],device=scores.device)
    if not characters.gt(0).all():raise ValueError("Empty choice")
    correct=row["choices"].index(row["answer"])
    return {"scores":scores.tolist(),"target_lengths":lengths.tolist(),"character_lengths":characters.tolist(),
            "predicted":int(scores.argmax()),"correct":int(scores.argmax())==correct,
            "predicted_character_normalized":int((scores/characters).argmax()),
            "correct_character_normalized":int((scores/characters).argmax())==correct,
            "format":"literal_raw_continuation_no_eos"}


@torch.inference_mode()
def suffix_scores(model,tokenizer,probe):
    rows=[]
    for candidate in probe["candidates"]:
        prefix=probe["prefix"]+candidate;full=prefix+probe["suffix"]
        context=tokenizer.encode(prefix,add_special_tokens=False);ids=tokenizer.encode(full,add_special_tokens=False)
        if ids[:len(context)]!=context or len(ids)<=len(context):raise ValueError("Suffix boundary changed")
        rows.append({"input_ids":ids,"labels":[-100]*len(context)+ids[len(context):],
                     "target_tokens":len(ids)-len(context),"processed_tokens":len(ids),"prompt_length":len(context)})
    scores,_=response_logps(model,rows)
    return {"scores":scores.tolist(),"correct":int(scores.argmax())==probe["answer_index"],
            "format":"candidate_in_context_score_suffix_only"}


def family_metrics(rows):
    groups=defaultdict(list)
    for r in rows:groups[r["family"]].append(r)
    return {f:{"accuracy":sum(r["correct"] for r in values)/len(values),"examples":len(values)}
            for f,values in groups.items()}


def evaluate(model_path,manifest_path,config,phase,output,device="cpu",*,confirmation_claim=None):
    if phase not in ("development","confirmation"):raise ValueError("Invalid evaluation phase")
    model_path=resolve(model_path);manifest_path=resolve(manifest_path);output=resolve(output)
    model_identity=identity(model_path,config);data=load(manifest_path)
    if phase=="confirmation":
        from .selection_v2 import verify_claim
        if confirmation_claim is None:raise ValueError("Confirmation requires a frozen one-use claim")
        verify_claim(confirmation_claim,model_identity["export_model_sha256"],data,config)
    from .sota_train import source_contract
    contract={"model":model_identity,"manifest_sha256":sha256(manifest_path),"config_sha256":digest(config),
              "phase":phase,"source_sha256":source_contract(),"device":device,"precision":"float32"}
    key=digest(contract);done=output/"result.json"
    if done.exists():
        result=read(done)
        if result["contract_sha256"]!=key:raise ValueError("Evaluation cache contract changed")
        for name,checksum in result["files"].items():
            if sha256(output/name)!=checksum:raise ValueError("Evaluation evidence changed")
        return result
    output.mkdir(parents=True,exist_ok=True);write_json(output/"registration.json",contract)
    rows=read_jsonl(manifest_path.parent/data["files"][phase]["file"])
    if any(r["split"]!=phase for r in rows):raise ValueError("Evaluation split changed")
    legacy=read_jsonl(manifest_path.parent/data["files"]["legacy_"+phase]["file"])
    documents=raw_panels(config,phase);settings=config["evaluation"]
    model=load_model(model_path).to(device).eval();tokenizer=AutoTokenizer.from_pretrained(str(model_path),local_files_only=True)
    try:
        primary=[];choices=[];suffix=[]
        for row in rows:
            if row.get("choices"):
                score=raw_choices(model,tokenizer,row)
                prediction={"id":row["id"],"group":row["group_id"],"family":row["family"],"domain":row.get("domain"),**score}
                choices.append(prediction)
                if row.get("domain") in PRIMARY_DOMAINS:primary.append(prediction)
            if row.get("suffix_probe"):
                suffix.append({"id":row["id"],"group":row["group_id"],**suffix_scores(model,tokenizer,row["suffix_probe"])})
        domains={d:[r for r in primary if r["domain"]==d] for d in PRIMARY_DOMAINS}
        if any(not values for values in domains.values()):raise ValueError("Missing primary domain")
        domain_metrics={d:{"accuracy":sum(r["correct"] for r in v)/len(v),
                           "character_normalized_accuracy":sum(r["correct_character_normalized"] for r in v)/len(v),
                           "examples":len(v),"groups":len({r["group"] for r in v})} for d,v in domains.items()}
        verified=[r for r in rows if r["verifier"]!="none" and not r.get("control")]
        decoded,stats=sample(model,tokenizer,verified,{"samples_per_prompt":1,"do_sample":False,
                           "max_new_tokens":settings["max_new_tokens"],"batch_size":settings["batch_size"]},
                           seed=config["seed"],policy_id=model_identity["export_model_sha256"])
        generated=[{"id":r["id"],"group":r["group_id"],"family":r["family"],
                    "correct":bool(a["reward"]),"response":a["response"],"ended_eos":a["ended_eos"]}
                   for r,a in zip(verified,decoded)]
        legacy_groups=defaultdict(list)
        for row in legacy:
            if row["source"]=="procedural":legacy_groups[row["category"]].append(row)
        chosen=[r for f in sorted(legacy_groups) for r in generation_sample(legacy_groups[f],settings["legacy_per_family"])]
        answers=generate_many(model,tokenizer,[render_prompt(r["messages"][:-1]) for r in chosen],
                              max_new_tokens=settings["max_new_tokens"],batch_size=settings["batch_size"])
        legacy_predictions=[{"id":r["id"],"group":r["group"],"family":r["category"],"system_case":r.get("system_case","none"),
                             "pair_id":r.get("pair_id"),"correct":legacy_correct(r,a["response"]),**a} for r,a in zip(chosen,answers)]
        # Human prose remains a loss diagnostic; it never gets an invented correctness label.
        human=[r for r in rows if r["verifier"]=="none"]
        human_loss=evaluate_records(model,human)[0] if human else None
        raw=evaluate_documents(model,documents,context_length=config["max_length"],stride=config["max_length"]//2,
                               max_predictable_tokens=settings["max_raw_tokens_per_document"])
        metrics={"primary":sum(v["accuracy"] for v in domain_metrics.values())/len(domain_metrics),
                 "domains":domain_metrics,"choice_families":family_metrics(choices),"families":family_metrics(generated),
                 "legacy_families":family_metrics(legacy_predictions),"system":system_metrics(legacy_predictions),
                 "raw_lm":raw_metrics(raw,config),"repetition_rate":sum(repetition(r["response"]) for r in generated)/len(generated),
                 "eos_rate":sum(r["ended_eos"] for r in generated)/len(generated),"human_response_loss":human_loss,
                 "suffix_accuracy":sum(r["correct"] for r in suffix)/len(suffix) if suffix else None,
                 "generation":stats,"official_scores_used":False}
        files={}
        for name,values in (("primary",primary),("choices",choices),("suffix",suffix),("generated",generated),
                            ("legacy",legacy_predictions),("raw",raw)):
            path=output/(name+".jsonl");write_jsonl(path,values);files[path.name]=sha256(path)
        write_json(output/"metrics.json",metrics);files["metrics.json"]=sha256(output/"metrics.json")
        if sha256(model_path/"model.safetensors")!=model_identity["export_model_sha256"]:raise ValueError("Model changed during scoring")
        result={**model_identity,"contract_sha256":key,"metrics":metrics,"phase":phase,"files":files,"artifacts":str(output)}
        write_json(done,result);return result
    finally:
        del model;gc.collect()
        if str(device).startswith("cuda"):torch.cuda.empty_cache()
