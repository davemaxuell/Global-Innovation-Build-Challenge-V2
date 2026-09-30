"""Recoverable full-parameter SFT with replay for the selected model lineage."""
from __future__ import annotations
from collections import Counter, defaultdict
import fcntl
import importlib.metadata
import math
import os
from pathlib import Path
import random
import shutil
import signal
import time
import numpy as np
import torch
from transformers import AutoTokenizer
from scglm.model import load_model,save_model,audit_parameters
from scglm_post.common import append_event,publish_export,CHAT_TEMPLATE
from scglm_post.sft import response_loss_sum,learning_rate
from .common import ROOT,read,read_jsonl,write_json,write_jsonl,sha256,digest,resolve
from .posttraining_lineage import identity
from .sota_data import load as load_data


def source_contract():
    paths=[*(ROOT/"src/scglm_post").glob("*.py"), *(ROOT/"src/scglm_pipeline").glob("*.py"),
           ROOT/"src/scglm/model.py", ROOT/"src/scglm/evaluate.py", ROOT/"scripts/validation_history.py",
           ROOT/"scripts/run_pipeline.py", ROOT/"scripts/run_final_evaluation.py", ROOT/"scripts/release_tools.py",
           ROOT/"scripts/verify_sota_package.py", ROOT/"configs/evaluation.json", ROOT/"requirements.lock.txt"]
    return {str(p.relative_to(ROOT)):sha256(p) for p in sorted(paths)}


def balanced_plan(rows, config, mixture=None):
    pools=defaultdict(list)
    for epoch in range(config["epochs"]):
        order=list(range(len(rows))); random.Random(config["seed"]+epoch).shuffle(order)
        for i in order: pools[rows[i]["bucket"] if mixture else "all"].append(i)
    weights=mixture or {"all":1.}
    if any(not pools[k] for k in weights) or abs(sum(weights.values())-1)>1e-8:
        raise ValueError("Missing mixture bucket or invalid weights")
    capacities={k:sum(rows[i]["target_tokens"] for i in pools[k]) for k in weights}
    budget=min(config["max_targets"],int(min(capacities[k]/w for k,w in weights.items())))
    if budget<1: raise ValueError("No target budget")
    cursors=Counter(); used=Counter(); batches=[];batch=[];n=0;total=0;format_targets=0
    while True:
        active=[k for k,w in weights.items() if cursors[k]<len(pools[k]) and used[k]<budget*w]
        if not active:break
        bucket=min(active,key=lambda k:used[k]/weights[k]);i=pools[bucket][cursors[bucket]];cursors[bucket]+=1
        row=rows[i];tokens=row["target_tokens"]
        if total+tokens>budget:continue
        if row.get("formatting_only") and format_targets+tokens>budget*config.get("max_format_share",.05):continue
        used[bucket]+=tokens;total+=tokens;batch.append(i);n+=tokens
        if row.get("formatting_only"):format_targets+=tokens
        if n>=config["target_tokens_per_update"]:batches.append(batch);batch=[];n=0
    if batch:batches.append(batch)
    if not batches:raise ValueError("No complete training batch")
    return batches,{"targets":total,"bucket_targets":dict(used),"budget":budget,"format_targets":format_targets}


def sft_update(model,optimizer,rows,replay,config):
    model.train();optimizer.zero_grad(set_to_none=True)
    amount=config.get("replay_lambda",0.)
    if not 0<=amount<1 or (amount>0 and not replay):raise ValueError("Invalid replay objective")
    metrics={}
    for name,items,weight in (("response",rows,1-amount),("replay",replay,amount)):
        if not items:continue
        total=sum(r["target_tokens"] for r in items);loss_sum=0.
        for start in range(0,len(items),config["microbatch_examples"]):
            micro=items[start:start+config["microbatch_examples"]]
            device=next(model.parameters()).device.type
            with torch.autocast(device_type=device,dtype=torch.bfloat16,enabled=device=="cuda"):
                loss,_=response_loss_sum(model,micro,pad_token_id=model.config.pad_token_id)
            (weight*loss/total).backward();loss_sum+=loss.detach().item()
        metrics[name+"_nll"]=loss_sum/total
        metrics[name+"_targets"]=total
        metrics[name+"_processed_tokens"]=sum(r["processed_tokens"] for r in items)
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),config.get("gradient_clip",1.),error_if_nonfinite=True)
    optimizer.step();metrics["gradient_norm"]=float(norm);return metrics




class Trainer:
    def __init__(self,config,recipe,*,parent,manifest,output,stage="sft",device="cpu",on_evaluate=None):
        if stage != "sft":
            raise ValueError("Only the retained full-parameter SFT stage is supported")
        if recipe.get("legacy_control"):
            raise ValueError("Only the selected balanced SFT data path is supported")
        self.config=config;self.recipe=recipe;self.stage=stage;self.device=torch.device(device)
        self.parent=resolve(parent).resolve();self.manifest_path=resolve(manifest);self.output=resolve(output).resolve()
        if self.output==self.parent or self.output.is_relative_to(self.parent) or self.parent.is_relative_to(self.output):
            raise ValueError("Training output overlaps its parent")
        self.parent_identity=identity(self.parent,config,allow_stages=["pretraining"])
        self.manifest=load_data(self.manifest_path)
        if sha256(self.parent/"tokenizer.json")!=self.manifest["tokenizer_sha256"]:raise ValueError("Tokenizer changed")
        if self.device.type=="cuda":
            if torch.cuda.device_count()!=1:raise ValueError("Expose exactly one allocated GPU per trainer")
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
        torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.benchmark=False
        torch.manual_seed(recipe["seed"]);random.seed(recipe["seed"]);np.random.seed(recipe["seed"])
        self.tokenizer=AutoTokenizer.from_pretrained(str(self.parent),local_files_only=True);self.tokenizer.chat_template=CHAT_TEMPLATE
        self.model=load_model(self.parent).to(self.device)
        if any(getattr(m,"p",0)>0 for m in self.model.modules() if isinstance(m,torch.nn.Dropout)):
            raise ValueError("Selected SFT requires dropout disabled")
        self.optimizer=torch.optim.AdamW([
            {"params":[p for p in self.model.parameters() if p.ndim>=2],"weight_decay":recipe.get("weight_decay",0.)},
            {"params":[p for p in self.model.parameters() if p.ndim<2],"weight_decay":0.}],
            lr=recipe["learning_rate"],betas=(.9,.95))
        self.rows=read_jsonl(self.manifest_path.parent/self.manifest["files"]["train"]["file"])
        self.on_evaluate=on_evaluate
        self.plan=[];self.planned={};self.replay=defaultdict(list)
        self.rows=[r for r in self.rows if r.get("balanced",True)]
        self.plan,self.planned=balanced_plan(self.rows,recipe,config["mixture"])
        self.stop_after_updates=recipe.get("stop_after_updates",len(self.plan))
        if type(self.stop_after_updates) is not int or not 1 <= self.stop_after_updates <= len(self.plan):
            raise ValueError("Selected endpoint must be inside the unchanged full SFT schedule")
        if recipe.get("replay_lambda",0):
            for r in read_jsonl(self.manifest_path.parent/self.manifest["files"]["replay"]["file"]):self.replay[r["source"]].append(r)
            if set(self.replay)!={"edu","dclm","wiki"}:raise ValueError("Missing replay source")
        self.contract={"pipeline":"selected-sft-reproduction-v1","stage":stage,"config_sha256":digest(config),"recipe":recipe,
            "parent":self.parent_identity,"manifest_sha256":sha256(self.manifest_path),
            "source_sha256":source_contract(),"device":str(device),"plan_sha256":digest(self.plan),
            "versions":{n:importlib.metadata.version(n) for n in ("torch","transformers","tokenizers","numpy","safetensors")}}
        self.contract_hash=digest(self.contract)
        self.step=0;self.updates=0;self.counters=Counter();self.replay_cursors=Counter();self.replay_used=Counter()
        self.previous_seconds=0.;self.started=time.monotonic();self.stop_requested=False
        self.monitor={"best":None,"stale":0,"uninformative":0,"stop_reason":None};self.endpoints=[]
        self.output.mkdir(parents=True,exist_ok=True)
        self.lock=(self.output/".trainer.lock").open("a")
        try:fcntl.flock(self.lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:self.lock.close();raise ValueError("Another process owns training output") from None
        try:
            pointer=self.output/"checkpoint_latest.json"
            if pointer.exists():
                if read(self.output/"run.json")["contract_sha256"]!=self.contract_hash:raise ValueError("Resume contract changed")
                self.restore()
            else:
                run=self.output/"run.json"
                if run.exists() and read(run)["contract_sha256"]!=self.contract_hash:raise ValueError("Interrupted initialization contract changed")
                leftovers=[p for p in self.output.iterdir() if p.name not in (".trainer.lock","initialization_attempts")]
                if leftovers:
                    archive=self.output/"initialization_attempts"/str(time.time_ns());archive.mkdir(parents=True)
                    for p in leftovers:shutil.move(str(p),archive/p.name)
                write_json(run,{"contract":self.contract,"contract_sha256":self.contract_hash,
                                "planned":self.planned,"planned_updates":len(self.plan),"parameter_audit":audit_parameters(self.model)})
                for relative in self.contract["source_sha256"]:
                    dest=self.output/"snapshot"/relative;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/relative,dest)
                self.checkpoint();self.status("initialized")
        except BaseException:self.lock.close();raise

    @property
    def elapsed(self):return self.previous_seconds+time.monotonic()-self.started

    def event(self,name,**fields):
        append_event(self.output/"events.jsonl",{"event":name,"step":self.step,"time_unix":time.time(),**fields})

    def status(self,status,**fields):
        write_json(self.output/"status.json",{"status":status,"stage":self.stage,"pid":os.getpid(),"step":self.step,
            "optimizer_updates":self.updates,"elapsed_seconds":self.elapsed,"counters":dict(self.counters),
            "monitor":self.monitor,"endpoints":self.endpoints,**fields})

    def checkpoint(self):
        path=self.output/f"checkpoint_{self.step:07d}_{time.time_ns()}.pt";temporary=path.with_suffix(".tmp")
        state={"contract_sha256":self.contract_hash,"model":self.model.state_dict(),"optimizer":self.optimizer.state_dict(),
            "step":self.step,"updates":self.updates,"counters":dict(self.counters),"replay_cursors":dict(self.replay_cursors),
            "replay_used":dict(self.replay_used),
            "monitor":self.monitor,"endpoints":self.endpoints,"elapsed":self.elapsed,
            "torch_rng":torch.get_rng_state(),"numpy_rng":np.random.get_state(),"python_rng":random.getstate(),
            "cuda_rng":torch.cuda.get_rng_state_all() if self.device.type=="cuda" else None}
        torch.save(state,temporary)
        with temporary.open("rb") as stream:os.fsync(stream.fileno())
        temporary.replace(path)
        write_json(self.output/"checkpoint_latest.json",{"file":path.name,"sha256":sha256(path),"step":self.step})
        for p in sorted(self.output.glob("checkpoint_*.pt"))[:-2]:p.unlink()

    def restore(self):
        pointer=read(self.output/"checkpoint_latest.json");path=self.output/pointer["file"]
        if not path.resolve().is_relative_to(self.output) or sha256(path)!=pointer["sha256"]:raise ValueError("Checkpoint identity changed")
        state=torch.load(path,map_location=self.device,weights_only=False)
        if state["contract_sha256"]!=self.contract_hash:raise ValueError("Checkpoint contract changed")
        self.model.load_state_dict(state["model"]);self.optimizer.load_state_dict(state["optimizer"])
        self.step=state["step"];self.updates=state["updates"];self.counters=Counter(state["counters"])
        self.replay_cursors=Counter(state["replay_cursors"]);self.replay_used=Counter(state["replay_used"])
        self.monitor=state["monitor"];self.endpoints=state["endpoints"];self.previous_seconds=state["elapsed"]
        torch.set_rng_state(state["torch_rng"].cpu());np.random.set_state(state["numpy_rng"]);random.setstate(state["python_rng"])
        if self.device.type=="cuda":torch.cuda.set_rng_state_all([v.cpu() for v in state["cuda_rng"]])

    def replay_batch(self,targets):
        amount=self.recipe.get("replay_lambda",0.);needed=targets*amount/(1-amount)
        result=[];used=0;weights={"edu":.6,"dclm":.3,"wiki":.1}
        while used<needed:
            source=min(weights,key=lambda s:self.replay_used[s]/weights[s]);pool=self.replay[source]
            row=pool[self.replay_cursors[source]%len(pool)]
            self.replay_cursors[source]+=1;self.replay_used[source]+=row["target_tokens"]
            result.append(row);used+=row["target_tokens"]
        return result

    def export(self,label):
        if source_contract()!=self.contract["source_sha256"] or sha256(self.manifest_path)!=self.contract["manifest_sha256"]:
            raise ValueError("Training source or data registration changed")
        if sha256(self.parent/"model.safetensors")!=self.parent_identity["export_model_sha256"]:raise ValueError("Parent changed")
        destination=self.output/"endpoints"/label/"export";destination.parent.mkdir(parents=True,exist_ok=True)
        temporary=destination.with_name("export.tmp")
        if temporary.exists():shutil.rmtree(temporary)
        save_model(self.model,temporary);self.tokenizer.save_pretrained(str(temporary))
        shutil.copy2(self.parent/"tokenizer.json",temporary/"tokenizer.json")
        provenance={"pipeline":"selected-sft-reproduction-v1","stage":self.stage,"parent_model":str(self.parent),
            "parent_model_sha256":self.parent_identity["export_model_sha256"],"scratch_base_sha256":self.config["parent_sha256"],
            "cumulative_pretraining_tokens":self.config["expected_pretraining_tokens"],"manifest_sha256":sha256(self.manifest_path),
            "contract_sha256":self.contract_hash,"step":self.step,"optimizer_updates":self.updates,"counters":dict(self.counters),
            "disposable_smoke_test":bool(self.config.get("disposable_smoke_test")),"endpoint_kind":"completed_training_prefix"}
        write_json(temporary/"training_provenance.json",provenance)
        write_json(temporary/"endpoint.json",{"weights_sha256":sha256(temporary/"model.safetensors"),
            "provenance_sha256":sha256(temporary/"training_provenance.json"),"source_run":str(self.output),"step":self.step})
        publish_export(temporary,destination)
        entry={"label":label,"model":str(destination),"step":self.step,"weights_sha256":sha256(destination/"model.safetensors")}
        if entry not in self.endpoints:self.endpoints.append(entry)
        return destination



    def run(self,max_updates=None):
        """Stop at the selected prefix while scheduling against the full plan."""
        previous={s:signal.getsignal(s) for s in (signal.SIGTERM,signal.SIGINT)}
        for s in previous:signal.signal(s,lambda *args:setattr(self,"stop_requested",True))
        limit=self.stop_after_updates
        completed_now=0
        try:
            if (self.output/"status.json").exists() and read(self.output/"status.json")["status"]=="completed":
                return self.endpoints
            self.monitor["stop_reason"]=None
            self.status("running")
            while self.step<limit:
                if self.elapsed>=self.recipe["max_seconds"]:
                    self.monitor["stop_reason"]="wall_time_budget"
                    break
                if self.stop_requested or (max_updates is not None and completed_now>=max_updates):break
                self.event("work_started",kind="optimizer",attempt_step=self.step)
                rows=[self.rows[i] for i in self.plan[self.step]]
                targets=sum(r["target_tokens"] for r in rows)
                rate=learning_rate(self.counters["response_targets"]+targets,self.planned["targets"],self.recipe)
                for g in self.optimizer.param_groups:g["lr"]=rate
                metrics=sft_update(self.model,self.optimizer,rows,self.replay_batch(targets),self.recipe)
                self.step+=1;completed_now+=1;self.updates+=1
                for key,value in metrics.items():
                    if key.endswith(("tokens","targets")):self.counters[key]+=value
                self.event("work_completed",kind="optimizer",metrics=metrics,learning_rate=rate)
                boundary=self.step in {max(1,math.ceil(len(self.plan)*f)) for f in self.recipe.get("endpoint_fractions",[1.])}
                if boundary or self.step==limit:
                    path=self.export(f"step_{self.step:07d}")
                    if self.on_evaluate:self.on_evaluate(path,self)
                if self.step%self.recipe["checkpoint_every"]==0 or boundary:self.checkpoint()
                self.status("running")
            complete=self.step==limit
            if complete and not any(e["step"]==self.step for e in self.endpoints):self.export(f"step_{self.step:07d}")
            self.checkpoint();self.status("completed" if complete else "paused")
            return self.endpoints
        except BaseException as error:
            self.event("failed",error=repr(error));self.status("failed",error=repr(error));raise
        finally:
            for s,handler in previous.items():signal.signal(s,handler)
            self.lock.close()
