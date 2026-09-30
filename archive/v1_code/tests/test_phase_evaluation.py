"""Safety, recovery and real CPU inference tests for the phase observer."""
import json
import os
from pathlib import Path
import sys
import time

import pytest

from scglm_phase.common import artifact_hashes, finite, jsonl, lease, read, sha256, verify_receipt, write
from scglm_phase.evaluate import evaluate, generation_metrics, load_target, repetition
from scglm_phase.runner import Pipeline, command, load_comparator, readiness


def test_log_reader_ignores_only_unfinished_line(tmp_path):
    path=tmp_path/'events'
    path.write_bytes(b'{"step":1}\n{"step":')
    assert jsonl(path)==[{'step':1}]
    path.write_bytes(b'{"step":1}\ninvalid\n')
    with pytest.raises(json.JSONDecodeError):
        jsonl(path)


def test_nonfinite_nested_metrics_are_rejected():
    assert finite({'a':[1,2,None,False]})
    assert not finite({'a':[{'loss':float('nan')}]})
    assert not finite({'a':float('inf')})


def test_gpu_lease_excludes_other_open_and_inherited_close_does_not_unlock(tmp_path):
    path=tmp_path/'gpu.lock'
    with lease(path) as parent:
        with pytest.raises(BlockingIOError):
            with lease(path):
                pytest.fail('Overlapping GPU lease')
        with lease(path,inherited_fd=parent.fileno()):
            pass
        with pytest.raises(BlockingIOError):
            with lease(path):
                pytest.fail('Child released parent GPU lease')
        other=tmp_path/'wrong.lock'; other.touch()
        with pytest.raises(ValueError,match='different'):
            with lease(other,inherited_fd=parent.fileno()):
                pass
    with lease(path):
        pass


def test_receipt_rejects_changed_or_added_results_and_wrong_registration(tmp_path):
    folder=tmp_path/'out';folder.mkdir();write(folder/'result.json',{'x':1})
    record={'status':'completed','registration_sha256':'abc','output':str(folder),'artifacts':artifact_hashes(folder)}
    verify_receipt(record,'abc')
    with pytest.raises(ValueError,match='registration'):
        verify_receipt(record,'def')
    (folder/'extra').write_text('unexpected')
    with pytest.raises(ValueError,match='changed'):
        verify_receipt(record,'abc')
    (folder/'extra').unlink();write(folder/'result.json',{'x':2})
    with pytest.raises(ValueError,match='changed'):
        verify_receipt(record,'abc')


def test_readiness_does_not_confuse_pilot_pause_with_supervisor_stop():
    now=time.time();training={'status':'paused','updated_unix':now}
    supervisor={'stage':'pilot_training','pid':os.getpid(),'updated_unix':now}
    assert readiness(training,supervisor,False,now,900)[0]=='waiting'
    assert readiness(training,{**supervisor,'stage':'paused'},False,now,900)[0]=='blocked'
    assert readiness(training,{**supervisor,'pid':None},False,now,900)[0]=='blocked'
    assert readiness(training,supervisor,False,now+901,900)[0]=='blocked'


def test_readiness_requires_both_completed_training_and_final_comparison():
    sup={'stage':'completed'}
    assert readiness({'status':'completed'},sup,True,0,900)[0]=='ready'
    assert readiness({'status':'running'},sup,True,0,900)[0]=='blocked'
    assert readiness({'status':'completed'},sup,False,0,900)[0]=='blocked'


def test_timeout_stops_only_owned_subprocess_and_keeps_log(tmp_path):
    code='import os,time; print(os.getpid(),flush=True); time.sleep(30)'
    with pytest.raises(TimeoutError):
        command([sys.executable,'-c',code],tmp_path,tmp_path/'log',dict(os.environ),.4,lambda pid:None)
    pid=int((tmp_path/'log').read_text().strip())
    with pytest.raises(ProcessLookupError):
        os.kill(pid,0)
    assert os.getpid()!=pid


def test_repetition_metrics_do_not_imply_accuracy():
    assert repetition('one two three four '*4)['fourfold_4gram_loop']
    assert not repetition('one two three four five six')['fourfold_4gram_loop']
    rows=[{'id':'a','response':'one two three four '*4,'ended_eos':False,'generated_tokens':20,'elapsed_seconds':2},
          {'id':'b','response':'','ended_eos':True,'generated_tokens':1,'elapsed_seconds':1}]
    scores=generation_metrics(rows)
    assert scores['loop_rate']==scores['eos_rate']==scores['nonempty_rate']==.5
    assert scores['tokens_per_second']==7
    assert 'accuracy' not in scores
    with pytest.raises(ValueError):
        generation_metrics([rows[0],rows[0]])


@pytest.fixture
def stage_pipeline(tmp_path,monkeypatch):
    p=Pipeline.__new__(Pipeline)
    p.root=tmp_path;p.work=tmp_path/'work';p.work.mkdir()
    p.config={'command_attempts':2,'cpu_threads':1};p.registration={'sha256':'registered'};p.receipts={}
    monkeypatch.setattr(p,'check_frozen',lambda:None)
    monkeypatch.setattr(p,'state',lambda *a,**k:None)
    monkeypatch.setattr(p,'event',lambda *a,**k:None)
    monkeypatch.setattr(p,'report',lambda:None)
    def validate(label,out):
        result=read(out/'result.json')
        if result.get('status')!='completed':
            raise ValueError('Incomplete result')
    monkeypatch.setattr(p,'validate_result',validate)
    return p


def child_result(output, *, fail_first=False, always_fail=False):
    code='import pathlib,json,sys\np=pathlib.Path(sys.argv[1])\n'
    if always_fail:
        code+='raise RuntimeError("intentional failure")\n'
    elif fail_first:
        code+='if p.parent.name=="01": raise RuntimeError("first attempt failure")\n'
    code+='p.mkdir()\n(p/"result.json").write_text(json.dumps({"status":"completed"}))\n'
    return [sys.executable,'-c',code,str(output)]


def test_real_subprocess_retry_and_idempotent_receipt(stage_pipeline):
    p=stage_pipeline
    with lease(p.work/'gpu') as gpu:
        p.run_stage('phase',gpu,lambda out:child_result(out,fail_first=True),5)
        assert (p.work/'attempts/phase/01/failure.json').exists()
        assert (p.work/'attempts/phase/02/completed.json').exists()
        p.run_stage('phase',gpu,lambda out:pytest.fail('Successful stage ran twice'),5)
    verify_receipt(p.receipts['phase'],'registered')


def test_failed_stages_stop_at_persisted_retry_budget(stage_pipeline):
    p=stage_pipeline
    with lease(p.work/'gpu') as gpu:
        with pytest.raises(RuntimeError,match='exhausted'):
            p.run_stage('phase',gpu,lambda out:child_result(out,always_fail=True),5)
        with pytest.raises(RuntimeError,match='exhausted'):
            p.run_stage('phase',gpu,lambda out:pytest.fail('Exceeded retry budget'),5)
    assert len(list((p.work/'attempts/phase').iterdir()))==2
    assert not p.receipts


def test_successful_output_recovers_after_missing_receipt(stage_pipeline):
    p=stage_pipeline
    folder=p.work/'attempts/phase/01';folder.mkdir(parents=True)
    write(folder/'command.json',{'registration_sha256':'registered'})
    write(folder/'output/result.json',{'status':'completed'})
    with lease(p.work/'gpu') as gpu:
        p.run_stage('phase',gpu,lambda out:pytest.fail('Recovered output was rerun'),5)
    assert p.receipts['phase']['status']=='completed'


def test_recovery_cannot_adopt_output_from_another_registration(stage_pipeline):
    p=stage_pipeline
    folder=p.work/'attempts/phase/01';folder.mkdir(parents=True)
    write(folder/'command.json',{'registration_sha256':'another'})
    write(folder/'output/result.json',{'status':'completed'})
    with lease(p.work/'gpu') as gpu:
        p.run_stage('phase',gpu,lambda out:child_result(out),5)
    assert Path(p.receipts['phase']['output']).parent.name=='02'


def test_real_cpu_evaluation_persists_all_documents_generations_and_checks(tmp_path,monkeypatch):
    import torch
    from scglm.model import build_model,audit_parameters
    import scglm_phase.evaluate as worker
    config={'vocab_size':32,'hidden_size':8,'intermediate_size':16,'num_hidden_layers':1,
            'num_attention_heads':2,'num_key_value_heads':2,'max_position_embeddings':1024,'seed':10}
    model=build_model(config)
    class Tokenizer:
        pad_token_id=0;eos_token_id=2
        def encode(self,s,add_special_tokens=False):
            return [4+ord(c)%28 for c in s]
        def decode(self,ids,skip_special_tokens=True):
            return ' '.join(str(i) for i in ids if i>3)
    original={k:v.detach().clone() for k,v in model.state_dict().items()}
    monkeypatch.setattr(worker,'load_target',lambda root,spec:(model,Tokenizer(),{'cumulative_tokens':64},audit_parameters(model),
        lambda:all(torch.equal(v,original[k]) for k,v in model.state_dict().items())))
    panel=tmp_path/'panel.jsonl'
    panel.write_text(''.join(json.dumps({'source':s,'id':str(i),'tokens':[4,5,6,7,8]})+'\n'
                             for s in ('edu','dclm','wiki') for i in range(2)))
    (tmp_path/'prompts.py').write_text('CASES='+repr([(f'p{i}','A short text') for i in range(12)]))
    cfg={'cpu_threads':1,'seed':1,'development_panel':{'path':'panel.jsonl','sha256':sha256(panel)},
         'source_weights':{'edu':.6,'dclm':.3,'wiki':.1},'prompt_script':'prompts.py'}
    result=evaluate(tmp_path,cfg,{'label':'tiny'},tmp_path/'out',device='cpu')
    assert result['status']=='completed'
    assert len(result['raw']['documents'])==6
    assert all(r['token_count']==4 for r in result['raw']['documents'])
    assert len(jsonl(tmp_path/'out/responses.jsonl'))==12
    assert result['checks']['artifacts_unchanged']
    assert result['checks']['full_context_logits_finite']
    assert result['checks']['deterministic_repeat_matches']
    assert 'p11' in (tmp_path/'out/GENERATIONS.md').read_text()


def test_cpu_milestone_load_is_exact_and_rejects_bad_accounting(tmp_path,monkeypatch):
    import torch
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast
    import scglm.model as modeling
    config={'vocab_size':32,'hidden_size':8,'intermediate_size':16,'num_hidden_layers':1,
            'num_attention_heads':2,'num_key_value_heads':2,'max_position_embeddings':1024,'seed':10}
    model=modeling.build_model(config)
    run=tmp_path/'run';folder=run/'milestones';folder.mkdir(parents=True)
    export=tmp_path/'parent_export';export.mkdir()
    vocab={'<pad>':0,'<bos>':1,'<eos>':2,'<unk>':3,**{f't{i}':i for i in range(4,32)}}
    tok=PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordLevel(vocab,unk_token='<unk>')),
         pad_token='<pad>',bos_token='<bos>',eos_token='<eos>',unk_token='<unk>')
    tok.save_pretrained(export)
    write(run/'run.json',{'science_digest':'science','model_config':config,'config':{'sequence_length':10,'batch_sequences':1},
                         'parent':{'cumulative_tokens':100}})
    write(run/'status.json',{'status':'completed'})
    path=folder/'milestone_148.pt'
    torch.save({'science_digest':'science','step':5,'main_tokens':50,'model':model.state_dict()},path)
    marker={'sha256':sha256(path),'path':str(path),'step':5,'main_tokens':50,'cumulative_tokens':150}
    write(path.with_suffix('.json'),marker)
    audit=modeling.audit_parameters
    # Test loader accounting and actual tensor equality with a small real model.
    # The production model-size invariant is separately covered by core model tests.
    monkeypatch.setattr(modeling,'audit_parameters',lambda m,*a,**kw:{**audit(m,*a,**kw),'total_unique_parameters':46346752})
    spec={'kind':'milestone','run':'run','nominal_tokens':148,'tokenizer_export':'parent_export','tokenizer_sha256':sha256(export/'tokenizer.json')}
    loaded,_,identity,_,unchanged=load_target(tmp_path,spec)
    assert identity['cumulative_tokens']==150 and unchanged()
    assert all(torch.equal(v,loaded.state_dict()[k]) for k,v in model.state_dict().items())
    write(path.with_suffix('.json'),{**marker,'cumulative_tokens':149})
    with pytest.raises(ValueError,match='accounting'):
        load_target(tmp_path,spec)
    write(path.with_suffix('.json'),{**marker,'sha256':'wrong'})
    with pytest.raises(ValueError,match='checksum'):
        load_target(tmp_path,spec)


@pytest.mark.parametrize('gain',[True,False])
def test_final_decision_reproduces_gates_and_rejects_wrong_recommendation(tmp_path,monkeypatch,gain):
    import scglm_phase.runner as runner
    original_compare=load_comparator(Path(__file__).resolve().parents[1])
    monkeypatch.setattr(runner,'load_comparator',lambda root:original_compare)
    p=Pipeline.__new__(Pipeline);p.root=tmp_path;p.run=tmp_path/'candidate';p.source=tmp_path/'supervisor'
    (p.run/'export').mkdir(parents=True);(p.run/'export/model.safetensors').write_bytes(b'fixture-weights')
    candidate_hash=sha256(p.run/'export/model.safetensors')
    weights={'edu':.6,'dclm':.3,'wiki':.1}
    protocol={'bootstrap_seed':1,'bootstrap_replicates':100,'min_mixture_nll_improvement':.005,'max_source_nll_regression':.02}
    p.config={'training_contract_sha256':'contract','target_tokens':100,'parent_model_sha256':'parent-hash',
              'parent_run':'parent','source_weights':weights}
    write(p.source/'contract.json',{'protocol':protocol})
    panels={};results={}
    for phase in ('development','confirmation') if gain else ('development',):
        panel=p.source/'panels'/f'{phase}.jsonl';panel.parent.mkdir(parents=True,exist_ok=True);panel.write_text('fixture\n')
        panels[phase]={'path':str(panel.relative_to(p.source)),'sha256':sha256(panel)}
        base={'documents':[{'source':s,'id':str(i),'token_count':10,'nll_sum':30.} for s in weights for i in range(2)]}
        candidate={'documents':[{**r,'nll_sum':r['nll_sum']+(-1 if gain else 1)} for r in base['documents']]}
        for label,model_hash,record in [('parent','parent-hash',base),('candidate',candidate_hash,candidate)]:
            write(p.source/f'{label}_{phase}.json',{**record,'identity':{'model_sha256':model_hash,'panel_sha256':sha256(panel),'contract_sha256':'contract'}})
        results[phase]=original_compare(base,candidate,weights,protocol)
    write(p.source/'preflight.json',{'panels':panels})
    decision={'status':'completed','contract_sha256':'contract','candidate_sha256':candidate_hash,
              'candidate_cumulative_tokens':100,'development':results['development'],
              'confirmation':results.get('confirmation'),'recommended_model':('candidate' if gain else 'parent')+'/export'}
    write(p.source/'comparison.json',decision)
    assert p.verify_decision()==decision
    write(p.source/'comparison.json',{**decision,'recommended_model':('parent' if gain else 'candidate')+'/export'})
    with pytest.raises(ValueError,match='contradicts'):
        p.verify_decision()
