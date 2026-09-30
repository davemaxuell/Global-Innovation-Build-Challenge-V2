"""Durable phase evaluator. Observes training; never changes training policy."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback

from .common import artifact_hashes, digest, finite, jsonl, lease, read, sha256, verify_receipt, write


def process_alive(pid):
    try:
        os.kill(int(pid), 0)
        return True
    except (ProcessLookupError, ValueError, TypeError):
        return False


def readiness(training, supervisor, comparison_exists, now, stale_seconds):
    if supervisor.get('stage') in {'failed', 'failed_preflight', 'paused'}:
        return 'blocked', f"Training supervisor is {supervisor['stage']}; no automatic retraining"
    if supervisor.get('stage') == 'completed':
        if training.get('status') != 'completed' or not comparison_exists:
            return 'blocked', 'Completion is missing a completed model or comparison'
        return 'ready', 'Training and its fixed held-out decision are complete'
    # A pilot writes a temporary paused status while its supervisor continues.
    if not process_alive(supervisor.get('pid')):
        return 'blocked', 'Training supervisor exited without a completed decision'
    latest = max(training.get('updated_unix', 0), supervisor.get('updated_unix', 0))
    if now-latest > stale_seconds:
        return 'blocked', 'Training heartbeat is stale; inspect the supervisor before resuming'
    return 'waiting', 'Waiting for training and its registered comparison; GPU untouched'


def command(command, cwd, log, env, timeout, heartbeat, pass_fds=()):
    """Terminate only the subprocess we launched; preserve logs on every failure."""
    started = time.monotonic()
    with Path(log).open('w') as f:
        child = subprocess.Popen(command, cwd=cwd, env=env, stdout=f, stderr=subprocess.STDOUT,
                                 start_new_session=True, pass_fds=pass_fds)
        last_beat = 0
        try:
            while child.poll() is None:
                if time.monotonic()-started > timeout:
                    raise TimeoutError(f'Phase exceeded {timeout} seconds; see {log}')
                if time.monotonic()-last_beat >= 15:
                    heartbeat(child.pid)
                    last_beat = time.monotonic()
                time.sleep(.25)
            if child.returncode != 0:
                raise RuntimeError(f'Phase exited {child.returncode}; see {log}')
        finally:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
    return time.monotonic()-started


def load_comparator(root):
    spec = importlib.util.spec_from_file_location('registered_continuation', root/'scripts/run_pretraining_next.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.compare


class Pipeline:
    def __init__(self, root, config_path):
        self.root = Path(root).resolve()
        self.config_path = self.root/config_path
        self.config = read(self.config_path)
        self.work = self.root/self.config['output']
        self.work.mkdir(parents=True, exist_ok=True)
        self.source = self.root/self.config['supervisor_output']
        self.run = self.root/self.config['training_run']
        self.receipts = {}
        self.runtime = {'stage': 'initializing', 'pid': os.getpid()}
        self.registration = None

    def verify_training_contract(self):
        contract = read(self.source/'contract.json')
        if contract['sha256'] != self.config['training_contract_sha256']:
            raise ValueError('Training registration differs from the evaluator target')
        if digest({k:v for k,v in contract.items() if k != 'sha256'}) != contract['sha256']:
            raise ValueError('Training contract content does not match its digest')
        for path, checksum in contract['files'].items():
            if sha256(path) != checksum:
                raise ValueError(f'Active training implementation changed: {path}')
        return contract

    def freeze(self):
        training_contract=self.verify_training_contract()
        paths = [self.config_path, *sorted((self.root/'src/scglm_phase').glob('*.py')),
                 self.root/'scripts/run_phase_evaluation.py', self.root/'tests/test_phase_evaluation.py',
                 self.root/'PHASE_EVALUATION.md', self.root/self.config['prompt_script'],
                 self.root/'scripts/run_final_evaluation.py', self.root/'scripts/validation_history.py',
                 self.source/'contract.json', *[Path(p).resolve() for p in training_contract['files']]]
        for path, checksum in self.config['reference_files'].items():
            if sha256(self.root/path) != checksum:
                raise ValueError(f'Protected reference changed: {path}')
            paths.append(self.root/path)
        body = {'schema_version': 1, 'config': self.config,
                'files': {str(p.relative_to(self.root)): sha256(p) for p in sorted(set(paths))},
                'versions': {k: importlib.metadata.version(k) for k in
                    ('torch','transformers','datasets','tokenizers','safetensors','numpy','matplotlib')},
                'training_contract_sha256': self.config['training_contract_sha256']}
        path = self.work/'registration.json'
        if path.exists():
            record = read(path)
            if record['body'] != body or record['sha256'] != digest(body):
                raise ValueError('Evaluator code/config/reference changed: use a new output registration')
        else:
            record = {'body': body, 'sha256': digest(body), 'registered_unix': time.time(),
                      'training_observation_at_registration': read(self.run/'status.json'),
                      'disclosure': 'Registered during 13B training. 12B official scores and completion examples were already observed. No claim of a fully blind experiment.'}
            write(path, record)
            for p in sorted(set(paths)):
                target = self.work/'snapshot'/p.relative_to(self.root)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p, target)
        self.registration = record
        for p in sorted((self.work/'receipts').glob('*.json')):
            receipt = read(p)
            verify_receipt(receipt, record['sha256'])
            self.receipts[p.stem] = receipt

    def check_frozen(self):
        self.verify_training_contract()
        for path, checksum in self.registration['body']['files'].items():
            if sha256(self.root/path) != checksum:
                raise ValueError(f'Frozen evaluation input changed: {path}')

    def validate_prior_evidence(self):
        preflight=self.evidence('preflight.json')
        recovery=self.evidence('gpu_recovery.json')
        pilot=self.evidence('pilot_gate.json')
        if preflight['status']!='passed' or preflight['cumulative_target']!=self.config['target_tokens']:
            raise ValueError('Registered preflight did not pass for this target')
        if (recovery['status']!='passed' or any(recovery[k] is not True for k in
                ('inherited_weights_optimizer_streams_exact','resumed_state_and_loss_exact','export_exact'))):
            raise ValueError('Registered GPU recovery checks did not pass')
        protocol=read(self.source/'contract.json')['protocol']
        before,after=pilot['before'],pilot['after']
        if (pilot['status']!='passed' or pilot['failures']
                or after['fixed_q_nll']-before['fixed_q_nll'] > protocol['pilot_max_mixture_nll_increase']
                or any(after['sources'][s]['nll']-before['sources'][s]['nll'] > protocol['pilot_max_source_nll_increase']
                       for s in self.config['source_weights'])):
            raise ValueError('Registered pilot stability checks did not pass')

    def state(self, stage, **extra):
        self.runtime = {'stage': stage, 'pid': os.getpid(), 'updated_unix': time.time(),
                        'registration_sha256': self.registration['sha256'] if self.registration else None, **extra}
        write(self.work/'state.json', self.runtime)

    def event(self, event, **extra):
        record = {'event': event, 'time_unix': time.time(), **extra}
        with (self.work/'events.jsonl').open('a') as f:
            f.write(json.dumps(record, allow_nan=False)+'\n')
        print(json.dumps(record), flush=True)

    def evidence(self, name):
        path = self.source/name
        if not path.exists():
            return None
        result = read(path)
        if result.get('contract_sha256') != self.config['training_contract_sha256'] or not finite(result):
            raise ValueError(f'Invalid registered evidence: {name}')
        return result

    def verify_decision(self):
        result = self.evidence('comparison.json')
        if result is None:
            raise ValueError('Registered final decision missing')
        if result['status'] != 'completed':
            raise ValueError('Incomplete final comparison')
        sha = sha256(self.run/'export/model.safetensors')
        if sha != result['candidate_sha256'] or result['candidate_cumulative_tokens'] != self.config['target_tokens']:
            raise ValueError('Final model differs from the comparison')
        compare = load_comparator(self.root)
        protocol = read(self.source/'contract.json')['protocol']
        phases = ('development', 'confirmation') if result['development']['passed'] else ('development',)
        for phase in phases:
            panel = read(self.source/'preflight.json')['panels'][phase]
            if sha256(self.source/panel['path']) != panel['sha256']:
                raise ValueError('Decision panel changed')
            records = []
            for label, model_sha in [('parent', self.config['parent_model_sha256']), ('candidate', sha)]:
                record = read(self.source/f'{label}_{phase}.json')
                expected = {'model_sha256': model_sha, 'panel_sha256': panel['sha256'],
                            'contract_sha256': self.config['training_contract_sha256']}
                if record['identity'] != expected:
                    raise ValueError('Decision evidence identity mismatch')
                records.append(record)
            if compare(*records, self.config['source_weights'], protocol) != result[phase]:
                raise ValueError('Decision does not reproduce from document likelihoods')
        keep = result['development']['passed'] and result['confirmation'] is not None and result['confirmation']['passed']
        if not result['development']['passed'] and result['confirmation'] is not None:
            raise ValueError('Confirmation was opened despite a failed development gate')
        expected = (self.run if keep else self.root/self.config['parent_run'])/'export'
        if (self.root/result['recommended_model']).resolve() != expected.resolve():
            raise ValueError('Recommended endpoint contradicts fixed likelihood gates')
        return result

    def report(self):
        training, supervisor = read(self.run/'status.json'), read(self.source/'status.json')
        phases = []
        def add(name, status, detail, **values):
            phases.append({'phase': name, 'status': status, 'detail': detail, **values})
        for name, file in [('Corpus and continuation preflight','preflight.json'), ('Exact GPU recovery','gpu_recovery.json')]:
            evidence = self.evidence(file)
            add(name, evidence['status'] if evidence else 'pending',
                'Existing registered evidence; verified identity' if evidence else 'Waiting for evidence', evidence=file)
        pilot = self.evidence('pilot_gate.json')
        if pilot:
            change = pilot['after']['fixed_q_nll']-pilot['before']['fixed_q_nll']
            add('1,000-update pilot', pilot['status'], f'Monitor NLL change {change:+.6f}; stability gate, not a quality-improvement claim',
                before=pilot['before'], after=pilot['after'], nll_change=change)
        else:
            add('1,000-update pilot','pending','Waiting for registered pilot gate')
        rows = jsonl(self.run/'events.jsonl')
        curves = [r for r in rows if r['event'] in ('train','validation')]
        if not finite(curves) or not finite(training):
            raise ValueError('Nonfinite training/evaluation telemetry')
        progress = training['main_tokens']/training['target_tokens']
        add('Continuation training', training['status'],
            f"{training['main_tokens']:,}/{training['target_tokens']:,} added targets ({progress:.1%}); step {training['step']:,}",
            training=training)
        comparison = self.evidence('comparison.json')
        for phase in ('development', 'confirmation'):
            gate = comparison[phase] if comparison else None
            status = ('passed' if gate['passed'] else 'rejected') if gate else (
                'skipped' if comparison and phase=='confirmation' else 'pending')
            detail = f"Candidate − 12B NLL {gate['candidate_minus_parent_nll']:+.6f}, 95% interval [{gate['lower_95']:+.6f}, {gate['upper_95']:+.6f}]" if gate else (
                'Development gate did not pass; confirmation remains unopened' if status=='skipped' else 'Owned by the registered training supervisor')
            add(f'Final {phase} quality gate', status, detail, result=gate)
        checkpoints = []
        for spec in self.config['checkpoints']:
            receipt = self.receipts.get(spec['label'])
            if receipt:
                r = read(Path(receipt['output'])/'result.json')
                checkpoints.append(r)
                add(f"{spec['label']} checkpoint assessment", 'completed',
                    f"Development NLL {r['raw']['metrics']['fixed_q_nll']:.6f}; generation loop rate {r['generation']['loop_rate']:.1%}")
            else:
                active=self.runtime['stage']=='evaluating' and self.runtime.get('phase')==spec['label']
                add(f"{spec['label']} checkpoint assessment",'running' if active else 'queued','Same-panel FP32 likelihood, completion diagnostics and numerical/identity checks')
        add('12B official benchmarks','completed','Reuses the checksummed full benchmark run from September 24')
        official_active=self.runtime['stage']=='evaluating' and self.runtime.get('phase')=='official_13b'
        add('13B official benchmarks','completed' if 'official_13b' in self.receipts else ('running' if official_active else 'queued'),
            'Full HellaSwag, ARC-Easy, PIQA, WinoGrande and WikiText-103; report-only after fixed raw-text decision')
        add('Post-training assessment','out_of_scope','This report covers the completed raw-pretraining phase; selected full-SFT results are recorded in BEST_CHECKPOINT_TRAINING.md')
        add('Evidence handoff','completed' if self.runtime['stage']=='completed' else 'pending','JSON, Markdown, CSV, curves, per-example outputs and checksums')
        report = {'updated_utc':datetime.now(timezone.utc).isoformat(), 'runtime':self.runtime, 'phases':phases,
                  'training':training, 'checkpoints':checkpoints, 'registered_decision':comparison,
                  'limitations': ['Stability is separate from quality; lower training loss alone does not establish knowledge gains.',
                    'BF16 monitor losses and FP32 development losses use different panels; compare only within each protocol.',
                    'Intermediate checkpoints are assessed after the final decision and never promoted by these diagnostics.',
                    'The 12 illustrative prompts do not measure representative accuracy or competition rank.',
                    '12B official scores were previously observed; benchmark comparisons are descriptive and never change the frozen recommendation.']}
        self.write_report(report, curves)
        return report

    def write_report(self, report, curves):
        write(self.work/'report.json', report)
        lines = ['# Training phase performance', '', f"Updated: {report['updated_utc']}", '',
                 f"Workflow: **{report['runtime']['stage']}**. Cumulative training targets: **{report['training']['cumulative_tokens']:,}**.", '',
                 '| Phase | Status | Evidence / interpretation |', '| --- | --- | --- |']
        for p in report['phases']:
            lines.append(f"| {p['phase']} | {p['status']} | {p['detail']} |")
        lines += ['', '## Checkpoint quality on the same development panel', '',
                  'Mixture NLL uses fixed source weights 60% Edu / 30% DCLM / 10% Wikipedia. Lower is better. Generation diagnostics use 12 fixed prompts; no factual-accuracy score is inferred.', '',
                  '| Checkpoint | Actual cumulative targets | NLL | Perplexity | Δ NLL vs prior evaluated checkpoint | Loop rate | EOS rate |',
                  '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
        flat, previous = [], None
        for r in report['checkpoints']:
            m, g = r['raw']['metrics'], r['generation']
            delta = m['fixed_q_nll']-previous if previous is not None else None
            lines.append(f"| {r['label']} | {r['identity']['cumulative_tokens']:,} | {m['fixed_q_nll']:.6f} | {m['fixed_q_perplexity']:.4f} | {delta if delta is not None else '—'} | {g['loop_rate']:.1%} | {g['eos_rate']:.1%} |")
            for source, metrics in {'mixture':{'nll':m['fixed_q_nll'], 'perplexity':m['fixed_q_perplexity']}, **m['sources']}.items():
                for metric in ('nll','perplexity'):
                    flat.append([r['label'],'development',source,metric,metrics[metric]])
            for metric in ('eos_rate','loop_rate','mean_duplicate_4gram_fraction','tokens_per_second'):
                flat.append([r['label'],'generation','12 illustrative prompts',metric,g[metric]])
            previous = m['fixed_q_nll']
        base = read(self.root/self.config['baseline_official_summary'])
        candidate = read(Path(self.receipts['official_13b']['output'])/'summary.json') if 'official_13b' in self.receipts else None
        lines += ['', '## Official benchmark reporting', '',
                  'Zero-shot frozen project protocol; both raw and character-normalized accuracy are shown where defined. These scores do not revise the likelihood-based recommendation. Differences are point estimates, not significance claims.', '',
                  '| Task / metric | 12B | 13B | Change |', '| --- | ---: | ---: | ---: |']
        for task, values in base['metrics']['benchmarks'].items():
            for metric in ('acc','acc_norm'):
                if metric not in values:
                    continue
                a = values[metric]
                b = candidate['metrics']['benchmarks'][task][metric] if candidate else None
                lines.append(f"| {task} / {metric} | {a:.6f} | {f'{b:.6f}' if b is not None else 'pending'} | {f'{b-a:+.6f}' if b is not None else '—'} |")
                flat.append(['12b','official',task,metric,a])
                if b is not None:
                    flat.append(['13b','official',task,metric,b])
        a = base['metrics']['wikitext103']['perplexity']
        b = candidate['metrics']['wikitext103']['perplexity'] if candidate else None
        lines.append(f"| WikiText-103 perplexity ↓ | {a:.6f} | {f'{b:.6f}' if b is not None else 'pending'} | {f'{b-a:+.6f}' if b is not None else '—'} |")
        flat.append(['12b','official','wikitext103','perplexity',a])
        if b is not None:
            flat.append(['13b','official','wikitext103','perplexity',b])
        if report['registered_decision']:
            lines += ['', f"Frozen recommendation: **{report['registered_decision']['recommendation']}**."]
        lines += ['', '## Training curves', '', '![Training performance](training_curves.png)', '',
                  '[Machine-readable report](report.json) · [Metric table](performance.csv) · [Curve data](training_curves.csv) · [Registration](registration.json)', '',
                  '## Interpretation limits', '', *['- '+s for s in report['limitations']], '']
        tmp = self.work/'REPORT.md.tmp'; tmp.write_text('\n'.join(lines)); os.replace(tmp,self.work/'REPORT.md')
        with (self.work/'performance.csv').open('w') as f:
            writer=csv.writer(f); writer.writerow(['checkpoint','evaluation','source_or_task','metric','value']); writer.writerows(flat)
        curve_rows = []
        for r in curves:
            m = r.get('metrics',{})
            curve_rows.append([r['event'],r['step'],r['main_tokens'],r['elapsed_seconds'],r.get('loss'),r.get('gradient_norm'),
                r.get('learning_rate'),r.get('tokens_per_second'),m.get('fixed_q_nll'),m.get('fixed_q_perplexity'),
                *[m.get('sources',{}).get(s,{}).get('nll') for s in self.config['source_weights']]])
        with (self.work/'training_curves.csv').open('w') as f:
            writer=csv.writer(f); writer.writerow(['kind','step','added_targets','elapsed_seconds','train_loss','gradient_norm','learning_rate',
                'tokens_per_second','monitor_nll','monitor_perplexity',*[s+'_monitor_nll' for s in self.config['source_weights']]]); writer.writerows(curve_rows)
        version = [(r['event'],r['step']) for r in curves]
        if getattr(self,'curve_version',None) != version:
            self.plot(curves)
            self.curve_version = version

    def plot(self, rows):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        train=[r for r in rows if r['event']=='train']; monitor=[r for r in rows if r['event']=='validation']
        fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
        x=lambda rows:[r['main_tokens']/1e9 for r in rows]
        axes[0,0].plot(x(train),[r['loss'] for r in train],lw=1); axes[0,0].set_title('Training loss (logged updates)')
        for source in self.config['source_weights']:
            axes[0,1].plot(x(monitor),[r['metrics']['sources'][source]['nll'] for r in monitor],label=source)
        axes[0,1].plot(x(monitor),[r['metrics']['fixed_q_nll'] for r in monitor],color='black',label='fixed mixture')
        axes[0,1].set_title('BF16 monitor NLL (separate from final panel)'); axes[0,1].legend()
        axes[1,0].plot(x(train),[r['tokens_per_second'] for r in train]); axes[1,0].set_title('Training targets / second')
        axes[1,1].plot(x(train),[r['learning_rate'] for r in train]); axes[1,1].set_title('Registered learning rate')
        for ax in axes.flat:
            ax.set_xlabel('Added training targets (billions)'); ax.grid(alpha=.2)
        fig.savefig(self.work/'training_curves.tmp.png',dpi=140)
        plt.close(fig); os.replace(self.work/'training_curves.tmp.png',self.work/'training_curves.png')

    def validate_result(self, label, output):
        if label=='official_13b':
            result = read(output/'summary.json'); completion = read(output/'completed.json')
            if (result['protocol_sha256'] != sha256(self.root/self.config['official_config'])
                    or result['artifact_sha256']['model.safetensors'] != sha256(self.run/'export/model.safetensors')
                    or not completion.get('scores_calculated') or not completion.get('model_loaded')):
                raise ValueError('Official result identity/completion mismatch')
            protocol=read(self.root/self.config['official_config'])
            for task, definition in protocol['harness']['task_definitions'].items():
                scores=result['metrics']['benchmarks'][task]
                if scores['examples'] != definition['expected_examples'] or scores['split'] != definition['evaluation_split']:
                    raise ValueError('Official benchmark is incomplete')
            if result['metrics']['wikitext103']['token_count'] != 269569:
                raise ValueError('WikiText-103 coverage differs from the frozen baseline')
        else:
            from scglm.evaluate import fixed_mixture_metrics
            result=read(output/'result.json')
            expected=next(s for s in self.config['checkpoints'] if s['label']==label)
            identity=result['identity']
            source=self.root/expected['run']
            if expected['kind']=='export':
                if identity['model_sha256'] != sha256(source/'export/model.safetensors') or identity['cumulative_tokens'] != expected['cumulative_tokens']:
                    raise ValueError('Checkpoint result identity mismatch')
            elif identity['checkpoint_sha256'] != sha256(source/'milestones'/f"milestone_{expected['nominal_tokens']}.pt"):
                raise ValueError('Milestone result identity mismatch')
            if (result['status'] != 'completed' or result['label'] != label
                    or result['raw']['panel_sha256'] != self.config['development_panel']['sha256']
                    or len(result['raw']['documents']) != 768 or result['generation']['examples'] != 12
                    or any(result['checks'][k] is not True for k in
                           ('all_weights_finite','full_context_logits_finite','deterministic_repeat_matches','artifacts_unchanged'))):
                raise ValueError('Checkpoint diagnostics failed or have incomplete coverage')
            expected_docs={(r['source'],str(r['id'])):min(1024,len(r['tokens'])-1)
                           for r in jsonl(self.root/self.config['development_panel']['path'])}
            docs=result['raw']['documents']
            if (len({(r['source'],str(r['id'])) for r in docs})!=len(docs)
                    or {(r['source'],str(r['id'])):r['token_count'] for r in docs}!=expected_docs
                    or fixed_mixture_metrics(docs,self.config['source_weights'])!=result['raw']['metrics']):
                raise ValueError('Raw evaluation identities, token coverage or aggregate differ')
        if not finite(result):
            raise ValueError('Nonfinite phase result')
        return result

    def run_stage(self, label, gpu, build_command, timeout):
        if label in self.receipts:
            verify_receipt(self.receipts[label],self.registration['sha256'])
            self.validate_result(label,Path(self.receipts[label]['output']))
            return
        attempts=self.work/'attempts'/label
        attempts.mkdir(parents=True,exist_ok=True)
        # Recover a successful subprocess if the observer crashed before its receipt.
        for folder in sorted(attempts.iterdir()):
            if (folder/'command.json').exists() and (folder/'output').is_dir():
                try:
                    if read(folder/'command.json')['registration_sha256'] != self.registration['sha256']:
                        raise ValueError('Recovered output belongs to another registration')
                    self.validate_result(label,folder/'output')
                except (ValueError,KeyError,FileNotFoundError):
                    continue
                self.save_receipt(label,folder/'output')
                return
        used=len(list(attempts.iterdir()))
        for number in range(used+1,self.config['command_attempts']+1):
            self.check_frozen()
            folder=attempts/f'{number:02d}'; folder.mkdir()
            output=folder/'output'; argv=build_command(output)
            env={**os.environ,'CUDA_VISIBLE_DEVICES':'0','PYTHONPATH':str(self.root/'src'),
                 'OMP_NUM_THREADS':str(self.config['cpu_threads']),'PYTHONDONTWRITEBYTECODE':'1',
                 'HF_HUB_DISABLE_TELEMETRY':'1','SCGLM_PHASE_GPU_LOCK_FD':str(gpu.fileno())}
            write(folder/'command.json',{'argv':argv,'started_unix':time.time(),'registration_sha256':self.registration['sha256']})
            self.state('evaluating',phase=label,attempt=number); self.report()
            self.event('phase_started',phase=label,attempt=number)
            try:
                elapsed=command(argv,self.root,folder/'console.log',env,timeout,
                    lambda pid:self.state('evaluating',phase=label,attempt=number,child_pid=pid),pass_fds=(gpu.fileno(),))
                self.check_frozen(); self.validate_result(label,output)
                write(folder/'completed.json',{'elapsed_seconds':elapsed,'finished_unix':time.time()})
                self.save_receipt(label,output)
                self.event('phase_completed',phase=label,elapsed_seconds=elapsed)
                self.report()
                return
            except (RuntimeError,TimeoutError,ValueError,KeyError,FileNotFoundError):
                write(folder/'failure.json',{'error':traceback.format_exc(),'finished_unix':time.time()})
                self.event('phase_attempt_failed',phase=label,attempt=number)
        raise RuntimeError(f'{label}: exhausted {self.config["command_attempts"]} recorded attempts; see {attempts}')

    def save_receipt(self,label,output):
        receipt={'status':'completed','registration_sha256':self.registration['sha256'],
                 'output':str(output),'artifacts':artifact_hashes(output),'completed_unix':time.time()}
        write(self.work/'receipts'/f'{label}.json',receipt)
        self.receipts[label]=receipt

    def reporting_selection(self,decision):
        recommended=(self.root/decision['recommended_model']).resolve()
        body={'panel':'selection','selection_kind':'registered_13b_development_then_conditional_confirmation',
              'official_scores_used':False,'official_scores_previously_observed':True,
              'selected_model':str(recommended),'selected_model_sha256':sha256(recommended/'model.safetensors'),
              'comparison_sha256':sha256(self.source/'comparison.json'),
              'registration_sha256':self.registration['sha256'],
              'actual_selection_panels':read(self.source/'preflight.json')['panels'],
              'confirmation_opened':decision['confirmation'] is not None,
              'purpose':'Supplemental report of completed endpoints; neither intermediate diagnostics nor official scores alter the recorded raw-likelihood recommendation or the original 12B competition package.',
              'results':[{'run':str(self.root/path),'export_model_sha256':sha256(self.root/path/'export/model.safetensors')}
                         for path in (self.config['parent_run'],self.config['training_run'])]}
        path=self.work/'reporting_selection.json'
        if path.exists() and read(path)!=body:
            raise ValueError('Frozen reporting selection changed')
        write(path,body)
        return path

    def execute(self,gpu):
        self.check_frozen()
        decision=self.verify_decision()
        selection=self.reporting_selection(decision)  # Frozen before any new checkpoint diagnostics.
        for spec in self.config['checkpoints']:
            self.run_stage(spec['label'],gpu,lambda output,spec=spec:[sys.executable,'-m','scglm_phase.evaluate',
                '--config',str(self.config_path),'--label',spec['label'],'--output',str(output),'--device','cuda:0'],
                self.config['checkpoint_timeout_seconds'])
        self.run_stage('official_13b',gpu,lambda output:[sys.executable,'scripts/run_final_evaluation.py',
            '--config',str(self.root/self.config['official_config']),'--output',str(output),
            '--history-dir',str(self.work/'official_history'),'--label','13B supplemental phase report',
            '--model',str(self.run/'export'),'--selection-record',str(selection),'--device','cuda:0','--final-evaluation'],
            self.config['official_timeout_seconds'])
        self.check_frozen(); self.verify_decision()
        write(self.work/'handoff.json',{'status':'completed','decision':decision,
              'reporting_selection_sha256':sha256(selection),'stages':self.receipts,
              'model_published':False,'original_12b_package_modified':False,
              'training_actions_by_this_evaluator':0,'finished_unix':time.time()})
        self.state('completed',recommendation=decision['recommendation']); self.event('completed')
        self.report()
        artifacts={str(p.relative_to(self.work)):sha256(p) for p in sorted(self.work.rglob('*'))
                   if p.is_file() and p.name not in {'EVIDENCE_MANIFEST.json','workflow.log','observer.lock'}
                   and '__pycache__' not in p.parts}
        write(self.work/'EVIDENCE_MANIFEST.json',{'files':artifacts,'note':'Model weights remain in their source runs; this is an evaluation evidence archive.'})

    def watch(self,once=False):
        self.freeze()
        self.validate_prior_evidence()
        if (self.work/'handoff.json').exists() and (self.work/'EVIDENCE_MANIFEST.json').exists():
            self.check_frozen(); self.verify_decision()
            for name,receipt in self.receipts.items():
                self.validate_result(name,Path(receipt['output']))
            manifest=read(self.work/'EVIDENCE_MANIFEST.json')
            for path,checksum in manifest['files'].items():
                if sha256(self.work/path)!=checksum:
                    raise ValueError(f'Completed evidence archive changed: {path}')
            print('Evaluation pipeline already completed and verified',flush=True)
            return
        self.event('observer_started',mode='observe_once' if once else 'watch',pid=os.getpid(),registration_sha256=self.registration['sha256'])
        while True:
            self.check_frozen()
            training,supervisor=read(self.run/'status.json'),read(self.source/'status.json')
            readiness_state,reason=readiness(training,supervisor,(self.source/'comparison.json').exists(),time.time(),self.config['stale_seconds'])
            if readiness_state=='blocked':
                raise RuntimeError(reason)
            if time.time()-self.registration['registered_unix'] > self.config['max_watch_seconds']:
                raise TimeoutError('Observer deadline reached; training is left untouched')
            self.state('ready' if readiness_state=='ready' else 'watching_training',detail=reason)
            self.report()
            if once:
                return
            if readiness_state=='ready':
                try:
                    with lease(self.root/self.config['gpu_lock']) as gpu:
                        uuid=subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=uuid',
                                                      '--format=csv,noheader'],text=True).strip()
                        if uuid!=self.config['gpu_uuid']:
                            raise ValueError('Physical GPU 0 does not match the assigned GPU UUID')
                        occupied=subprocess.check_output(['nvidia-smi','-i','0','--query-compute-apps=pid',
                                                          '--format=csv,noheader'],text=True).strip()
                        if not occupied:
                            self.execute(gpu)
                            return
                        self.state('waiting_gpu',detail='GPU 0 still has an active compute process')
                except BlockingIOError:
                    self.state('waiting_gpu',detail='GPU lease is still held; no concurrent evaluation')
            time.sleep(self.config['poll_seconds'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=Path('configs/phase_evaluation.json'))
    group=parser.add_mutually_exclusive_group()
    group.add_argument('--once',action='store_true',help='Record current phase report without GPU inference')
    group.add_argument('--status',action='store_true',help='Print last observer state without changing files')
    group.add_argument('--watch',action='store_true',help='Watch and run the queued evaluations (default)')
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[2]
    pipeline=Pipeline(root,args.config)
    if args.status:
        print(json.dumps(read(pipeline.work/'state.json'),indent=2))
        return
    def stopped(signum,frame):
        raise KeyboardInterrupt(f'Observer received signal {signum}; trainer is unaffected')
    signal.signal(signal.SIGTERM,stopped)
    with lease(pipeline.work/'observer.lock'):
        try:
            pipeline.watch(once=args.once)
        except BaseException:
            pipeline.state('failed',error=traceback.format_exc())
            pipeline.event('failed',error=traceback.format_exc())
            try:
                pipeline.report()
            except Exception:
                pass
            raise


if __name__=='__main__':
    main()
