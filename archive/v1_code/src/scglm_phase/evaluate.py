"""Same-panel likelihood and fixed completion diagnostics, with no optimization."""
from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path
import time

from .common import artifact_hashes, jsonl, lease, read, sha256, write


def repetition(text):
    words = text.casefold().split()
    grams = [tuple(words[i:i+4]) for i in range(max(0, len(words)-3))]
    repeated_fraction = 0.0 if not grams else 1 - len(set(grams)) / len(grams)
    loop = any(words[i:i+4] * 4 == words[i:i+16] for i in range(max(0, len(words)-15)))
    return {'duplicate_4gram_fraction': repeated_fraction, 'fourfold_4gram_loop': loop}


def generation_metrics(rows):
    if not rows or len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Empty or duplicate generation diagnostic IDs')
    n = len(rows)
    duration = sum(r['elapsed_seconds'] for r in rows)
    return {'examples': n, 'nonempty_rate': sum(bool(r['response'].strip()) for r in rows)/n,
            'eos_rate': sum(r['ended_eos'] for r in rows)/n,
            'loop_rate': sum(repetition(r['response'])['fourfold_4gram_loop'] for r in rows)/n,
            'mean_duplicate_4gram_fraction': sum(repetition(r['response'])['duplicate_4gram_fraction'] for r in rows)/n,
            'generated_tokens': sum(r['generated_tokens'] for r in rows),
            'generation_seconds': duration,
            'tokens_per_second': sum(r['generated_tokens'] for r in rows)/duration if duration > 0 else None}


def load_target(root, spec):
    import torch
    from transformers import AutoTokenizer
    from scglm.model import build_model, load_model, audit_parameters
    run = root / spec['run']
    record, status = read(run/'run.json'), read(run/'status.json')
    if status['status'] != 'completed':
        raise ValueError('Checkpoint assessment waits for the parent run to complete')
    if spec['kind'] == 'export':
        folder = run/'export'
        before = artifact_hashes(folder)
        if spec.get('model_sha256') and before['model.safetensors'] != spec['model_sha256']:
            raise ValueError('Wrong endpoint weights')
        prov = read(folder/'training_provenance.json')
        cumulative = prov.get('cumulative_tokens', prov['main_tokens'])
        if (cumulative != spec['cumulative_tokens'] or prov['main_tokens'] != status['main_tokens']
                or prov['science_digest'] != record['science_digest']):
            raise ValueError('Endpoint provenance does not match completed run')
        model = load_model(folder)
        tokenizer = AutoTokenizer.from_pretrained(str(folder), local_files_only=True)
        identity = {'kind': 'export', 'model_sha256': before['model.safetensors'],
                    'artifact_sha256': before, 'run': str(run), 'cumulative_tokens': cumulative}
        verify = lambda: before == artifact_hashes(folder)
    elif spec['kind'] == 'milestone':
        path = run/'milestones'/f"milestone_{spec['nominal_tokens']}.pt"
        marker = read(path.with_suffix('.json'))
        if sha256(path) != marker['sha256'] or Path(marker['path']).resolve() != path.resolve():
            raise ValueError('Milestone checksum/path mismatch')
        # Only our checksummed local torch checkpoint is deserialized.
        payload = torch.load(path, map_location='cpu', weights_only=False)
        step_tokens = record['config']['sequence_length'] * record['config']['batch_sequences']
        cumulative = record['parent']['cumulative_tokens'] + payload['main_tokens']
        if (payload['science_digest'] != record['science_digest']
                or payload['step'] != marker['step'] or payload['main_tokens'] != marker['main_tokens']
                or payload['main_tokens'] != payload['step'] * step_tokens
                or cumulative != marker['cumulative_tokens']
                or not spec['nominal_tokens'] <= cumulative < spec['nominal_tokens'] + step_tokens):
            raise ValueError('Milestone accounting mismatch')
        model = build_model(record['model_config'])
        model.load_state_dict(payload['model'], strict=True)
        del payload
        folder = root/spec['tokenizer_export']
        tokenizer = AutoTokenizer.from_pretrained(str(folder), local_files_only=True)
        identity = {'kind': 'milestone', 'checkpoint_sha256': marker['sha256'], 'run': str(run),
                    'checkpoint': str(path), 'cumulative_tokens': cumulative, 'step': marker['step']}
        verify = lambda: sha256(path) == marker['sha256']
    else:
        raise ValueError('Unsupported checkpoint kind')
    if sha256(folder/'tokenizer.json') != spec['tokenizer_sha256']:
        raise ValueError('Tokenizer differs from frozen diagnostic protocol')
    audit = audit_parameters(model)
    if (audit['total_unique_parameters'] != 46346752 or len(tokenizer) != model.config.vocab_size
            or model.config.max_position_embeddings != 1024):
        raise ValueError('Model/tokenizer architecture mismatch')
    if not all(bool(torch.isfinite(p).all()) for p in model.parameters()):
        raise ValueError('Nonfinite model weights')
    return model, tokenizer, identity, audit, verify


def evaluate(root, config, spec, output, *, device):
    import torch
    from scglm.evaluate import evaluate_documents, fixed_mixture_metrics

    started = time.monotonic()
    torch.set_num_threads(config['cpu_threads'])
    torch.manual_seed(config['seed'])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model, tokenizer, identity, audit, verify = load_target(root, spec)
    model = model.to(device).eval()
    if device.startswith('cuda'):
        torch.cuda.reset_peak_memory_stats()
    panel_path = root/config['development_panel']['path']
    if sha256(panel_path) != config['development_panel']['sha256']:
        raise ValueError('Development panel changed')
    panel = jsonl(panel_path)
    raw_started = time.monotonic()
    documents = evaluate_documents(model, panel, context_length=1024, stride=512,
                                   max_predictable_tokens=1024, device=device)
    raw_seconds = time.monotonic() - raw_started
    raw = fixed_mixture_metrics(documents, config['source_weights'])
    # Reuse exactly the previously inspected 12 prompts, preserving all responses.
    module_spec = importlib.util.spec_from_file_location('frozen_smoke', root/config['prompt_script'])
    prompts = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(prompts)

    def sync():
        if device.startswith('cuda'):
            torch.cuda.synchronize()

    @torch.inference_mode()
    def generate(prompt, limit=64):
        values = tokenizer.encode(prompt, add_special_tokens=False)
        if not values or len(values) + limit > model.config.max_position_embeddings:
            raise ValueError('Invalid prompt length')
        ids = torch.tensor([values], device=device)
        sync(); begin = time.monotonic()
        result = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), do_sample=False,
                                max_new_tokens=limit, use_cache=True, pad_token_id=tokenizer.pad_token_id,
                                eos_token_id=tokenizer.eos_token_id)
        sync(); elapsed = time.monotonic() - begin
        generated = result[0, len(values):].tolist()
        return {'response': tokenizer.decode(generated, skip_special_tokens=True), 'generated_ids': generated,
                'generated_tokens': len(generated), 'elapsed_seconds': elapsed,
                'ended_eos': bool(generated and generated[-1] == tokenizer.eos_token_id)}

    generate('The', 4)
    rows = [{'id': key, 'prompt': prompt, **generate(prompt)} for key, prompt in prompts.CASES]
    deterministic = generate(prompts.CASES[0][1])['generated_ids'] == rows[0]['generated_ids']
    values = tokenizer.encode('The quick brown fox jumps over the lazy dog. ', add_special_tokens=False)
    context = model.config.max_position_embeddings
    ids = torch.tensor([(values*(context//len(values)+1))[:context]], device=device)
    with torch.inference_mode():
        logits = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).logits
    logits_finite = bool(torch.isfinite(logits).all())
    unchanged = verify()
    if not (deterministic and logits_finite and unchanged):
        raise ValueError('Deterministic replay, numerical or identity check failed')
    output.mkdir(parents=True, exist_ok=False)
    (output/'responses.jsonl').write_text(''.join(__import__('json').dumps(r)+'\n' for r in rows))
    (output/'GENERATIONS.md').write_text('# Fixed completion diagnostics\n\nIllustrative; no representative accuracy claim.\n\n' +
        '\n\n'.join(f"## {r['id']}\n\nPrompt:\n```text\n{r['prompt']}\n```\n\nResponse:\n```text\n{r['response']}\n```" for r in rows)+'\n')
    result = {'status': 'completed', 'label': spec['label'], 'identity': identity,
              'raw': {'panel_sha256': config['development_panel']['sha256'], 'metrics': raw,
                      'documents': documents, 'elapsed_seconds': raw_seconds, 'dtype': 'float32'},
              'generation': generation_metrics(rows), 'decode': {'max_new_tokens':64, 'do_sample':False,
                  'chat_template':False, 'use_cache':True, 'dtype':'float32'},
              'checks': {'parameter_audit': audit, 'all_weights_finite': True, 'full_context_logits_finite': logits_finite,
                         'deterministic_repeat_matches': deterministic, 'artifacts_unchanged': unchanged},
              'elapsed_seconds': time.monotonic()-started,
              'peak_gpu_allocated_bytes': torch.cuda.max_memory_allocated() if device.startswith('cuda') else None,
              'limitations': ['Completion prompts were inspected on 12B before this registration; descriptive only.',
                  'Repetition and EOS measure output behavior, not factual knowledge.',
                  'Single-process generation timing is not a serving benchmark.',
                  'Development comparisons are diagnostic and do not select an intermediate checkpoint.']}
    write(output/'result.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--label', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=['cpu','cuda:0'], default='cpu')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    config = read(args.config)
    matches = [s for s in config['checkpoints'] if s['label'] == args.label]
    if len(matches) != 1:
        raise ValueError('Checkpoint label must be unique')
    if args.device.startswith('cuda'):
        if os.environ.get('CUDA_VISIBLE_DEVICES') != '0':
            raise ValueError('Expose only assigned physical GPU 0')
        inherited = os.environ.get('SCGLM_PHASE_GPU_LOCK_FD')
        with lease(root/config['gpu_lock'], inherited_fd=int(inherited) if inherited else None):
            evaluate(root, config, matches[0], args.output, device=args.device)
    else:
        evaluate(root, config, matches[0], args.output, device=args.device)


if __name__ == '__main__':
    main()
