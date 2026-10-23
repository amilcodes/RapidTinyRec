"""Paired beam-50 benchmark on a frozen cohort of distinct MiniOneRec users.

Paired CUDA benchmark of the released generation path. One measured request per user and
variant, with per-shape warm-up excluded.
Resume uses atomic per-user records and verifies the input/code manifest.
The default experiment varies PrefillOnce with the released processor.
The bulk_constraints experiment uses PrefillOnce in both variants and varies
only the constraint processor. Raw variant keys remain baseline/shared for
compatibility; manifest.variant_definitions records their precise meanings.
"""
import argparse
import collections
import contextlib
import csv
import datetime
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ.setdefault('HF_HOME', str(Path(os.environ.get('RAPIDTINYREC_OUTPUT_DIR', Path('results')))/'hf_cache'))
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, LogitsProcessorList
from .source.data import EvalSidDataset
from .source.LogitProcessor import ConstrainedLogitsProcessor
from .shared_prefill import PrefillOnce
from .bulk_constraints import BulkPrefixConstrainedLogitsProcessor


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        while chunk := f.read(8*1024*1024):
            h.update(chunk)
    return h.hexdigest()


def save(path, obj):
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False))
    tmp.replace(path)


def announce(obj):
    print(json.dumps(obj, allow_nan=False), flush=True)


def percentile(values, q):
    ordered = sorted(values)
    index = (len(ordered)-1)*q
    lo = math.floor(index); hi = math.ceil(index)
    return ordered[lo]+(ordered[hi]-ordered[lo])*(index-lo)


def distribution(values):
    return {'n': len(values), 'median_s': statistics.median(values),
            'p95_s': percentile(values, .95), 'min_s': min(values),
            'max_s': max(values), 'mean_s': statistics.mean(values)}


def compare(a, b):
    sa = [tuple(t) for t in a['token_ids']]
    sb = [tuple(t) for t in b['token_ids']]
    assert len(sa) == len(sb) == 50
    assert len(set(sa)) == len(set(sb)) == 50, 'Expected 50 distinct generated SIDs'
    ma = {s: (i, a['sequence_logp'][i]) for i, s in enumerate(sa)}
    mb = {s: (i, b['sequence_logp'][i]) for i, s in enumerate(sb)}
    common = set(ma) & set(mb)
    return {'ranked_tokens_equal': sa == sb,
            'same_top50_set': set(sa) == set(sb), 'overlap_count': len(common),
            'exact_rank_matches': sum(x == y for x, y in zip(sa, sb)),
            'top1_equal': sa[0] == sb[0],
            'sequence_scores_equal_in_rank_order': a['sequence_logp'] == b['sequence_logp'],
            'transition_scores_equal_in_rank_order': a['transition_logp'] == b['transition_logp'],
            'max_abs_score_difference_common': max((abs(ma[s][1]-mb[s][1]) for s in common), default=None),
            'changed_ranks': [{'token_ids': list(s), 'baseline_rank': ma[s][0]+1, 'shared_rank': mb[s][0]+1,
                               'baseline_score': ma[s][1], 'shared_score': mb[s][1]}
                              for s in sa if s in mb and ma[s][0] != mb[s][0]],
            'baseline_only': [list(s) for s in sa if s not in mb],
            'shared_only': [list(s) for s in sb if s not in ma]}


def summarize(records, manifest, complete=False):
    baseline = distribution([r['runs']['baseline']['elapsed_s'] for r in records])
    shared = distribution([r['runs']['shared']['elapsed_s'] for r in records])
    checks = [r['comparison'] for r in records]
    return {'complete': complete, 'updated_at': utc_now(), 'completed_users': len(records),
            'planned_users': len(manifest['users']), 'manifest_sha256': manifest['manifest_sha256'],
            'baseline': baseline, 'shared': shared,
            'median_latency_ratio': baseline['median_s']/shared['median_s'],
            'p95_latency_ratio': baseline['p95_s']/shared['p95_s'],
            'median_paired_speedup': statistics.median(r['runs']['baseline']['elapsed_s']/r['runs']['shared']['elapsed_s'] for r in records),
            'all50_rankings_match_users': sum(c['ranked_tokens_equal'] for c in checks),
            'same_top50_set_users': sum(c['same_top50_set'] for c in checks),
            'same_top1_users': sum(c['top1_equal'] for c in checks),
            'same_sequence_scores_users': sum(c['sequence_scores_equal_in_rank_order'] for c in checks),
            'same_transition_scores_users': sum(c['transition_scores_equal_in_rank_order'] for c in checks),
            'matching_rank_positions': sum(c['exact_rank_matches'] for c in checks),
            'total_rank_positions': 50*len(records),
            'mismatching_cohort_indices': [r['cohort_index'] for r in records if not r['comparison']['ranked_tokens_equal']],
            'max_abs_score_difference_common': max(c['max_abs_score_difference_common'] or 0 for c in checks),
            'latency_by_first_variant': {
                first: {v: distribution([r['runs'][v]['elapsed_s'] for r in records if r['order'][0] == first])
                        for v in ('baseline', 'shared')}
                for first in ('baseline', 'shared') if any(r['order'][0] == first for r in records)},
            'latency_by_prompt_length': {
                str(length): {v: distribution([r['runs'][v]['elapsed_s'] for r in records if r['prompt_tokens'] == length])
                              for v in ('baseline', 'shared')}
                for length in sorted(set(r['prompt_tokens'] for r in records))}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--users', type=int, default=500)
    parser.add_argument('--dtype', choices=('bfloat16', 'float32'), default='float32')
    parser.add_argument('--experiment', choices=('shared_prefill', 'bulk_constraints'), default='shared_prefill')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--release-root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, default=Path(os.environ.get('RAPIDTINYREC_OUTPUT_DIR', Path('results'))))
    args = parser.parse_args()
    model_dtype = getattr(torch, args.dtype)
    torch.manual_seed(42)
    torch.set_num_threads(4)
    assert torch.cuda.is_available(), 'CUDA GPU allocation required'
    assert torch.cuda.get_device_capability(0)[0] >= 8, 'Native BF16 requires Ampere or newer'
    assert torch.cuda.is_bf16_supported()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device('cuda:0')
    expected = json.loads((ROOT/'expected.json').read_text())
    assert args.users == expected['num_users'], 'This experiment uses the frozen 500-user cohort'
    release = args.release_root.resolve()
    outdir = args.output_root.resolve()/f'{args.experiment}_{args.users}_users'
    outdir.mkdir(parents=True, exist_ok=True)
    for rel, want in expected['release_sha256'].items():
        assert digest(release/rel) == want, f'Release hash mismatch: {rel}'
    for rel, want in expected['source_sha256'].items():
        assert digest(ROOT/rel) == want, f'Source hash mismatch: {rel}'
    requests_dir = outdir/'users'
    requests_dir.mkdir(exist_ok=True)
    if list(requests_dir.glob('*.json')) and not args.resume:
        raise RuntimeError('Existing results: use --resume rather than overwrite')
    ckpt = release/'Office_ckpt'
    source_csv = release/'Amazon/test/Office_Products_5_2016-10-2018-11.csv'
    catalog_file = release/'Amazon/info/Office_Products_5_2016-10-2018-11.txt'
    all_rows = list(csv.DictReader(source_csv.open()))
    selected = []
    seen = set()
    for source_index, row in enumerate(all_rows):
        if row['user_id'] not in seen:
            selected.append((source_index, row))
            seen.add(row['user_id'])
        if len(selected) == args.users:
            break
    assert len(selected) == args.users
    cohort_csv = outdir/'cohort.csv'
    with cohort_csv.open('w') as f:
        writer = csv.DictWriter(f, fieldnames=all_rows[0].keys())
        writer.writeheader()
        writer.writerows(row for _, row in selected)
    tok = AutoTokenizer.from_pretrained(ckpt, local_files_only=True)
    tok.pad_token = tok.eos_token
    tok.padding_side = 'left'
    dataset = EvalSidDataset(str(cohort_csv), tok, max_len=2560, test=True,
                            category='office products', K=0, seed=42)
    users = [{'cohort_index': i, 'source_row_index': source_index, 'user_id': row['user_id'],
              'target_sid': row['item_sid'], 'prompt_token_ids': dataset[i]['input_ids'],
              'prompt_tokens': len(dataset[i]['input_ids'])}
             for i, (source_index, row) in enumerate(selected)]
    cohort_hash = hashlib.sha256(json.dumps(users, sort_keys=True).encode()).hexdigest()
    assert cohort_hash == expected['users_sha256'], 'Cohort/prompt tokens differ from frozen cohort'
    provenance_files = [Path(__file__), ROOT/'shared_prefill.py', ROOT/'source/data.py',
                        ROOT/'source/LogitProcessor.py', ROOT/'expected.json']
    if args.experiment == 'bulk_constraints':
        provenance_files.append(ROOT/'bulk_constraints.py')
    variant_definitions = {
        'baseline': {'shared_prefill': args.experiment == 'bulk_constraints',
                     'constraint_processor': 'ConstrainedLogitsProcessor'},
        'shared': {'shared_prefill': True,
                   'constraint_processor': 'BulkPrefixConstrainedLogitsProcessor'
                       if args.experiment == 'bulk_constraints' else 'ConstrainedLogitsProcessor'}}
    manifest = {'selection': 'First occurrence of each of the first N distinct user IDs, in test CSV order; one example per user.',
        'experiment': args.experiment, 'variant_definitions': variant_definitions,
        'hardware': torch.cuda.get_device_name(0), 'device': 'CUDA', 'dtype': args.dtype,
        'gpu_total_memory_bytes': torch.cuda.get_device_properties(0).total_memory,
        'cuda_runtime': torch.version.cuda, 'device_capability': list(torch.cuda.get_device_capability(0)),
        'gpu_info': subprocess.check_output(['nvidia-smi', '--query-gpu=name,uuid,driver_version,memory.total', '--format=csv,noheader'], text=True),
        'cohort_sha256': cohort_hash, 'expected_release': expected,
        'attention': 'PyTorch SDPA', 'num_beams': 50, 'num_return_sequences': 50,
        'length_penalty': 0.0, 'max_new_tokens': 8, 'request_batch': 1, 'use_cache': True,
        'output_scores': True, 'threads': 4, 'seed': 42, 'users': users,
        'cuda_settings': {'allow_tf32': torch.backends.cuda.matmul.allow_tf32,
                          'allow_bf16_reduced_precision_reduction': torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction},
        'versions': {name: importlib.metadata.version(name) for name in ['torch', 'transformers', 'tokenizers']},
        'sha256': {str(p.relative_to(ROOT)): digest(p) for p in provenance_files},
        'protocol': 'Warm both variants once for each prompt length. Then run one timed request per variant per user; alternate order by cohort index. Synchronize CUDA before and after generation. Exclude model load, input tokenization, input transfer, warmup, output comparison and persistence. Include generation setup, score outputs and cache expansion. No prompt cache reused between requests. Preserve all measured pairs, including mismatches and latency outliers.',
        'p95_method': 'Linear interpolation at index (N-1)*0.95 in ascending latency order.',
        'scope': 'Latency distribution across 500 distinct Office test users on one allocated NVIDIA GPU; not an M4 latency comparison, concurrent serving throughput, or all-user equivalence proof.'}
    manifest['manifest_sha256'] = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    manifest_path = outdir/'manifest.json'
    if manifest_path.exists():
        assert json.loads(manifest_path.read_text()) == manifest, 'Resume manifest changed'
    else:
        save(manifest_path, manifest)
    records = []
    for path in sorted(requests_dir.glob('*.json')):
        record = json.loads(path.read_text())
        assert record['manifest_sha256'] == manifest['manifest_sha256']
        assert record['cohort_index'] == len(records), 'Expected contiguous completed prefix'
        records.append(record)
    announce({'stage': 'cohort_ready', 'distinct_users': len(users), 'already_completed': len(records),
              'prompt_length_counts': dict(collections.Counter(u['prompt_tokens'] for u in users)),
              'manifest_sha256': manifest['manifest_sha256']})
    if len(records) == len(users):
        save(outdir/'summary.json', summarize(records, manifest, complete=True))
        return

    constraints = collections.defaultdict(set)
    for line in catalog_file.read_text().splitlines():
        sid = line.split('\t')[0].strip()
        ids = tok('### Response:\n'+sid+'\n').input_ids+[tok.eos_token_id]
        assert all(u['prompt_token_ids'][-3:] == ids[:3] for u in users)
        for j in range(3, len(ids)):
            constraints[tuple(ids[:j] if j == 3 else ids[3:j])].add(ids[j])
    constraints = {k: list(v) for k, v in constraints.items()}

    def allowed(batch_id, ids):
        return constraints.get(tuple(ids), [])

    model = AutoModelForCausalLM.from_pretrained(ckpt, torch_dtype=model_dtype,
        local_files_only=True, attn_implementation='sdpa').to(device).eval()
    assert all(p.dtype == model_dtype for p in model.parameters()), 'Unexpected parameter dtype'
    model.config.pad_token_id = tok.eos_token_id
    model.config.eos_token_id = tok.eos_token_id
    model.config.bos_token_id = tok.bos_token_id
    torch.cuda.synchronize()
    announce({'stage': 'model_loaded', 'at': utc_now(), 'dtype': str(model_dtype),
              'cuda_allocated_bytes': torch.cuda.memory_allocated(),
              'cuda_reserved_bytes': torch.cuda.memory_reserved()})

    @torch.inference_mode()
    def generate(variant, input_ids, attention_mask):
        cfg = GenerationConfig(max_new_tokens=8, length_penalty=0.0, num_beams=50,
            num_return_sequences=50, pad_token_id=tok.eos_token_id, eos_token_id=tok.eos_token_id,
            bos_token_id=tok.bos_token_id, do_sample=False, top_k=None, top_p=None, use_cache=True)
        processor_class = (BulkPrefixConstrainedLogitsProcessor
            if args.experiment == 'bulk_constraints' and variant == 'shared'
            else ConstrainedLogitsProcessor)
        proc = processor_class(allowed, 50, base_model=str(ckpt), eos_token_id=tok.eos_token_id)
        context = (PrefillOnce(model, 50)
            if variant_definitions[variant]['shared_prefill'] else contextlib.nullcontext())
        with context:
            return model.generate(input_ids, attention_mask=attention_mask, generation_config=cfg,
                do_sample=False, return_dict_in_generate=True, output_scores=True,
                logits_processor=LogitsProcessorList([proc]))

    def unpack(output, prompt_length):
        transitions = model.compute_transition_scores(output.sequences, output.scores,
            output.beam_indices, normalize_logits=False).float().cpu().tolist()
        return {'token_ids': output.sequences[:, prompt_length:].cpu().tolist(),
                'sequence_logp': output.sequences_scores.float().cpu().tolist(),
                'transition_logp': transitions}

    representatives = {}
    for user in users[len(records):]:
        representatives.setdefault(user['prompt_tokens'], user)
    for length, user in sorted(representatives.items()):
        x = torch.tensor([user['prompt_token_ids']], device=device)
        mask = torch.ones_like(x)
        for variant in ('baseline', 'shared'):
            start = time.perf_counter()
            generated = generate(variant, x, mask)
            torch.cuda.synchronize()
            del generated
            announce({'stage': 'warmup', 'prompt_tokens': length, 'variant': variant,
                      'excluded_seconds': time.perf_counter()-start,
                      'cuda_allocated_bytes': torch.cuda.memory_allocated(),
                      'cuda_reserved_bytes': torch.cuda.memory_reserved()})

    session_id = utc_now()
    started = time.perf_counter()
    session_completed = 0
    for user in users[len(records):]:
        i = user['cohort_index']
        order = ['baseline', 'shared'] if i % 2 == 0 else ['shared', 'baseline']
        x = torch.tensor([user['prompt_token_ids']], device=device)
        mask = torch.ones_like(x)
        runs = {}
        for variant in order:
            torch.cuda.synchronize()
            start = time.perf_counter()
            generated = generate(variant, x, mask)
            torch.cuda.synchronize()
            elapsed = time.perf_counter()-start
            runs[variant] = {'elapsed_s': elapsed, **unpack(generated, user['prompt_tokens'])}
            del generated
        comparison = compare(runs['baseline'], runs['shared'])
        record = {'cohort_index': i, 'source_row_index': user['source_row_index'],
                  'user_id': user['user_id'], 'prompt_tokens': user['prompt_tokens'],
                  'target_sid': user['target_sid'], 'order': order, 'session_id': session_id,
                  'completed_at': utc_now(), 'manifest_sha256': manifest['manifest_sha256'],
                  'runs': runs, 'comparison': comparison,
                  'cuda_current_allocated_bytes': torch.cuda.memory_allocated(),
                  'cuda_reserved_bytes': torch.cuda.memory_reserved()}
        save(requests_dir/f'{i:04d}.json', record)
        records.append(record)
        session_completed += 1
        current = summarize(records, manifest, complete=len(records) == len(users))
        save(outdir/'summary.json', current)
        announce({'stage': 'user_complete', 'completed': len(records), 'planned': len(users),
                  'cohort_index': i, 'prompt_tokens': user['prompt_tokens'],
                  'baseline_s': runs['baseline']['elapsed_s'], 'shared_s': runs['shared']['elapsed_s'],
                  'ranked_tokens_equal': comparison['ranked_tokens_equal'],
                  'same_sequence_scores': comparison['sequence_scores_equal_in_rank_order'],
                  'mismatch_users_so_far': len(current['mismatching_cohort_indices']),
                  'median_baseline_s': current['baseline']['median_s'],
                  'median_shared_s': current['shared']['median_s'],
                  'eta_seconds': (time.perf_counter()-started)/session_completed*(len(users)-len(records))})
    announce({'stage': 'finished', **summarize(records, manifest, complete=True)})


if __name__ == '__main__':
    main()
