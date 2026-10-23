"""One isolated phase of shared-input versus physical-prefix cache comparison."""
import argparse
import collections
import contextlib
import functools
import hashlib
import json
from pathlib import Path
import time
from unittest.mock import patch

from . import benchmark as b
from .compiled_forward import CompiledForward
from .decode_replay import DecodeReplayPool
from .gpu_constraints import GpuSidConstraintTables,GpuSidConstrainedLogitsProcessor

torch=b.torch


def select_users(reference_users, full_cohort):
    assert len(reference_users)==500
    lengths=sorted({u['prompt_tokens'] for u in reference_users})
    if full_cohort:
        assert len(lengths)==10
        return reference_users,[next(u for u in reference_users if u['prompt_tokens']==n) for n in lengths]
    users=[u for length in (lengths[0],lengths[-1])
           for u in [x for x in reference_users if x['prompt_tokens']==length][:2]]
    assert len(users)==4
    return users,users


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--release-root',type=Path,required=True)
    ap.add_argument('--reference-results',type=Path,required=True)
    ap.add_argument('--output-root',type=Path,required=True)
    ap.add_argument('--full-cohort',action='store_true')
    ap.add_argument('--phase',type=int,choices=range(4),required=True)
    args=ap.parse_args()
    variant=('arena_replay','prefix_replay','prefix_replay','arena_replay')[args.phase]
    half=0 if args.phase<2 else 1
    assert torch.cuda.is_available() and torch.cuda.device_count()==1
    assert torch.cuda.get_device_name(0)=='NVIDIA RTX A6000'
    torch.manual_seed(42);torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    expected=json.loads((b.ROOT/'expected.json').read_text())
    reference=json.loads((args.reference_results/'manifest.json').read_text())
    assert reference['experiment']=='gqa_three_way' and reference['dtype']=='float32'
    content={k:v for k,v in reference.items() if k!='manifest_sha256'}
    assert hashlib.sha256(json.dumps(content,sort_keys=True).encode()).hexdigest()==reference['manifest_sha256']
    assert hashlib.sha256(json.dumps(reference['users'],sort_keys=True).encode()).hexdigest()==expected['users_sha256']
    for name,want in expected['release_sha256'].items():assert b.digest(args.release_root/name)==want,name
    users,warm_users=select_users(reference['users'],args.full_cohort)
    out=args.output_root/f'phase_{args.phase}'
    out.mkdir(parents=True,exist_ok=True)
    assert not (out/'manifest.json').exists(),'Use a new output directory'
    (out/'users').mkdir()
    tok=b.AutoTokenizer.from_pretrained(args.release_root/'Office_ckpt',local_files_only=True)
    tok.pad_token=tok.eos_token;tok.padding_side='left'
    constraints=collections.defaultdict(set)
    for line in (args.release_root/'Amazon/info/Office_Products_5_2016-10-2018-11.txt').read_text().splitlines():
        ids=tok('### Response:\n'+line.split('\t')[0].strip()+'\n').input_ids+[tok.eos_token_id]
        assert all(u['prompt_token_ids'][-3:]==ids[:3] for u in users)
        for j in range(3,len(ids)):constraints[tuple(ids[:j] if j==3 else ids[3:j])].add(ids[j])
    tables=GpuSidConstraintTables({k:list(v) for k,v in constraints.items()},users[0]['prompt_token_ids'][-3:],198,tok.eos_token_id,'cuda')
    model=b.AutoModelForCausalLM.from_pretrained(args.release_root/'Office_ckpt',torch_dtype=torch.float32,
        local_files_only=True,attn_implementation='sdpa').to('cuda').eval()
    assert model.config.model_type=='qwen2' and not model.model.has_sliding_layers
    assert set(model.config.layer_types)=={'full_attention'}
    model.config.pad_token_id=tok.eos_token_id;model.config.eos_token_id=tok.eos_token_id;model.config.bos_token_id=tok.bos_token_id
    assert all(p.dtype==torch.float32 for p in model.parameters())
    inputs={u['cohort_index']:torch.tensor([u['prompt_token_ids']],device='cuda') for u in users}
    masks={i:torch.ones_like(x) for i,x in inputs.items()}
    manifest={'experiment':'shared_prefix_500' if args.full_cohort else 'shared_prefix_pilot',
        'phase':args.phase,'variant':variant,'half':half,
        'users':users,'dtype':'float32','tf32':False,'hardware':torch.cuda.get_device_name(0),
        'cohort_sha256':expected['users_sha256'],'reference_manifest_sha256':reference['manifest_sha256'],
        'versions':{n:b.importlib.metadata.version(n) for n in ['torch','transformers','tokenizers']},
        'sha256':{n:b.digest(b.ROOT/n) for n in ['benchmark_shared_prefix.py','shared_prefix.py','prefix_replay.py','input_arena.py','decode_replay.py',
            'compiled_forward.py','shared_prefill.py','gpu_constraints.py','expected.json','direct_reorder.py']},
        'protocol':'Four isolated processes on one GPU allocation: arena half0, prefix half0, prefix half1, arena half1. Full cohort halves use even/odd cohort positions; pilot halves use repeats 0-2/3-5. Every phase warms all selected history lengths. FP32 beam50, all five scored tokens, full-vocabulary normalization. Same Qwen2 weights and native GQA operations; candidate keeps a single-row physical prefix and temporarily materializes dense exact-length attention K/V per layer. Only suffixes are retained and reordered. Cold compilation, capture, warmup and output extraction excluded; prefill, staging, replay, constraints and cache reordering included. No per-request process switching.'}
    b.save(out/'manifest.json',manifest)
    compiler=CompiledForward(model,out,b.announce)
    from .input_arena import SharedInputArena
    from .direct_reorder import ReorderIntoReplay
    from .shared_prefix import PrefixArena
    from .prefix_replay import PrefixReplayPool, ReorderSuffix
    prefix_mode=variant=='prefix_replay'
    maximum=max(u['prompt_tokens'] for u in users)
    arena=PrefixArena(maximum) if prefix_mode else SharedInputArena(maximum+3)
    pool_class=PrefixReplayPool if prefix_mode else DecodeReplayPool
    pool=pool_class(compiler.compiled,max_graphs=40 if args.full_cohort else 8,arena=arena)
    direct=ReorderSuffix(pool) if prefix_mode else ReorderIntoReplay(pool)

    @torch.inference_mode()
    def generate(user,replay,direct_reorder=False):
        calls=0
        @functools.wraps(compiler.eager)
        def forward(*a,**kw):
            nonlocal calls
            assert not a
            step=calls;calls+=1
            ids,cache=kw['input_ids'],kw['past_key_values']
            assert ids.shape==((1,user['prompt_tokens']) if step==0 else (50,1))
            assert cache.get_seq_length()==(0 if step==0 else user['prompt_tokens']+step-1)
            kw['attention_mask']={'full_attention':None}
            return pool(**kw) if replay and step else compiler.compiled(**kw)
        cfg=b.GenerationConfig(max_new_tokens=8,length_penalty=0.,num_beams=50,num_return_sequences=50,
            pad_token_id=tok.eos_token_id,eos_token_id=tok.eos_token_id,bos_token_id=tok.bos_token_id,
            do_sample=False,top_k=None,top_p=None,use_cache=True)
        with patch.object(model,'forward',forward),b.PrefillOnce(model,50,cache_expander=arena.start_request if prefix_mode and replay else None), \
                (direct.install(model) if direct_reorder else contextlib.nullcontext()):
            output=model.generate(inputs[user['cohort_index']],attention_mask=masks[user['cohort_index']],
                generation_config=cfg,return_dict_in_generate=True,output_scores=True,
                logits_processor=b.LogitsProcessorList([GpuSidConstrainedLogitsProcessor(tables)]))
        assert calls==5
        return output

    def unpack(output,user):
        return {'token_ids':output.sequences[:,user['prompt_tokens']:].cpu().tolist(),
            'sequence_logp':output.sequences_scores.float().cpu().tolist(),
            'transition_logp':model.compute_transition_scores(output.sequences,output.scores,output.beam_indices,normalize_logits=False).float().cpu().tolist()}

    def memory():
        return {'allocated_bytes':torch.cuda.memory_allocated(),'reserved_bytes':torch.cuda.memory_reserved(),
                'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved()}

    setup_started=time.perf_counter()
    warm_outputs={}
    (out/'warmup_outputs').mkdir()
    for user in warm_users:
        for mode in ('compiled_gqa',variant):
            torch.cuda.synchronize();started=time.perf_counter()
            output=generate(user,mode!='compiled_gqa',mode!='compiled_gqa')
            torch.cuda.synchronize();elapsed=time.perf_counter()-started
            raw=unpack(output,user);del output
            key=user['cohort_index']
            b.save(out/'warmup_outputs'/f'{key}_{mode}.json',raw)
            if mode=='compiled_gqa':warm_outputs[key]=raw
            # Cross-variant exactness is audited after preserving all pilot outputs.
            # Never turn an FP32 mismatch into a tolerance-based pass.
            compiler.add_warmup({'variant':mode,'elapsed_s':elapsed,'prompt_tokens':user['prompt_tokens']})
    for user in reversed(warm_users):
        output=generate(user,True,True)
        raw=unpack(output,user);del output
        key=user['cohort_index']
        b.save(out/'warmup_outputs'/f'{key}_repeat.json',raw)
        previous=json.loads((out/'warmup_outputs'/f'{key}_{variant}.json').read_text())
        assert raw==previous, 'Mixed-length candidate instability'
        warm_outputs[key]=raw
    torch.cuda.synchronize()
    assert torch.cuda.memory_allocated()<40*1024**3
    assert torch.cuda.memory_reserved()<46*1024**3
    pool.freeze();compiler.freeze()
    b.save(out/'setup.json',{'combined_setup_and_warmup_s':time.perf_counter()-setup_started,
        'replay':pool.report(),'direct_reorder':direct.report(),
        'warmup_cohort_indices':[u['cohort_index'] for u in warm_users],'memory':memory(),
        'note':'Isolated process residency. Setup includes nested compiler and capture timings; do not add. Peak measurement counters reset after this record.'})
    torch.cuda.reset_peak_memory_stats()
    rows=[]
    trials=[(0,u) for i,u in enumerate(users) if i%2==half] if args.full_cohort else [
        (repeat,u) for repeat in range(half*3,half*3+3) for u in users]
    for repeat,user in trials:
        torch.cuda.synchronize();started=time.perf_counter()
        output=generate(user,True,True)
        torch.cuda.synchronize();elapsed=time.perf_counter()-started
        raw=unpack(output,user);del output
        row={'repeat':repeat,'cohort_index':user['cohort_index'],'prompt_tokens':user['prompt_tokens'],
             'variant':variant,'run':{'elapsed_s':elapsed,**raw}}
        rows.append(row)
        b.save(out/'users'/f"{repeat}_{user['cohort_index']:04d}.json",row)
        if user['cohort_index'] in warm_outputs:
            assert raw==warm_outputs[user['cohort_index']], 'Replay drifted'
        b.announce({'stage':'measured','variant':variant,'repeat':repeat,'cohort_index':user['cohort_index']})
    b.save(out/'requests.json',rows)
    compiler.finish()
    summary={'complete':True,'measured_requests':len(rows),'latency':b.distribution([r['run']['elapsed_s'] for r in rows]),
        'replay':pool.report(),'memory':memory(),'direct_reorder':direct.report()}
    b.save(out/'summary.json',summary);b.announce(summary)


if __name__=='__main__':main()
