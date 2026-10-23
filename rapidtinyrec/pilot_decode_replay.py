"""Short/long-history A6000 pilot; no 500-user performance claim."""
import argparse
import collections
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


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--release-root',type=Path,required=True)
    ap.add_argument('--reference-results',type=Path,required=True)
    ap.add_argument('--output-root',type=Path,required=True)
    ap.add_argument('--repeats',type=int,default=8)
    args=ap.parse_args()
    assert args.repeats>=2 and args.repeats%2==0
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
    lengths=sorted({u['prompt_tokens'] for u in reference['users']})
    users=[u for length in (lengths[0],lengths[-1])
           for u in [x for x in reference['users'] if x['prompt_tokens']==length][:2]]
    assert len(users)==4
    out=args.output_root/'decode_replay_pilot'
    out.mkdir(parents=True,exist_ok=True)
    assert not (out/'manifest.json').exists(),'Use a new output directory'
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
    manifest={'experiment':'decode_replay_pilot','scope':'Four representative users, repeated trials; not a 500-user estimate.',
        'users':users,'dtype':'float32','tf32':False,'hardware':torch.cuda.get_device_name(0),
        'cohort_sha256':expected['users_sha256'],'reference_manifest_sha256':reference['manifest_sha256'],
        'versions':{n:b.importlib.metadata.version(n) for n in ['torch','transformers','tokenizers']},
        'sha256':{n:b.digest(b.ROOT/n) for n in ['pilot_decode_replay.py','decode_replay.py','compiled_forward.py','shared_prefill.py','gpu_constraints.py','expected.json']},
        'protocol':'FP32 beam50. Same compiled GQA forward on both sides. CUDA synchronization around each request. Prefill, staging copies, graph replay, constraints and beam cache reordering included. Compilation, capture, warmup and output extraction excluded. Alternate order for every repeat.'}
    b.save(out/'manifest.json',manifest)
    compiler=CompiledForward(model,out,b.announce)
    pool=DecodeReplayPool(compiler.compiled,max_graphs=8)

    @torch.inference_mode()
    def generate(user,replay):
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
        with patch.object(model,'forward',forward),b.PrefillOnce(model,50):
            output=model.generate(inputs[user['cohort_index']],attention_mask=masks[user['cohort_index']],
                generation_config=cfg,return_dict_in_generate=True,output_scores=True,
                logits_processor=b.LogitsProcessorList([GpuSidConstrainedLogitsProcessor(tables)]))
        assert calls==5
        return output

    def unpack(output,user):
        return {'token_ids':output.sequences[:,user['prompt_tokens']:].cpu().tolist(),
            'sequence_logp':output.sequences_scores.float().cpu().tolist(),
            'transition_logp':model.compute_transition_scores(output.sequences,output.scores,output.beam_indices,normalize_logits=False).float().cpu().tolist()}

    setup_started=time.perf_counter()
    warm_outputs={}
    for user in users:
        for variant in ('compiled_gqa','replay'):
            torch.cuda.synchronize();started=time.perf_counter()
            output=generate(user,variant=='replay')
            torch.cuda.synchronize();elapsed=time.perf_counter()-started
            warm_outputs[(user['cohort_index'],variant)]=unpack(output,user)
            del output
            compiler.add_warmup({'variant':variant,'elapsed_s':elapsed,'prompt_tokens':user['prompt_tokens']})
    for user in users:
        for variant in ('compiled_gqa','replay'):
            output=generate(user,variant=='replay')
            assert unpack(output,user)==warm_outputs[(user['cohort_index'],variant)],'Warmup instability'
            del output
    torch.cuda.synchronize()
    pool.freeze();compiler.freeze()
    b.save(out/'setup.json',{'combined_setup_and_warmup_s':time.perf_counter()-setup_started,'replay':pool.report(),
        'note':'Replay preparation includes warmup calls that can compile. Timings are nested and must not be added. All setup is excluded from measured requests.'})
    rows=[]
    for repeat in range(args.repeats):
        for user in users:
            order=('compiled_gqa','replay') if repeat%2==0 else ('replay','compiled_gqa')
            runs={}
            for variant in order:
                torch.cuda.synchronize();started=time.perf_counter()
                output=generate(user,variant=='replay')
                torch.cuda.synchronize();elapsed=time.perf_counter()-started
                raw=unpack(output,user);del output
                assert raw==warm_outputs[(user['cohort_index'],variant)],'Replay or control drifted'
                runs[variant]={'elapsed_s':elapsed,**raw}
            old=json.loads((args.reference_results/'users'/f"{user['cohort_index']:04d}.json").read_text())
            row={'repeat':repeat,'cohort_index':user['cohort_index'],'prompt_tokens':user['prompt_tokens'],
                'order':order,'runs':runs,'comparison':b.compare(runs['compiled_gqa'],runs['replay']),
                'historical_control':b.compare(old['runs']['compiled_gqa'],runs['compiled_gqa'])}
            rows.append(row);b.save(out/'requests.json',rows)
            b.announce({'stage':'measured','repeat':repeat,'cohort_index':user['cohort_index'],
                'rankings_equal':row['comparison']['ranked_tokens_equal'],
                'scores_equal':row['comparison']['sequence_scores_equal_in_rank_order']})
    compiler.finish()
    summary={'complete':True,'scope':manifest['scope'],'measured_pairs':len(rows),
        'latency':{v:b.distribution([r['runs'][v]['elapsed_s'] for r in rows]) for v in ('compiled_gqa','replay')},
        'matching_rank_lists':sum(r['comparison']['ranked_tokens_equal'] for r in rows),
        'matching_sequence_vectors':sum(r['comparison']['sequence_scores_equal_in_rank_order'] for r in rows),
        'matching_transition_arrays':sum(r['comparison']['transition_scores_equal_in_rank_order'] for r in rows),
        'replay':pool.report(),'allocated_bytes':torch.cuda.memory_allocated(),'reserved_bytes':torch.cuda.memory_reserved(),
        'latency_by_length':{str(length):{v:b.distribution([r['runs'][v]['elapsed_s'] for r in rows if r['prompt_tokens']==length])
            for v in ('compiled_gqa','replay')} for length in (lengths[0],lengths[-1])}}
    b.save(out/'summary.json',summary);b.announce(summary)


if __name__=='__main__':main()
