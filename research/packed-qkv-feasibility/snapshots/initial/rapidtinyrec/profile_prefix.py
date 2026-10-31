"""Profile physical-prefix replay without changing the model or cache operations.

Ten representatives retain the same 40 graph shapes as the full comparison.
Uninstrumented timings precede all profiler initialization. Uncaptured compiled
prefix calls are a separate attribution diagnostic, not a serving candidate.
"""
import contextlib
import functools
import gzip
import os
from pathlib import Path
import shutil
import time
from unittest.mock import patch

from . import benchmark as b
from .audit_decode_pilot import compare
from .gpu_constraints import GpuSidConstrainedLogitsProcessor
from .prefix_replay import PrefixBuffers, PrefixOutput, PrefixReplayPool, ReorderSuffix
from .profile_shared import Scopes, summarize_scopes
from .shared_prefix import PrefixArena, split_pairs

torch=b.torch


@contextlib.contextmanager
def diagnostic(pool, scopes=None):
    """Use captured input layouts but launch the unchanged compiled forward."""
    @torch.inference_mode()
    def call(self, **kwargs):
        assert self is pool
        buffers, _, _ = self.entries[self.signature(kwargs)]
        buffers.load(kwargs)
        with scopes.scope('diagnostic_decode') if scopes else contextlib.nullcontext():
            output=self.forward(**buffers.kwargs())
        retained=PrefixOutput(output)
        self.arena.check_output(retained)
        return retained.fresh()
    with patch.object(PrefixReplayPool,'__call__',call):
        yield


def warm_diagnostic(pool, users, generate, unpack, out):
    started=time.perf_counter()
    rows=[]
    with diagnostic(pool):
        for user in users:
            output=generate(user,True,True)
            rows.append({'cohort_index':user['cohort_index'], 'prompt_tokens':user['prompt_tokens'],
                         'output':unpack(output,user)})
            del output
    torch.cuda.synchronize()
    b.save(out/'diagnostic_warmup.json',{'elapsed_s':time.perf_counter()-started,'rows':rows})


@contextlib.contextmanager
def instrument(scopes, model, compiler):
    def scoped(original,name):
        @functools.wraps(original)
        def call(*args,**kwargs):
            with scopes.scope(name):return original(*args,**kwargs)
        return call
    original=compiler.compiled
    @functools.wraps(original)
    def prefill(**kwargs):
        assert kwargs['input_ids'].shape[0]==1
        with scopes.scope('compiled_prefill'):return original(**kwargs)
    original_reorder=ReorderSuffix.reorder
    def reorder(self,cache,beam_idx):
        suffix=split_pairs(cache)[1][0][0].shape[2]
        name='final_unused_cache_reorder' if suffix>self.pool.arena.max_suffix_tokens else 'suffix_reorder'
        with scopes.scope(name):return original_reorder(self,cache,beam_idx)
    with contextlib.ExitStack() as stack:
        stack.enter_context(patch.object(compiler,'compiled',prefill))
        stack.enter_context(patch.object(ReorderSuffix,'reorder',reorder))
        for obj,attr,name in [
            (PrefixBuffers,'load','replay_input_staging'),
            (PrefixOutput,'fresh','replay_output_container'),
            (PrefixArena,'start_request','prefix_copy'),
            (torch.cuda.CUDAGraph,'replay','graph_decode'),
            (GpuSidConstrainedLogitsProcessor,'__call__','constraints'),
            (torch.nn.functional,'log_softmax','vocabulary_log_softmax'),
            (model,'prepare_inputs_for_generation','prepare_inputs'),
            (model,'_update_model_kwargs_for_generation','update_model_kwargs'),
            (model,'_get_top_k_continuations','beam_topk_continuations'),
            (model,'_get_running_beams_for_next_iteration','beam_running_update'),
            (model,'_update_finished_beams','beam_finished_update'),
            (model,'_check_early_stop_heuristic','beam_early_stop'),
            (model,'_beam_search_has_unfinished_sequences','beam_stop_sync')]:
            stack.enter_context(patch.object(obj,attr,scoped(getattr(obj,attr),name)))
        yield


def profile(model,compiler,pool,direct,users,generate,unpack,out,memory):
    assert compiler.phase=='measurement' and pool.frozen and len(pool.entries)==40
    report={'complete':False,'uninstrumented':[],'light_profiles':[],'torch_profiles':[],
            'before_replay':pool.report(),'source_sha256':{p.name:b.digest(p) for p in b.ROOT.glob('*.py')},
            'note':'Ten representative users, not a 500-user benchmark. Timing precedes profiler initialization. Diagnostic mode uses the unchanged compiled prefix forward and stable inputs without graph replay, for operation attribution only. Host times overlap asynchronous GPU execution; CUDA-event spans include idle. Trace kernel sums are instrumented and may overlap; do not add host and GPU times.'}
    references={}
    # Six alternating-length sweeps, all 40 graph shapes resident.
    for repeat in range(6):
        for user in users if repeat%2==0 else list(reversed(users)):
            torch.cuda.synchronize();started=time.perf_counter()
            output=generate(user,True,True)
            torch.cuda.synchronize();elapsed=time.perf_counter()-started
            raw=unpack(output,user);del output
            references.setdefault(user['cohort_index'],raw)
            report['uninstrumented'].append({'repeat':repeat,'cohort_index':user['cohort_index'],
                'prompt_tokens':user['prompt_tokens'],'elapsed_s':elapsed,'output':raw})
    report['uninstrumented_latency']=b.distribution([r['elapsed_s'] for r in report['uninstrumented']])
    report['uninstrumented_memory']=memory()
    b.save(out/'prefix_profile.json',report)
    b.announce({'stage':'uninstrumented_complete',**report['uninstrumented_latency']})

    for repeat in range(3):
        for user in users:
            scopes=Scopes('light')
            torch.cuda.synchronize()
            with instrument(scopes,model,compiler),scopes.scope('request'):
                output=generate(user,True,True)
                with scopes.scope('final_sync'):torch.cuda.synchronize()
            torch.cuda.synchronize()
            raw=unpack(output,user);del output
            rows=scopes.data()
            report['light_profiles'].append({'repeat':repeat,'cohort_index':user['cohort_index'],
                'prompt_tokens':user['prompt_tokens'],'scopes':rows,'summary':summarize_scopes(rows),'output':raw})
    b.save(out/'prefix_profile.json',report)

    # Actual replay first. Diagnostic traces expose operation/shape correlations
    # hidden inside a graph launch, while leaving native GQA math unchanged.
    selected=[users[0],users[len(users)//2],users[-1]]
    for mode in ('replay','diagnostic'):
        for user in selected:
            scopes=Scopes('torch');torch.cuda.synchronize()
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],
                    record_shapes=True,profile_memory=False,with_stack=False) as prof:
                with instrument(scopes,model,compiler), (diagnostic(pool,scopes) if mode=='diagnostic' else contextlib.nullcontext()), scopes.scope('request'):
                    output=generate(user,True,True)
                    with scopes.scope('final_sync'):torch.cuda.synchronize()
            raw=unpack(output,user);del output
            trace=out/f'prefix_trace_{mode}_{user["prompt_tokens"]}.json'
            prof.export_chrome_trace(str(trace))
            with trace.open('rb') as src,gzip.open(str(trace)+'.gz','wb') as dst:shutil.copyfileobj(src,dst)
            trace.unlink()
            events=[{'name':e.key,'input_shapes':e.input_shapes,'count':e.count,'cpu_total_us':e.cpu_time_total,
                     'self_cpu_us':e.self_cpu_time_total,'device_total_us':e.device_time_total,'self_device_us':e.self_device_time_total}
                    for e in prof.key_averages(group_by_input_shape=True)]
            report['torch_profiles'].append({'mode':mode,'cohort_index':user['cohort_index'],
                'prompt_tokens':user['prompt_tokens'],'trace':trace.name+'.gz','events':events,'output':raw})
            b.save(out/'prefix_profile.json',report)
            b.announce({'stage':'prefix_trace_complete','mode':mode,'prompt_tokens':user['prompt_tokens']})
    # Preserve generated code for attributing fused cats/layout operations.
    generated=out/'inductor_sources';generated.mkdir()
    generated_manifest={}
    for source in Path(os.environ['TORCHINDUCTOR_CACHE_DIR']).rglob('*.py'):
        destination=generated/(b.digest(source)+'.py')
        shutil.copyfile(source,destination)
        generated_manifest[destination.name]={'original_relative_path':str(source.relative_to(Path(os.environ['TORCHINDUCTOR_CACHE_DIR']))),'sha256':b.digest(destination)}
    b.save(out/'inductor_sources.json',generated_manifest)
    compiler.finish()
    assert report['before_replay']['events']==pool.report()['events']
    pairs=[(references[r['cohort_index']],r['output']) for group in ('uninstrumented','light_profiles','torch_profiles') for r in report[group]]
    comparison=compare(pairs)
    # Save evidence even when exactness fails. Independent audit is authoritative.
    report.update(complete=True,after_replay=pool.report(),measurement_compilations=0,
        captures_during_profiling=0,comparison=comparison,after_memory=memory(),direct_reorder=direct.report())
    b.save(out/'prefix_profile.json',report)
    b.announce({'stage':'prefix_profile_complete','comparison':comparison})

