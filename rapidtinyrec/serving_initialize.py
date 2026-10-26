"""Allocated-compute model initialization and serialized warm evidence production.

This is not a submission entrypoint. A harness must additionally bind source,
run/allocation identity, deadlines and independent audits before any GPU trial.
"""
import collections
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

from .serving_context import OwnedCudaContext
from .serving_runtime import PreparedFoldedLane


def save_exclusive(path, value):
    with Path(path).open('x') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write('\n')


def warm_lane(lane, users, out):
    """No freeze here: other lanes must finish setup before any compiler freezes."""
    seen_lengths = set()
    for user in users:
        key = user['cohort_index']
        raw = lane.warm(user, replay=False)
        save_exclusive(out/f'{key}_dense.json', raw)
        retain = user['prompt_tokens'] not in seen_lengths
        raw = lane.warm(user, replay=True, retain=retain)
        save_exclusive(out/f'{key}_replay.json', raw)
        seen_lengths.add(user['prompt_tokens'])
    for user in reversed(users):
        raw = lane.warm(user, replay=True)
        save_exclusive(out/f"{user['cohort_index']}_repeat.json", raw)
    # Preserve drift too; the separate setup auditor decides whether it passes.
    save_exclusive(out/'retained_after_mixed.json', lane.scan_retained(users))


def _allocate(args, users, out):
    from . import benchmark as b
    from .compiled_forward import CompiledForward
    from .gpu_constraints import GpuSidConstraintTables
    from .decode_replay import DecodeReplayPool
    torch = b.torch
    # Preserve normal model/parameter construction from the accepted source.
    # The surrounding owner context uses inference mode for execution.
    with torch.inference_mode(False):
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
        compiler=CompiledForward(model,out,b.announce)
        from .input_arena import SharedInputArena
        from .direct_reorder import ReorderIntoReplay
        from .shared_prefix import PrefixArena
        from .prefix_replay import PrefixReplayPool, ReorderSuffix
        prefix_mode=True
        maximum=max(u['prompt_tokens'] for u in users)
        arena=PrefixArena(maximum) if prefix_mode else SharedInputArena(maximum+3)
        pool_class=PrefixReplayPool if prefix_mode else DecodeReplayPool
        pool=pool_class(compiler.compiled,max_graphs=40 if args.full_cohort else 8,arena=arena)
        direct=ReorderSuffix(pool) if prefix_mode else ReorderIntoReplay(pool)

    return model, compiler, arena, pool, direct, tok, tables, inputs, masks


def lane_initializer(release_root, reference_results, output_root):
    """Verify identity/resources once, then return the serialized lane callback.

    Must be constructed on allocated compute. It requires a pinned runtime and
    active process-wide folded patch; prepared_service enters that patch before
    invoking the callback. This check is not the complete allocation audit.
    """
    if not os.environ.get('SLURM_JOB_ID', '').isdigit():
        raise RuntimeError('Model initialization requires allocated compute')
    from . import benchmark as b
    import torch
    import torch._dynamo
    import transformers
    if (torch.__version__ != '2.14.0+cu126' or torch.version.git_version !=
            '08187d9e0fba026dc8217405802ab5381dc88d90' or transformers.__version__ != '4.55.4'):
        raise RuntimeError('Pinned serving runtime required')
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1 or torch.cuda.get_device_name(0) != 'NVIDIA RTX A6000':
        raise RuntimeError('One regular NVIDIA RTX A6000 required')
    torch.manual_seed(42)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    expected = json.loads((b.ROOT/'expected.json').read_text())
    reference = json.loads((Path(reference_results)/'manifest.json').read_text())
    content = {key: value for key, value in reference.items() if key != 'manifest_sha256'}
    if (reference['experiment'] != 'gqa_three_way' or reference['dtype'] != 'float32' or
            hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest() != reference['manifest_sha256'] or
            hashlib.sha256(json.dumps(reference['users'], sort_keys=True).encode()).hexdigest() != expected['users_sha256']):
        raise ValueError('Historical reference identity mismatch')
    from .benchmark_folded_prefix import select_users
    users, _ = select_users(reference['users'], False)
    release_root = Path(release_root)
    observed = {name: b.digest(release_root/name) for name in expected['release_sha256']}
    if observed != expected['release_sha256']:
        raise ValueError('Release resources differ before model load')
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=False)
    save_exclusive(root/'initialization-provenance.json', dict(users=users,
        observed_release_sha256=observed, reference_manifest_sha256=reference['manifest_sha256'],
        torch_version=torch.__version__, torch_git=torch.version.git_version,
        transformers_version=transformers.__version__, tf32=False))
    def initialize(index):
        if type(index) is not int or index not in (0, 1):
            raise ValueError('Only one/two owned lanes supported')
        from .folded_attention import folded_sdpa
        import torch.nn.functional as functional
        if functional.scaled_dot_product_attention is not folded_sdpa:
            raise RuntimeError('Global folded attention patch must precede lane setup')
        if torch._dynamo.config.error_on_recompile:
            raise RuntimeError('All lane initialization must precede compiler freeze')
        out = root/f'lane_{index}'
        out.mkdir(exist_ok=False)
        warm = out/'warmup'; warm.mkdir()
        context = OwnedCudaContext(0)
        try:
            args = SimpleNamespace(release_root=release_root, full_cohort=False)
            components = context.call(_allocate, args, users, out)
            lane = PreparedFoldedLane(context, *components)
            warm_lane(lane, users, warm)
            return lane
        except BaseException as error:
            try:
                context.close()
            except BaseException as drain_error:
                error.add_note(f'Initialization stream cleanup failed: {drain_error!r}')
            raise
    return initialize
