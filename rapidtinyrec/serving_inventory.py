"""Explicit untimed inventory of pinned folded-runtime Python tensor owners.

The returned tuple holds strong tensor references through cross-lane comparison.
It does not enumerate hidden allocator/graph-private intermediates, prove stream
ordering, or replace runtime/retained-output and graph-pool audits.
"""
from .serving_storage import describe_tensors


def _fields(obj, expected, label):
    if set(vars(obj)) != set(expected):
        raise ValueError(f'{label} owner schema changed: {sorted(vars(obj))}')


def _pairs(rows, name, role, pairs):
    if not isinstance(pairs,(tuple,list)) or not pairs:
        raise ValueError('Nonempty explicit layer pairs required')
    for layer,pair in enumerate(pairs):
        if not isinstance(pair,(tuple,list)) or len(pair)!=2:
            raise ValueError('Each layer needs key and value tensors')
        for kind,tensor in zip(('key','value'),pair):
            rows.append((f'{name}.{layer}.{kind}',role,tensor))


def enumerate_folded_lane(model, tables, arena, replay, prompts, retained):
    """No copies, model calls, synchronization, or mutations; call while idle.

    prompts maps stable request names to named prepared input tensors. retained
    maps stable sample names to explicitly flattened independent output tensors.
    An empty retained mapping is allowed before any output is retained.
    """
    rows=[]
    for kind,items in (('parameter',model.named_parameters(remove_duplicate=False)),
                       ('buffer',model.named_buffers(remove_duplicate=False))):
        for name,tensor in items:rows.append((f'model.{kind}.{name}','model_'+kind,tensor))
    _fields(tables,('sizes_python','offsets','sizes','strides','root_prefix','first_tokens',
                    'eos','newline','tables','valid_sids','metadata'),'constraint')
    for name in ('offsets','sizes','root_prefix','first_tokens','eos','newline','valid_sids'):
        rows.append(('constraints.'+name,'constraint',getattr(tables,name)))
    for name,keys in (('strides',{1,2,3}),('tables',{1,2})):
        mapping=getattr(tables,name)
        if set(mapping)!=keys:raise ValueError('Constraint table keys changed')
        for key,tensor in sorted(mapping.items()):rows.append((f'constraints.{name}.{key}','constraint',tensor))
    _fields(arena,('max_prefix_tokens','max_suffix_tokens','storage','layout','prefix_views','suffix_views',
                  'requests_loaded','prefix_bytes_copied','beams','prefix_storage','suffix_storage'),'prefix arena')
    if arena.storage is None:raise ValueError('Initialize arena before inventory')
    for name in ('prefix_storage','suffix_storage'):
        _pairs(rows,'arena.'+name,'arena_backing',getattr(arena,name))
    # The aggregate storage tuple must refer to exactly the declared owners.
    expected=arena.prefix_storage+arena.suffix_storage
    if len(arena.storage)!=len(expected) or any(len(p)!=2 for p in arena.storage) or any(a is not b for p,q in zip(arena.storage,expected) for a,b in zip(p,q)):
        raise ValueError('Arena aggregate owners differ')
    for name in ('prefix_views','suffix_views'):
        for key,pairs in sorted(getattr(arena,name).items()):_pairs(rows,f'arena.{name}.{key}','arena_view',pairs)
    _fields(replay,('forward','arena','max_graphs','entries','events','frozen','replays'),'replay')
    if replay.arena is not arena or not replay.frozen or not replay.entries:
        raise ValueError('Frozen replay must own the supplied initialized arena')
    for index,entry in enumerate(replay.entries.values()):
        if len(entry)!=3:raise ValueError('Replay entry schema changed')
        buffers,graph,output=entry
        _fields(buffers,('signature','tensors','prefixes','pairs','fixed'),'prefix buffers')
        _fields(output,('logits','prefixes','pairs'),'prefix output')
        if set(buffers.tensors)!={'input_ids','position_ids','cache_position'}:
            raise ValueError('Replay tensor arguments changed')
        if any(value is not None and type(value) not in (bool,int) for value in buffers.fixed.values()):
            raise ValueError('Unexpected fixed replay tensor/object')
        for name,tensor in sorted(buffers.tensors.items()):rows.append((f'replay.{index}.input.{name}','replay_input',tensor))
        for kind,obj in (('input',buffers),('output',output)):
            for name in ('prefixes','pairs'):_pairs(rows,f'replay.{index}.{kind}.{name}','replay_'+kind,getattr(obj,name))
        rows.append((f'replay.{index}.output.logits','graph_logits',output.logits))
    if not prompts:raise ValueError('Explicit prepared prompt tensors required')
    for group,mapping in (('prompt',prompts),('retained',retained)):
        for sample,tensors in mapping.items():
            if not isinstance(sample,str) or not sample or not tensors:raise ValueError('Named nonempty tensor sample required')
            for name,tensor in tensors.items():
                if not isinstance(name,str) or not name:raise ValueError('Explicit tensor name required')
                rows.append((f'{group}.{sample}.{name}',group,tensor))
    result=tuple(rows)
    describe_tensors(result)  # Validate actual storages without reading their values.
    return result
