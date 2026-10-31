"""Independent expected inventory: pinned config, declared users and SID catalog.

No model weights are loaded: the expected model is instantiated on meta. The
producer's observed paths/table metadata never define the required schema.
"""
import collections
import hashlib
import json
import math
import re
from pathlib import Path

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

TABLE_PATHS = frozenset(["table['" + n + "']" for n in
    ('offsets', 'sizes', 'root_prefix', 'first_tokens', 'eos', 'newline', 'valid_sids')]
    + [f"table['strides'][{n}]" for n in (1, 2, 3)]
    + [f"table['tables'][{n}]" for n in (1, 2)])


def pinned_config(path, expected_sha256):
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError('Config source hash differs')
    config = Qwen2Config.from_dict(json.loads(raw))
    if config.model_type != 'qwen2' or set(config.layer_types) != {'full_attention'}:
        raise ValueError('Unsupported model config')
    return config


def spec(shape, dtype):
    shape = list(shape)
    stride = [math.prod(shape[i+1:]) for i in range(len(shape))]
    size = torch.empty((), dtype=dtype).element_size()
    return dict(shape=shape, dtype=str(dtype), stride=stride,
                logical_bytes=math.prod(shape)*size)


def expected_schema(config, users, constraints, root_prefix, packed=False):
    """Auditor builds this itself from hash-bound inputs, never observed inventory."""
    # Match from_pretrained(torch_dtype=BF16) initialization, NOT model.bfloat16():
    # Qwen explicitly initializes its rotary inv_freq buffer as FP32.
    previous_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.bfloat16)
        with torch.device('meta'):
            model = Qwen2ForCausalLM(config).eval()
    finally:
        torch.set_default_dtype(previous_dtype)
    result = {'model.parameter.'+n: spec(t.shape, t.dtype)
              for n, t in model.named_parameters(remove_duplicate=False)}
    result.update({'model.buffer.'+n: spec(t.shape, t.dtype)
                   for n, t in model.named_buffers(remove_duplicate=False)})
    if packed:
        d = config.hidden_size // config.num_attention_heads
        width = d*(config.num_attention_heads + 2*config.num_key_value_heads)
        for i in range(config.num_hidden_layers):
            base = f'model.buffer.model.layers.{i}.self_attn._research_packed_qkv.'
            result[base+'weight'] = spec((width, config.hidden_size), torch.bfloat16)
            result[base+'bias'] = spec((width,), torch.bfloat16)
    seen = set()
    for u in users:
        idx, length = u['cohort_index'], u['prompt_tokens']
        if type(idx) is not int or idx in seen or length <= 0:
            raise ValueError('Invalid or duplicate cohort index')
        seen.add(idx)
        for n in (0, 1):
            result[f'input[{idx}][{n}]'] = spec((1, length), torch.int64)
    if not seen:
        raise ValueError('Empty declared cohort')
    sids = sorted(k for k in constraints if len(k) == 3 and tuple(k) != tuple(root_prefix))
    if not sids:
        raise ValueError('Empty SID catalog')
    sizes = [max(s[i] for s in sids)-min(s[i] for s in sids)+1 for i in range(3)]
    shapes = {'offsets': (3,), 'sizes': (3,), 'root_prefix': (3,),
              'first_tokens': (len({s[0] for s in sids}),), 'eos': (), 'newline': (),
              'valid_sids': (math.prod(sizes),)}
    for name, shape in shapes.items():
        result[f"table['{name}']"] = spec(shape, torch.bool if name == 'valid_sids' else torch.int64)
    for n in (1, 2, 3):
        result[f"table['strides'][{n}]"] = spec((n,), torch.int64)
    for n in (1, 2):
        allowed = collections.defaultdict(set)
        for sid in sids:
            allowed[sid[:n]].add(sid[n])
        width = max(map(len, allowed.values()))
        result[f"table['tables'][{n}]"] = spec((math.prod(sizes[:n]), width), torch.int64)
    assert {k for k in result if k.startswith('table')} == TABLE_PATHS
    return result


def validate_snapshot(observed, schema):
    if set(observed) != set(schema):
        raise ValueError(f'Inventory coverage differs: missing={sorted(set(schema)-set(observed))}, extra={sorted(set(observed)-set(schema))}')
    for name, required in schema.items():
        row = observed[name]
        if any(row.get(k) != v for k, v in required.items()):
            raise ValueError('Inventory metadata differs: '+name)
        if not isinstance(row.get('sha256'), str) or not re.fullmatch('[0-9a-f]{64}', row['sha256']):
            raise ValueError('Invalid digest: '+name)
        if row.get('storage_offset') != 0 or row.get('storage_bytes', -1) < row['logical_bytes']:
            raise ValueError('Invalid storage span: '+name)
        if any(type(row.get(k)) is not int or row[k] <= 0 for k in ('object_id', 'storage_ptr')):
            raise ValueError('Missing live tensor identity: '+name)
        if not isinstance(row.get('device'), str):
            raise ValueError('Missing device: '+name)
    return dict(tensors=len(schema), logical_bytes=sum(r['logical_bytes'] for r in schema.values()))


def validate_unchanged(before, after, schema):
    validate_snapshot(before, schema)
    validate_snapshot(after, schema)
    if before != after:
        raise ValueError('Immutable tensor identity, storage or bytes changed')


def validate_packing(evidence, before, after, original_schema, packed_schema, config):
    """Audit complete producer proof against independent layer and byte counts."""
    validate_snapshot(before, original_schema)
    validate_snapshot(after, packed_schema)
    if {n:after[n] for n in original_schema} != before:
        raise ValueError('Original tensors changed during packing')
    d = config.hidden_size // config.num_attention_heads
    widths = [d*config.num_attention_heads, d*config.num_key_value_heads, d*config.num_key_value_heads]
    per_layer = sum(widths)*(config.hidden_size+1)*2
    layers = evidence.get('layers', [])
    if len(layers) != config.num_hidden_layers or evidence.get('extra_bytes') != per_layer*config.num_hidden_layers:
        raise ValueError('Incomplete packing proof/byte count')
    old_ptrs = {r['storage_ptr'] for r in before.values()}
    new_ptrs = set()
    for i, row in enumerate(layers):
        module = f'model.layers.{i}.self_attn'
        if row.get('module') != module or row.get('widths') != widths or row.get('extra_bytes') != per_layer:
            raise ValueError('Packed layer contract differs')
        slices = row.get('slices', [])
        if len(slices) != 6:
            raise ValueError('Incomplete packed slices')
        for kind_index, kind in enumerate(('weight', 'bias')):
            ptr = after[f'model.buffer.{module}._research_packed_qkv.{kind}']['storage_ptr']
            if ptr in old_ptrs or ptr in new_ptrs:
                raise ValueError('Packed storage overlaps supplied storage')
            new_ptrs.add(ptr)
            offset = 0
            for part_index, (part, width) in enumerate(zip(('q_proj','k_proj','v_proj'), widths)):
                value = before[f'model.parameter.{module}.{part}.{kind}']['sha256']
                want = dict(kind=kind, offset=offset, width=width,
                            original_sha256=value, packed_sha256=value)
                if slices[kind_index*3+part_index] != want:
                    raise ValueError('Packed slice proof differs')
                offset += width
    return dict(layers=len(layers), extra_bytes=per_layer*len(layers), original_identity_preserved=True)
