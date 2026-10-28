"""Independent raw validation of persistent prompt KV snapshots.

The complete pilot auditor must additionally validate source, allocation,
request KV ownership, outputs, dispatch, timing and quality. This module alone
cannot certify a pilot or prove inventory completeness.
"""
import hashlib
import json
import math
import os
import re
from pathlib import Path

STAGES=('initial','after-warmup','after-measurement','after-profiles')
REQUESTS=(0,12,24,26)


def validate_metadata(rows,*,layers,kv_heads,head_dim,prefix_tokens,prefix_sha256,identity):
    assert type(rows) is list and [r['stage'] for r in rows]==list(STAGES)
    assert all(type(n) is int and n>0 for n in (layers,kv_heads,head_dim,prefix_tokens))
    shape=[1,kv_heads,prefix_tokens,head_dim];tensor_bytes=math.prod(shape)*4
    expected_bytes=layers*2*tensor_bytes
    first=rows[0]['metadata']
    assert re.fullmatch('[0-9a-f]{64}',first['model_config_sha256'])
    for row,requests in zip(rows,REQUESTS):
        assert row['file']==f"prompt-cache-{row['stage']}.pt"
        assert re.fullmatch('[0-9a-f]{64}',row['sha256'])
        metadata=row['metadata']
        assert type(metadata['requests']) is int and metadata['requests']==requests
        assert metadata['prefix_tokens']==prefix_tokens and metadata['prefix_token_sha256']==prefix_sha256
        assert metadata['dtype']=='torch.float32' and metadata['identity']==identity
        assert metadata['model_config_sha256']==first['model_config_sha256']
        assert metadata['policy']==first['policy']
        policy=metadata['policy']
        assert len(policy)==5 and policy[0] is True and policy[1] is False and policy[4]=='high'
        assert type(policy[2]) is bool and type(policy[3]) is bool
        assert metadata['cached_kv_bytes']==metadata['request_kv_copy_bytes']==expected_bytes
        inventory=metadata['storage_inventory']
        assert inventory==first['storage_inventory'],'Persistent storage identity changed'
        assert len(inventory)==layers*2
        ranges=[]
        for index,entry in enumerate(inventory):
            assert (entry['layer'],entry['kind'])==(index//2,('key','value')[index%2])
            assert entry['shape']==shape and entry['dtype']=='torch.float32'
            ptr=entry['storage_ptr'];size=entry['storage_bytes']
            assert type(ptr) is int and ptr>0 and ptr%4==0
            assert type(size) is int and size==tensor_bytes
            assert all(ptr+size<=a or b<=ptr for a,b in ranges),'Persistent KV storage overlaps'
            ranges.append((ptr,ptr+size))
    return dict(persistent_kv_bytes=expected_bytes,request_kv_copy_bytes=expected_bytes,
                storage_ranges=ranges,shape=shape,layers=layers)


def validate_tensors(snapshots,*,layers,shape):
    """Small synthetic tensors may exercise this validator in CPU tests."""
    import torch
    assert len(snapshots)==4
    first=None
    for snapshot in snapshots:
        assert type(snapshot) is tuple and len(snapshot)==layers
        flattened=[]
        for pair in snapshot:
            assert type(pair) is tuple and len(pair)==2
            for tensor in pair:
                assert type(tensor) is torch.Tensor
                assert tensor.device.type=='cpu' and tensor.dtype==torch.float32
                assert list(tensor.shape)==shape
                assert bool(torch.isfinite(tensor).all()),'Nonfinite cached KV'
                flattened.append(tensor.contiguous().view(torch.uint8))
        if first is None:first=flattened
        else:
            assert all(torch.equal(x,y) for x,y in zip(first,flattened)),'Persistent KV bits changed'
    return dict(stages_checked=4,tensors_per_stage=layers*2,bit_unchanged=True)


def audit_snapshots(root,*,config,protocol,release_sha256):
    assert os.environ.get('SLURM_JOB_ID'),'Raw model snapshot audit requires allocated compute'
    import torch
    root=Path(root)
    rows=json.loads((root/'prompt-cache-snapshots.json').read_text())
    model_pins={k:release_sha256[k] for k in ('Office_ckpt/model.safetensors','Office_ckpt/config.json')}
    token_pins={k:v for k,v in release_sha256.items() if k.startswith('Office_ckpt/') and k not in
        ('Office_ckpt/model.safetensors','Office_ckpt/config.json','Office_ckpt/generation_config.json')}
    manifest_hash=lambda pins:hashlib.sha256(json.dumps(pins,sort_keys=True).encode()).hexdigest()
    identity=dict(checkpoint_manifest_sha256=manifest_hash(model_pins),
                  tokenizer_manifest_sha256=manifest_hash(token_pins),precision='folded_tf32')
    assert config['hidden_size']%config['num_attention_heads']==0
    report=validate_metadata(rows,layers=config['num_hidden_layers'],kv_heads=config['num_key_value_heads'],
        head_dim=config['hidden_size']//config['num_attention_heads'],prefix_tokens=protocol['prefix_tokens'],
        prefix_sha256=protocol['prefix_token_sha256'],identity=identity)
    snapshots=[]
    for row in rows:
        path=root/row['file']
        assert path.is_file() and not path.is_symlink()
        assert hashlib.sha256(path.read_bytes()).hexdigest()==row['sha256']
        snapshots.append(torch.load(path,map_location='cpu',weights_only=True))
    report.update(validate_tensors(snapshots,layers=report['layers'],shape=report['shape']))
    report['request_counts']=list(REQUESTS)
    report['scope']='Persistent prefix snapshots only; no complete-pilot or request-ownership claim'
    return report
