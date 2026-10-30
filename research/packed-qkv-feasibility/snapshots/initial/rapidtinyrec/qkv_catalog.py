"""Allocated auditor's catalog reconstruction; no model weights or generation."""
import collections
import hashlib
import json
from pathlib import Path
from .audit_qkv_outputs import require
from .precision_quality_adapter import quality_inputs


def authenticate_catalog(release,users,expected):
    release=Path(release)
    # Includes tokenizer and original information text, independently of the
    # producer's reported verification record. Each file is read only once here.
    for name,want in expected['release_sha256'].items():
        if name.startswith('Office_ckpt/') and name.endswith(('.bin','.safetensors')):
            continue  # Source/model immutability and producer pre-load hashes cover weights.
        p=release/name
        require(hashlib.sha256(p.read_bytes()).hexdigest()==want,'Release resource differs: '+name)
    metadata=quality_inputs(release,users,[])
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(release/'Office_ckpt',local_files_only=True)
    root=users[0]['prompt_token_ids'][-3:]
    require(all(u['prompt_token_ids'][-3:]==root for u in users),'User constraint root differs')
    require(tokenizer.eos_token_id==151645,'Original EOS differs')
    constraints=collections.defaultdict(set);sids=set()
    for line in (release/'Amazon/info/Office_Products_5_2016-10-2018-11.txt').read_text().splitlines():
        ids=tokenizer('### Response:\n'+line.split('\t')[0].strip()+'\n').input_ids+[tokenizer.eos_token_id]
        require(len(ids)==8 and ids[:3]==root and ids[-2:]==[198,151645],'Original SID tokenizer boundary differs')
        sids.add(tuple(ids[3:6]))
        for step in range(3,len(ids)):
            constraints[tuple(ids[:step] if step==3 else ids[3:step])].add(ids[step])
    require(sids=={tuple(s) for s in metadata['catalog_sids']},'Information-text and index SID catalogs differ')
    return constraints,root,dict(catalog_sids=len(sids),independent_users=len(metadata['users']),weights_loaded=False,bootstrap_executed=False)
