"""Frozen BF16 versus audited TF32 pilot identity, separate from exact serving."""
import hashlib
import json
from pathlib import Path
PROTOCOL = 'benchmarks/preparation/quality-frontier/bf16-pilot-protocol.json'
PROTOCOL_SHA256 = '654ccf21c2d41c6cd37f0732507248c0d469065e7538e095ac9bee8cfc9988e3'
PHASES = ('folded_tf32','folded_bf16','folded_bf16','folded_tf32')
def load_protocol(path):
    data=Path(path).read_bytes()
    assert hashlib.sha256(data).hexdigest()==PROTOCOL_SHA256
    return json.loads(data)
def phase_record(protocol,phase,reused,users):
    assert [u['cohort_index'] for u in users]==protocol['cohort_indices']
    assert reused==(phase>=2)
    return dict(schema='bf16-vs-tf32-phase-v1',protocol_sha256=PROTOCOL_SHA256,phase=protocol['phases'][phase],requests=12,graphs=8)
