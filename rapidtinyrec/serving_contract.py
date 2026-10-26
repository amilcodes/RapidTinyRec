"""Hash-frozen exact-serving schedule; not a submission/readiness decision."""
import hashlib
from pathlib import Path
from .serving_setup_audit import read_unique

PROFILE = 'exact-serving-throughput-pilot'
PROTOCOL = 'benchmarks/preparation/serving-throughput/protocol.json'
PROTOCOL_SHA256 = '1df3fe229df48bc885d5f092185864919f993df97d14bc139a8aac584b2da0d8'

def load_protocol(path):
    path=Path(path)
    if hashlib.sha256(path.read_bytes()).hexdigest()!=PROTOCOL_SHA256:
        raise ValueError('Frozen serving protocol changed')
    value=read_unique(path)
    if value['profile']!=PROFILE or value['promotion'] is not False:
        raise ValueError('Serving protocol scope differs')
    return value
