"""Require a separately reviewed exact source tree before any executable entry."""
import hashlib,json
from pathlib import Path

def require_reviewed_source(code):
    code=Path(code).resolve(strict=True);path=code/'readiness-lock.json'
    if not path.is_file() or path.is_symlink():raise ValueError('QKV source has not passed final readiness review')
    lock=json.loads(path.read_text())
    if lock.get('schema')!='packed-qkv-readiness-lock-v1' or lock.get('submission_ready') is not True:raise ValueError('QKV readiness not approved')
    actual={str(p.relative_to(code)):hashlib.sha256(p.read_bytes()).hexdigest() for p in code.rglob('*') if p.is_file() and p!=path and '.oscar' not in p.relative_to(code).parts and '__pycache__' not in p.parts and p.suffix!='.pyc'}
    if actual!=lock['files'] or any(p.is_symlink() for p in code.rglob('*')):raise ValueError('Reviewed QKV source tree changed')
    return lock
