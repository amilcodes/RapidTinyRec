"""CPU process/launch evidence for a future externally supervised serving run.

These checks do not enforce a hard timeout. The parent supervisor must enforce
the same global deadline, including preflight, and preserve launch/exit evidence.
"""
import os
from pathlib import Path
import re
import socket
import time

from .serving_allocation import identical, require
from .serving_contract import PROTOCOL_SHA256


def process_identity(pid=None, *, proc=Path('/proc')):
    """Linux PID reuse protection; field 22 follows a possibly spaced comm."""
    pid = os.getpid() if pid is None else pid
    require(type(pid) is int and pid > 0, 'Positive process PID required')
    raw = (proc/str(pid)/'stat').read_text()
    close = raw.rfind(')')
    require(raw.startswith(str(pid)+' (') and close > 0, 'Malformed process stat')
    fields = raw[close+1:].split()
    require(len(fields) >= 20, 'Truncated process stat')
    parent, start = int(fields[1]), int(fields[19])
    boot = (proc/'sys/kernel/random/boot_id').read_text().strip()
    require(parent > 0 and start > 0 and re.fullmatch(r'[0-9a-f-]{36}', boot) is not None,
            'Incomplete process birth identity')
    return dict(pid=pid, parent_pid=parent, start_ticks=start, boot_id=boot,
                hostname=socket.gethostname(), clock_ticks=os.sysconf('SC_CLK_TCK'))


def check_deadline(launch, now_ns=None):
    now = time.perf_counter_ns() if now_ns is None else now_ns
    start, deadline, launched = (launch[k] for k in ('global_started_ns', 'deadline_ns', 'launch_started_ns'))
    require(all(type(v) is int and v > 0 for v in (start, deadline, launched, now)),
            'Integer global deadline clocks required')
    require(deadline-start == 1620*10**9 and start <= launched <= now < deadline,
            'Expired or altered serving global deadline')
    return now


def validate_launch(launch, run_root, phase, protocol, identity, parent, token, now_ns=None):
    require(type(phase) is int and phase in range(4), 'Frozen four-phase index required')
    spec = protocol['phases'][phase]
    require(launch.get('schema') == 'serving-phase-launch-v1' and
            launch.get('run_root') == str(run_root) and launch.get('phase') == phase and
            type(launch['phase']) is int and launch.get('protocol_sha256') == PROTOCOL_SHA256,
            'Launch/run/protocol binding differs')
    require(isinstance(token, str) and re.fullmatch(r'[0-9a-f]{32}', token) is not None and
            launch.get('launch_token') == token, 'Fresh parent launch token required')
    require(identical(launch['parent_identity'], parent) and identity['parent_pid'] == parent['pid'] and
            identity['pid'] != parent['pid'] and identity['boot_id'] == parent['boot_id'] and
            identity['hostname'] == parent['hostname'] and identity['clock_ticks'] == parent['clock_ticks'] and
            identity['start_ticks'] >= parent['start_ticks'], 'Phase parent/birth identity differs')
    cache = launch['cache']
    expected = Path(run_root)/'cache'/spec['cache_family']
    require(identical(cache, dict(family=spec['cache_family'], root=str(expected),
                                 reuse=spec['reuse_disk_cache'])), 'Phase cache family differs')
    check_deadline(launch, now_ns)
    return spec


def observe_cache(root):
    """Hash actual cold/reused disk contents; no cache creation or deletion."""
    from .serving_allocation import digest
    root = Path(root)
    require(root.is_dir() and not root.is_symlink() and root.resolve() == root,
            'Canonical cache directory required')
    result = {}; inodes = set()
    for path in sorted(root.rglob('*')):
        require(not path.is_symlink(), 'Linked cache entry')
        if path.is_dir():
            continue
        stat = path.stat(); inode = (stat.st_dev, stat.st_ino)
        require(path.is_file() and stat.st_nlink == 1 and inode not in inodes, 'Aliased/nonregular cache entry')
        inodes.add(inode)
        result[path.relative_to(root).as_posix()] = dict(bytes=stat.st_size, sha256=digest(path))
    return result


def validate_cache_environment(root, env):
    expected = {'TORCHINDUCTOR_CACHE_DIR': str(Path(root)/'inductor'),
                'TRITON_CACHE_DIR': str(Path(root)/'triton'),
                'CUDA_CACHE_PATH': str(Path(root)/'cuda')}
    require(all(env.get(k) == v for k, v in expected.items()), 'Phase cache environment differs')
    # Alternate roots/remote caches would make observed cold/reuse state ambiguous.
    forbidden = ('TORCHINDUCTOR_FORCE_DISABLE_CACHES', 'TORCHINDUCTOR_CACHE_KEY_TAG',
                 'TORCHINDUCTOR_FX_GRAPH_REMOTE_CACHE', 'TORCHINDUCTOR_AUTOTUNE_REMOTE_CACHE',
                 'TRITON_OVERRIDE_DIR', 'TRITON_DUMP_DIR', 'TRITON_KERNEL_OVERRIDE')
    require(not any(k in env for k in forbidden), 'Unreviewed cache override')
    return expected
