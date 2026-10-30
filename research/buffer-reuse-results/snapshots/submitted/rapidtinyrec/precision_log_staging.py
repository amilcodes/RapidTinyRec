"""CPU-testable native API log staging; no GPU computation in this module.

Preserve native logger filenames/coverage, move only their temporary sink.
Publication happens only after the caller has disabled the native loggers.
Failed publication retains local files for recovery. SIGKILL recovery requires
an external allocation wrapper and is deliberately not claimed here.
"""
import hashlib
import os
from pathlib import Path
import shutil
import tempfile

LOCAL_FILESYSTEMS = frozenset(('tmpfs', 'ext4', 'xfs'))
LOG_NAMES = ('capture-cublas.log', 'capture-cublaslt.log')


def filesystem_for(path, mountinfo):
    """Longest enclosing Linux mount; reject network/unknown filesystems."""
    path = Path(path).resolve(strict=True)
    matches = []
    def unescape(value):
        for escaped, plain in ((r'\040', ' '), (r'\011', '\t'), (r'\012', '\n'), (r'\134', '\\')):
            value = value.replace(escaped, plain)
        return value
    for line in mountinfo.splitlines():
        before, after = line.split(' - ', 1)
        fields = before.split(); fs = after.split()[0]
        mount = Path(unescape(fields[4]))
        if path == mount or mount in path.parents:
            matches.append((len(mount.parts), str(mount), fs))
    if not matches:
        raise ValueError('No enclosing filesystem record')
    _, mount, fs = max(matches)
    if fs not in LOCAL_FILESYSTEMS:
        raise ValueError('Not a verified local filesystem: '+fs)
    return {'mount': mount, 'filesystem': fs}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


class LogStage:
    def __init__(self, local_root, output, filesystem, minimum_free_bytes=256*1024*1024):
        local_root = Path(local_root).resolve(strict=True)
        self.output = Path(output).resolve(strict=True)
        if not local_root.is_dir() or not self.output.is_dir():
            raise ValueError('Existing local and output directories required')
        if filesystem['filesystem'] not in LOCAL_FILESYSTEMS:
            raise ValueError('Unverified filesystem')
        if shutil.disk_usage(local_root).free < minimum_free_bytes:
            raise OSError('Insufficient local space for diagnostic logs')
        self.root = Path(tempfile.mkdtemp(prefix='rapidtinyrec-api-', dir=local_root))
        self.filesystem = dict(filesystem)
        self.published = False

    @classmethod
    def for_allocation(cls, local_root, output):
        if not os.environ.get('SLURM_JOB_ID'):
            raise RuntimeError('Allocated compute is required for native API log staging')
        fs = filesystem_for(local_root, Path('/proc/self/mountinfo').read_text())
        return cls(local_root, output, fs)

    def publish(self, *, loggers_disabled, require_complete):
        if loggers_disabled is not True:
            raise RuntimeError('Native loggers must be disabled before copying')
        unexpected = {p.name for p in self.root.iterdir()} - set(LOG_NAMES)
        if unexpected:
            raise ValueError('Unexpected staging entries: '+repr(sorted(unexpected)))
        rows = []
        # Preflight all destinations before copying either file.
        for name in LOG_NAMES:
            src = self.root/name; dst = self.output/name
            if src.is_symlink() or dst.is_symlink():
                raise ValueError('Symlink log paths forbidden')
            if not src.exists():
                if require_complete:
                    raise FileNotFoundError(src)
                continue
            if not src.is_file():
                raise ValueError('Regular log files required')
            row = {'path': name, 'bytes': src.stat().st_size, 'sha256': digest(src)}
            if require_complete and row['bytes'] == 0:
                raise ValueError('Empty native API log')
            if dst.exists() and (not dst.is_file() or dst.stat().st_size != row['bytes'] or digest(dst) != row['sha256']):
                raise FileExistsError('Refusing to replace evidence: '+str(dst))
            rows.append(row)
        for row in rows:
            src = self.root/row['path']; dst = self.output/row['path']
            if dst.exists():
                continue  # Verified retry of a previous partial publication.
            fd, tmp_name = tempfile.mkstemp(prefix='.api-log-copy-', dir=self.output)
            tmp = Path(tmp_name)
            try:
                with os.fdopen(fd, 'wb') as target, src.open('rb') as source:
                    shutil.copyfileobj(source, target)
                    target.flush(); os.fsync(target.fileno())
                if tmp.stat().st_size != row['bytes'] or digest(tmp) != row['sha256']:
                    raise OSError('Copied log hash mismatch')
                os.link(tmp, dst)  # Atomic exclusive publication; cannot replace evidence.
            finally:
                tmp.unlink(missing_ok=True)
        self.published = True
        return {'schema': 'precision-log-staging-v1', 'filesystem': self.filesystem,
                'local_directory': str(self.root), 'files': rows,
                'complete': len(rows) == len(LOG_NAMES) and all(r['bytes'] for r in rows),
                'timing_scope': 'Untimed setup only; publish after native loggers disabled'}

    def cleanup(self):
        if not self.published:
            raise RuntimeError('Unpublished local evidence must be retained')
        # Recheck before deletion; destination loss/corruption must not erase originals.
        for src in self.root.iterdir():
            dst = self.output/src.name
            if src.is_symlink() or not src.is_file() or dst.is_symlink() or not dst.is_file() or digest(src) != digest(dst):
                raise OSError('Published evidence no longer matches; retaining local files')
        shutil.rmtree(self.root)


def save_record(path, record):
    import json
    path = Path(path)
    temp = path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(record, indent=2, allow_nan=False)+'\n')
    temp.replace(path)


class StagedCapture:
    """Own disable/flush/publication outside all generation measurement windows.

    Local files deliberately remain until the allocation wrapper observes process
    exit, so it can preserve partial logs if Python cleanup is interrupted.
    """
    def __init__(self, stage, logger, flush):
        self.stage, self.logger, self.flush = stage, logger, flush

    def __enter__(self):
        save_record(self.stage.output/'log-staging-location.json', {
            'schema': 'precision-log-location-v1', 'local_directory': str(self.stage.root),
            'output_directory': str(self.stage.output), 'filesystem': self.stage.filesystem})
        self.started = __import__('time').perf_counter()
        try:
            result = self.logger.begin('capture')
            if not result.get('available'):
                raise RuntimeError('Native API logger unavailable: '+repr(result))
            return result
        except BaseException:
            self.__exit__(*__import__('sys').exc_info())
            raise

    def __exit__(self, exc_type, exc, tb):
        import time
        record = {'schema': 'precision-staged-capture-v1', 'setup_succeeded': exc_type is None,
                  'setup_error': None if exc is None else repr(exc), 'logs_disabled': False,
                  'published': False, 'measurement_allowed': False}
        try:
            self.logger.end_verified()
            record['logs_disabled'] = True
            self.flush()
            copy_started = time.perf_counter()
            publication = self.stage.publish(loggers_disabled=True, require_complete=exc_type is None)
            record.update(published=True, publication=publication,
                          copy_s=time.perf_counter()-copy_started,
                          measurement_allowed=exc_type is None and publication['complete'])
        except BaseException as error:
            record['preservation_error'] = repr(error)
            raise
        finally:
            record['session_wall_s'] = time.perf_counter()-self.started
            save_record(self.stage.output/'log-staging.json', record)
        return False


def capture_session(output, local_root=None):
    """Return the original logger session or an explicitly requested local sink."""
    import contextlib
    import ctypes
    from .precision_dispatch import BlasLogs
    if local_root is None:
        @contextlib.contextmanager
        def original():
            logger = BlasLogs(output)
            try:
                yield logger.begin('capture')
            finally:
                logger.end()
        return original()
    stage = LogStage.for_allocation(local_root, output)
    logger = BlasLogs(stage.root)
    def flush():
        libc = ctypes.CDLL(None)
        libc.fflush.argtypes = [ctypes.c_void_p]
        libc.fflush.restype = ctypes.c_int
        if libc.fflush(None) != 0:
            raise OSError('Native log stream flush failed; retain local evidence')
    return StagedCapture(stage, logger, flush)
