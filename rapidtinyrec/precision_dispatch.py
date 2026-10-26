"""Native API diagnostics for untimed setup/capture only; no numerical changes."""
import ctypes
import os
from pathlib import Path

class BlasLogs:
    """Read-only library diagnostics, enabled only for untimed probe calls."""
    def __init__(self, root):
        self.root = root
        self.error = None
        try:
            maps = Path('/proc/self/maps').read_text().splitlines()
            def lib(stem):
                paths = {line.split()[-1] for line in maps if '/' in line and line.split()[-1].split('/')[-1].startswith(stem + '.so.')}
                if len(paths) != 1:
                    raise RuntimeError(f'Expected one loaded {stem}: {sorted(paths)}')
                return ctypes.CDLL(paths.pop())
            self.blas, self.lt = lib('libcublas'), lib('libcublasLt')
            self.blas.cublasLoggerConfigure.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_char_p]
            self.lt.cublasLtLoggerOpenFile.argtypes = [ctypes.c_char_p]
            self.lt.cublasLtLoggerSetLevel.argtypes = [ctypes.c_int]
        except Exception as exc:
            self.error = repr(exc)

    def begin(self, prefix):
        if self.error:
            return {'available': False, 'error': self.error}
        a, b = prefix + '-cublas.log', prefix + '-cublaslt.log'
        try:
            codes = [self.blas.cublasLoggerConfigure(1, 0, 0, os.fsencode(self.root / a)),
                     self.lt.cublasLtLoggerOpenFile(os.fsencode(self.root / b)), self.lt.cublasLtLoggerSetLevel(5)]
            if any(codes):
                raise RuntimeError(f'Logger status {codes}')
            return dict(available=True, cublas=a, cublaslt=b)
        except Exception as exc:
            self.end()
            return {'available': False, 'error': repr(exc)}

    def end(self):
        if not self.error:
            self.blas.cublasLoggerConfigure(0, 0, 0, None)
            self.lt.cublasLtLoggerSetLevel(0)


    def end_verified(self):
        """Staged logging must fail closed if diagnostics cannot be disabled."""
        if self.error:
            raise RuntimeError(self.error)
        codes = [self.blas.cublasLoggerConfigure(0, 0, 0, None),
                 self.lt.cublasLtLoggerSetLevel(0)]
        if any(codes):
            raise RuntimeError(f'Native logger disable failed: {codes}')
