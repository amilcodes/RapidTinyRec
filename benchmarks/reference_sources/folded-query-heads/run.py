"""Launch a benchmark in a fresh process with isolated compiler caches."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import tempfile

MODES = {
    'prefill': ('benchmark', ['--experiment', 'shared_prefill']),
    'batched': ('benchmark', ['--experiment', 'bulk_constraints']),
    'gpu': ('benchmark', ['--experiment', 'gpu_constraints']),
    'compiled': ('benchmark', ['--experiment', 'compile_forward']),
    'gqa': ('benchmark_gqa', []),
    'decode-pilot': ('pilot_decode_replay', []),
    'profile-shared': ('profile_shared', []),
    'profile-compiled': ('profile_compiled', []),
}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=MODES)
    parser.add_argument('--cache-root',type=Path,default=Path('compiler-cache'))
    args, extra=parser.parse_known_args()
    module, flags=MODES[args.mode]
    env=dict(os.environ)
    env['TOKENIZERS_PARALLELISM']='false'
    if args.mode in ('compiled','gqa','decode-pilot','profile-compiled'):
        args.cache_root.mkdir(parents=True,exist_ok=True)
        root=Path(tempfile.mkdtemp(prefix=args.mode+'-',dir=args.cache_root)).resolve()
        env['TORCHINDUCTOR_CACHE_DIR']=str(root/'inductor')
        env['TRITON_CACHE_DIR']=str(root/'triton')
        print(f'Fresh compiler cache: {root}',flush=True)
    raise SystemExit(subprocess.call([sys.executable,'-m','rapidtinyrec.'+module,*flags,*extra],env=env))


if __name__=='__main__':
    main()
