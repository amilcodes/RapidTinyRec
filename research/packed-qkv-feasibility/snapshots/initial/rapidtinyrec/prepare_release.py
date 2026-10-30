"""Fetch the public pinned release, then verify its hashes."""
import argparse
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        while chunk := f.read(8*1024*1024):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--release-root', type=Path, required=True)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    expected = json.loads((ROOT/'expected.json').read_text())
    if not args.verify_only:

        from huggingface_hub import hf_hub_download
        args.release_root.mkdir(parents=True, exist_ok=True)
        for relative in expected['release_sha256']:
            hf_hub_download(repo_id=expected['repo_id'], revision=expected['revision'],
                            filename=relative, local_dir=args.release_root, token=False)
    for relative, want in expected['release_sha256'].items():
        got = sha256(args.release_root/relative)
        assert got == want, f'Hash mismatch for {relative}: {got} != {want}'
        print(json.dumps({'verified': relative, 'sha256': got}), flush=True)


if __name__ == '__main__':
    main()
