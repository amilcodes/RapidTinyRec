"""Build an exclusive local serving snapshot from one clean Git commit; no SSH."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import subprocess

from .serving_allocation import CONFIG, COMMAND, PROJECT, SSH_TARGET, REMOTE_ROOT, require, identical
from .serving_contract import PROFILE, PROTOCOL, PROTOCOL_SHA256, load_protocol
from .serving_setup_audit import read_unique

TEMPLATES='benchmarks/preparation/serving-throughput/'


def git(repo,*args):
    return subprocess.check_output(['git',*args],cwd=repo)


def repository_mapping(names):
    mapping={n:n for n in names if n.startswith('rapidtinyrec/') or (n.startswith('tests/test_serving') and n.endswith('.py'))}
    mapping.update({PROTOCOL:PROTOCOL,CONFIG:TEMPLATES+CONFIG,COMMAND[1]:TEMPLATES+'serving-throughput.sh'})
    require(set(mapping.values())<=set(names),'Required committed serving templates absent')
    require({'rapidtinyrec/serving_runner.py','rapidtinyrec/serving_run_audit.py',
             'rapidtinyrec/serving_terminal_audit.py','rapidtinyrec/source/LogitProcessor.py','rapidtinyrec/expected.json'}<=set(mapping),
            'Recursive serving source dependencies absent')
    for name in (*mapping,*mapping.values()):
        path=PurePosixPath(name)
        require(not path.is_absolute() and '..' not in path.parts and path.as_posix()==name and
                '__pycache__' not in path.parts and path.suffix not in ('.pyc','.pyo'),'Non-source snapshot entry')
    return mapping


def validate_provenance(value, hashes):
    require(value['schema']=='serving-source-provenance-v1' and
            re.fullmatch('[0-9a-f]{40}',value['source_commit']) is not None and
            value['protocol_sha256']==PROTOCOL_SHA256 and value['submission_ready'] is False,
            'Serving source provenance identity differs')
    expected={k:v for k,v in hashes.items() if k!='source-provenance.json'}
    require(identical(value['source_files'],expected) and set(value['repository_paths'])==set(expected),
            'Serving provenance source inventory differs')
    mapping=value['repository_paths']
    require(len(set(mapping.values()))==len(mapping),'Aliased repository source mapping')
    for name,repo in mapping.items():
        expected_path=TEMPLATES+CONFIG if name==CONFIG else TEMPLATES+'serving-throughput.sh' if name==COMMAND[1] else name
        require(repo==expected_path,'Unexpected serving repository path mapping')
    require(repository_mapping(set(mapping.values()))==mapping,'Incomplete serving repository mapping')


def validate_snapshot(root, repository=None):
    root=Path(root);hashes={};inodes=set()
    require(root.is_dir() and not root.is_symlink(),'Regular harness directory required')
    for path in root.rglob('*'):
        require(not path.is_symlink(),'Linked harness entry')
        if path.is_dir():continue
        stat=path.stat();identity=(stat.st_dev,stat.st_ino)
        require(path.is_file() and stat.st_nlink==1 and identity not in inodes,'Aliased harness entry');inodes.add(identity)
        hashes[path.relative_to(root).as_posix()]=hashlib.sha256(path.read_bytes()).hexdigest()
    provenance=read_unique(root/'source-provenance.json');validate_provenance(provenance,hashes)
    protocol=load_protocol(root/PROTOCOL);config=read_unique(root/CONFIG)
    require(config['project']==PROJECT and config['ssh_target']==SSH_TARGET and config['remote_root']==REMOTE_ROOT and
            identical(config['profiles'],{PROFILE:dict(protocol['resources'],setup=['module purge'],command=COMMAND)}),
            'Serving harness configuration differs')
    if repository is not None:
        for target,source in provenance['repository_paths'].items():
            require((root/target).read_bytes()==git(repository,'show',provenance['source_commit']+':'+source),
                    'Harness differs from pinned repository bytes')
    return dict(source_commit=provenance['source_commit'],files=len(hashes),bytes=sum((root/n).stat().st_size for n in hashes),
                sha256=hashes,submission_ready=False)


def prepare(repository, destination):
    repo=Path(repository).resolve(strict=True);out=Path(destination).absolute()
    require(not out.exists(),'Harness destination already exists; no replacement')
    require(not git(repo,'status','--porcelain').strip(),'Clean committed serving source required')
    head=git(repo,'rev-parse','HEAD').decode().strip()
    names=git(repo,'ls-tree','-r','--name-only',head).decode().splitlines();mapping=repository_mapping(names)
    data={target:git(repo,'show',head+':'+source) for target,source in mapping.items()}
    for target,source in mapping.items():
        require(not (repo/source).is_symlink() and (repo/source).read_bytes()==data[target],'Working source differs from commit')
    hashes={name:hashlib.sha256(raw).hexdigest() for name,raw in data.items()}
    provenance=dict(schema='serving-source-provenance-v1',source_commit=head,protocol_sha256=PROTOCOL_SHA256,
                    repository_paths=mapping,source_files=hashes,submission_ready=False,
                    purpose='Exact-serving CPU preparation; submission guard and six-lane budget review required.')
    data['source-provenance.json']=(json.dumps(provenance,indent=2)+'\n').encode()
    out.mkdir()
    for name,raw in data.items():
        path=out/name;path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('xb') as handle:handle.write(raw)
    return validate_snapshot(out,repo)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repository',type=Path,required=True);parser.add_argument('--destination',type=Path,required=True)
    args=parser.parse_args();print(json.dumps(prepare(args.repository,args.destination),indent=2))


if __name__=='__main__':main()
