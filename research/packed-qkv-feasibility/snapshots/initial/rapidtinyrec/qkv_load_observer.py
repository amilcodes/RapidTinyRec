"""Setup-only actual load_kernel provenance, before runtime clears cubin paths."""
import hashlib
from pathlib import Path
from .qkv_compile_result import snapshot_result


class LoadObserver:
    def __init__(self,is_capturing,artifact_root):
        self.is_capturing=is_capturing;self.root=Path(artifact_root);self.active={};self.loads={}
        self.root.mkdir(parents=True,exist_ok=True)
        if self.root.is_symlink():raise ValueError('Binary artifact root is symlink')

    def event(self,frame,event,fingerprint):
        if event=='call' and frame.f_code.co_name=='load_kernel' and frame.f_code.co_filename.endswith('/torch/_inductor/runtime/static_triton_launcher.py'):
            if self.is_capturing():raise ValueError('Kernel load during CUDA capture is unsupported')
            owner=frame.f_locals.get('self');state=vars(owner)
            if state.get('device_agnostic') is not False:raise ValueError('Only single-device kernel loads supported')
            if state.get('function') is not None:
                if id(owner) not in self.loads:raise ValueError('Kernel was loaded before setup observer')
                return
            path=state.get('cubin_path')
            if type(path) is not str:raise ValueError('Actual load lacks binary path')
            path=Path(path)
            if path.is_symlink() or path.resolve(strict=True)!=path or not path.is_file() or not 0<path.stat().st_size<=64*1024*1024:raise ValueError('Invalid actual load binary file')
            data=path.read_bytes();digest=hashlib.sha256(data).hexdigest();artifact=self.root/(digest+'.cubin')
            if artifact.exists():
                if artifact.is_symlink() or artifact.read_bytes()!=data:raise ValueError('Binary artifact changed')
            else:
                with artifact.open('xb') as stream:stream.write(data)
            name,khash=state.get('name'),state.get('hash');device=frame.f_locals.get('device')
            if type(name) is not str or type(khash) is not str or type(device) is not int or device!=0:raise ValueError('Invalid kernel load identity')
            caller=frame.f_back
            if caller.f_code.co_name!='make_launcher' or not caller.f_code.co_filename.endswith('/torch/_inductor/runtime/triton_heuristics.py'):raise ValueError('Load lacks actual make_launcher parent')
            compile_result=snapshot_result(caller.f_locals.get('self'),owner)
            maker_source=Path(caller.f_code.co_filename).read_bytes()
            source=Path(frame.f_code.co_filename).read_bytes()
            self.active[id(frame)]=dict(owner=owner,identity=dict(owner_id=id(owner),name=name,kernel_hash=khash,device=device,binary_sha256=digest,binary_bytes=len(data),artifact=artifact.name,source_sha256=hashlib.sha256(source).hexdigest(),load_code_sha256=fingerprint(frame.f_code),compile_result=compile_result,maker_source_sha256=hashlib.sha256(maker_source).hexdigest(),maker_code_sha256=fingerprint(caller.f_code)),path=path)
        elif event=='return' and id(frame) in self.active:
            if self.is_capturing():raise ValueError('Kernel load returned during capture')
            pending=self.active.pop(id(frame));state=vars(pending['owner']);identity=pending['identity']
            if hashlib.sha256(pending['path'].read_bytes()).hexdigest()!=identity['binary_sha256']:raise ValueError('Binary file changed during load')
            for key in ('function','module'):
                value=state.get(key)
                if type(value) is not int or value<=0:raise ValueError('Kernel load did not publish valid handles')
                identity[key]=value
            if state.get('name')!=identity['name'] or state.get('hash')!=identity['kernel_hash']:raise ValueError('Kernel identity changed during load')
            if identity['owner_id'] in self.loads:raise ValueError('Duplicate kernel object load')
            self.loads[identity['owner_id']]=identity

    def bind(self,call):
        for launch in call['launch_observations']:
            runner=launch['static_runner'];m=runner['metadata'];load=self.loads.get(m['owner_id'])
            if load is None or any(load[k]!=m[v] for k,v in [('name','name'),('kernel_hash','hash'),('function','function'),('module','module')]):raise ValueError('Captured runner has no exact observed binary load')
            runner['load_identity']=dict(load)
        return call


def validate_load_identities(root,report,installed):
    root=Path(root);checked={};count=0
    for graph in report['graphs']:
        for call in graph['wrappers']:
            for launch in call['launch_observations']:
                runner=launch['static_runner'];m=runner['metadata'];load=runner['load_identity']
                for key,other in [('owner_id','owner_id'),('name','name'),('kernel_hash','hash'),('function','function'),('module','module')]:
                    if load[key]!=m[other]:raise ValueError('Runner/load identity mismatch')
                if load['device']!=0 or load['source_sha256']!=installed['static_runner']['source_sha256'] or load['load_code_sha256']!=installed['static_runner']['load_code_sha256']:raise ValueError('Load method is not independently installed source/code')
                if load['maker_source_sha256']!=installed['source_sha256'] or load['maker_code_sha256']!=installed['maker_code_sha256']:raise ValueError('Compile-result maker differs from independently installed source/code')
                digest=load['binary_sha256']
                if len(digest)!=64 or any(c not in '0123456789abcdef' for c in digest) or load['artifact']!=digest+'.cubin':raise ValueError('Noncanonical binary artifact identity')
                path=root/'loaded-binaries'/load['artifact']
                if path.is_symlink() or path.parent.is_symlink() or not path.is_file():raise ValueError('Missing/aliased binary artifact')
                if digest not in checked:
                    data=path.read_bytes()
                    if hashlib.sha256(data).hexdigest()!=digest:raise ValueError('Loaded binary artifact hash mismatch')
                    checked[digest]=len(data)
                if checked[digest]!=load['binary_bytes']:raise ValueError('Loaded binary artifact size mismatch')
                count+=1
    if not count:raise ValueError('No loaded runner provenance')
    return dict(captured_runners=count,unique_binary_artifacts=len(checked),actual_binary_load_identity_verified=True,
                source_to_binary_compilation_verified=False,complete_dispatch_verified=False)
