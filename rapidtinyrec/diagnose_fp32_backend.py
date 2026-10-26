"""Bounded native FP32 GEMM diagnostic on allocated A6000 compute only."""
import argparse
import collections
import ctypes
import datetime
import gzip
import hashlib
import json
import os
from pathlib import Path
import statistics
import time
import traceback

from .fp32_backend_probe import BACKENDS, USERS, LAYERS, ROLES, TORCH_COMMIT, backend_scope, validate_tensors


def save(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def raw(tensor):
    return tensor.detach().contiguous().cpu().numpy().astype('<f4', copy=False).tobytes()


class TensorStore:
    def __init__(self, root):
        self.root = root
        self.entries = {}
        (root / 'tensors').mkdir()

    def put(self, tensor):
        assert str(tensor.dtype) == 'torch.float32', 'Raw evidence requires FP32'
        data = raw(tensor)
        digest = hashlib.sha256(data).hexdigest()
        if digest not in self.entries:
            name = f'tensors/{digest}.f32.gz'
            encoded = gzip.compress(data, mtime=0)
            (self.root / name).write_bytes(encoded)
            self.entries[digest] = dict(file=name, bytes=len(data), compressed_bytes=len(encoded),
                                       compressed_sha256=hashlib.sha256(encoded).hexdigest())
        return dict(sha256=digest, shape=list(tensor.shape), stride=list(tensor.stride()),
                    dtype=str(tensor.dtype), storage_offset=tensor.storage_offset())


def memory(torch):
    return dict(allocated=torch.cuda.memory_allocated(), reserved=torch.cuda.memory_reserved(),
                peak_allocated=torch.cuda.max_memory_allocated(), peak_reserved=torch.cuda.max_memory_reserved())


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


def run(args):
    assert os.environ.get('SLURM_JOB_ID'), 'Allocated compute only'
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, LogitsProcessorList
    from .folded_attention import fold_decode_attention
    from .shared_prefill import PrefillOnce
    from .gpu_constraints import GpuSidConstraintTables, GpuSidConstrainedLogitsProcessor
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    assert torch.cuda.get_device_name() == 'NVIDIA RTX A6000'
    args.output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    report = dict(schema=1, complete=False, job_id=os.environ['SLURM_JOB_ID'],
                  hardware=torch.cuda.get_device_name(), torch=torch.__version__, cuda=torch.version.cuda,
                  torch_git_version=torch.version.git_version, transformers=transformers.__version__,
                  scope='Original-shaped fixed-input projection microdiagnostic. Eager folded/shared-prefill activations; not saved compiled intermediates or generation latency.',
                  cases=[], setup={}, runtime_precision={}, source_sha256={})
    save(args.output / 'diagnosis.json', report)
    try:
        assert torch.__version__ == '2.14.0+cu126' and torch.version.cuda == '12.6'
        assert torch.version.git_version == TORCH_COMMIT, 'Review installed build before relying on pinned dispatch source'
        assert transformers.__version__ == '4.55.4'
        torch.set_num_threads(4); torch.manual_seed(42)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.cuda.tunable.enable(False); torch.cuda.tunable.tuning_enable(False)
        assert not torch.cuda.tunable.is_enabled() and not torch.cuda.tunable.tuning_is_enabled()
        api = torch.backends.cuda.preferred_blas_library
        initial_backend = api()
        assert str(initial_backend).endswith('.Cublas'), 'Unexpected initial backend preference'
        report['runtime_precision'] = dict(tf32=torch.backends.cuda.matmul.allow_tf32,
            cudnn_tf32=torch.backends.cudnn.allow_tf32, matmul_precision=torch.get_float32_matmul_precision(),
            initial_backend=str(initial_backend), tunable_enabled=torch.cuda.tunable.is_enabled(),
            tuning_enabled=torch.cuda.tunable.tuning_is_enabled(),
            cublas_workspace_bytes=torch.backends.cuda.cublas_workspace_size(),
            cublaslt_workspace_bytes=torch.backends.cuda.cublaslt_workspace_size())
        for path in sorted(Path('rapidtinyrec').rglob('*')):
            if path.is_file() and '__pycache__' not in path.parts:
                report['source_sha256'][str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        expected = json.loads(Path('rapidtinyrec/expected.json').read_text())
        for name, digest in expected['release_sha256'].items():
            assert hashlib.sha256((args.release / name).read_bytes()).hexdigest() == digest, name
        historical = json.loads((args.historical / 'manifest.json').read_text())
        cohort_sha = hashlib.sha256(json.dumps(historical['users'], sort_keys=True).encode()).hexdigest()
        assert cohort_sha == expected['users_sha256']
        report['cohort_sha256'] = cohort_sha
        report['release_sha256'] = expected['release_sha256']
        users = [historical['users'][i] for i in USERS]
        assert [u['cohort_index'] for u in users] == list(USERS)
        tok = AutoTokenizer.from_pretrained(args.release / 'Office_ckpt', local_files_only=True)
        tok.pad_token = tok.eos_token
        constraints = collections.defaultdict(set)
        for line in (args.release / 'Amazon/info/Office_Products_5_2016-10-2018-11.txt').read_text().splitlines():
            ids = tok('### Response:\n' + line.split('\t')[0].strip() + '\n').input_ids + [tok.eos_token_id]
            for j in range(3, len(ids)):
                constraints[tuple(ids[:j] if j == 3 else ids[3:j])].add(ids[j])
        tables = GpuSidConstraintTables({k: list(v) for k, v in constraints.items()}, users[0]['prompt_token_ids'][-3:], 198, tok.eos_token_id, 'cuda')
        load_start = time.perf_counter()
        model = AutoModelForCausalLM.from_pretrained(args.release / 'Office_ckpt', torch_dtype=torch.float32,
                    local_files_only=True, attn_implementation='sdpa').to('cuda').eval()
        torch.cuda.synchronize()
        report['model_load_s'] = time.perf_counter() - load_start
        cfg = GenerationConfig(max_new_tokens=8, length_penalty=0., num_beams=50, num_return_sequences=50,
            pad_token_id=tok.eos_token_id, eos_token_id=tok.eos_token_id, bos_token_id=tok.bos_token_id,
            do_sample=False, top_k=None, top_p=None, use_cache=True)
        captured = {}
        def hook(layer, kind, width):
            def capture(module, inputs):
                x = inputs[0]
                if tuple(x.shape) == (50, 1, width) and (layer, kind) not in captured:
                    captured[layer, kind] = x[:, 0, :].clone()
            return capture
        handles = []
        for layer in LAYERS:
            mlp = model.model.layers[layer].mlp
            handles += [mlp.register_forward_pre_hook(hook(layer, 'gate_up', 1536)),
                        mlp.down_proj.register_forward_pre_hook(hook(layer, 'down', 8960))]
        cases = []
        collect_start = time.perf_counter()
        try:
            with fold_decode_attention(True):
                for user in users:
                    captured.clear()
                    ids = torch.tensor([user['prompt_token_ids']], device='cuda')
                    with PrefillOnce(model, 50):
                        model.generate(ids, attention_mask=torch.ones_like(ids), generation_config=cfg,
                            logits_processor=LogitsProcessorList([GpuSidConstrainedLogitsProcessor(tables)]))
                    assert len(captured) == 6
                    for layer in LAYERS:
                        cases.append((user, layer, captured[layer, 'gate_up'], captured[layer, 'down']))
        finally:
            for h in handles: h.remove()
        torch.cuda.synchronize()
        report['setup']['activation_collection_s'] = time.perf_counter() - collect_start
        report['memory_after_collection'] = memory(torch)
        report['model_parameter_logical_bytes'] = sum(p.numel() * p.element_size() for p in model.parameters())
        weights = {p.untyped_storage().data_ptr(): p.untyped_storage().nbytes() for p in model.parameters()}
        report['model_parameter_unique_bytes'] = sum(weights.values())
        store = TensorStore(args.output)
        logs = BlasLogs(args.output)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.inference_mode(), torch.cuda.stream(stream):
            for user, layer, x, down_x in cases:
                case_start = time.perf_counter()
                mlp = model.model.layers[layer].mlp
                ws = dict(gate=mlp.gate_proj.weight, up=mlp.up_proj.weight, down=mlp.down_proj.weight)
                original = dict(gate=x, up=x, down=down_x)
                staged = {r: original[r].clone() for r in ROLES}
                case = dict(cohort_index=user['cohort_index'], prompt_tokens=len(user['prompt_token_ids']), layer=layer,
                            inputs={r: store.put(original[r]) for r in ROLES},
                            weight_sha256={r: hashlib.sha256(raw(ws[r])).hexdigest() for r in ROLES},
                            weight_shapes={r: list(ws[r].shape) for r in ROLES},
                            variants={}, graph_count=0, staging_storage_disjoint=True)
                pools = {}
                for backend in BACKENDS:
                    with backend_scope(api, backend) as selected:
                        detail = dict(requested_backend=backend, selected_backend=selected, states={}, projection_us={r: [] for r in ROLES})
                        out = {r: torch.empty((50, ws[r].shape[0]), device='cuda', dtype=torch.float32) for r in ROLES}
                        graphs = {}
                        for role in ROLES:
                            validate_tensors(staged[role], ws[role], out[role], role)
                            for _ in range(3): torch.mm(staged[role], ws[role].t(), out=out[role])
                            torch.cuda.synchronize()
                            graph = torch.cuda.CUDAGraph()
                            capture_start = time.perf_counter()
                            with torch.cuda.graph(graph, stream=stream):
                                torch.mm(staged[role], ws[role].t(), out=out[role])
                            detail.setdefault('capture_warmup_s', {})[role] = time.perf_counter() - capture_start
                            graphs[role] = graph
                            case['graph_count'] += 1
                        pools[backend] = (graphs, out)
                        detail['output_ptrs'] = {r: out[r].untyped_storage().data_ptr() for r in ROLES}
                        # Original, row-reversed changed input, then restored original.
                        for state in ('original', 'changed', 'restored'):
                            for role in ROLES:
                                staged[role].copy_(original[role].flip(0) if state == 'changed' else original[role])
                            refs = {'inputs': {r: store.put(staged[r]) for r in ROLES}, 'roles': {}}
                            for role in ROLES:
                                direct = torch.empty_like(out[role]); repeat = torch.empty_like(out[role])
                                torch.mm(staged[role], ws[role].t(), out=direct)
                                torch.mm(staged[role], ws[role].t(), out=repeat)
                                graphs[role].replay(); torch.cuda.synchronize()
                                refs['roles'][role] = dict(direct=store.put(direct), direct_repeat=store.put(repeat),
                                    graph=store.put(out[role]))
                                graphs[role].replay(); torch.cuda.synchronize()
                                refs['roles'][role]['graph_repeat'] = store.put(out[role])
                            # Later-role graph executions must not overwrite earlier role outputs.
                            for role in ROLES:
                                refs['roles'][role]['retained_after_other_roles'] = store.put(out[role])
                            detail['states'][state] = refs
                        prefix = f'dispatch-u{user["cohort_index"]}-l{layer}-{backend}'
                        detail['logs'] = logs.begin(prefix)
                        try:
                            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                                   torch.profiler.ProfilerActivity.CUDA]) as prof:
                                for role in ROLES:
                                    with torch.profiler.record_function('fp32_probe_' + role):
                                        torch.mm(staged[role], ws[role].t(), out=out[role])
                                torch.cuda.synchronize()
                            trace = prefix + '.json'
                            prof.export_chrome_trace(str(args.output / trace))
                            detail['trace'] = trace
                        finally:
                            logs.end()
                        case['variants'][backend] = detail
                # All output storages are independent across role and backend pools.
                pointers = [ptr for d in case['variants'].values() for ptr in d['output_ptrs'].values()]
                assert len(set(pointers)) == 9
                stage_ptrs = [s.untyped_storage().data_ptr() for s in staged.values()]
                assert len(set(stage_ptrs)) == 3 and not set(stage_ptrs) & set(pointers)
                case['staging_ptrs'] = stage_ptrs
                case['input_logical_bytes'] = sum(t.numel()*4 for t in staged.values())
                case['output_logical_bytes'] = sum(o.numel()*4 for _, os_ in pools.values() for o in os_.values())
                case['unique_input_output_bytes'] = sum(t.untyped_storage().nbytes() for t in staged.values()) + sum(o.untyped_storage().nbytes() for _, os_ in pools.values() for o in os_.values())
                case['setup_s'] = time.perf_counter() - case_start
                case['before_timing_memory'] = memory(torch)
                torch.cuda.reset_peak_memory_stats()
                for round_index in range(8):
                    order = BACKENDS if round_index % 2 == 0 else tuple(reversed(BACKENDS))
                    for backend in order:
                        for role in ROLES:
                            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                            begin.record(stream)
                            for _ in range(32): pools[backend][0][role].replay()
                            end.record(stream); end.synchronize()
                            case['variants'][backend]['projection_us'][role].append(begin.elapsed_time(end)*1000/32)
                case['timing_memory'] = memory(torch)
                case['timed_graph_replays'] = 8*3*3*32
                case['measured_captures'] = case['measured_compilations'] = 0
                for backend in BACKENDS:
                    detail = case['variants'][backend]
                    detail['after_timing'] = {r: store.put(pools[backend][1][r]) for r in ROLES}
                    detail['timing_summary'] = {r: dict(median_us=statistics.median(v),
                        p95_us=statistics.quantiles(v,n=100,method='inclusive')[94],min_us=min(v),max_us=max(v)) for r,v in detail['projection_us'].items()}
                report['cases'].append(case)
                report['tensors'] = store.entries
                save(args.output / 'diagnosis.json', report)
                print(json.dumps({'completed_case': [user['cohort_index'], layer]}), flush=True)
                del pools, graphs, out, staged, direct, repeat, graph
        assert api() == initial_backend
        report['restored_backend'] = str(api())
        report['total_wall_s'] = time.perf_counter() - start
        report['memory_final'] = memory(torch)
        report['complete'] = True
        save(args.output / 'diagnosis.json', report)
    except Exception:
        report['failure'] = traceback.format_exc()
        save(args.output / 'diagnosis.json', report)
        raise


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--release', type=Path, required=True)
    ap.add_argument('--historical', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    # Inference mode also covers activation collection and weight ownership.
    import torch
    with torch.inference_mode(): run(args)


if __name__ == '__main__': main()
