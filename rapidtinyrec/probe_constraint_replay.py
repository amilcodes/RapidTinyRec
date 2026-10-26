"""Allocated-GPU warmup probes; preserve numeric failures as evidence."""
import torch

from .gpu_constraints import GpuSidConstrainedLogitsProcessor


def compare_tensors(left, right):
    same_finite = torch.isfinite(left) & torch.isfinite(right)
    delta = torch.where(same_finite, (left-right).abs(), 0.)
    return dict(values=left.numel(), matching_bits=int((left.view(torch.int32)==right.view(torch.int32)).sum()),
        max_abs_finite=float(delta.max()), finite_mask_equal=bool(torch.equal(torch.isfinite(left),torch.isfinite(right))),
        no_nan=bool(not torch.isnan(left).any() and not torch.isnan(right).any()))


@torch.inference_mode()
def probe(pool, constraints, vocabulary):
    tables=pool.tables; device=tables.eos.device
    root=tuple(tables.root_prefix.cpu().tolist())
    sid=next(k for k in sorted(constraints) if len(k)==3 and k!=root)
    newline=int(tables.newline); eos=int(tables.eos)
    suffixes=[root, (0,0,sid[0]), (0,*sid[:2]), sid, (*sid[1:],newline)]
    scenarios=('valid','unknown','mixed','valid_repeat')
    rows=[]; retained_rows=[]
    # Each scenario is a new request with changed scores and prompt lengths.
    for scenario_index, scenario in enumerate(scenarios):
        lease=pool.processor(); original=GpuSidConstrainedLogitsProcessor(tables)
        held=[]
        try:
            for step in range(5):
                ids=torch.full((50,76 if scenario_index%2==0 else 121),7,dtype=torch.long,device=device)
                ids[:,-3:]=torch.tensor(suffixes[step],dtype=torch.long,device=device)
                unknown_rows=range(50) if scenario=='unknown' else range(1,50,2) if scenario=='mixed' else ()
                # Out-of-range token IDs exercise the existing EOS fallback.
                for row in unknown_rows:ids[row,-3:]=-1
                scores=torch.randn((50,vocabulary),dtype=torch.float32,device=device)
                scores.add_(step/8+scenario_index/16)
                expected=original(ids,scores)
                actual=lease(ids,scores)
                detail=compare_tensors(expected,actual)
                detail.update(scenario=scenario,step=step,score_shape=list(scores.shape),prompt_tokens=ids.shape[1])
                detail['unknown_fallback_checked_rows']=len(unknown_rows)
                if len(unknown_rows):
                    # Negative token prefixes can only permit EOS on every step.
                    bad=actual[list(unknown_rows)]
                    detail['unknown_fallback_correct']=bool((torch.isfinite(bad).sum(dim=1)==1).all() and torch.isfinite(bad[:,eos]).all())
                else:detail['unknown_fallback_correct']=True
                rows.append(detail)
                held.append((actual,actual.clone()))
            pool.check_ownership()
            assert len({x.untyped_storage().data_ptr() for x,_ in held})==5
            for step,(actual,saved) in enumerate(held):
                detail=compare_tensors(saved,actual)
                detail.update(scenario=scenario,step=step)
                retained_rows.append(detail)
        finally:
            lease.close()
        del held,actual,saved,expected,scores
    pool.freeze()
    return dict(complete=True,graph_direct=rows,retained_after_five_steps=retained_rows,
                independent_step_outputs=True,scenarios=list(scenarios),replay=pool.report(),
                scope='Warmup-only synthetic FP32 score inputs with catalog-valid, unknown and mixed prefixes. All twenty graph/direct and retained-output comparisons are saved even when bits differ. Actual generation outputs are audited separately.')
