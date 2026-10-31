import copy
import json
from pathlib import Path
import tempfile
import unittest
from rapidtinyrec.audit_beam_wave_dispatch import inspect_dispatch,validate_replay_counts


def report(root,wave=True,api_batch=None,launches=None,correlated=True):
    batch=(100 if wave else 50) if api_batch is None else api_batch
    n=(4 if wave else 8) if launches is None else launches
    (root/'api.log').write_text(''.join(f'I! cuBLAS 12 function cublasSgemm(\nm: type=int; val={m}\nn: type=int; val={batch}\nk: type=int; val={k}\nMathMode=CUBLAS_TF32_TENSOR_OP_MATH\n' for m,k in ((8960,1536),(1536,8960))))
    profiles=[]
    for length,users in ((76,[1,2]),(121,[6,10])):
        events=[dict(name='cudaGraphLaunch',cat='cuda_runtime',args={'correlation':i}) for i in range(n)]
        events += [dict(name='ampere_s1688tf32_kernel',cat='kernel',dur=1,args={'correlation':i if correlated else 999}) for i in range(n)]
        name=f'{length}.json';(root/name).write_text(json.dumps({'traceEvents':events}))
        profiles.append(dict(prompt_tokens=length,cohort_indices=users,trace=name))
    return dict(capture_logs=dict(available=True,cublas='api.log'),profiles=profiles)


def counters(wave=True):
    def stage(n):
        prefix=dict(physical_prefix_batch=2 if wave else 1,suffix_batch=100 if wave else 50,max_prefix_tokens=121,
                    max_suffix_tokens=3,requests_loaded=2*n if wave else n)
        if wave:prefix['waves_loaded']=n
        replay=dict(graphs=8,frozen=True,replays=4*n,physical_prefix=prefix,
            events=[dict(cache_tokens=length+i,batch=100 if wave else 50) for length in (76,121) for i in range(4)])
        reorder=dict(direct_calls=4*n,fallback_calls=n,prefix_bytes_reordered=0)
        return replay,reorder
    setup={};summary=dict(complete=True,measured_pairs=6,measured_requests=12);dispatch={}
    setup['replay'],setup['direct_reorder']=stage(4 if wave else 8)
    summary['replay'],summary['direct_reorder']=stage(10 if wave else 20)
    dispatch['replay_after_profiles'],dispatch['direct_reorder_after_profiles']=stage(12 if wave else 24)
    return setup,summary,dispatch


class WaveDispatchTests(unittest.TestCase):
    def test_both_native_batch_shapes_and_graph_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for wave in (True,False):
                result=inspect_dispatch(root,report(root,wave),'beam_wave_tf32' if wave else 'serial_pair_tf32')
                self.assertTrue(result['verified']);self.assertEqual(result['decode_batch'],100 if wave else 50)

    def test_serial_batch_probe_cannot_prove_wave_dispatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);r=inspect_dispatch(root,report(root,api_batch=50),'beam_wave_tf32')
            self.assertFalse(r['verified']);self.assertFalse(r['native_policy_verified'])

    def test_unrelated_tf32_and_missing_or_extra_graph_launches_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for opts in ({'correlated':False},{'launches':3},{'launches':5}):
                self.assertFalse(inspect_dispatch(root,report(root,**opts),'beam_wave_tf32')['verified'])

    def test_wrong_profile_pair_and_path_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);r=report(root);r['profiles'][0]['cohort_indices']=[6,10]
            with self.assertRaises(ValueError):inspect_dispatch(root,r,'beam_wave_tf32')
            r=report(root);r['capture_logs']['cublas']='../api.log'
            with self.assertRaises(ValueError):inspect_dispatch(root,r,'beam_wave_tf32')

    def test_expected_phase_replay_reorder_and_residency_counts(self):
        for phase in range(4):self.assertTrue(validate_replay_counts(phase,*counters(phase in (1,2)))['phase_inventory_verified'])

    def test_missing_replay_wrong_batch_or_request_count_rejected(self):
        for change in ('replay','batch','physical','reorder','incomplete'):
            a,b,c=counters()
            if change=='replay':b['replay']['replays']-=4
            elif change=='batch':a['replay']['events'][0]['batch']=50
            elif change=='physical':b['replay']['physical_prefix']['requests_loaded']=10
            elif change=='reorder':c['direct_reorder_after_profiles']['fallback_calls']=0
            else:b['complete']=False
            with self.subTest(change=change),self.assertRaises(ValueError):validate_replay_counts(1,a,b,c)


if __name__=='__main__':unittest.main()
