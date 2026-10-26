import copy
import json
from pathlib import Path
import tempfile
import unittest
from rapidtinyrec.audit_precision import fidelity_gate, performance_gate, validate_run, validate_output_storage
from rapidtinyrec.audit_precision_dispatch import api_gemms, dispatch_evidence
from rapidtinyrec.audit_decode_pilot import compare


def run():
    return dict(token_ids=[[i,1,2,198,151645] for i in range(50)],sequence_logp=[-5.]*50,
        transition_logp=[[-1.]*5 for _ in range(50)],elapsed_s=.1,
        saved_score_storage_ptrs=[1,2,3,4,5],saved_score_dtypes=['torch.float32']*5,
        beam_indices=[[i]*5 for i in range(50)])


def block(mode, m=8960,n=50,k=1536):
    return f'I! cuBLAS (v12.3) function cublasStatus_t cublasSgemm_v2(...) called:\ni! m: type=int; val={m}\ni! n: type=int; val={n}\ni! k: type=int; val={k}\ni!Process=1; MathMode={mode}\n'


class PrecisionAuditTests(unittest.TestCase):
    def test_score_drift_allowed_only_between_policies(self):
        a=run();b=copy.deepcopy(a);b['sequence_logp'][0]=-5.25;b['transition_logp'][0][0]=-1.25
        valid=compare([(a,a)]);drift=compare([(a,b)])
        r=dict(historical_comparisons={'folded_fp32':valid},comparison=drift,repeated_comparison=valid,ancestry_stable=True,
            phases=[dict(variant='folded_fp32',warmup_historical_comparison=valid,warmup_comparison=valid,repeat_comparison=valid,profile_comparison=valid)])
        self.assertEqual({k:v for k,v in fidelity_gate(r).items() if k!='note'},dict(control_exact=True,stability_exact=True,candidate_bit_exact=False))
        r['repeated_comparison']=drift;self.assertFalse(fidelity_gate(r)['stability_exact'])
        r['historical_comparisons']['folded_fp32']=drift;self.assertFalse(fidelity_gate(r)['control_exact'])

    def test_invalid_outputs_are_kept_but_score_arithmetic_is_not_relaxed(self):
        a=run();a['token_ids'][0]=a['token_ids'][1];a['token_ids'][2][-1]=0
        validate_run(a)  # Quality evaluator separately counts these validity failures.
        a['sequence_logp'][0]=0
        with self.assertRaisesRegex(AssertionError,'sum'):validate_run(a)

    def test_signed_zero_and_ancestry_drift_remain_stability_failures(self):
        a=run();b=copy.deepcopy(a);a['sequence_logp'][0]=0.;b['sequence_logp'][0]=-0.
        self.assertNotEqual(compare([(a,b)])['matching_sequence_values'],50)
        a=run();a['saved_score_storage_ptrs'][-1]=1
        with self.assertRaises(AssertionError):validate_output_storage(a)
        a=run();a['beam_indices'][0][0]=50
        with self.assertRaises(AssertionError):validate_output_storage(a)

    def test_large_effect_required_in_both_halves_and_lengths(self):
        t={'folded_fp32':{'median_ms':100.,'p95_ms':110.},'folded_tf32':{'median_ms':80.,'p95_ms':90.}}
        r=dict(latency_ms=t,latency_by_length_ms={'76':copy.deepcopy(t)},latency_by_half_ms={'1':copy.deepcopy(t)},latency_by_length_half_ms={'76:1':copy.deepcopy(t)})
        self.assertTrue(performance_gate(r)['passed'])
        r['latency_by_half_ms']['1']['folded_tf32']['p95_ms']=109.
        self.assertFalse(performance_gate(r)['passed'])

    def test_api_parser_never_uses_unrelated_math_mode(self):
        text='I! cuBLAS (v12) function cublasSetMathMode(...) called:\nMathMode=CUBLAS_TF32_TENSOR_OP_MATH\n'+block('CUBLAS_DEFAULT_MATH')
        calls=api_gemms(text)
        self.assertEqual(len(calls),1);self.assertEqual(calls[0]['math_mode'],'CUBLAS_DEFAULT_MATH')

    def test_dispatch_needs_actual_model_shapes_graph_replay_and_symbols(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);mode='CUBLAS_TF32_TENSOR_OP_MATH'
            (root/'api.log').write_text(block(mode)+block(mode,1536,50,8960))
            events=[{'cat':'kernel','name':'ampere_s1688tf32_128x64','dur':10,'args':{'correlation':1}}]+[{'name':'cudaGraphLaunch','cat':'cuda_runtime','args':{'correlation':i}} for i in range(1,5)]
            for n in (76,121):(root/f'{n}.json').write_text(json.dumps({'traceEvents':events}))
            report={'capture_logs':{'available':True,'cublas':'api.log'},'profiles':[{'trace':f'{n}.json'} for n in (76,121)]}
            self.assertTrue(dispatch_evidence(root,report,True)['verified'])
            events[0]['args']['correlation']=99
            (root/'76.json').write_text(json.dumps({'traceEvents':events}))
            self.assertFalse(dispatch_evidence(root,report,True)['verified'])
            events[0]['args']['correlation']=1
            (root/'76.json').write_text(json.dumps({'traceEvents':events}))
            (root/'api.log').write_text(block(mode,16,16,16))
            self.assertFalse(dispatch_evidence(root,report,True)['verified'])
            (root/'api.log').write_text(block(mode)+block(mode,1536,50,8960))
            events[0]['name']='sgemm_128x64';(root/'76.json').write_text(json.dumps({'traceEvents':events}))
            self.assertFalse(dispatch_evidence(root,report,True)['verified'])
            report['capture_logs']['available']=False
            self.assertFalse(dispatch_evidence(root,report,True)['verified'])
