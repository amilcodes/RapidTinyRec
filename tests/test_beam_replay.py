import unittest
from unittest.mock import patch, MagicMock

import torch
from transformers.generation.utils import GenerationMixin
from rapidtinyrec.beam_replay import (
    RUNNING, FINISHED, PINNED_SOURCE_SHA256, BeamReplayBuffers, BeamReplayPool,
    verify_helpers,
)


def inputs(prompt=7, step=0, penalty=0., early=False, scenario='ties', beams=3):
    count = 2*beams
    scores = torch.tensor([[0., -0., -1., -1., -1e9, float('-inf')]])
    if scenario == 'equal':
        scores.fill_(-1.)
    elif scenario == 'near':
        scores = torch.arange(count, dtype=torch.float32).reshape(1, count) * torch.finfo(torch.float32).eps
    elif scenario == 'infinity':
        scores[0, 0] = float('inf')
    sequences = torch.arange(count*(prompt+8), dtype=torch.int64).reshape(1, count, prompt+8)
    indices = torch.arange(count*8, dtype=torch.int32).reshape(1, count, 8) % beams
    stopped = torch.tensor([[False, True, False, True, False, True]])
    running = dict(topk_log_probs=scores, topk_running_sequences=sequences,
                   topk_running_beam_indices=indices, next_token_hits_stopping_criteria=stopped,
                   num_beams=beams)
    finished = dict(sequences=sequences[:, :beams].clone(), topk_running_sequences=sequences,
                    beam_scores=torch.full((1, beams), -1e9), topk_log_probs=scores,
                    beam_indices=indices[:, :beams].clone(), topk_running_beam_indices=indices,
                    is_early_stop_heuristic_unsatisfied=torch.tensor([[True]]),
                    is_sent_finished=torch.tensor([[False, True, False]]),
                    next_token_hits_stopping_criteria=stopped,
                    top_num_beam_mask=torch.arange(count)<beams, num_beams=beams,
                    cur_len=prompt+step, decoder_prompt_len=prompt,
                    length_penalty=penalty, early_stopping=early)
    return running, finished


def equal_bits(test, left, right):
    test.assertEqual(len(left), len(right))
    for a, b in zip(left, right):
        test.assertEqual(a.dtype, b.dtype)
        test.assertEqual(a.shape, b.shape)
        test.assertTrue(torch.equal(a.view(torch.int32), b.view(torch.int32)) if a.dtype == torch.float32
                        else torch.equal(a, b))


class BeamReplayTests(unittest.TestCase):
    def test_pinned_helpers_and_reject_override(self):
        model = GenerationMixin()
        self.assertEqual(verify_helpers(model), PINNED_SOURCE_SHA256)
        from rapidtinyrec.audit_beam_helpers import HASHES
        self.assertEqual(PINNED_SOURCE_SHA256, HASHES)
        model._get_running_beams_for_next_iteration = lambda **kwargs: None
        with self.assertRaisesRegex(RuntimeError, 'Custom beam'):
            verify_helpers(model)

    @torch.inference_mode()
    def test_owned_staging_preserves_helper_ties_and_scalars(self):
        model = GenerationMixin()
        for scenario in ('ties', 'equal', 'near', 'infinity'):
            for prompt in (7, 11):
                for penalty, early in ((0., False), (1.2, True), (-1., 'never')):
                    for name, arguments in zip((RUNNING, FINISHED), inputs(prompt, 2, penalty, early, scenario)):
                        with self.subTest(scenario=scenario, prompt=prompt, helper=name, penalty=penalty):
                            buffers = BeamReplayBuffers(arguments)
                            direct = getattr(model, name)(**arguments)
                            staged = getattr(model, name)(**buffers.arguments)
                            equal_bits(self, direct, staged)
                            for key, tensor in buffers.arguments.items():
                                if isinstance(tensor, torch.Tensor):
                                    self.assertNotEqual(tensor.data_ptr(), arguments[key].data_ptr())
                            # Future source mutations cannot change owned staging or earlier returned state.
                            old = tuple(x.clone() for x in staged)
                            arguments['topk_log_probs'].fill_(-17.)
                            equal_bits(self, old, staged)
                            buffers.load(arguments)
                            equal_bits(self, getattr(model, name)(**arguments), getattr(model, name)(**buffers.arguments))
                            changed = dict(arguments, num_beams=4)
                            with self.assertRaises(ValueError):
                                buffers.load(changed)

    def test_staging_rejects_layout_and_dtype_changes(self):
        arguments = inputs()[0]
        buffers = BeamReplayBuffers(arguments)
        with self.assertRaises(ValueError):
            buffers.load(dict(arguments, topk_log_probs=arguments['topk_log_probs'].double()))
        with self.assertRaises(ValueError):
            BeamReplayBuffers(dict(arguments, topk_log_probs=torch.zeros(6, 2)[:, 0].reshape(1, 6)))
        with self.assertRaises(ValueError):
            BeamReplayBuffers(dict(arguments, topk_log_probs=arguments['topk_log_probs'].requires_grad_()))

    @torch.inference_mode()
    def test_instance_wrappers_lease_order_and_recurrent_state(self):
        model, other = GenerationMixin(), GenerationMixin()
        pool = BeamReplayPool(model).install()
        original = other._get_running_beams_for_next_iteration
        # CPU adapter tests lease/integration only, not CUDA graph correctness.
        def cpu_execute(method, prompt, step, arguments):
            buffers = BeamReplayBuffers(arguments)
            return pool.originals[method](**buffers.arguments), (method, prompt, step)
        with patch.object(pool, 'replay', side_effect=cpu_execute):
            with self.assertRaises(RuntimeError):
                model._get_running_beams_for_next_iteration(**inputs()[0])
            for prompt in (7, 11, 7):
                lease = pool.request(prompt)
                with self.assertRaises(RuntimeError): pool.request(prompt)
                with self.assertRaises(RuntimeError): lease.close()
                retained = []
                previous_finished = None
                for step in range(5):
                    running, finished = inputs(prompt, step)
                    if previous_finished is not None:
                        for name, tensor in zip(('sequences', 'beam_scores', 'beam_indices', 'is_sent_finished'), previous_finished):
                            finished[name] = tensor
                    expected = pool.originals[RUNNING](**running)
                    equal_bits(self, expected, model._get_running_beams_for_next_iteration(**running))
                    expected = pool.originals[FINISHED](**finished)
                    previous_finished = model._update_finished_beams(**finished)
                    equal_bits(self, expected, previous_finished)
                    retained.append((previous_finished, tuple(t.clone() for t in previous_finished)))
                for output, saved in retained:
                    equal_bits(self, saved, output)
                self.assertEqual(len(lease.entries), 10)
                with self.assertRaises(RuntimeError): pool.uninstall()
                lease.close()
                with self.assertRaises(RuntimeError): lease.close()
        pool.uninstall()
        self.assertIs(model._get_running_beams_for_next_iteration.__func__, original.__func__)
        self.assertNotIn(RUNNING, model.__dict__)
        self.assertNotIn(FINISHED, other.__dict__)

    def test_out_of_order_abort_and_freeze_guard(self):
        model = GenerationMixin()
        pool = BeamReplayPool(model).install()
        lease = pool.request(7)
        with self.assertRaises(RuntimeError):
            model._update_finished_beams(**inputs()[1])
        with self.assertRaises(RuntimeError): pool.freeze([7])
        lease.abort()
        self.assertEqual(pool.aborted_requests, 1)
        with self.assertRaises(RuntimeError): pool.freeze([7])
        pool.uninstall()

    def test_stream_device_and_capture_freeze_guards(self):
        pool=BeamReplayPool(GenerationMixin())
        tensor=MagicMock(spec=torch.Tensor);tensor.device=torch.device('cuda:0')
        with patch('torch.cuda.current_stream',return_value=MagicMock(cuda_stream=1)):
            pool._stream({'x':tensor})
        with patch('torch.cuda.current_stream',return_value=MagicMock(cuda_stream=2)):
            with self.assertRaisesRegex(RuntimeError,'stream/device'):pool._stream({'x':tensor})
        with self.assertRaisesRegex(ValueError,'CUDA'):pool._stream({'x':torch.zeros(1)})
        pool.frozen=True
        with self.assertRaisesRegex(RuntimeError,'capture after freeze'):pool.prepare(None,{})


    @torch.inference_mode()
    def test_gpu_probe_inputs_use_actual_helper_shapes_on_cpu(self):
        from rapidtinyrec.probe_beam_replay import inputs as probe_inputs, SCENARIOS, compare_tensors
        from rapidtinyrec.audit_beam_helpers import input_spec, output_spec
        model = GenerationMixin()
        for scenario in SCENARIOS:
            for request, prompt in enumerate((76,121,121,76)):
                previous = None
                for step in range(5):
                    running, finished = probe_inputs(prompt,step,scenario,request,'cpu')
                    if previous is not None:
                        for name,value in zip(('sequences','beam_scores','beam_indices','is_sent_finished'),previous):
                            finished[name]=value
                    for method,args in ((RUNNING,running),(FINISHED,finished)):
                        actual_spec={k:(list(v.shape),str(v.dtype)) for k,v in args.items() if isinstance(v,torch.Tensor)}
                        self.assertEqual(actual_spec,input_spec(method,prompt))
                        direct=getattr(model,method)(**args)
                        owned=getattr(model,method)(**BeamReplayBuffers(args).arguments)
                        equal_bits(self,direct,owned)
                        self.assertEqual([(list(t.shape),str(t.dtype)) for t in direct],output_spec(method,prompt))
                        self.assertTrue(all(compare_tensors(a,b)['no_nan'] for a,b in zip(direct,owned)))
                        if method==FINISHED:previous=owned
                    if scenario=='signed_zero':
                        self.assertEqual(int((running['topk_log_probs'].view(torch.int32)<0).sum()),50)


if __name__ == '__main__':
    unittest.main()
