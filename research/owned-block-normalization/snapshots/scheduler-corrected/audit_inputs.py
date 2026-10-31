"""Allocated input regeneration only; never locally re-audit released workloads."""
import hashlib
import torch


def verify_generated_inputs(records, *, width=152265,device='cuda:0'):
    # Independent native sampler path; no producer imports or supplied hashes as
    # seeds. This is input identity, not model/output recomputation.
    def digest(t):return hashlib.sha256(t.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()
    expected={}
    for seed in (17,29):
        generator=torch.Generator(device=device);generator.manual_seed(seed)
        initial=digest(generator.get_state())
        values=torch.randn((100,width),dtype=torch.float32,device=device,generator=generator)
        expected[seed]=dict(algorithm='torch.randn-dedicated-generator-v1',seed=seed,device=str(values.device),shape=[100,width],dtype='torch.float32',initial_generator_sha256=initial,final_generator_sha256=digest(generator.get_state()),generated_sha256=digest(values),staged_sha256=digest(values))
        del values
    if any(record['input_generation']!=expected[record['seed']] for record in records):
        raise ValueError('Independent seed/input regeneration differs')
    return dict(seed_generation_independently_verified=True,seeds=2,output_recomputed=False)
