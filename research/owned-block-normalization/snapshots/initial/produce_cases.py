"""Finite seeded input/producer assembly, CUDA CLI disabled until readiness."""
import torch
from compare_intervention import _hash,_shifted_copy,compare_case,two_site
from graph_intervention import CapturedTwoSite


def seeded_input(seed,offset,width,device):
    if type(seed) is not int or seed not in (17,29) or type(offset) is not int or offset not in range(4):
        raise ValueError('Undeclared seed/offset')
    if type(width) is not int or width<=0:raise ValueError('Invalid width')
    generator=torch.Generator(device=device).manual_seed(seed)
    initial=_hash(generator.get_state())
    base=torch.randn((100,width),generator=generator,dtype=torch.float32,device=device)
    x=_shifted_copy(base,offset)
    return x,dict(algorithm='torch.randn-dedicated-generator-v1',seed=seed,device=str(x.device),shape=[100,width],
                  dtype='torch.float32',initial_generator_sha256=initial,final_generator_sha256=_hash(generator.get_state()),
                  generated_sha256=_hash(base),staged_sha256=_hash(x))


def produce_cases(*,width=152265,device='cuda:0',graph_factory=CapturedTwoSite,case_sink=None):
    records=[]
    for seed in (17,29):
        for offset in range(4):
            for mode in ('eager','cuda_graph'):
                x,generation=seeded_input(seed,offset,width,device)
                candidate=two_site if mode=='eager' else graph_factory(x)
                record=compare_case(x,seed=seed,offset=offset,candidate=candidate,mode=mode)
                record['input_generation']=generation
                records.append(record)
                if case_sink is not None:case_sink(record)
                del candidate,x
    return records

if __name__=='__main__':
    raise SystemExit('Disabled: allocated independent numerical audit, source/config/deadline/log closure/readiness pending')
