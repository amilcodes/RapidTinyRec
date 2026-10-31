"""Seeded synthetic row-permutation diagnostic, never a released-model audit."""
import argparse,json,os,pathlib,time

SHAPES=((100,1536,8960),(100,8960,1536),(100,1536,152265),(2,1536,152265))

def compare(a,b):
    import torch
    aa=a.contiguous().view(torch.int32);bb=b.contiguous().view(torch.int32)
    different=aa!=bb
    return {'bit_exact':bool(torch.equal(aa,bb)), 'different_values':int(different.sum().item()),
            'different_rows':int(different.any(dim=1).sum().item()),
            'max_abs':float((a-b).abs().max().item())}

def execute(device,shapes,seeds,policies,modes):
    import torch
    torch.set_num_threads(4)
    def sync():
        if device=='cuda':torch.cuda.synchronize()
        elif device=='mps':torch.mps.synchronize()
    result=[]
    with torch.inference_mode():
      for shape in shapes:
       m,k,n=shape; assert m%2==0
       for seed in seeds:
        torch.manual_seed(seed)
        x=torch.randn(m,k,device=device,dtype=torch.float32)/k**0.5
        w=torch.randn(k,n,device=device,dtype=torch.float32)/k**0.5
        original=x.clone(); perm=torch.cat((torch.arange(m//2,m,device=device),torch.arange(m//2,device=device)))
        xrev=original.index_select(0,perm)
        for tf32 in policies:
         torch.backends.cuda.matmul.allow_tf32=tf32
         for mode in modes:
          graph=None
          def forward():
              p=torch.mm(x,w)
              first=torch.log_softmax(p,dim=-1)
              second=torch.log_softmax(first,dim=-1)
              return p,first,second
          if mode=='graph':
              assert device=='cuda'
              stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
              with torch.cuda.stream(stream):
                  for _ in range(3):forward()
              torch.cuda.current_stream().wait_stream(stream);sync()
              graph=torch.cuda.CUDAGraph()
              with torch.cuda.graph(graph):borrowed=forward()
          def run(src):
              x.copy_(src)
              if graph is None:out=forward()
              else:graph.replay();out=borrowed
              owned=tuple(t.clone() for t in out);sync();return owned
          # Each fixed sequence contains two same-order and two reversed observations.
          baseline=run(original); repeat=run(original); reverse=run(xrev); reverse_repeat=run(xrev); restored=run(original)
          stages={}
          for j,name in enumerate(('projection','first_log_softmax','second_log_softmax')):
              stages[name]={'same_order':compare(baseline[j],repeat[j]),
                            'reversed_order':compare(baseline[j],reverse[j].index_select(0,perm)),
                            'reverse_repeat':compare(reverse[j],reverse_repeat[j]),
                            'restored_order':compare(baseline[j],restored[j])}
          result.append({'shape':list(shape),'seed':seed,'tf32':tf32,'mode':mode,'stages':stages})
          del baseline,repeat,reverse,reverse_repeat,restored
          if graph is not None:del graph,borrowed
        del x,w,original,xrev
    return result

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--local-smoke',action='store_true');ap.add_argument('--output',type=pathlib.Path,required=True);a=ap.parse_args()
    if a.output.exists():raise ValueError('Preserve prior evidence')
    if a.local_smoke:
        device='cpu';shapes=((4,16,32),);seeds=(17,);policies=(False,);modes=('eager',);provenance=None
    else:
        from provenance import validate
        provenance=validate();device='cuda';shapes=SHAPES;seeds=(17,29);policies=(False,True);modes=('eager','graph')
    import torch
    if not a.local_smoke:
        assert torch.cuda.is_available() and torch.cuda.get_device_name()=='NVIDIA RTX A6000'
        assert torch.__version__=='2.14.0+cu126'
    started=time.monotonic();old=torch.backends.cuda.matmul.allow_tf32
    try:cases=execute(device,shapes,seeds,policies,modes)
    finally:torch.backends.cuda.matmul.allow_tf32=old
    record={'schema':'synthetic-row-permutation-probe-v1','device':device,'torch':torch.__version__,
            'torch_git':torch.version.git_version,'cuda':torch.version.cuda,'cases':cases,
            'elapsed_s':time.monotonic()-started,'provenance':provenance,
            'released_weights_used':False,'generation_rerun':False,'latency_claim':False,
            'limitation':'Synthetic primitive attribution only; neither reproduces nor repairs complete beam-wave generation.'}
    if not a.local_smoke:
        assert time.monotonic()-started<480
        assert validate()==provenance,'Submitted source changed during diagnostic'
    with a.output.open('x') as f:json.dump(record,f,indent=2);f.write('\n')
    print(json.dumps({'cases':len(cases),'device':device,'elapsed_s':record['elapsed_s']}))
if __name__=='__main__':main()
