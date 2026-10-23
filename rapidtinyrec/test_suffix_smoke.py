"""CPU-only functional test using a tiny random Qwen2, never a latency benchmark."""
import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM, GenerationConfig, LogitsProcessor, LogitsProcessorList
from .combined_suffix import register_suffix_attention, suffix_split_gqa, SuffixGeneration

torch.set_num_threads(4)
torch.manual_seed(42)


class ForcedSuffix(LogitsProcessor):
    def __init__(self):
        self.step=0
    def __call__(self,input_ids,scores):
        scores=torch.log_softmax(scores,dim=-1)
        mask=torch.full_like(scores,-torch.inf)
        columns=range(1000,1050) if self.step==0 else [1100,1200,198,151645][self.step-1:self.step]
        mask[:,list(columns)]=0
        self.step+=1
        return scores+mask


def main():
    register_suffix_attention()
    # A separate explicit full attention computation checks the visibility rule.
    q=torch.randn(3,12,2,8,dtype=torch.float64)
    k=torch.randn(3,2,11,8,dtype=torch.float64)
    v=torch.randn_like(k)
    got,_=suffix_split_gqa(None,q,k,v,None,scaling=8**-0.5)
    s=q@k.repeat_interleave(6,1).transpose(-1,-2)*8**-0.5
    s[:,:,0,-1]=-torch.inf
    want=(s.softmax(-1)@v.repeat_interleave(6,1)).transpose(1,2)
    assert torch.allclose(got,want,atol=1e-12,rtol=0)
    cfg=Qwen2Config(vocab_size=151646,hidden_size=48,intermediate_size=96,num_hidden_layers=2,
                    num_attention_heads=6,num_key_value_heads=2,max_position_embeddings=256,
                    bos_token_id=1,eos_token_id=151645,pad_token_id=151645,use_sliding_window=False,
                    layer_types=['full_attention','full_attention'],attention_dropout=0.0)
    model=Qwen2ForCausalLM(cfg).double().eval()
    model.config._attn_implementation='sdpa'
    compiler=SimpleNamespace(eager=model.forward,compiled=model.forward)
    result={'purpose':'Functional checks only; tiny random Qwen2 on allocated CPU. No A6000 benchmark results.',
            'causal_oracle_max_abs':(got-want).abs().max().item(),'cases':[]}
    with torch.inference_mode():
        for length in [4,9,16]:
            ids=torch.randint(2,900,(1,length))
            runs=[]
            for combine in [False,True]:
                gen=GenerationConfig(max_new_tokens=8,length_penalty=0.0,num_beams=50,num_return_sequences=50,
                                     pad_token_id=151645,eos_token_id=151645,bos_token_id=1,
                                     do_sample=False,top_k=None,top_p=None,use_cache=True)
                with SuffixGeneration(model,compiler,length,combine) as state:
                    output=model.generate(ids,attention_mask=torch.ones_like(ids),generation_config=gen,
                                          return_dict_in_generate=True,output_scores=True,
                                          logits_processor=LogitsProcessorList([ForcedSuffix()]))
                # No GPU-only tracker.audit here; independently check the same counters.
                assert state.generation_calls==5 and state.model_forwards==(4 if combine else 5)
                if combine:
                    assert state.suffix_reorders==1 and bool((state.newline_input==198).all())
                    assert sorted(state.reorder_indices.tolist())==list(range(50))
                assert not hasattr(model,'_reorder_cache')
                assert model.config._attn_implementation=='sdpa'
                tokens=output.sequences[:,length:]
                assert tokens.shape==(50,5) and bool((tokens[:,-2:]==torch.tensor([198,151645])).all())
                transitions=model.compute_transition_scores(output.sequences,output.scores,output.beam_indices,normalize_logits=False)
                runs.append((tokens,output.sequences_scores,transitions))
            assert torch.equal(runs[0][0],runs[1][0]),'Toy ranking changed'
            seq_delta=(runs[0][1]-runs[1][1]).abs().max().item()
            tr_delta=(runs[0][2]-runs[1][2]).abs().max().item()
            assert seq_delta<1e-4 and tr_delta<1e-4,(seq_delta,tr_delta)
            result['cases'].append({'prompt_tokens':length,'rankings_equal':True,
                'max_sequence_difference':seq_delta,'max_transition_difference':tr_delta,
                'reference_forwards':5,'combined_forwards':4,
                'nontrivial_newline_reorder':state.reorder_indices.tolist()!=list(range(50)),
                'newline_beam_permutation':state.reorder_indices.tolist()})
    result['passed']=True
    p=Path(os.environ.get('RAPIDTINYREC_OUTPUT_DIR', 'results'))/'suffix_smoke.json'
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    main()
