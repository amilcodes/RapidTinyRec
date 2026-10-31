"""Head/row mapping and unsupported-path checks; GPU bit equality is separate."""
import unittest
from unittest.mock import patch
import torch
from torch.nn.attention import sdpa_kernel,SDPBackend
from rapidtinyrec.folded_attention import folded_sdpa,fold_decode_attention,_SDPA


class FoldedAttentionTests(unittest.TestCase):
    def test_dynamo_traces_packing_without_graph_breaks(self):
        compiled=torch.compile(folded_sdpa,backend='eager',fullgraph=True,dynamic=True)
        with sdpa_kernel(SDPBackend.MATH):
            for length in (7,12):
                q=torch.randn(2,12,1,8);k=torch.randn(2,2,length,8);v=torch.randn_like(k)
                a=folded_sdpa(q,k,v,enable_gqa=True);b=compiled(q,k,v,enable_gqa=True)
                self.assertTrue(torch.equal(a,b))

    def test_head_to_kv_mapping_and_inverse(self):
        torch.manual_seed(312)
        q=torch.randn(3,12,1,128);k=torch.randn(3,2,17,128);v=torch.randn_like(k)
        with sdpa_kernel(SDPBackend.MATH):
            want=_SDPA(q,k,v,scale=0.07,enable_gqa=True)
            got=folded_sdpa(q,k,v,scale=0.07,enable_gqa=True)
        self.assertEqual(got.shape,q.shape)
        torch.testing.assert_close(got,want,rtol=1e-5,atol=1e-6)
        # Each output head must draw only from its assigned KV head.
        for g in range(2):
            v[:,g].fill_(3+g)
        got=folded_sdpa(q,k,v,enable_gqa=True)
        for h in range(12):torch.testing.assert_close(got[:,h],torch.full_like(got[:,h],3+h//6))

    def test_prefill_mask_causal_and_dropout_keep_original_dispatch(self):
        for length,mask,causal,dropout,gqa in [(5,None,False,0.,True),(1,torch.zeros(1,7),False,0.,True),
                (1,None,True,0.,True),(1,None,False,0.1,True),(1,None,False,0.,False)]:
            q=torch.randn(2,12,length,8);k=torch.randn(2,2,7,8);v=torch.randn_like(k)
            sentinel=object()
            with patch('rapidtinyrec.folded_attention._SDPA',return_value=sentinel) as native:
                self.assertIs(folded_sdpa(q,k,v,mask,dropout,causal,scale=.2,enable_gqa=gqa),sentinel)
                a,kw=native.call_args;self.assertIs(a[0],q);self.assertIs(a[1],k);self.assertIs(a[2],v)
                self.assertEqual(kw,dict(attn_mask=mask,dropout_p=dropout,is_causal=causal,scale=.2,enable_gqa=gqa))

    def test_pack_does_not_copy_contiguous_query_and_context_restores(self):
        q=torch.randn(2,12,1,8);k=torch.randn(2,2,7,8);v=torch.randn_like(k)
        def native(packed,keys,values,**kw):
            self.assertEqual(packed.shape,(2,2,6,8));self.assertEqual(packed.data_ptr(),q.data_ptr())
            self.assertIs(keys,k);self.assertIs(values,v);self.assertFalse(kw['enable_gqa'])
            return packed.clone()
        with patch('rapidtinyrec.folded_attention._SDPA',side_effect=native):
            self.assertTrue(torch.equal(folded_sdpa(q,k,v,enable_gqa=True),q))
        with fold_decode_attention():self.assertIs(torch.nn.functional.scaled_dot_product_attention,folded_sdpa)
        self.assertIs(torch.nn.functional.scaled_dot_product_attention,_SDPA)


if __name__=='__main__':unittest.main()
