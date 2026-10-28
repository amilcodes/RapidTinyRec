import copy
import os
import unittest
from unittest.mock import patch
from rapidtinyrec.audit_prompt_cache_storage import STAGES,REQUESTS,validate_metadata,audit_snapshots


class MetadataTests(unittest.TestCase):
    def fixture(self):
        identity=dict(checkpoint_manifest_sha256='a'*64,tokenizer_manifest_sha256='b'*64,precision='folded_tf32')
        metadata=dict(prefix_tokens=3,prefix_token_sha256='c'*64,dtype='torch.float32',identity=identity,
            model_config_sha256='d'*64,policy=[True,False,True,True,'high'],
            cached_kv_bytes=192,request_kv_copy_bytes=192,storage_inventory=[
                dict(layer=i//2,kind=('key','value')[i%2],shape=[1,2,3,2],dtype='torch.float32',
                     storage_ptr=4096+48*i,storage_bytes=48) for i in range(4)])
        rows=[dict(stage=s,file=f'prompt-cache-{s}.pt',sha256='e'*64,
                   metadata=dict(copy.deepcopy(metadata),requests=n)) for s,n in zip(STAGES,REQUESTS)]
        args=dict(layers=2,kv_heads=2,head_dim=2,prefix_tokens=3,prefix_sha256='c'*64,identity=identity)
        return rows,args

    def test_complete_fixed_identity_and_exact_byte_counts(self):
        rows,args=self.fixture();result=validate_metadata(rows,**args)
        self.assertEqual(result['persistent_kv_bytes'],192)

    def test_missing_or_reordered_snapshot_rejected(self):
        rows,args=self.fixture()
        for changed in (rows[:-1],list(reversed(rows)),rows+[rows[-1]]):
            with self.assertRaises(AssertionError):validate_metadata(changed,**args)

    def test_path_injection_rejected(self):
        rows,args=self.fixture();rows[0]['file']='../outside.pt'
        with self.assertRaises(AssertionError):validate_metadata(rows,**args)

    def test_counter_identity_and_precision_changes_rejected(self):
        for key,value in (('requests',25),('identity',{}),('dtype','torch.bfloat16'),
                          ('prefix_token_sha256','f'*64),('model_config_sha256','f'*64),
                          ('request_kv_copy_bytes',0),('policy',[False,False,True,True,'highest'])):
            rows,args=self.fixture();rows[2]['metadata'][key]=value
            with self.subTest(key=key),self.assertRaises(AssertionError):validate_metadata(rows,**args)

    def test_storage_overlap_wrong_shape_and_incomplete_inventory_rejected(self):
        for key,value in (('storage_ptr',4096),('shape',[1,2,2,3]),('storage_bytes',96),('kind','key')):
            rows,args=self.fixture()
            for row in rows:row['metadata']['storage_inventory'][1][key]=value
            with self.subTest(key=key),self.assertRaises(AssertionError):validate_metadata(rows,**args)
        rows,args=self.fixture()
        for row in rows:row['metadata']['storage_inventory'].pop()
        with self.assertRaises(AssertionError):validate_metadata(rows,**args)

    def test_raw_file_entrypoint_refuses_local_before_tensor_loading(self):
        with patch.dict(os.environ,{},clear=True):
            with self.assertRaisesRegex(AssertionError,'allocated compute'):
                audit_snapshots(None,config=None,protocol=None,release_sha256=None)


if __name__=='__main__':unittest.main()
