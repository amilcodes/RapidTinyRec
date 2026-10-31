import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from rapidtinyrec.precision_quality_adapter import quality_inputs


class AdapterTests(unittest.TestCase):
    def fixture(self, root):
        files={'Office_ckpt/added_tokens.json':json.dumps({'<a_1>':11,'<b_2>':22,'<c_3>':33}),
            'Amazon/index/Office_Products.index.json':json.dumps({'1':['<a_1>','<b_2>','<c_3>'],'2':['<a_1>','<b_2>','<c_3>']}),
            'Amazon/index/Office_Products.item.json':json.dumps({'1':{},'2':{}}),
            'Amazon/test/Office_Products_5_2016-10-2018-11.csv':'user_id,item_sid,item_id\nu,<a_1><b_2><c_3>,1\n'}
        for n,s in files.items():
            p=root/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(s)
        users=[dict(cohort_index=1,prompt_tokens=76,source_row_index=0,user_id='u',target_sid='<a_1><b_2><c_3>')]
        paired=[dict(cohort_index=1,repeat=0,runs={v:{'token_ids':[[11,22,33,198,151645]]*50} for v in ('folded_fp32','folded_tf32')})]
        pins={n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in files}
        original=Path.read_text
        def reader(path,*a,**kw):
            return json.dumps(pins) if path.name=='quality_resources.json' else original(path,*a,**kw)
        return users,paired,reader

    def test_ambiguous_target_stays_in_primary_and_original_rows_stay(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);users,paired,reader=self.fixture(root)
            with patch.object(Path,'read_text',reader):data=quality_inputs(root,users,paired)
            self.assertEqual(data['users'][0]['target_item_count'],2)
            self.assertEqual(data['users'][0]['target_token_ids'],[11,22,33])
            self.assertEqual(data['catalog_sids'],[[11,22,33]])
            self.assertEqual(len(data['observations'][0]['token_ids']),50)

    def test_resource_mutation_is_not_reinterpreted_as_quality_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);users,paired,reader=self.fixture(root)
            (root/'Office_ckpt/added_tokens.json').write_text('{}')
            with patch.object(Path,'read_text',reader),self.assertRaisesRegex(ValueError,'resource changed'):
                quality_inputs(root,users,paired)

    def test_target_row_identity_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);users,paired,reader=self.fixture(root);users[0]['user_id']='different'
            with patch.object(Path,'read_text',reader),self.assertRaisesRegex(ValueError,'target identity'):
                quality_inputs(root,users,paired)
