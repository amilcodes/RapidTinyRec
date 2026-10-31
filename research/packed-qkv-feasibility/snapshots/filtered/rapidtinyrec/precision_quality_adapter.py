"""Build paired SID evaluation inputs from independently verified release resources."""
import collections
import csv
import hashlib
import json
import re
from pathlib import Path
from .quality_frontier import evaluate


def quality_inputs(release, users, paired):
    release = Path(release)
    pinned = json.loads(Path(__file__).with_name('quality_resources.json').read_text())
    for name, sha in pinned.items():
        if hashlib.sha256((release/name).read_bytes()).hexdigest() != sha:
            raise ValueError(f'Quality resource changed: {name}')
    tokenmap = json.loads((release/'Office_ckpt/added_tokens.json').read_text())
    index = json.loads((release/'Amazon/index/Office_Products.index.json').read_text())
    items = json.loads((release/'Amazon/index/Office_Products.item.json').read_text())
    if set(items) != set(index):
        raise ValueError('Item/index identity mismatch')
    counts = collections.Counter(''.join(v) for v in index.values())
    catalog = []
    def sid_tokens(sid):
        pieces = re.findall(r'<[abc]_\d+>', sid)
        if len(pieces) != 3 or ''.join(pieces) != sid:
            raise ValueError('Invalid target SID')
        return [tokenmap[t] for t in pieces]
    for sid in sorted(counts):
        catalog.append(sid_tokens(sid))
    with (release/'Amazon/test/Office_Products_5_2016-10-2018-11.csv').open() as f:
        rows = list(csv.DictReader(f))
    q_users = []
    for u in users:
        row = rows[u['source_row_index']]
        if (row['user_id'], row['item_sid']) != (u['user_id'], u['target_sid']):
            raise ValueError('User/held-out target identity mismatch')
        if ''.join(index[row['item_id']]) != u['target_sid']:
            raise ValueError('Held-out item/SID association mismatch')
        q_users.append(dict(cohort_index=u['cohort_index'],prompt_tokens=u['prompt_tokens'],
            target_token_ids=sid_tokens(u['target_sid']),target_item_count=counts[u['target_sid']]))
    observations = [dict(cohort_index=r['cohort_index'],repeat=r['repeat'],variant=label,token_ids=r['runs'][variant]['token_ids'])
                    for r in paired for variant,label in [('folded_fp32','control'),('folded_tf32','candidate')]]
    return dict(users=q_users, observations=observations,catalog_sids=catalog,eos_token_id=151645)
