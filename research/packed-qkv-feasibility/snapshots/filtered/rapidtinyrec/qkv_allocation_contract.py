"""Distinct QKV allocation contract. Final submission still needs reviewed lock.

Runtime authenticity is the exclusive uploaded source manifest plus actual job
identity; no invocation here submits a run or authorizes an unreviewed protocol.
"""
import json
from pathlib import Path
from .audit_qkv_outputs import require
PROTOCOL='protocol.json'
RESOURCES=dict(partition='gpu',qos='norm-gpu',constraint='ampere',gpu_type='nvidia_rtx_a6000',gpus=1,cpus=4,memory_gb=32,time='00:30:00',setup=['module purge'],command=['bash','jobs/packed-qkv-pilot.sh'])


def load_protocol(path):
    protocol=json.loads(Path(path).read_text())
    require(protocol['schema']=='packed-qkv-bf16-pilot-v1','Distinct QKV protocol required')
    require(protocol['arms']==['original_bf16','packed_qkv_bf16'],'Original BF16 objective differs')
    require(protocol['cohort_indices']==[1,2,6,10] and protocol['prompt_lengths']==[76,121]
            and protocol['repeats_per_half']==3 and protocol['paired_observations']==24 and protocol['independent_users']==4,'QKV cohort or repetition contract differs')
    require(protocol['phases']==[dict(phase=p,arm=arm,half=p//2,reuse_disk_artifacts=p>=2) for p,arm in enumerate(('original_bf16','packed_qkv_bf16','packed_qkv_bf16','original_bf16'))],'QKV ABBA phases differ')
    require(protocol['gate']==dict(minimum_median_reduction_percent=5,p95_no_regression=True,all_overall_length_half_cross_strata=True,exact_engineering=True,full500_requires_review=True,promotion=False),'QKV strict standalone gate differs')
    require(protocol['resources']==dict(gpu='regular RTX A6000',cpus=4,memory_gb=32,max_minutes=30,concurrency=1),'Resource bound differs')
    require(all(protocol['budget'][k]==v for k,v in dict(global_s=1620,generation_max_s=1500,audit_max_s=120).items()),'Inclusive stage deadlines differ')
    return protocol
