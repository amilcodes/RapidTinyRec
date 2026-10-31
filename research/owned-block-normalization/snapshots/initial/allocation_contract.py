"""Distinct primitive allocation contract; no submission entry point."""
PROFILE='owned-block-normalization-v1'
PROJECT='rapidtinyrec-owned-block-normalization'
SSH_TARGET='aagraw73@sshcampus.ccv.brown.edu'
REMOTE_ROOT='/oscar/scratch/aagraw73/agent-runs'
COMPUTE_SECONDS=480
GLOBAL_SECONDS=540
ALLOCATION_SECONDS=600
RESOURCES=dict(partition='gpu',qos='norm-gpu',constraint='ampere',gpu_type='nvidia_rtx_a6000',
               gpus=1,cpus=4,memory_gb=32,time='00:10:00',setup=['module purge'],
               command=['bash','jobs/owned-block.sh'])
# Complete intended upload inventory, including outer entry files still pending.
REQUIRED_FILES=('owned_blocks.py','compare_intervention.py','graph_intervention.py','produce_cases.py',
 'audit_evidence.py','audit_inputs.py','native_binding.py','audit_live.py','loaded_binding.py',
 'publication.py','final_audit.py','allocation_contract.py','allocation.py','bounded_runner.py','whole_run.py','guarded_whole.py','terminal_metadata.py','allocated_entry.py',
 'jobs/owned-block.sh','oscar.json','protocol.json','readiness-lock.json')
