"""One complete synchronized generation; all extraction and hashing follow timing."""
from .buffer_reuse_protocol import phase_schedule,validate_timing

def measure_request(spec,generate,synchronize,clock,unpack):
 if spec not in phase_schedule(spec['phase']):raise ValueError('Outside frozen request schedule')
 users=[spec['cohort_index']]
 synchronize();start=clock();output=generate(users);synchronize();duration=clock()-start
 timing=dict(spec,schema='released-buffer-reuse-timing-v1',duration_s=duration)
 validate_timing(timing)
 raw=unpack(output,users)
 if len(raw)!=1 or raw[0]['cohort_index']!=users[0]:raise ValueError('Wrong request extraction')
 return dict(schema='released-buffer-reuse-trial-v1',timing=timing,requests=raw)
