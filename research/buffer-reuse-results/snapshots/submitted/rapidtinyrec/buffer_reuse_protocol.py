"""Distinct paired single-request protocol; no amortized wave timings."""
import hashlib,json,math,statistics
from pathlib import Path
from .beam_wave_protocol import percentile
PROTOCOL='protocol.json'
PROTOCOL_SHA256='f484617583ff74b99bd1949b857563af9c395cf0f87229a13ae02f86d4d1782c'
PHASES=('released_tf32','reuse_normalized_tf32','reuse_normalized_tf32','released_tf32')
USERS=(1,2,6,10)
LENGTHS={1:76,2:76,6:121,10:121}

def load_protocol(path):
 raw=Path(path).read_bytes()
 if hashlib.sha256(raw).hexdigest()!=PROTOCOL_SHA256:raise ValueError('Frozen reuse protocol changed')
 p=json.loads(raw)
 if p['schema']!='released-normalized-buffer-reuse-cuda-v1' or p['profile']!=p['schema']:raise ValueError('Distinct protocol required')
 if p['phases']!=[dict(phase=i,variant=v,half=i//2,reuse_disk_artifacts=i>=2) for i,v in enumerate(PHASES)]:raise ValueError('ABBA scope differs')
 if p['cohort_indices']!=list(USERS) or p['paired_observations']!=24 or p['timed_requests_per_phase']!=12:raise ValueError('Cohort scope differs')
 if p['promotion'] or p['noninferiority_established']:raise ValueError('No automatic adoption')
 return p

def phase_schedule(phase):
 if type(phase) is not int or phase not in range(4):raise ValueError('Invalid phase')
 return [dict(phase=phase,variant=PHASES[phase],half=phase//2,repeat=r,cohort_index=u,prompt_tokens=LENGTHS[u]) for r in range(phase//2*3,phase//2*3+3) for u in ((1,2,6,10) if r%2==0 else (2,1,10,6))]

def validate_timing(row):
 if row['schema']!='released-buffer-reuse-timing-v1':raise ValueError('Wrong timing schema')
 schedule=phase_schedule(row['phase'])
 if {k:row[k] for k in schedule[0]} not in schedule:raise ValueError('Wrong request schedule')
 if type(row['duration_s']) not in (int,float) or not math.isfinite(row['duration_s']) or row['duration_s']<=0:raise ValueError('Finite positive request duration required')
 return row

def validate_phase_rows(phase,rows):
 expected=phase_schedule(phase)
 if len(rows)!=len(expected):raise ValueError('Incomplete phase')
 for row,spec in zip(rows,expected):
  validate_timing(row)
  if any(row[k]!=v for k,v in spec.items()):raise ValueError('Duplicate or reordered request')

def compare_phases(phases):
 if set(phases)!={0,1,2,3}:raise ValueError('Four complete phases required')
 for phase,rows in phases.items():validate_phase_rows(phase,rows)
 key=lambda r:(r['repeat'],r['cohort_index'])
 c={key(r):r for r in phases[0]+phases[3]};t={key(r):r for r in phases[1]+phases[2]}
 if set(c)!=set(t) or len(c)!=24:raise ValueError('Matched paired observations required')
 groups={'overall':list(c)}
 for length in (76,121):groups[f'length_{length}']=[k for k in c if c[k]['prompt_tokens']==length]
 for half in (0,1):
  groups[f'half_{half}']=[k for k in c if c[k]['half']==half]
  for length in (76,121):groups[f'length_{length}_half_{half}']=[k for k in c if c[k]['half']==half and c[k]['prompt_tokens']==length]
 stats=lambda xs:dict(median_ms=statistics.median(xs)*1000,p95_ms=percentile(xs,.95)*1000,requests=len(xs))
 strata={}
 for name,keys in groups.items():
  a=stats([c[k]['duration_s'] for k in keys]);b=stats([t[k]['duration_s'] for k in keys])
  strata[name]=dict(control=a,candidate=b,median_change_percent=100*(b['median_ms']/a['median_ms']-1),p95_change_percent=100*(b['p95_ms']/a['p95_ms']-1))
 return dict(schema='released-buffer-reuse-timing-review-v1',strata=strata,timing_screen_passed=all(v['median_change_percent']<=-5 and v['p95_change_percent']<=0 for v in strata.values()),raw_audit_passed=None,promotion=False,quality_equivalence=False)
