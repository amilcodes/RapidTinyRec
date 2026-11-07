"""Fetch per-config logs of Oscar runs and print a compact summary. usage: peek.py RUN [RUN...]"""
import json, subprocess, sys
from pathlib import Path
local = Path(__file__).resolve().parents[2] / 'runs'
for run in sys.argv[1:]:
    dst = local / run
    dst.mkdir(parents=True, exist_ok=True)
    subprocess.run(['rsync', '-az', '-e', 'ssh -o BatchMode=yes -o LogLevel=ERROR', '--include=*.log', '--include=*/',
                    '--include=results.json', '--include=quality_users.json', '--include=latency_users.json',
                    '--include=profile.txt', '--include=roofline.json', '--exclude=*',
                    f'aagraw73@sshcampus.ccv.brown.edu:/oscar/scratch/aagraw73/speedrun/runs/{run}/out/', str(dst) + '/'],
                   check=False)
    for log in sorted(dst.glob('*.log')):
        if log.name == 'slurm.log':
            continue
        print(f'## {run}/{log.stem}')
        for l in log.read_text(errors='replace').splitlines():
            l = l.strip()
            if l.startswith('{"stage"'):
                d = json.loads(l)
                if d['stage'] == 'latency':
                    ag = d.get('agreement_vs_fp32_reference') or {}
                    print(f"  LAT {d['device_only']['median_ms']:.2f} / p95 {d['device_only']['p95_ms']:.2f} ms  cert={d.get('certified_rate')}  top1=FP32 {ag.get('top1_equal')}")
                elif d['stage'] == 'quality':
                    m = d['metrics']
                    print(f"  QUAL HR@1 {m['HR@1']:.4f} HR@5 {m['HR@5']:.4f} HR@10 {m['HR@10']:.4f} NDCG@10 {m['NDCG@10']:.4f} HR@50 {m['HR@50']:.4f} cert={d.get('certified_rate')}")
                elif d['stage'] == 'throughput':
                    print(f"  TP B={d['B']} {d['users_per_s']:.1f} users/s")
                elif d['stage'] == 'profile':
                    print(f"  PROFILE {d}")
            elif 'Error' in l or l.startswith('Traceback'):
                print('  ' + l[:240])
    sl = dst / 'slurm.log'
    if sl.exists():
        tail = [l for l in sl.read_text(errors='replace').splitlines() if l.strip()][-1:]
        print(f'  [slurm tail] {tail[0][:200] if tail else ""}')
