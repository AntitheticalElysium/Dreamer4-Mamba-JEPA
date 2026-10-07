"""Recovery status, append-only events and atomic current snapshot. No numerical verdict inference."""
import datetime
import json
import subprocess
import time
from pathlib import Path

HERE=Path(__file__).parent
LOG=Path('artifacts/eda/levers_logs')
UNITS=['d4mj-oct05-lead','d4mj-oct05-e19-diagnose','d4mj-oct05-e19-mechanism','d4mj-oct05-e17-clock',
       'd4mj-oct06-diagnostic-replicas','d4mj-oct06-cpu-recurrence','d4mj-oct06-cpu-recurrence-full',
       'd4mj-oct06-cpu-conv-channels','d4mj-oct06-cpu-endpoints',
       'd4mj-oct06-cpu-health-census','d4mj-oct06-router-oracle','d4mj-oct06-cpu-incoming',
       'd4mj-oct06-cpu-canvas36k','d4mj-oct06-h16-resume']

def command(args):
    p=subprocess.run(args,capture_output=True,text=True,timeout=15)
    return {'exit_code':p.returncode,'stdout':p.stdout.strip(),'stderr':p.stderr.strip()}

while True:
    s={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
       'note':'Runtime only: an active service can be waiting; stale log rows are not resumed updates.',
       'units':{u:command(['systemctl','--user','show',u,'-p','ActiveState','-p','Result','-p','MemoryCurrent']) for u in UNITS},
       'python':command(['ps','-C','python','-o','pid,etimes,pcpu,args']),
       'gpu':command(['nvidia-smi','--query-gpu=memory.used,memory.free,utilization.gpu','--format=csv,noheader']),
       'events':(LOG/'lanes.log').read_text().splitlines()[-12:]}
    names=['e17_M16_s7','e17_M16_s8','tworld_c6_s7','tworld_c6_s8','oct05_h16_m6_s8','e17_h16traj_w15']
    names.extend(['e17_recurrence_cpu','e18_access_s7_at12000','e18_reroute_s7_at12000','e18_carry_s7_at12000'])
    names.append('e17_recurrence_full_cpu')
    names.append('e17_conv_channels_cpu')
    names.append('e18_endpoint_s7_cpu')
    names.extend(['e18_access_s7_at36000','e18_carry_s7_at36000','e18_reroute_s7_at36000'])
    names.extend(p.stem for p in LOG.glob('e19_*.log'))
    s['logs']={name:(LOG/f'{name}.log').read_text().splitlines()[-3:] if (LOG/f'{name}.log').exists() else []
               for name in names}
    tmp=HERE/'live_status.tmp'
    tmp.write_text(json.dumps(s,indent=2)+'\n')
    tmp.replace(HERE/'live_status.json')
    with (HERE/'status_events.jsonl').open('a') as f:
        f.write(json.dumps(s)+'\n')
    if not any('ActiveState=active' in v['stdout'] or 'ActiveState=activating' in v['stdout'] for v in s['units'].values()):
        break
    time.sleep(30)
