"""Read-only audit of the actual F2 treatment, fixed control and queued readings."""
from pathlib import Path
import hashlib,json,datetime
import torch
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
RUN=HERE.parent/'20261103_p1f2';BASE=HERE.parent/'20261102_p1f16'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()

def main():
    pins=json.loads((RUN/'SOURCE_PINS.json').read_text());bad=[]
    for path,expected in pins['files'].items():
        actual=sha(ROOT/path)
        if actual!=expected:bad.append(dict(path=path,expected=expected,actual=actual))
    a=torch.load(BASE/'run/ckpt_10.pt',map_location='cpu',weights_only=False)
    b=torch.load(RUN/'run/ckpt_10.pt',map_location='cpu',weights_only=False)
    status=json.loads((RUN/'STATUS.json').read_text());head=b['fixation_controller']
    assert a['cfg']==b['cfg'] and a['world_seed']==b['world_seed']==1
    assert status['phase']=='complete' and status['selection']['checkpoint']==10
    assert sha(RUN/'run/ckpt_10.pt')==status['selection']['checkpoint_sha256']
    assert b['state']['frames']==head['frames']==head['trace_rows']==25600000
    assert head['updates']==25000
    assert (RUN/'run/gaze.bin').stat().st_size==25600000*16
    assert all(a['model'][k].shape==b['model'][k].shape for k in a['model'])
    schemas={}
    for policy in ['random','guided']:
        j=json.loads((RUN/f'readings/interface_{policy}.json').read_text())
        old=json.loads((BASE/f'readings/interface_{policy}.json').read_text())
        assert j['classes']==old['classes']
        assert j['checkpoints']['p1f16']['classes']==old['checkpoints']['p1f16']['classes']
        schemas[policy]=dict(reference_reproduces_exactly=True,histories_by_class=j['classes'])
    result=dict(audited_at=datetime.datetime.now().astimezone().isoformat(),
        source_pins_checked=len(pins['files']),source_mismatches=bad,
        checkpoint_sha256=sha(RUN/'run/ckpt_10.pt'),control_sha256=sha(BASE/'run/ckpt_10.pt'),
        cfg_equal=True,world_seed=1,world_parameter_shapes_equal=True,
        world_frames=b['state']['frames'],head_updates=head['updates'],trace_rows=head['trace_rows'],
        trace_bytes=(RUN/'run/gaze.bin').stat().st_size,monitor_selection=status['selection'],
        started_at=status['started_at'],training_completed_at=status['training_completed_at'],
        evaluation_completed_at=status['completed_at'],fixed_gaze_reference=schemas)
    (HERE/'PROVENANCE.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['monitor_selection','fixed_gaze_reference']},indent=2))
    assert not bad

if __name__=='__main__':main()
