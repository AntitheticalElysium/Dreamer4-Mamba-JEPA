"""Read Claude's completed M1-M3 and the independently repaired M4, without modifying parent artifacts."""
import json
from pathlib import Path
HERE=Path(__file__).resolve().parent
PARENT=HERE.parent/'20261107_f4_followups/readings/danger/denoiser_teaching_f4.json'

def main():
    r=json.loads(PARENT.read_text());m=r['M1 mean |dL/d bp| x1e6 (pass / class / flow, boot)']
    v=json.loads((HERE/'VISIBILITY.json').read_text())
    result={'parent':str(PARENT),'microbatches':r['microbatches'],'magnitude':{},'reach':r['M3 reach'],
            'state_gradient_consistency_descriptive_only':r['M2 consistency |sum g| / sum |g| (and 1/sqrt(n))'],
            'visibility':v,'interpretation':[
                'Student bootstrap gradients are nonzero; both branches have event-sensitive memory gradients.',
                'The declared combined <=1%-at-lag10 hard-credit-cutoff condition is not met.',
                'Low state-gradient consistency would not prove update conflict; see the exact counterexample.',
                'M4 detects visible digit changes, not every hidden-counter reset.',
                'Readability is conditional on this reader family, not proof of irrecoverable erasure.',
                'None of these end-checkpoint derivatives identifies the causal training repair.'
            ]}
    for event,part in [('food_down','food'),('recovery','health'),('hit','health')]:
        result['magnitude'][event]={}
        for branch in ['flow','boot']:
            a,b=m[f'g1/{part} / {event}'][branch],m[f'g1/{part} / ordinary'][branch]
            result['magnitude'][event][branch]={'event_grad_x1e6':a,'ordinary_grad_x1e6':b,'ratio':a/b}
    for event,rec in result['reach'].items():
        rec['lag10_below_one_percent']=rec['relative to k = 1 at k = 5, 10, 20, 30, 40'][1]<.01
    (HERE/'SUMMARY.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({'magnitude':result['magnitude'],'changed_AP':{k:{p:x[p]['ap'] for p in ['HUD_abs','embedding_abs']} for k,x in v['changed'].items()}},indent=2))

if __name__=='__main__':main()
