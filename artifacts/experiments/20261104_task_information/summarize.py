"""Aggregate retained results and plot the only source passing both checks.

This is reporting, with no selection fed back into fitting or simulation.
"""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/saccade-task-info-mpl')
from pathlib import Path
import json,hashlib
from collections import Counter
import numpy as np
import torch
import torch.nn.functional as F
HERE=Path(__file__).resolve().parent

def read(name):return json.loads((HERE/name).read_text())

def main():
    focus=read('focus_balanced_results.json');paired=read('focus_paired_readers.json')
    causal=torch.load(HERE/'balanced_consequences.pt',map_location='cpu',weights_only=False)
    byframe={r['meta']['frame']:r for r in causal['rows']};witness={r['meta']['frame']:r for r in read('witness_checks.json')}
    paired_causal=[]
    for r in paired:
        cr=byframe[r['meta']['frame']];v=cr['values'];d=v[0,:,:,0]-v[1,:,:,0]
        if r['qualified']:
            paired_causal.append(dict(meta=r['meta'],has_any_causal_damage_difference=bool((d!=0).any()),
                damage_mean_by_plan=v[:,:,:,0].mean(-1).tolist(),
                reward_mean_by_plan=v[:,:,:,1].mean(-1).tolist(),
                differing_damage_keys_by_plan=(d!=0).sum(-1).tolist(),
                mean_paired_damage_difference_by_plan=d.mean(-1).tolist(),witness=witness[r['meta']['frame']]))
    summary=dict(checkpoint_sha256='34bda77fe5ec5226c4f003301c972d918e2d4a4a554a5a6f279fcabbe922b045',
        full_context_contract=read('checks.json'),focused_sources=len(focus['rows']),
        qualified_sources=sum(r['qualified'] for r in focus['rows']),
        qualified_by_cohort=dict(Counter(r['cohort'] for r in focus['rows'] if r['qualified'])),
        selected_sources=len(focus['selected']),paired_survivors=len(paired_causal),
        paired_survivors_by_cohort=dict(Counter(r['meta']['cohort'] for r in paired if r['qualified'])),
        causal_survivors=sum(r['has_any_causal_damage_difference'] for r in paired_causal),
        cases=paired_causal,plans=causal['plans'],keys=causal['keys'],horizon=causal['horizon'],
        interpretation='One natural-frame, finite-reader, task-relevant witness. No robust confirmation or information-theoretic necessity claim.')
    (HERE/'SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n')
    # Display unchanged uint8 sensor samples; no brightness/contrast adjustment.
    arrays=torch.load(HERE/'witness_arrays.pt',map_location='cpu',weights_only=False)
    passing=[r for r in paired_causal if r['has_any_causal_damage_difference']]
    if passing:
        assert len(passing)==1
        m=passing[0]['meta'];frames=next(r['current'] for r in arrays if r['meta']['frame']==m['frame'])
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
        rr,cc=m['focus'];r0,c0=(rr-1)*7,(cc-1)*7
        crop=frames[0,r0:r0+21,c0:c0+21];absent=frames[1,r0:r0+21,c0:c0+21]
        cell=torch.as_tensor(crop.copy()).reshape(3,7,3,7,3).permute(0,2,4,1,3).reshape(9,3,7,7).float()
        pooled=F.adaptive_avg_pool2d(cell,3).round().byte().reshape(3,3,3,3,3).permute(0,3,1,4,2).reshape(9,9,3).numpy()
        fig,axs=plt.subplots(1,4,figsize=(12,3.5),layout='constrained')
        for ax,img,title in zip(axs,[frames[0],pooled,crop,absent],['Natural source frame','Coarse crop: 3 px per cell','Fine crop: 7 px per cell','Fine crop: threat removed']):
            ax.imshow(img,interpolation='nearest');ax.set_title(title,fontsize=10);ax.axis('off')
        row,col=divmod(m['cell'],9)
        axs[0].add_patch(Rectangle((col*7-.5,row*7-.5),7,7,fill=False,edgecolor='red',linewidth=1.2))
        for ax,scale in [(axs[1],3),(axs[2],7),(axs[3],7)]:
            ax.add_patch(Rectangle(((col-(cc-1))*scale-.5,(row-(rr-1))*scale-.5),scale,scale,fill=False,edgecolor='red',linewidth=1.2))
        fig.suptitle(f"Episode {m['episode']}, step {m['time']}: skeleton cue; original pixel intensities",fontsize=12)
        fig.savefig(HERE/'natural_witness.png',dpi=170);plt.close(fig)
    print(json.dumps({k:v for k,v in summary.items() if k not in ['cases']},indent=2))

if __name__=='__main__':main()
