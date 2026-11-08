"""Archived head utility, combined gaze and exact-feedback timing."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent


def main():
    r=json.loads((HERE/'COMPARISONS.json').read_text())['confirmation']
    cost=json.loads((HERE/'COST_V2.json').read_text())
    full=json.loads((HERE/'cost_full_compile.json').read_text())
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig=plt.figure(figsize=(12,8),layout='constrained');gs=fig.add_gridspec(2,2)
    ax=fig.add_subplot(gs[0,0]);colors=['#008d8a','#cf7434']
    for j,s in enumerate(['611601','611602']):
        qs=[r[s]['all']['contrasts'][key]['percent_mse_reduction'] for key in ['belief_at1','belief_at8']]
        x=np.array([q['mean'] for q in qs]);lo=np.array([q['low'] for q in qs]);hi=np.array([q['high'] for q in qs])
        ax.errorbar(x,np.arange(2)+(j-.5)*.13,xerr=np.stack([x-lo,hi-x]),fmt='o',capsize=3,
                    color=colors[j],label='Head '+('A' if j==0 else 'B'))
    ax.set_yticks([0,1],['1 coarse frame','8 coarse frames']);ax.invert_yaxis();ax.axvline(0,color='#999999',lw=.8)
    ax.set_xlabel('Prediction MSE reduction from native belief (%)')
    ax.set_title('A. Redundancy effect depends on the fit\nSame 12k schedule; fresh 978-row cohort; 95% intervals',loc='left');ax.legend()
    ax=fig.add_subplot(gs[0,1]);x=np.arange(2);w=.2
    variants=[('Coarse only',full,'C','#aab3b9'),('1B eager',cost['eager'],'B','#dbbaa2'),
              ('1B compiled Mamba',cost['compiled'],'B','#769cbe'),('1B compiled scorer + Mamba',full,'B','#008d8a')]
    for j,(label,bank,arm,color) in enumerate(variants):
        v=[bank['shapes'][shape][arm]['collection']['median'] for shape in ['16x64','4x256']]
        ax.bar(x+(j-1.5)*w,v,width=w,color=color,label=label)
    ax.set_xticks(x,['16 streams × 64 steps','4 streams × 256 steps']);ax.set_ylabel('Collection seconds / 1024 new frames')
    ax.set_title('B. Fresh fine-belief feedback remains costly\nOwn selected glimpses; native reset/replay checks',loc='left')
    ax.legend(fontsize=8,loc='upper left')
    ax=fig.add_subplot(gs[1,:]);classes=['drink','wood','sapling','stone','coal','iron'];arms=['1C','1B','8C','8B']
    rows=[];labels=[]
    for seed,letter in [('611601','A'),('611602','B')]:
        for arm in arms:
            rows.append([100*r[seed]['class_'+cl]['arms'][arm]['policies']['existing']['remote']['mean'] for cl in classes])
            labels.append(arm+' — head '+letter)
    values=np.array(rows);im=ax.imshow(values,cmap='YlGnBu',vmin=0,vmax=50,aspect='auto')
    for i in range(8):
        for j in range(6):ax.text(j,i,f'{values[i,j]:.1f}%',ha='center',va='center',color='white' if values[i,j]>32 else '#1f2730')
    ax.axhline(3.5,color='white',lw=3);ax.set_xticks(range(6),classes);ax.set_yticks(range(8),labels)
    ax.set_title('C. Combined gaze has no consistent belief advantage\nRemote-counter crop coverage; 20% uniform retained; labels used only for evaluation',loc='left')
    fig.colorbar(im,ax=ax,label='Crop coverage probability (%)',fraction=.025,pad=.015)
    fig.suptitle('Coarse history, native belief and the cost of fresh feedback',fontsize=15)
    fig.savefig(HERE/'history_belief_cost.png',dpi=170);fig.savefig(HERE/'history_belief_cost.pdf');plt.close(fig)


if __name__=='__main__':main()
