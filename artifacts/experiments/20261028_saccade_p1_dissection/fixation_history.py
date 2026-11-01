"""Actual S1 fixation gaps and conditional coverage; no learned policy is assumed."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .common import *


if __name__=='__main__':
    L=labels('dev');enc=torch.load(OUT/'real_dev.pt',weights_only=False,mmap=True)
    values=hud_values(L);off=np.r_[0,np.cumsum(L['ep_T']+1)];rows=[]
    for e,r in enumerate(enc):
        cells=crop_cells(r['fix']);n=len(cells);seen=(cells[:,:,None]==HUD_CELLS[None,None]).any(1)
        last=np.full(16,-1);gap=np.empty((n,16),np.int32)
        for t in range(n):
            last[seen[t]]=t;gap[t]=np.where(last>=0,t-last,-1)
        y=values[off[e]:off[e+1]];changed=np.r_[np.zeros((1,16),bool),y[1:]!=y[:-1]]
        rows.append({'seen':seen,'gap':gap,'changed':changed,'values':y,'ep':np.full(n,e)})
    R={k:np.concatenate([r[k] for r in rows]) for k in rows[0]};out={}
    for j,name in enumerate(HUD):
        gap=R['gap'][:,j];seen=R['seen'][:,j];m=np.ones(len(gap),bool)
        out[name]={'seen_now':G.estimate(seen,m,R['ep']),'never_seen_yet':G.estimate(gap<0,m,R['ep']),
                   'last_seen_gap_quantiles_if_seen':np.quantile(gap[gap>=0],[.5,.9,.99]).tolist(),
                   'seen_after_value_change':G.estimate(seen,R['changed'][:,j],R['ep']),
                   'seen_when_value_le_3':G.estimate(seen,R['values'][:,j]<=3,R['ep'])}
    save('fixation_history.json',out)
    train=json.loads((OUT/'training_fixations.json').read_text());centres=np.array(train['centres'])/train['frames']*100
    coverage=np.array(train['analytic_cell_coverage'])*100
    fig,axes=plt.subplots(1,3,figsize=(14,4.8),gridspec_kw={'width_ratios':[1,1,1.4]})
    for ax,a,title in zip(axes[:2],(centres,coverage),('Training crop centres (%)','Cell included in random crop (%)')):
        im=ax.imshow(a,vmin=0,cmap='Blues');ax.set_title(title);fig.colorbar(im,ax=ax,fraction=.046)
        for (r,c),v in np.ndenumerate(a):ax.text(c,r,f'{v:.1f}',ha='center',va='center',fontsize=7,color='black')
    axes[1].axhline(6.5,color='orange');axes[1].set_yticks(range(9),[f'map {i}' if i<7 else f'HUD {i-7}' for i in range(9)])
    axes[0].set_xticks(range(7),range(1,8));axes[0].set_yticks(range(7),range(1,8));axes[0].set_xlabel('Centre column');axes[0].set_ylabel('Centre row')
    axes[2].barh(HUD,[100*train['hud_inclusion_rates'][h] for h in HUD]);axes[2].invert_yaxis();axes[2].set_xlabel('Training frames whose crop includes field (%)');axes[2].set_title('HUD exposure: geometry, not attention')
    fig.tight_layout();fig.savefig(OUT/'fixation_coverage.png',dpi=180);plt.close(fig)
