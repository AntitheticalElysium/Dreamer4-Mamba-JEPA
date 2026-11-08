"""Static scientific figures from the archived paired measurements."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
g=json.loads((HERE/'gaze.json').read_text())
b=json.loads((HERE/'belief.json').read_text())


def err(ax,y,stats,color,label,offset=0,scale=1):
    v=np.array([[s[k]*scale for k in ('mean','low','high')] for s in stats])
    ax.errorbar(v[:,0],np.asarray(y)+offset,xerr=np.stack([v[:,0]-v[:,1],v[:,2]-v[:,0]]),
                fmt='o',ms=4,capsize=2,color=color,label=label,lw=1)


def main():
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig=plt.figure(figsize=(12,10),layout='constrained')
    gs=fig.add_gridspec(2,2,height_ratios=[1,1.08])
    ax=fig.add_subplot(gs[0,0])
    cats=['any_entity','zombies','cows','skeletons','arrows','entering_map','changed_hud']
    labels=['Any entity (404)','Zombie (231)','Cow (220)','Skeleton (102)',
            'Arrow (61)','Entering strip (239)','Changed HUD (144)']
    y=np.arange(len(cats))
    names=[('native','Native','#495d8a',-.2),('E2_carry_611601','Cheap A','#008d8a',0),
           ('E2_carry_611602','Cheap B','#cf7434',.2)]
    for name,label,color,offset in names:
        stats=[g['saved'][name]['existing']['coverage_'+k]['policy'] for k in cats]
        err(ax,y,stats,color,label,offset,100)
    uniform=[g['saved']['native']['uniform']['coverage_'+k]['policy']['mean']*100 for k in cats]
    ax.scatter(uniform,y,color='#999999',marker='x',label='Uniform',zorder=3)
    ax.set_yticks(y,labels);ax.invert_yaxis();ax.set_xlim(0,65)
    ax.set_xlabel('Probability crop covers target (%)')
    ax.set_title('A. Broad gaze: matched random frames\nCounts in parentheses; 95% episode-bootstrap intervals',loc='left')
    ax.legend(loc='lower right',fontsize=8)
    ax=fig.add_subplot(gs[0,1])
    selections=[('native','Native'),('E1_611601','4k A'),('E1_611602','4k B'),
                ('E2_carry_611601','12k A'),('E2_carry_611602','12k B')]
    categories=[(['terrain_common','terrain_other'],'Other terrain','#bec7cb'),
                (['terrain_resources','terrain_structures'],'Resources / structures','#90ad79'),
                (['zombies','cows','skeletons','arrows'],'Entities','#b26b52'),
                (['player'],'Player','#696c8d'),
                (['health','food','drink','energy','inventory','blank_hud'],'HUD','#c5ad68')]
    x=np.arange(len(selections));bottom=np.zeros(len(x))
    for ks,label,color in categories:
        val=[100*sum(g['saved'][name]['existing']['all_random']['center_share_'+k]['mean'] for k in ks)
             for name,_ in selections]
        ax.bar(x,val,bottom=bottom,color=color,label=label,width=.7);bottom+=val
    ax.set_xticks(x,[label for _,label in selections]);ax.set_ylim(0,100)
    ax.set_ylabel('Expected fixation centers (%)')
    ax.set_title('B. Counters are a minority of overall gaze\n600 random frames, original mixed policy',loc='left')
    ax.legend(loc='upper center',bbox_to_anchor=(.5,-.1),ncol=2,fontsize=8,frameon=False)
    ax=fig.add_subplot(gs[1,:])
    pairs=[('all',None,'Whole screen (978)'),('action_move',None,'Movement frames (51)'),
           ('all','any_entity','Entity cells (609)'),('all','near_entity','Near-entity cells (359)'),
           ('all','terrain_edit','Edited terrain cells (470)'),
           ('all','changed_hud','Changed HUD cells (672)'),
           ('age_over128',None,'Age >128 (473)'),('health_loss',None,'Health-loss frames (16)')]
    y=np.arange(len(pairs))
    for seed,label,color,off in [('611601','Head A','#008d8a',-.12),('611602','Head B','#cf7434',.12)]:
        stats=[]
        for group,target,_ in pairs:
            z=b['confirmation'][seed][group]
            stats.append((z if target is None else z['targets'][target])['percent_reduction'])
        err(ax,y,stats,color,label,off)
    ax.axvline(0,color='#777777',lw=.8)
    ax.set_yticks(y,[x[2] for x in pairs]);ax.invert_yaxis()
    ax.set_xlabel('Reduction in coarse-token prediction MSE from native belief (%)     ← worse     better →')
    ax.set_title('C. Belief input: selective gains, not a consistent rescue\nFresh cohort; counts are evaluation rows; no world-model training',loc='left')
    ax.legend(loc='lower right')
    fig.suptitle('Where the frozen cheap chooser looks — and what the belief arm changes',fontsize=15)
    fig.savefig(HERE/'gaze_belief.png',dpi=175)
    fig.savefig(HERE/'gaze_belief.pdf')
    plt.close(fig)


if __name__=='__main__':main()
