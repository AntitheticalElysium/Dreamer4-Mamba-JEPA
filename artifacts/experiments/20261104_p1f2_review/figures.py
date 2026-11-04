"""Standalone figures from frozen descriptive outputs; no fitting or inference."""
from pathlib import Path
import csv
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np

HERE=Path(__file__).resolve().parent
RUN=HERE.parent/'20261103_p1f2'
BLUE='#2463a6';ORANGE='#cf6b24';GRAY='#a8b0b8';GREEN='#25805b'
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
    'figure.facecolor':'white','axes.titleweight':'bold','savefig.facecolor':'white'})

def save(fig,name):
    for suffix in ['png','pdf']:fig.savefig(HERE/f'{name}.{suffix}',dpi=190,bbox_inches='tight')
    plt.close(fig)

def main():
    trace=json.loads((HERE/'TRACE.json').read_text())
    dev=json.loads((HERE/'DEVELOPMENT.json').read_text())
    provenance=json.loads((HERE/'PROVENANCE.json').read_text())
    h=json.loads((RUN/'readings/chooser_trace.json').read_text())
    fig,axes=plt.subplots(1,2,figsize=(12.8,5.4),gridspec_kw={'width_ratios':[1.1,1]})
    names=[('any_entity','Any entity'),('zombies','Zombie'),('cows','Cow'),('skeletons','Skeleton'),
        ('arrows','Arrow'),('entering_map','Entering terrain'),('terrain_edit','Terrain edit'),('changed_hud','Changed HUD')]
    y=np.arange(len(names));a=axes[0]
    a.barh(y+.16,[trace['coverage'][k]['all']['paired_uniform']['mean'] for k,_ in names],height=.30,color=GRAY,label='Paired uniform')
    a.barh(y-.16,[trace['coverage'][k]['all']['learned']['mean'] for k,_ in names],height=.30,color=BLUE,label='Executed F2 gaze')
    for i,(k,_) in enumerate(names):
        x=trace['coverage'][k]['all']['learned']['mean'];a.text(x+.008,i-.16,f'{x:.1%}',va='center',fontsize=9)
    a.set(yticks=y,yticklabels=[v for _,v in names],xlim=(0,.61),xlabel='Crop includes target, when target exists',title='Where F2 actually looked')
    a.invert_yaxis();a.xaxis.set_major_formatter(PercentFormatter(1));a.legend(loc='lower right',frameon=False)
    a=axes[1]
    for kind,label,color in [('any_entity','Entities',BLUE),('entering_map','Entering terrain',ORANGE),('changed_hud','Changed HUD',GREEN)]:
        learned=[trace['coverage'][kind][f'decile_{i}']['learned']['mean'] for i in range(1,11)]
        uniform=[trace['coverage'][kind][f'decile_{i}']['paired_uniform']['mean'] for i in range(1,11)]
        a.plot(range(1,11),learned,'o-',color=color,label=label)
        a.plot(range(1,11),uniform,':',color=color,alpha=.7)
    a.set(xlabel='Training tenth',ylabel='Conditional crop coverage',ylim=(0,.6),xticks=range(1,11),title='Targeting developed during training')
    a.yaxis.set_major_formatter(PercentFormatter(1));a.legend(frameon=False,loc='upper left')
    fig.suptitle('Executed acquisition: 1.30M labelled training occurrences from 383 source episodes',fontsize=13,y=1.01)
    fig.text(.12,-.045,'Overlapping target categories. Repeated corpus passes are clustered by source episode; dotted lines are paired uniform opportunity.',fontsize=9)
    fig.tight_layout();save(fig,'gaze_coverage')

    fig,axes=plt.subplots(2,3,figsize=(12.5,7.5),sharex=True)
    for cl,a in zip(['drink','wood','stone','coal','iron'],axes.flat):
        rr=[dev[str(c)]['cohorts']['confirmation']['counter'][cl]['increment'] for c in range(1,11)]
        a.plot(range(1,11),[r['normalized_error'] for r in rr],'o-',color=ORANGE,label='Unpredicted counter change')
        a.plot(range(1,11),[r['normalized_C'] for r in rr],'s-',color=BLUE,label='Action-conditioned counter change')
        a.axhline(1,color=GRAY,ls=':',lw=1);a.set(title=cl.capitalize(),ylim=(-.04,1.1),xticks=[1,3,5,7,10])
        a.grid(axis='y',alpha=.15)
    a=axes.flat[-1]
    colors=[BLUE,ORANGE,GREEN,'#9057a3','#9c822d']
    for cl,color in zip(['drink','wood','stone','coal','iron'],colors):
        vals=[dev[str(c)]['cohorts']['confirmation']['policy'][cl]['combined']['remote_mass']['mean'] for c in range(1,11)]
        a.plot(range(1,11),vals,'o-',ms=3,color=color,label=cl)
    a.set(title='Probability crop covers its counter',ylim=(0,.7),xticks=[1,3,5,7,10]);a.yaxis.set_major_formatter(PercentFormatter(1));a.legend(ncol=2,fontsize=8,frameon=False)
    for a in axes[1]:a.set_xlabel('Saved checkpoint (2.56M world frames each)')
    for a in axes[:,0]:a.set_ylabel('Squared token change / copying error')
    handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.52,.965),ncol=2,frameon=False)
    fig.suptitle('The saved online chooser learned a selective surprise-to-contingency handover',fontsize=13,y=1.015)
    fig.text(.08,-.02,'Same confirmation contexts at every checkpoint; each head uses its own encoder. Counter-only signals, not neighbouring HUD overlap.\nCopying error = actual counter token change at that checkpoint. These are chooser forecasts, not main-world generations.',fontsize=9)
    fig.tight_layout(rect=(0,0,1,.92));save(fig,'handover')

    fig,axes=plt.subplots(1,3,figsize=(12,4))
    fields=[]
    for dec in ['1','10']:
        x=np.array(h['deciles'][dec]['window_histogram'],float);fields.append((x/x.sum()*49).reshape(7,7))
    fields.append(fields[1]-fields[0])
    for i,(a,z,title) in enumerate(zip(axes,fields,['First training tenth','Final training tenth','Final minus first'])):
        im=a.imshow(z,cmap='RdBu_r' if i==2 else 'viridis',vmin=-.6 if i==2 else .5,vmax=.6 if i==2 else 1.9)
        a.set(title=title,xticks=range(7),xticklabels=range(1,8),yticks=range(7),yticklabels=range(1,8),xlabel='Crop-center column')
        a.axhline(5.5,color='white',lw=1,ls='--');fig.colorbar(im,ax=a,fraction=.046,pad=.04)
    axes[0].set_ylabel('Crop-center row');fig.suptitle('Whole-run spatial averages: center frequency relative to uniform (=1)',fontsize=13)
    fig.text(.05,-.08,'All 25.6M executed gazes. Last two center rows can include the HUD. Spatial averages hide context-dependent tracking;\nuse the conditional target-coverage figure to assess whether relevant content was included.',fontsize=9)
    fig.tight_layout();save(fig,'center_distribution')

    # Compact all-class outcome file is already available; this panel retains
    # selected successes and regressions without selecting a best sampler.
    rows=list(csv.DictReader((HERE/'INTERFACE.csv').open()))
    selected=['ach_COLLECT_COAL','ach_COLLECT_IRON','ach_EAT_COW','ach_DEFEAT_SKELETON',
        'ach_COLLECT_WOOD','ach_COLLECT_STONE','ach_DEFEAT_ZOMBIE','ach_WAKE_UP']
    rr=[next(r for r in rows if r['policy']=='random' and r['metric']=='unlock_p' and r['context']==cl) for cl in selected]
    fig,a=plt.subplots(figsize=(9,4.5));x=np.array([float(r['difference']) for r in rr])*100
    low=np.array([float(r['ci95_low']) for r in rr])*100;high=np.array([float(r['ci95_high']) for r in rr])*100
    a.errorbar(x,np.arange(len(rr)),xerr=np.stack([x-low,high-x]),fmt='o',color=BLUE,capsize=3)
    a.axvline(0,color=GRAY,lw=1);a.set(yticks=range(len(rr)),yticklabels=['Collect coal','Collect iron','Eat cow','Defeat skeleton','Collect wood','Collect stone','Defeat zombie','Wake up'],xlabel='F2 minus F1.6: generated reward>0.5 head mass (percentage points)',title='Main-world consequences remain mixed');a.invert_yaxis()
    fig.text(.1,-.035,'Same fixed random gaze, histories and production sampler. Outcome-selected achievement slices, not unconditional calibration.\nIntervals reflect evaluation sampling, not between-training-seed uncertainty; one matched world seed.',fontsize=9)
    fig.tight_layout();save(fig,'world_outcomes')

if __name__=='__main__':main()
