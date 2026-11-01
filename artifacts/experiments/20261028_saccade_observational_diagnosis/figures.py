"""Standalone figures from observational outputs; no model calls."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .common import *


def read(name):return json.loads((OUT/name).read_text())


if __name__=='__main__':
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    n=read('native_outcome_trace.json');o=read('production_outcomes.json');fig,axes=plt.subplots(1,3,figsize=(13,4.2))
    for ax,group,metric,title in [(axes[0],'all','native_dead_probability','Death: all 30,000 contexts'),(axes[1],'past_low_health_zombie_adjacent','native_dead_probability','Death: past low health + adjacent zombie'),(axes[2],'all','native_negative_reward_probability','Negative reward: all contexts')]:
        key='termination' if metric=='native_dead_probability' else 'negative_reward'
        rr=o[group][key]['recorded'];values=[rr,n[group]['real'][metric],n[group]['generated'][metric]]
        means=[v['est']*100 for v in values];se=[v['se']*196 for v in values]
        ax.bar(range(3),means,yerr=se,capsize=3,color=['#555555','#3465a4','#cc5500']);ax.set_xticks(range(3),['Recorded','Real-input\nhead','Full\nprediction']);ax.set_ylabel('Event probability (%)');ax.set_title(title,fontsize=10)
        for j,value in enumerate(means):ax.text(j,value+se[j]+max(means)*.035,f'{value:.4g}%',ha='center',fontsize=9)
        ax.set_ylim(0,max(means)*1.4)
    fig.suptitle('Same real prefixes; complete observations; native production recipe',fontsize=11);fig.tight_layout();fig.savefig(OUT/'native_outcomes.png',dpi=180);plt.close(fig)

    r=read('risk_timecourse.json');x=np.array([2.56,12.8,25.6]);fig,ax=plt.subplots(figsize=(8.5,4.4))
    group='past_low_health_zombie_adjacent';ax.axhline(100*r['10'][group]['termination']['recorded']['est'],ls='--',c='#555555',label='Recorded rate')
    for which,color,label in [('teacher_forced_dead_probability','#3465a4','Head on real successor'),('generated','#cc5500','Full generated prediction')]:
        q=[r[str(s)][group][which] if which=='teacher_forced_dead_probability' else r[str(s)][group]['termination']['generated'] for s in (1,5,10)]
        y=np.array([v['est']*100 for v in q]);e=np.array([v['se']*196 for v in q]);ax.errorbar(x,y,yerr=e,marker='o',capsize=3,label=label,color=color)
    ax.set_xlabel('Training frames (millions)');ax.set_ylabel('Death probability (%)');ax.set_ylim(-.5,15);ax.legend();ax.set_title('373 fixed risky contexts; four native draws/history per saved snapshot')
    fig.tight_layout();fig.savefig(OUT/'risk_training.png',dpi=180);plt.close(fig)

    a=read('activation_readout.json');d=read('density_semantics.json');arms=['g','adapter','hidden','mu','params','cdf_logits','scale'];names=['Glimpse token','Reconstruction adapter','Mean hidden layer','Location parameters','Location + scale','Endpoint CDF logits','Scale-only context']
    matrix=np.array([[np.mean([a[f'{arm}/{s}']['metrics'][h]['exact']['est'] for s in (0,1)]) for h in HUD[:4]] for arm in arms]+[[d[h]['canonical_digit_likelihood_accuracy']['est'] for h in HUD[:4]]])
    fig,ax=plt.subplots(figsize=(8.5,6));im=ax.imshow(matrix,vmin=0,vmax=1,cmap='Blues');ax.set_xticks(range(4),HUD[:4]);ax.set_yticks(range(8),names+['Native glyph likelihood ranking']);ax.axhline(6.5,c='black',lw=2)
    for (i,j),v in np.ndenumerate(matrix):ax.text(j,i,f'{v*100:.1f}%',ha='center',va='center',color='white' if v>.65 else 'black')
    ax.set_title('Readable value information versus the native pixel distribution');fig.colorbar(im,ax=ax,label='Accuracy');fig.tight_layout();fig.savefig(OUT/'decoder_information.png',dpi=180);plt.close(fig)

    m=read('multiclass_risk.json');old=json.loads((OLD/'headline_metrics.json').read_text())['belief_body'];fig,axes=plt.subplots(1,2,figsize=(11.5,3.8));reps=['full_percept','x','b']
    for ax,metric,title in [(axes[0],'exact','Exact value accuracy'),(axes[1],'ap','Critical-state ranking (AP)')]:
        matrix=np.array([[old[rep][h]['exact'] if metric=='exact' else np.mean([m[f'{rep}/{h}/{s}']['ap'] for s in (0,1)]) for h in HUD[:4]] for rep in reps])
        ax.imshow(matrix,vmin=0,vmax=1,cmap='Blues');ax.set_xticks(range(4),HUD[:4]);ax.set_yticks(range(3),['Current percept','Projected token','Recurrent output']);ax.set_title(title)
        for (i,j),v in np.ndenumerate(matrix):ax.text(j,i,f'{v:.3f}',ha='center',va='center',color='white' if v>.65 else 'black')
    fig.suptitle('Existing multiclass readers; living states for danger; health ≤2, others =0',fontsize=10);fig.tight_layout();fig.savefig(OUT/'value_vs_danger.png',dpi=180);plt.close(fig)

    c=read('reference_support_by_class.json');fig,ax=plt.subplots(figsize=(9,4));xx=np.arange(5)
    for i,domain in enumerate(['real','generated']):
        ax.bar(xx+(i-.5)*.32,[c[f'p/{domain}'][name]['support_fraction']*100 for name in previous.KINDS],.30,label=domain)
    ax.set_xticks(xx,previous.KINDS);ax.set_ylabel('Meets calibrated support rule (%)');ax.set_ylim(0,105);ax.legend();ax.set_title('Periphery reference support by NN-assigned class; bulk coverage hides entity uncertainty')
    fig.tight_layout();fig.savefig(OUT/'reference_support_by_class.png',dpi=180);plt.close(fig)

    d=read('natural_state_readouts.json');fig,ax=plt.subplots(figsize=(8,4));xx=np.arange(4)
    for seed in (0,1):
        q=[d[f'b/{seed}'][h]['natural_exposure_standardised_exact_error'] for h in HUD[:4]]
        ax.errorbar(xx+(seed-.5)*.08,[v['difference']*100 for v in q],yerr=[v['se']*196 for v in q],marker='o',ls='none',capsize=3,label=f'Reader seed {seed}')
    ax.axhline(0,c='gray',ls='--');ax.set_xticks(xx,HUD[:4]);ax.set_ylabel('Exposed minus unexposed error (percentage points)');ax.set_title('Actual recorded glimpses only; standardised observational association');ax.legend();fig.tight_layout();fig.savefig(OUT/'natural_glimpses.png',dpi=180);plt.close(fig)
