"""Standalone scientific figures from saved diagnostics; no model calls."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .common import *


def read(name):return json.loads((OUT/name).read_text())


if __name__=='__main__':
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    b=read('belief_readers.json');fig,axes=plt.subplots(1,2,figsize=(12,4))
    for ax,(task,sel,title) in zip(axes,[('joint','original_s1_stop','Original multi-task reader / stop'),('body','body','Dedicated body reader; frozen world')]):
        a=np.array([[np.mean([b[f'{rep}/{task}/{sel}/{s}']['metrics'][h]['exact']['est'] for s in (0,1)]) for h in HUD[:4]] for rep in ('full_percept','x','b')])
        im=ax.imshow(a,vmin=0,vmax=1,cmap='Blues');ax.set_xticks(range(4),HUD[:4]);ax.set_yticks(range(3),['Current percept','Projected token (256)','Recurrent output (256)']);ax.set_title(title)
        for (r,c),v in np.ndenumerate(a):ax.text(c,r,f'{100*v:.1f}%',ha='center',va='center',color='white' if v>.68 else 'black')
    fig.subplots_adjust(left=.17,right=.90,wspace=.80,bottom=.18,top=.84)
    fig.colorbar(im,cax=fig.add_axes([.94,.2,.014,.6]),label='Exact-value accuracy')
    fig.savefig(OUT/'body_readout.png',dpi=180);plt.close(fig)

    d=read('decoder_timecourse.json');x=np.arange(1,11)*2.56;fig,axes=plt.subplots(1,3,figsize=(12.5,3.7))
    for key,color in [('map','#3465a4'),('HUD','#75507b'),('food','#cc5500')]:
        for ax,measure in zip(axes,['nll','mae','scale']):ax.plot(x,[d[str(i)][key][measure]['est'] for i in range(1,11)],'-o',ms=3,label=key,color=color)
    for ax,title in zip(axes,['Reconstruction NLL / RGB scalar','Clipped-location RGB error','Mean logistic scale (RGB units)']):ax.set_title(title);ax.set_xlabel('Training frames (millions)');ax.grid(alpha=.2)
    axes[2].set_yscale('log');axes[0].legend();fig.suptitle('Same 8,192 real frames re-encoded by each frozen snapshot',y=1.01);fig.tight_layout();fig.savefig(OUT/'decoder_training.png',dpi=180,bbox_inches='tight');plt.close(fig)

    a=read('sampler_conditional_counts.json')['population'];es=read('sampler_energy_scores.json');fig,axes=plt.subplots(1,2,figsize=(12.5,4.5),gridspec_kw={'width_ratios':[1.65,1]})
    arms=['reader_real_count','production_3_churn1','ode_3','ode_8','ode_32'];names=['Real reader','Production 3','ODE 3','ODE 8','ODE 32'];xx=np.arange(4)
    for j,(arm,name) in enumerate(zip(arms,names)):
        vals=[a[k][arm]['est'] for k in KINDS[1:]];err=[1.96*a[k][arm]['se'] for k in KINDS[1:]]
        axes[0].bar(xx+(j-2)*.15,vals,.145,yerr=err,capsize=2,label=name)
    axes[0].set_xticks(xx,KINDS[1:]);axes[0].set_ylabel('Mean visible count through same frozen reader');axes[0].set_title('Counts move with sampler settings');axes[0].legend(fontsize=8,ncol=2)
    ar=arms[1:];axes[1].bar(range(4),[es[k]['joint']['energy_score']['est'] for k in ar],yerr=[1.96*es[k]['joint']['energy_score']['se'] for k in ar],capsize=3)
    axes[1].set_xticks(range(4),names[1:],rotation=20);axes[1].set_ylabel('Joint latent energy score (lower is better)');axes[1].set_title('Count agreement is not the whole distribution')
    fig.suptitle('1,024 fixed population histories, four independent futures per arm; episode-bootstrap intervals',fontsize=10);fig.tight_layout();fig.savefig(OUT/'sampling_counts.png',dpi=180);plt.close(fig)

    f=read('fovea_readers.json');fig,ax=plt.subplots(figsize=(10,4.5));arms=['periphery','fovea','joint','pixels_full'];names=['Periphery','Fovea cell','Joint','Raw full pixels'];cols=['#3465a4','#cc5500','#4e9a06','#75507b']
    for j,(arm,name,color) in enumerate(zip(arms,names,cols)):
        vals=np.array([[f[f'map/{arm}/{s}']['metrics']['night/r1_2'][k+'_recall']['est'] for k in KINDS[1:]] for s in (0,1)])
        pos=xx+(j-1.5)*.19;ax.bar(pos,vals.mean(0),.18,label=name,color=color,alpha=.8)
        for s in (0,1):ax.scatter(pos+(s-.5)*.035,vals[s],s=13,c='black',zorder=4)
    ax.set_xticks(xx,KINDS[1:]);ax.set_ylim(0,1.05);ax.set_ylabel('Kind recall on entity cells');ax.set_title('Same selected crop cells at night, rings 1–2; dots are the two reader seeds');ax.legend(ncol=4,fontsize=9)
    fig.tight_layout();fig.savefig(OUT/'fovea_readout.png',dpi=180);plt.close(fig)
