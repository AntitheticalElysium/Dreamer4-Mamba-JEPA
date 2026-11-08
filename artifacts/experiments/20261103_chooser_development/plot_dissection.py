"""Standalone research figure from archived, endpoint-identical replay measurements."""
import importlib
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

C=importlib.import_module('artifacts.experiments.20261103_chooser_development.common')


def main():
    steps=np.array([0,100,1000,2000,4000,6000,8000,10000,12000])
    data={seed:[json.loads((C.HERE/f'replay_fresh_{seed}_u{u:05d}_confirmation.json').read_text())
                for u in steps] for seed in (611601,611602)}
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                         'axes.spines.right':False,'axes.titlesize':11,'axes.labelsize':10})
    fig,axs=plt.subplots(2,2,figsize=(12.8,8.5),layout='constrained')
    colors={'C_softmax':'#176b9b','S_softmax':'#d17c1c','existing':'#303941'}
    for ax,seed,title in zip(axs[0],(611601,611602),('A. Fixed encoder, head seed A','B. Same recipe, head seed B')):
        for pol,label in (('C_softmax','Contingency only'),('S_softmax','Surprise only'),('existing','Combined policy')):
            y=[100*r['drink']['policies'][pol]['remote_mass']['mean'] for r in data[seed]]
            ax.plot(steps/1000,y,color=colors[pol],label=label,lw=2,marker='o',ms=3)
        ax.axhline(100*6/49,color='#92999e',lw=1,ls=':',label='Uniform (12.2%)')
        ax.axvline(4,color='#b6bbc0',ls='--',lw=1)
        ax.set(title=title,xlabel='Head updates (thousands)',ylabel='Drink-counter coverage probability (%)',ylim=(0,65),xlim=(0,12))
        ax.grid(axis='y',alpha=.15)
    axs[0,0].legend(fontsize=9,loc='upper left')
    ax=axs[1,0]
    for seed,style,lab in ((611601,'-','A'),(611602,'--','B')):
        for cl,col in (('drink','#176b9b'),('stone','#8d4992')):
            y=[r[cl]['errors']['counter_mse']/r[cl]['errors']['counter_copy_mse'] for r in data[seed]]
            ax.plot(steps/1000,y,color=col,ls=style,lw=2,label=f'{cl.capitalize()}, seed {lab}')
    ax.axhline(1,color='#92999e',ls=':',lw=1)
    ax.axvline(4,color='#b6bbc0',ls='--',lw=1)
    ax.set(title='C. Counter prediction improves where contingency emerges',xlabel='Head updates (thousands)',
           ylabel='Counter-token MSE / copy-previous error',ylim=(0,1.08),xlim=(0,12))
    ax.grid(axis='y',alpha=.15);ax.legend(fontsize=9)
    paired=json.loads((C.HERE/'paired_dissection.json').read_text())['confirmation']
    ax=axs[1,1];classes=('drink','stone','coal','iron');x=np.arange(4)
    for seed,shift,col,lab in ((611601,-.12,'#176b9b','Head seed A'),(611602,.12,'#d17c1c','Head seed B')):
        q=[paired[f'e3_rich_minus_coarse_{seed}'][cl]['existing']['remote'] for cl in classes]
        y=np.array([v['mean'] for v in q])*100
        lo=np.array([v['low'] for v in q])*100;hi=np.array([v['high'] for v in q])*100
        ax.errorbar(x+shift,y,yerr=[y-lo,hi-y],fmt='o',color=col,label=lab,capsize=4,ms=5)
    ax.axhline(0,color='#92999e',lw=1)
    ax.set(title='D. Richer history: small changes in the 4k input test',
           ylabel='Gain in counter coverage (percentage points)',xticks=x,xticklabels=[c.capitalize() for c in classes],ylim=(-3,5))
    ax.grid(axis='y',alpha=.15);ax.legend(fontsize=9)
    fig.suptitle('Cheap fixation: a learnable handover, with unstable acquisition',fontsize=15,fontweight='bold',y=.98)
    fig.get_layout_engine().set(rect=(0,.12,1,.78))
    fig.text(.025,.025,'A–C: exact replays of E2 fresh 12k fits, fixed P1-F1.5 checkpoint10. Dashed vertical line = 4k within this schedule;\n'
             'E1 used a different 4k decay schedule. D: paired episode-bootstrap 95% intervals, conditional on each fitted head.\n'
             'Fresh cohort; 128 drink/stone/coal contexts, 82 iron. Coverage is not prediction accuracy or measured world-learning benefit.',
             fontsize=9,color='#505861')
    fig.savefig(C.HERE/'chooser_dissection.png',dpi=180)
    fig.savefig(C.HERE/'chooser_dissection.pdf')


if __name__=='__main__':main()
