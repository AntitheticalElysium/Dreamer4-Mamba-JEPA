"""Saved-output analysis. Intervals cluster episodes, conditional on the trained worlds."""
import importlib, json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
A=importlib.import_module('artifacts.experiments.20261101_regression_localization.analyze')
E=A.AN.estimate

def main():
    out={'scope':'Bounded frozen-world diagnostics; episode intervals are not world-seed generalization.',
         'solver':{},'gradients':{},'recursion':{}}
    base=A.read(HERE/'solver/e3c1.npz',4)
    for name in ('e3c1','e8c0','e32c1'):
        z=A.read(HERE/f'solver/{name}.npz',4)
        out['solver'][name]={'delta_vs_production':A.compare(base,z),
            'absolute':{cl:{k:E(v,rr['ep']) for k,v in rr.items() if k.startswith(('real_','gen_','truth_'))}
                        for cl,rr in z.items()}}
    zz=json.loads((HERE/'gradients/summary.json').read_text())
    for tag,c in zz['cases'].items():
        bb=c['blocks'];oo=out['gradients'][tag]={'blocks':len(bb),'frames':len(bb)*128,
             'instrument_full_gradient_error':c['instrument_full_gradient_max_error'],
             'route_addition_relative_error_max':max(b['gradient_sum_relative_error'] for b in bb),'modules':{}}
        for m in bb[0]['module_total_norm']:
            f=lambda values:{'median':float(np.median(values)),'q25':float(np.quantile(values,.25)),
                             'q75':float(np.quantile(values,.75))}
            oo['modules'][m]={'total_norm':f([b['module_total_norm'][m] for b in bb]),
                'routes':{r:{key:f([b['routes'][r][m][key] for b in bb]) for key in ('norm','projection_fraction')}
                          for r in bb[0]['routes']},
                'adam_fine_removal_difference_over_full':f([b['adam']['modules'][m]['difference_over_full'] for b in bb])}
    z=np.load(HERE/'recursion/raw.npz');ep=z['episode'][::4]
    out['recursion']['windows']=len(ep);out['recursion']['episodes']=len(np.unique(ep))
    out['recursion']['first_step_equal']=json.loads((HERE/'recursion/identity.json').read_text())['first_step_equal']
    mean_draw=lambda a:a.reshape(len(ep),4,*a.shape[1:]).mean(1)
    for h in (1,4,8,16):
        rr=out['recursion'][h]={};j=h-1
        for source in ('truth','real_reader','teacher','replay','free'):
            if source=='truth':counts=z['truth_count'][:,j]
            elif source=='real_reader':counts=z['teacher__real_reader_count'][:,j]
            else:counts=z[source+'__count'][:,j]*z[source+'__active'][:,j,None]
            counts=mean_draw(counts)
            rr[source]={k:E(counts[:,i],ep) for i,k in enumerate(('zombie','cow','skeleton','arrow'))}
            rr[source]['all']=E(counts.sum(-1),ep)
        for mode in ('teacher','replay','free'):
            rr[mode]['positive_probability']=E(mean_draw(z[mode+'__positive'][:,j]*z[mode+'__active'][:,j]),ep)
            rr[mode]['active']=E(mean_draw(z[mode+'__active'][:,j].astype(float)),ep)
        for kind,i in [('all',None),('cow',1),('zombie',0)]:
            diff=mean_draw(z['free__count'][:,j]*z['free__active'][:,j,None]-z['teacher__count'][:,j])
            rr['free_minus_teacher_'+kind]=E(diff.sum(-1) if i is None else diff[:,i],ep)
    (HERE/'analysis.json').write_text(json.dumps(out,indent=1))
    # Scientific figure: measured reader outputs with clustered intervals; no implied generated ground truth.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    xs=np.array([1,4,8,16]);fig,ax=plt.subplots(figsize=(8,4.4))
    for key,label,color in [('truth','Recorded simulator count','#333333'),('real_reader','Reader on real percepts','#8a8a8a'),
                            ('teacher','Generated, real history','#2677b8'),('replay','Generated, stepwise real replay','#50a090'),
                            ('free','Generated, own history','#cc493c')]:
        q=[out['recursion'][int(h)][key]['all'] for h in xs]
        y=np.array([r['mean'] for r in q]);lo=np.array([r['ci95'][0] for r in q]);hi=np.array([r['ci95'][1] for r in q])
        ax.plot(xs,y,'o-',color=color,label=label);ax.fill_between(xs,lo,hi,color=color,alpha=.10)
    ax.set(xlabel='Future step',ylabel='Visible entity count (four kinds)',xticks=xs,
           title='P1-F1.5: entity readouts decline under generated history')
    ax.legend(fontsize=8);ax.spines[['top','right']].set_visible(False);ax.grid(alpha=.15)
    fig.text(.01,.005,'128 exogenous windows, 93 episodes, 4 futures. Bands: episode bootstrap 95% intervals.\n'
             'The frozen reader is checked on real percepts; generated-domain semantic accuracy is not established.',fontsize=8)
    fig.tight_layout(rect=(0,.075,1,1));fig.savefig(HERE/'recursion_counts.png',dpi=170);plt.close(fig)
    print('saved analysis.json and recursion_counts.png')

if __name__=='__main__':main()
