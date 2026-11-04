"""Case/episode-paired analysis of the predeclared semantic test. CPU only."""
from pathlib import Path
import hashlib,json
import numpy as np
import torch
from saccade.eval.s1 import MLP
HERE=Path(__file__).resolve().parent
BOOT={}

def interval(x,episodes):
    x=np.asarray(x,float);episodes=np.asarray(episodes);u=np.unique(episodes)
    if not len(x):return None
    key=tuple(episodes.tolist())
    if key not in BOOT:
        rng=np.random.default_rng(614600);ids=rng.integers(len(u),size=(4000,len(u)))
        w=(ids[:,:,None]==np.arange(len(u))).sum(1)
        n=np.array([(episodes==e).sum() for e in u]);BOOT[key]=(w,n)
    w,n=BOOT[key];s=np.array([x[episodes==e].sum() for e in u]);draws=(w@s)/(w@n)
    return dict(mean=float(x.mean()),ci95=np.quantile(draws,[.025,.975]).tolist(),cases=len(x),episodes=len(u))

def joined(draw):
    out={}
    for c in range(4):
        a=[b for b in draw['batches'] if b['condition']==c]
        out[c]={k:torch.cat([z[k] for z in a]).numpy() for k in a[0] if k not in ('condition','batch')}
    return out

def main():
    torch.set_num_threads(2)
    raw=torch.load(HERE/'native_results.pt',weights_only=False,map_location='cpu');rows=raw['results'];stages=raw['stages']
    assert len(rows)==67,(len(rows),'Incomplete run; preserve partial outputs, do not call complete.')
    summary={};detail=[]
    for i,row in enumerate(rows):
        r=row['meta'];cohort='heading_dark' if row['kind']=='heading' and r['light']<.25 else 'heading_day' if row['kind']=='heading' else r['cohort']+('_coarse_missed' if r['coarse_missed'] else '_coarse_clear')
        control={k:v.numpy() for k,v in row['control'].items()}
        truth_health=np.repeat([r['next'][0]['health'],r['next'][1]['health']],2)
        for draw in row['draws']:
            a=joined(draw)
            val={}
            for c in range(4):
                x=a[c];val[c]={k:float(np.asarray(v).mean()) for k,v in x.items()}
                val[c]['supported_drop']=float((x['drop']&x['supported']).mean())
                val[c]['health_accuracy']=float((x['health']==truth_health[c]).mean())
                val[c]['supported_correct_health']=float(((x['health']==truth_health[c])&x['supported']).mean())
            met=dict(
                health_focus_minus_control_A=val[0]['health']-val[1]['health'],
                health_contrast_focus=val[2]['health']-val[0]['health'],
                health_contrast_control=val[3]['health']-val[1]['health'],
                health_contrast_gaze_gain=(val[2]['health']-val[0]['health'])-(val[3]['health']-val[1]['health']),
                health_correct_gaze_gain=.5*((val[0]['health_accuracy']-val[1]['health_accuracy'])+(val[2]['health_accuracy']-val[3]['health_accuracy'])),
                health_supported_correct_gaze_gain=.5*((val[0]['supported_correct_health']-val[1]['supported_correct_health'])+(val[2]['supported_correct_health']-val[3]['supported_correct_health'])),
                arrow_focus_A=val[0]['arrow'],arrow_control_A=val[1]['arrow'],
                arrow_focus_B=val[2]['arrow'],arrow_control_B=val[3]['arrow'],
                arrow_gaze_gain_A=val[0]['arrow']-val[1]['arrow'],
                arrow_causal_separation_focus=val[0]['arrow']-val[2]['arrow'],
                arrow_causal_separation_control=val[1]['arrow']-val[3]['arrow'],
                arrow_causal_separation_gaze_gain=(val[0]['arrow']-val[2]['arrow'])-(val[1]['arrow']-val[3]['arrow']),
                arrow_score_gaze_gain_A=val[0]['arrow_score']-val[1]['arrow_score'],
                negative_focus_A=val[0]['negative'],negative_control_A=val[1]['negative'],
                negative_focus_B=val[2]['negative'],negative_control_B=val[3]['negative'],
                negative_causal_separation_gaze_gain=(val[0]['negative']-val[2]['negative'])-(val[1]['negative']-val[3]['negative']),
                health_support_focus_A=val[0]['supported'],health_support_control_A=val[1]['supported'],
                real_health_accuracy=float((control['health']==truth_health).mean()),
                real_health_support=float(control['supported'].mean()),real_arrow_A=float(control['arrow'][0]),
                real_arrow_B=float(control['arrow'][2]),real_negative_A=float(control['negative'][0]),real_negative_B=float(control['negative'][2]))
            detail.append(dict(case=i,cohort=cohort,episode=r['e'],steps=draw['steps'],first_visible=r.get('first_visible',True),
                conditions=val,metrics=met))
    for cohort in sorted(set(r['cohort'] for r in detail)):
        summary[cohort]={}
        for steps in [3,8]:
            a=[r for r in detail if r['cohort']==cohort and r['steps']==steps]
            summary[cohort][str(steps)]={k:interval([r['metrics'][k] for r in a],[r['episode'] for r in a]) for k in a[0]['metrics']}
            summary[cohort][str(steps)]['conditions']={str(c):{k:float(np.mean([r['conditions'][c][k] for r in a])) for k in a[0]['conditions'][c]} for c in range(4)}
    # Preserve first-visible results separately to expose prior-information redundancy.
    first=[]
    for steps in [3,8]:
        a=[r for r in detail if r['cohort']=='distant_arrows_coarse_missed' and r['steps']==steps and r['first_visible']]
        first.append(dict(steps=steps,metrics={k:interval([r['metrics'][k] for r in a],[r['episode'] for r in a]) for k in a[0]['metrics']}))
    field=lambda k:[s[k]['rms'] for s in stages]
    stage=dict(cases=len(stages),max_prefix_rms=max(s['prefix_rms'] for s in stages),
        fine_only_embedding_collisions=sum(s['mixed_fine_only']['equal'] for s in stages),
        fine_only_predicted_collisions=sum(s['fine_only_predicted']['equal'] for s in stages),
        fine_only_full_ssm_collisions=sum(all(c['equal'] for c in s['fine_only_carry_ssm']) for s in stages),
        fine_only_full_conv_collisions=sum(all(c['equal'] for c in s['fine_only_carry_conv']) for s in stages),
        median_fine_embedding_rms=float(np.median(field('mixed_fine_only'))),
        median_fine_predicted_rms=float(np.median(field('fine_only_predicted'))),
        physical_cache_parity=all(all(k['physical_keys']['equal'] and k['physical_values']['equal'] for k in s['cache_key_deltas']) for s in stages),
        decoder_heading_correct=int(sum(sum(s['fine_pixel_semantic_correct']) for s in stages[:24])),
        decoder_heading_trials=48)
    aliases=[];source_checks=[]
    saved=torch.load(HERE.parents[0]/'20261103_p1f16_review/p1f16/readers.pt',weights_only=False,map_location='cpu')
    readers=[]
    for seed in [0,1]:
        rd=saved[f'g_{seed}'];net=MLP(rd['dim'],{'kind':5});net.load_state_dict(rd['state']);net.eval();readers.append(net)
    for i in range(len(rows)):
        a=torch.load(HERE/'activations'/f'case_{i:03d}.pt',weights_only=False,map_location='cpu')
        aliases.append(bool(torch.equal(a['p'][0],a['p'][2]) and torch.equal(a['v'][0],a['v'][2])))
        meta=rows[i]['meta'];cell=meta.get('cell',(meta.get('delta',[0,0])[0]+3)*9+meta.get('delta',[0,0])[1]+4)
        fr,fc=a['fix'][0].tolist();rr,cc=divmod(cell,9);slot=(rr-fr+1)*3+cc-fc+1
        feats=torch.cat([a['g'][[0,2],slot].float(),a['v'][[0,2]].float()],-1)
        with torch.no_grad():pred=[rd(feats)['kind'].argmax(-1).tolist() for rd in readers]
        ky=4 if meta.get('kind','arrows')=='arrows' else 1
        source_checks.append(dict(case=i,pred=pred,specific_to_presence=all(p[0]==ky and p[1]!=ky for p in pred)))
    stage['exact_coarse_and_light_aliases']=sum(aliases)
    pins=json.loads((HERE/'SOURCE_PINS.json').read_text());bad=[]
    for p,sha in pins.items():
        h=hashlib.sha256(Path(p).read_bytes()).hexdigest()
        if sha!=h:bad.append(p)
    assert not bad,('Pinned source changed',bad)
    out=dict(summary=summary,first_visible_distant_missed=first,stage=stage,detail=detail,source_checks=source_checks,sources_unchanged=True,
        non_claim='Reader accuracy is not a proof of complete information absence; counterfactual removals/headings do not necessarily share a feasible episode past.')
    (HERE/'RESULTS.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
    print(json.dumps(stage,indent=2))
    for k in summary:
        for n in ['3','8']:
            z=summary[k][n];metric='arrow_gaze_gain_A' if k.startswith('distant') else 'health_contrast_gaze_gain'
            print(k,n,metric,z[metric], 'focus_arrow_A',z['arrow_focus_A']['mean'],'focus_negative_A',z['negative_focus_A']['mean'])

if __name__=='__main__':main()
