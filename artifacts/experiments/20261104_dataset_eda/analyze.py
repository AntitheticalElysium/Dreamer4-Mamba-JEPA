"""Tables and scientific figures from the reconciled, final TRAIN census."""
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/saccade_dataset_matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
BUDGET = 25_600_000
NAMES = {'old': 'Original', 'advanced': 'Advanced addition', 'danger': 'Danger addition', 'combined': 'Full candidate union'}
COLORS = {'old': '#556879', 'advanced': '#208b84', 'danger': '#c76452', 'combined': '#7754a0'}
ACTIONS = ['NOOP','LEFT','RIGHT','UP','DOWN','DO','SLEEP','PLACE_STONE','PLACE_TABLE','PLACE_FURNACE',
           'PLACE_PLANT','MAKE_WOOD_PICKAXE','MAKE_STONE_PICKAXE','MAKE_IRON_PICKAXE',
           'MAKE_WOOD_SWORD','MAKE_STONE_SWORD','MAKE_IRON_SWORD']


def main():
    d = json.loads((HERE/'CENSUS.json').read_text())
    assert d['complete'] and d['snapshot_rounds'] == {'advanced': 128, 'danger': 512}
    ep = json.loads(gzip.open(HERE/'episodes.json.gz', 'rt').read())
    pools = {**d['pools'], 'combined': d['combined']}
    def n(pool, key): return pools[pool].get(key, {}).get('n', 0)
    def support(pool, key): return pools[pool].get(key, {}).get('episodes', 0)
    def rate(pool, key): return n(pool, key)/n(pool, 'total/frames')
    def fold(key): return rate('combined', key)/rate('old', key) if n('old', key) else None
    def fmt(x): return f'{x:,}' if isinstance(x, (int, np.integer)) else f'{x:,.2f}'
    def table(headers, rows):
        return ['|'+'|'.join(headers)+'|', '|'+'|'.join(['---']*len(headers))+'|',
                *['|'+'|'.join(str(x) for x in row)+'|' for row in rows]]
    def csv_out(name, headers, rows):
        with open(HERE/(name+'.csv'), 'w', newline='') as f:
            w=csv.writer(f);w.writerow(headers);w.writerows(rows)
    def title(key): return key.split('/')[-1].replace('_', ' ').lower()
    ach = [k for k in pools['old'] if k.startswith('ach/')]
    inventory = ['wood','stone','coal','iron','diamond','sapling','wood_pickaxe','stone_pickaxe',
                 'iron_pickaxe','wood_sword','stone_sword','iron_sword']
    body = ['health','food','drink','energy']
    selected = [('Plant first unlock','ach/EAT_PLANT'), ('Diamond first unlock','ach/COLLECT_DIAMOND'),
                ('Coal first unlock','ach/COLLECT_COAL'), ('Iron first unlock','ach/COLLECT_IRON'),
                ('Coal increments','inc/coal'), ('Iron increments','inc/iron'),
                ('Drink increases','body_inc/drink'), ('Damage (net health loss)','event/damage'),
                ('Deaths','event/death'), ('Low-health zombie risk','event/risk'),
                ('Deaths in that risk class','event/risk_death')]
    allkeys = sorted(set().union(*(set(p) for p in pools.values())))
    raw_rows=[]
    for key in allkeys:
        oldrate=rate('old',key)
        for pool in pools:
            raw_rows.append([key,pool,n(pool,key),support(pool,key),rate(pool,key)*1e6,
                             rate(pool,key)*BUDGET,rate(pool,key)/oldrate if oldrate else ''])
    csv_out('all_metrics',['metric','pool','count','episodes','per_million_emitted_frames',
                           'expected_occurrences_in_25_6M_frames','rate_fold_vs_original'],raw_rows)
    # Every frame is emitted once per corpus pass, including one reset frame per episode.
    # Queue shuffling does not make the long episodes frame-balanced with short episodes.
    summary={'complete':True,'frame_budget':BUDGET,'exposure_is_approximate':True,
             'denominator':'transitions + episodes, matching the emitted-frame budget in saccade.train',
             'pools':{},'selected':{},'source_pins':d['source_pins'],
             'census_sha256':hashlib.sha256((HERE/'CENSUS.json').read_bytes()).hexdigest()}
    for pool in pools:
        lengths=np.array([e['T'] for e in ep if pool=='combined' or e['group'].startswith(pool+'_')])
        summary['pools'][pool]={
            'episodes':n(pool,'total/episodes'),'transitions':n(pool,'total/transitions'),
            'frames':n(pool,'total/frames'),'passes_at_budget':BUDGET/n(pool,'total/frames'),
            'length_quantiles':dict(zip(['min','p10','p50','p90','p99','max'],np.percentile(lengths,[0,10,50,90,99,100]).tolist())),
            'episodes_reaching_1024':int((lengths>=1024).sum()),
            'truncated':n(pool,'total/truncated')}
    for label,key in selected:
        summary['selected'][key]={'label':label,'unique_growth':n('combined',key)/n('old',key),
                                 'rate_fold':fold(key),'original':n('old',key),'combined':n('combined',key),
                                 'expected_original':rate('old',key)*BUDGET,
                                 'expected_combined':rate('combined',key)*BUDGET}
    old_manifest=json.loads((HERE.parents[1]/'craftax_support_v2/manifest.json').read_text())
    new_manifests=[json.loads((HERE/f).read_text()) for f in ['advanced_manifest_snapshot.json','danger_manifest_snapshot.json']]
    summary['archive']={
        'all_episodes':old_manifest['episodes']+sum(m['episodes'] for m in new_manifests),
        'all_transitions':old_manifest['transitions']+sum(m['transitions'] for m in new_manifests),
        'original_dev_episodes':old_manifest['split_episode_counts']['dev'],
        'original_final_episodes':old_manifest['split_episode_counts']['final'],
        'new_monitor_episodes':sum(m['coverage']['monitor']['episodes'] for m in new_manifests),
        'new_monitor_transitions':sum(m['coverage']['monitor']['transitions'] for m in new_manifests)}
    (HERE/'SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n')

    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                         'savefig.dpi':170,'font.family':'DejaVu Sans'})
    def growth_plot(items, filename, heading, height):
        labels=[x[0] for x in items];keys=[x[1] for x in items];y=np.arange(len(keys))
        fig,(ax,bx)=plt.subplots(1,2,figsize=(13,height),gridspec_kw={'width_ratios':[1.75,1]},sharey=True)
        a=np.array([n('old',k) for k in keys]);b=np.array([n('combined',k) for k in keys])
        ax.barh(y-.18,a,.34,color=COLORS['old'],label='Original training data')
        ax.barh(y+.18,b,.34,color=COLORS['combined'],label='Full candidate union')
        ax.set_yticks(y,labels);ax.invert_yaxis();ax.set_xscale('log');ax.set_xlim(min(a)*.6,max(b)*2)
        for i,x in enumerate(b):ax.text(x*1.03,i+.18,f'{x:,}',va='center',fontsize=9)
        ax.set_xlabel('Distinct recorded event occurrences (log scale)');ax.legend(loc='lower right',fontsize=9)
        ratios=np.array([fold(k) for k in keys]);bx.axvline(1,color='#555',ls='--',lw=1)
        bx.barh(y,ratios,color=['#208b84' if r>=1 else '#c76452' for r in ratios],height=.6)
        bx.set_xlim(0,max(1.2,ratios.max()+.28));bx.set_xlabel('Expected exposure at fixed training budget\n(relative to original; dashed = unchanged)')
        for i,r in enumerate(ratios):bx.text(r+.025,i,f'{r:.2f}×',va='center',fontsize=9)
        bx.tick_params(axis='y',left=False);ax.grid(axis='x',alpha=.15);bx.grid(axis='x',alpha=.15)
        fig.suptitle(heading,fontweight='bold',y=.988)
        fig.text(.01,.012,'TRAIN only; fixed monitor excluded. Exposure = count / (transitions + episodes) × 25.6M. More distinct examples can accompany fewer updates per event.',fontsize=8)
        fig.tight_layout(rect=(0,.045,1,.96));fig.savefig(HERE/filename);fig.savefig(HERE/filename.replace('.png','.pdf'));plt.close(fig)
    growth_plot(selected,'coverage_vs_exposure.png','Coverage grew; optimization exposure did not grow uniformly',7.2)
    growth_plot([(title(k),k) for k in ach],'achievement_coverage.png','All 22 achievements: independent unlocks and fixed-budget exposure',11.6)

    bands=['1_7','8_63','64_255','256_1023','1024_plus']
    band_labels=['1–7','8–63','64–255','256–1023','1024+']
    fig,axes=plt.subplots(1,3,figsize=(14,4.9))
    names=list(pools);y=np.arange(len(names))
    axes[0].barh(y,[n(p,'total/transitions')/1e6 for p in names],color=[COLORS[p] for p in names])
    axes[0].set_yticks(y,[NAMES[p] for p in names]);axes[0].invert_yaxis();axes[0].set_xlabel('Million transitions');axes[0].set_title('Actual training sources')
    for i,p in enumerate(names):axes[0].text(n(p,'total/transitions')/1e6+.07,i,f"{n(p,'total/episodes'):,} eps",va='center',fontsize=9)
    axes[0].set_xlim(0,11.5)
    left=np.zeros(4)
    for j,k in enumerate(bands):
        width=np.array([n(p,'age/'+k)/n(p,'total/transitions')*100 for p in names])
        axes[1].barh(y,width,left=left,label=band_labels[j],color=plt.cm.viridis((j+.3)/5.3));left+=width
    axes[1].invert_yaxis();axes[1].set_yticks(y,[NAMES[p] for p in names]);axes[1].set_xlabel('% of transitions');axes[1].set_title('History depth when a target occurs');axes[1].legend(title='Episode age (steps)',fontsize=8,loc='upper center',bbox_to_anchor=(.5,-.16),ncol=3)
    kinds=['zombies','cows','skeletons','arrows'];y=np.arange(4)
    for shift,p in [(-.18,'old'),(.18,'combined')]:
        vals=[n(p,'entity/'+k+'/night_instances')/n(p,'total/transitions')*1000 for k in kinds]
        axes[2].barh(y+shift,vals,height=.34,color=COLORS[p],label=NAMES[p])
    axes[2].set_yticks(y,kinds);axes[2].invert_yaxis();axes[2].set_xlabel('Night object-frame instances\nper 1,000 transitions');axes[2].set_title('Night entity density');axes[2].legend(fontsize=8)
    fig.suptitle('Long histories and advanced states expanded; night creatures remain plentiful',fontweight='bold')
    fig.text(.01,.012,'Entity visibility here is simulator geometry, not pixel detectability. Source policies differ; these are empirical composition statistics.',fontsize=8)
    fig.tight_layout(rect=(0,.07,1,.92));fig.savefig(HERE/'composition.png');fig.savefig(HERE/'composition.pdf');plt.close(fig)

    lines=['# Full training-dataset census — 2026-11-04','',
           'Final, complete census of the original store plus both authorized collections. No model fit, training run, sampler or architecture change. All counts below exclude the stable 5% monitor; original dev/final remain untouched. The full union is an accounting candidate, not an adopted training mixture.','',
           f"The entire three-store archive has {summary['archive']['all_episodes']:,} episodes and {summary['archive']['all_transitions']:,} transitions. The eligible training pool has 22,267 episodes/8,914,902 transitions; held-out monitor episodes are excluded. Original dev (1,009 episodes) and final (1,002) remain separate, unchanged and unused in this audit.",'',
           '**Main finding:** independent coverage is much broader, especially long histories, plant/diamond skills and diverse danger. At the fixed 25.6M-frame budget, raw concatenation reduces repetition from '+f"{summary['pools']['old']['passes_at_budget']:.2f} to {summary['pools']['combined']['passes_at_budget']:.2f} passes. Several first unlocks and danger events become less frequent per emitted frame, even after the separate danger collection.",'',
           '![Coverage and exposure](coverage_vs_exposure.png)','', '## Source composition','']
    lines+=table(['Source','Train episodes','Transitions','Deaths','Truncated at cap','Median / p90 / max length','Frames after step 1023'],[
        [NAMES[p],fmt(n(p,'total/episodes')),fmt(n(p,'total/transitions')),fmt(n(p,'event/death')),fmt(n(p,'total/truncated')),
         ' / '.join(fmt(summary['pools'][p]['length_quantiles'][k]) for k in ['p50','p90','max']),
         f"{n(p,'age/1024_plus'):,} ({100*n(p,'age/1024_plus')/n(p,'total/transitions'):.1f}%)"] for p in pools])
    lines+=['','Long episodes are complete natural histories, not cutouts or repeated prefixes. In the advanced addition, '+f"{summary['pools']['advanced']['episodes_reaching_1024']:,} episodes reach step 1,024 and {n('advanced','total/truncated'):,} survive to the 2,500-step cap. A cap is truncation, not death.",'',
            'Original/new reset keys were checked disjoint across all source splits. New sources use the same frozen symbolic expert and environment, with fixed exploration schedules, no success/death acceptance filtering, and full actual behavior probabilities/random-action flags. These diagnostic state labels never become world-model inputs.','',
            'Advanced policies: competent epsilon 0/.02/.05/.10, interleaved with fixed 16-action uniform bursts after 512/1,024-step competent prefixes. Danger policies: reset exploration at epsilon .25/.5/1, interleaved with a fixed 512/1,024-step competent prefix then uniform actions until termination/cap. No state was hand edited.','',
            'Whether scheduled late exploration was actually reached (survival is not assumed from the assigned policy name):','']
    reach=[]
    for group in d['groups']:
        if 'late_failure' not in group and 'bursts' not in group:continue
        prefix=int(group.rsplit('_',1)[1]);r=[e for e in ep if e['group']==group];z=[e for e in r if e['T']>prefix]
        reach.append([group,len(r),len(z),fmt(sum(e['T']-prefix for e in z)),sum(e['death'] for e in z)])
    lines+=table(['Assigned policy','Train episodes','Reached exploration switch','Actual transitions after prefix','Deaths after switch'],reach)
    lines+=['','For burst policies, post-prefix transitions include both bursts and base-policy intervals; for late-failure policies every post-prefix action is uniform. The danger addition includes 1,016 genuinely reached late-failure trajectories (90,432 post-prefix random actions), rather than merely 1,278 assigned trajectories.','',
            'New held-out monitor support (excluded throughout this training census):','']
    monitors=[]
    for name,filename in [('Advanced addition','advanced_manifest_snapshot.json'),('Danger addition','danger_manifest_snapshot.json')]:
        m=json.loads((HERE/filename).read_text())['coverage']['monitor']
        monitors.append([name,fmt(m['episodes']),fmt(m['transitions']),fmt(m['deaths']),fmt(m['risk_deaths']),
                         *[fmt(m['unlocks'][k]) for k in ['EAT_PLANT','COLLECT_DIAMOND','COLLECT_COAL','COLLECT_IRON']]])
    lines+=table(['Held-out source','Episodes','Transitions','Deaths','Risk deaths','Plant','Diamond','Coal','Iron'],monitors)
    lines+=['','Keep original evaluation/monitor results separately visible. New held-out data can check generalization on the acquired distribution, but its rare-class support remains small and silently changing an aggregate monitor would change checkpoint selection. This audit does not make that selection change.','',
            '## All 22 first unlocks','',
            'Each unlock occurs at most once per episode, so these counts are also independent episode support. First unlocks are distinct from repeated counter increases. Rates use emitted frames (T+1 per episode), matching the P1 budget.','']
    rows=[]
    for key in ach:
        rows.append([title(key),*[fmt(n(p,key)) for p in pools],f'{n("combined",key)/n("old",key):.2f}×',f'{fold(key):.2f}×'])
    lines+=table(['Unlock','Original','Advanced','Danger','Union','Unique growth','Fixed-budget exposure ratio'],rows)
    lines+=['','First unlocks by episode age (union):','']
    lines+=table(['Unlock',*band_labels],[[title(k),*[fmt(n('combined','ach_age/'+k.split('/')[1]+'/'+b)) for b in bands]]
         for k in ['ach/EAT_PLANT','ach/COLLECT_DIAMOND','ach/COLLECT_COAL','ach/COLLECT_IRON','ach/MAKE_IRON_PICKAXE','ach/MAKE_IRON_SWORD']])
    lines+=['','This separates late-life coverage from merely collecting more early events: 435 of 1,081 plant unlocks occur at age≥1,024; only 6 of 9,281 coal and 21 of 6,804 iron first unlocks do. Longer episodes are especially relevant to plant maturation, later survival, repeated mining and inventory states; missing long histories alone is not an explanation for first coal/iron failure.','',
            '![All achievements](achievement_coverage.png)','','## Repeated consequences and numeric value coverage','']
    lines+=table(['Consequence','Original events / episodes','Union events / episodes','Expected old / union per 25.6M frames','Exposure ratio'],[
        [label,f'{n("old",key):,} / {support("old",key):,}',f'{n("combined",key):,} / {support("combined",key):,}',
         f'{BUDGET*rate("old",key):,.0f} / {BUDGET*rate("combined",key):,.0f}',f'{fold(key):.2f}×']
        for label,key in [('Wood','inc/wood'),('Stone','inc/stone'),('Coal','inc/coal'),('Iron','inc/iron'),('Diamond','inc/diamond'),('Sapling','inc/sapling'),('Drink','body_inc/drink'),('Health recovery','body_inc/health')]])
    lines+=['','Prior numeric value for an increment (0 means 0→1; decreases/consumption can allow repeated 0→1 without another unlock):','']
    values=[]
    for name in ['wood','stone','coal','iron','diamond','sapling']:
        for p in ['old','combined']:values.append([name,NAMES[p],*[fmt(n(p,f'inc_value/{name}/{i}')) for i in range(9)]])
    lines+=table(['Counter','Source',*[f'{i}→{i+1}' for i in range(9)]],values)
    lines+=['','**Remaining holes:** diamond increments above 2→3 are absent; high-valued iron and sapling changes still have tens or single examples. Even wood 8→9 has only 479 examples. More source frames cannot repair a state/action combination that never occurred. The main rare-event gains are not uniformly spread across all values.','',
            '## Damage, death and survival controls','',
            'Damage below means net health loss in a transition, not a simulator-attributed attack count. Risk means health≤2 and an active zombie at Manhattan distance 1; it includes awake and sleeping states. Contexts overlap; empty hunger/thirst or a visible arrow is not proof of the cause of death.','']
    lines+=table(['Event','Original','Advanced','Danger','Union','Old / union per million frames','Exposure ratio'],[
        [title(key),*[fmt(n(p,key)) for p in pools],f'{1e6*rate("old",key):.1f} / {1e6*rate("combined",key):.1f}',f'{fold(key):.2f}×']
        for key in ['event/damage','event/death','event/risk','event/risk_death','event/risk_survival']])
    conds=['day','night','health1','health2','sleeping','awake','food_empty','drink_empty','arrow_visible','skeleton_visible','iron_equipment',
           'risk_day','risk_night','risk_ready','risk_unready','risk_sleeping','risk_awake']
    crows=[]
    for c in conds:
        f0=n('old','condition/'+c+'/frames');f1=n('combined','condition/'+c+'/frames')
        t0=n('old','condition/'+c+'/death');t1=n('combined','condition/'+c+'/death')
        crows.append([c.replace('_',' '),fmt(f0),fmt(f1),fmt(t0),fmt(t1),f'{100*t0/f0:.3f}% / {100*t1/f1:.3f}%'])
    lines+=['']+table(['Prior context','Old transitions','Union transitions','Old deaths','Union deaths','Empirical death probability old / union'],crows)
    lines+=['','These probabilities are unadjusted behavior-policy mixtures. They do not estimate a causal effect of equipment, sleeping, action or night. Readiness is the prior-state cooldown of an already-adjacent zombie; attack/update order and movement can still alter the outcome.','']
    lines+=table(['Death state','Original','Union'],[[k,fmt(n('old','death_context/'+k)),fmt(n('combined','death_context/'+k))]
         for k in ['lava','health_zero','lava_and_health_zero','nonlava_health_zero','other']])
    lines+=['','The simulator sets health to zero in lava; lava and health-zero counts overlap. Nonlava health-zero is not automatically a zombie death. No attack/starvation causal labels are inferred from overlapping contexts.','',
            'Deaths by episode age:','']
    lines+=table(['Age band','Original deaths','Union deaths','Original risk deaths','Union risk deaths'],[
        [b,fmt(n('old','event_age/death/'+b)),fmt(n('combined','event_age/death/'+b)),fmt(n('old','event_age/risk_death/'+b)),fmt(n('combined','event_age/risk_death/'+b))] for b in bands])
    lines+=['','## Action support at usable resources','',
            'Opportunity = awake, faced matching block, counter<9 and the exact required pickaxe held (wood for stone/coal, stone for iron, iron for diamond). NOOP/DO are drawn from different visited states, so these counts alone are not a matched causal comparison. Actual uniform-action-branch successes add stronger local action support.','']
    rr=[]
    for name in ['drink','wood','sapling','stone','coal','iron','diamond']:
        for p in ['old','combined']:
            rr.append([name,NAMES[p],*[fmt(n(p,'resource/'+name+'/'+k)) for k in ['DO','DO_increment','NOOP','NOOP_increment','DO_no_tool','DO_random_increment']]])
    lines+=table(['Resource','Source','Qualified DO','DO increments','Qualified NOOP','NOOP increments','DO without tool','Random-branch DO increments'],rr)
    lines+=['','In the saved support, qualified DO always increments drink/wood/stone/coal/iron/diamond; sapling collection is stochastic (17,986 increases from 300,460 qualified DOs). Negative controls exist (NOOP/no tool), but diamond still has only '+f"{n('combined','resource/diamond/NOOP'):,} qualified NOOP opportunities and {n('combined','resource/diamond/DO_random_increment'):,} randomized successful DO transitions. Rare skills are still expert-action dominated.",'',
            '## Scene and action composition','', '![Composition](composition.png)','',
            'Creature counts below are nominal simulator-visible instances (active and inside the camera rectangle), not distinct tracked identities and not a guarantee the night pixels reveal them. Background terrain counts include cells occluded by objects.','']
    er=[]
    for kind in ['zombies','cows','skeletons','arrows']:
        for p in ['old','combined']:
            er.append([kind,NAMES[p],fmt(n(p,'entity/'+kind+'/instances')),fmt(n(p,'entity/'+kind+'/night_instances')),
                       fmt(n(p,'entity/'+kind+'/near2_frames')),f"{1e3*n(p,'entity/'+kind+'/night_instances')/n(p,'total/transitions'):.1f}"])
    lines+=table(['Entity','Source','Object-frame instances','Night instances','Frames with entity within two cells','Night instances / 1k transitions'],er)
    terrainkeys=[k for k in pools['combined'] if k.startswith('terrain/')]
    lines+=['']+table(['Background cell','Original view-cell count','Union view-cell count','Density ratio'],[
        [title(k),fmt(n('old',k)),fmt(n('combined',k)),f'{fold(k):.2f}×' if n('old',k) else 'new'] for k in terrainkeys])
    editkeys=sorted([k for k in pools['combined'] if k.startswith('edit/')],key=lambda k:n('combined',k),reverse=True)
    lines+=['','Camera-aligned terrain edits (same world location; entering borders excluded):','']
    lines+=table(['Edit','Original cells / episodes','Union cells / episodes'],[
        [title(k),f'{n("old",k):,} / {support("old",k):,}',f'{n("combined",k):,} / {support("combined",k):,}'] for k in editkeys])
    lines+=['','All 17 action frequencies:','']
    lines+=table(['Action','Original','Union','Original / union % of transitions'],[
        [name,fmt(n('old',f'action/{i}')),fmt(n('combined',f'action/{i}')),
         f'{100*n("old",f"action/{i}")/n("old","total/transitions"):.2f}% / {100*n("combined",f"action/{i}")/n("combined","total/transitions"):.2f}%'] for i,name in enumerate(ACTIONS)])

    lines+=['','## P2: actual exogenous sequences, with terminal suffixes','',
            'A real window qualifies only when every observed action in it came from the actual uniform-random branch. After a true terminal, the remaining horizon may be absorbing. A surviving cap supplies no imaginary future. Windows overlap: event/episode support below is deduplicated. This audits support; it does not adopt a P2 objective or train on labels.','']
    lines+=table(['Horizon','Original windows','Union windows','Union full real horizon','Union absorbing-tail windows'],[
        [H,fmt(n('old',f'exog/{H}/windows')),fmt(n('combined',f'exog/{H}/windows')),
         fmt(n('combined',f'exog/{H}/full_length_windows')),fmt(n('combined',f'exog/{H}/absorbing_suffix_windows'))] for H in [2,4,8,16]])
    pk=[('damage','unique_event/damage'),('deaths','unique_event/death'),('risk deaths','unique_event/risk_death'),
        ('coal first unlock','unique_ach/COLLECT_COAL'),('iron first unlock','unique_ach/COLLECT_IRON'),('diamond first unlock','unique_ach/COLLECT_DIAMOND'),
        ('coal increments','unique_inc/coal'),('iron increments','unique_inc/iron'),('diamond increments','unique_inc/diamond'),('drink increases','unique_inc/drink')]
    lines+=['']+table(['Event inside qualifying H16 support','Original events / episodes','Union events / episodes'],[
        [name,f'{n("old","exog/16/"+k):,} / {support("old","exog/16/"+k):,}',
         f'{n("combined","exog/16/"+k):,} / {support("combined","exog/16/"+k):,}'] for name,k in pk])
    lines+=['','Many terminal events are covered by a short actual random suffix plus absorption, not 16 live steps before death. This avoids survival censoring but must not be sold as 16-step live trajectory diversity. Long advanced-skill P2 windows remain a separate coverage issue.','',
            '## Policy-component accounting','',
            'Policies were fixed before their outcomes. This table exposes which collection generates each support class; no new event-driven acceptance or training weight is chosen here.','']
    grows=[]
    for group,q in d['groups'].items():
        get=lambda k:q.get(k,{}).get('n',0)
        grows.append([group,fmt(get('total/episodes')),fmt(get('total/transitions')),
                      *[fmt(get(k)) for k in ['ach/EAT_PLANT','ach/COLLECT_DIAMOND','ach/COLLECT_COAL','ach/COLLECT_IRON','event/death','event/risk_death']]])
    lines+=table(['Policy','Episodes','Transitions','Plant unlock','Diamond unlock','Coal unlock','Iron unlock','Deaths','Risk deaths'],grows)
    lines+=['','## What tonight can and cannot test','',
            '- Better-supported learning hypotheses: plant eating/diamond and advanced-tool events, late-state inventory changes, more distinct death/damage contexts, and later-state exploratory actions. Greater diversity can reduce memorization even where frequency falls.','- Raw-union caveat: at unchanged 25.6M frames, common first unlocks, coal first unlock, damage and deaths do not all receive more updates. An unchanged result on these would not isolate an architecture failure or refute a coverage hypothesis; unique diversity and exposure changed together.','- High-value iron/diamond counters, action support for very rare skills, and advanced-skill exogenous H16 windows remain thin or absent. Data collected so far does not close these holes.','- Extra night-entity examples create learning opportunity, not proof the fovea supplies incremental information over coarse vision/history or that its information reaches generation. Native fine uptake, sampler quality and P2 self-feeding are separate questions.','- Recommended scope for a later approved data run: same F2 architecture/learned chooser, 3×3 coarse input, losses and evaluations. A concurrent 2×2 change would alter sensing and obscure what the data changed. No run is launched by this audit.','- Final mixture and compute exposure must be declared together. Simply taking every source once is a defensible diversity experiment, but is not an experiment that increases optimization pressure on every rare failure. No event balancing or longer budget has been silently adopted.','',
            'Expected readings, to keep the run interpretable:','',
            '| Learning hypothesis | Evidence supporting it | Required reading | What a null would not prove |',
            '|---|---|---|---|',
            '| Better plant/diamond prediction | >5× independent unlock episodes; 1.60×/1.45× fixed-budget exposure | First unlocks and repeated increments, event recall AND false positives, on fixed old evaluation and separately the fresh monitor | Not proof that foveal integration works/fails |',
            '| Better advanced counter behavior | Iron increments 5,981→23,657; high values filled unevenly | Per-prior-value increments in generated periphery AND generated glimpse; real-target heads as a separate control | Not proof that all high values are supported |',
            '| Better danger generalization | 21,579 deaths and 11,097 risk deaths; true late-failure trajectories | Risk-conditional sampled damage/death, by health, day/night, sleep and cooldown; false death on surviving controls | At fixed frames, not a clean refutation of extra optimization pressure: death exposure is lower |',
            '| Better late-history handling | 1.69M transitions at age≥1,024, versus 0.25M | Fixed evaluation stratified by prefix length; action-conditioned outcomes with deep real prefixes | Not proof of P2 self-feeding readiness or exact spatial/entity memory |',
            '| Better coal/iron prediction | More source episodes and repeated increments, but first-unlock exposure falls | Keep first unlock distinct from later increments; compare same histories and gaze | A miss alone cannot identify data diversity versus optimization, memory or generator cause |','',
            '## Validation and provenance','',
            'All 22 original unlock counts reconcile to the previous train census; all new unlocks, transitions, damage, deaths and risk counts reconcile to the final committed manifests. New audit-sidecar hashes were checked against each manifest. All 22,267 actual-train ids are unique. Synthetic checks cover camera-aligned edits and terminal/truncation handling. Collection processes both exited successfully.','',
            'The collection QA includes native-GPU legacy parity and full native-GPU replay checks. The independent post-prefix CPU replay recovered all saved rewards/termination/random-action flags, but cross-backend pixels differed by at most one intensity value in 530 of 13,931,190 channels; that original exact-pixel failure remains recorded. Stored GPU observations and sidecars are authoritative. See `../20261104_rare_collection/RANDOM_TAIL_CHECK.json`; do not call cross-backend replay byte-identical.','',
            'Reproduction: `.venv/bin/python -m unittest artifacts.experiments.20261104_dataset_eda.test_census -v`; '
            '`.venv/bin/python -m artifacts.experiments.20261104_dataset_eda.census`; '
            '`.venv/bin/python -m artifacts.experiments.20261104_dataset_eda.analyze`.','',
            'All individual metrics, independent-episode support, densities and expected fixed-budget exposures are in `all_metrics.csv`; raw complete census in `CENSUS.json`, compact summary in `SUMMARY.json`, per-episode diagnostics in `episodes.json.gz`, and final manifest snapshots beside them. No probe was fitted and no simulator state is fed to the world model.','']
    (HERE/'RESULTS.md').write_text('\n'.join(lines))
    print(json.dumps({'pools':summary['pools'],'selected':summary['selected']},indent=2))


if __name__=='__main__':main()
