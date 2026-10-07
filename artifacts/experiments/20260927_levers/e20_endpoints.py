"""E20 paired endpoints: two health readers, physical cost, and parent memory.

Windows 5/15 reuse hash-verified teval generated/teacher states without inference.
Window 4 needs teacher token63 that the historical damage cache did not save;
its compact pass scores both readers and saves only scalars/alignment per batch.
Paired memory uncertainty uses the preselected sample-0 factual trajectories;
the main queue separately reports check_recall's original all-five-key metric.
All comparisons are reused diagnostic evidence, never actor or promotion gates.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
for p in (HERE, HERE.parent/'20260926_diagnosis', HERE.parent/'20260921_readout_ladder'):
    sys.path.insert(0, str(p))
import e20_train as TRAIN
import e19_eval as EVAL
import check_recall as RECALL
import check_damage_rule as RULE
import check_decision_step as CD
import spatial as SP
import scroll
R, T, OUT = TRAIN.R, CD.T, TRAIN.D.OUT


def inputs():
    meta, fit, seeds = T.split()
    cache = T.build_cache('raw', torch.device('cpu'))
    probes = T.Probes(cache, meta, fit, seeds)
    reader = torch.load(OUT/'health_reader.pt', map_location='cpu', weights_only=False)
    def health(x):
        x = x.float().cpu()
        shape = x.shape[:-2]
        flat = x.reshape(-1,81,192)
        return torch.stack((probes.hud(flat[:, 63:81, :].flatten(-2))[..., 0]*9,
                            TRAIN.read_health(flat[:, 63, :], reader)*9), -1).reshape(*shape,2)
    masks = {k: v[:, 0] for k,v in RULE.masks(meta).items()}
    frames = torch.cat((cache['ctx'], cache['fut']), 1).cpu()
    actions = torch.cat((cache['ctx_a'], cache['fut_a']), 1).cpu()
    truth = torch.cat((meta['root_visible'][:, None, 1512], meta['future_visible'][:, 0, :, 1512]), 1).float()*9
    tensors = {**{'cache_'+k:v for k,v in cache.items() if torch.is_tensor(v)},
               **{'meta_'+k:v for k,v in meta.items() if torch.is_tensor(v)},
               'train_roots':fit, 'train_seeds':seeds}
    return meta, fit, cache, health, masks, frames, actions, truth, tensors


def verified_teval(path, window, expected):
    st = torch.load(path, map_location='cpu', weights_only=False)
    hashes = {k:R.tensor_hash(v) for k,v in expected.items()}
    checkpoints = {str(path):R.file_hash(path), str(SP.CHECKPOINT):R.file_hash(SP.CHECKPOINT)}
    matches = []
    root = TRAIN.D.ROOT/'artifacts/eda/frozen_eval_resume_v1'
    for directory in root.glob(st['name']+f'__teval_w{window}__*'):
        contract = json.loads((directory/'contract.json').read_text())
        if contract['checkpoints'] != checkpoints or contract['inputs'] != hashes:
            continue
        options = contract['options']
        if options['window'] != window or options['hard'] or options['snap'] != 'None':
            continue
        if any(R.file_hash(p) != sha for p,sha in contract['sources'].items()):
            raise RuntimeError(f'Changed cached numerical source: {directory}')
        store = R.Store(directory, contract)
        if store.load('result') is not None:
            matches.append(store)
    if len(matches) != 1:
        raise RuntimeError(f'Need exactly one completed compatible teval cache: {path}, w{window}, got {len(matches)}')
    return matches[0], st


def offsets(frames):
    shifts = torch.cat([scroll.estimate(frames[i:i+16, :-1].float(), frames[i:i+16, 1:].float())
                        for i in range(0, len(frames), 16)])
    return CD.DA.offsets(shifts).long()


def health_report(meta, fit, masks, real, truth, teacher, generated, aligned):
    summaries = {}
    positive = {}
    for j,label in enumerate(('independent_hud', 'factual_loss_reader')):
        cur = real[:, :-1, j]
        selfcur = torch.cat((real[:, :1,j], generated[:, :-1,j]), 1)
        positive[label] = {'mae_health':float((real[...,j]-truth).abs().mean()),
                           'strict_real_control':EVAL.metrics(real[:,1:,j] < cur-1.5, masks, torch.ones(len(meta['seed']),dtype=torch.bool))}
        summaries[label] = {}
        for mode,pred,baseline,extra in (('teacher',teacher[...,j],cur,torch.ones_like(aligned)),
                                       ('selffed',generated[...,j],selfcur,aligned)):
            mode_masks = dict(masks)
            mode_masks['valid'] = masks['valid'] & extra
            summaries[label][mode] = {}
            for threshold in (1.5,.5):
                drawn = pred < baseline-threshold
                summaries[label][mode][str(threshold)] = {
                    subset:EVAL.metrics(drawn,mode_masks,selection) for subset,selection in
                    (('all',torch.ones(len(fit),dtype=torch.bool)),('test',~fit))}
            valid = mode_masks['valid'] & mode_masks['k3']
            delta = pred-baseline
            summaries[label][mode]['change_mae_health'] = float((delta-masks['dh'])[valid].abs().mean())
            summaries[label][mode]['recovery_catch'] = float((delta>.5)[valid & (masks['dh']>=1)].float().mean())
            summaries[label][mode]['starvation_catch'] = float((delta<-.5)[valid & (masks['dh']==-1)].float().mean())
    return {'positive_controls':positive,'metrics':summaries,
            'teacher_hp':teacher,'generated_hp':generated,'real_hp':real,'aligned_before':aligned,
            'masks':masks,'seed':meta['seed'],'test':~fit}


def memory_rows(frames,teacher,valid):
    rows = {g:{k:torch.zeros(len(frames),dtype=torch.float64) for k in ('n','world','sighting','neighbour')}
            for g in ('same_6_15','moved_6_15','same_2_5')}
    for j in range(len(frames)):
        t,ce,cls,sf,sc,nb = RECALL.cells(frames[j].float())
        eligible = (t>=4)&(cls==1)
        live = torch.zeros_like(eligible)
        live[eligible] = valid[j,t[eligible]-4]
        age = t-sf
        for group,q in (('same_6_15',(sc==ce)&(age>=6)&(age<=15)),
                        ('moved_6_15',(sc!=ce)&(age>=6)&(age<=15)),
                        ('same_2_5',(sc==ce)&(age>=2)&(age<=5))):
            use = live&q
            tt,cc = t[use],ce[use]
            target = frames[j,tt,cc].float()
            rows[group]['n'][j] = use.sum()
            rows[group]['world'][j] = (teacher[j,tt-4,cc].float()-target).square().sum()
            rows[group]['sighting'][j] = (frames[j,sf[use],sc[use]].float()-target).square().sum()
            rows[group]['neighbour'][j] = (frames[j,tt,nb[use]].float()-target).square().sum()
    return rows


def cached_endpoint(path, window, data):
    meta,fit,cache,health,masks,frames,actions,truth,expected = data
    original,st = verified_teval(path,window,expected)
    spec = {'version':'e20-endpoints-cached-v1','sources':TRAIN.source_pins(),
            'original_contract':original.contract,'world_sha256':R.file_hash(path),
            'reader_sha256':R.file_hash(OUT/'health_reader.pt'),'window':window,
            'memory_keys':'sample0 factual; all-five-key aggregate measured separately'}
    store = R.Store(OUT/'endpoints'/f'{st["name"]}_w{window}',spec)
    with store.lock():
        done = store.load('result')
        if done is not None:
            return done
        Rn = len(meta['seed'])
        gen = torch.empty(Rn,16,81,192,dtype=torch.float16)
        tf = torch.empty_like(gen)
        bs = original.load('result')  # verifies completed payload before reading batch generations
        del bs
        batch = json.loads((original.root/'contract.json').read_text())['options']['batch']
        for i in range(0,Rn,batch):
            item = original.load(f'batch_{i}')
            if item is None:raise RuntimeError(f'Missing committed raw batch {i}')
            b = len(item['generated'])
            gen[i:i+b],tf[i:i+b] = item['generated'],item['teacher']
        real = health(frames[:,3:])
        gh,th = health(gen),health(tf)
        true_off = offsets(frames[:,3:])
        gen_off = offsets(torch.cat((frames[:,3:4],gen),1))
        aligned = torch.cat((torch.ones(Rn,1,dtype=torch.bool),(true_off==gen_off).all(-1)[:,:-1]),1)
        report = health_report(meta,fit,masks,real,truth,th,gh,aligned)
        # Preserve per-root sums on exactly the same physical cells for paired intervals.
        valid = ~meta['future_dead'][:,0].cumsum(1).bool()
        memory = memory_rows(frames,tf,valid)
        report.update(name=st['name'],window=window,memory=memory,
                      original_contract=original.contract,original_precision='stored fp16 teacher/generated states',
                      zero_world_forward_calls=True)
        store.save('result',report,1)
        brief = {k:v for k,v in report.items() if k in ('name','window','positive_controls','metrics','zero_world_forward_calls','original_contract')}
        R.atomic_json(OUT/'endpoints'/f'{st["name"]}_w{window}.json',brief,immutable=True)
        print(json.dumps({'name':st['name'],'window':window,'zero_world_forward_calls':True}),flush=True)
        return report


def four(path,data):
    meta,fit,cache,health,masks,frames,actions,truth,_ = data
    device = torch.device('cuda')
    from d4mj.config import config_from_dict
    config = config_from_dict(torch.load(SP.CHECKPOINT,map_location='cpu',weights_only=False)['config'])
    world,st = T.load_world(path,device)
    spec = {'version':'e20-endpoints-four-v1','sources':TRAIN.source_pins(),
            'world_sha256':R.file_hash(path),'reader_sha256':R.file_hash(OUT/'health_reader.pt'),
            'frames':R.tensor_hash(frames),'actions':R.tensor_hash(actions),'window':4,
            'runtime':{'torch':str(torch.__version__),'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name()},
            'masks':{k:R.tensor_hash(v) for k,v in masks.items()}}
    store = R.Store(OUT/'endpoints'/f'{st["name"]}_w4',spec)
    with store.lock():
        if store.load('result') is not None:return
        teacher,generated = torch.zeros(len(fit),16,2),torch.zeros(len(fit),16,2)
        aligned = torch.zeros(len(fit),16,dtype=torch.bool)
        with torch.no_grad():
            for i in range(0,len(fit),16):
                part = store.load(f'batch_{i}')
                if part is None:
                    s = frames[i:i+16].float();a = actions[i:i+16]
                    g = [s[:,j] for j in range(4)]
                    tf,gh = [],[]
                    for k in range(16):
                        current = 3+k
                        acts = a[:,current-3:current+1]
                        pp = T.step(world,s[:,current-3:current+1],acts,device,config)
                        gg = T.step(world,torch.stack(g[-4:],1),acts,device,config)
                        tf.append(health(pp));gh.append(health(gg));g.append(gg)
                    go = offsets(torch.stack(g[3:],1));to = offsets(s[:,3:])
                    ok = torch.cat((torch.ones(len(s),1,dtype=torch.bool),(go==to).all(-1)[:,:-1]),1)
                    part = {'teacher':torch.stack(tf,1),'generated':torch.stack(gh,1),'aligned':ok}
                    store.save(f'batch_{i}',part,1)
                b = len(part['teacher'])
                teacher[i:i+b],generated[i:i+b],aligned[i:i+b] = part['teacher'],part['generated'],part['aligned']
        result = health_report(meta,fit,masks,health(frames[:,3:]),truth,teacher,generated,aligned)
        result.update(name=st['name'],window=4)
        store.save('result',result,1)
        R.atomic_json(OUT/'endpoints'/f'{st["name"]}_w4.json',
                      {k:result[k] for k in ('name','window','positive_controls','metrics')},immutable=True)
        print(json.dumps({'name':st['name'],'window':4,'stage':'complete'}),flush=True)


def paired_ratio(numerators,denominator,seeds,draws=2000):
    groups = [torch.where(seeds==s)[0] for s in seeds.unique()]
    nn = np.array([float(numerators[g].sum()) for g in groups])
    dd = np.array([float(denominator[g].sum()) for g in groups])
    if dd.sum()<=0:return {'point':None,'interval95':None,'denominator':float(dd.sum())}
    ix = np.random.default_rng(20261007).integers(len(groups),size=(draws,len(groups)))
    den = dd[ix].sum(1);values = nn[ix].sum(1)[den>0]/den[den>0]
    return {'point':float(nn.sum()/dd.sum()),'interval95':np.quantile(values,[.025,.975]).tolist(),
            'episode_clusters':len(groups),'denominator':float(dd.sum())}


def compare(paths, window, data):
    states = [cached_endpoint(p,window,data) for p in paths]
    result = {'window':window,'scope':'paired sample0 factual diagnostic roots; no actor/promotion',
              'contrasts':{},'parent_retention':{}}
    for a,b in zip(states,states[1:]):
        assert torch.equal(a['seed'],b['seed'])
        assert torch.equal(a['test'],b['test'])
        assert R.digest({k:R.tensor_hash(v) for k,v in a['masks'].items()}) == R.digest({k:R.tensor_hash(v) for k,v in b['masks'].items()})
        key = a['name']+':'+b['name']
        m = a['masks'];v = m['valid']&m['k3']
        selectors = {'ordinary':v&m['drop2'],'fresh':v&m['drop2']&m['adjacent']&~m['win']&~m['adjwin'],
                     'unchanged':v&(m['dh']==0)}
        result['contrasts'][key] = {}
        for j,reader in enumerate(('independent_hud','factual_loss_reader')):
            result['contrasts'][key][reader] = {}
            for mode in ('teacher','generated'):
                ca = a['real_hp'][:,:-1,j] if mode=='teacher' else torch.cat((a['real_hp'][:,:1,j],a['generated_hp'][:,:-1,j]),1)
                cb = b['real_hp'][:,:-1,j] if mode=='teacher' else torch.cat((b['real_hp'][:,:1,j],b['generated_hp'][:,:-1,j]),1)
                pa,pb = a[mode+'_hp'][...,j]<ca-1.5,b[mode+'_hp'][...,j]<cb-1.5
                common = torch.ones_like(v) if mode=='teacher' else a['aligned_before']&b['aligned_before']
                result['contrasts'][key][reader][mode] = {s:EVAL.cluster_contrast(pa,pb,a['seed'],use&common)
                                                       for s,use in selectors.items()}
    parent = states[0]
    for st in states[1:]:
        rows = {}
        for group in parent['memory']:
            a,b = parent['memory'][group],st['memory'][group]
            for k in ('n','sighting','neighbour'):assert torch.equal(a[k],b[k])
            den = a['neighbour']-a['sighting']
            contrast = paired_ratio(a['world']-b['world'],den,parent['seed'])
            contrast['n'] = int(a['n'].sum())
            contrast['parent_capture'] = float((a['neighbour']-a['world']).sum()/den.sum())
            contrast['arm_capture'] = float((b['neighbour']-b['world']).sum()/den.sum())
            contrast['noninferior_margin_0.05'] = contrast['interval95'] is not None and contrast['interval95'][0]>-.05
            rows[group] = contrast
        result['parent_retention'][st['name']] = rows
    raw = []
    for path in paths:
        source,_ = verified_teval(path,window,data[-1])
        raw.append(source.load('result')['_per_root'])
    result['physical_cost'] = {}
    for index,st in enumerate(states[1:],1):
        a,b = raw[0],raw[index]
        for k in ('seed','alive','onestep_copy','V'):
            if torch.is_tensor(a[k]):assert torch.equal(a[k],b[k])
            else:assert a[k]==b[k]
        one = paired_ratio((b['onestep_err']-a['onestep_err']).sum(1),a['onestep_copy'].sum(1),a['seed'])
        one['upper95_le_0.005'] = one['interval95'] is not None and one['interval95'][1]<=.005
        live = a['alive'][:,-1].float()
        depth16 = paired_ratio((b['gen_err'][:,-1]-a['gen_err'][:,-1])*live,
                               live*float(a['V']),a['seed'])
        result['physical_cost'][st['name']] = {'onestep_all_copy_normalized':one,'depth16_variance_normalized':depth16}
    R.atomic_json(OUT/'endpoints'/f'contrasts_{states[-1]["name"]}_w{window}.json',result,immutable=True)
    print(json.dumps(result),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['cached','four','compare'])
    p.add_argument('paths',type=Path,nargs='+')
    p.add_argument('--window',type=int,choices=[5,15],default=15)
    args=p.parse_args();torch.set_num_threads(4)
    data=inputs()
    if args.stage=='compare':compare(args.paths,args.window,data)
    else:
        for path in args.paths:
            four(path,data) if args.stage=='four' else cached_endpoint(path,args.window,data)


if __name__=='__main__':main()
