"""CPU HUD transfer control on saved worlds; same episode split, no TEST tuning."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import prefix_readout_refit as G
import loss_intervention_eval as L

NAMES = ('health', 'food', 'drink', 'energy')


def log(**row):
    print(json.dumps(row), flush=True)


def scores(head, x):
    return ((x.float() - head['mu']) / head['sd']).double() @ head['w'][:-1] + head['w'][-1]


def fit(x, y, fit_ids, dev_ids):
    xf = x[fit_ids].flatten(0, 1).float()
    xd = x[dev_ids].flatten(0, 1).float()
    yf = y[fit_ids].flatten(0, 1).double()
    yd = y[dev_ids].flatten(0, 1).double()
    mu, sd = xf.mean(0), xf.std(0).clamp_min(1e-6)
    xf = torch.cat([((xf-mu)/sd).double(), torch.ones(len(xf), 1, dtype=torch.float64)], 1)
    xd = torch.cat([((xd-mu)/sd).double(), torch.ones(len(xd), 1, dtype=torch.float64)], 1)
    # Exact dual ridge, equivalent to the existing standardized primal family.
    gram = xf @ xf.T
    eye = torch.eye(len(xf), dtype=torch.float64)
    best, errors = None, {}
    for alpha in G.ALPHAS:
        w = xf.T @ torch.linalg.solve(gram + alpha*len(xf)*eye, yf)
        error = float((xd@w-yd).square().mean())
        errors[str(alpha)] = error
        if best is None or error < best[0]:
            best = error, alpha, w
    return {'mu':mu, 'sd':sd, 'w':best[2], 'alpha':best[1], 'dev_mse':best[0], 'dev_errors':errors}


def main():
    torch.set_num_threads(8)
    root = G.S.OUT/'generated_prefix'
    out = root/'hud_readout_refit'
    out.mkdir(exist_ok=True)
    manifest = json.loads((root/'readout_refit/contract.json').read_text())
    panel = torch.load(G.S.OUT/'panel.pt', map_location='cpu', weights_only=False)
    visible = torch.load(G.S.OUT/'snap_visible.pt', map_location='cpu', weights_only=False)['visible']
    episodes = [r['episode_id'] for r in panel['roots']]
    ids = {name:np.array([episodes.index(e) for e in selected]) for name,selected in manifest['split'].items()}
    assert [len(ids[k]) for k in ('FIT','DEV','TEST')] == [100,50,50]
    assert not set(ids['FIT']) & set(ids['DEV']) and not set(ids['FIT']) & set(ids['TEST'])
    assert not set(ids['DEV']) & set(ids['TEST'])
    targets = visible[:,16:32,1512:1516].float()*9
    paths = {(0,'teacher'):root/'eval_u0000/pred_teacher.pt',
             (3000,'teacher'):root/'eval_u3000/pred_teacher.pt',
             (3000,'prefix'):root/'eval_u3000/pred_prefix.pt'}
    contract = {'format':'oracle_hud_refit_v1', 'source_sha256':G.S.sha(__file__),
                'inputs':{str(p):G.S.sha(p) for p in [G.S.OUT/'panel.pt',G.S.OUT/'snap_visible.pt',
                                                    G.S.OUT/'snap_bank_readers.pt',*paths.values()]},
                'split':manifest['split'], 'capacity':'3456 HUD features plus regularized intercept ->4 native counters',
                'fit_depths':'1-16', 'test_depths':'9-16', 'alphas':G.ALPHAS,
                'selection':'minimum joint four-counter DEV MSE', 'bootstrap':'2000 paired TEST-episode draws seed20261012',
                'scope':'post-hoc readout control; reused world panel; no fresh judgment or information ceiling'}
    cid = hashlib.sha256(json.dumps(contract,sort_keys=True).encode()).hexdigest()
    contract_path = out/'contract.json'
    if contract_path.exists():
        assert json.loads(contract_path.read_text()) == json.loads(json.dumps(contract))
    else:
        G.S.atomic(contract, contract_path, True)
    # Dual/primal agreement before fitting actual high-dimensional heads.
    rng = torch.Generator().manual_seed(10)
    xx = torch.randn(17,11,generator=rng,dtype=torch.float64)
    yy = torch.randn(17,4,generator=rng,dtype=torch.float64)
    primal = torch.linalg.solve(xx.T@xx+.2*17*torch.eye(11,dtype=torch.float64),xx.T@yy)
    dual = xx.T@torch.linalg.solve(xx@xx.T+.2*17*torch.eye(17,dtype=torch.float64),yy)
    assert torch.allclose(primal,dual,atol=1e-12,rtol=1e-12)
    real = panel['tokens'][:,:,63:].float().flatten(2)
    heads = {}
    hp = out/'real_head.pt'
    if hp.exists():
        st=torch.load(hp,map_location='cpu',weights_only=False);assert st['contract_id']==cid;heads['real']=st['head']
    else:
        log(stage='fit_real_start')
        heads['real']=fit(real[:,16:32],targets,ids['FIT'],ids['DEV'])
        G.S.atomic({'contract_id':cid,'head':heads['real']},hp)
    reader = G.S.Readers(torch.load(G.S.OUT/'snap_bank_readers.pt',map_location='cpu',weights_only=False))
    masks=L.health_masks(visible);test=ids['TEST']
    test_masks={k:(v[test] if isinstance(v,np.ndarray) else v) for k,v in masks.items()}
    truth=targets[test].numpy()
    boot=np.random.default_rng(20261012).multinomial(50,np.full(50,1/50),size=2000)
    results,rows={},{}
    for (update,arm),path in paths.items():
        destination=out/f'u{update:04d}_{arm}.json'
        if destination.exists():
            assert json.loads(destination.read_text())['contract_id']==cid
            log(stage='result_resume',update=update,arm=arm);continue
        prediction=torch.load(path,map_location='cpu',weights_only=False)
        for mode in ('tf','gen'):
            hx=prediction[mode][:,:,63:].float().flatten(2)
            hp=out/f'head_u{update:04d}_{arm}_{mode}.pt'
            if hp.exists():
                st=torch.load(hp,map_location='cpu',weights_only=False);assert st['contract_id']==cid;head=st['head']
            else:
                log(stage='fit_start',update=update,arm=arm,mode=mode)
                head=fit(hx,targets,ids['FIT'],ids['DEV'])
                G.S.atomic({'contract_id':cid,'head':head},hp)
            pred_h={name:scores(h,hx[test].flatten(0,1)).float().view(50,16,4).numpy()
                    for name,h in {'real':heads['real'],'matched':head}.items()}
            pred_h['original']=reader.hud(hx[test].flatten(0,1))[:,:4].view(50,16,4).numpy()
            model={}
            for name,pred in pred_h.items():
                delta=pred-truth
                statistics={}
                for j,counter in enumerate(NAMES):
                    mse=(delta[:,8:,j]**2).mean(1);mae=np.abs(delta[:,8:,j]).mean(1)
                    statistics[counter]={'mse':float(mse.mean()),'mse_ci95':np.quantile(boot@mse/50,[.025,.975]).tolist(),
                                         'mae':float(mae.mean()),'r2':float(1-(delta[:,8:,j]**2).mean()/truth[:,8:,j].var())}
                current=np.concatenate([reader.hud(real[test,15])[:,:1].numpy(),pred[:,:-1,0]],1)
                ev=L.health_events(pred[:,:,0],current,test_masks)
                statistics['events']={k:int(v[:,8:].sum())for k,v in ev.items()}
                model[name]=statistics
                rows[f'{mode}/{name}']=pred.tolist()
            contrast={}
            for counter,j in zip(NAMES,range(4)):
                error_matched=(pred_h['matched'][:,8:,j]-truth[:,8:,j])**2
                error_original=(pred_h['original'][:,8:,j]-truth[:,8:,j])**2
                diff=(error_matched-error_original).mean(1)
                contrast[counter]={'difference':float(diff.mean()),'ci95':np.quantile(boot@diff/50,[.025,.975]).tolist()}
            results[mode]={'alpha':head['alpha'],'readings':model,'matched_minus_original':contrast}
        G.S.atomic({'contract_id':cid,'update':update,'arm':arm,'TEST_episodes':manifest['split']['TEST'],
                    'results':results,'prediction_rows':rows,'truth_rows':truth.tolist()},destination,True)
        log(stage='model_done',update=update,arm=arm,results=results)
        results,rows={},{}
    log(stage='done')


if __name__=='__main__':
    main()
