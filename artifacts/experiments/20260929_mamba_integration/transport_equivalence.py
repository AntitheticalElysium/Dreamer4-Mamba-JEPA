"""Frozen-weight equivalence of carry transport and the prior fcanvas variant.

This is a mechanism audit, not a train/eval comparison: loads fcanvas weights into both worlds,
feeds exactly the same true six-frame windows, and reports token outputs by no/any scroll and
whether a canvas cell leaves and reenters. No new training data or judgement seeds.
"""
import hashlib
import json
import sys
from pathlib import Path
import torch
ROOT=Path('/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA')
HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT),str(ROOT/'artifacts/experiments/20260927_levers'),str(HERE)]
import teval as E  # noqa: E402
import tworld as T  # noqa: E402
from transport_world import TransportTWorld  # noqa: E402
from scroll import estimate,SHIFTS  # noqa: E402

CKPT=ROOT/'artifacts/eda/levers_tworlds_v1/corrt_raw_suffix_s7_fcanvas.pt'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def reentry(shifts):
    # View window 7x9 at cumulative world offsets; a cell returns after missing >=1 frame.
    origin=[0,0];seen=set();absent=set()
    for k,s in enumerate(shifts):
        dr,dc=SHIFTS[int(s)]
        origin[0]+=dr;origin[1]+=dc
        cells={(r+origin[0],c+origin[1]) for r in range(7) for c in range(9)}
        if cells & absent:return True
        absent |= seen-cells;seen |= cells
    return False

@torch.no_grad()
def main():
    torch.manual_seed(7)
    device='cuda'
    payload=torch.load(CKPT,map_location='cpu',weights_only=False)
    baseline=T.TWorld('corrt',backbone='fcanvas').to(device).eval()
    proposal=TransportTWorld('corrt').to(device).eval()
    baseline.load_state_dict(payload['world']);proposal.load_state_dict(payload['world'])
    cache=torch.load(Path(str(E.CACHE).format('raw')),map_location='cpu',weights_only=False,mmap=True)
    rows=[]
    # Stratify the previously inspected panel by input-derived motion only, before any world outputs.
    # Reentry is rare; a fixed first-40 sample would leave that mechanism untested.
    candidates={'no_scroll':[], 'scroll_no_reentry':[], 'scroll_reentry':[]}
    for idx in range(len(cache['ctx'])):
        frames=torch.cat([cache['ctx'][idx],cache['fut'][idx,:2]],0).float()
        shifts=estimate(frames[:-1],frames[1:]).tolist()
        motion=any(x!=0 for x in shifts)
        label='scroll_reentry' if reentry([0]+shifts) else ('scroll_no_reentry' if motion else 'no_scroll')
        candidates[label].append(idx)
    ids=[idx for label in candidates for idx in candidates[label][:15]]
    for idx in ids:
        frames=torch.cat([cache['ctx'][idx],cache['fut'][idx,:2]],0).float()[None].to(device)
        acts=torch.cat([cache['ctx_a'][idx],cache['fut_a'][idx,:3]],0)[None].to(device)
        shifts=estimate(frames[:,:-1],frames[:,1:])[0].cpu().tolist()
        a=baseline(frames,acts)[0]
        b=proposal(frames,acts)[0]
        d=(a-b).abs().float()
        rows.append({'root':idx,'shifts':shifts,'any_scroll':any(x!=0 for x in shifts),
                     'reentry':reentry([0]+shifts),
                     'max_abs':float(d.max()),'rmse':float(d.square().mean().sqrt()),
                     'normalizer':float(a.square().mean().sqrt())})
        if len(rows)%10==0:print(json.dumps({'done':len(rows),'of':len(ids)}),flush=True)
    by={}
    for name,sub in [('no_scroll',[r for r in rows if not r['any_scroll']]),
                     ('scroll_no_reentry',[r for r in rows if r['any_scroll'] and not r['reentry']]),
                     ('scroll_reentry',[r for r in rows if r['reentry']])]:
        by[name]={'n':len(sub),'max_abs':max((x['max_abs'] for x in sub),default=None),
                  'mean_rmse':sum(x['rmse'] for x in sub)/len(sub) if sub else None,
                  'mean_relative_rmse':sum(x['rmse']/x['normalizer'] for x in sub)/len(sub) if sub else None}
    # The checkpoint was written before ldad1 was added to the pool registry. Reconstruct the
    # original source bytes and verify the executable model code itself has not drifted.
    source=(ROOT/'artifacts/experiments/20260927_levers/tworld.py').read_text()
    current_pool='POOLS = {"raw": ROOT / "artifacts/eda/spatial_pool_v1", "tc": ROOT / "artifacts/eda/spatial_pool_tc_v1",\n         "ldad10": ROOT / "artifacts/eda/spatial_pool_ldad10_v1",\n         "ldad1": ROOT / "artifacts/eda/spatial_pool_ldad1_v1"}'
    former_pool='POOLS = {"raw": ROOT / "artifacts/eda/spatial_pool_v1", "tc": ROOT / "artifacts/eda/spatial_pool_tc_v1",\n         "ldad10": ROOT / "artifacts/eda/spatial_pool_ldad10_v1"}'
    assert current_pool in source
    reconstructed_sha=hashlib.sha256(source.replace(current_pool,former_pool).encode()).hexdigest()
    assert reconstructed_sha==payload['script_sha256'], 'historical model source drift'
    out={'status':'complete','scope':'frozen fcanvas weights on inspected diagnosis true windows',
         'candidate_counts':{k:len(v) for k,v in candidates.items()},'selected_ids':ids,
         'checkpoint_source_sha256':payload['script_sha256'],'reconstructed_source_sha256':reconstructed_sha,
         'checkpoint_sha256':sha(CKPT),'raw_cache_sha256':sha(str(E.CACHE).format('raw')),
         'transport_source_sha256':sha(HERE/'transport_world.py'),'carry_source_sha256':sha(HERE/'carry_transport.py'),
         'groups':by,'rows':rows}
    HERE.joinpath('transport_equivalence.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({'stage':'complete','groups':by}),flush=True)

if __name__=='__main__':main()
