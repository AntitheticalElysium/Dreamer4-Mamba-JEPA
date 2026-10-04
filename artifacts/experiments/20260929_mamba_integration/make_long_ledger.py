"""Immutable 64-frame TRAIN-only window ledger shared by Raw, LDAD1 and LDAD10 encoders."""
import hashlib
import json
from pathlib import Path
import torch
ROOT=Path('/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA')
HERE=Path(__file__).resolve().parent
AUDIT=ROOT/'artifacts/lewm_m4_canonical/raw/dataset_audit.json'
LENGTH=64
MAIN=12288
SEED=20260929

def main():
    out=HERE/'long_ledger.jsonl'
    assert not out.exists(), 'ledger exists; refusing overwrite'
    audit=json.loads(AUDIT.read_text())
    eligible=[e for e in audit['episodes'] if e['split']=='train' and e['uniform'] and e['steps']+1>=LENGTH]
    counts=torch.tensor([e['steps']+2-LENGTH for e in eligible],dtype=torch.float64)
    rng=torch.Generator().manual_seed(SEED)
    chosen=torch.multinomial(counts,MAIN,replacement=True,generator=rng).tolist()
    rows=[]
    for i in chosen:
        e=eligible[i]
        start=int(torch.randint(int(counts[i]),(),generator=rng))
        rows.append({'episode_id':e['id'],'start':start,'terminal':False})
    terminal=[e for e in eligible if e['terminals']>0]
    rows += [{'episode_id':e['id'],'start':e['steps']+1-LENGTH,'terminal':True} for e in terminal]
    assert len(eligible)==7755 and len(terminal)==7501 and len(rows)==19789
    assert all(r['start']>=0 for r in rows)
    content=''.join(json.dumps(r,sort_keys=True,separators=(',',':'))+'\n' for r in rows)
    out.write_text(content)
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    meta={'status':'sealed','length':LENGTH,'main':MAIN,'terminal':len(terminal),
          'eligible_train_episodes':len(eligible),'possible_train_windows':int(counts.sum()),
          'seed':SEED,'row_count':len(rows),'unique_episode_count':len({r['episode_id'] for r in rows}),
          'unique_window_count':len({(r['episode_id'],r['start']) for r in rows}),
          'ledger_sha256':sha(out),'dataset_audit_sha256':sha(AUDIT)}
    (HERE/'long_ledger.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(json.dumps(meta),flush=True)

if __name__=='__main__':main()
