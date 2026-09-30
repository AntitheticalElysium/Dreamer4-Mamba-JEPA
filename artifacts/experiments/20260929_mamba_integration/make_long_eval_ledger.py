"""Fixed DEV-only 128-frame windows for long-state mechanics; FINAL remains unopened."""
import hashlib,json
from pathlib import Path
import torch
HERE=Path(__file__).resolve().parent
AUDIT=Path('artifacts/lewm_m4_canonical/raw/dataset_audit.json')
LENGTH=128;N=128;SEED=20260929

def main():
 out=HERE/'long_eval_ledger.jsonl'
 assert not out.exists(),'eval ledger exists; refusing overwrite'
 j=json.loads(AUDIT.read_text())
 eligible=[e for e in j['episodes'] if e['split']=='dev' and e['steps']+1>=LENGTH]
 counts=torch.tensor([e['steps']+2-LENGTH for e in eligible],dtype=torch.float64)
 rng=torch.Generator().manual_seed(SEED)
 chosen=torch.multinomial(counts,N,replacement=False,generator=rng).tolist()
 rows=[]
 for i in chosen:
  e=eligible[i];start=int(torch.randint(int(counts[i]),(),generator=rng))
  rows.append({'episode_id':e['id'],'start':start})
 content=''.join(json.dumps(r,sort_keys=True,separators=(',',':'))+'\n' for r in rows)
 out.write_text(content)
 meta={'status':'sealed','split':'dev','length':LENGTH,'windows':N,'unique_episodes':len({r['episode_id'] for r in rows}),
       'seed':SEED,'eligible_dev_episodes':len(eligible),'ledger_sha256':hashlib.sha256(content.encode()).hexdigest(),
       'dataset_audit_sha256':hashlib.sha256(AUDIT.read_bytes()).hexdigest(),'final_opened':False}
 (HERE/'long_eval_ledger.json').write_text(json.dumps(meta,indent=2)+'\n')
 print(json.dumps(meta),flush=True)
if __name__=='__main__':main()
