"""Compare two existing runtime paths on identical actual recorded histories. No state substitutions."""
from .common import *


@torch.no_grad()
def main():
    before=fingerprints();hist=torch.load(OUT/'training_histories.pt',weights_only=False);cfg=TR.load_cfg();episodes=D.load_split(cfg)['train'];out={}
    for snapshot,streams in hist.items():
        ck=torch.load(RUN/f'ckpt_{snapshot:02d}.pt',map_location='cpu',weights_only=False,mmap=True);model=M.M0(cfg).cuda().eval().requires_grad_(False);model.load_state_dict(ck['model']);mh=model_hash(model);rows=[]
        for i,r in enumerate(streams):
            if not r['t']:rows.append({'stream':i,'depth':0,'reset_pending':True});continue
            E=episodes[r['episode']];records=[[(0,r['uid'],*x[2:]) for x in r['history']]]
            data=TR.tensors(D.gather([E],records),'cuda');parts=[]
            with TR.autocast(torch.device('cuda'),cfg):
                for start in range(0,r['t'],512):parts.append(model.perception(data['frames'][:,start:start+512],data['fixation'][:,start:start+512]))
                P=M.Percept(*(torch.cat([p.parts()[k] for p in parts],1) for k in range(3)),data['fixation'])
                tok=model.belief.tokens(model.belief.percept_embed(P),data['reward'],data['terminated'],data['action'])
                y,fresh=model.belief(tok,model.belief.initial(1,'cuda'))
            current=y[0,-1].float().cpu();stored=ck['cache'][i,-1].float();delta=current-stored
            row={'stream':i,'depth':r['t'],'belief_rmse':float(delta.square().mean().sqrt()),'belief_max_abs':float(delta.abs().max()),
                 'belief_cosine':float(torch.nn.functional.cosine_similarity(current,stored,dim=0)),
                 'stored_rms':float(stored.square().mean().sqrt()),'fresh_rms':float(current.square().mean().sqrt())}
            row['ssm_relative_rmse_by_layer']=[]
            for a,z in zip(fresh,ck['carry']):
                source=z.ssm[i].float();diff=a.ssm[0].cpu()-source
                row['ssm_relative_rmse_by_layer'].append(float(diff.square().mean().sqrt()/source.square().mean().sqrt().clamp(min=1e-8)))
            rows.append(row)
        out[snapshot]=rows;save('state_compatibility.json',out);print(snapshot,[(r['depth'],round(r.get('belief_rmse',0),4)) for r in rows],flush=True)
        assert model_hash(model)==mh;del model,ck;torch.cuda.empty_cache()
    assert before==fingerprints()


if __name__=='__main__':main()
