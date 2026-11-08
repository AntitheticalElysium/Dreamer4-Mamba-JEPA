"""Post-result stronger coarse control: use the extra branch for the full pre-crop TSP image.

No new observations enter. Same initialization, batches, optimizer and1200-step
budget as the original readers. Baseline choice is made on train calibration
episodes, not discovery/confirmation performance.
"""
import json,time,gc
from pathlib import Path
import torch
import torch.nn.functional as F
from . import fit as A
HERE=Path(__file__).resolve().parent

def main():
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=True
    feat=A.Features(A.load('train'));kp,hp=feat.pops();z=feat.z;val=feat.queries[z['episode'][feat.queries//63]>=360]
    old=torch.load(HERE/'readers.pt',map_location='cpu',weights_only=False);saved={};stats={}
    for family in ['flat','split']:
        for seed in [0,1]:
            key=f'{family}_{seed}_coarse_tsp';torch.manual_seed(617001+seed);net=A.Reader(feat.dim,family).to(A.DEV)
            opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.0001);gen=torch.Generator(device=A.DEV).manual_seed(617101+seed);curve=[];start=time.time()
            for step in range(1200):
                q=torch.cat([p[torch.randint(len(p),(100,),device=A.DEV,generator=gen)] for p in kp]+[p[torch.randint(len(p),(32,),device=A.DEV,generator=gen)] for p in hp])
                f,c=q//63,q%63;kind=z['truth'][f,c].long();heading=z['heading'][f,c].long();has=heading>=0
                with torch.autocast('cuda',dtype=torch.bfloat16):y=net(feat.make(q,'coarse_tsp'));loss=F.cross_entropy(y[:,:5].float(),kind)+F.cross_entropy(y[has,5:].float(),heading[has])
                opt.zero_grad(set_to_none=True);loss.backward();norm=torch.nn.utils.clip_grad_norm_(net.parameters(),1.);opt.step()
                if step%300==0 or step==1199:curve.append(dict(step=step+1,loss=float(loss.detach()),gradient_norm=float(norm)))
            net.eval();pred=A.evaluate(net,feat,'coarse_tsp',val);ff,cc=val//63,val%63;labels={'kind':z['truth'][ff,cc].cpu().long(),'heading':z['heading'][ff,cc].cpu().long()};cal={};cal_scores={}
            for task,sl in [('kind',slice(0,5)),('heading',slice(5,9))]:
                ok=labels[task]>=0;cal[task]=A.calibrate(pred[ok,sl],labels[task][ok])
                ca=cal[task];lp=F.log_softmax(pred[ok,sl]/torch.exp(torch.tensor(ca['log_temperature']))+torch.tensor(ca['bias']),-1)
                cal_scores[task]=float(F.nll_loss(lp,labels[task][ok]))
            saved[key]=dict(state={k:v.cpu() for k,v in net.state_dict().items()},calibration=cal,family=family,seed=seed,mode='coarse_tsp',dimension=feat.dim)
            # Re-evaluate the original coarse reader on the SAME natural calibration queries.
            basekey=f'{family}_{seed}_coarse';base=A.Reader(feat.dim,family).to(A.DEV);base.load_state_dict(old[basekey]['state']);base.eval();bp=A.evaluate(base,feat,'coarse',val);bs={}
            for task,sl in [('kind',slice(0,5)),('heading',slice(5,9))]:
                ok=labels[task]>=0;ca=old[basekey]['calibration'][task];lp=F.log_softmax(bp[ok,sl]/torch.exp(torch.tensor(ca['log_temperature']))+torch.tensor(ca['bias']),-1)
                bs[task]=float(F.nll_loss(lp,labels[task][ok]))
            stats[key]=dict(curve=curve,seconds=time.time()-start,new_calibration_nll=cal_scores,old_calibration_nll=bs,
                chosen_by_task={task:key if cal_scores[task]<bs[task] else basekey for task in ['kind','heading']})
            torch.save(saved,HERE/'coarse_control_readers.pt');(HERE/'coarse_control.json').write_text(json.dumps(stats,indent=2)+'\n');print(key,stats[key]['chosen_by_task'],cal_scores,bs,flush=True)
            del net,opt,base;gc.collect();torch.cuda.empty_cache()
    del feat;gc.collect();torch.cuda.empty_cache();dev=A.Features(A.load('dev'))
    for key,s in saved.items():
        net=A.Reader(dev.dim,s['family']).to(A.DEV);net.load_state_dict(s['state']);net.eval();pred=A.evaluate(net,dev,'coarse_tsp',dev.queries)
        torch.save(pred.half(),HERE/f'dev_logits_{key}.pt');print('control dev',key,flush=True);del net;torch.cuda.empty_cache()

if __name__=='__main__':main()
