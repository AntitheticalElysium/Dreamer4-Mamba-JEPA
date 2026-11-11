"""GPU mechanical comparison of new full-frame auxiliary versus historical F4, no training."""
import os
os.environ.setdefault('JAX_PLATFORMS','cpu')
import ast,json,subprocess
from pathlib import Path
import torch
from saccade import model as M,train as TR
HERE=Path(__file__).resolve().parent

@torch.no_grad()
def main():
    torch.set_num_threads(2)
    source=subprocess.check_output(['git','show','HEAD:saccade/model.py'],text=True)
    tree=ast.parse(source);node=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='ChangeMap')
    scope=dict(vars(M));exec(compile(ast.Module(body=[node],type_ignores=[]),'legacy_change_head','exec'),scope)
    cls=M.ChangeMap;ck=torch.load(TR.ROOT/'artifacts/experiments/20261106_p1f4/run/ckpt_10.pt',map_location='cpu',weights_only=False)
    M.ChangeMap=scope['ChangeMap']
    try:old=M.M0(ck['cfg']).cuda().eval();old.load_state_dict(ck['model'])
    finally:M.ChangeMap=cls
    new=M.M0(ck['cfg']).cuda().eval();new.load_state_dict(ck['model'])
    bank=torch.load(HERE/'phase_bank.pt',weights_only=False)[0]
    f=torch.as_tensor(bank['frames'][0,:12],device='cuda')[None]
    fix=torch.full((1,12,2),4,device='cuda',dtype=torch.long);act=torch.zeros((1,12),device='cuda',dtype=torch.long)
    report={}
    outputs=[]
    for model in (old,new):
        with TR.autocast(f.device,ck['cfg']):
            p=model.perception(f,fix);emb=model.belief.percept_embed(p)
            y,_=model.belief(model.belief.tokens(emb,torch.zeros(1,12,device='cuda'),act.bool(),act),model.belief.initial(1,'cuda'))
            mem=model.belief.memory(y[:,-1:]);cache=model.denoiser.new_cache()
            xs=M.sample(model.denoiser,[tuple(x.shape[2:]) for x in p.parts()],act[:,:1],mem,fix[:,:1],cache,ck['cfg'],torch.Generator(device='cuda').manual_seed(42))
            logits=model.change(p.p[:,:-1],p.p[:,1:],y[:,1::2][:,:-1])
        outputs.append((p.parts(),y,xs,logits))
    a,b=outputs
    report['perception_bitwise']=all(torch.equal(x,y) for x,y in zip(a[0],b[0]))
    report['belief_bitwise']=torch.equal(a[1],b[1]);report['samples_bitwise']=all(torch.equal(x,y) for x,y in zip(a[2],b[2]))
    report['old_auxiliary_map_max_abs']=float((a[3]-b[3][:,:,:7, :, 1]).abs().max())
    report['new_auxiliary_shape']=list(b[3].shape)
    assert report['perception_bitwise'] and report['belief_bitwise'] and report['samples_bitwise']
    (HERE/'LEGACY_CHECK.json').write_text(json.dumps(report,indent=2));print(report,flush=True)

if __name__=='__main__':main()
