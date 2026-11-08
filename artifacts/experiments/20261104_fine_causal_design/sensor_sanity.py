"""Small renderer-input sanity check. No world inference/training or night claim."""
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
ASSET=ROOT/'.venv/lib/python3.12/site-packages/craftax/craftax_classic/assets/arrow-up.png'


def main():
    torch.set_num_threads(1)
    a=np.array(Image.open(ASSET).convert('RGBA'));a[...,3]=(a[...,3]//255)*255
    a=np.array(Image.fromarray(a).resize((7,7),Image.Resampling.NEAREST))
    rgb,alpha=a[...,:3],a[...,3:4]/255
    parts=dict(up=(rgb,alpha),down=(rgb[::-1].copy(),alpha[::-1].copy()),
        left=(rgb.transpose(1,0,2).copy(),alpha.transpose(1,0,2).copy()),
        right=(rgb[::-1].transpose(1,0,2).copy(),alpha[::-1].transpose(1,0,2).copy()))
    raw=np.array([np.rint(r*m+np.array([63,104,69])*(1-m)).astype(np.uint8) for r,m in parts.values()])
    q=F.adaptive_avg_pool2d(torch.tensor(raw).permute(0,3,1,2).float(),3).round().flatten(1)
    names=list(parts);out={}
    for i in range(4):
        for j in range(i+1,4):
            out[names[i]+'_'+names[j]]=dict(equal_3x3_input=bool(torch.equal(q[i],q[j])),
                different_channels=int((q[i]!=q[j]).sum()),max_pixel_difference=float((q[i]-q[j]).abs().max()))
    report=dict(scope='Clean arrow asset on a constant grass-color background, exact resize/flip/pool operations only.',
        non_claim='No night noise, trained encoder, model-use or natural-episode result.',pairs=out)
    (HERE/'sensor_sanity.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
