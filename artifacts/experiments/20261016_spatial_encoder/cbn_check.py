"""C-BN (FROZEN_STATE_CONTROL_PREREG.md): is the joint worlds' mob erasure a BatchNorm-mode artefact? Read-only.

A joint predictor (J) was trained on deployed tokens normalized with TRAIN-mode BN (batch statistics over 128 windows x 4 frames x
81 patches), but is evaluated with eval-mode BN (running statistics). Here the same J predictor runs on the same held-out DEV
windows (erasure_diagnosis.dev_windows: 2,048 seeded 4-frame windows of the training corpus's dev split; never FINAL) with the
projector in (a) eval mode and (b) train mode over training-shaped batches of 128 windows (a deep copy; its buffer updates are
discarded). Labels: Raw@10000's Tier B readers on the real f3 frame (pseudo-labels). Retention = refit AP on the expected
state / refit AP on the copy state, readers fitted on one episode-disjoint half and scored on the other (as erasure_diagnosis).
If (b) restores retention, the erasure is (partly) a BN-mode / cohort artefact; a positive result shows mode / cohort sensitivity,
NOT the absence of co-adaptation (train-mode statistics include target-future frames).
Corrected cohort (Codex audit, before any C-BN run): the full arm's BN saw 128 clean + 32 AUGMENTED windows (spatial_encoder.py:
the first INV_B = 32 windows recoloured by augment()), i.e. 640 frames / 51,840 patches; the projector copy is applied ONCE to the
whole cohort's backbone tokens (never through PatchState's 256-frame chunks). MoP arms have no augmented windows (128 clean).
Usage: cbn_check.py <label>=<spatial ckpt>[,aug] ... -> erasure_v2/cbn_<label>.json   (',aug' = the full arm's cohort)
"""
import copy
import json
import os
import sys
from pathlib import Path

os.environ.setdefault('TRITON_F32_DEFAULT', 'ieee')
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import erasure_diagnosis as ED                                                 # noqa: E402
import spatial_encoder as SE                                                   # noqa: E402
import spatial_eval as EV                                                      # noqa: E402
import tierb_eval as TB                                                        # noqa: E402

import torch                                                                   # noqa: E402


@torch.no_grad()
def main():
    frames, acts, ep = ED.dev_windows()
    raw_probes = torch.load(EV.OUT / 'probes2_raw10000.pt', weights_only=False)
    rawenc = EV.PatchState('raw@10000')
    lab = {}
    for nm in ('zombie', 'cow'):
        sc = torch.cat([EV.apply(raw_probes[nm], rawenc(frames[i:i + 256, 3])[:, :63].reshape(-1, 192).cpu()) for i in range(0, len(frames), 256)])
        lab[nm] = (sc[:, 0] > raw_probes[f'{nm}_threshold']).view(-1, 63)
    del rawenc
    fit = (ep % 2) == 0
    for arg in sys.argv[1:]:
        label, ck = arg.split('=', 1)
        use_aug = ck.endswith(',aug'); ck = ck.split(',')[0]
        aug_g = torch.Generator().manual_seed(20261106)
        ps = EV.PatchState('spatial:' + ck)
        proj_tr = copy.deepcopy(ps.proj).train()
        sd = torch.load(ck, map_location=EV.DEV, weights_only=False)['predictor']
        pred = SE.PatchPredictor(sd['router.weight'].shape[0]).to(EV.DEV).eval(); pred.load_state_dict(sd)
        g = torch.Generator(device=EV.DEV).manual_seed(20261071)
        res = {'label': label, 'checkpoint': ck, 'windows': len(frames),
               'bn_cohort': ('128 clean + 32 augmented windows' if use_aug else '128 clean windows') + ' x 4 frames x 81 patches, one projector call'}
        for mode in ('eval', 'train'):
            U = {k: [] for k in ('exp', 'copy', 'real')}
            for i in range(0, len(frames), 128):
                x = frames[i:i + 128].flatten(0, 1).to(EV.DEV)
                if mode == 'eval':
                    z = ps(frames[i:i + 128].flatten(0, 1))
                else:
                    nw = len(x) // 4
                    if use_aug:                                                    # the training cohort: + recoloured first 32 windows
                        xa = SE.augment(frames[i:i + SE.INV_B].flatten(0, 1), aug_g).to(EV.DEV)
                        x = torch.cat([x, xa])
                    _, _, tok, _, _ = ps.enc._hidden(x[:, None])
                    tok = tok.float()
                    z = proj_tr(tok.reshape(-1, tok.shape[-1])).view(tok.shape).float()[:nw * 4]   # one BN call; keep the clean windows
                z = z.view(-1, 4, 81, z.shape[-1])
                exp, _, _, _ = TB.step(pred, z[:, 1], z[:, 2], acts[i:i + 128, 2], g)
                U['exp'].append(exp[:, :63].cpu()); U['copy'].append(z[:, 2, :63].cpu()); U['real'].append(z[:, 3, :63].cpu())
            U = {k: torch.cat(v) for k, v in U.items()}
            D = U['exp'].shape[-1]
            res[mode] = {}
            for nm in ('zombie', 'cow'):
                ex = lab[nm]
                o = {}
                for k in U:
                    Wr = EV.ridge(U[k][fit].reshape(-1, D).float(), ex[fit].reshape(-1).float()[:, None])
                    o[f'refit_ap_{k}'] = ED.rnd(ED.AP(ED.score(Wr, U[k][~fit].reshape(-1, D))[:, 0], ex[~fit].reshape(-1)))
                o['retention'] = ED.rnd(o['refit_ap_exp'] / o['refit_ap_copy']) if o['refit_ap_copy'] else None
                res[mode][nm] = o
        (ED.OUTD / f'cbn_{label}.json').write_text(json.dumps(res, indent=1))
        print(json.dumps(res), flush=True)
        del ps, proj_tr, pred


if __name__ == '__main__':
    main()
