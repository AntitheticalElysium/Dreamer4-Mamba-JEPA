"""Where are the imagined mobs? (R60 check of R59's census-based level-3 reading; descriptive, evaluation only, CPU.)

B4 counts mobs (a census) and B2 found that Spatial's expected one-step state reads no mob at cells where every key's real next
frame has one. This measures LOCATION at depth 1 on the Tier B test roots, for each recorded future (the B4 setting: root's 2-frame
context, the recorded first action, 4 router samples, same probes2 readers and thresholds):
  imagined (each sample, and the expected state), decoded-real (the reader on the real next frame) and copy (the reader on the
  root's own state, i.e. "nothing happens") cells vs the exact mob cells of the real next frame:
  count per frame, location precision |pred & exact| / |pred|, location recall |pred & exact| / |exact| (pooled over units).
State-space fidelity (reader-free): squared error of the expected / sampled predicted map tokens to the real next frame's tokens,
relative to copying the root's tokens (sum pred / sum copy; < 1 = better than copy), at exact mob cells and at mob-free cells.
Usage: census_localization.py <name>=<encoder spec>,<predictor checkpoint>,<probes2 file> ... -> prints, census_loc_<names>.json
"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault('TRITON_F32_DEFAULT', 'ieee')
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import spatial_encoder as SE                                                   # noqa: E402
import spatial_eval as EV                                                      # noqa: E402
import spatial_eval_v2 as EV2                                                  # noqa: E402
import tierb_eval as TB                                                        # noqa: E402

import torch                                                                   # noqa: E402

import objective_audit as OA                                                   # noqa: E402


@torch.no_grad()
def main():
    test = json.loads(EV2.SPLIT.read_text())['test']
    rows, i = {}, 0                                                            # keep only the test roots (same global order as Tier B)
    for f in sorted(OA.DIAG.glob('seed-*.pt')):
        for r in torch.load(f, weights_only=False):
            if i in set(test):
                rows[i] = r
            i += 1
    res = {}
    for arg in sys.argv[1:]:
        name, spec = arg.split('=', 1)
        enc_spec, pred_ck, probe_file = spec.split(',')
        enc = EV.PatchState(enc_spec)
        sd = torch.load(pred_ck, map_location=EV.DEV, weights_only=False)['predictor']
        pred = SE.PatchPredictor(sd['router.weight'].shape[0]).to(EV.DEV).eval(); pred.load_state_dict(sd)
        probes = torch.load(probe_file, weights_only=False)
        g = torch.Generator(device=EV.DEV).manual_seed(20261071)
        th = {'zombie': probes['zombie_threshold'], 'cow': probes['cow_threshold']}
        acc = {m: {k: [0, 0, 0] for k in ('sample', 'expected', 'decoded_real', 'copy')} for m in th}   # [pred cells, hits, units]
        exact_n = {m: 0 for m in th}
        sq = {k: [0., 0.] for k in ('expected', 'sample', 'copy')}                       # [mob cells, mob-free cells]
        units = 0
        for ri in test:
            r = rows[ri]
            st = enc(r['context'][-2:])
            _, zc, cc = TB.read(probes, st[1:2])
            for f in range(5):
                if bool(r['future_dead'][f][0]):
                    continue
                exp, smp, _, _ = TB.step(pred, st[0:1].expand(4, -1, -1), st[1:2].expand(4, -1, -1), r['future_actions'][0:1].repeat(4), g)
                _, ze, ce = TB.read(probes, exp[0:1].cpu()); _, zs, cs = TB.read(probes, smp.cpu())
                real = enc(r['future_frames'][f][0:1])
                _, zr, cr = TB.read(probes, real.cpu())
                _, lay, _ = OA.layout(r['future_visible'][f][0:1])
                units += 1
                mob = (lay[0] & 3) > 0
                for k, x in (('expected', exp[0:1]), ('sample', smp), ('copy', st[1:2])):
                    e = (x[:, :63].cpu().float() - real[:, :63].cpu().float()).pow(2).sum(-1).mean(0)   # [63]
                    sq[k][0] += float(e[mob].sum()); sq[k][1] += float(e[~mob].sum())
                for m, bit, s_, e_, r_, c_ in (('zombie', 1, zs, ze, zr, zc), ('cow', 2, cs, ce, cr, cc)):
                    ex = (lay[0] & bit) > 0
                    exact_n[m] += int(ex.sum())
                    for k, sc in (('sample', s_), ('expected', e_), ('decoded_real', r_), ('copy', c_)):
                        p = sc > th[m]                                                   # [n, 63]
                        acc[m][k][0] += float(p.float().sum(-1).mean()); acc[m][k][1] += float((p & ex).float().sum(-1).mean())
        out = {'units': units, 'sq_err_ratio_to_copy': {k: {'mob_cells': round(sq[k][0] / max(sq['copy'][0], 1e-9), 4),
                                                             'mob_free_cells': round(sq[k][1] / max(sq['copy'][1], 1e-9), 4)} for k in ('expected', 'sample')}}
        for m in th:
            out[m] = {'exact_per_frame': round(exact_n[m] / units, 4)}
            for k, (n, hit, _) in acc[m].items():
                out[m][k] = {'per_frame': round(n / units, 4), 'location_precision': round(hit / max(n, 1e-9), 4),
                             'location_recall': round(hit / max(exact_n[m], 1), 4)}
        res[name] = out
        print(json.dumps({name: out}), flush=True)
        del enc, pred
    (EV.OUT / f"census_loc_{'_'.join(res)}.json").write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
