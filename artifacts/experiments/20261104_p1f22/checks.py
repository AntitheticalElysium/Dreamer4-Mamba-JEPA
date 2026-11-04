"""Bounded actual-path preflight for P1-F2.2 (F2.1's checks.run), plus the arm's own: the periphery's first layer reads
2 x 2 x 3 inputs, and the chooser's coarse path (chooser.coarse) equals the world's periphery tokens on real frames,
so the head reads exactly what the world sees. No efficacy reading, no launch."""
import importlib
from pathlib import Path
import torch
from saccade import data as D, model as M, train as TR

K = importlib.import_module('artifacts.experiments.20261104_p1f21.checks')
CH = importlib.import_module('artifacts.experiments.20261103_p1f2.chooser')


def acuity(cfg, out):
    torch.manual_seed(0); model = M.M0(cfg).cuda().eval()
    ep = D.load_split(cfg)['dev'][0]
    c = D.gather([ep], [[(0, 1, t, 1 + t % 7, 1 + (t // 7) % 7) for t in range(min(ep.T + 1, 256))]])
    frames, fix = (torch.as_tensor(c[k], device='cuda') for k in ('frames', 'fixation'))
    with torch.no_grad(), TR.autocast(frames.device, cfg):
        world, (chooser, _) = model.perception(frames, fix).p, CH.coarse(model.perception, frames)
    res = {'periphery_first_layer_in': model.perception.p_layers[0].in_features,
           'chooser_coarse_equals_world_periphery': bool(torch.equal(world, chooser)), 'frames_compared': int(frames.shape[1])}
    assert res['periphery_first_layer_in'] == 3 * 2 ** 2 and res['chooser_coarse_equals_world_periphery'], res
    return res


def main():
    L = importlib.import_module('artifacts.experiments.20261104_p1f22.train_p1f22')
    K.run(L, Path(__file__).resolve().parent, 'artifacts.experiments.20261104_p1f22.train_p1f22', extra=acuity)


if __name__ == '__main__':
    main()
