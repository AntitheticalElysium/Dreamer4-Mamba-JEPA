"""Faithful decoder routes (PLAN.md checks 4-5), one guided P1 chunk at initialisation (world seed 1):
4. Gradient routes: backpropagating the `noise` term alone reaches the pixel decoder's scale network (and its cell
   embedding) and nothing else (not the mean head, the TSP / reconstruction adapters, the encoders including the
   lighting latent); the faithful tsp / rec terms never reach the scale network.
5. Encoder gradient per loss (saccade.train.encoder_grad_norms) under faithful vs nll on the same chunk and weights:
   how much the grounding routes' pull on the encoders changes at initialisation (the harmoniser re-weights G2 later;
   the run logs the same norms at every checkpoint).
Usage (project venv, GPU): python -m artifacts.experiments.20261030_p1_guided.check_faithful
"""
import copy, functools, importlib, json, os
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np
import torch

from saccade import data as D, model as M, train as TR

FT2 = importlib.import_module("artifacts.experiments.20261030_fovea_placement.finetune")
M05 = importlib.import_module("artifacts.experiments.20261030_m05_guided_fovea.finetune")
HERE = Path(__file__).resolve().parent


def main():
    dev = torch.device("cuda")
    base = TR.load_cfg()
    split = D.load_split(base)
    offsets = D.frame_offsets(split["train"])
    targets = np.load(TR.ROOT / base["p1"]["out_dir"] / "targets/change_targets.npy", mmap_mode="r")
    d, P = base["data"], base["data"]["context_prefix"]
    streams = FT2.GuidedStreams(split["train"], d["streams"], P, seed=1, targets_offsets=offsets, place=np.load(M05.PLACE))
    chunk = D.select(streams.next(64), np.arange(2))
    res = {}
    torch.manual_seed(1)
    model = M.M0(base).to(dev)
    TR.set_reward_values(model, split["train"])
    for mode in ("faithful", "nll"):
        cfg = copy.deepcopy(base)
        cfg["heads"]["pixel_decoder"]["mean_loss"] = mode
        carry = M.carry_rows(model.belief.initial(d["streams"], dev), [0, 1])
        cache = torch.zeros(2, P + 1, base["belief"]["width"], device=dev)
        with torch.random.fork_rng(devices=[0]):
            torch.manual_seed(7)
            terms, _, _ = TR.p1_losses(model, chunk, carry, cache, targets, cfg)
        res[mode] = {"terms": {k: float(v) for k, v in terms.items()}, "encoder_grad_norms": TR.encoder_grad_norms(model, terms)}
        if mode == "faithful":
            groups = {"pixels.scale": list(model.pixels.scale.parameters()) + list(model.pixels.cell.parameters()),
                      "pixels.mean": list(model.pixels.mean.parameters()), "tsp": list(model.tsp.parameters()),
                      "perception": list(model.perception.parameters()), "belief": list(model.belief.parameters()),
                      "denoiser": list(model.denoiser.parameters())}
            routes = {}
            for term in ("noise", "tsp", "rec"):
                gs = torch.autograd.grad(terms[term], [p for ps in groups.values() for p in ps], retain_graph=True, allow_unused=True)
                i, routes[term] = 0, {}
                for name, ps in groups.items():
                    sq = [g.float().square().sum() for g in gs[i:i + len(ps)] if g is not None]
                    routes[term][name] = float(torch.stack(sq).sum().sqrt()) if sq else 0.0
                    i += len(ps)
            res["faithful"]["routes"] = routes
    ok = res["faithful"]["routes"]
    res["pass"] = {"noise_only_scale": ok["noise"]["pixels.scale"] > 0 and all(ok["noise"][k] == 0 for k in ok["noise"] if k != "pixels.scale"),
                   "mse_never_scale": ok["tsp"]["pixels.scale"] == 0 and ok["rec"]["pixels.scale"] == 0,
                   "noise_not_in_encoder_norms": res["faithful"]["encoder_grad_norms"].get("noise", 0.0) == 0.0}
    (HERE / "check_faithful.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    assert all(res["pass"].values())


if __name__ == "__main__":
    main()
