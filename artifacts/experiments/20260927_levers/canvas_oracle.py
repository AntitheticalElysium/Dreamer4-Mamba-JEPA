"""Post-hoc causal substitution: frozen fcanvas with oracle true view shifts.

The only intervention is replacing tworld.estimate's shift indices for each
input window with shifts computed from the corresponding TRUE token frames.
Both arms use identical checkpoint, actions, initial context and self-fed
generated tokens. This reveals how much mistaken world-frame alignment
contributes to its rollout error; the oracle is not deployable.
"""
import hashlib
import json
from pathlib import Path

import torch

import teval as T
import tworld as TW
from scroll import estimate

HERE = Path(__file__).parent
WORLD = Path("artifacts/eda/levers_tworlds_v1/corrt_raw_suffix_s7_fcanvas.pt")


@torch.no_grad()
def main():
    from d4mj.config import config_from_dict
    import spatial as S

    device = torch.device("cuda")
    meta, train_roots, _ = T.split()
    ix = torch.where(~train_roots)[0]
    cache = T.build_cache("raw", device)
    world, payload = T.load_world(WORLD, device)
    assert payload["script_sha256"] == hashlib.sha256(Path(TW.__file__).read_bytes()).hexdigest()
    config = config_from_dict(torch.load(S.CHECKPOINT, map_location="cpu", weights_only=False)["config"])
    predicted = {arm: torch.empty(len(ix), T.H, 81, 192, dtype=torch.float16)
                 for arm in ("native", "true_scroll")}
    for i in range(0, len(ix), 8):
        sl = ix[i:i + 8]
        ctx = cache["ctx"][sl].float()
        ca, fa = cache["ctx_a"][sl], cache["fut_a"][sl]
        true_all = torch.cat([ctx, cache["fut"][sl].float()], 1)
        true_shifts = estimate(true_all[:, :-1], true_all[:, 1:])
        for arm in predicted:
            frames = [ctx[:, j] for j in range(4)]
            actions = [ca[:, j] for j in range(3)]
            for k in range(T.H):
                window = 4 if k == 0 else 5
                start = len(frames) - window
                a = torch.stack(actions[-(window - 1):] + [fa[:, k]], 1)
                if arm == "true_scroll":
                    original = TW.estimate
                    def oracle(_before, _after):
                        return true_shifts[:, start:start + window - 1].to(_before.device)
                    TW.estimate = oracle
                    try:
                        nxt = T.step(world, torch.stack(frames[-window:], 1), a, device, config)
                    finally:
                        TW.estimate = original
                else:
                    nxt = T.step(world, torch.stack(frames[-window:], 1), a, device, config)
                predicted[arm][i:i + len(sl), k] = nxt.half()
                frames.append(nxt)
                actions.append(fa[:, k])
    fut = cache["fut"][ix].float()
    alive = ~meta["future_dead"][ix, 0].cumsum(1).bool()
    all_fut = cache["fut"].float()
    V = float(((all_fut - all_fut.flatten(0, 1).mean(0)) ** 2).sum((-1, -2)).mean())
    errors = {arm: ((pred.float() - fut) ** 2).sum((-1, -2))
              for arm, pred in predicted.items()}
    out = {"checkpoint_sha256": hashlib.sha256(WORLD.read_bytes()).hexdigest(),
           "model_source_sha256": payload["script_sha256"],
           "n_test_roots": len(ix), "walk_seeds": int(meta["seed"][ix].unique().numel()),
           "normalizer_V": V, "depth": {}}
    for k in (1, 4, 8, 16):
        m = alive[:, k - 1]
        a = float(errors["native"][m, k - 1].mean() / V)
        b = float(errors["true_scroll"][m, k - 1].mean() / V)
        out["depth"][k] = {"n": int(m.sum()), "native": a, "true_scroll": b,
                           "difference": b - a, "relative_change": (b - a) / a}
    seeds = meta["seed"][ix]
    groups = [torch.where(seeds == seed)[0] for seed in seeds.unique()]
    g = torch.Generator().manual_seed(20260929)
    d = []
    for _ in range(2000):
        chosen = torch.randint(len(groups), (len(groups),), generator=g)
        rows = torch.cat([groups[j] for j in chosen])
        m = alive[rows, 15]
        a = float(errors["native"][rows, 15][m].mean() / V)
        b = float(errors["true_scroll"][rows, 15][m].mean() / V)
        d.append(b - a)
    out["depth16_difference_ci95"] = [float(x) for x in torch.tensor(d).quantile(torch.tensor([.025, .975]))]
    torch.save({"native_err": errors["native"], "true_scroll_err": errors["true_scroll"],
                "alive": alive, "seed": seeds, "root_index": ix, "V": V},
               HERE / "canvas_oracle_per_root.pt")
    (HERE / "canvas_oracle.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2), flush=True)


if __name__ == "__main__":
    main()
