"""Matched HUD-feedback intervention for factored attention and raster-scan Mamba.

Both frozen worlds roll out the factual action sequence. After each generated
frame, substitute the true 18 HUD tokens only before feeding it back. Score the
unswapped prediction at depth 16 against the same real successor, with paired
walk-seed uncertainty. Post-hoc on the inspected diagnosis panel; not a gate.
"""
import hashlib
import json
from pathlib import Path

import torch

import teval as T
import tworld as TW

HERE = Path(__file__).parent
WORLDS = {name: Path(f"artifacts/eda/levers_tworlds_v1/corrt_raw_suffix_s7_{name}.pt")
          for name in ("fattn", "fscan")}


@torch.no_grad()
def main():
    from d4mj.config import config_from_dict
    import spatial as S

    device = torch.device("cuda")
    meta, _, _ = T.split()
    cache = T.build_cache("raw", device)
    R = len(meta["seed"])
    config = config_from_dict(torch.load(S.CHECKPOINT, map_location="cpu", weights_only=False)["config"])
    source_hash = hashlib.sha256(Path(TW.__file__).read_bytes()).hexdigest()
    native, hud, hashes = {}, {}, {}
    for name, path in WORLDS.items():
        world, payload = T.load_world(path, device)
        assert payload["script_sha256"] == source_hash
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        old = torch.load(HERE / "evals" / f"{payload['name']}_per_root.pt", weights_only=False)
        assert torch.equal(old["seed"], meta["seed"])
        native[name] = old["gen_err"]
        hud[name] = torch.empty(R, T.H)
        for start in range(0, R, 16):
            stop = min(start + 16, R)
            ctx = cache["ctx"][start:stop].float()
            ca, fa = cache["ctx_a"][start:stop], cache["fut_a"][start:stop]
            true_f = cache["fut"][start:stop].float()
            frames = [ctx[:, j] for j in range(4)]
            actions = [ca[:, j] for j in range(3)]
            for k in range(T.H):
                width = 4 if k == 0 else 5
                act = torch.stack(actions[-(width - 1):] + [fa[:, k]], 1)
                g = T.step(world, torch.stack(frames[-width:], 1), act, device, config)
                hud[name][start:stop, k] = ((g - true_f[:, k]) ** 2).sum((-1, -2))
                g[:, 63:] = true_f[:, k, 63:]
                frames.append(g)
                actions.append(fa[:, k])
        del world
        torch.cuda.empty_cache()
    alive = ~meta["future_dead"][:, 0].cumsum(1).bool()
    m = alive[:, 15]
    seeds = meta["seed"]
    groups = [torch.where(seeds == seed)[0] for seed in seeds.unique()]
    V = float(old["V"])
    def stats(rows):
        mask = m[rows]
        per = {name: {"native": float(native[name][rows, 15][mask].mean() / V),
                      "hud": float(hud[name][rows, 15][mask].mean() / V)} for name in WORLDS}
        effects = {name: per[name]["hud"] - per[name]["native"] for name in WORLDS}
        gaps = {kind: per["fscan"][kind] - per["fattn"][kind] for kind in ("native", "hud")}
        return per, effects, gaps, effects["fscan"] - effects["fattn"]
    per, effects, gaps, diff_in_diff = stats(torch.arange(R))
    gen = torch.Generator().manual_seed(20260929)
    draws = []
    for _ in range(2000):
        rows = torch.cat([groups[j] for j in torch.randint(len(groups), (len(groups),), generator=gen)])
        draws.append(stats(rows)[3])
    interval = [float(x) for x in torch.tensor(draws).quantile(torch.tensor([.025, .975]))]
    out = {"checkpoint_sha256": hashes, "model_source_sha256": source_hash,
           "roots": R, "alive_depth16": int(m.sum()), "walk_seeds": len(groups),
           "normalizer_V": V, "per_world": per, "hud_minus_native": effects,
           "scan_minus_attention": gaps, "difference_in_hud_effect": diff_in_diff,
           "difference_in_effect_ci95": interval}
    torch.save({"native_err": native, "hud_err": hud, "alive": alive,
                "seed": seeds, "V": V}, HERE / "fscan_hud_feedback_per_root.pt")
    (HERE / "fscan_hud_feedback.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2), flush=True)


if __name__ == "__main__":
    main()
