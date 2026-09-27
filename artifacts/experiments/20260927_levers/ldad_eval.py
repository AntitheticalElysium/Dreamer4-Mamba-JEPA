"""E4 evaluation. What Delta-JEPA's LDAD changes in the encoder and in the joint world, against the paired canonical runs.

Runs (all joint step 10,000, same init / batches / projections): raw, tc (canonical) and raw_lam10, tc_lam10 (LDAD).
  encoder   tc_encoder.py's suite on z and patch tokens: within-4-frame-window variance share; Fisher d' of one-tile
            twin edits (mobs vs static terrain) and adjacent-vs-far zombie; natural-root ridge probes (zombie
            adjacent / in view, cow, lava adjacent, passability, health, per-cell tile) fitted 55k+56k, read 57k+58k
  action    a fresh ridge probe decoding the executed action (17-way, one-vs-rest) from dz = z_{t+1} - z_t on the
            futures' factual transitions (train seeds -> test seeds): accuracy overall, on moves that succeeded, on
            moves that were blocked, on non-move actions; and the fraction of blocked moves decoded as the attempted
            direction
  world     each run's own joint world under le-wm's 3-frame window: one-step squared error x copy on 4,000
            held-out (DEV/FINAL) transitions by class (insample.classes: moved / blocked / ambiguous / other / sleep
            onset); 16-step imagined error / V on the futures (factual actions), copy-root baseline
"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
import torch

HERE = Path(__file__).parent
ROOT = Path("/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA")
DIAG = ROOT / "artifacts/experiments/20260926_diagnosis"
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(DIAG))
sys.path.insert(0, str(ROOT / "artifacts/experiments/20260921_readout_ladder"))

RUNS = {"raw": ROOT / "artifacts/lewm_m4_canonical/raw/joint/step-010000.pt",
        "tc": ROOT / "artifacts/lewm_m4_canonical/tc/joint/step-010000.pt",
        "raw_lam10": ROOT / "artifacts/eda/levers_ldad_v1/raw_lam10/step-010000.pt",
        "tc_lam10": ROOT / "artifacts/eda/levers_ldad_v1/tc_lam10/step-010000.pt",
        "raw_lam1": ROOT / "artifacts/eda/levers_ldad_v1/raw_lam1/step-010000.pt",
        "raw_lam0.1": ROOT / "artifacts/eda/levers_ldad_v1/raw_lam0.1/step-010000.pt",
        "tc_lam1": ROOT / "artifacts/eda/levers_ldad_v1/tc_lam1/step-010000.pt",
        "tc_lam10": ROOT / "artifacts/eda/levers_ldad_v1/tc_lam10/step-010000.pt"}


def load(path, device):
    from d4mj.config import config_from_dict
    from d4mj.world_api import ModelBundle
    p = torch.load(path, map_location="cpu", weights_only=False)
    bundle = ModelBundle.create(config_from_dict(p["config"]))
    bundle.encoder.load_state_dict(p["modules"]["encoder"])
    bundle.world.load_state_dict(p["modules"]["world"])
    bundle.encoder.to(device).freeze()
    bundle.world.to(device).eval()
    return bundle


@torch.no_grad()
def action_probe(enc, device):
    from coverage import classify
    from rollouts import load_roots
    from teval import split
    meta, train_roots, _ = split()
    roots = load_roots()
    frames = torch.stack([torch.cat([r["context"][-1:], r["future_frames"][0]]) for r in roots])     # [R,17,...]
    z = torch.cat([enc._hidden(frames[i:i + 64].flatten(0, 1)[:, None].to(device))[0].float().cpu()
                   for i in range(0, len(frames), 64)]).view(len(roots), 17, -1)
    dz = (z[:, 1:] - z[:, :-1]).flatten(0, 1)
    acts = torch.stack([r["future_actions"] for r in roots]).flatten()
    before, after = frames[:, :-1].flatten(0, 1), frames[:, 1:].flatten(0, 1)
    scroll = classify(before, after, acts)                     # +1 moved, -1 blocked, 0 ambiguous (moves only)
    tr = train_roots[:, None].expand(-1, 16).flatten()
    te = ~tr
    y = torch.nn.functional.one_hot(acts, 17).double() * 2 - 1
    mu, sd = dz[tr].mean(0), dz[tr].std(0).clamp_min(1e-6)
    x = torch.cat([((dz - mu) / sd).double(), torch.ones(len(dz), 1, dtype=torch.float64)], 1)
    w = torch.linalg.solve(x[tr].T @ x[tr] + 1.0 * int(tr.sum()) * torch.eye(x.shape[1], dtype=x.dtype), x[tr].T @ y[tr])
    pred = (x @ w).argmax(-1)
    move = (acts >= 1) & (acts <= 4)
    ok = pred == acts
    return {"acc_all": float(ok[te].float().mean()),
            "acc_move_succeeded": float(ok[te & move & (scroll == 1)].float().mean()),
            "acc_move_blocked": float(ok[te & move & (scroll == -1)].float().mean()),
            "n_test_moved_blocked": [int((te & move & (scroll == 1)).sum()), int((te & move & (scroll == -1)).sum())],
            "acc_non_move": float(ok[te & ~move].float().mean()),
            "chance_majority": float((acts[te] == acts[tr].mode().values).float().mean())}


@torch.no_grad()
def world_eval(bundle, device):
    from insample import classes, sample
    from d4mj.train import autocast_context
    enc, world, config = bundle.encoder, bundle.world, bundle.config
    d = sample({"dev", "final"}, 4000, 1)
    cls = classes(d)
    zf = lambda f: torch.cat([enc(f[i:i + 64][:, None].to(device)).float()[:, 0, 0].cpu() for i in range(0, len(f), 64)])
    ctx = torch.stack([zf(d["frames"][:, j]) for j in (1, 2, 3)], 1)                                 # window of 3
    nxt = zf(d["next"])
    preds = []
    with autocast_context(config):
        for i in range(0, len(ctx), 256):
            c, a = ctx[i:i + 256].to(device), d["ctx_a"][i:i + 256, 1:].to(device)
            st = world.teacher(c[:, :, None], a).state
            adv, _ = world.advance(st, d["a"][i:i + 256, None].to(device))
            preds.append(adv.latent[:, 0, 0].float().cpu())
    pred = torch.cat(preds)
    err, cp = ((pred - nxt) ** 2).sum(-1), ((ctx[:, -1] - nxt) ** 2).sum(-1)
    out = {"onestep_window3_x_copy": {"all": float(err.mean() / cp.mean())}}
    for c in sorted(set(cls)):
        m = torch.tensor([x == c for x in cls])
        if m.sum() >= 30:
            out["onestep_window3_x_copy"][c] = float(err[m].mean() / cp[m].mean())
    # 16-step imagination on the futures, window 3
    from rollouts import load_roots
    roots = load_roots()
    meta = torch.load(ROOT / "artifacts/eda/diagnosis_rollouts_v1/meta.pt")
    alive = ~meta["future_dead"][:, 0].cumsum(1).bool()
    c3 = torch.stack([zf(torch.stack([r["context"][j] for r in roots])) for j in (1, 2, 3)], 1)
    true = torch.stack([zf(torch.stack([r["future_frames"][0, k] for r in roots])) for k in range(16)], 1)
    ca = torch.stack([r["context_actions"] for r in roots])
    fa = torch.stack([r["future_actions"] for r in roots])
    V = float(((true - true.flatten(0, 1).mean(0)) ** 2).sum(-1).mean())
    gen = []
    with autocast_context(config):
        for i in range(0, len(roots), 128):
            lat = [c3[i:i + 128, j].to(device) for j in range(3)]
            acts = [ca[i:i + 128, 1].to(device), ca[i:i + 128, 2].to(device)]
            g = []
            for k in range(16):
                a = fa[i:i + 128, k].to(device)
                st = world.teacher(torch.stack(lat[-3:], 1)[:, :, None], torch.stack(acts[-2:], 1)).state
                nx, _ = world.advance(st, a[:, None])
                q = nx.latent[:, 0, 0].float()
                g.append(q.cpu()); lat.append(q.to(lat[0].dtype)); acts.append(a)
            gen.append(torch.stack(g, 1))
    gen = torch.cat(gen)
    out["imagine16_window3_over_V"] = [float(((gen[:, k] - true[:, k]) ** 2).sum(-1)[alive[:, k]].mean() / V) for k in (0, 3, 7, 15)]
    out["copy_root_over_V"] = [float(((c3[:, -1] - true[:, k]) ** 2).sum(-1)[alive[:, k]].mean() / V) for k in (0, 3, 7, 15)]
    return out


SCREENS = {"raw@2000": ROOT / "artifacts/lewm_m4_canonical/raw/joint/step-002000.pt",
           "tc@2000": ROOT / "artifacts/lewm_m4_canonical/tc/joint/step-002000.pt",
           "raw_lam1@2000": ROOT / "artifacts/eda/levers_ldad_v1/raw_lam1_screen2000/step-002000.pt",
           "raw_lam0.1@2000": ROOT / "artifacts/eda/levers_ldad_v1/raw_lam0.1_screen2000/step-002000.pt",
           "raw_lam10@2000": ROOT / "artifacts/eda/levers_ldad_v1/raw_lam10/step-002000.pt",
           "tc_lam1@2000": ROOT / "artifacts/eda/levers_ldad_v1/tc_lam1_screen2000/step-002000.pt",
           "tc_lam10@2000": ROOT / "artifacts/eda/levers_ldad_v1/tc_lam10_screen2000/step-002000.pt"}


def main():
    import tc_encoder as TE
    if "--screens" in sys.argv:
        RUNS.clear(); RUNS.update(SCREENS)
    import twins as TW
    device = torch.device("cuda")
    states = TW.base_states()
    frames = TW.render_all(states)
    nat = TW.natural_frames()
    out_path = HERE / ("ldad_screens.json" if "--screens" in sys.argv else "ldad_eval.json")
    result = json.loads(out_path.read_text()) if out_path.exists() else {}
    for run, path in RUNS.items():
        if run in result or not path.exists():
            continue
        bundle = load(path, device)
        enc = bundle.encoder
        nz = TE.z_of(enc, nat, device)
        Wz = TW.inverse_sqrt_cov(nz)
        Z = {k: TE.z_of(enc, f, device).double() for k, f in frames.items()}
        tw = {}
        for light in ("day", "night"):
            base = Z[(light, "base")]
            for e in TW.EDITS:
                tw[f"{light}/{e}"] = float((Wz @ (Z[(light, e)] - base).mean(0)).norm())
            dn, df = Z[(light, "zombie")] - base, Z[(light, "zombie_far")] - base
            tw[f"{light}/cos_adjacent_vs_far"] = float(torch.nn.functional.cosine_similarity(dn.mean(0), df.mean(0), dim=0))
        result[run] = {"twins_z": tw, "natural_z": TE.natural(enc, device), "natural_tokens": TE.natural_tokens(enc, device),
                       "temporal": TE.temporal(enc, device), "action_from_dz": action_probe(enc, device),
                       "world": world_eval(bundle, device)}
        print(run, json.dumps({k: v for k, v in result[run].items() if k != "twins_z"}), flush=True)
        out_path.write_text(json.dumps(result, indent=2) + "\n")
        del bundle
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
