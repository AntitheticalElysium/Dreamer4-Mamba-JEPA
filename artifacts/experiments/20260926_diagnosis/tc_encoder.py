"""D23. Does TC-LeWM's encoder keep what Raw's z drops? Same measurements, both joint step-10000 encoders.

TC-LeWM (arXiv 2607.26924): Raw LeWM "biases variance allocation toward the temporally persistent component";
TC applies SIGReg to temporally centered residuals. A zombie's position is non-persistent, so TC predicts more
of it in z. Measured on z (projected CLS), frozen, eval mode:
  twins      300 twin states (twins.py): Fisher d' of each one-tile edit, adjacent-vs-far zombie cosine and
             held-out AUC, zombie present-vs-absent in the same scene (held-out AUC)
  natural    roots of the four opened blocks (55k-58k): ridge probes fitted on 55k+56k, read on 57k+58k:
             zombie adjacent, zombie anywhere in view, cow anywhere, lava adjacent, each move's passability
             (the target tile walkable and free; lava excluded), health (R^2), per-tile content accuracy
  temporal   share of z's variance that is within-window (4 consecutive frames) residual, on corpus windows
  tokens     the same natural probes on each encoder's PATCH TOKENS (frame grid 9x9; view tile (r, c) is token
             9r + c; player token 31): passability of each move from the target neighbour's token, zombie
             adjacent from the 4 neighbour tokens, per-cell tile class from the cell's own token (one shared probe),
             health from the HUD health token (63), zombie / cow in view from the mean token; and the
             within-window share of token variance
"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
import torch

HERE = Path(__file__).parent
ROOT = Path("/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA")
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "artifacts/experiments/20260921_readout_ladder"))
import twins as TW  # noqa: E402
from choices import move_table  # noqa: E402
from compound import auc  # noqa: E402
from frozen_ladder import strata  # noqa: E402

RUNS = {"raw": ROOT / "artifacts/lewm_m4_canonical/raw/joint/step-010000.pt",
        "tc": ROOT / "artifacts/lewm_m4_canonical/tc/joint/step-010000.pt"}
BLOCKS = ("observe_fresh_v6", "observe_fresh_v7", "observe_fresh_v8", "observe_fresh_v9")


def encoder(path, device):
    from d4mj.checkpoint import read_lewm_bundle
    from d4mj.config import config_from_dict
    from d4mj.world_api import ModelBundle
    payload = read_lewm_bundle(path)
    bundle = ModelBundle.create(config_from_dict(payload["config"]))
    bundle.encoder.load_state_dict(payload["modules"]["encoder"])
    return bundle.encoder.to(device).eval()


@torch.no_grad()
def z_of(enc, frames, device, batch=64):
    return torch.cat([enc._hidden(frames[i:i + batch, None].to(device))[0].float().cpu() for i in range(0, len(frames), batch)])


def ridge_scores(x, y, tr, te, lams=(1e-3, 1e-2, 1e-1, 1, 10)):
    mu, sd = x[tr].mean(0), x[tr].std(0).clamp_min(1e-6)
    xs = torch.cat([((x - mu) / sd).double(), torch.ones(len(x), 1, dtype=torch.float64)], 1)
    idx = torch.where(tr)[0]
    a, b = idx[: len(idx) // 2], idx[len(idx) // 2:]
    best = None
    for lam in lams:
        w = torch.linalg.solve(xs[a].T @ xs[a] + lam * len(a) * torch.eye(xs.shape[1], dtype=xs.dtype), xs[a].T @ y[a].double())
        err = float(((xs[b] @ w - y[b].double()) ** 2).mean())
        if best is None or err < best[0]:
            best = (err, lam)
    w = torch.linalg.solve(xs[tr].T @ xs[tr] + best[1] * int(tr.sum()) * torch.eye(xs.shape[1], dtype=xs.dtype), xs[tr].T @ y[tr].double())
    return xs[te] @ w


@torch.no_grad()
def tokens_of(enc, frames, device, batch=64):
    return torch.cat([enc._hidden(frames[i:i + batch, None].to(device))[2].float().cpu() for i in range(0, len(frames), batch)])


def natural_tokens(enc, device):
    rows, block = [], []
    for i, s in enumerate(BLOCKS):
        r = [r for f in sorted((ROOT / "artifacts/eda" / s).glob("seed-*.pt")) for r in torch.load(f, weights_only=False)]
        rows += r; block += [i] * len(r)
    block = torch.tensor(block)
    tr, te = block <= 1, block >= 2
    vis = torch.stack([r["visible"].float() for r in rows])
    tok = tokens_of(enc, torch.stack([r["frames"][-1] for r in rows]), device)          # [n, 81, 192]
    st = strata(vis)
    mobs = vis[:, 1071:1512].reshape(-1, 7, 9, 7)
    cat, _ = move_table(vis)
    out = {}
    nb = {1: 30, 2: 32, 3: 22, 4: 40}
    pas = []
    for a, k in nb.items():
        m = cat[:, a] <= 1
        y = (cat[:, a] == 0).float()
        pas.append(auc(ridge_scores(tok[m, k], y[m], tr[m], te[m]), y[m][te[m]]))
    out["passability_mean_4_directions"] = sum(pas) / 4
    four = tok[:, [22, 40, 30, 32]].flatten(1)
    out["zombie_adjacent"] = auc(ridge_scores(four, st["zombie_adjacent"].float(), tr, te), st["zombie_adjacent"][te])
    mean = tok.mean(1)
    for name, y in (("zombie_in_view", mobs[..., 0].sum((1, 2)) > 0), ("cow_in_view", mobs[..., 1].sum((1, 2)) > 0)):
        out[name] = auc(ridge_scores(mean, y.float(), tr, te), y[te])
    h = vis[:, 1512]
    p = ridge_scores(tok[:, 63], h, tr, te).float()
    out["health_r2"] = float(1 - ((p - h[te]) ** 2).sum() / ((h[te] - h[te].mean()) ** 2).sum())
    t = vis[:, :1071].reshape(-1, 7, 9, 17).argmax(-1).flatten(1)                          # [n, 63]
    cells = tok[:, :63]
    trc, tec = tr[:, None].expand(-1, 63).flatten(), te[:, None].expand(-1, 63).flatten()
    pred = ridge_scores(cells.flatten(0, 1), torch.nn.functional.one_hot(t.flatten(), 17).float(), trc, tec)
    out["tile_accuracy"] = float((pred.argmax(-1) == t.flatten()[tec]).float().mean())
    return out


def natural(enc, device):
    rows, block = [], []
    for i, s in enumerate(BLOCKS):
        r = [r for f in sorted((ROOT / "artifacts/eda" / s).glob("seed-*.pt")) for r in torch.load(f, weights_only=False)]
        rows += r; block += [i] * len(r)
    block = torch.tensor(block)
    tr, te = block <= 1, block >= 2
    vis = torch.stack([r["visible"].float() for r in rows])
    z = z_of(enc, torch.stack([r["frames"][-1] for r in rows]), device)
    st = strata(vis)
    mobs = vis[:, 1071:1512].reshape(-1, 7, 9, 7)
    tiles = vis[:, :1071].reshape(-1, 7, 9, 17)
    cat, _ = move_table(vis)
    out = {}
    for name, y in (("zombie_adjacent", st["zombie_adjacent"]), ("zombie_in_view", mobs[..., 0].sum((1, 2)) > 0),
                    ("cow_in_view", mobs[..., 1].sum((1, 2)) > 0), ("lava_adjacent", st["lava_adjacent"])):
        out[name] = auc(ridge_scores(z, y.float(), tr, te), y[te])
    pas = []
    for a in (1, 2, 3, 4):
        m = cat[:, a] <= 1
        y = (cat[:, a] == 0).float()
        s = ridge_scores(z[m], y[m], tr[m], te[m])
        pas.append(auc(s, y[m][te[m]]))
    out["passability_mean_4_directions"] = sum(pas) / 4
    h = vis[:, 1512]
    p = ridge_scores(z, h, tr, te).float()
    out["health_r2"] = float(1 - ((p - h[te]) ** 2).sum() / ((h[te] - h[te].mean()) ** 2).sum())
    t = tiles.argmax(-1).flatten(1)                                        # [n, 63] tile class per cell
    onehot = torch.nn.functional.one_hot(t, 17).float().flatten(1)
    pred = ridge_scores(z, onehot, tr, te).view(-1, 63, 17)
    majority = torch.nn.functional.one_hot(t[tr], 17).sum(0).argmax(-1)
    out["tile_accuracy"] = float((pred.argmax(-1) == t[te]).float().mean())
    out["tile_majority_baseline"] = float((majority[None] == t[te]).float().mean())
    return out


@torch.no_grad()
def temporal(enc, device):
    import bnmode
    bnmode.BATCHES, bnmode.B, bnmode.T = 8, 64, 4
    frames, _ = bnmode.windows(0)
    out = {}
    hidden = [enc._hidden(frames[i:i + 32].to(device)) for i in range(0, len(frames), 32)]
    z = torch.cat([h[0].float().cpu().view(-1, bnmode.T, 192) for h in hidden])
    tok = torch.cat([h[2].float().cpu().view(-1, bnmode.T, 81, 192) for h in hidden])
    for name, x in (("z", z), ("tokens", tok.flatten(2))):
        total = float(((x - x.flatten(0, 1).mean(0)) ** 2).sum(-1).mean())
        within = float(((x - x.mean(1, keepdim=True)) ** 2).sum(-1).mean())
        out[f"{name}_within_window_share_of_variance"] = within / total
    return out


def main():
    device = torch.device("cuda")
    states = TW.base_states()
    frames = TW.render_all(states)
    nat = TW.natural_frames()
    result = {}
    for run, path in RUNS.items():
        enc = encoder(path, device)
        nz = z_of(enc, nat, device)
        Wz = TW.inverse_sqrt_cov(nz)
        Z = {k: z_of(enc, f, device).double() for k, f in frames.items()}
        tw = {}
        for light in ("day", "night"):
            base = Z[(light, "base")]
            for e in TW.EDITS:
                d = (Z[(light, e)] - base)
                tw[f"{light}/{e}"] = float((Wz @ d.mean(0)).norm())
            dn, df = Z[(light, "zombie")] - base, Z[(light, "zombie_far")] - base
            half = len(dn) // 2
            y = torch.cat([torch.ones(len(dn) - half), torch.zeros(len(dn) - half)])
            w = Wz @ Wz @ (Z[(light, "zombie")][:half].mean(0) - Z[(light, "zombie_far")][:half].mean(0))
            w2 = Wz @ Wz @ dn[:half].mean(0)
            tw[f"{light}/cos_adjacent_vs_far"] = float(torch.nn.functional.cosine_similarity(dn.mean(0), df.mean(0), dim=0))
            tw[f"{light}/auc_adjacent_vs_far"] = auc(torch.cat([Z[(light, "zombie")][half:] @ w, Z[(light, "zombie_far")][half:] @ w]), y)
            tw[f"{light}/auc_zombie_present"] = auc(torch.cat([Z[(light, "zombie")][half:] @ w2, base[half:] @ w2]), y)
        result[run] = {"twins": tw, "natural": natural(enc, device), "tokens": natural_tokens(enc, device),
                       "temporal": temporal(enc, device)}
        print(run, json.dumps({k: {kk: round(vv, 3) for kk, vv in v.items()} for k, v in result[run].items()}), flush=True)
        del enc
        torch.cuda.empty_cache()
    (HERE / "tc_encoder.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
