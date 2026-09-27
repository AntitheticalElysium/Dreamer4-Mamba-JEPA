"""E1. Option (b): does training the Mamba world at the context it is deployed at fix it, and does it USE the memory?

Diagnosis (TRANSITION.md Part 1d/4): the canonical joint world (trained on 4-frame windows, as le-wm's
history_size 3 + num_preds 1) beats copying only inside its window and degrades past it; under le-wm's own
3-frame truncation it is the best world measured. Chen et al. 2025 (Stuffed Mamba): Mamba trained on contexts too
short for its state does not learn to forget; the minimum length grows with state size.

Controlled test on the frozen canonical encoder's cached eval-mode latents (`raw/cache`, TRAIN split):
fresh canonical LeWMWorld (joint config: Mamba-2, 6 layers, width 256, d_state 64), the joint prediction loss
(teacher-forced next-latent MSE; no SIGReg -- the encoder is frozen), joint optimizer / schedule (AdamW 5e-5,
wd 1e-3, clip 1, 500 warmup, cosine to 5e-6, 10,000 updates), predictor BN in train mode (as joint), bf16.
Arms differ only in the training window length L; batch B set so B x (L-1) ~ 384 predicted transitions per
update (the joint's 128 x 3): L=4 B=128, L=16 B=26, L=64 B=6. Same init seed (7) and window sampler seed.

Evaluation on DEV/FINAL episodes (never trained on), eval mode:
  position   teacher-forced error / copy-previous error by position, 256-frame windows (128 of them)
  memory     the same positions with le-wm's 3-frame window instead of the full history:
             benefit = 1 - err(full) / err(window3); > 0 = the history helps
  revisit    transitions whose next latent is closer to a latent 4+ frames back than to the current one
             (the view returns to something seen before): the same benefit on that subset
  imagine    16 imagined steps after a 48-frame teacher context, recurrent vs window-3; error / variance
"""
import json
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).parent
ROOT = Path("/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA")
sys.path.insert(0, str(ROOT))

ARMS = {4: 128, 16: 26, 64: 6}
UPDATES = 10_000
OUT = ROOT / "artifacts/eda/levers_context_v1"
BINS = ((1, 3), (4, 7), (8, 15), (16, 31), (32, 63), (64, 127), (128, 255))


def load(device):
    from d4mj.cache import load_latent_cache
    from d4mj.checkpoint import read_lewm_bundle
    from d4mj.config import config_from_dict
    from d4mj.world_api import ModelBundle
    payload = read_lewm_bundle(ROOT / "artifacts/lewm_m4_canonical/raw/joint/step-010000.pt")
    config = config_from_dict(payload["config"])
    bundle = ModelBundle.create(config)
    bundle.encoder.load_state_dict(payload["modules"]["encoder"])
    bundle.encoder.freeze()
    episodes = load_latent_cache(ROOT / "artifacts/lewm_m4_canonical/raw/cache", bundle.encoder, config)
    return config, episodes


class Windows:
    """Uniform over transitions: episode weighted by its number of full windows, start uniform."""

    def __init__(self, episodes, length, seed):
        self.eps = [e for e in episodes if len(e.actions_taken) + 1 >= length]
        self.length = length
        self.counts = torch.tensor([len(e.actions_taken) + 2 - length for e in self.eps], dtype=torch.float64)
        self.gen = torch.Generator().manual_seed(seed)

    def sample(self, batch):
        z, a = [], []
        for i in torch.multinomial(self.counts, batch, replacement=True, generator=self.gen).tolist():
            e = self.eps[i]
            s = int(torch.randint(int(self.counts[i]), (), generator=self.gen))
            z.append(e.latents[s:s + self.length]); a.append(e.actions_taken[s:s + self.length - 1])
        return torch.stack(z), torch.stack(a)


def new_world(config, device):
    from d4mj.lewm import LeWMWorld
    torch.manual_seed(7)
    world = LeWMWorld(config).to(device)
    world.train().requires_grad_(True)
    world.agent_readout.requires_grad_(False)
    return world


def train(config, episodes, length, batch, device, log):
    from d4mj.train import autocast_context, learning_rate, optimizer, optimizer_step
    world = new_world(config, device)
    opt = optimizer([world], config.joint, exclude_vectors=True)
    params = [p for g in opt.param_groups for p in g["params"]]
    sampler = Windows([e for e in episodes if e.split == "train"], length, seed=11)
    history, started = [], time.time()
    for update in range(UPDATES):
        z, a = sampler.sample(batch)
        z, a = z.to(device), a.to(device)
        with autocast_context(config):
            pred = world.teacher(z, a).predicted
        loss = (pred.float() - z[:, 1:].float()).square().mean()
        norm = optimizer_step(opt, loss, params, learning_rate=learning_rate(config, update),
                              grad_clip=config.joint.grad_clip, strict=True, zero_grad=True)
        if (update + 1) % 1000 == 0:
            row = {"update": update + 1, "loss": float(loss), "copy": float((z[:, 1:] - z[:, :-1]).square().mean()),
                   "gradient_norm": float(norm), "seconds": round(time.time() - started, 1)}
            history.append(row)
            log(stage="train", L=length, **row)
    return world.eval(), history


@torch.no_grad()
def window3(world, z, a):
    """Teacher-forced prediction of every position t >= 3 from only z[t-3..t-1] (le-wm's truncation)."""
    n, T = z.shape[:2]
    out = torch.zeros_like(z[:, 1:])
    out[:, :2] = world.teacher(z[:, :3], a[:, :2]).predicted[:, :2]
    for t in range(3, T):
        out[:, t - 1] = _advance_pred(world, z[:, t - 3:t], a[:, t - 3:t])[:, 0]
    return out


def _advance_pred(world, z3, a3):
    st = world.teacher(z3, a3[:, :2]).state
    nxt, _ = world.advance(st, a3[:, 2:3])
    return nxt.latent


@torch.no_grad()
def evaluate(world, config, episodes, device):
    from d4mj.train import autocast_context
    held = [e for e in episodes if e.split in ("dev", "final")]
    long = Windows(held, 256, seed=2)
    z, a = long.sample(128)
    errs = {"full": [], "window3": []}
    with autocast_context(config):
        for i in range(0, len(z), 8):
            zi, ai = z[i:i + 8].to(device), a[i:i + 8].to(device)
            errs["full"].append(((world.teacher(zi, ai).predicted.float() - zi[:, 1:].float()) ** 2).mean(-1)[:, :, 0].cpu())
            errs["window3"].append(((window3(world, zi, ai).float() - zi[:, 1:].float()) ** 2).mean(-1)[:, :, 0].cpu())
    full, w3 = torch.cat(errs["full"]), torch.cat(errs["window3"])          # [n, 255] per predicted position
    copy = ((z[:, 1:] - z[:, :-1]) ** 2).mean(-1)[:, :, 0]
    res = {"position": {}, "memory_benefit": {}}
    for lo, hi in BINS:
        s = slice(lo - 1, hi)
        res["position"][f"{lo}-{hi}"] = {"full_vs_copy": float(full[:, s].mean() / copy[:, s].mean()),
                                         "window3_vs_copy": float(w3[:, s].mean() / copy[:, s].mean())}
        if lo >= 4:
            res["memory_benefit"][f"{lo}-{hi}"] = float(1 - full[:, s].mean() / w3[:, s].mean())
    # revisits: next latent closer to one 4+ frames back than to the current latent
    zz = z[:, :, 0].float()
    rev = torch.zeros_like(full, dtype=torch.bool)
    for t in range(4, zz.shape[1] - 1):
        d_now = (zz[:, t + 1] - zz[:, t]).norm(dim=-1)
        d_old = torch.cdist(zz[:, t + 1:t + 2], zz[:, :t - 2]).squeeze(1).min(-1).values
        rev[:, t] = d_old < 0.5 * d_now
    res["revisit_share"] = float(rev[:, 3:].float().mean())
    res["revisit_benefit"] = float(1 - full[rev].mean() / w3[rev].mean())
    res["revisit_full_vs_copy"] = float(full[rev].mean() / copy[rev].mean())
    # imagination after 48 frames of context, 16 steps
    V = float(zz.flatten(0, 1).var(0).sum())
    gen = {"recurrent": [], "window3": []}
    with autocast_context(config):
        for i in range(0, len(z), 8):
            zi, ai = z[i:i + 8].to(device), a[i:i + 8].to(device)
            st = world.teacher(zi[:, :48], ai[:, :47]).state
            g = []
            for k in range(16):
                st, _ = world.advance(st, ai[:, 47 + k:48 + k])
                g.append(st.latent[:, 0, 0].float())
            gen["recurrent"].append(torch.stack(g, 1).cpu())
            lat = [zi[:, j, 0] for j in range(45, 48)]
            g = []
            for k in range(16):
                z3 = torch.stack(lat[-3:], 1)[:, :, None]
                a3 = ai[:, 45 + k:48 + k]
                nxt = _advance_pred(world, z3, a3)[:, 0, 0].float()
                g.append(nxt); lat.append(nxt.to(zi.dtype))
            gen["window3"].append(torch.stack(g, 1).cpu())
    truth = zz[:, 48:64]
    res["imagine_after_48"] = {k: [float(((torch.cat(v) - truth) ** 2).sum(-1)[:, d].mean() / V) for d in (0, 3, 7, 15)]
                               for k, v in gen.items()}
    res["imagine_after_48"]["copy_root"] = [float(((zz[:, 47:48] - truth) ** 2).sum(-1)[:, d].mean() / V) for d in (0, 3, 7, 15)]
    return res


def main():
    device = torch.device("cuda")
    started = time.time()
    log = lambda **kw: print(json.dumps({**kw, "seconds": round(time.time() - started, 1)}), flush=True)
    config, episodes = load(device)
    OUT.mkdir(parents=True, exist_ok=True)
    result = {}
    for length, batch in ARMS.items():
        path = OUT / f"L{length}.pt"
        if path.exists():
            world = new_world(config, device)
            state = torch.load(path, map_location="cpu", weights_only=False)
            world.load_state_dict(state["world"]); world.eval()
            history = state["history"]
        else:
            world, history = train(config, episodes, length, batch, device, log)
            torch.save({"world": world.state_dict(), "history": history, "L": length, "B": batch}, path)
        result[f"L{length}"] = {"batch": batch, "history": history, **evaluate(world, config, episodes, device)}
        log(stage="eval", L=length, **{k: v for k, v in result[f"L{length}"].items() if k != "history"})
        (HERE / "context_length.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
