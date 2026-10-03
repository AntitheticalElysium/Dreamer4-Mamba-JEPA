"""E16 (design and readings predeclared in NOTEBOOK.md, 2026-10-03, commit 7e353eba): Delta-IRIS's stochastic channel on the
per-tile world. Source: Micheli et al., ICML 2024 (arXiv 2406.19320, sections 2.2-2.4) and vmicheli/delta-iris f8d4173
(src/models/tokenizer/{tokenizer,quantizer}.py, src/models/convnet.py, config/params/crafter.yaml), read 2026-10-03.

Stage A (this file, `--stage a`): a posterior encoder E reads (frame t tokens, action t, frame t+1 tokens) on the 9 x 9 token grid
(zero-padded to 10 x 10), emits K = 4 Delta-tokens (one per 5 x 5 region), quantized (cosine codebook 1024 x 64, EMA 0.99,
commitment 0.02, revival only on collapse: crafter.yaml max_codebook_updates_with_revival 0). Each token's post-quantized vector
covers its region (Delta-IRIS rearranges it onto the latent grid) and enters the corrt world's frame-t tile embeddings through a
zero-initialized projection (TWorld.delta), so at initialization the decoder IS the deterministic world it continues from.
Loss: the world's teacher L1 over the 5 targets of each 6-frame pool window + commitment. Data, batch, optimizer, schedule and
resumable state as tworld.train.
Usage: dworld.py --stage a --seed 7 --init <corrt_raw_teacher_s7_u36000.pt> --updates 18000
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import tworld as TW  # noqa: E402
S = TW.S
K, CODES, CDIM, LAT, GRID, REGION = 4, 1024, 64, 64, 10, 5


class ResidualBlock(nn.Module):
    """Delta-IRIS convnet.ResidualBlock."""

    def __init__(self, cin, cout, groups=32):
        super().__init__()
        self.f = nn.Sequential(nn.GroupNorm(groups, cin), nn.SiLU(), nn.Conv2d(cin, cout, 3, 1, 1),
                               nn.GroupNorm(groups, cout), nn.SiLU(), nn.Conv2d(cout, cout, 3, 1, 1))
        self.skip = nn.Identity() if cin == cout else nn.Conv2d(cin, cout, 1)

    def forward(self, x):
        return self.skip(x) + self.f(x)


class Quantizer(nn.Module):
    """Delta-IRIS tokenizer/quantizer.py: cosine nearest code, EMA codebook (0.99), frequencies (0.98), commitment 0.02,
    revival of expired codes only while the codebook entropy is below 1 bit (max_codebook_updates_with_revival = 0)."""

    def __init__(self, size, dim, input_dim):
        super().__init__()
        self.revival_entropy_threshold = int(math.log2(size)) - 2
        self.pre, self.post = nn.Linear(input_dim, dim), nn.Linear(dim, input_dim)
        self.register_buffer("codebook", torch.empty(size, dim).uniform_(-1.0 / size, 1.0 / size))
        self.register_buffer("freqs", torch.ones(size) / size)

    def entropy(self):
        p = self.freqs[self.freqs != 0]
        return float(-(torch.log2(p) * p).sum())

    def forward(self, z):
        with torch.autocast(device_type=z.device.type, enabled=False):      # nearest code and EMA in float32
            return self._forward(self.pre(z.float()))

    def _forward(self, z):
        z = F.normalize(z, dim=-1)
        shape = z.shape
        z = z.reshape(-1, shape[-1])
        tokens = (z @ self.codebook.t()).argmax(-1)
        q = self.codebook[tokens]
        loss = 0.02 * (z - q.detach()).pow(2).mean()
        if self.training:
            with torch.no_grad():
                onehot = F.one_hot(tokens, len(self.codebook)).float()
                counts = onehot.sum(0)
                update = F.normalize(onehot.t() @ z / counts.clamp(min=1)[:, None], dim=-1)
                self.codebook.lerp_(update, 1 - 0.99)
                self.freqs.lerp_(counts / len(z), 1 - 0.98)
                if self.entropy() < 1 and self.entropy() < self.revival_entropy_threshold:
                    expired = torch.where(self.freqs < 1 / (10 * len(self.freqs)))[0]
                    expired = expired[torch.randperm(len(expired), device=z.device)[:len(z)]]
                    self.codebook[expired] = z[torch.randperm(len(z), device=z.device)[:len(expired)]]
                    self.freqs[expired] = 1 / len(self.freqs)
                self.codebook.copy_(F.normalize(self.codebook, dim=-1))
        q = z + (q - z).detach()
        return self.post(q).view(*shape[:-1], -1), tokens.view(shape[:-1]), loss

    @torch.no_grad()
    def embed(self, tokens):
        return self.post(self.codebook[tokens])


class Posterior(nn.Module):
    """Delta-IRIS's encoder over (x1, action plane, x2), on the token grid: [192 + 192 + 1, 10, 10] -> 4 x 64 -> codes."""

    def __init__(self, channels=64, mult=(1, 1, 2, 2, 4)):
        super().__init__()
        self.action = nn.Embedding(S.N, GRID * GRID)
        layers, c = [nn.Conv2d(2 * S.WIDTH + 1, channels, 3, 1, 1)], channels
        for m in mult:
            layers.append(ResidualBlock(c, m * channels)); c = m * channels
        layers += [nn.GroupNorm(32, c), nn.SiLU(), nn.Conv2d(c, LAT, 3, 1, 1)]
        self.net = nn.Sequential(*layers)
        self.quantizer = Quantizer(CODES, CDIM, LAT * REGION * REGION)
        self.cond = nn.Linear(LAT, S.D)                 # region vector -> tile embedding offset; zero-init: D = the world at start
        nn.init.zeros_(self.cond.weight); nn.init.zeros_(self.cond.bias)

    @staticmethod
    def grid(x):                                        # [M,81,C] -> [M,C,10,10] (zero pad row / column 9)
        return F.pad(x.transpose(1, 2).reshape(len(x), -1, 9, 9), (0, 1, 0, 1))

    def encode(self, s0, a, s1):
        """[M,81,192] x2, [M] -> region features [M,4,1600]"""
        x = torch.cat([self.grid(s0), self.action(a).view(-1, 1, GRID, GRID), self.grid(s1)], 1)
        z = self.net(x)                                                                 # [M,64,10,10]
        return z.unflatten(2, (2, REGION)).unflatten(4, (2, REGION)).permute(0, 2, 4, 3, 5, 1).flatten(3).flatten(1, 2)

    def condition(self, q):
        """post-quantized [M,4,1600] -> per-tile offsets [M,81,D]"""
        g = q.view(len(q), 2, 2, REGION, REGION, LAT).permute(0, 5, 1, 3, 2, 4).reshape(len(q), LAT, GRID, GRID)
        return self.cond(g[:, :, :9, :9].flatten(2).transpose(1, 2))

    def forward(self, s, a):
        """s [B,T,81,192], a [B,T-1] -> conditioning [B,T,81,D] (last frame zero), codes [B,T-1,4], commitment loss"""
        b, t = s.shape[:2]
        q, tokens, loss = self.quantizer(self.encode(s[:, :-1].flatten(0, 1), a.flatten(), s[:, 1:].flatten(0, 1)))
        cond = self.condition(q).view(b, t - 1, 81, S.D)
        return F.pad(cond, (0, 0, 0, 0, 0, 1)), tokens.view(b, t - 1, K), loss


def train_a(seed, init, updates, device, log, state_path):
    from d4mj.config import config_from_dict
    from d4mj.train import _phase_lr, autocast_context, optimizer_step, phase_optimizer
    config = config_from_dict(torch.load(S.CHECKPOINT, map_location="cpu", weights_only=False)["config"])
    pool = torch.load(TW.POOLS["raw"] / "pool.pt", weights_only=False, mmap=True)
    main_rows = torch.where(~pool["terminal"])[0]
    held = main_rows[torch.randperm(len(main_rows), generator=torch.Generator().manual_seed(1))[:2048]]
    rows = torch.cat([main_rows[~torch.isin(main_rows, held)], torch.where(pool["terminal"])[0]])
    with torch.random.fork_rng(devices=[0]):
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        world = TW.TWorld("corrt")
        post = Posterior()
    world.load_state_dict(torch.load(init, map_location="cpu", weights_only=False)["world"])
    world, post = world.to(device), post.to(device)
    opt = phase_optimizer([world, post], config)
    params = [p for g in opt.param_groups for p in g["params"]]
    order = torch.Generator().manual_seed(11)
    history, start, started = [], 0, time.time()
    if state_path.exists():
        st = torch.load(state_path, map_location="cpu", weights_only=False)
        world.load_state_dict(st["world"]); post.load_state_dict(st["post"]); opt.load_state_dict(st["optimizer"])
        order.set_state(st["order"]); torch.set_rng_state(st["rng_cpu"])
        if st["rng_cuda"] is not None:
            torch.cuda.set_rng_state_all(st["rng_cuda"])
        history, start = st["history"], st["update"]
        log(stage="resume", update=start)

    def save_state(u):
        tmp = state_path.with_suffix(".tmp")
        torch.save({"update": u, "world": world.state_dict(), "post": post.state_dict(), "optimizer": opt.state_dict(),
                    "order": order.get_state(), "rng_cpu": torch.get_rng_state(), "history": history,
                    "rng_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}, tmp)
        tmp.replace(state_path)
    for update in range(start, updates):
        idx = rows[torch.randint(len(rows), (TW.BATCH,), generator=order)]
        b = S.batch_of(pool, idx, "tokens", device)
        s, a = b["s"], b["actions"]
        with autocast_context(config):
            cond, tokens, commit = post(s, a)
            world.delta = cond
            teacher = TW.rollout_losses(world, s, a, "teacher")
            world.delta = None
            objective = teacher + commit
        norm = optimizer_step(opt, objective, params, learning_rate=_phase_lr(config, update), grad_clip=config.agent.grad_clip,
                              strict=True, zero_grad=True)
        if (update + 1) % 500 == 0:
            used = int(torch.unique(tokens).numel())
            row = {"update": update + 1, "teacher": float(teacher), "commitment": float(commit), "gradient_norm": float(norm),
                   "codebook_entropy": round(post.quantizer.entropy(), 3), "codes_in_batch": used,
                   "seconds": round(time.time() - started, 1)}
            history.append(row)
            log(stage="train", **row)
        if (update + 1) % TW.STATE_EVERY == 0 or update + 1 == updates:
            save_state(update + 1)
            if update + 1 < updates:
                torch.save({"world": world.state_dict(), "post": post.state_dict(), "history": history},
                           state_path.parent.parent / f"{state_path.name.split('.')[0]}_at{update + 1}.pt")
    return world, post, history


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=("a",), required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--init", type=Path, required=True)
    p.add_argument("--updates", type=int, required=True)
    a = p.parse_args()
    started = time.time()
    log = lambda **kw: print(json.dumps({**kw, "seconds_total": round(time.time() - started, 1)}), flush=True)
    name = f"dworld_a_s{a.seed}_from{a.init.stem.split('_u')[-1]}_u{a.updates}"
    out = TW.OUT / f"{name}.pt"
    if out.exists():
        log(status="exists", name=name)
        return 0
    (TW.OUT / "state").mkdir(exist_ok=True)
    world, post, history = train_a(a.seed, a.init, a.updates, torch.device("cuda"), log, TW.OUT / "state" / f"{name}.state.pt")
    from d4mj.data import _sha256
    torch.save({"name": name, "args": {k: str(v) for k, v in vars(a).items()}, "world": world.state_dict(), "post": post.state_dict(),
                "history": history, "script_sha256": _sha256(Path(__file__))}, out)
    log(status="saved", name=name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
