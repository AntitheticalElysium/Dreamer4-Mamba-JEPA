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
Stage B (`--stage b`): the dynamics prior G (Delta-IRIS world_model.py / transformer.py / crafter.yaml): blocks of [I-token, action,
4 Delta-tokens], 21 blocks, causal pre-norm transformer (3 layers, 8 heads, width 512, GELU 4x, no dropout, learned absolute
positions over 126 tokens), heads Linear-ReLU-Linear: Delta logits at [action, Delta1..3] -> Delta1..4, episode end at action.
I-token = per-tile Linear 192 -> 8, flattened 648 -> 512, LayerNorm (Delta-IRIS: an 8-channel frame CNN flattened to 512).
Targets: codes of the FROZEN stage-A posterior on 21-frame windows of the 64-frame Raw TRAIN ledger (26.4% end-aligned on a
death, tworld.TERMINAL_SHARE); end_t = the frame after action t is dead. CE, weights 1 / 1. minGPT AdamW (lr 1e-4, wd 0.01 on
Linear weights only), grad clip 10, no warmup, batch 32 (crafter.yaml). The last block's Delta-tokens are dummies (no loss).
Usage: dworld.py --stage a --seed 7 --init <corrt_raw_teacher_s7_u36000.pt> --updates 18000
       dworld.py --stage b --seed 7 --init <dworld_a_...pt> --updates 20000
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

    def __init__(self, size, dim, input_dim, max_revival=0):
        super().__init__()
        self.revival_entropy_threshold = int(math.log2(size)) - 2
        self.max_revival = max_revival                  # Delta-IRIS max_codebook_updates_with_revival: crafter 0, atari 400, None
        self.register_buffer("updates", torch.zeros((), dtype=torch.long))
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
                can = self.entropy() < 1 or self.max_revival is None or int(self.updates) < self.max_revival
                if can and self.entropy() < self.revival_entropy_threshold:
                    expired = torch.where(self.freqs < 1 / (10 * len(self.freqs)))[0]
                    expired = expired[torch.randperm(len(expired), device=z.device)[:len(z)]]
                    self.codebook[expired] = z[torch.randperm(len(z), device=z.device)[:len(expired)]]
                    self.freqs[expired] = 1 / len(self.freqs)
                self.codebook.copy_(F.normalize(self.codebook, dim=-1))
                self.updates += 1
        q = z + (q - z).detach()
        return self.post(q).view(*shape[:-1], -1), tokens.view(shape[:-1]), loss

    @torch.no_grad()
    def embed(self, tokens):
        return self.post(self.codebook[tokens])


class Posterior(nn.Module):
    """Delta-IRIS's encoder over (x1, action plane, x2), on the token grid: [192 + 192 + 1, 10, 10] -> 4 x 64 -> codes."""

    def __init__(self, channels=64, mult=(1, 1, 2, 2, 4), max_revival=0):
        super().__init__()
        self.action = nn.Embedding(S.N, GRID * GRID)
        layers, c = [nn.Conv2d(2 * S.WIDTH + 1, channels, 3, 1, 1)], channels
        for m in mult:
            layers.append(ResidualBlock(c, m * channels)); c = m * channels
        layers += [nn.GroupNorm(32, c), nn.SiLU(), nn.Conv2d(c, LAT, 3, 1, 1)]
        self.net = nn.Sequential(*layers)
        self.quantizer = Quantizer(CODES, CDIM, LAT * REGION * REGION, max_revival)
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


def train_a(seed, init, updates, device, log, state_path, max_revival=0):
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
        post = Posterior(max_revival=max_revival)
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
            with torch.no_grad(), autocast_context(config):                           # the information the decoder takes from Delta:
                no_delta = float(TW.rollout_losses(world, s, a, "teacher"))            # same batch, Delta zeroed (world.delta None)
            row = {"update": update + 1, "teacher": float(teacher), "teacher_no_delta": no_delta, "commitment": float(commit),
                   "gradient_norm": float(norm),
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


BLOCKS, PER, WIDTH = 21, 2 + K, 512


class Prior(nn.Module):
    def __init__(self):
        super().__init__()
        self.tile = nn.Linear(S.WIDTH, 8)
        self.frame = nn.Sequential(nn.Linear(81 * 8, WIDTH), nn.LayerNorm(WIDTH))
        self.act, self.lat = nn.Embedding(S.N, WIDTH), nn.Embedding(CODES, WIDTH)
        self.pos = nn.Embedding(BLOCKS * PER, WIDTH)
        layer = nn.TransformerEncoderLayer(WIDTH, 8, 4 * WIDTH, dropout=0.0, activation="gelu", batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, 3, enable_nested_tensor=False)
        self.ln = nn.LayerNorm(WIDTH)
        self.head_lat = nn.Sequential(nn.Linear(WIDTH, WIDTH), nn.ReLU(), nn.Linear(WIDTH, CODES))
        self.head_end = nn.Sequential(nn.Linear(WIDTH, WIDTH), nn.ReLU(), nn.Linear(WIDTH, 2))
        self.register_buffer("causal", torch.ones(BLOCKS * PER, BLOCKS * PER, dtype=torch.bool).triu(1), persistent=False)

    def sequence(self, s, a, codes):
        """s [B,T,81,192], a [B,T], codes [B,T,4] -> [B,T*6,512] (I-token, action, Delta1..4 per block)"""
        itok = self.frame(self.tile(s.float()).flatten(-2))[:, :, None]
        x = torch.cat([itok, self.act(a)[:, :, None], self.lat(codes)], 2).flatten(1, 2)
        return x + self.pos(torch.arange(x.shape[1], device=x.device))

    def forward(self, s, a, codes):
        """-> Delta logits [B,T,4,1024] (from the action and Delta1..3 positions), end logits [B,T,2]"""
        x = self.sequence(s, a, codes)
        n = x.shape[1]
        y = self.ln(self.blocks(x, mask=self.causal[:n, :n])).view(len(s), -1, PER, WIDTH)
        return self.head_lat(y[:, :, 1:PER - 1]), self.head_end(y[:, :, 1])

    @torch.no_grad()
    def sample(self, s, a, codes, temperature=1.0, generator=None):
        """History blocks s [B,T,81,192], a [B,T] (a[:, -1] = the action now), codes [B,T-1,4] of the past transitions ->
        sampled Delta codes [B,4] for the transition after a[:, -1], and P(end) [B]."""
        b, t = a.shape
        cur = torch.cat([codes, codes.new_zeros(b, 1, K)], 1)
        for j in range(K):
            logits, end = self(s, a, cur)
            p = torch.softmax(logits[:, -1, j].float() / temperature, -1)
            cur[:, -1, j] = torch.multinomial(p, 1, generator=generator)[:, 0]
        return cur[:, -1], torch.softmax(end[:, -1].float(), -1)[:, 1]


def adamw_mingpt(model, lr=1e-4, wd=0.01):
    """Delta-IRIS utils.configure_optimizer: decay on Linear weights only."""
    decay = [p for m in model.modules() if isinstance(m, nn.Linear) for n, p in m.named_parameters(recurse=False) if n == "weight"]
    ids = {id(p) for p in decay}
    rest = [p for p in model.parameters() if id(p) not in ids]
    return torch.optim.AdamW([{"params": decay, "weight_decay": wd}, {"params": rest, "weight_decay": 0.0}], lr=lr)


def load_a(path, device):
    st = torch.load(path, map_location="cpu", weights_only=False)
    world, post = TW.TWorld("corrt"), Posterior()
    world.load_state_dict(st["world"]); post.load_state_dict(st["post"])
    return world.to(device).eval(), post.to(device).eval(), st


def train_b(seed, init, updates, device, log, state_path, batch=32):
    import numpy as np
    from d4mj.config import config_from_dict
    from d4mj.train import autocast_context
    config = config_from_dict(torch.load(S.CHECKPOINT, map_location="cpu", weights_only=False)["config"])
    _, post, _ = load_a(init, device)
    lab = torch.load(TW.POOLS["rawlong"] / "labels.pt", weights_only=False, mmap=True)
    tokens = np.memmap(TW.POOLS["rawlong"] / "tokens.f16", dtype=np.float16, mode="r", shape=(len(lab["terminal"]), 64, 81, 192))
    main_rows, term_rows = torch.where(~lab["terminal"])[0], torch.where(lab["terminal"])[0]
    with torch.random.fork_rng(devices=[0]):
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        prior = Prior().to(device)
    opt = adamw_mingpt(prior)
    order = torch.Generator().manual_seed(11)
    history, start, started = [], 0, time.time()
    if state_path.exists():
        st = torch.load(state_path, map_location="cpu", weights_only=False)
        prior.load_state_dict(st["prior"]); opt.load_state_dict(st["optimizer"]); order.set_state(st["order"])
        torch.set_rng_state(st["rng_cpu"]); history, start = st["history"], st["update"]
        log(stage="resume", update=start)
    for update in range(start, updates):
        term = torch.rand(batch, generator=order) < TW.TERMINAL_SHARE
        r = torch.where(term, term_rows[torch.randint(len(term_rows), (batch,), generator=order)],
                        main_rows[torch.randint(len(main_rows), (batch,), generator=order)])
        t0 = torch.where(term, 64 - BLOCKS, torch.randint(0, 64 - BLOCKS, (batch,), generator=order))
        s = torch.stack([torch.from_numpy(np.array(tokens[int(i), int(j):int(j) + BLOCKS])) for i, j in zip(r, t0)]).to(device).float()
        acts = F.pad(lab["actions"], (0, 1))                                             # a_63 does not exist: NOOP, never used
        a = torch.stack([acts[int(i), int(j):int(j) + BLOCKS] for i, j in zip(r, t0)]).to(device)
        dead = torch.stack([~lab["alive"][int(i), int(j) + 1:int(j) + BLOCKS] for i, j in zip(r, t0)]).to(device).long()   # [B,20]
        with torch.no_grad():
            _, codes, _ = post.quantizer(post.encode(s[:, :-1].flatten(0, 1), a[:, :-1].flatten(), s[:, 1:].flatten(0, 1)))
        codes = F.pad(codes.view(batch, BLOCKS - 1, K), (0, 0, 0, 1))                     # last block: dummy
        with autocast_context(config):
            lat, end = prior(s, a, codes)
            loss_lat = F.cross_entropy(lat[:, :-1].flatten(0, 2).float(), codes[:, :-1].flatten())
            loss_end = F.cross_entropy(end[:, :-1].flatten(0, 1).float(), dead.flatten())
            loss = loss_lat + loss_end
        opt.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(prior.parameters(), 10.0)
        opt.step()
        if (update + 1) % 500 == 0:
            acc = float((lat[:, :-1].argmax(-1) == codes[:, :-1]).float().mean())
            row = {"update": update + 1, "loss_latents": float(loss_lat), "loss_ends": float(loss_end), "latent_accuracy": round(acc, 4),
                   "gradient_norm": float(norm), "seconds": round(time.time() - started, 1)}
            history.append(row)
            log(stage="train_b", **row)
        if (update + 1) % TW.STATE_EVERY == 0 or update + 1 == updates:
            tmp = state_path.with_suffix(".tmp")
            torch.save({"update": update + 1, "prior": prior.state_dict(), "optimizer": opt.state_dict(), "order": order.get_state(),
                        "rng_cpu": torch.get_rng_state(), "history": history}, tmp)
            tmp.replace(state_path)
    return prior, history


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=("a", "b"), required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--init", type=Path, required=True)
    p.add_argument("--updates", type=int, required=True)
    p.add_argument("--revival", default="crafter", choices=("crafter", "atari", "always"),
                   help="Delta-IRIS codebook revival: crafter.yaml 0, atari.yaml 400 (steps_first_epoch), the class default None")
    a = p.parse_args()
    started = time.time()
    log = lambda **kw: print(json.dumps({**kw, "seconds_total": round(time.time() - started, 1)}), flush=True)
    name = (f"dworld_a_s{a.seed}_from{a.init.stem.split('_u')[-1]}" + ("" if a.revival == "crafter" else f"_rev{a.revival}") + f"_u{a.updates}" if a.stage == "a"
            else f"{a.init.stem}_prior_u{a.updates}")
    out = TW.OUT / f"{name}.pt"
    if out.exists():
        log(status="exists", name=name)
        return 0
    (TW.OUT / "state").mkdir(exist_ok=True)
    from d4mj.data import _sha256
    meta = {"name": name, "args": {k: str(v) for k, v in vars(a).items()}, "script_sha256": _sha256(Path(__file__))}
    if a.stage == "a":
        world, post, history = train_a(a.seed, a.init, a.updates, torch.device("cuda"), log, TW.OUT / "state" / f"{name}.state.pt",
                                       {"crafter": 0, "atari": 400, "always": None}[a.revival])
        torch.save(meta | {"world": world.state_dict(), "post": post.state_dict(), "history": history}, out)
    else:
        prior, history = train_b(a.seed, a.init, a.updates, torch.device("cuda"), log, TW.OUT / "state" / f"{name}.state.pt")
        torch.save(meta | {"prior": prior.state_dict(), "stage_a": str(a.init), "history": history}, out)
    log(status="saved", name=name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
