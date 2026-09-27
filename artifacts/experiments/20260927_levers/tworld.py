"""Per-tile world arms (E2 discretization, E3 TC+T, E5 copy/Delta variants): one recipe, one switch at a time.

Backbone: spatial.World exactly (V-JEPA 2-AC frame layout at H2's scale: per frame [action token, 81 tile
tokens], frame block-causal attention, 6 pre-norm layers, width 256, 4 heads, learned space/time positions).
Only the OUTPUT differs between arms; h = the backbone's normalized output for tile i at frame t:
  direct       LN(proj(h))                                       spatial.World as trained (no copy path)
  residual     LN(s_t + proj(h)), proj zero-init                 Nagabandi et al. 2018 (predict the change)
  gated        LN(g s_t + (1-g) proj(h)), g = sigmoid(w.h + 3)   copy-or-generate at the same tile (ITC's
                                                                 decision, continuous, same position)
  corrg        corr, plus one logit per frame from the action token's output (which attends to the whole frame),
               added to the 4 neighbour logits of every tile: a shared "did the view move" decision; zero-init
  corrt        corrg, plus (move actions only) a logit read from the backbone output AT the tile the move enters
               (player token 31 + direction): bypasses routing the target's passability to the action token
  corr         LN(sum_c w_c cand_c + w_gen proj(h)), w = softmax over {self, up, down, left, right, generate},
               cand = the tile's own and its 4 neighbours' tokens in frame t (zeros off-grid), self logit +3
               (ITC's displacement-capped copy-or-generate, continuous, cap = 1 tile)
  categorical  input tokens quantized to the nearest of K k-means codes (codebook.py); output = K logits per tile;
               cross-entropy on the next frame's code (Dedieu et al. 2025 / DreamerV2: discrete targets)
Training (fixed for every arm): spatial_pool_v1 (Raw tokens) or spatial_pool_tc_v1 (TC tokens), 2,048 main windows
held out (seed 1, as parameterization.py); batches of 40 windows (seed 11); AdamW lr 1e-4, wd 0.01, 1,000 warmup,
clip 1 (H2 phase optimizer); bf16; init seed given (default 7). Loss: `suffix` = spatial.losses' dynamics L1
(teacher-forced + depth-2 generated suffix from frame 3); `teacher` = teacher-forced only. Categorical is
teacher-forced CE (argmax is not differentiable). No heads in any arm.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

HERE = Path(__file__).parent
ROOT = Path("/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "artifacts/experiments/20260921_readout_ladder"))
import spatial as S  # noqa: E402

POOLS = {"raw": ROOT / "artifacts/eda/spatial_pool_v1", "tc": ROOT / "artifacts/eda/spatial_pool_tc_v1"}
OUT = ROOT / "artifacts/eda/levers_tworlds_v1"
BATCH = 40
NEIGHBOURS = ((-1, 0), (1, 0), (0, -1), (0, 1))


class TWorld(S.World):
    def __init__(self, head, codebook=None):
        super().__init__(S.TOKENS, False)
        self.head = head
        if head in ("residual",):
            nn.init.zeros_(self.proj.weight); nn.init.zeros_(self.proj.bias)
        if head == "gated":
            self.gate = nn.Linear(S.D, 1)
            nn.init.zeros_(self.gate.weight); nn.init.constant_(self.gate.bias, 3.0)
        if head in ("corrg", "corrt"):            # corr + a frame-level move gate from the action token
            self.frame = nn.Linear(S.D, 1)
            nn.init.zeros_(self.frame.weight); nn.init.zeros_(self.frame.bias)
        if head == "corrt":                       # + the target tile's own output, indexed by the move direction
            self.target_gate = nn.Linear(S.D, 1)
            nn.init.zeros_(self.target_gate.weight); nn.init.zeros_(self.target_gate.bias)
        if head in ("corr", "corrg", "corrt"):
            self.choose = nn.Linear(S.D, 6)
            nn.init.zeros_(self.choose.weight)
            with torch.no_grad():
                self.choose.bias.copy_(torch.tensor([3.0, 0, 0, 0, 0, 0]))
        if head == "categorical":
            self.register_buffer("codes", codebook.float())
            self.logits = nn.Linear(S.D, len(codebook))

    def backbone(self, s, a):
        b, t = s.shape[:2]
        x = torch.cat([self.action(a)[:, :, None], self.embed(s)], 2) + self.space + self.time[:t, None]
        k = t * (self.n + 1)
        x = self.blocks(x.flatten(1, 2), mask=self.blocked[:k, :k]).view(b, t, self.n + 1, S.D)[:, :, 1:]
        return self.norm(x)

    def backbone_full(self, s, a):
        """backbone(), also returning the action token's output (it attends to the whole frame)."""
        b, t = s.shape[:2]
        x = torch.cat([self.action(a)[:, :, None], self.embed(s)], 2) + self.space + self.time[:t, None]
        k = t * (self.n + 1)
        x = self.norm(self.blocks(x.flatten(1, 2), mask=self.blocked[:k, :k]).view(b, t, self.n + 1, S.D))
        return x[:, :, 1:], x[:, :, 0]

    def forward(self, s, a):
        if self.head in ("corrg", "corrt"):
            h, h_action = self.backbone_full(s, a)
        else:
            h = self.backbone(s, a)
        if self.head == "categorical":
            logits = self.logits(h).float()
            return self.codes[logits.argmax(-1)], h, logits
        gen = self.proj(h).float()
        s = s.float()
        if self.head == "direct":
            out = gen
        elif self.head == "residual":
            out = s + gen
        elif self.head == "gated":
            g = torch.sigmoid(self.gate(h).float())
            out = g * s + (1 - g) * gen
        elif self.head in ("corr", "corrg", "corrt"):
            grid = s.view(*s.shape[:2], 9, 9, s.shape[-1])
            pad = F.pad(grid, (0, 0, 1, 1, 1, 1))
            cands = [grid] + [pad[:, :, 1 + dr:10 + dr, 1 + dc:10 + dc] for dr, dc in NEIGHBOURS]
            cands = torch.stack([c.reshape(s.shape) for c in cands] + [gen], -2)          # [B,T,81,6,D]
            logits = self.choose(h).float()
            if self.head in ("corrg", "corrt"):  # one "did the view move" logit per frame, added to the 4 neighbours
                moved = self.frame(h_action).float()[:, :, None, :]                  # [B,T,1,1]
                if self.head == "corrt":         # the tile the move enters (player token 31 + direction), moves only
                    target = torch.tensor([31, 30, 32, 22, 40], device=a.device)[a.clamp(max=4)]      # [B,T]
                    h_t = h.gather(2, target[:, :, None, None].expand(-1, -1, 1, h.shape[-1]))[:, :, 0]
                    is_move = ((a >= 1) & (a <= 4)).float()[:, :, None, None]
                    moved = moved + is_move * self.target_gate(h_t).float()[:, :, None, :]
                logits = logits + moved * logits.new_tensor([0, 1, 1, 1, 1, 0])
            w = torch.softmax(logits, -1)[..., None]
            out = (w * cands).sum(-2)
        return F.layer_norm(out, (S.WIDTH,)), h, None


def quantize(x, codes, chunk=16384):
    flat = x.reshape(-1, x.shape[-1]).float()
    idx = torch.cat([torch.cdist(flat[i:i + chunk], codes).argmin(-1) for i in range(0, len(flat), chunk)])
    return idx.view(x.shape[:-1])


def rollout_losses(world, s, a, loss):
    """spatial.rollout + the dynamics L1 (suffix) or teacher-forced L1 only."""
    a = F.pad(a, (0, 1))
    predicted, history, _ = world(s, a)
    teacher = (predicted[:, :S.W - 1] - s[:, 1:]).abs().mean()
    if loss == "teacher":
        return teacher
    first = predicted[:, S.ANCHOR]
    second_all, _, _ = world(torch.cat([s[:, :S.ANCHOR + 1], first[:, None].to(s.dtype)], 1), a[:, :S.ANCHOR + 2])
    generated = torch.stack([first, second_all[:, S.ANCHOR + 1]], 1)
    return teacher + (generated - s[:, S.ANCHOR + 1:]).abs().mean()


def train(head, pool_name, loss, seed, updates, device, log, codebook=None):
    from d4mj.config import config_from_dict
    from d4mj.train import _phase_lr, autocast_context, optimizer_step, phase_optimizer
    config = config_from_dict(torch.load(S.CHECKPOINT, map_location="cpu", weights_only=False)["config"])
    pool = torch.load(POOLS[pool_name] / "pool.pt", weights_only=False, mmap=True)
    main_rows = torch.where(~pool["terminal"])[0]
    held = main_rows[torch.randperm(len(main_rows), generator=torch.Generator().manual_seed(1))[:2048]]
    rows = torch.cat([main_rows[~torch.isin(main_rows, held)], torch.where(pool["terminal"])[0]])
    with torch.random.fork_rng(devices=[0]):
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        world = TWorld(head, codebook).to(device)
    opt = phase_optimizer([world], config)
    params = [p for g in opt.param_groups for p in g["params"]]
    order = torch.Generator().manual_seed(11)
    codes = codebook.to(device) if codebook is not None else None
    history, started = [], time.time()
    for update in range(updates):
        b = S.batch_of(pool, rows[torch.randint(len(rows), (BATCH,), generator=order)], "tokens", device)
        s, a = b["s"], b["actions"]
        with autocast_context(config):
            if head == "categorical":
                idx = quantize(s, codes)
                _, _, logits = world(codes[idx], F.pad(a, (0, 1)))
                objective = F.cross_entropy(logits[:, :S.W - 1].flatten(0, 2), idx[:, 1:].flatten())
            else:
                objective = rollout_losses(world, s, a, loss)
        norm = optimizer_step(opt, objective, params, learning_rate=_phase_lr(config, update),
                              grad_clip=config.agent.grad_clip, strict=True, zero_grad=True)
        if (update + 1) % 500 == 0:
            row = {"update": update + 1, "objective": float(objective), "gradient_norm": float(norm),
                   "seconds": round(time.time() - started, 1)}
            history.append(row)
            log(stage="train", head=head, pool=pool_name, **row)
    return world.eval(), history, held


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--head", required=True, choices=("direct", "residual", "gated", "corr", "corrg", "corrt", "categorical"))
    parser.add_argument("--pool", default="raw", choices=tuple(POOLS))
    parser.add_argument("--loss", default="suffix", choices=("suffix", "teacher"))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--updates", type=int, default=6000)
    parser.add_argument("--codebook", type=Path, default=None)
    args = parser.parse_args(argv)
    started = time.time()
    log = lambda **kw: print(json.dumps({**kw, "seconds_total": round(time.time() - started, 1)}), flush=True)
    device = torch.device("cuda")
    codebook = torch.load(args.codebook, weights_only=False)["codes"] if args.codebook else None
    name = f"{args.head}_{args.pool}_{args.loss}_s{args.seed}" + (f"_K{len(codebook)}" if codebook is not None else "") \
        + ("" if args.updates == 6000 else f"_u{args.updates}")
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / f"{name}.pt").exists():
        log(status="exists", name=name)
        return 0
    world, history, held = train(args.head, args.pool, args.loss, args.seed, args.updates, device, log, codebook)
    from d4mj.data import _sha256
    torch.save({"name": name, "args": {k: str(v) for k, v in vars(args).items()}, "world": world.state_dict(),
                "history": history, "script_sha256": _sha256(Path(__file__))}, OUT / f"{name}.pt")
    log(status="saved", name=name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
