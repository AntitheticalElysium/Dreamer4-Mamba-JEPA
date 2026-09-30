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
Backbones (stage 2; `--backbone`, default `full`; every other part of the recipe identical):
  full         spatial.World: one block-causal attention over all frames' 82 tokens (V-JEPA 2-AC layout)
  fattn        per layer: space attention within the frame (82 tokens), causal time attention per token position,
               MLP -- Dreamer 4's factorized layout (arXiv 2509.24527 s3; nicklashansen/dreamer4 model.py
               BlockCausalLayer), time mixing in every layer
  fmamba       fattn with the time mixer = the canonical Mamba-2 block (RMSNorm + FunctionalMamba2, canonical
               DynamicsSettings) per token position: Po et al. 2025 (arXiv 2505.20171) block size 1
  fcanvas      fmamba on a world-aligned canvas: frame t's 63 map tokens sit at canvas cell (r, c) + o_t, o_t the
               cumulative view scroll estimated between consecutive INPUT frames (scroll.estimate: 0.994 accurate);
               each canvas cell is scanned over time with delta = 0 (an exact hold: decay 1, no update) at frames
               where it is out of view; outputs gathered back to screen positions. CMP's egocentric memory warped
               by ego-motion (Gupta et al. 2017), done as a change of coordinates. HUD and action tokens unshifted
               (their own sequences)
  fscan        one Mamba-2 scan over the window in spatial-major order (T x 82 tokens): Po et al.'s variant
               "without block-wise scan"
ITC's training and decoding (arXiv 2605.16457, verified in the PDF 2026-09-28), two switches on the corr heads:
  --gen-loss   the generate candidate also gets its own teacher-forced L1 on every token. ITC "leaves the
               transformer and its training loss unchanged": the transformer's next-token predictions are trained
               on all tokens (their appendix loss 1) and copying is chosen at decoding. Our corr heads train the
               generator only through its mixture weight (measured 0.002-0.06 outside sleep onset: where.py)
  --regions itc  ITC's Craftax rule (appendix, "Choosing Between Transformer and Optimal Transport Output"): copy
               only in the central region; the screen edges (map border ring, 28 cells) and the inventory (HUD,
               18 cells) take the generator's prediction
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
import torch.utils.checkpoint
from torch import nn

HERE = Path(__file__).parent
ROOT = Path("/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "artifacts/experiments/20260921_readout_ladder"))
import spatial as S  # noqa: E402
from scroll import SHIFTS, estimate  # noqa: E402

POOLS = {"raw": ROOT / "artifacts/eda/spatial_pool_v1", "tc": ROOT / "artifacts/eda/spatial_pool_tc_v1",
         "ldad10": ROOT / "artifacts/eda/spatial_pool_ldad10_v1"}
OUT = ROOT / "artifacts/eda/levers_tworlds_v1"
BATCH = 40
NEIGHBOURS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def masked_scan(mixer, inputs, keep):
    """FunctionalMamba2.scan from a zero carry (its Triton path, same equations), with delta = 0 wherever keep is
    False: decay exp(0) = 1 and no state update, so the state is held exactly through those steps."""
    from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
    c = mixer.core
    z, xbc, dt = torch.split(c.in_proj(inputs), [c.d_ssm, c.d_ssm + 2 * c.d_state, c.nheads], dim=-1)
    filtered = F.silu(F.conv1d(F.pad(xbc.transpose(1, 2), (c.d_conv - 1, 0)), c.conv1d.weight, c.conv1d.bias,
                               groups=c.conv1d.groups)).transpose(1, 2)
    x, b, cc = torch.split(filtered, [c.d_ssm, c.d_state, c.d_state], dim=-1)
    dt = torch.where(keep[..., None], dt, dt.new_tensor(-1e4))             # softplus(-1e4 + dt_bias) == 0
    y = mamba_chunk_scan_combined(x.reshape(*x.shape[:2], c.nheads, c.headdim), dt, -c.A_log.float().exp(),
                                  b.unsqueeze(2), cc.unsqueeze(2), chunk_size=c.chunk_size, D=c.D,
                                  dt_bias=c.dt_bias, dt_softplus=True)
    y = y.flatten(2).to(z.dtype)
    gated = y.float() * F.silu(z.float())
    normalized = gated * torch.rsqrt(gated.square().mean(-1, keepdim=True) + c.norm.eps)
    return c.out_proj((normalized * c.norm.weight.float()).to(y.dtype))


class Factored(nn.Module):
    """One factorized layer (see the module docstring): space attention, a time mixer, MLP; pre-norm residuals."""

    def __init__(self, time, settings):
        super().__init__()
        self.time = time
        self.n1, self.n3 = nn.LayerNorm(S.D), nn.LayerNorm(S.D)
        self.space = nn.MultiheadAttention(S.D, 4, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(S.D, 4 * S.D), nn.GELU(), nn.Linear(4 * S.D, S.D))
        if time == "fattn":
            self.n2, self.mix = nn.LayerNorm(S.D), nn.MultiheadAttention(S.D, 4, batch_first=True)
        else:                                     # the canonical _MambaBlock's norm and mixer
            from d4mj.mamba_recurrence import FunctionalMamba2
            self.n2, self.mix = nn.RMSNorm(S.D, eps=settings.norm_eps), FunctionalMamba2(settings)

    def forward(self, x, shifts):
        b, t, n, d = x.shape
        h = self.n1(x).flatten(0, 1)
        x = x + self.space(h, h, h, need_weights=False)[0].view(b, t, n, d)
        h = self.n2(x)
        per_position = h.transpose(1, 2).flatten(0, 1)                                  # [B*82, T, D]
        if self.time == "fattn":
            causal = torch.ones(t, t, dtype=torch.bool, device=x.device).triu(1)
            y = self.mix(per_position, per_position, per_position, attn_mask=causal, need_weights=False)[0]
            y = y.view(b, n, t, d).transpose(1, 2)
        elif self.time == "fmamba":
            y = self.mix(per_position)[0].view(b, n, t, d).transpose(1, 2)
        elif self.time == "fscan":
            y = self.mix(h.flatten(1, 2))[0].view(b, t, n, d)
        else:                                                                           # fcanvas
            pad = t - 1
            cols, cells = 9 + 2 * pad, (7 + 2 * pad) * (9 + 2 * pad)
            offset = torch.tensor(SHIFTS, device=x.device)[shifts].cumsum(1)            # [B,T,2]
            r = torch.arange(7, device=x.device)[:, None] + offset[..., 0, None, None] + pad
            c = torch.arange(9, device=x.device)[None] + offset[..., 1, None, None] + pad
            cell = (r * cols + c).flatten(2)                                            # [B,T,63]
            # scan only the canvas cells in view at least once (<= 63 + 9 per scroll), compacted per window
            present = torch.zeros(b, cells, dtype=torch.bool, device=x.device).scatter(1, cell.flatten(1), True)
            cell = (present.long().cumsum(1) - 1).gather(1, cell.flatten(1)).view(b, t, 63)
            cells = int(present.sum(1).max())
            seq = torch.cat([h.new_zeros(b, t, cells, d), h[:, :, :1], h[:, :, 64:]], 2)  # canvas, action, HUD
            seq = seq.scatter(2, cell[..., None].expand(-1, -1, -1, d), h[:, :, 1:64])
            keep = torch.zeros(b, t, cells + 19, dtype=torch.bool, device=x.device)
            keep[:, :, cells:] = True
            keep = keep.scatter(2, cell, True)
            y = masked_scan(self.mix, seq.transpose(1, 2).flatten(0, 1), keep.transpose(1, 2).flatten(0, 1))
            y = y.view(b, cells + 19, t, d).transpose(1, 2)
            tiles = y.gather(2, cell[..., None].expand(-1, -1, -1, d))
            y = torch.cat([y[:, :, cells:cells + 1], tiles, y[:, :, cells + 1:]], 2)
        x = x + y
        return x + self.mlp(self.n3(x))


class TWorld(S.World):
    def __init__(self, head, codebook=None, backbone="full", regions="all"):
        super().__init__(S.TOKENS, False)
        self.head, self.backbone_kind, self.regions = head, backbone, regions
        self.hard_decode = False               # evaluation only: ITC's binarized decoding (one source per token, Eq. 4)
        if regions == "itc":
            ring = [r * 9 + c for r in range(7) for c in range(9) if r in (0, 6) or c in (0, 8)]
            mask = torch.zeros(81, dtype=torch.bool)
            mask[ring + list(range(63, 81))] = True
            self.register_buffer("generated_region", mask, persistent=False)
        if backbone != "full":
            from d4mj.config import config_from_dict
            from dataclasses import replace
            settings = config_from_dict(torch.load(S.CHECKPOINT, map_location="cpu", weights_only=False)["config"]).dynamics
            # kernel tiling only (SSD is exact for any chunking): the canonical 256 allocates B*82 x 256 x 256 for
            # 6-step sequences (~860 MB at batch 40); 64 keeps it at ~54 MB
            settings = replace(settings, chunk_size=64)
            del self.blocks
            self.layers = nn.ModuleList(Factored(backbone, settings) for _ in range(6))
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
        if self.backbone_kind != "full":
            return self.backbone_full(s, a)[0]
        b, t = s.shape[:2]
        x = torch.cat([self.action(a)[:, :, None], self.embed(s)], 2) + self.space + self.time[:t, None]
        k = t * (self.n + 1)
        x = self.blocks(x.flatten(1, 2), mask=self.blocked[:k, :k]).view(b, t, self.n + 1, S.D)[:, :, 1:]
        return self.norm(x)

    def backbone_full(self, s, a):
        """backbone(), also returning the action token's output (it attends to the whole frame)."""
        b, t = s.shape[:2]
        x = torch.cat([self.action(a)[:, :, None], self.embed(s)], 2) + self.space + self.time[:t, None]
        if self.backbone_kind == "full":
            k = t * (self.n + 1)
            x = self.norm(self.blocks(x.flatten(1, 2), mask=self.blocked[:k, :k]).view(b, t, self.n + 1, S.D))
            return x[:, :, 1:], x[:, :, 0]
        shifts = F.pad(estimate(s[:, :-1], s[:, 1:]), (1, 0)) if self.backbone_kind == "fcanvas" else None
        for layer in self.layers:
            if torch.is_grad_enabled():   # per-token SSM states (B*82 x 4 x 64 x 64 fp32) exceed 6 GB otherwise; same math
                x = torch.utils.checkpoint.checkpoint(layer, x, shifts, use_reentrant=False)
            else:
                x = layer(x, shifts)
        x = self.norm(x)
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
            if self.hard_decode:
                w = F.one_hot(w[..., 0].argmax(-1), 6).to(w.dtype)[..., None]
            self.last_weights, self.last_generated = w.detach()[..., 0], gen.detach()   # read by where.py
            out = (w * cands).sum(-2)
            if self.regions == "itc":
                out = torch.where(self.generated_region[:, None], gen, out)
        return F.layer_norm(out, (S.WIDTH,)), h, gen


def quantize(x, codes, chunk=16384):
    flat = x.reshape(-1, x.shape[-1]).float()
    idx = torch.cat([torch.cdist(flat[i:i + chunk], codes).argmin(-1) for i in range(0, len(flat), chunk)])
    return idx.view(x.shape[:-1])


def rollout_losses(world, s, a, loss, gen_loss=False):
    """spatial.rollout + the dynamics L1 (suffix) or teacher-forced L1 only; gen_loss: + the generate candidate's
    own teacher-forced L1 on every token."""
    a = F.pad(a, (0, 1))
    predicted, history, gen = world(s, a)
    teacher = (predicted[:, :S.W - 1] - s[:, 1:]).abs().mean()
    if gen_loss:
        teacher = teacher + (F.layer_norm(gen[:, :S.W - 1], (S.WIDTH,)) - s[:, 1:]).abs().mean()
    if loss == "teacher":
        return teacher
    first = predicted[:, S.ANCHOR]
    second_all, _, _ = world(torch.cat([s[:, :S.ANCHOR + 1], first[:, None].to(s.dtype)], 1), a[:, :S.ANCHOR + 2])
    generated = torch.stack([first, second_all[:, S.ANCHOR + 1]], 1)
    return teacher + (generated - s[:, S.ANCHOR + 1:]).abs().mean()


def train(head, pool_name, loss, seed, updates, device, log, codebook=None, backbone="full", gen_loss=False,
          regions="all", snapshot=None):
    from d4mj.config import config_from_dict
    from d4mj.train import _phase_lr, autocast_context, optimizer_step, phase_optimizer
    config = config_from_dict(torch.load(S.CHECKPOINT, map_location="cpu", weights_only=False)["config"])
    pool = torch.load(POOLS[pool_name] / "pool.pt", weights_only=False, mmap=True)
    main_rows = torch.where(~pool["terminal"])[0]
    held = main_rows[torch.randperm(len(main_rows), generator=torch.Generator().manual_seed(1))[:2048]]
    rows = torch.cat([main_rows[~torch.isin(main_rows, held)], torch.where(pool["terminal"])[0]])
    with torch.random.fork_rng(devices=[0]):
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        world = TWorld(head, codebook, backbone, regions).to(device)
    log(stage="init", parameters=sum(p.numel() for p in world.parameters()))
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
                objective = rollout_losses(world, s, a, loss, gen_loss)
        norm = optimizer_step(opt, objective, params, learning_rate=_phase_lr(config, update),
                              grad_clip=config.agent.grad_clip, strict=True, zero_grad=True)
        if snapshot is not None and (update + 1) % 6000 == 0 and update + 1 < updates:
            snapshot(update + 1, world, history)          # held-out learning curve: every 6k, evaluated by teval
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
    parser.add_argument("--backbone", default="full", choices=("full", "fattn", "fmamba", "fcanvas", "fscan"))
    parser.add_argument("--gen-loss", action="store_true")
    parser.add_argument("--regions", default="all", choices=("all", "itc"))
    parser.add_argument("--snapshots", action="store_true", help="also save the world every 6,000 updates")
    args = parser.parse_args(argv)
    started = time.time()
    log = lambda **kw: print(json.dumps({**kw, "seconds_total": round(time.time() - started, 1)}), flush=True)
    device = torch.device("cuda")
    codebook = torch.load(args.codebook, weights_only=False)["codes"] if args.codebook else None
    name = f"{args.head}_{args.pool}_{args.loss}_s{args.seed}" + (f"_K{len(codebook)}" if codebook is not None else "") \
        + ("" if args.backbone == "full" else f"_{args.backbone}") + ("_gl" if args.gen_loss else "") \
        + ("" if args.regions == "all" else f"_{args.regions}") + ("" if args.updates == 6000 else f"_u{args.updates}")
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / f"{name}.pt").exists():
        log(status="exists", name=name)
        return 0
    from d4mj.data import _sha256
    save = lambda tag, w, h: torch.save({"name": tag, "args": {k: str(v) for k, v in vars(args).items()},
                                         "world": w.state_dict(), "history": h,
                                         "script_sha256": _sha256(Path(__file__))}, OUT / f"{tag}.pt")
    snapshot = (lambda u, w, h: save(f"{name}_at{u}", w, h)) if args.snapshots else None
    world, history, held = train(args.head, args.pool, args.loss, args.seed, args.updates, device, log, codebook,
                                 args.backbone, args.gen_loss, args.regions, snapshot)
    save(name, world, history)
    log(status="saved", name=name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
