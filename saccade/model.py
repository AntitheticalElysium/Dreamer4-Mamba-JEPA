"""M0 networks (saccade/design/M0_CODE.md; STEP_SPEC §7): perception, the Mamba belief, the space-time denoiser with
shortcut forcing (canonical since P1-F4, 11-07), and the grounding and outcome heads. Every number comes from saccade/m0.yaml (cfg); reasons live in the specs.
Shapes: N = 81 periphery cells (9 x 9, rows 7-8 the HUD), G = 9 glimpse cells (a 3 x 3 window), K memory tokens.
"""
import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

GRID, CELL, N, G = 9, 7, 81, 9
SYMLOG = lambda x: torch.sign(x) * torch.log1p(x.abs())


def mlp(sizes, act=nn.GELU):
    layers = []
    for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:])):
        layers += [nn.Linear(a, b)] + ([act()] if i < len(sizes) - 2 else [])
    return nn.Sequential(*layers)


def out_map(x: Tensor, s: float) -> Tensor:
    """Token scale (LOSSES principle 7): layer norm over the token's channels without affine, then C(z) = s tanh(z / s)."""
    return s * torch.tanh(F.layer_norm(x.float(), x.shape[-1:]) / s)


def window(fix: Tensor) -> Tensor:
    """Fixation centres [..., 2] (row, col) -> the 9 cell indices of their 3 x 3 window, row-major [..., 9]."""
    d = torch.arange(-1, 2, device=fix.device)
    r, c = fix[..., 0, None, None] + d[:, None], fix[..., 1, None, None] + d[None, :]
    return (r * GRID + c).flatten(-2)


def crop(frames: Tensor, fix: Tensor) -> Tensor:
    """uint8 frames [..., 63, 63, 3] -> the full-resolution 3 x 3-cell crop at the fixation [..., 21, 21, 3]."""
    lead, f, x = frames.shape[:-3], frames.reshape(-1, 63, 63, 3), fix.reshape(-1, 2)
    a = torch.arange(3 * CELL, device=frames.device)
    r, c = (x[:, 0, None] - 1) * CELL + a, (x[:, 1, None] - 1) * CELL + a
    return f[torch.arange(len(f), device=f.device)[:, None, None], r[:, :, None], c[:, None, :]].reshape(*lead, 21, 21, 3)


def cell_pixels(crop_: Tensor) -> Tensor:
    """[..., 21, 21, 3] -> per cell of the window [..., 9, 147] (7 x 7 x 3 each, cells row-major)."""
    x = crop_.reshape(*crop_.shape[:-3], 3, CELL, 3, CELL, 3).transpose(-4, -3)
    return x.reshape(*crop_.shape[:-3], G, CELL * CELL * 3)


@dataclass
class Percept:
    p: Tensor                                                            # [..., 81, 16] periphery tokens
    g: Tensor                                                            # [..., 9, 64] glimpse tokens, window order
    v: Tensor                                                            # [..., 8] lighting latent
    fix: Tensor                                                          # [..., 2] fixation centre (row, col)

    def parts(self):
        return self.p, self.g, self.v

    def replace(self, p, g, v):
        return Percept(p, g, v, self.fix)


class Perception(nn.Module):
    """Lighting latent from pooled frame statistics; periphery = shared MLP on each cell at reduced acuity, FiLM by the
    lighting latent; fovea = small conv on the full-resolution crop, one token per cell; every token through out_map."""

    def __init__(self, cfg):
        super().__init__()
        pp, pl, pf = (cfg["perception"][k] for k in ("periphery", "lighting", "fovea"))
        self.acuity, self.s = pp["acuity"], pp["tanh_clamp"]
        h = pp["hidden"]
        self.light = mlp([6, *pl["hidden"], pl["dim"]])
        self.p_layers = nn.ModuleList([nn.Linear(3 * self.acuity ** 2, h[0]), nn.Linear(h[0], h[1])])
        self.p_out = nn.Linear(h[1], pp["token_dim"])
        self.film = nn.ModuleList([nn.Linear(pl["dim"], 2 * w) for w in h])
        for f in self.film:
            nn.init.zeros_(f.weight), nn.init.zeros_(f.bias)
        ch = pf["channels"]
        self.fovea = nn.Sequential(nn.Conv2d(3, ch, 3, padding=1), nn.GELU(), nn.Conv2d(ch, ch, 3, padding=1), nn.GELU(),
                                   nn.Conv2d(ch, ch, CELL, stride=CELL), nn.GELU(),
                                   nn.Conv2d(ch, pf["token_dim"], 3, padding=1))

    def forward(self, frames: Tensor, fix: Tensor) -> Percept:
        lead = frames.shape[:-3]
        f = frames.reshape(-1, 63, 63, 3)
        x = f.float() / 255
        v = out_map(self.light(torch.cat([x.mean((1, 2)), x.std((1, 2))], -1)), self.s)
        cells = f.reshape(-1, GRID, CELL, GRID, CELL, 3).permute(0, 1, 3, 5, 2, 4).reshape(-1, 3, CELL, CELL).float()
        cells = torch.round(F.adaptive_avg_pool2d(cells, self.acuity)) if self.acuity < CELL else cells
        h = cells.reshape(len(f), N, -1) / 127.5 - 1
        for layer, film in zip(self.p_layers, self.film):
            gain, shift = film(v).unsqueeze(1).chunk(2, -1)
            h = F.gelu(layer(h)) * (1 + gain) + shift
        p = out_map(self.p_out(h), self.s)
        c = crop(f, fix.reshape(-1, 2)).permute(0, 3, 1, 2).float() / 127.5 - 1
        g = out_map(self.fovea(c).flatten(2).transpose(1, 2), self.s)
        return Percept(p.reshape(*lead, N, -1), g.reshape(*lead, G, -1), v.reshape(*lead, -1), fix)


# ---- belief: Mamba-2 (ported from d4mj/mamba_recurrence.py) ----

@dataclass(frozen=True)
class MambaCarry:
    conv: Tensor                                                         # B, inner + 2 d_state, d_conv (newest right)
    ssm: Tensor                                                          # B, heads, headdim, d_state; FP32


class FunctionalMamba2(nn.Module):
    """d4mj's functional carry around upstream Mamba2 (state-spaces/mamba; Tri Dao, Albert Gu; Apache-2.0), extended
    with episode starts inside a chunk (CURRICULUM §1): seq_idx is LOCAL, 0 for the segment continuing the carried
    state and incremented at each episode start (the installed kernel drops the carried state when the first index
    differs from 0); the short convolution is reset at the same starts (the kernel's seq_idx does not reach it)."""

    def __init__(self, b):
        super().__init__()
        from mamba_ssm.modules.mamba2 import Mamba2

        self.backend = b["backend"]
        self.core = Mamba2(
            d_model=b["width"], d_state=b["d_state"], d_conv=b["d_conv"], conv_init=None, expand=b["expand"],
            headdim=b["headdim"], d_ssm=None, ngroups=b["ngroups"], A_init_range=tuple(b["A_init_range"]),
            D_has_hdim=False, rmsnorm=True, norm_before_gate=b["norm_before_gate"], dt_min=b["dt_range"][0],
            dt_max=b["dt_range"][1], dt_init_floor=b["dt_init_floor"], dt_limit=(0.0, float("inf")), bias=b["bias"],
            conv_bias=b["conv_bias"], chunk_size=b["chunk_size"], use_mem_eff_path=False, layer_idx=0,
            process_group=None, sequence_parallel=False)

    def initial(self, batch: int, device=None) -> MambaCarry:
        c, dev = self.core, device or self.core.in_proj.weight.device
        return MambaCarry(torch.zeros(batch, c.d_ssm + 2 * c.d_state, c.d_conv, device=dev),
                          torch.zeros(batch, c.nheads, c.headdim, c.d_state, device=dev))

    def forward(self, inputs: Tensor, carry: MambaCarry, seq_idx: Tensor | None = None, backend: str | None = None):
        c, K = self.core, self.core.d_conv
        z, xbc, dt = torch.split(c.in_proj(inputs), [c.d_ssm, c.d_ssm + 2 * c.d_state, c.nheads], dim=-1)
        raw = xbc.transpose(1, 2)
        full = torch.cat((carry.conv.to(raw.dtype), raw), dim=-1)                    # B, C, K + L
        win = full[:, :, 1:].unfold(-1, K, 1)                                         # B, C, L, K: inputs t-K+1..t
        final_conv = full[:, :, -K:]
        if seq_idx is not None:
            seg = F.pad(seq_idx, (K, 0))                                              # carried inputs: segment 0
            win = win * (seg[:, 1:].unfold(-1, K, 1) == seq_idx[..., None])[:, None].to(win.dtype)
            final_conv = final_conv * (seg[:, -K:] == seq_idx[:, -1:])[:, None].to(win.dtype)
        filtered = F.silu((win * c.conv1d.weight[:, 0, None, :]).sum(-1) + c.conv1d.bias[:, None]).transpose(1, 2)
        x, b, cc = torch.split(filtered, [c.d_ssm, c.d_state, c.d_state], dim=-1)
        x = x.reshape(*x.shape[:2], c.nheads, c.headdim)
        if (backend or self.backend) == "reference":
            y, final_ssm = self._reference_ssm(x, b, cc, dt, carry.ssm, seq_idx)
        else:
            from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined

            y, final_ssm = mamba_chunk_scan_combined(
                x, dt, -c.A_log.float().exp(), b.unsqueeze(2), cc.unsqueeze(2), chunk_size=c.chunk_size, D=c.D,
                dt_bias=c.dt_bias, initial_states=carry.ssm,
                seq_idx=None if seq_idx is None else seq_idx.to(torch.int32).contiguous(), dt_softplus=True,
                return_final_states=True, state_dtype=torch.float32)
        # Source RMSNormGated(norm_before_gate=False): norm(y * silu(z)).
        gated = y.flatten(2).float() * F.silu(z.float())
        normalized = gated * torch.rsqrt(gated.square().mean(-1, keepdim=True) + c.norm.eps) * c.norm.weight.float()
        return c.out_proj(normalized.to(z.dtype)), MambaCarry(final_conv.clone(), final_ssm.float())

    def _reference_ssm(self, x, b, c, dt, initial, seq_idx):
        core = self.core
        with torch.autocast(device_type=x.device.type, enabled=False):
            x, b, c = x.float(), b.float(), c.float()
            delta = F.softplus(dt.float() + core.dt_bias.float())
            a = -core.A_log.float().exp()
            state, outputs = initial.float(), []
            for t in range(x.shape[1]):
                if seq_idx is not None:
                    prev = seq_idx[:, t - 1] if t else torch.zeros_like(seq_idx[:, 0])
                    state = state * (seq_idx[:, t] == prev).float()[:, None, None, None]
                decay = (delta[:, t] * a).exp()[:, :, None, None]
                state = decay * state + delta[:, t, :, None, None] * x[:, t, :, :, None] * b[:, t, None, None, :]
                outputs.append((state * c[:, t, None, None, :]).sum(-1) + core.D[None, :, None].float() * x[:, t])
            return torch.stack(outputs, 1), state


class SpatialFusion(nn.Module):
    """Joint coarse/fine evidence in crop coordinates; replaces the raw fine-to-belief path."""

    def __init__(self, periphery_dim, fine_dim, lighting_dim, hidden, clamp):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(periphery_dim + fine_dim + lighting_dim, hidden), nn.GELU(),
                                 nn.Linear(hidden, fine_dim))
        self.clamp = clamp

    def forward(self, P: Percept, cells: Tensor) -> Tensor:
        index = cells[..., None].expand(*cells.shape, P.p.shape[-1])
        coarse = P.p.gather(-2, index)
        light = P.v.unsqueeze(-2).expand(*P.g.shape[:-1], P.v.shape[-1])
        return out_map(self.net(torch.cat([coarse, P.g, light], -1)), self.clamp)


class Belief(nn.Module):
    """Mamba over the interleaved sequence [percept_t + reward + termination], [a_t] (STEP_SPEC 7). The output after
    the percept token is the corrected belief b_t (what the actor reads); after the action token, the predicted belief
    of t + 1, projected to K memory tokens for the denoiser. The percept enters flattened and linearly projected
    (DRAMA); glimpse tokens are placed on a 9 x 9 canvas at their fixation's cells, with a mask, so the projection
    has weights per screen position."""

    def __init__(self, cfg):
        super().__init__()
        b, pc = cfg["belief"], cfg["perception"]
        W, self.K, Wd = b["width"], cfg["denoiser"]["memory_tokens"], cfg["denoiser"]["width"]
        self.percept_in = nn.Linear(N * (pc["periphery"]["token_dim"] + pc["fovea"]["token_dim"] + 1)
                                    + pc["lighting"]["dim"], W)
        self.outcome_in = nn.Linear(2, W)
        self.action = nn.Embedding(cfg["data"]["actions"], W)
        self.norms = nn.ModuleList([nn.RMSNorm(W, eps=b["norm_eps"]) for _ in range(b["layers"])])
        self.mixers = nn.ModuleList([FunctionalMamba2(b) for _ in range(b["layers"])])
        self.final_norm = nn.RMSNorm(W, eps=b["norm_eps"])
        self.memory_out = nn.Linear(W, self.K * Wd)
        fusion = b["spatial_fusion"]                                     # P1-F1.6, canonical since F2
        assert fusion["kind"] == "joint_fine", "unknown spatial fusion kind"
        with torch.random.fork_rng(devices=[]):                          # its own draws: every other parameter keeps the
            torch.random.default_generator.manual_seed(fusion["init_seed"])   # earlier initial draws
            self.spatial_fusion = SpatialFusion(pc["periphery"]["token_dim"], pc["fovea"]["token_dim"],
                                                pc["lighting"]["dim"], fusion["hidden"], pc["periphery"]["tanh_clamp"])

    def initial(self, batch: int, device=None):
        return tuple(m.initial(batch, device) for m in self.mixers)

    def percept_embed(self, P: Percept) -> Tensor:
        cells = window(P.fix)
        fine = self.spatial_fusion(P, cells)
        canvas = fine.new_zeros(*fine.shape[:-2], N, fine.shape[-1]).scatter(-2, cells[..., None].expand_as(fine), fine)
        mask = P.g.new_zeros(*P.g.shape[:-2], N).scatter(-1, cells, 1.0)
        return self.percept_in(torch.cat([P.p.flatten(-2), canvas.flatten(-2), mask, P.v], -1))

    def tokens(self, embed: Tensor, reward: Tensor, terminated: Tensor, action: Tensor) -> Tensor:
        """[B, T, W] percept embeddings + outcomes, actions [B, T] -> interleaved [B, 2T, W]."""
        pt = embed + self.outcome_in(torch.stack([SYMLOG(reward), terminated.float()], -1).to(embed.dtype))
        return torch.stack([pt, self.action(action).to(pt.dtype)], 2).flatten(1, 2)

    def forward(self, x: Tensor, carry, seq_idx: Tensor | None = None, backend: str | None = None):
        out = []
        for norm, mixer, c in zip(self.norms, self.mixers, carry):
            y, c = mixer(norm(x), c, seq_idx, backend)
            x = x + y
            out.append(c)
        return self.final_norm(x), tuple(out)

    def memory(self, b: Tensor) -> Tensor:
        return self.memory_out(b).unflatten(-1, (self.K, -1))


def carry_rows(carry, rows):
    return tuple(MambaCarry(c.conv[rows], c.ssm[rows]) for c in carry)


def carry_put(carry, rows, part):
    for c, p in zip(carry, part):
        c.conv[rows], c.ssm[rows] = p.conv.detach().to(c.conv.dtype), p.ssm.detach()


# ---- denoiser: Dreamer-4-style space-time transformer, adaLN-Zero (DiT), shortcut forcing ----

def rope(x: Tensor, ang: Tensor) -> Tensor:
    """Rotate adjacent channel pairs of x [..., n, hd] by angles [..., n, hd / 2]."""
    x2 = x.float().unflatten(-1, (-1, 2))
    cos, sin = ang.cos(), ang.sin()
    out = torch.stack([x2[..., 0] * cos - x2[..., 1] * sin, x2[..., 0] * sin + x2[..., 1] * cos], -1)
    return out.flatten(-2).to(x.dtype)


class Attention(nn.Module):
    def __init__(self, W, H):
        super().__init__()
        self.H, self.hd = H, W // H
        self.qkv, self.out = nn.Linear(W, 3 * W, bias=False), nn.Linear(W, W)
        self.qn, self.kn = nn.RMSNorm(self.hd), nn.RMSNorm(self.hd)          # QK-norm (Dreamer 4)

    def project(self, h: Tensor, ang: Tensor):
        """h [..., n, W], angles [..., n, hd / 2] -> q, k, v [..., H, n, hd] (RoPE on q and k)."""
        q, k, v = (t.transpose(-3, -2) for t in self.qkv(h).unflatten(-1, (3, self.H, self.hd)).unbind(-3))
        a = ang.unsqueeze(-3)                                                # [..., H, n, hd]
        return rope(self.qn(q), a), rope(self.kn(k), a), v

    def attend(self, q, k, v, mask=None):
        """q [..., H, n, hd], k / v [..., H, m, hd], mask broadcastable to [..., 1, n, m]; flattened to 4-D for SDPA."""
        lead = q.shape[:-3]
        q4, k4, v4 = (t.reshape(-1, *t.shape[-3:]) for t in (q, k, v))
        m4 = None if mask is None else mask.expand(*lead, 1, q.shape[-2], k.shape[-2]).reshape(-1, 1, q.shape[-2], k.shape[-2])
        o = F.scaled_dot_product_attention(q4, k4, v4, attn_mask=m4).reshape(*lead, *q.shape[-3:])
        return self.out(o.transpose(-3, -2).flatten(-2))


class Block(nn.Module):
    """Pre-norm RMSNorm block with adaLN-Zero modulation; space-only or dense temporal attention (Dreamer 4), SwiGLU MLP."""

    def __init__(self, W, H, hidden, temporal):
        super().__init__()
        self.temporal = temporal
        self.n1, self.n2 = nn.RMSNorm(W, elementwise_affine=False), nn.RMSNorm(W, elementwise_affine=False)
        self.attn = Attention(W, H)
        self.gate_up, self.down = nn.Linear(W, 2 * hidden, bias=False), nn.Linear(hidden, W, bias=False)
        self.mod = nn.Linear(W, 6 * W)
        nn.init.zeros_(self.mod.weight), nn.init.zeros_(self.mod.bias)

    def mlp(self, h):
        a, b = self.gate_up(h).chunk(2, -1)
        return self.down(F.silu(a) * b)

    def forward(self, x, c, ang, mask=None, cache=None, phys=None, pkv=None):
        """x [B, T, n, W]. c = the (physical, glimpse) conditionings [B, T, W], applied to their token groups (glimpse
        tokens N:N+G, the rest physical; TEACHING 10 / 10a), or one conditioning in a glimpse-only pass (x = the 9
        glimpse tokens). Spatial: ang [B, T, n, hd/2]; with phys (the physical token indices) the physical queries attend
        only to physical keys, whose keys / values are returned; pkv (glimpse-only pass): those keys / values, around
        which the glimpse's own are placed in the full frame's order. Temporal: dense_time (mask = its window, cache =
        streaming, pkv = this frame's full keys / values in a glimpse-only pass)."""
        if isinstance(c, tuple):
            (sh1, sc1, g1, sh2, sc2, g2), mg = (self.mod(F.silu(z)).unsqueeze(2).chunk(6, -1) for z in c)
            grp = lambda t, f: torch.cat([f(t[:, :, :N], 0), f(t[:, :, N:N + G], 1), f(t[:, :, N + G:], 0)], 2)
            pick = lambda i, j: (sh1, sc1, g1, sh2, sc2, g2)[i] if j == 0 else mg[i]
            h = grp(self.n1(x), lambda t, j: t * (1 + pick(1, j)) + pick(0, j))
        else:
            sh1, sc1, g1, sh2, sc2, g2 = self.mod(F.silu(c)).unsqueeze(2).chunk(6, -1)
            h = self.n1(x) * (1 + sc1) + sh1
        new = None
        if not self.temporal:
            q, k, v = self.attn.project(h, ang)
            if pkv is not None:
                k, v = (torch.cat([pt[..., :N, :], t, pt[..., N:, :]], -2) for pt, t in zip(pkv, (k, v)))
            if phys is None:                                                 # glimpse-only pass: every key
                a = self.attn.attend(q, k, v)
            else:
                sel = lambda t: t.index_select(-2, phys)
                new = sel(k), sel(v)
                ap, ag = self.attn.attend(sel(q), *new), self.attn.attend(q[..., N:N + G, :], k, v)
                a = torch.cat([ap[:, :, :N], ag, ap[:, :, N:]], 2)
        else:
            a, new = self.dense_time(h, ang, mask, cache, phys, pkv)
        if isinstance(c, tuple):
            x = x + grp(a, lambda t, j: t * pick(2, j))
            return x + grp(self.mlp(grp(self.n2(x), lambda t, j: t * (1 + pick(4, j)) + pick(3, j))), lambda t, j: t * pick(5, j)), new
        x = x + g1 * a
        return x + g2 * self.mlp(self.n2(x) * (1 + sc2) + sh2), new

    def dense_time(self, h, ang, win=None, cache=None, phys=None, cur=None):
        """Dense temporal attention (P1-F4): Dreamer 4's long-context layers before its "time factorized long context"
        step (2509.24527 Table 2, FVD 71 -> 91 when factorised): every token of a frame attends every token of its own
        frame and of the window's earlier frames, with 1D RoPE over frame positions; physical queries read physical keys
        only (decision 1, as in the spatial layers). h [B, T, n, W], ang [B, T, hd/2]. Chunked: win = (frame index
        [T, w + 1] oldest first, allowed [B, T, w + 1]). Streaming (T = 1): cache = this layer's (k, v) of the committed
        frames [B, F, H, n, hd] (or None); this frame's (k, v) is returned to commit. Glimpse-only pass (h = the 9
        glimpse tokens): cur = this frame's full (k, v) from the full step, its glimpse rows replaced."""
        q, k, v = self.attn.project(h, ang[:, :, None].expand(-1, -1, h.shape[2], -1))   # [B, T, H, n, hd]
        if cur is not None:
            k, v = (torch.cat([c[..., :N, :], t, c[..., N + G:, :]], -2) for c, t in zip(cur, (k, v)))
        new, n = (k, v), k.shape[-2]
        if win is not None:                                              # [B, H, T, (w + 1) n, hd] from shifted slices of
            idx, ok = win                                                # the front-padded frames (slot j = frame t - (w - j)):
            w, T = idx.shape[1] - 1, idx.shape[0]                        # an index gather's backward would accumulate the
            pad = lambda t: F.pad(t.transpose(1, 2), (0, 0, 0, 0, w, 0))  # repeated frames with atomics (not bitwise)
            kw, vw = (torch.stack([p[:, :, j:j + T] for j in range(w + 1)], 3).flatten(3, 4) for p in (pad(k), pad(v)))
            allowed = ok[:, None, :, None].repeat_interleave(n, -1)         # [B, 1, T, 1, (w + 1) n]: broadcast, never expanded
        else:
            if cache is not None and cache[0] is not None:
                k, v = torch.cat([cache[0], k], 1), torch.cat([cache[1], v], 1)         # [B, F + 1, H, n, hd]
            kw, vw = (t.permute(0, 2, 1, 3, 4).flatten(2, 3)[:, :, None] for t in (k, v))   # [B, H, 1, (F + 1) n, hd]
            allowed = None
        qh = q.transpose(1, 2)                                           # [B, H, T, n, hd]

        def attend(qq, mask):                                            # SDPA on 4-D [B H, T, ., hd] (fused kernels)
            B, H, T = kw.shape[:3]
            m = None if mask is None else mask.expand(B, H, T, 1, kw.shape[-2]).reshape(B * H, T, 1, -1)
            o = F.scaled_dot_product_attention(qq.reshape(B * H, T, *qq.shape[-2:]), kw.reshape(B * H, T, *kw.shape[-2:]),
                                               vw.reshape(B * H, T, *vw.shape[-2:]), attn_mask=m)
            return self.attn.out(o.unflatten(0, (B, H)).permute(0, 2, 3, 1, 4).flatten(-2))
        if cur is not None:                                              # glimpse-only pass: its queries read every key
            return attend(qh, allowed), new
        physical = torch.zeros(n, dtype=torch.bool, device=h.device).index_fill(0, phys, True).repeat(kw.shape[-2] // n)
        ap = attend(qh.index_select(-2, phys), physical if allowed is None else allowed & physical)
        ag = attend(qh[..., N:N + G, :], allowed)
        return torch.cat([ap[:, :, :N], ag, ap[:, :, N:]], 2), new


class Denoiser(nn.Module):
    """Tokens per frame: 81 periphery (at their cells), 9 glimpse (at their fixation's cells), 1 lighting, K memory
    (the predicted belief). Positions: learned absolute 2D + axial 2D RoPE for cell tokens (Heo et al.), none for
    lighting and memory; 1D RoPE over frames in the dense temporal layers. Factorised (decision 1, TEACHING 10 / 10a):
    physical tokens never attend to glimpse tokens, and the two groups have their own noise levels and adaLN
    conditioning (UniDiffuser). Conditioning per frame and group: discrete embeddings of the shortcut signal level and
    step size (Dreamer 4) plus the previous action's embedding (zero at an episode's first frame), projected, by
    adaLN-Zero."""

    def __init__(self, cfg):
        super().__init__()
        d, pc, e = cfg["denoiser"], cfg["perception"], cfg["edm"]
        assert d["factorised"] and d["temporal_attention"] == "dense" and e["objective"] == "shortcut_forcing", \
            "the canonical denoiser (P1-F4, 11-07); earlier recipes run from commit fa6edd1a"
        W, H = d["width"], d["heads"]
        self.W, self.hd, self.K, self.window = W, W // H, d["memory_tokens"], d["temporal_window"]
        dims = (pc["periphery"]["token_dim"], pc["fovea"]["token_dim"], pc["lighting"]["dim"])
        self.inp = nn.ModuleList([nn.Linear(k, W) for k in dims])
        self.pos = nn.Parameter(torch.randn(N, W) * 0.02)
        self.kind = nn.Parameter(torch.randn(4, W) * 0.02)
        nf = d["cond_channels"]
        self.register_buffer("fourier", torch.randn(nf // 2))           # unused since P1-F4 (EDM's noise features); its
        self.action = nn.Embedding(cfg["data"]["actions"], nf)          # draw keeps every later initial draw, and F4's
        self.cond_proj = nn.Sequential(nn.Linear(nf, W), nn.SiLU(), nn.Linear(W, W))   # checkpoints load unchanged
        self.blocks = nn.ModuleList([Block(W, H, d["mlp_hidden"], (i + 1) % d["temporal_every"] == 0) for i in range(d["layers"])])
        self.final_norm = nn.RMSNorm(W, elementwise_affine=False)
        self.final_mod = nn.Linear(W, 2 * W)
        self.out = nn.ModuleList([nn.Linear(W, k) for k in dims])
        for m in (self.final_mod, *self.out):
            nn.init.zeros_(m.weight), nn.init.zeros_(m.bias)
        hd = self.hd
        self.register_buffer("f_space", 1.0 / d["rope_theta_space"] ** (torch.arange(0, hd, 4)[: hd // 4].float() / hd))
        self.register_buffer("f_time", 1.0 / d["rope_theta_time"] ** (torch.arange(0, hd, 2).float() / hd))
        self.checkpoint = False
        self.register_buffer("phys", torch.cat([torch.arange(N), torch.arange(N + G, N + G + 1 + self.K)]), persistent=False)
        self.k_max = e["shortcut"]["k_max"]                              # Dreamer 4: discrete signal-level and step-size
        assert self.k_max >= 2 and self.k_max & (self.k_max - 1) == 0   # embeddings (nicklashansen/dreamer4 model.py:
        with torch.random.fork_rng(devices=[]):                          # signal_embed k_max + 1, step_embed log2 + 1),
            torch.random.default_generator.manual_seed(e["shortcut"]["init_seed"])   # their own draws
            self.signal_embed = nn.Embedding(self.k_max + 1, nf)
            self.step_embed = nn.Embedding(self.k_max.bit_length(), nf)

    def conds(self, code, prev_action):
        """The (physical, glimpse) conditionings from a pair of codes; one code conditions both groups."""
        return tuple(self.cond(z, prev_action) for z in (code if isinstance(code, tuple) else (code, code)))

    def cond(self, code: Tensor, prev_action: Tensor) -> Tensor:
        """code: shortcut_code's integer (signal index, step index); sum = concatenation then cond_proj's linear."""
        a = self.action(prev_action.clamp(min=0)) * (prev_action >= 0)[..., None]
        return self.cond_proj(self.signal_embed(code % (self.k_max + 1)) + self.step_embed(code // (self.k_max + 1)) + a)

    def embed(self, x, memory: Tensor, fix: Tensor):
        """x = (p, g, v) [B, T, ...] -> tokens [B, T, n, W] and spatial RoPE angles."""
        p, g, v = x
        cells = window(fix)                                                  # [B, T, 9]
        tok = torch.cat([self.inp[0](p) + self.pos + self.kind[0], self.inp[1](g) + self.pos[cells] + self.kind[1],
                         (self.inp[2](v) + self.kind[2]).unsqueeze(-2), memory + self.kind[3]], -2)
        allc = torch.arange(N, device=p.device).expand(*p.shape[:-2], N)
        idx = torch.cat([allc, cells], -1)
        ang = torch.cat([(idx % GRID)[..., None] * self.f_space, (idx // GRID)[..., None] * self.f_space], -1)
        ang = torch.cat([ang, ang.new_zeros(*ang.shape[:-2], 1 + self.K, ang.shape[-1])], -2)
        return tok, ang

    def head(self, h: Tensor, c):
        """Each output group read under its own conditioning."""
        (sp, cp), (sg, cg) = (self.final_mod(F.silu(z)).unsqueeze(2).chunk(2, -1) for z in c)
        h = self.final_norm(h)
        hp, hg = h * (1 + cp) + sp, h[:, :, N:N + G] * (1 + cg) + sg
        return self.out[0](hp[:, :, :N]), self.out[1](hg), self.out[2](hp[:, :, N + G])

    def forward(self, x, code, prev_action, memory, fix, episode, valid):
        """Chunked (training) form over T frames; returns the outputs x1_hat = (p, g, v). Temporal attention: a frame
        sees itself and the previous `window` frames of its own episode (valid ones)."""
        tok, ang = self.embed(x, memory, fix)
        c = self.conds(code, prev_action)
        T = tok.shape[1]
        t = torch.arange(T, device=tok.device)
        lag = t[:, None] - t[None, :]
        ok = (lag >= 0) & (lag <= self.window) & (episode[:, :, None] == episode[:, None, :]) & valid[:, None, :]
        allowed = ok | torch.eye(T, dtype=torch.bool, device=tok.device)
        idx = t[:, None] - torch.arange(self.window, -1, -1, device=tok.device)   # the rule per window slot, oldest first
        win = (idx.clamp(min=0), (idx >= 0) & allowed.gather(2, idx.clamp(min=0).expand(len(allowed), -1, -1)))
        ang_t = t.float()[:, None] * self.f_time
        for blk in self.blocks:
            a = (ang_t.expand(tok.shape[0], -1, -1), win, None, self.phys) if blk.temporal else (ang, None, None, self.phys)
            if self.checkpoint and self.training:
                tok = checkpoint(lambda z, b=blk, a=a: b(z, c, *a)[0], tok, use_reentrant=False)
            else:
                tok = blk(tok, c, *a)[0]
        return self.head(tok, c)

    def step(self, x, code, prev_action, memory, fix, cache: dict, keep=None):
        """Streaming form for one frame per stream (T = 1) at frame position cache["t"]; cache["kv"] holds, per temporal
        layer, the keys / values of the previous <= `window` committed frames. Returns (outputs, kv of this frame);
        commit() appends them. With a list `keep`: every layer's keys / values of this frame (spatial: the physical
        tokens'), in block order, for glimpse_step."""
        tok, ang = self.embed(x, memory, fix)
        c = self.conds(code, prev_action)
        ang_t = torch.full((tok.shape[0], 1, 1), float(cache["t"]), device=tok.device) * self.f_time
        new, i = [], 0
        for blk in self.blocks:
            if blk.temporal:
                tok, kv = blk(tok, c, ang_t, None, cache["kv"][i], self.phys)
                new.append(kv)
                if keep is not None:
                    keep.append(kv)
                i += 1
            else:
                tok, pkv = blk(tok, c, ang, None, None, self.phys)
                if keep is not None:
                    keep.append(pkv)
        return self.head(tok, c), new

    def glimpse_step(self, g, code, prev_action, fix, cache: dict, kept):
        """The glimpse tokens alone: the physical tokens never read them, so with the physical inputs and level held (the
        cascade's stage 2) their per-layer keys / values are those `kept` by one step(). Returns the glimpse output;
        equal to step()'s at the same inputs."""
        cells = window(fix)
        tok = self.inp[1](g) + self.pos[cells] + self.kind[1]
        ang = torch.cat([(cells % GRID)[..., None] * self.f_space, (cells // GRID)[..., None] * self.f_space], -1)
        c = self.cond(code, prev_action)
        ang_t = torch.full((tok.shape[0], 1, 1), float(cache["t"]), device=tok.device) * self.f_time
        i = j = 0
        for blk in self.blocks:
            if blk.temporal:
                tok, i, j = blk(tok, c, ang_t, None, cache["kv"][i], pkv=kept[j])[0], i + 1, j + 1
            else:
                tok, j = blk(tok, c, ang, pkv=kept[j])[0], j + 1
        shift, scale = self.final_mod(F.silu(c)).unsqueeze(2).chunk(2, -1)
        return self.out[1](self.final_norm(tok) * (1 + scale) + shift)

    def new_cache(self):
        return {"t": 0, "kv": [(None, None) for b in self.blocks if b.temporal]}

    def commit(self, cache: dict, new) -> dict:
        kv = []
        for (k0, v0), (k, v) in zip(cache["kv"], new):                  # [B, frames, H, n, hd]: every token of a frame
            k, v = (k, v) if k0 is None else (torch.cat([k0, k], 1), torch.cat([v0, v], 1))
            kv.append((k[:, -self.window:], v[:, -self.window:]))
        return {"t": cache["t"] + 1, "kv": kv}


# ---- shortcut forcing (P1-F4): Dreamer 4 (2509.24527) eq. 6-8 and section 3.2, as implemented by both public
# reimplementations (nicklashansen/dreamer4 train_dynamics.py: dynamics_pretrain_loss, sample_one_timestep_packed;
# edwhu/dreamer4-jax scripts/train_dynamics.py). Signal level tau in [0, 1) on the grid of a step size d = 2^-e:
# z_tau = (1 - tau) z0 + tau x1 (z0 ~ N(0, I), tau = 0 pure noise); the network predicts x1 (x-prediction) from z_tau,
# conditioned on the signal index tau k_max and the step index e (e_max = log2 k_max: the finest step d_min). ----

def _bcast(c: Tensor, x: Tensor) -> Tensor:
    return c.reshape(*c.shape, *([1] * (x.ndim - c.ndim)))


def _parts(pair):
    """A (physical, glimpse) pair -> the value of each percept part (p, g, v)."""
    return pair[0], pair[1], pair[0]


def shortcut_code(signal, step, k_max):
    """(signal index 0..k_max, step index 0..log2 k_max) -> the integer code Denoiser.cond reads."""
    return step * (k_max + 1) + signal


def shortcut_levels(shape, k_max, bootstrap, device, gen=None):
    """Per frame: the step index (finest on flow rows; uniform over the coarser ones on bootstrap rows, `bootstrap` [B]
    bool) and a signal index on that step's grid (j uniform in 0..2^e - 1, tau = j / 2^e, index j k_max / 2^e)."""
    e_max = k_max.bit_length() - 1
    coarse = torch.randint(0, e_max, shape, device=device, generator=gen)
    e = torch.where(bootstrap.reshape(-1, *([1] * (len(shape) - 1))), coarse, torch.full_like(coarse, e_max))
    K = 2 ** e
    j = (torch.rand(shape, device=device, generator=gen) * K).floor().long()
    return j * (k_max // K), e


def shortcut_loss(den: Denoiser, x0, prev_action, memory, fix, episode, valid, sc, bootstrap, start, levels=None, noise=None):
    """LOSSES G1, shortcut forcing, per frame [B, T] (mean over each frame's percept scalars, every scalar equal; ramp
    weight w(tau) = 0.9 tau + 0.1 per element at its group's level). Rows `bootstrap` [B] bool train the self-consistency
    (bootstrap) term at a coarser step, the others the flow term at the finest step; before the bootstrap start (start
    False) the bootstrap rows contribute 0, as in both reimplementations. Physical (p, v) and glimpse levels are drawn
    independently (decision 1). levels / noise (tests): ((signal, step) per group, z0 per part)."""
    k = sc["k_max"]
    B, T = x0[0].shape[:2]
    levels = levels or [shortcut_levels((B, T), k, bootstrap, x0[0].device) for _ in range(2)]
    noise = noise or tuple(torch.randn_like(t) for t in x0)
    tau = [s.float() / k for s in _parts(tuple(l[0] for l in levels))]
    code = lambda s_, e_: tuple(shortcut_code(a, b, k) for a, b in zip(s_, e_))
    zt = tuple((1 - _bcast(t, x)) * n + _bcast(t, x) * x for x, n, t in zip(x0, noise, tau))
    out = den(zt, code(*zip(*levels)), prev_action, memory, fix, episode, valid)
    w = [sc["ramp"][0] * t + sc["ramp"][1] for t in tau]
    per_frame = lambda terms: sum(e.flatten(2).sum(-1) for e in terms) / sum(t[0, 0].numel() for t in x0)
    flow = per_frame([_bcast(wt, x) * (o.float() - x) ** 2 for o, x, wt in zip(out, x0, w)])
    loss = torch.where(bootstrap[:, None], torch.zeros_like(flow), flow)
    if start and bootstrap.any():
        r = bootstrap
        sub = lambda a: a[r]
        zr, tr, wr = tuple(sub(z).float() for z in zt), [sub(t) for t in tau], [sub(t) for t in w]
        half = [(sub(l[0]), sub(l[1]) + 1) for l in levels]               # d / 2: the next finer step
        plus = [(s_ + k // 2 ** e_, e_) for s_, e_ in half]                # tau + d / 2 on that grid
        args = (sub(prev_action), sub(memory).detach(), sub(fix), sub(episode), sub(valid))
        with torch.no_grad():
            h1 = den(tuple(z.to(t.dtype) for z, t in zip(zr, zt)), code(*zip(*half)), *args)
            b1 = tuple((h.float() - z) / _bcast(1 - t, z) for h, z, t in zip(h1, zr, tr))
            dh = _parts(tuple(2.0 ** -(l[1].float()) for l in half))     # d / 2 per part
            z1 = tuple(z + b * _bcast(d, z) for z, b, d in zip(zr, b1, dh))
            h2 = den(tuple(z.to(t.dtype) for z, t in zip(z1, zt)), code(*zip(*plus)), *args)
            b2 = tuple((h.float() - z) / _bcast(1 - t - d, z) for h, z, t, d in zip(h2, z1, tr, dh))
            target = tuple((a + b) / 2 for a, b in zip(b1, b2))
        vhat = tuple((sub(o).float() - z) / _bcast(1 - t, z) for o, z, t in zip(out, zr, tr))
        boot = per_frame([_bcast(wt * (1 - t) ** 2, z) * (v - g) ** 2 for v, g, z, t, wt in zip(vhat, target, zr, tr, wr)])
        loss = loss.index_put((r.nonzero()[:, 0],), boot)
    return loss


def shortcut_ladder(cfg):
    """The sampler's K = edm.sampler.steps evaluations (a power of two <= k_max, Dreamer 4: 4): (signal index, tau)
    from tau = 0, the step index log2 K, the step size 1 / K."""
    k, K = cfg["edm"]["shortcut"]["k_max"], cfg["edm"]["sampler"]["steps"]
    assert K >= 1 and K & (K - 1) == 0 and K <= k, f"shortcut sampler steps {K} must be a power of two <= k_max {k}"
    return [(i * (k // K), i / K) for i in range(K)], K.bit_length() - 1, 1 / K


def sample(den: Denoiser, shapes, prev_action, memory, fix, cache, cfg, gen=None, choose=None):
    """One frame per stream, Dreamer 4's sampler (nicklashansen sample_one_timestep_packed): from pure noise, K
    evaluations at the step size 1 / K, z <- z + (x1_hat - z) / (1 - tau) d (the last lands on x1_hat), x1_hat clipped
    to its data range. These evaluations never commit to the cache; the finished frame then enters the context once
    (context_pass). The cascade of decision 1: periphery and lighting first (the glimpse tokens, which they never read,
    held at a placeholder), then the glimpse given them, slightly corrupted to the context signal level and labelled at
    the finest step; `choose(p, v) -> fix` may set the fixation between the two stages (TEACHING 14), and is then
    returned beside the frame."""
    ladder, e, d = shortcut_ladder(cfg)
    k, dev, B = den.k_max, memory.device, memory.shape[0]
    clamp = cfg["perception"]["periphery"]["tanh_clamp"]
    randn = lambda shp: torch.randn(shp, device=dev, generator=gen)
    code = lambda s_, e_: torch.full((B, 1), shortcut_code(s_, e_, k), device=dev)
    move = lambda z, x, tau: z + (x.float().clamp(-clamp, clamp) - z) / (1 - tau) * d
    p, v = randn((B, 1, *shapes[0])), randn((B, 1, *shapes[2]))
    g0 = torch.zeros((B, 1, *shapes[1]), device=dev)                   # never read by the physical tokens
    for s_, tau in ladder:
        out, _ = den.step((p, g0, v), code(s_, e), prev_action, memory, fix, cache)
        p, v = move(p, out[0], tau), move(v, out[2], tau)
    chosen = choose(p, v) if choose is not None else None
    fix = fix if chosen is None else chosen
    tc = cfg["edm"]["shortcut"]["context_signal"]
    pc, vc = ((1 - tc / k) * randn(t.shape) + tc / k * t for t in (p, v))
    g, kept = randn((B, 1, *shapes[1])), []
    for i, (s_, tau) in enumerate(ladder):
        if i == 0:                                                       # one full pass keeps the physical keys / values
            xg = den.step((pc, g, vc), (code(tc, k.bit_length() - 1), code(s_, e)), prev_action, memory, fix, cache, kept)[0][1]
        else:                                                            # then the 9 glimpse tokens alone
            xg = den.glimpse_step(g, code(s_, e), prev_action, fix, cache, kept)
        g = move(g, xg, tau)
    return (p, g, v) if chosen is None else ((p, g, v), fix)


def context_pass(den: Denoiser, x, prev_action, memory, fix, cache, cfg, gen=None):
    """Enter a finished frame into the context once (STEP_SPEC 7): Dreamer 4's past inputs "slightly corrupted", one
    fresh draw to the context signal level, labelled at the finest step; one pass in its context role writes its keys
    / values. Returns the new cache."""
    k = den.k_max
    tau = cfg["edm"]["shortcut"]["context_signal"] / k
    xs = tuple((1 - tau) * torch.randn(t.shape, device=t.device, generator=gen) + tau * t for t in x)
    code = torch.full((x[0].shape[0], 1), shortcut_code(cfg["edm"]["shortcut"]["context_signal"], k.bit_length() - 1, k),
                      device=x[0].device)
    _, new = den.step(xs, code, prev_action, memory, fix, cache)
    return den.commit(cache, new)


# ---- heads ----

def log_bin_prob(lo: Tensor, hi: Tensor, mu: Tensor, log_s: Tensor) -> Tensor:
    """log P(lo < X <= hi) for a logistic(mu, s); lo = -inf / hi = +inf give the absorbing edge bins (PixelCNN++)."""
    inv = torch.exp(-log_s)
    lo_inf, hi_inf = torch.isinf(lo), torch.isinf(hi)
    a = (torch.where(hi_inf, mu + 1, hi) - mu) * inv
    b = (torch.where(lo_inf, mu - 1, lo) - mu) * inv
    mid = a + torch.log(-torch.expm1(torch.minimum(b - a, torch.full_like(a, -1e-12)))) - F.softplus(a) - F.softplus(b)
    return torch.where(lo_inf & hi_inf, torch.zeros_like(a),
                       torch.where(lo_inf, -F.softplus(-a), torch.where(hi_inf, -F.softplus(b), mid)))


def pixel_nll(mean: Tensor, log_s: Tensor, pixels: Tensor) -> Tensor:
    """Discretised logistic NLL (PixelCNN++, one component, log-scale floor -7) of uint8 pixels, nats per pixel."""
    y = pixels.float() / 127.5 - 1
    lo = torch.where(pixels == 0, torch.full_like(y, -math.inf), y - 1 / 255)
    hi = torch.where(pixels == 255, torch.full_like(y, math.inf), y + 1 / 255)
    return -log_bin_prob(lo, hi, mean.float(), log_s.float().clamp(min=-7))


class PixelDecoder(nn.Module):
    """Per-cell pixel likelihood shared by TSP and the fovea reconstruction: mean from a per-cell feature, scale from
    the lighting latent and the cell's screen position only (PERCEPTION, TSP noise model)."""

    def __init__(self, cfg):
        super().__init__()
        h, dv = cfg["heads"]["pixel_decoder"], cfg["perception"]["lighting"]["dim"]
        self.cell = nn.Embedding(N, h["position_dim"])
        self.mean = mlp([h["hidden"], h["hidden"], CELL * CELL * 3])
        self.scale = mlp([dv + h["position_dim"], h["hidden"], CELL * CELL * 3])

    def forward(self, feat: Tensor, v: Tensor, cells: Tensor):
        vv = v.unsqueeze(-2).expand(*cells.shape, v.shape[-1])
        return self.mean(feat), self.scale(torch.cat([vv, self.cell(cells)], -1))


class TSP(nn.Module):
    """Trans-saccadic prediction: the glimpse cells' pixels from (predicted belief, periphery tokens of the window, v)
    before the glimpse is read. Also the fovea reconstruction adapter (glimpse tokens, v)."""

    def __init__(self, cfg):
        super().__init__()
        h, pc = cfg["heads"]["pixel_decoder"], cfg["perception"]
        dp, dg, dv = pc["periphery"]["token_dim"], pc["fovea"]["token_dim"], pc["lighting"]["dim"]
        self.cell = nn.Embedding(N, h["position_dim"])
        self.belief = nn.Linear(cfg["belief"]["width"], h["belief_dim"])
        self.tsp = nn.Sequential(mlp([dp + dv + h["position_dim"] + h["belief_dim"], h["hidden"], h["hidden"]]), nn.GELU())
        self.rec = nn.Sequential(mlp([dg + dv + h["position_dim"], h["hidden"], h["hidden"]]), nn.GELU())

    def predict(self, belief, p, v, cells):
        pw = torch.gather(p, -2, cells[..., None].expand(*cells.shape, p.shape[-1]))
        ctx = torch.cat([v, self.belief(belief)], -1).unsqueeze(-2).expand(*cells.shape, -1)
        return self.tsp(torch.cat([pw, self.cell(cells), ctx], -1))

    def reconstruct(self, g, v, cells):
        return self.rec(torch.cat([g, self.cell(cells), v.unsqueeze(-2).expand(*cells.shape, v.shape[-1])], -1))


class ChangeMap(nn.Module):
    """Soft change probability per map cell of frame t from the periphery of t, the 3 x 3 neighbourhood of each cell
    in t - 1 (the camera moves at most one cell) and the predicted belief (LOSSES G2, A9b target)."""

    def __init__(self, cfg):
        super().__init__()
        h, dp = cfg["heads"]["change_map"], cfg["perception"]["periphery"]["token_dim"]
        self.cell = nn.Embedding(7 * GRID, h["position_dim"])
        self.belief = nn.Linear(cfg["belief"]["width"], h["belief_dim"])
        self.net = mlp([10 * dp + h["position_dim"] + h["belief_dim"], h["hidden"], h["hidden"], 1])

    def forward(self, p_prev, p, belief):
        lead = p.shape[:-2]
        grid = p_prev.reshape(-1, GRID, GRID, p.shape[-1]).permute(0, 3, 1, 2)
        nb = F.unfold(grid, 3, padding=1).reshape(-1, p.shape[-1] * 9, GRID, GRID)[:, :, :7].flatten(2).transpose(1, 2)
        x = torch.cat([p.reshape(-1, N, p.shape[-1])[:, : 7 * GRID], nb, self.cell.weight.expand(len(nb), -1, -1),
                       self.belief(belief).reshape(-1, 1, self.belief.out_features).expand(-1, 7 * GRID, -1)], -1)
        return self.net(x).reshape(*lead, 7, GRID)


class Outcome(nn.Module):
    """Reward: a categorical over uniform bins of symlog(r) (LOSSES G3, revised 10-27: the discretised logistic mixture's
    tail gradients, ~1 / scale, dominated the clipped update); a sampled bin decodes to the train-split mean of its
    rewards (buffer bin_value). Continuation: Bernoulli given (belief, percept, reward)."""

    def __init__(self, cfg):
        super().__init__()
        h, r = cfg["heads"]["outcome"], cfg["heads"]["reward"]
        W = cfg["belief"]["width"]
        self.bins = r["bins"]
        self.trunk = mlp([2 * W, h["hidden"], h["hidden"]])
        self.trunk.append(nn.GELU())
        self.reward = nn.Linear(h["hidden"], self.bins)
        self.cont = mlp([h["hidden"] + 1, h["hidden"], 1])
        lo, hi = (math.copysign(math.log1p(abs(x)), x) for x in r["support"])
        self.register_buffer("edges", torch.linspace(lo, hi, self.bins + 1))
        self.register_buffer("bin_value", torch.sign((self.edges[:-1] + self.edges[1:]) / 2)
                             * torch.expm1(((self.edges[:-1] + self.edges[1:]) / 2).abs()))

    def features(self, belief, embed):
        return self.trunk(torch.cat([belief, embed], -1))

    def bin_of(self, r: Tensor) -> Tensor:
        """The LOSSES G3 bin of a raw reward (as saccade.eval.calib.reward_bin)."""
        lo, hi = self.edges[0], self.edges[-1]
        return ((SYMLOG(r.float()) - lo) / (hi - lo) * self.bins).floor().long().clamp(0, self.bins - 1)

    def bin_logp(self, h: Tensor) -> Tensor:
        """log P(bin) for every bin [..., bins]."""
        return self.reward(h).float().log_softmax(-1)

    def reward_logp(self, h: Tensor, r: Tensor) -> Tensor:
        return self.bin_logp(h).gather(-1, self.bin_of(r)[..., None])[..., 0]

    def sample_reward(self, h: Tensor, gen=None) -> Tensor:
        k = torch.multinomial(self.bin_logp(h).exp().reshape(-1, self.bins), 1, generator=gen).reshape(h.shape[:-1])
        return self.bin_value[k]

    def cont_logit(self, h: Tensor, r: Tensor) -> Tensor:
        return self.cont(torch.cat([h, SYMLOG(r.float())[..., None].to(h.dtype)], -1))[..., 0]


class Harmoniser(nn.Module):
    """HarmonyDream: sum over groups of L_g / sigma_g + log(1 + sigma_g), sigma_g learned (LOSSES principle 4)."""

    def __init__(self, groups, init_sigma):
        super().__init__()
        self.groups = list(groups)
        self.log_sigma = nn.Parameter(torch.full((len(self.groups),), math.log(init_sigma)))

    def forward(self, losses: dict) -> Tensor:
        s = self.log_sigma.exp()
        return sum(losses[g] / s[i] + torch.log1p(s[i]) for i, g in enumerate(self.groups))


class M0(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.perception, self.belief, self.denoiser = Perception(cfg), Belief(cfg), Denoiser(cfg)
        self.pixels, self.tsp, self.change, self.outcome = PixelDecoder(cfg), TSP(cfg), ChangeMap(cfg), Outcome(cfg)
        self.harmoniser = Harmoniser(cfg["losses"]["groups"], cfg["losses"]["harmoniser_init_sigma"])

    def param_groups(self):
        """AdamW groups: encoder (perception) at encoder_lr_scale; weight decay only on Linear / Conv weights, as
        DIAMOND's configure_opt (biases, norms, embeddings, raw tables, Mamba's A / D / dt and the harmoniser excluded)."""
        o = self.cfg["optimizer"]
        decayed = {id(m.weight) for m in self.modules() if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d))}
        groups = {}
        for name, prm in self.named_parameters():
            enc = name.startswith("perception.")
            groups.setdefault((enc, id(prm) in decayed), []).append(prm)
        return [{"params": v, "lr_scale": o["encoder_lr_scale"] if enc else 1.0,
                 "weight_decay": o["weight_decay"] if decay else 0.0} for (enc, decay), v in groups.items()]
