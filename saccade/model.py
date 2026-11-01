"""M0 networks (saccade/design/M0_CODE.md; STEP_SPEC §7): perception, the Mamba belief, the space-time denoiser with
EDM, and the grounding and outcome heads. Every number comes from saccade/m0.yaml (cfg); reasons live in the specs.
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

    def initial(self, batch: int, device=None):
        return tuple(m.initial(batch, device) for m in self.mixers)

    def percept_embed(self, P: Percept) -> Tensor:
        cells = window(P.fix)
        canvas = P.g.new_zeros(*P.g.shape[:-2], N, P.g.shape[-1]).scatter(-2, cells[..., None].expand_as(P.g), P.g)
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


# ---- denoiser: Dreamer-4-style space-time transformer, adaLN-Zero (DiT), EDM preconditioning ----

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
    """Pre-norm RMSNorm block with adaLN-Zero modulation; space-only or time-only attention (Dreamer 4), SwiGLU MLP."""

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

    def forward(self, x, c, ang, mask=None, cache=None):
        """x [B, T, n, W], c [B, T, W]. Spatial: ang [B, T, n, hd/2]. Temporal: ang [B, T, hd/2] (frame positions) and
        mask [B, 1, 1, T, T]; with cache (streaming, T = 1) the cached keys / values of earlier frames are prepended and
        this frame's (k, v) is returned for the caller to commit."""
        sh1, sc1, g1, sh2, sc2, g2 = self.mod(F.silu(c)).unsqueeze(2).chunk(6, -1)
        h = self.n1(x) * (1 + sc1) + sh1
        new = None
        if not self.temporal:
            q, k, v = self.attn.project(h, ang)
            a = self.attn.attend(q, k, v)
        else:
            ht = h.transpose(1, 2)                                            # [B, n, T, W]
            q, k, v = self.attn.project(ht, ang[:, None].expand(-1, ht.shape[1], -1, -1))
            if cache is not None:
                new = (k, v)
                if cache[0] is not None:
                    k, v = torch.cat([cache[0], k], -2), torch.cat([cache[1], v], -2)
            a = self.attn.attend(q, k, v, mask).transpose(1, 2)
        x = x + g1 * a
        return x + g2 * self.mlp(self.n2(x) * (1 + sc2) + sh2), new


class Denoiser(nn.Module):
    """Tokens per frame: 81 periphery (at their cells), 9 glimpse (at their fixation's cells), 1 lighting, K memory
    (the predicted belief). Positions: learned absolute 2D + axial 2D RoPE for cell tokens (Heo et al.), none for
    lighting and memory; 1D RoPE over frames in temporal layers. Conditioning per frame: noise level (random Fourier
    features of c_noise plus the previous action's embedding, projected, as DIAMOND; zero action at an episode's
    first frame), by adaLN-Zero."""

    def __init__(self, cfg):
        super().__init__()
        d, pc = cfg["denoiser"], cfg["perception"]
        W, H = d["width"], d["heads"]
        self.W, self.hd, self.K, self.window = W, W // H, d["memory_tokens"], d["temporal_window"]
        dims = (pc["periphery"]["token_dim"], pc["fovea"]["token_dim"], pc["lighting"]["dim"])
        self.inp = nn.ModuleList([nn.Linear(k, W) for k in dims])
        self.pos = nn.Parameter(torch.randn(N, W) * 0.02)
        self.kind = nn.Parameter(torch.randn(4, W) * 0.02)
        nf = d["cond_channels"]
        self.register_buffer("fourier", torch.randn(nf // 2))
        self.action = nn.Embedding(cfg["data"]["actions"], nf)
        self.cond_proj = nn.Sequential(nn.Linear(nf, W), nn.SiLU(), nn.Linear(W, W))
        self.blocks = nn.ModuleList([Block(W, H, d["mlp_hidden"], (i + 1) % d["temporal_every"] == 0)
                                     for i in range(d["layers"])])
        self.final_norm = nn.RMSNorm(W, elementwise_affine=False)
        self.final_mod = nn.Linear(W, 2 * W)
        self.out = nn.ModuleList([nn.Linear(W, k) for k in dims])
        for m in (self.final_mod, *self.out):
            nn.init.zeros_(m.weight), nn.init.zeros_(m.bias)
        hd = self.hd
        self.register_buffer("f_space", 1.0 / d["rope_theta_space"] ** (torch.arange(0, hd, 4)[: hd // 4].float() / hd))
        self.register_buffer("f_time", 1.0 / d["rope_theta_time"] ** (torch.arange(0, hd, 2).float() / hd))
        self.checkpoint = False

    def cond(self, c_noise: Tensor, prev_action: Tensor) -> Tensor:
        f = 2 * math.pi * c_noise[..., None].float() * self.fourier
        a = self.action(prev_action.clamp(min=0)) * (prev_action >= 0)[..., None]
        return self.cond_proj(torch.cat([f.cos(), f.sin()], -1) + a)            # DIAMOND: proj(noise_emb + act_emb)

    def embed(self, x, memory: Tensor, fix: Tensor):
        """x = (p, g, v) [B, T, ...] (already scaled by c_in) -> tokens [B, T, n, W] and spatial RoPE angles."""
        p, g, v = x
        cells = window(fix)                                                  # [B, T, 9]
        tok = torch.cat([self.inp[0](p) + self.pos + self.kind[0], self.inp[1](g) + self.pos[cells] + self.kind[1],
                         (self.inp[2](v) + self.kind[2]).unsqueeze(-2), memory + self.kind[3]], -2)
        allc = torch.arange(N, device=p.device).expand(*p.shape[:-2], N)
        idx = torch.cat([allc, cells], -1)
        ang = torch.cat([(idx % GRID)[..., None] * self.f_space, (idx // GRID)[..., None] * self.f_space], -1)
        ang = torch.cat([ang, ang.new_zeros(*ang.shape[:-2], 1 + self.K, ang.shape[-1])], -2)
        return tok, ang

    def head(self, h: Tensor, c: Tensor):
        shift, scale = self.final_mod(F.silu(c)).unsqueeze(2).chunk(2, -1)
        h = self.final_norm(h) * (1 + scale) + shift
        return self.out[0](h[:, :, :N]), self.out[1](h[:, :, N:N + G]), self.out[2](h[:, :, N + G])

    def forward(self, x, c_noise, prev_action, memory, fix, episode, valid):
        """Chunked (training) form over T frames; returns the raw outputs F = (Fp, Fg, Fv). Temporal attention: a frame
        sees itself and the previous `window` frames of its own episode (valid ones)."""
        tok, ang = self.embed(x, memory, fix)
        c = self.cond(c_noise, prev_action)
        T = tok.shape[1]
        t = torch.arange(T, device=tok.device)
        lag = t[:, None] - t[None, :]
        ok = (lag >= 0) & (lag <= self.window) & (episode[:, :, None] == episode[:, None, :]) & valid[:, None, :]
        mask = (ok | torch.eye(T, dtype=torch.bool, device=tok.device))[:, None, None]
        ang_t = t.float()[:, None] * self.f_time
        for blk in self.blocks:
            a = (ang_t.expand(tok.shape[0], -1, -1), mask) if blk.temporal else (ang, None)
            if self.checkpoint and self.training:
                tok = checkpoint(lambda z, b=blk, a=a: b(z, c, *a)[0], tok, use_reentrant=False)
            else:
                tok = blk(tok, c, *a)[0]
        return self.head(tok, c)

    def step(self, x, c_noise, prev_action, memory, fix, cache: dict):
        """Streaming form for one frame per stream (T = 1) at frame position cache["t"]; cache["kv"] holds, per temporal
        layer, the keys / values of the previous <= `window` committed frames. Returns (F, kv of this frame); commit()
        appends them."""
        tok, ang = self.embed(x, memory, fix)
        c = self.cond(c_noise, prev_action)
        ang_t = torch.full((tok.shape[0], 1, 1), float(cache["t"]), device=tok.device) * self.f_time
        new, i = [], 0
        for blk in self.blocks:
            if blk.temporal:
                tok, kv = blk(tok, c, ang_t, None, cache["kv"][i])
                new.append(kv)
                i += 1
            else:
                tok = blk(tok, c, ang)[0]
        return self.head(tok, c), new

    def new_cache(self):
        return {"t": 0, "kv": [(None, None) for b in self.blocks if b.temporal]}

    def commit(self, cache: dict, new) -> dict:
        kv = []
        for (k0, v0), (k, v) in zip(cache["kv"], new):
            k, v = (k, v) if k0 is None else (torch.cat([k0, k], -2), torch.cat([v0, v], -2))
            kv.append((k[..., -self.window:, :], v[..., -self.window:, :]))
        return {"t": cache["t"] + 1, "kv": kv}


# ---- EDM (Karras et al. 2022): preconditioning, loss, Algorithm 2 sampler ----

def precondition(sigma: Tensor, sd: float = 1.0):
    s2 = sigma ** 2 + sd ** 2
    return sd ** 2 / s2, sigma * sd / s2.sqrt(), 1 / s2.sqrt(), sigma.log() / 4      # c_skip, c_out, c_in, c_noise


def _bcast(c: Tensor, x: Tensor) -> Tensor:
    return c.reshape(*c.shape, *([1] * (x.ndim - c.ndim)))


def denoise_loss(den: Denoiser, x0, sigma, prev_action, memory, fix, episode, valid, sd=1.0, noise=None):
    """LOSSES G1: MSE of the raw output against (x0 - c_skip x) / c_out (DIAMOND; EDM lambda = 1 / c_out^2), mean over
    each frame's percept scalars (every scalar equal) -> [B, T]. x0 = (p, g, v) stop-gradient targets."""
    noise = noise or tuple(torch.randn_like(t) for t in x0)
    c_skip, c_out, c_in, c_noise = precondition(sigma, sd)
    xs = tuple(t + _bcast(sigma, t) * n for t, n in zip(x0, noise))
    Fo = den(tuple(_bcast(c_in, t) * t for t in xs), c_noise, prev_action, memory, fix, episode, valid)
    se = [((f.float() - (t - _bcast(c_skip, t) * s) / _bcast(c_out, t)) ** 2).flatten(2).sum(-1)
          for f, t, s in zip(Fo, x0, xs)]
    return sum(se) / sum(t[0, 0].numel() for t in x0)


def denoised(den: Denoiser, x, sigma, prev_action, memory, fix, cache, s, sd=1.0):
    """One streaming evaluation at sigma [B, 1]; preserve an already valid clean prediction."""
    c_skip, c_out, c_in, c_noise = precondition(sigma, sd)
    Fo, new = den.step(tuple(_bcast(c_in, t) * t for t in x), c_noise, prev_action, memory, fix, cache)
    return clean_estimate(x, Fo, c_skip, c_out, s), new


def clean_estimate(x, raw_output, c_skip, c_out, bound):
    """EDM's trained clean estimate, clipped only to its data range (not normalized again).

    LayerNorm/tanh belongs to the encoder. Applying it again here changes even an exact prediction.
    Shared with the batched S1 sampler so training and both inference paths use the same coordinates.
    """
    return tuple((_bcast(c_skip, t) * t + _bcast(c_out, t) * f.float()).clamp(-bound, bound)
                 for t, f in zip(x, raw_output))


def sample(den: Denoiser, shapes, prev_action, memory, fix, cache, cfg, gen=None):
    """EDM Algorithm 2 (NVIDIA generate.py), order 1, for one frame per stream: sigma_max-scaled start, churn
    gamma = min(S_churn / N, sqrt 2 - 1), the network evaluated at sigma_hat, final step to 0. These evaluations never
    commit to the cache; the finished frame then enters the context once (context_pass)."""
    e, s = cfg["edm"], cfg["edm"]["sampler"]
    n, dev = s["steps"], memory.device
    i = torch.arange(n, device=dev, dtype=torch.float64)
    sig = (s["sigma_max"] ** (1 / s["rho"]) + i / (n - 1) * (s["sigma_min"] ** (1 / s["rho"]) - s["sigma_max"] ** (1 / s["rho"]))) ** s["rho"]
    sig = torch.cat([sig, sig.new_zeros(1)]).tolist()
    B = memory.shape[0]
    randn = lambda shp: torch.randn(shp, device=dev, generator=gen)
    x = tuple(sig[0] * randn((B, 1, *shp)) for shp in shapes)
    clamp = cfg["perception"]["periphery"]["tanh_clamp"]
    for k in range(n):
        gamma = min(s["s_churn"] / n, math.sqrt(2) - 1) if s["s_tmin"] <= sig[k] <= s["s_tmax"] else 0.0
        hat = sig[k] * (1 + gamma)
        if gamma > 0:
            x = tuple(t + math.sqrt(hat ** 2 - sig[k] ** 2) * s["s_noise"] * randn(t.shape) for t in x)
        D, _ = denoised(den, x, torch.full((B, 1), hat, device=dev), prev_action, memory, fix, cache, clamp, e["sigma_data"])
        x = tuple(t + (sig[k + 1] - hat) * (t - d) / hat for t, d in zip(x, D))
    return x


def context_pass(den: Denoiser, x, prev_action, memory, fix, cache, cfg, gen=None):
    """Enter a finished frame into the context once: one fresh draw noises it to the context level, one pass in its
    context role writes its keys / values (STEP_SPEC 7). Returns the new cache."""
    e = cfg["edm"]
    sc = e["inference_context_sigma"]
    xs = tuple(t + sc * torch.randn(t.shape, device=t.device, generator=gen) for t in x)
    c_skip, c_out, c_in, c_noise = precondition(torch.full((x[0].shape[0], 1), sc, device=x[0].device), e["sigma_data"])
    _, new = den.step(tuple(_bcast(c_in, t) * t for t in xs), c_noise, prev_action, memory, fix, cache)
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
