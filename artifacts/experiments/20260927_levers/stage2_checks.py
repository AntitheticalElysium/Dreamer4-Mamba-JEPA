"""Stage 2 correctness checks for tworld's factorized backbones (run before any arm is trained; stage2_checks.json).

1. masked_scan with every step kept == FunctionalMamba2.scan (Triton); with holds == FunctionalMamba2's own
   reference recurrence given delta = 0 at the held steps (fp32).
2. No scroll: fcanvas == fmamba under the same weights (the canvas only re-indexes positions).
3. Canvas alignment: views cut from a random 30 x 30 world along a random walk (plus random HUD tokens); the scroll is
   estimated from the tokens; every canvas cell must receive the same world token from every frame that writes it.
4. Causality: perturbing frames > t leaves every backbone's outputs at frames <= t unchanged.
5. Parameter counts.
"""
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import tworld as TW  # noqa: E402
from scroll import SHIFTS, estimate  # noqa: E402


def views(b, t, g):
    world = F.layer_norm(torch.randn(b, 30, 30, 192, generator=g), (192,))
    pos = torch.full((b, 2), 11)
    frames, moves = [], []
    for k in range(t):
        if k:
            m = torch.randint(0, 5, (b,), generator=g)
            pos = pos + torch.tensor(SHIFTS)[m]                        # new view (r,c) = old view (r+dr, c+dc)
            moves.append(m)
        v = torch.stack([world[i, pos[i, 0]:pos[i, 0] + 7, pos[i, 1]:pos[i, 1] + 9] for i in range(b)]).flatten(1, 2)
        frames.append(torch.cat([v, F.layer_norm(torch.randn(b, 18, 192, generator=g), (192,))], 1))
    return torch.stack(frames, 1), torch.stack(moves, 1)


@torch.no_grad()
def main():
    device = torch.device("cuda")
    g = torch.Generator().manual_seed(0)
    out = {}
    # 1
    from d4mj.config import config_from_dict
    import spatial as S
    settings = config_from_dict(torch.load(S.CHECKPOINT, map_location="cpu", weights_only=False)["config"]).dynamics
    from d4mj.mamba_recurrence import FunctionalMamba2
    torch.manual_seed(0)
    mix = FunctionalMamba2(settings).to(device)
    x = torch.randn(64, 6, 256, device=device)
    keep = torch.ones(64, 6, dtype=torch.bool, device=device)
    out["masked_scan_all_kept_vs_scan_maxabs"] = float((TW.masked_scan(mix, x, keep) - mix.scan(x)[0]).abs().max())
    keep = torch.rand(64, 6, device=device, generator=torch.Generator(device).manual_seed(1)) > 0.4
    xm = torch.where(keep[..., None], x, 0)
    c = mix.core
    z, xbc, dt = torch.split(c.in_proj(xm), [c.d_ssm, c.d_ssm + 2 * c.d_state, c.nheads], dim=-1)
    filtered = F.silu(F.conv1d(F.pad(xbc.transpose(1, 2), (c.d_conv - 1, 0)), c.conv1d.weight, c.conv1d.bias,
                               groups=c.conv1d.groups)).transpose(1, 2)
    xx, bb, cc = torch.split(filtered, [c.d_ssm, c.d_state, c.d_state], dim=-1)
    y, _ = mix._reference_ssm(xx.reshape(*xx.shape[:2], c.nheads, c.headdim), bb, cc,
                              torch.where(keep[..., None], dt, dt.new_tensor(-1e4)), mix.initial(64, device=device).ssm)
    y = y.flatten(2)
    gated = y * F.silu(z)
    ref = c.out_proj(gated * torch.rsqrt(gated.square().mean(-1, keepdim=True) + c.norm.eps) * c.norm.weight)
    out["masked_scan_holds_vs_reference_maxabs"] = float((TW.masked_scan(mix, xm, keep) - ref).abs().max())
    out["reference_scale"] = float(ref.abs().max())
    # 2
    torch.manual_seed(3)
    wm = TW.TWorld("corrt", None, "fmamba").to(device).eval()
    wc = TW.TWorld("corrt", None, "fcanvas").to(device).eval()
    wc.load_state_dict(wm.state_dict())
    s = F.layer_norm(torch.randn(8, 1, 81, 192, generator=g), (192,)).expand(-1, 6, -1, -1).contiguous().to(device)
    a = torch.randint(0, 17, (8, 6), generator=g).to(device)
    out["static_fcanvas_vs_fmamba_maxabs"] = float((wc(s, a)[0] - wm(s, a)[0]).abs().max())
    # 3
    frames, moves = views(16, 6, g)
    est = estimate(frames[:, :-1], frames[:, 1:])
    out["synthetic_scroll_estimate_accuracy"] = float((est == moves).float().mean())
    t = 6
    pad = t - 1
    cols = 9 + 2 * pad
    offset = torch.tensor(SHIFTS)[F.pad(est, (1, 0))].cumsum(1)
    r = torch.arange(7)[:, None] + offset[..., 0, None, None] + pad
    cc_ = torch.arange(9)[None] + offset[..., 1, None, None] + pad
    cell = (r * cols + cc_).flatten(2)                                                   # [B,T,63]
    worst = 0.0
    for i in range(16):
        written = {}
        for k in range(t):
            for j in range(63):
                key = int(cell[i, k, j])
                if key in written:
                    worst = max(worst, float((written[key] - frames[i, k, j]).abs().max()))
                written[key] = frames[i, k, j]
    out["canvas_same_world_token_maxabs"] = worst
    # 4 + 5
    s = F.layer_norm(torch.randn(4, 6, 81, 192, generator=g), (192,)).to(device)
    s[:, :, :63] = frames[:4, :, :63].to(device)
    a = torch.randint(0, 17, (4, 6), generator=g).to(device)
    s2 = s.clone()
    s2[:, 4:] = F.layer_norm(torch.randn(4, 2, 81, 192, generator=g), (192,)).to(device)
    a2 = a.clone()
    a2[:, 4:] = (a[:, 4:] + 1) % 17
    for kind in ("full", "fattn", "fmamba", "fcanvas", "fscan"):
        torch.manual_seed(5)
        w = TW.TWorld("corrt", None, kind).to(device).eval()
        o1, o2 = w(s, a)[0], w(s2, a2)[0]
        out[f"causal_{kind}_maxabs_frames_0_3"] = float((o1[:, :4] - o2[:, :4]).abs().max())
        out[f"causal_{kind}_changes_frames_4_5"] = float((o1[:, 4:] - o2[:, 4:]).abs().max())
        out[f"parameters_{kind}"] = sum(p.numel() for p in w.parameters())
    (HERE / "stage2_checks.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=1), flush=True)


if __name__ == "__main__":
    main()
