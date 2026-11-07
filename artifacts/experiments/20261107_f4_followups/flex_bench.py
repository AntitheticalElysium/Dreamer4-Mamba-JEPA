"""Follow-up 4 (PLAN.md): FlexAttention for P1-F4's dense layers, benchmark-only (no core change before adoption).
`flex_dense_time` replaces Block.dense_time's chunked (window) path inside this process: one flex_attention call per
dense layer on the frame-major flattened tokens; the mask is the window rule rebuilt from the existing (index, allowed)
tuple (key frame lag 0..w and allowed[b, query frame, w - lag]) and decision 1 (physical queries read physical keys
only). The streaming / glimpse-only paths keep SDPA. Stages (each MAMBA_DETERMINISTIC=1, GPU):
  correct   F4's checkpoint, one real 64-step dev batch (two rows spliced across an episode boundary, 2 bootstrap rows,
            bootstrap on): shortcut_loss per frame and parameter gradients, flex vs SDPA, bf16 and fp32
  det       two fresh processes per path (flex, SDPA), 4 updates of F4's preflight configuration each; checkpoints
            compared within each pair
  cost      sec / update and peak from train.p1 (F4's checks.py protocol) for F3's recipe at both splits and F4 with
            SDPA / flex at both splits; the projected training hours
Usage: MAMBA_DETERMINISTIC=1 python -m artifacts.experiments.20261107_f4_followups.flex_bench correct|det|cost"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import copy, importlib, json, subprocess, sys, time
from pathlib import Path

import numpy as np
import torch
import torch._dynamo
import torch._inductor.config as IC
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

from saccade import model as M, train as TR
from saccade.eval import s1 as S

L4 = importlib.import_module("artifacts.experiments.20261106_p1f4.train_p1f4")
F2C = importlib.import_module("artifacts.experiments.20261103_p1f2.checks")
CV = importlib.import_module("artifacts.experiments.20261106_determinant_curves.curves")
HERE = Path(__file__).resolve().parent
OUT = HERE / "readings/flex"
CKPT = "artifacts/experiments/20261106_p1f4/run/ckpt_10.pt"
IC.deterministic = True                                                 # deterministic backward (PLAN 4)
torch._dynamo.config.cache_size_limit = 64                              # one static compile per batch / length shape
FLEX = torch.compile(flex_attention, dynamic=False)
SDPA_DENSE_TIME = M.Block.dense_time
_masks = {}


def block_mask(ok, n, w, glimpse):
    key = (ok.data_ptr(), ok._version, tuple(ok.shape), n, w, glimpse is None)
    if key not in _masks:
        B, T = ok.shape[:2]
        g = torch.zeros(n, dtype=torch.bool, device=ok.device) if glimpse is None else glimpse

        def mask_mod(b, h, qi, ki):
            lag = qi // n - ki // n
            return (lag >= 0) & (lag <= w) & ok[b, qi // n, (w - lag).clamp(0, w)] & (g[qi % n] | ~g[ki % n])
        _masks.clear()
        _masks[key] = create_block_mask(mask_mod, B, None, T * n, T * n, device=ok.device, _compile=True)
    return _masks[key]


def flex_dense_time(self, h, ang, win=None, cache=None, phys=None, cur=None):
    if win is None or cur is not None:
        return SDPA_DENSE_TIME(self, h, ang, win, cache, phys, cur)
    q, k, v = self.attn.project(h, ang[:, :, None].expand(-1, -1, h.shape[2], -1))   # [B, T, H, n, hd]
    B, T, H, n, hd = q.shape
    glimpse = None if phys is None else ~torch.zeros(n, dtype=torch.bool, device=h.device).index_fill(0, phys, True)
    dt = torch.get_autocast_dtype("cuda") if torch.is_autocast_enabled("cuda") else q.dtype   # SDPA's autocast cast
    flat = lambda t: t.permute(0, 2, 1, 3, 4).reshape(B, H, T * n, hd).to(dt)
    o = FLEX(flat(q), flat(k), flat(v), block_mask=block_mask(win[1], n, win[0].shape[1] - 1, glimpse))
    return self.attn.out(o.reshape(B, H, T, n, hd).permute(0, 2, 3, 1, 4).flatten(-2)), (k, v)


def use(path):
    M.Block.dense_time = flex_dense_time if path == "flex" else SDPA_DENSE_TIME


def correct():
    model, cfg = CV.IF.load(CKPT)
    den, dev, sc = model.denoiser, CV.DEV, cfg["edm"]["shortcut"]
    L = S.first_episodes(dict(np.load(CV.LABELS / "s0_dev_labels.npz")), 30)
    enc = S.encode(model, L, cfg)
    eps = [e for e in range(len(enc)) if enc[e]["p"].shape[0] >= 64][:6]
    rows = [[(eps[0], 0, 32), (eps[1], 0, 32)], [(eps[2], 0, 32), (eps[3], 0, 32)]] + [[(e, 0, 64)] for e in eps[4:6]]
    cat = lambda key, f=lambda z: z: torch.stack([torch.cat([f(enc[e][key][a:a + n].to(dev)) for e, a, n in r]) for r in rows])
    x0 = tuple(cat(k).float() for k in ("p", "g", "v"))
    with torch.no_grad():
        mem = cat("bp", lambda z: model.belief.memory(z.float())).float()
    pa, fix = cat("prev_action"), cat("fix")
    episode = torch.stack([torch.cat([torch.full((n,), e, device=dev) for e, a, n in r]) for r in rows])
    valid = torch.ones(episode.shape, dtype=torch.bool, device=dev)
    boot = torch.zeros(len(rows), dtype=torch.bool, device=dev); boot[[1, 3]] = True
    g = torch.Generator(device=dev).manual_seed(1107)
    levels = [M.shortcut_levels(episode.shape, sc["k_max"], boot, dev, g) for _ in range(2)]
    noise = tuple(torch.randn(x.shape, device=dev, generator=g) for x in x0)
    del enc; torch.cuda.empty_cache()                                   # 4 rows: the fp32 pass of 8 x 64 frames exceeds 6 GB
    den.train()
    out = {}
    for path in ("sdpa", "flex"):
        for amp in (True, False):
            use(path); den.zero_grad(set_to_none=True)
            with torch.autocast("cuda", torch.bfloat16, enabled=amp):
                loss = M.shortcut_loss(den, x0, pa, mem, fix, episode, valid, sc, boot, True, levels, noise)
            loss.mean().backward()
            out[(path, amp)] = (loss.detach().float(), torch.cat([p.grad.flatten().float() for p in den.parameters() if p.grad is not None]).cpu())
            torch.cuda.empty_cache()
    rel = lambda a, b: float((a - b).norm() / b.norm())
    res = {"rows": len(rows), "frames": int(episode.shape[1]), "episode_splices": 2, "bootstrap_rows": int(boot.sum()),
           **{f"{w}_rel": {"flex_vs_sdpa_fp32": rel(out[("flex", False)][i], out[("sdpa", False)][i]),
                           "flex_vs_sdpa_bf16": rel(out[("flex", True)][i], out[("sdpa", True)][i]),
                           "sdpa_bf16_vs_fp32 (tolerance scale)": rel(out[("sdpa", True)][i], out[("sdpa", False)][i]),
                           "flex_bf16_vs_fp32": rel(out[("flex", True)][i], out[("flex", False)][i])}
              for i, w in ((0, "loss_per_frame"), (1, "gradients"))}}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "correct.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res), flush=True)


def small_cfg():
    cfg = L4.check_recipe(L4.recipe(TR.load_cfg()))
    cfg["p1"]["frames"] = 20480; cfg["p1"]["monitor"]["windows"] = 4
    cfg["edm"]["shortcut"]["bootstrap_start"] = 2
    return cfg


def det():
    OUT.mkdir(parents=True, exist_ok=True)
    if len(sys.argv) > 2:                                               # child: one 4-update run on one path
        use(sys.argv[3])
        L4.T.p1(small_cfg(), 1, OUT / "det" / sys.argv[2], updates=4, log_every=1)
        return
    for name, path in (("flex_a", "flex"), ("flex_b", "flex"), ("sdpa_a", "sdpa"), ("sdpa_b", "sdpa")):
        subprocess.run([sys.executable, "-u", "-m", "artifacts.experiments.20261107_f4_followups.flex_bench", "det", name, path],
                       cwd=TR.ROOT, check=True)
    res = {}
    for path in ("flex", "sdpa"):
        a, b = (torch.load(max((OUT / "det" / f"{path}_{x}").glob("ckpt_*.pt")), map_location="cpu", weights_only=False) for x in "ab")
        res[path] = {"model_max_abs": max(float((a["model"][k].float() - b["model"][k].float()).abs().max()) for k in a["model"]),
                     "model_bitwise_tensors": f"{sum(torch.equal(a['model'][k], b['model'][k]) for k in a['model'])} / {len(a['model'])}",
                     "optimizer_max_abs": F2C.optimizer_error(a["opt"], b["opt"])}
        la, lb = ([json.loads(x) for x in (OUT / "det" / f"{path}_{w}" / "log.jsonl").read_text().splitlines()] for w in "ab")
        strip = lambda rows: [{k: v for k, v in r.items() if k not in ("sec_per_update", "peak_gib")} for r in rows]
        res[path]["logged_terms_equal"] = strip(la) == strip(lb)
    (OUT / "det.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res), flush=True)


def cost():
    OUT.mkdir(parents=True, exist_ok=True)
    f4 = L4.check_recipe(L4.recipe(TR.load_cfg()))
    f3 = torch.load(L4.CONTROL, map_location="cpu", weights_only=False)["cfg"]
    split1 = {"accumulate_at_64": 1, "accumulate_at_256": 4}               # F3's split
    def variant(base, split, boot0=None):
        c = copy.deepcopy(base); c["data"].update(split)
        if boot0 is not None:
            c["edm"]["shortcut"]["bootstrap_start"] = boot0
        return c
    runs = [("f3_split1", f3, split1, None, "sdpa"), ("f3_split2", f3, L4.MEMORY, None, "sdpa"),
            ("f4_sdpa_split2_boot_off", f4, L4.MEMORY, None, "sdpa"), ("f4_sdpa_split2", f4, L4.MEMORY, 0, "sdpa"),
            ("f4_flex_split2_boot_off", f4, L4.MEMORY, None, "flex"), ("f4_flex_split2", f4, L4.MEMORY, 0, "flex"),
            ("f4_flex_split1_boot_off", f4, split1, None, "flex"), ("f4_flex_split1", f4, split1, 0, "flex")]
    res = json.loads((OUT / "cost.json").read_text()) if (OUT / "cost.json").exists() else {}
    for name, base, split, boot0, path in runs:
        for steps, u, every in ((64, 60, 10), (256, 3, 1)):
            key = f"{name}_{steps}"
            if key in res or (steps == 256 and name.endswith("boot_off") and name.startswith("f4")):
                continue                                                 # 256-step phase is always past the bootstrap start
            use(path); torch.cuda.reset_peak_memory_stats()
            d = OUT / "cost" / key
            try:
                model = L4.T.p1(variant(base, split, boot0), 1, d, updates=u, steps=steps, checkpoints=False, log_every=every)
                rows = [json.loads(x) for x in (d / "log.jsonl").read_text().splitlines()]
                res[key] = {"sec_per_update_final_half": float(np.mean([r["sec_per_update"] for r in rows[len(rows) // 2:]])),
                            "peak_gib": torch.cuda.max_memory_allocated() / 2 ** 30, "updates": u}
                del model
            except torch.OutOfMemoryError as e:
                res[key] = {"oom": str(e)[:200]}
            torch.cuda.empty_cache()
            (OUT / "cost.json").write_text(json.dumps(res, indent=1))
            print(key, json.dumps(res[key]), flush=True)
    n64, n256, b0 = 12500, 3125, f4["edm"]["shortcut"]["bootstrap_start"]
    s = lambda k: res.get(k, {}).get("sec_per_update_final_half", float("nan"))
    res["projected_training_hours"] = {
        "f3_split1": (n64 * s("f3_split1_64") + n256 * s("f3_split1_256")) / 3600,
        **{f"f4_{p}": (b0 * s(f"f4_{p}_boot_off_64") + (n64 - b0) * s(f"f4_{p}_64") + n256 * s(f"f4_{p}_256")) / 3600
           for p in ("sdpa_split2", "flex_split2", "flex_split1")}}
    (OUT / "cost.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res["projected_training_hours"]), flush=True)


if __name__ == "__main__":
    assert os.environ.get("MAMBA_DETERMINISTIC") == "1"
    {"correct": correct, "det": det, "cost": cost}[sys.argv[1]]()
