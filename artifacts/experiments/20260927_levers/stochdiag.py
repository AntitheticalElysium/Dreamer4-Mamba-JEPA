"""E11 diagnostic. Where does a per-tile world's imagined error come from: irreducible randomness, or the model? By content
class and depth, in exact numbers. (Asked 2026-10-01: is a stochastic head the fix, or an assumption?)

Data: the diagnosis futures (1,002 roots, 143 seeds). Every root has FIVE sampled 16-step futures under the SAME 16 actions
(independent env keys; sample 0 is the factual one teval scores) and four sampled successors per first action. Layer-normed
raw-encoder tokens of all of them (teval's encoder), cached once.
Per world (frozen): the imagined rollout g_k of the shared actions (teval's convention: 4 context frames, then a 5-frame
window), and the 17 one-step predictions from the root.

Decomposition at depth k, per root and token (squared error summed over the 192 dims), on depths where all five samples
are alive:
  floor    s^2 = sum_s |x_s - mu|^2 / (S - 1): the unbiased variance of the sampled true tokens = the expected error of the
           BEST deterministic prediction given root and actions (their conditional mean). No deterministic world can go
           below it, however good.
  excess   |g - mu|^2 - s^2 / S: the world's own error against the conditional mean (unbiased)
  total    excess + floor = the expected error against a sampled future (teval scores one sample, sample 0)
All divided by teval's V (the total token variance of the factual futures), so values read like teval's gen_k.
Token classes (exclusive, in this priority): hud (the 18 bottom tokens), player (token 31), mob (any of the five samples has
a zombie / cow / skeleton / arrow in the cell at depth k or k-1), entering (the leading row or column of sample 0's view
scroll at step k, scroll.estimate), static (every other map cell).
Also reported:
  one-step   the same decomposition for the 17 first actions, floor from the four keys (S = 4); classes per action (mobs
             from all four keys' states or the root; entering from key 0's scroll against the root)
  blur       on mob tokens with s^2 > 0: |g - mu|^2 / (s^2 (S-1)/S), i.e. how close the prediction sits to the conditional mean
             relative to a typical true sample (a true sample scores 1 on average; a mean-predictor scores 0)
  manifold   squared distance from each token to its nearest of the 4,096 k-means codes of TRUE training tokens
             (levers_codebooks_v1), for imagined tokens and for sample 0's true tokens, per class
Readings, declared before any run:
  1. aleatoric share at depth 16 = floor / total over all tokens; >= 0.5 -> aleatoric_dominated (most of the measured
     depth-16 error cannot be removed by any deterministic world)
  2. growth of the model's own error, excess_16 - excess_1, split by class: mob share >= 0.5 -> stochastic_growth;
     hud + player + entering + static share >= 0.5 -> deterministic_growth
  3. mode averaging: on mob tokens at depth 16, blur < 0.5 AND imagined manifold distance > 1.5 x true -> mode_averaged
A stochastic (sampling) head is indicated by this diagnostic only if readings 2 and 3 both point to stochastic content.
Usage: stochdiag.py <world.pt> ... -> evals/stochdiag_<name>.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
import teval as T  # noqa: E402

CACHE = ROOT / "artifacts/eda/stochdiag_v1"
H, N, S, KEYS = 16, 17, 5, 4
CLASSES = ("hud", "player", "mob", "entering", "static")
DEPTHS = (1, 2, 4, 8, 16)


def memmap(path, shape, mode="c"):
    return np.memmap(path, dtype=np.float16, mode=mode, shape=tuple(shape))


@torch.no_grad()
def token_cache(device):
    """fut5 [R,5,16,81,192] and one4 [R,4,17,81,192] layer-normed raw tokens (fp16 memmaps), from the futures store."""
    from d4mj.checkpoint import read_lewm_bundle
    from d4mj.config import config_from_dict
    from d4mj.world_api import ModelBundle
    from rollouts import load_roots
    CACHE.mkdir(parents=True, exist_ok=True)
    done = CACHE / "DONE"
    roots = None
    if not done.exists():
        roots = load_roots()
        R = len(roots)
        payload = read_lewm_bundle(T.ENCODERS["raw"])
        bundle = ModelBundle.create(config_from_dict(payload["config"]))
        bundle.encoder.load_state_dict(payload["modules"]["encoder"])
        enc = bundle.encoder.to(device).freeze()
        tok = lambda f: torch.cat([F.layer_norm(enc._hidden(f[i:i + 128, None].to(device))[2].float(), (192,)).half().cpu()
                                   for i in range(0, len(f), 128)])
        fut = memmap(CACHE / "fut5.f16", (R, S, H, 81, 192), "w+")
        one = memmap(CACHE / "one4.f16", (R, KEYS, N, 81, 192), "w+")
        for i, r in enumerate(roots):
            fut[i] = tok(r["future_frames"].flatten(0, 1)).view(S, H, 81, 192).numpy()
            one[i] = tok(r["onestep_frames"].flatten(0, 1)).view(KEYS, N, 81, 192).numpy()
        fut.flush(); one.flush()
        done.write_text(str(R))
    R = int(done.read_text())
    return (torch.from_numpy(memmap(CACHE / "fut5.f16", (R, S, H, 81, 192))),
            torch.from_numpy(memmap(CACHE / "one4.f16", (R, KEYS, N, 81, 192))))


def classes(meta, fut0, root_tok):
    """[R,16,81] class index per token at each depth (CLASSES order)."""
    from scroll import SHIFTS, estimate
    R = len(meta["seed"])
    vis = meta["future_visible"].float()                                        # [R,5,16,1534]
    mobs = vis[..., 1071:1071 + 441].reshape(R, S, H, 7, 9, 7).sum(-1) > 0       # [R,5,16,7,9]
    root_mobs = meta["root_visible"].float()[:, 1071:1071 + 441].reshape(R, 7, 9, 7).sum(-1) > 0
    now = mobs.any(1)                                                            # [R,16,7,9]
    before = torch.cat([root_mobs[:, None], now[:, :-1]], 1)
    mob = (now | before).flatten(2)                                              # [R,16,63]
    prev = torch.cat([root_tok[:, None], fut0[:, :-1]], 1)                       # [R,16,81,192]
    shift = torch.cat([estimate(prev[i:i + 32].float(), fut0[i:i + 32].float()) for i in range(0, R, 32)])   # [R,16]
    entering = torch.zeros(R, H, 7, 9, dtype=torch.bool)
    for s, (dr, dc) in enumerate(SHIFTS):
        m = shift == s
        if dr == 1: entering[..., 6, :] |= m[..., None]
        if dr == -1: entering[..., 0, :] |= m[..., None]
        if dc == 1: entering[..., :, 8] |= m[..., None]
        if dc == -1: entering[..., :, 0] |= m[..., None]
    cls = torch.full((R, H, 81), CLASSES.index("static"))
    cls[..., :63][entering.flatten(2)] = CLASSES.index("entering")
    cls[..., :63][mob] = CLASSES.index("mob")
    cls[..., 31] = CLASSES.index("player")
    cls[..., 63:] = CLASSES.index("hud")
    return cls, shift


def onestep_classes(meta, root_tok, one0):
    """[R,17,81] class index per token of each first action's successor: mobs from all four keys (or the root), the scroll
    of key 0's successor against the root."""
    from scroll import SHIFTS, estimate
    R = len(meta["seed"])
    mobs = meta["onestep_visible"].float()[..., 1071:1071 + 441].reshape(R, KEYS, N, 7, 9, 7).sum(-1) > 0   # [R,4,17,7,9]
    root_mobs = meta["root_visible"].float()[:, 1071:1071 + 441].reshape(R, 7, 9, 7).sum(-1) > 0
    mob = (mobs.any(1) | root_mobs[:, None]).flatten(2)                                              # [R,17,63]
    shift = torch.cat([estimate(root_tok[i:i + 32, None].float(), one0[i:i + 32].float()) for i in range(0, R, 32)])
    entering = torch.zeros(R, N, 7, 9, dtype=torch.bool)
    for s, (dr, dc) in enumerate(SHIFTS):
        m = shift == s
        if dr == 1: entering[..., 6, :] |= m[..., None]
        if dr == -1: entering[..., 0, :] |= m[..., None]
        if dc == 1: entering[..., :, 8] |= m[..., None]
        if dc == -1: entering[..., :, 0] |= m[..., None]
    cls = torch.full((R, N, 81), CLASSES.index("static"))
    cls[..., :63][entering.flatten(2)] = CLASSES.index("entering")
    cls[..., :63][mob] = CLASSES.index("mob")
    cls[..., 31] = CLASSES.index("player")
    cls[..., 63:] = CLASSES.index("hud")
    return cls


def nearest(codes, x, device):
    """squared distance of every token x [..., 192] to its nearest code."""
    flat = x.reshape(-1, 192).float()
    out = []
    for i in range(0, len(flat), 8192):
        c = flat[i:i + 8192].to(device)
        out.append(torch.cdist(c, codes).min(1).values.square().cpu())
    return torch.cat(out).view(x.shape[:-1])


@torch.no_grad()
def main():
    from d4mj.config import config_from_dict
    import spatial as Sp
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = config_from_dict(torch.load(Sp.CHECKPOINT, map_location="cpu", weights_only=False)["config"])
    meta, _, _ = T.split()
    fut5, one4 = token_cache(device)
    cache = T.build_cache("raw", device)
    R = len(meta["seed"])
    ctx, ca, fa = cache["ctx"], cache["ctx_a"], cache["fut_a"]
    drift = float((fut5[:64, 0].float() - cache["fut"][:64].float()).abs().max())
    if drift > 0.01:
        raise SystemExit(f"sample-0 tokens differ from teval's cache (max {drift})")
    fut0 = fut5[:, 0]
    V = float(((cache["fut"].float() - cache["fut"].float().flatten(0, 1).mean(0)) ** 2).sum((-1, -2)).mean())
    cls, shift = classes(meta, fut0, ctx[:, -1])
    cls1 = onestep_classes(meta, ctx[:, -1], one4[:, 0])
    alive = ~meta["future_dead"].cumsum(2).bool().any(1)                         # [R,16]: all five samples alive
    codes = torch.load(ROOT / "artifacts/eda/levers_codebooks_v1/raw_K4096.pt", weights_only=False)["codes"].float().to(device)
    out_dir = HERE / "evals"
    for path in [Path(p) for p in sys.argv[1:]]:
        world, st = T.load_world(path, device)
        name = st["name"]
        batch = 16 if world.backbone_kind == "full" else 4
        acc = {k: {c: {"total0": 0.0, "floor": 0.0, "excess": 0.0, "tokens": 0, "blur_num": 0.0, "blur_den": 0.0,
                       "code_gen": 0.0, "code_true": 0.0} for c in CLASSES} for k in range(H)}
        one_acc = {c: {"floor": 0.0, "excess": 0.0, "tokens": 0} for c in CLASSES}
        n_alive = [0] * H
        for i in range(0, R, batch):
            b = min(batch, R - i)
            c4, a3, fk = ctx[i:i + b].float(), ca[i:i + b], fa[i:i + b]
            frames, hist, gen = [c4[:, j] for j in range(4)], [a3[:, j] for j in range(3)], []
            for k in range(H):
                w = 4 if k == 0 else 5
                g = T.step(world, torch.stack(frames[-w:], 1), torch.stack(hist[-(w - 1):] + [fk[:, k]], 1), device, config)
                gen.append(g); frames.append(g); hist.append(fk[:, k])
            g = torch.stack(gen, 1)                                              # [b,16,81,192]
            x = fut5[i:i + b].float()                                            # [b,5,16,81,192]
            mu = x.mean(1)
            s2 = ((x - mu[:, None]) ** 2).sum((1, -1)) / (S - 1)                 # [b,16,81]
            d_mu = ((g - mu) ** 2).sum(-1)                                       # [b,16,81]
            e0 = ((g - x[:, 0]) ** 2).sum(-1)
            excess = d_mu - s2 / S
            cg, ct = nearest(codes, g, device), nearest(codes, x[:, 0], device)
            for k in range(H):
                m_root = alive[i:i + b, k]
                n_alive[k] += int(m_root.sum())
                for ci, c in enumerate(CLASSES):
                    m = (cls[i:i + b, k] == ci) & m_root[:, None]
                    if not m.any():
                        continue
                    a = acc[k][c]
                    a["total0"] += float(e0[:, k][m].sum()); a["floor"] += float(s2[:, k][m].sum())
                    a["excess"] += float(excess[:, k][m].sum()); a["tokens"] += int(m.sum())
                    a["code_gen"] += float(cg[:, k][m].sum()); a["code_true"] += float(ct[:, k][m].sum())
                    if c == "mob":
                        sm = m & (s2[:, k] > 0)
                        a["blur_num"] += float(d_mu[:, k][sm].sum()); a["blur_den"] += float((s2[:, k][sm] * (S - 1) / S).sum())
            # one-step, all 17 actions from the root; floor over the four keys
            fan = c4.repeat_interleave(N, 0)
            acts = torch.cat([a3.repeat_interleave(N, 0), torch.arange(N).repeat(b)[:, None]], 1)
            p1 = T.step(world, fan, acts, device, config).view(b, N, 81, 192)
            y = one4[i:i + b].float()                                            # [b,4,17,81,192]
            mu1 = y.mean(1)
            s21 = ((y - mu1[:, None]) ** 2).sum((1, -1)) / (KEYS - 1)            # [b,17,81]
            ex1 = ((p1 - mu1) ** 2).sum(-1) - s21 / KEYS
            c1 = cls1[i:i + b]
            for ci, c in enumerate(CLASSES):
                m = c1 == ci
                one_acc[c]["floor"] += float(s21[m].sum()); one_acc[c]["excess"] += float(ex1[m].sum())
                one_acc[c]["tokens"] += int(m.sum())
        res = {"world": name, "V": V, "roots": R, "alive_roots_by_depth": n_alive, "depths": {}, "one_step": {}}
        for k in range(H):
            n = max(n_alive[k], 1)
            row = {c: {"total0": acc[k][c]["total0"] / n / V, "floor": acc[k][c]["floor"] / n / V,
                       "excess": acc[k][c]["excess"] / n / V, "tokens_per_root": acc[k][c]["tokens"] / n,
                       "code_gen": acc[k][c]["code_gen"] / max(acc[k][c]["tokens"], 1),
                       "code_true": acc[k][c]["code_true"] / max(acc[k][c]["tokens"], 1)} for c in CLASSES}
            row["mob"]["blur"] = acc[k]["mob"]["blur_num"] / max(acc[k]["mob"]["blur_den"], 1e-9)
            row["all"] = {q: sum(row[c][q] for c in CLASSES) for q in ("total0", "floor", "excess")}
            res["depths"][k + 1] = row
        n1 = R * N
        res["one_step"] = {c: {"floor": one_acc[c]["floor"] / n1 / V, "excess": one_acc[c]["excess"] / n1 / V,
                               "tokens_per_branch": one_acc[c]["tokens"] / n1} for c in CLASSES}
        d1, d16 = res["depths"][1], res["depths"][16]
        growth = {c: d16[c]["excess"] - d1[c]["excess"] for c in CLASSES}
        total_growth = sum(growth.values())
        share = {c: growth[c] / total_growth for c in CLASSES} if total_growth > 0 else {}
        aleatoric = d16["all"]["floor"] / (d16["all"]["floor"] + d16["all"]["excess"])
        mob16 = d16["mob"]
        res["readings"] = {
            "aleatoric_share_16": aleatoric, "aleatoric_dominated": aleatoric >= 0.5,
            "excess_growth_1_to_16": growth, "excess_growth_share": share,
            "growth": ("stochastic_growth" if share.get("mob", 0) >= 0.5 else
                       "deterministic_growth" if share and sum(share[c] for c in CLASSES if c != "mob") >= 0.5 else "mixed"),
            "mob_blur_16": mob16["blur"], "mob_code_ratio_16": mob16["code_gen"] / max(mob16["code_true"], 1e-9),
            "mode_averaged": mob16["blur"] < 0.5 and mob16["code_gen"] > 1.5 * mob16["code_true"]}
        (out_dir / f"stochdiag_{name}.json").write_text(json.dumps(res, indent=2) + "\n")
        print(json.dumps({"world": name, **{k: v for k, v in res["readings"].items() if not isinstance(v, dict)},
                          "share": {c: round(v, 3) for c, v in share.items()}}), flush=True)
        del world
        torch.cuda.empty_cache()


if __name__ == "__main__":
    sys.path.insert(0, str(HERE.parent / "20260926_diagnosis"))
    main()
