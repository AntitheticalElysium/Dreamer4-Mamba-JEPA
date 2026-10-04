"""D13. Is W's better reading of asleep deaths a HEAD-conditioning effect? The clean test headwhite.py was not.

headwhite.py trained the U recipe with a whitened readout input, but the world kept training during the head
phase, and W itself also changes the world's loss weighting. Here the world is FROZEN (the seed-1 U world) and
only a fresh continuation head is trained, twice, identically but for its input:
  raw     cat(u_t, h_{t-1})            h = the frozen U world's Mamba output (teacher path)
  white   cat(u_t / std, h_{t-1})      std = W's per-component TRAIN std (whiten.py)
Head: H2's agent readout shape (Linear -> LayerNorm -> GELU) + Linear -> 1 logit. Data: the interface pool
(24,576 main + 8,071 terminal 6-frame windows), every position labelled alive / dead; batches of 32 main + 8
terminal windows (interface phase 2's ratio), AdamW lr 1e-4 wd 0.01 (H2's phase optimizer settings), 1,000
warmup, 9,333 updates, BCE. Head seeds 1, 2, 3 per arm, the same batch order in both arms.
Read on the zombie roots of 55k-58k, TRUE successors through the same frozen world (`observe_latent`):
fatal-vs-survived AUC and mean P(dead) for SLEEP (rendered asleep) and NOOP (awake).
Reading: white - raw on SLEEP AUC > 0 for all 3 seeds -> head_conditioning_isolated; otherwise not.
"""
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

HERE = Path(__file__).parent
ROOT = Path("/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA")
LADDER = ROOT / "artifacts/experiments/20260921_readout_ladder"
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(LADDER)); sys.path.insert(0, str(HERE))
import interface as I  # noqa: E402
from compound import auc  # noqa: E402
from frozen_ladder import strata  # noqa: E402

BLOCKS = ("observe_fresh_v6", "observe_fresh_v7", "observe_fresh_v8", "observe_fresh_v9")
UPDATES, WARMUP, MAIN, TERMINAL = 9_333, 1_000, 32, 8


@torch.no_grad()
def pool_features(world, pool, device, batch=512):
    """cat(u_t, h_{t-1}) for every pool window position, and alive labels."""
    xs = []
    for i in range(0, len(pool["u"]), batch):
        u = pool["u"][i:i + batch].float().to(device)
        a = pool["actions"][i:i + batch].to(device)
        _, history, _ = world.scan_pairs(u[:, :-1, None], a)
        aligned = torch.cat([torch.zeros_like(history[:, :1]), history], 1)
        xs.append(torch.cat([u, aligned.float()], -1).half().cpu())
    return torch.cat(xs), pool["alive"].float()


@torch.no_grad()
def judge_features(bundle, pca, encoder, device):
    """Zombie roots' TRUE successors of SLEEP and NOOP through the frozen world: cat(u', h(a)) and death labels."""
    from d4mj.train import autocast_context
    out = {6: [], 0: []}
    labels = {6: [], 0: []}
    for store in BLOCKS:
        rows = [r for f in sorted((ROOT / "artifacts/eda" / store).glob("seed-*.pt")) for r in torch.load(f, weights_only=False)]
        rows = [r for r in rows if r["p_death1"].max() > r["p_death1"].min()]
        zom = strata(torch.stack([r["visible"].float() for r in rows]))["zombie_adjacent"]
        rows = [r for r, k in zip(rows, zom) if k]
        for i in range(0, len(rows), 32):
            rb = rows[i:i + 32]
            z, g = I.encode(encoder, torch.stack([r["frames"][-4:] for r in rb]), device)
            s = I.state_of("U", pca, z, g).to(device)
            past = torch.stack([r["led_to_action"][-3:] for r in rb]).to(device)
            with autocast_context(bundle.config):
                st = bundle.world.teacher(s[:, :, None], past).state
                for a in (6, 0):
                    adv, _ = bundle.advance(st, torch.full((len(rb), 1), a, device=device))
                    rz, rg = I.encode(encoder, torch.stack([r["successors"][a] for r in rb])[:, None], device)
                    real = I.state_of("U", pca, rz, rg)[:, 0].to(device)
                    out[a].append(torch.cat([real.float(), adv.history[:, 0].float()], -1).cpu())
                    labels[a].append(torch.tensor([bool(r["terminated"][a]) for r in rb]))
    return {a: torch.cat(v) for a, v in out.items()}, {a: torch.cat(v) for a, v in labels.items()}


def train_head(x, y, terminal, scale, seed, device):
    torch.manual_seed(seed)
    head = nn.Sequential(nn.Linear(x.shape[-1], 256), nn.LayerNorm(256), nn.GELU(), nn.Linear(256, 1)).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-4, weight_decay=0.01)
    order = torch.Generator().manual_seed(1000 + seed)
    main_rows, term_rows = torch.where(~terminal)[0], torch.where(terminal)[0]
    for step in range(UPDATES):
        rows = torch.cat([main_rows[torch.randint(len(main_rows), (MAIN,), generator=order)],
                          term_rows[torch.randint(len(term_rows), (TERMINAL,), generator=order)]])
        xb = x[rows].float().to(device) * scale
        yb = y[rows].to(device)
        for group in opt.param_groups:
            group["lr"] = 1e-4 * min(1.0, (step + 1) / WARMUP)
        loss = F.binary_cross_entropy_with_logits(head(xb).squeeze(-1), yb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        opt.step()
    return head.eval()


def main():
    device = torch.device("cuda")
    from whiten import whitened_pool
    pool, std = whitened_pool()
    encoder, config = I.load_bridge()
    bundle = I.world_bundle(config, encoder, device)
    bundle.world.load_state_dict(torch.load(ROOT / "artifacts/eda/interface_worlds_v1/U.pt", map_location="cpu",
                                            weights_only=False)["world"])
    bundle.world.eval()
    x, alive = pool_features(bundle.world, pool, device)
    jx, jy = judge_features(bundle, pool["pca"], encoder, device)
    scales = {"raw": torch.ones(x.shape[-1]), "white": torch.cat([1 / std, torch.ones(x.shape[-1] - 192)])}
    result = {}
    for arm, scale in scales.items():
        result[arm] = []
        for seed in (1, 2, 3):
            head = train_head(x, alive, pool["terminal"], scale.to(device), seed, device)
            with torch.no_grad():
                row = {}
                for a, name in ((6, "SLEEP"), (0, "NOOP")):
                    p = 1 - torch.sigmoid(head(jx[a].to(device) * scale.to(device)).squeeze(-1)).cpu()
                    row[name] = {"auc": auc(p, jy[a]), "mean_p_dead": float(p.mean()), "death_rate": float(jy[a].float().mean())}
            result[arm].append(row)
            print(arm, seed, json.dumps(row), flush=True)
    diff = [result["white"][i]["SLEEP"]["auc"] - result["raw"][i]["SLEEP"]["auc"] for i in range(3)]
    result["white_minus_raw_sleep_auc"] = diff
    result["reading"] = "head_conditioning_isolated" if all(d > 0 for d in diff) else "not_isolated"
    (HERE / "headonly.json").write_text(json.dumps(result, indent=2) + "\n")
    print(result["reading"], diff, flush=True)


if __name__ == "__main__":
    main()
