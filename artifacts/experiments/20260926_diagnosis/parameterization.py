"""D16. Why the spatial transformer worlds (T, sZ) never beat copying the input, even on their training pool.

Measured (D15 follow-up): teacher-forced L1 on their own training pool, T 0.1475 vs copy 0.1281, sZ 0.1204
vs copy 0.0780. spatial.World outputs layer_norm(proj(history)): no path from a frame's state to its own
prediction, so even the identity must be learned through 6 layers. Test: the SAME architecture, pool, batches
(seed 11), optimizer (AdamW lr 1e-4, wd 0.01, 1,000 warmup, clip 1.0, bf16) and dynamics loss (spatial.losses'
L1: teacher-forced + depth-2 generated suffix), heads removed in both arms so only the output differs:
  direct     layer_norm(proj(history))                         (spatial.World as trained)
  residual   layer_norm(s_t + proj(history)), proj zero-initialised (starts as a copy)
4,000 updates each, for T and sZ. Read: teacher-forced L1 on 2,048 held-out main windows vs copying, split by
whether the action is a move (1-4) or not.
"""
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).parent
ROOT = Path("/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "artifacts/experiments/20260921_readout_ladder"))
import spatial as S  # noqa: E402

UPDATES, BATCH = 4_000, 40


class Residual(S.World):
    def __init__(self, n):
        super().__init__(n, False)
        torch.nn.init.zeros_(self.proj.weight); torch.nn.init.zeros_(self.proj.bias)

    def forward(self, s, a):
        b, t = s.shape[:2]
        x = torch.cat([self.action(a)[:, :, None], self.embed(s)], 2) + self.space + self.time[:t, None]
        k = t * (self.n + 1)
        x = self.blocks(x.flatten(1, 2), mask=self.blocked[:k, :k]).view(b, t, self.n + 1, S.D)[:, :, 1:]
        history = self.norm(x)
        return F.layer_norm((s.float() + self.proj(history).float()), (S.WIDTH,)), history


def dynamics_loss(world, s, a):
    predicted, generated, _, _ = S.rollout(world, s, a)
    return (predicted[:, :S.W - 1] - s[:, 1:]).abs().mean() + (generated - s[:, S.ANCHOR + 1:]).abs().mean()


@torch.no_grad()
def evaluate(world, pool, rows, state, config, device):
    from d4mj.train import autocast_context
    err, copy, move = [], [], []
    for i in range(0, len(rows), 64):
        b = S.batch_of(pool, rows[i:i + 64], state, device)
        with autocast_context(config):
            pred, _ = world(b["s"], F.pad(b["actions"], (0, 1)))
        err.append((pred[:, :5].float() - b["s"][:, 1:]).abs().mean((-1, -2)).cpu())
        copy.append((b["s"][:, :-1] - b["s"][:, 1:]).abs().mean((-1, -2)).cpu())
        move.append(((b["actions"] >= 1) & (b["actions"] <= 4)).cpu())
    err, copy, move = torch.cat(err), torch.cat(copy), torch.cat(move)
    return {"l1": float(err.mean()), "copy": float(copy.mean()),
            "l1_move": float(err[move].mean()), "copy_move": float(copy[move].mean()),
            "l1_other": float(err[~move].mean()), "copy_other": float(copy[~move].mean())}


def main():
    from d4mj.config import config_from_dict
    from d4mj.train import _phase_lr, autocast_context, optimizer_step, phase_optimizer
    device = torch.device("cuda")
    config = config_from_dict(torch.load(S.CHECKPOINT, map_location="cpu", weights_only=False)["config"])
    pool = torch.load(S.POOL / "pool.pt", weights_only=False, mmap=True)
    main_rows = torch.where(~pool["terminal"])[0]
    held = main_rows[torch.randperm(len(main_rows), generator=torch.Generator().manual_seed(1))[:2048]]
    train_rows = torch.cat([main_rows[~torch.isin(main_rows, held)], torch.where(pool["terminal"])[0]])
    result = {}
    for state, n in (("z", 1), ("tokens", 81)):
        for arm in ("direct", "residual"):
            with torch.random.fork_rng(devices=[0]):
                torch.manual_seed(7); torch.cuda.manual_seed_all(7)
                world = (S.World(n, False) if arm == "direct" else Residual(n)).to(device)
            opt = phase_optimizer([world], config)
            params = [p for g in opt.param_groups for p in g["params"]]
            order = torch.Generator().manual_seed(11)
            for update in range(UPDATES):
                b = S.batch_of(pool, train_rows[torch.randint(len(train_rows), (BATCH,), generator=order)], state, device)
                with autocast_context(config):
                    loss = dynamics_loss(world, b["s"], b["actions"])
                optimizer_step(opt, loss, params, learning_rate=_phase_lr(config, update),
                               grad_clip=config.agent.grad_clip, strict=True, zero_grad=True)
            world.eval()
            row = evaluate(world, pool, held, state, config, device)
            result[f"{'T' if n == 81 else 'sZ'}_{arm}"] = row
            print(f"{'T' if n == 81 else 'sZ'} {arm}", json.dumps({k: round(v, 4) for k, v in row.items()}), flush=True)
    (HERE / "parameterization.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
