"""Exploratory frozen-H2 readout challenge, fixed before this run.

Question: does the *existing* one-step generated feature contain a death signal that
the trained continuation head fails to read? No world or encoder weights change.

Use canonical Raw H2 checkpoint/cache, 800 TRAIN and 400 DEV terminal episodes
(len >= 16), sampled independently with seed 20261004. For each episode generate
the death, pre-death, and alive-10 targets at the same L5/depth-1 position (four observed frames, then one generated). Fit a
canonical-width SwiGLU continuation probe separately on (a) generated features,
(b) real successor features, and (c) root features plus the recorded action.
Fit on the first 80% of TRAIN episodes, choose among steps 100..800 on the rest
of TRAIN, and evaluate DEV once. Three probe seeds. Compare with checkpoint head.

Primary: paired within-episode death-vs-alive10 and death-vs-predeath AUC. These
are factual discrimination diagnostics, *not* all-action safety-choice evidence.
"""

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from d4mj.backbone import SwiGLU
from d4mj.cache import load_latent_cache
from d4mj.data import _sha256
from d4mj.experiments import _load_bridge_parent
from d4mj.train import autocast_context, bridge_rollout

HERE = Path(__file__).parent
CHECKPOINT = ROOT / "artifacts/lewm_m4_canonical/raw/bridge/step-002000.pt"
CACHE = ROOT / "artifacts/lewm_m4_canonical/raw/cache"
BACK = (0, 1, 10)
PER_SPLIT = {"train": 800, "dev": 400}
SEED = 20261004
STEPS = 800
PROBE_SEEDS = (7, 11, 19)


def log(**kwargs):
    print(json.dumps(kwargs), flush=True)


@torch.no_grad()
def extract(bundle, heads, episodes, device):
    z_windows, a_windows = [], []
    for episode in episodes:
        for back in BACK:
            end = len(episode) - back
            start = end - 4
            z_windows.append(episode.latents[start:end + 1])
            a_windows.append(torch.as_tensor(episode.actions_taken[start:end]).long())
    generated, observed, root, action, old_generated, old_observed = [], [], [], [], [], []
    for start in range(0, len(z_windows), 64):
        z = torch.stack(z_windows[start:start + 64]).to(device)
        a = torch.stack(a_windows[start:start + 64]).to(device)
        with autocast_context(bundle.config):
            teacher, _, g_features, anchor = bridge_rollout(bundle, z, a, 1, None)
            merged = torch.cat((teacher.features[:, :anchor + 1], g_features), 1)
            pg = 1 - torch.sigmoid(heads(merged)["continuation"][:, -1, 0].float())
            po = 1 - torch.sigmoid(heads(teacher.features)["continuation"][:, -1, 0].float())
        generated.append(g_features[:, -1].mean(1).float().cpu())
        observed.append(teacher.features[:, -1].mean(1).float().cpu())
        root.append(teacher.features[:, anchor].mean(1).float().cpu())
        action.append(a[:, -1].cpu())
        old_generated.append(pg.cpu())
        old_observed.append(po.cpu())
    pack = lambda parts: torch.cat(parts, 0)
    r = pack(root)
    a = F.one_hot(pack(action), bundle.config.n_actions).float()
    return {"generated": pack(generated), "observed": pack(observed),
            "root_action": torch.cat((r, a), 1),
            "old_generated": pack(old_generated), "old_observed": pack(old_observed)}


def paired(score):
    s = score.reshape(-1, 3)
    death = s[:, 0]
    out = {}
    for name, neg in (("predeath", s[:, 1]), ("alive10", s[:, 2])):
        wins = ((death > neg).float() + 0.5 * (death == neg).float()).numpy()
        rng = np.random.default_rng(20261004)
        samples = rng.integers(0, len(wins), (1000, len(wins)))
        means = wins[samples].mean(1)
        out[name] = {"auc": float(wins.mean()), "ci95": np.quantile(means, [.025, .975]).tolist()}
    return out


class Probe(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.body = SwiGLU(width, 2.0)
        self.out = nn.Linear(width, 1)

    def forward(self, x):
        return self.out(self.body(x)).squeeze(-1)


def fit(source, train, dev, device, seed):
    torch.manual_seed(seed)
    x = train[source].float()
    mean = x[:640 * 3].mean(0)
    std = x[:640 * 3].std(0).clamp(min=1e-4)
    x = ((x - mean) / std).to(device)
    xd = ((dev[source].float() - mean) / std).to(device)
    y = torch.tensor([1., 0., 0.] * 800, device=device)
    model = Probe(x.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    gen = torch.Generator().manual_seed(seed + 100)
    best_loss, best_state, best_step = float("inf"), None, None
    train_rows, select_rows = 640 * 3, torch.arange(640 * 3, 800 * 3, device=device)
    for step in range(1, STEPS + 1):
        ids = torch.randint(train_rows, (256,), generator=gen).to(device)
        logits = model(x[ids])
        weights = torch.where(y[ids] > .5, 2., 1.)
        loss = (F.binary_cross_entropy_with_logits(logits, y[ids], reduction="none") * weights).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 100 == 0:
            model.eval()
            with torch.no_grad():
                logits = model(x[select_rows])
                weights = torch.where(y[select_rows] > .5, 2., 1.)
                v = float((F.binary_cross_entropy_with_logits(logits, y[select_rows], reduction="none") * weights).mean())
            if v < best_loss:
                best_loss, best_step = v, step
                best_state = {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}
            model.train()
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        score = torch.sigmoid(model(xd)).cpu()
    if source == "generated":
        torch.save({"state": best_state, "mean": mean, "std": std,
                    "checkpoint_sha256": _sha256(CHECKPOINT), "seed": seed},
                   HERE / f"head_generated_{seed}.pt")
    return {"seed": seed, "step": best_step, "train_select_bce": best_loss,
            "dev_paired": paired(score),
            "dev_mean_pdeath": score.reshape(-1, 3).mean(0).tolist(),
            "dev_scores": score.tolist()}


def main():
    started = time.time()
    assert _sha256(CHECKPOINT) == "ffb852c1f650a106dd94943c0458615cb8bc42be0181c2c465663db82a9eb20e"
    assert _sha256(CACHE / "manifest.json") == "80d3d60521424ddb61318e6ebda9020e304cbcd3a96a7f2d07b1055bb1c30618"
    bundle, heads, _ = _load_bridge_parent(CHECKPOINT)
    bundle.encoder.freeze()
    bundle.world.eval()
    heads.eval()
    for module in bundle.world.modules():
        if isinstance(module, torch.nn.BatchNorm1d):
            module.eval()
    device = torch.device(bundle.config.runtime.device)
    corpus = load_latent_cache(CACHE, bundle.encoder, bundle.config)
    rng = torch.Generator().manual_seed(SEED)
    populations = {}
    for split, n in PER_SPLIT.items():
        pool = [e for e in corpus if e.split == split and bool(e.terminated[-1]) and len(e) >= 16]
        assert len(pool) >= n, (split, len(pool))
        chosen = torch.randperm(len(pool), generator=rng)[:n].tolist()
        populations[split] = [pool[i] for i in chosen]
        log(stage="sample", split=split, available=len(pool), sampled=n)
    features = {}
    for split, eps in populations.items():
        features[split] = extract(bundle, heads, eps, device)
        log(stage="extract", split=split, seconds=round(time.time() - started, 1))
    result = {"status": "exploratory_frozen_h2_refit", "checkpoint_sha256": _sha256(CHECKPOINT),
              "cache_manifest_sha256": _sha256(CACHE / "manifest.json"),
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "episode_counts": PER_SPLIT, "old_head": {}, "refit": {}}
    for path in ("generated", "observed"):
        score = features["dev"]["old_" + path]
        result["old_head"][path] = {"dev_paired": paired(score),
                                     "dev_mean_pdeath": score.reshape(-1, 3).mean(0).tolist(),
                                     "dev_scores": score.tolist()}
    for source in ("generated", "observed", "root_action"):
        result["refit"][source] = []
        for seed in PROBE_SEEDS:
            row = fit(source, features["train"], features["dev"], device, seed)
            result["refit"][source].append(row)
            log(stage="fit", source=source, seed=seed, step=row["step"],
                paired=row["dev_paired"], seconds=round(time.time() - started, 1))
    result["seconds"] = round(time.time() - started, 1)
    (HERE / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    log(stage="done", seconds=result["seconds"], result=str(HERE / "result.json"))


if __name__ == "__main__":
    main()
