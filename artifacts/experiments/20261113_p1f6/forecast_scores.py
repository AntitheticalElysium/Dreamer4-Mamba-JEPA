"""Held-out forecast scores for F6: the token forecast (own head, change form) against copying the previous frame's
tokens, the reactive TSP on matched uniform query windows, and the outcome heads in both modes. F5's
forecast_scores.py protocol (same episodes, histories and query windows)."""
import argparse
import importlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from saccade import data as D, model as M, train as TR
from saccade.eval import s1 as S

IF = importlib.import_module("artifacts.experiments.20261030_actor_interface.interface")


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--episodes", type=int, default=120)
    args = p.parse_args()
    torch.set_num_threads(2)
    model, cfg = IF.load(args.checkpoint.resolve())
    dev = IF.DEV
    L = dict(np.load(S.LABELS / "s0_dev_labels.npz"))
    chosen = np.sort(np.random.default_rng(55112).choice(len(L["ep_T"]), args.episodes, replace=False))
    episode_list = S.episodes(L)
    rows = []
    for i in chosen:
        ep = episode_list[i]
        history = D.fixation_prior(np.random.default_rng([0, ep.shard, ep.slot]), (ep.T + 1,))
        query = D.fixation_prior(np.random.default_rng([615001, ep.shard, ep.slot]), (ep.T,))
        chunk = TR.tensors(D.gather([ep], [[(0, 1, t, *f) for t, f in enumerate(history)]]), dev)
        query = torch.as_tensor(query, device=dev)[None]
        cells = M.window(query)
        target = M.cell_pixels(M.crop(chunk["frames"][:, 1:], query)).float() / 127.5 - 1
        with TR.autocast(dev, cfg):
            parts = [model.perception(chunk["frames"][:, a:a + 512], chunk["fixation"][:, a:a + 512])
                     for a in range(0, ep.T + 1, 512)]
            percept = M.Percept(*(torch.cat(z, 1) for z in zip(*(p.parts() for p in parts))), chunk["fixation"])
            embed = model.belief.percept_embed(percept)
            y, _ = model.belief(model.belief.tokens(embed, chunk["reward"], chunk["terminated"], chunk["action"]),
                                model.belief.initial(1, dev))
            bp = y[:, 1::2][:, :-1]
            reward = chunk["reward"][:, 1:].float()
            alive = (~chunk["terminated"][:, 1:]).float()
            row = {"episode": int(i), "n": ep.T}
            for name in ("forecast", "reactive"):
                h = model.outcome.features(bp, embed[:, 1:] if name == "reactive" else None)
                row[name] = {"reward_nll": float(-model.outcome.reward_logp(h, reward).mean()),
                             "continuation_nll_given_reward": float(F.binary_cross_entropy_with_logits(
                                 model.outcome.cont_logit(h, reward).float(), alive))}
            guess = model.pixels.mean(model.tsp.predict(bp, percept.p[:, 1:], percept.v[:, 1:], cells))
            row["reactive"]["tsp_mse"] = float((guess.float() - target).square().mean())
        prev, nxt = percept.p[:, :-1].float(), percept.p[:, 1:].float()
        row["forecast"]["token_mse"] = float((prev + model.token_forecast(bp.float()) - nxt).square().mean())
        row["forecast"]["token_mse_copy_previous"] = float((prev - nxt).square().mean())
        rows.append(row)
    n = sum(r["n"] for r in rows)
    means = {k: {m: sum(r["n"] * r[k][m] for r in rows) / n for m in rows[0][k]} for k in ("forecast", "reactive")}
    args.out.write_text(json.dumps({"checkpoint_sha256": TR.file_sha256(args.checkpoint), "frames": n,
                                    "means": means, "episodes": rows}, indent=2) + "\n")
    print(json.dumps(means), flush=True)


if __name__ == "__main__":
    main()
