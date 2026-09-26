"""D8. The follow-up measurements behind DIAGNOSIS.md, in one reproducible pass (writes followups.json).

  counterfactual   each world's zombie-root score if its stay-type choices were replaced by a random move
  asleep_heads     each head's fatal-vs-survived AUC and mean P(dead) on TRUE and GENERATED successors of
                   SLEEP (rendered asleep) and NOOP (awake), zombie roots
  death_direction  where the fatal-vs-survived information lives in u's spectrum (d^2/var by PCA rank), and
                   held-out AUC of the raw mean difference in u vs the same direction in w (= d/var in u)
  zombie_position  z's response to a zombie adjacent vs 3 tiles away (twin states), init and 10k
  count_vs_adjacency  how well the COUNT of lava / zombies in view predicts adjacency (real roots)
  noop_death       within NOOP on zombie roots: death from visible health vs hidden cooldown
  asleep_deaths    corpus death frames rendered asleep
"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np
import torch

HERE = Path(__file__).parent
ROOT = Path("/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA")
LADDER = ROOT / "artifacts/experiments/20260921_readout_ladder"
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(LADDER))
import twins as T  # noqa: E402  (pins JAX to CPU before anything imports it)
from choices import move_table  # noqa: E402
from frozen_ladder import strata  # noqa: E402
from sleep import asleep  # noqa: E402

DUMP = ROOT / "artifacts/eda/diagnosis_dump_v1"
BLOCKS = {"55k": "observe_fresh_v6", "56k": "observe_fresh_v7", "57k": "observe_fresh_v8", "58k": "observe_fresh_v9"}


def auc(score, label):
    score, label = score.double(), label.bool()
    pos, neg = score[label], score[~label]
    ranks = torch.cat([pos, neg]).argsort().argsort().double() + 1
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def counterfactual_and_heads():
    cf, heads = {}, {}
    acc = {}
    for b in BLOCKS:
        m = torch.load(DUMP / f"{b}_meta.pt")
        zom = strata(m["visible"])["zombie_adjacent"]
        cat, _ = move_table(m["visible"])
        safe = 1 - m["p_death1"]
        rand_move = safe[:, 1:5].mean(1)
        dead = m["terminated"][zom].bool()
        cf[b] = {}
        for w in ("H2", "Z", "ZW", "U", "W", "W_s2"):
            d = torch.load(DUMP / f"{b}_{w}.pt")
            ch = d["p_dead"].argmin(1)
            act = safe.gather(1, ch[:, None]).squeeze(1)
            move = (ch >= 1) & (ch <= 4)
            cc = cat.gather(1, ch[:, None]).squeeze(1)
            cf[b][w] = {"actual": float(act[zom].mean()),
                        "stay_replaced_by_random_move": float(torch.where(move, act, rand_move)[zom].mean()),
                        "move_choices_ok_share": float((cc[zom & move] == 0).float().mean()),
                        "random_move_ok_share": float((cat[zom][:, 1:5] == 0).float().mean())}
            for key in ("p_dead_real", "p_dead"):
                for a, name in ((6, "SLEEP"), (0, "NOOP")):
                    acc.setdefault((w, key, name), ([], []))
                    acc[(w, key, name)][0].append(d[key][zom][:, a]); acc[(w, key, name)][1].append(dead[:, a])
    for (w, key, name), (s, y) in acc.items():
        s, y = torch.cat(s), torch.cat(y)
        heads.setdefault(w, {})[f"{'true' if key == 'p_dead_real' else 'generated'}/{name}"] = {
            "auc_fatal_vs_survived": auc(s, y), "mean_p_dead": float(s.mean()), "true_death_rate": float(y.float().mean())}
    return cf, heads


@torch.no_grad()
def death_direction():
    import interface as I
    pool = torch.load(I.POOL / "pool.pt", weights_only=False, mmap=True)
    var = pool["u"][~pool["terminal"]].reshape(-1, 192).double().var(0)
    encoder, _ = I.load_bridge()
    device = torch.device("cuda")
    out = {}
    for a, name in ((6, "SLEEP"), (0, "NOOP")):
        S, Y, B = [], [], []
        for bi, store in enumerate(BLOCKS.values()):
            rows = [r for f in sorted((ROOT / "artifacts/eda" / store).glob("seed-*.pt")) for r in torch.load(f, weights_only=False)]
            rows = [r for r in rows if r["p_death1"].max() > r["p_death1"].min()]
            zom = strata(torch.stack([r["visible"].float() for r in rows]))["zombie_adjacent"]
            rows = [r for r, k in zip(rows, zom) if k]
            _, g = I.encode(encoder, torch.stack([r["successors"][a] for r in rows])[:, None], device)
            S.append(I.project(pool["pca"], g[:, 0])); Y.append(torch.tensor([bool(r["terminated"][a]) for r in rows]))
            B.append(torch.full((len(rows),), bi))
        S, Y, B = torch.cat(S).double(), torch.cat(Y), torch.cat(B)
        fit, test = B <= 1, B >= 2
        d = S[fit & Y].mean(0) - S[fit & ~Y].mean(0)
        info = d ** 2 / var
        out[name] = {"heldout_auc_raw_mean_difference_in_u": auc(S[test] @ d, Y[test]),
                     "heldout_auc_mean_difference_in_w": auc(S[test] @ (d / var), Y[test]),
                     "share_of_information_in_u_ranks_43_192": float(info[42:].sum() / info.sum()),
                     "share_of_raw_difference_energy_in_ranks_43_192": float((d[42:] ** 2).sum() / (d ** 2).sum())}
    return out


@torch.no_grad()
def zombie_position():
    import jax
    states = T.base_states()
    env, _ = T._env()
    obs = jax.jit(env.get_obs)
    fr = {e: torch.from_numpy(np.stack([np.asarray(obs(s if e == "base" else T.edited(s, e, r, f)))
                                        for s, r, f in states]) * 255.0).round().to(torch.uint8)
          for e in ("base", "zombie", "zombie_far")}
    device = torch.device("cuda")
    natural = T.natural_frames()
    out = {}
    for step in (0, 10000):
        enc = T.encoder_at(step, device)
        Wz = T.inverse_sqrt_cov(T.encode(enc, natural, device)[0])
        Z = {e: T.encode(enc, f, device)[0].double() for e, f in fr.items()}
        dn, df = Z["zombie"] - Z["base"], Z["zombie_far"] - Z["base"]
        half = len(dn) // 2
        y = torch.cat([torch.ones(len(dn) - half), torch.zeros(len(dn) - half)])
        w = Wz @ Wz @ (Z["zombie"][:half].mean(0) - Z["zombie_far"][:half].mean(0))
        w2 = Wz @ Wz @ dn[:half].mean(0)
        out[str(step)] = {
            "cosine_mean_change_adjacent_vs_far": float(torch.nn.functional.cosine_similarity(dn.mean(0), df.mean(0), dim=0)),
            "heldout_auc_adjacent_vs_far": auc(torch.cat([Z["zombie"][half:] @ w, Z["zombie_far"][half:] @ w]), y),
            "heldout_auc_with_vs_without_zombie_same_scene": auc(torch.cat([Z["zombie"][half:] @ w2, Z["base"][half:] @ w2]), y)}
        del enc
    return out


def count_vs_adjacency():
    V = torch.stack([r["visible"].float() for s in ("observe_fresh_v6", "observe_fresh_v7")
                     for f in sorted((ROOT / "artifacts/eda" / s).glob("seed-*.pt")) for r in torch.load(f, weights_only=False)])
    st = strata(V)
    tiles = V[:, :1071].reshape(-1, 7, 9, 17)
    mobs = V[:, 1071:1512].reshape(-1, 7, 9, 7)
    return {"rows": len(V), "auc_lava_count_for_lava_adjacency": auc(tiles[..., 14].sum((1, 2)), st["lava_adjacent"]),
            "auc_zombie_count_for_zombie_adjacency": auc(mobs[..., 0].sum((1, 2)), st["zombie_adjacent"])}


def noop_death():
    H, R, Y = [], [], []
    for b in BLOCKS:
        m = torch.load(DUMP / f"{b}_meta.pt")
        zom = strata(m["visible"])["zombie_adjacent"]
        v, hid = m["visible"][zom], m["hidden"][zom]
        mobs = v[:, 1071:1512].reshape(-1, 7, 9, 7)[..., 0]
        cool = hid[:, :315].reshape(-1, 7, 9, 5)[..., 0] * 5
        ready = torch.stack([(mobs[:, r, c] > 0) & (cool[:, r, c] <= 0) for r, c in ((2, 4), (4, 4), (3, 3), (3, 5))], 1).sum(1)
        H.append(v[:, 1512] * 9); R.append(ready); Y.append(m["terminated"][zom][:, 0].bool())
    H, R, Y = torch.cat(H), torch.cat(R).double(), torch.cat(Y)
    return {"roots": len(Y), "death_rate": float(Y.float().mean()), "auc_visible_health": auc(-H, Y),
            "auc_hidden_cooldown": auc(R, Y), "auc_both": auc(-(H - 2 * R), Y)}


def asleep_deaths():
    n = {"death_frames": 0, "rendered_asleep": 0}
    for path, shards in ((ROOT / "artifacts/craftax_expert_store_v1", 44), (ROOT / "artifacts/craftax_support_v2", 200)):
        for s in sorted(path.glob("shard-*.pt"))[:shards]:
            for e in torch.load(s, weights_only=False, mmap=True)["episodes"]:
                t = e["terminated"].nonzero().flatten()
                if len(t):
                    n["death_frames"] += len(t)
                    n["rendered_asleep"] += int(asleep(e["observations"][t + 1]).sum())
    return n


def main():
    out = {}
    out["counterfactual"], out["asleep_heads"] = counterfactual_and_heads()
    out["count_vs_adjacency"] = count_vs_adjacency()
    out["noop_death"] = noop_death()
    out["death_direction"] = death_direction()
    out["zombie_position"] = zombie_position()
    out["asleep_deaths"] = asleep_deaths()
    (HERE / "followups.json").write_text(json.dumps(out, indent=2) + "\n")
    for k, v in out.items():
        print(k, json.dumps(v)[:600], flush=True)


if __name__ == "__main__":
    main()
