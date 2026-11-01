"""Independent reproduction of the 10-28 finding (observational diagnosis, section 1) before any diagnosis is designed.

CLAIM (Codex, 10-28): from real histories, M0's one-step generated successors carry far fewer deaths and negative
rewards than recorded (death 0.0034% vs 0.28%; none of 42 deaths in low-health / adjacent-zombie contexts), while the
same heads on the recorded successors are calibrated.
COMPETING EXPLANATION: an accounting or implementation defect in that measurement (it used S1's batched recompute
sampler and its tuple bookkeeping), or a sampling accident of one tuple draw.
CONTROL: different code and draws: the PRODUCTION streaming sampler (model.sample + model.context_pass on a cache built
from the 8 real context frames), fresh tuples (seed 2029), the heads also applied to the recorded successors (must
reproduce the recorded rates), and many futures per risky history so a small probability is estimated, not hit or miss.
OUTCOMES: deficit reproduced -> it is a property of the frozen world, and diagnosis can start from it; not reproduced ->
the 10-28 numbers are an artefact and the diagnosis target changes.
Frozen world (checkpoint 10, world seed 1); dev split; simulator labels only to select contexts and count outcomes.
Usage (project venv, GPU): python -m artifacts.experiments.20261029_consequence_reproduction.reproduce OUT_JSON
"""
import json, sys
from pathlib import Path

import numpy as np
import torch

from saccade import model as M, train as TR
from saccade.eval import s1 as S, gates as G

ROOT = Path(__file__).resolve().parents[3]
CKPT = ROOT / "artifacts/saccade_m0/p1_world1/ckpt_10.pt"
DEV = torch.device("cuda")
P = 8                                                                    # real context frames before the target


def contexts(L, enc, rng, n_random):
    """(episode, t) targets with t >= P. Random: uniform over such frames. Risky: health <= 2 and a zombie at L1
    distance 1 at state t-1 (the attack rule's adjacency, game_logic.update_mobs). Random-behaviour: eps = 1.0."""
    s_off = np.r_[0, np.cumsum(L["ep_T"] + 1)]
    pos = L["s_pos"].astype(int)
    zp, zm = L["s_zombies_p"].astype(int), L["s_zombies_m"]
    adjacent = (zm & (np.abs(zp - pos[:, None]).sum(-1) == 1)).any(1)
    low = L["s_body"][:, 0] <= 2
    cand, risky = [], []
    for e in range(len(enc)):
        for t in range(P, len(enc[e]["p"])):
            cand.append((e, t))
            g = s_off[e] + t - 1
            if low[g] and adjacent[g]:
                risky.append((e, t))
    pick = rng.choice(len(cand), n_random, replace=False)
    return [cand[i] for i in pick], risky


@torch.no_grad()
def generate(model, enc, tuples, K, cfg, gen):
    """K independent futures per tuple through the production streaming sampler; heads on generated and recorded
    successors. Returns per-future arrays."""
    den, out = model.denoiser, {k: [] for k in ("gen_death_p", "gen_death", "gen_neg_p", "gen_reward",
                                                  "real_death_p", "real_neg_p", "rec_term", "rec_reward", "tuple")}
    neg_bins = model.outcome.bin_value < -0.05
    step = max(1, 256 // K)              # <= 256 futures per call: streaming temporal attention flattens B x 95 slots,
    for start in range(0, len(tuples), step):    # and SDPA failed at B = 1,024 (97,280 rows; "invalid argument")
        batch = tuples[start:start + step]
        rows = [i for i in range(len(batch)) for _ in range(K)]
        B = len(rows)
        take = lambda key, dt: torch.stack([enc[batch[i][0]][key][batch[i][1] + dt] for i in rows]).to(DEV)
        cache = den.new_cache()
        with TR.autocast(DEV, cfg):
            for dt in range(-P, 0):                                      # real context, oldest first
                x = tuple(take(k, dt).float()[:, None] for k in ("p", "g", "v"))
                cache = M.context_pass(den, x, take("prev_action", dt)[:, None],
                                       model.belief.memory(take("bp", dt).float())[:, None], take("fix", dt)[:, None],
                                       cache, cfg, gen)
            shapes = [tuple(enc[0][k].shape[1:]) for k in ("p", "g", "v")]
            bp = take("bp", 0).float()
            xg = M.sample(den, shapes, take("prev_action", 0)[:, None], model.belief.memory(bp)[:, None],
                          take("fix", 0)[:, None], cache, cfg, gen)
            for name, parts, r_in in (("gen", tuple(a[:, 0] for a in xg), None),
                                      ("real", tuple(take(k, 0).float() for k in ("p", "g", "v")), take("reward", 0).float())):
                Pc = M.Percept(*parts, take("fix", 0))
                h = model.outcome.features(bp, model.belief.percept_embed(Pc)).float()
                lp = model.outcome.bin_logp(h)
                r = model.outcome.sample_reward(h, gen) if r_in is None else r_in
                death_p = 1 - torch.sigmoid(model.outcome.cont_logit(h, r).float())
                out[f"{name}_death_p"].append(death_p.cpu()); out[f"{name}_neg_p"].append((lp.exp() * neg_bins).sum(-1).cpu())
                if r_in is None:
                    out["gen_reward"].append(r.cpu())
                    out["gen_death"].append((torch.rand(B, device=DEV, generator=gen) < death_p).cpu())
        out["rec_term"].append(take("terminated", 0).cpu()); out["rec_reward"].append(take("reward", 0).cpu())
        out["tuple"].append(torch.tensor([start + i for i in rows]))
    return {k: torch.cat(v).numpy() for k, v in out.items()}


def summary(o, ep):
    one = np.ones(len(ep), bool)
    est = lambda v: G.estimate(np.asarray(v, float), one, ep)
    return {"futures": int(len(ep)), "histories": int(len(np.unique(o["tuple"]))),
            "recorded_death_rate": est(o["rec_term"]), "head_on_recorded_death_p": est(o["real_death_p"]),
            "generated_death_p": est(o["gen_death_p"]), "generated_death_sampled": est(o["gen_death"]),
            "recorded_negative_reward_rate": est(o["rec_reward"] < -0.05), "head_on_recorded_negative_p": est(o["real_neg_p"]),
            "generated_negative_p": est(o["gen_neg_p"]), "generated_negative_sampled": est(o["gen_reward"] < -0.05)}


def main():
    out_path = Path(sys.argv[1]); out_path.parent.mkdir(parents=True, exist_ok=True)
    ck = torch.load(CKPT, weights_only=False, map_location="cpu")
    cfg = ck["cfg"]
    model = M.M0(cfg).to(DEV); model.load_state_dict(ck["model"]); model.eval()
    L = dict(np.load(ROOT / "artifacts/eda/s0/s0_dev_labels.npz"))
    enc = S.encode(model, L, cfg)
    rng = np.random.default_rng(2029)
    rand, risky = contexts(L, enc, rng, 6000)
    gen = torch.Generator(device=DEV).manual_seed(2029)
    res = {"checkpoint_sha256": TR.file_sha256(CKPT), "sampler": cfg["edm"]["sampler"], "context_frames": P}
    eps_rate = L["ep_eps"]
    for name, tuples, K in (("random", rand, 4), ("risky_low_health_adjacent_zombie", risky, 16)):
        o = generate(model, enc, tuples, K, cfg, gen)
        ep = np.array([tuples[i][0] for i in o["tuple"]])
        res[name] = summary(o, ep)
        if name == "random":
            rb = eps_rate[ep] == 1.0
            res["random_behaviour_subset"] = summary({k: v[rb] for k, v in o.items()}, ep[rb])
        print(name, json.dumps({k: (v["est"] if isinstance(v, dict) else v) for k, v in res[name].items()}), flush=True)
    out_path.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
