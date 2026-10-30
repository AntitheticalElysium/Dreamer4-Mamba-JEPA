"""The world at the actor's interface (user, 10-30: before more mechanism work, measure what P3 would consume). P3
trains on imagined returns read by the outcome heads (reward, continuation) from generated futures; 85% of dev reward
is achievement unlocks. One-step futures from real histories, P1 world 1 checkpoint 10 and M0.5 guided_1500.
CLAIM: the consequence failures (P1 generation diagnosis, M0.5 dissection) reach the actor's imagined rewards and
terminations. COMPETING EXPLANATION: the heads read the predicted belief too (it knows the action and, per S1.2, the
unlocked achievements), so they could recover an event the generated percept misses. CONTROL: the same heads on the
RECORDED successor of the same histories (calibration); the same histories and noise draws for every checkpoint.
Classes (dev; reproduce.py's production path: streaming sampler after 8 real context frames, S1-seeded fixations):
- RANDOM histories: unconditional totals (law of total expectation holds without conditioning).
- per ACHIEVEMENT: transitions that unlocked it (outcome-defined; recall for deterministic unlocks; the stochastic ones,
  sapling, eat cow, defeat zombie / skeleton, wake up, are read with the context classes below).
- SAPLING context: DO facing grass, awake, sapling not yet unlocked (context-defined, recorded rate ~0.07).
- RISKY (health <= 2, zombie adjacent) and ADJACENT (health >= 3, zombie adjacent, awake): context-defined damage.
Per class also the head's P(unlock) split by whether the recorded transition unlocked anything (false unlocks).
Per future: P(unlock) = head mass on reward bins decoding above 0.5; expected reward, its positive and negative parts;
P(reward < -0.05); P(termination). Recorded: reward, unlock, termination.
Usage (project venv, GPU): python -m artifacts.experiments.20261030_actor_interface.interface OUT_JSON [--episodes=K]
  [--classes=a,b] LABEL=CKPT ...   (--classes keeps the class lists and seeds of the full run, generating only those)
"""
import importlib, json, os, sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")          # craftax's constants import JAX, which would preallocate the GPU
import numpy as np
import torch
import craftax.craftax_classic.constants as C

from saccade import model as M, train as TR
from saccade.eval import s1 as S, gates as G

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "artifacts/saccade_m0/p1_world1/ckpt_10.pt"
DEV = torch.device("cuda")
P = 8
DIRS = importlib.import_module("artifacts.experiments.20261029_generation_diagnosis.field_calibration").DIRS
NAMES = [a.name for a in C.Achievement]
PER_ACH, K_ACH = 300, 4


def load(path):
    ck = torch.load(ROOT / path, weights_only=False, map_location="cpu")
    cfg = ck.get("cfg") or torch.load(BASE, weights_only=False, map_location="cpu")["cfg"]
    model = M.M0(cfg).to(DEV); model.load_state_dict(ck["model"]); model.eval()
    return model, cfg


def classes(L, rng):
    """(episode, t) targets, t >= P, per class (labels at state t - 1 and t)."""
    s_off, a_off = np.r_[0, np.cumsum(L["ep_T"] + 1)], np.r_[0, np.cumsum(L["ep_T"])]
    pos, d = L["s_pos"].astype(int), DIRS[L["s_dir"]]
    face = L["s_view"][np.arange(len(d)), 3 + d[:, 0], 4 + d[:, 1]]
    zp, zm = L["s_zombies_p"].astype(int), L["s_zombies_m"]
    adjacent = (zm & (np.abs(zp - pos[:, None]).sum(-1) == 1)).any(1)
    health, sleep, ach = L["s_body"][:, 0], L["s_sleep"], L["s_ach"]
    out = {"random": [], "sapling_context": [], "risky": [], "adjacent": [], **{f"ach_{n}": [] for n in NAMES}}
    for e in range(len(L["ep_T"])):
        for t in range(P, int(L["ep_T"][e]) + 1):
            g = s_off[e] + t
            a = int(L["t_action"][a_off[e] + t - 1])
            out["random"].append((e, t))
            for k in np.nonzero((ach[g] > 0) & (ach[g - 1] == 0))[0]:
                out[f"ach_{NAMES[k]}"].append((e, t))
            if a == C.Action.DO.value and face[g - 1] == C.BlockType.GRASS.value and not sleep[g - 1] \
                    and not ach[g - 1, C.Achievement.COLLECT_SAPLING.value]:
                out["sapling_context"].append((e, t))
            if adjacent[g - 1] and health[g - 1] <= 2:
                out["risky"].append((e, t))
            if adjacent[g - 1] and health[g - 1] >= 3 and not sleep[g - 1]:
                out["adjacent"].append((e, t))
    caps = {"random": 6000, "sapling_context": 2000, "risky": 4000, "adjacent": 3000}
    for k, v in out.items():
        n = caps.get(k, PER_ACH)
        if len(v) > n:
            out[k] = [v[i] for i in np.sort(rng.choice(len(v), n, replace=False))]
    return out


@torch.no_grad()
def futures(model, enc, L, tuples, K, cfg, gen):
    """K production futures per history; the outcome heads on the generated and on the recorded successor."""
    den, ob = model.denoiser, model.outcome
    v = ob.bin_value
    s_off = np.r_[0, np.cumsum(L["ep_T"] + 1)]
    keys = [f"{w}_{q}" for w in ("gen", "rec_head") for q in ("unlock_p", "exp_reward", "pos", "neg", "neg_p", "term_p")]
    out = {k: [] for k in keys + ["rec_reward", "rec_unlock", "rec_term", "tuple"]}
    step = max(1, 256 // K)
    for start in range(0, len(tuples), step):
        batch = tuples[start:start + step]
        rows = [i for i in range(len(batch)) for _ in range(K)]
        take = lambda key, dt: torch.stack([enc[batch[i][0]][key][batch[i][1] + dt] for i in rows]).to(DEV)
        cache = den.new_cache()
        with TR.autocast(DEV, cfg):
            for dt in range(-P, 0):
                x = tuple(take(k, dt).float()[:, None] for k in ("p", "g", "v"))
                cache = M.context_pass(den, x, take("prev_action", dt)[:, None],
                                       model.belief.memory(take("bp", dt).float())[:, None], take("fix", dt)[:, None],
                                       cache, cfg, gen)
            shapes = [tuple(enc[0][k].shape[1:]) for k in ("p", "g", "v")]
            bp = take("bp", 0).float()
            xg = M.sample(den, shapes, take("prev_action", 0)[:, None], model.belief.memory(bp)[:, None],
                          take("fix", 0)[:, None], cache, cfg, gen)
            for name, parts, r_in in (("gen", tuple(a[:, 0] for a in xg), None),
                                      ("rec_head", tuple(take(k, 0).float() for k in ("p", "g", "v")), take("reward", 0).float())):
                h = ob.features(bp, model.belief.percept_embed(M.Percept(*parts, take("fix", 0)))).float()
                p = ob.bin_logp(h).exp()
                r = ob.sample_reward(h, gen) if r_in is None else r_in
                for q, val in (("unlock_p", (p * (v > 0.5)).sum(-1)), ("exp_reward", (p * v).sum(-1)),
                               ("pos", (p * v.clamp(min=0)).sum(-1)), ("neg", (p * v.clamp(max=0)).sum(-1)),
                               ("neg_p", (p * (v < -0.05)).sum(-1)), ("term_p", 1 - torch.sigmoid(ob.cont_logit(h, r).float()))):
                    out[f"{name}_{q}"].append(val.float().cpu())
        g = np.array([s_off[batch[i][0]] + batch[i][1] for i in rows])
        out["rec_reward"].append(take("reward", 0).float().cpu())
        out["rec_unlock"].append(torch.as_tensor(((L["s_ach"][g] > 0) & (L["s_ach"][g - 1] == 0)).any(1)))
        out["rec_term"].append(take("terminated", 0).float().cpu())
        out["tuple"].append(torch.tensor([start + i for i in rows]))
    return {k: torch.cat(v).numpy() for k, v in out.items()}


def summary(o, ep):
    one = np.ones(len(ep), bool)
    est = lambda x: G.estimate(np.asarray(x, float), one, ep)
    r = o["rec_reward"]
    return {"futures": int(len(ep)), "histories": int(len(np.unique(o["tuple"]))),
            "recorded": {"unlock": est(o["rec_unlock"]), "exp_reward": est(r), "pos": est(np.clip(r, 0, None)),
                         "neg": est(np.clip(r, None, 0)), "neg_p": est(r < -0.05), "term_p": est(o["rec_term"])},
            **{w: {q: est(o[f"{w}_{q}"]) for q in ("unlock_p", "exp_reward", "pos", "neg", "neg_p", "term_p")}
               for w in ("rec_head", "gen")},
            "split_by_recorded_unlock": {k: {w: G.estimate(np.asarray(o[f"{w}_unlock_p"], float), m, ep) for w in ("rec_head", "gen")}
                                         for k, m in (("unlocked", o["rec_unlock"].astype(bool)), ("not_unlocked", ~o["rec_unlock"].astype(bool)))
                                         if m.any()}}


def main():
    out_path = Path(sys.argv[1]); out_path.parent.mkdir(parents=True, exist_ok=True)
    args = sys.argv[2:]
    n_ep = int(args.pop(0).split("=")[1]) if args and args[0].startswith("--episodes=") else 0
    only = args.pop(0).split("=")[1].split(",") if args and args[0].startswith("--classes=") else None
    pairs = [a.split("=", 1) for a in args]
    L = S.first_episodes(dict(np.load(ROOT / "artifacts/eda/s0/s0_dev_labels.npz")), n_ep)
    cls = classes(L, np.random.default_rng(2031))
    if only:
        cls = {k: (v if k in only else []) for k, v in cls.items()}
    res = {"classes": {k: len(v) for k, v in cls.items()}, "checkpoints": {}, "differences": {}}
    per = {}
    for label, path in pairs:
        model, cfg = load(path)
        enc = S.encode(model, L, cfg)
        res["checkpoints"][label] = {"path": path, "sha256": TR.file_sha256(ROOT / path), "classes": {}}
        for i, (name, tuples) in enumerate(cls.items()):
            if not tuples:
                continue
            K = 4
            gen = torch.Generator(device=DEV).manual_seed(3000 + i)
            o = futures(model, enc, L, tuples, K, cfg, gen)
            ep = np.array([tuples[j][0] for j in o["tuple"]])
            per[(label, name)] = (o, ep)
            res["checkpoints"][label]["classes"][name] = summary(o, ep)
            s = res["checkpoints"][label]["classes"][name]
            print(label, name, "rec unlock", round(s["recorded"]["unlock"]["est"], 3), "head-rec", round(s["rec_head"]["unlock_p"]["est"], 3),
                  "gen", round(s["gen"]["unlock_p"]["est"], 3), "| term rec", round(s["recorded"]["term_p"]["est"], 4),
                  "gen", round(s["gen"]["term_p"]["est"], 5), "| neg_p rec", round(s["recorded"]["neg_p"]["est"], 3),
                  "gen", round(s["gen"]["neg_p"]["est"], 3), flush=True)
        out_path.write_text(json.dumps(res, indent=1, default=str))
        del model, enc
        torch.cuda.empty_cache()
    labels = [l for l, _ in pairs]
    for b in labels[1:]:
        a = labels[0]
        res["differences"][f"{b}-{a}"] = {}
        for name in cls:
            if (a, name) not in per:
                continue
            (oa, ep), (ob, _) = per[(a, name)], per[(b, name)]
            m = np.ones(len(ep), bool)
            res["differences"][f"{b}-{a}"][name] = {}
            for q in ("unlock_p", "exp_reward", "neg_p", "term_p"):
                e = G.estimate(ob[f"gen_{q}"] - oa[f"gen_{q}"], m, ep)
                res["differences"][f"{b}-{a}"][name][q] = {**e, "ci95": [e["est"] - 1.96 * e["se"], e["est"] + 1.96 * e["se"]]}
    out_path.write_text(json.dumps(res, indent=1, default=str))
    print("saved", out_path, flush=True)


if __name__ == "__main__":
    main()
