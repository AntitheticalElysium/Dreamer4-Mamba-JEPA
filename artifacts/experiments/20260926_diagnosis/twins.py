"""D4. How strongly does the encoder register ONE tile's content, and did training make it so?

Twin game states: a real state, and the same state with exactly one edit on the player's right-hand
neighbour (a grass tile, no mob): add a zombie, add a cow, turn it to lava / stone / tree / water / table.
Also a zombie added far from the player (view row 0 or 6, a grass tile). Each twin is rendered day
(light 1.0) and night (light 0.15, same night-noise RNG), encoded by the Raw encoder at joint steps
0 (init), 500, 2000, 5000, 10000.

Per edit: pixel change; Euclidean change of z, CLS and the edited tile's patch token; and the
DETECTABILITY of the edit against the natural variation of that representation -- Fisher d' of the
mean edit direction, d' = sqrt(dmu^T Sigma^-1 dmu), Sigma = covariance of the representation over
4,000 natural root frames (55k block), eigenvalues floored at 1e-4 of the largest. d' is what an ideal
linear probe can achieve for "this edit is present" against everything else that varies in real frames.
Consistency: mean cosine of each twin's change with the mean change.
"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
import jax
import jax.numpy as jnp
import numpy as np
import torch

ROOT = Path("/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA")
sys.path.insert(0, str(ROOT))
from d4mj.env import _env  # noqa: E402

OUT = Path(__file__).parent / "twins.json"
STEPS = (0, 500, 2000, 5000, 10000)
EDITS = ("zombie", "cow", "lava", "stone", "tree", "water", "table", "zombie_far")
BLOCK = {"lava": 14, "stone": 4, "tree": 5, "water": 3, "table": 11}
GRASS = 2
N_STATES = 300


def base_states():
    env, params = _env()
    step = jax.jit(lambda k, s, a: env.step(k, s, a, params))
    states, seed = [], 900_000
    rng = np.random.default_rng(0)
    while len(states) < N_STATES:
        _, s = env.reset(jax.random.PRNGKey(seed), params)
        for t in range(int(rng.integers(5, 120))):
            _, s, _, done, _ = step(jax.random.PRNGKey(seed * 1000 + t), s, int(rng.integers(0, 5)))
            if bool(done):
                break
        seed += 1
        if bool(done):
            continue
        p = np.asarray(s.player_position)
        right = p + np.array([0, 1])
        mobs = [np.asarray(g.position)[np.asarray(g.mask)] for g in (s.zombies, s.cows, s.skeletons)]
        occupied = {tuple(x) for m in mobs for x in m}
        far = [p + np.array([dr, dc]) for dr in (-3, 3) for dc in range(-4, 5)]
        far = [f for f in far if int(s.map[f[0], f[1]]) == GRASS and tuple(f) not in occupied]
        if (int(s.map[right[0], right[1]]) != GRASS or tuple(right) in occupied or not far
                or bool(np.asarray(s.zombies.mask).all()) or bool(np.asarray(s.cows.mask).all())):
            continue
        states.append((s, right, far[0]))
    return states


def with_mob(s, group, position, health):
    g = getattr(s, group)
    slot = int(np.argmin(np.asarray(g.mask)))
    g = g.replace(position=g.position.at[slot].set(jnp.asarray(position, jnp.int32)),
                  mask=g.mask.at[slot].set(True), health=g.health.at[slot].set(health),
                  attack_cooldown=g.attack_cooldown.at[slot].set(0))
    return s.replace(**{group: g}, mob_map=s.mob_map.at[position[0], position[1]].set(True))


def edited(s, edit, right, far):
    if edit == "zombie":
        return with_mob(s, "zombies", right, 5)
    if edit == "zombie_far":
        return with_mob(s, "zombies", far, 5)
    if edit == "cow":
        return with_mob(s, "cows", right, 3)
    return s.replace(map=s.map.at[right[0], right[1]].set(BLOCK[edit]))


def render_all(states):
    env, params = _env()
    obs = jax.jit(env.get_obs)
    frames = {}
    for light in ("day", "night"):
        level = 1.0 if light == "day" else 0.15
        for edit in ("base",) + EDITS:
            out = []
            for s, right, far in states:
                t = s if edit == "base" else edited(s, edit, right, far)
                t = t.replace(light_level=jnp.asarray(level, jnp.float32))
                out.append(np.asarray(obs(t)))
            frames[(light, edit)] = torch.from_numpy(np.stack(out) * 255.0).round().to(torch.uint8)
    return frames


def encoder_at(step, device):
    from d4mj.checkpoint import read_lewm_bundle
    from d4mj.config import config_from_dict
    from d4mj.world_api import ModelBundle
    payload = read_lewm_bundle(ROOT / f"artifacts/lewm_m4_canonical/raw/joint/step-{step:06d}.pt")
    bundle = ModelBundle.create(config_from_dict(payload["config"]))
    bundle.encoder.load_state_dict(payload["modules"]["encoder"])
    return bundle.encoder.to(device).eval()


@torch.no_grad()
def encode(encoder, frames, device, batch=64):
    zs, cs, ts = [], [], []
    for i in range(0, len(frames), batch):
        z, cls, tokens, _, _ = encoder._hidden(frames[i:i + batch, None].to(device))
        zs.append(z.float().cpu()); cs.append(cls.float().cpu()); ts.append(tokens.float().cpu())
    return torch.cat(zs), torch.cat(cs), torch.cat(ts)


def inverse_sqrt_cov(x):
    x = x.double() - x.double().mean(0)
    values, vectors = torch.linalg.eigh(x.T @ x / (len(x) - 1))
    values = values.clamp_min(values.max() * 1e-4)
    return vectors @ torch.diag(values.rsqrt()) @ vectors.T


def natural_frames():
    files = sorted((ROOT / "artifacts/eda/observe_fresh_v6").glob("seed-*.pt"))
    frames = [r["frames"][-1] for f in files for r in torch.load(f, weights_only=False)]
    return torch.stack(frames[:4000])


def main():
    device = torch.device("cuda")
    states = base_states()
    print("states", len(states), flush=True)
    frames = render_all(states)
    natural = natural_frames()
    tile = 3 * 9 + 5                                        # right neighbour's token in the 9x9 grid
    result = {"n_states": len(states), "pixel_change": {}, "by_step": {}}
    for (light, edit), f in frames.items():
        if edit != "base":
            d = (f.float() - frames[(light, "base")].float()) / 255.0
            result["pixel_change"][f"{light}/{edit}"] = float(d.flatten(1).norm(dim=1).mean())
    for step in STEPS:
        encoder = encoder_at(step, device)
        nz, nc, nt = encode(encoder, natural, device)
        whiten = {"z": inverse_sqrt_cov(nz), "cls": inverse_sqrt_cov(nc), "token": inverse_sqrt_cov(nt[:, tile])}
        scale = {"z": float((nz - nz.mean(0)).norm(dim=1).median()), "cls": float((nc - nc.mean(0)).norm(dim=1).median()),
                 "token": float((nt[:, tile] - nt[:, tile].mean(0)).norm(dim=1).median())}
        enc = {k: encode(encoder, f, device) for k, f in frames.items()}
        row = {}
        for (light, edit) in frames:
            if edit == "base":
                continue
            base = enc[(light, "base")]
            cur = enc[(light, edit)]
            entry = {}
            for name, i in (("z", 0), ("cls", 1), ("token", 2)):
                a, b = base[i], cur[i]
                if name == "token":
                    a, b = a[:, tile], b[:, tile]
                delta = (b - a).double()
                mean = delta.mean(0)
                entry[name] = {"euclid_over_natural_radius": float(delta.norm(dim=1).mean() / scale[name]),
                               "fisher_dprime": float((whiten[name] @ mean).norm()),
                               "consistency": float(torch.nn.functional.cosine_similarity(delta, mean[None], dim=1).mean())}
            row[f"{light}/{edit}"] = entry
        result["by_step"][str(step)] = row
        print(step, json.dumps({k: round(v["z"]["fisher_dprime"], 2) for k, v in row.items()}), flush=True)
        del encoder
        torch.cuda.empty_cache()
    OUT.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
