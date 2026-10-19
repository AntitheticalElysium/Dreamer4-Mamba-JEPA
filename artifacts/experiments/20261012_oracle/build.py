"""10-12 oracle-innovation test (NOTEBOOK "10-12 ORACLE-INNOVATION TEST", predeclared): data.
  layouts  for every support-v2 row of the Raw long-pool ledger (20260929_mamba_integration/long_ledger.jsonl; 17,133 of 19,789
           rows; expert_v1 rows are not replayable and are excluded from every arm), replay the episode exactly and write the
           visible mob layout of each of the row's 64 frames: [rows, 64, 63, 4] uint8 (zombie, cow, skeleton, any arrow; the cell
           geometry of observe.state_features). Checks: the episode id matches the replayed slot; replayed renders equal the stored
           frames; pool tokens equal the Raw encoder on the stored frames (fp16 rounding).
  panel    200 roots, one per support-v2 DEV-split episode (never in the TRAIN ledger), root t uniform in [15, len - 17]: Raw tokens
           of frames t-15 .. t+16 from the stored observations, their mob layouts (replay), the 31 actions between them.
Usage: build.py layouts|panel  -> artifacts/eda/oracle_v1/
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
import torch.nn.functional as F

ROOT = Path("/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA")
sys.path.insert(0, str(ROOT / "artifacts/experiments/20260927_levers"))
sys.path.insert(0, str(ROOT / "artifacts/experiments/20260926_diagnosis"))
sys.path.insert(0, str(ROOT / "artifacts/experiments/20260921_readout_ladder"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "artifacts/eda"))                          # first: other folders also have a replay.py
import replay as RP  # noqa: E402
assert hasattr(RP, "env_and_render"), RP.__file__

OUT = ROOT / "artifacts/eda/oracle_v1"
LEDGER = ROOT / "artifacts/experiments/20260929_mamba_integration/long_ledger.jsonl"
POOL = ROOT / "artifacts/eda/levers_mamba_long_pools_v1/raw"
SEED = 20261012


@jax.jit
def _mob_scan(reset_key, env_keys, actions, stop):
    env, params, _, _, _ = RP.env_and_render()
    _, state = env.reset(reset_key, params)

    def rec(s):
        return (s.player_position, s.zombies.position, s.zombies.mask, s.cows.position, s.cows.mask,
                s.skeletons.position, s.skeletons.mask, s.arrows.position, s.arrows.mask)

    def body(carry, xs):
        s, i = carry
        k, a = xs
        _, nxt, _, _, _ = env.step(k, s, a, params)
        nxt = jax.tree.map(lambda x, y: jnp.where(i < stop, x, y), nxt, s)
        return (nxt, i + 1), rec(nxt)

    first = rec(state)
    _, seq = jax.lax.scan(body, (state, 0), (env_keys, actions))
    return jax.tree.map(lambda f, q: jnp.concatenate([f[None], q]), first, seq)


def mob_layouts(shard, slot, upto):
    """[upto + 1, 63, 4] uint8: frame f = state after f actions"""
    reset_key, env_keys = RP._slot_keys(shard, slot)
    acts = RP.episode_fields(shard, slot)["actions_taken"].numpy()
    bucket = next(b for b in RP.BUCKETS if b >= max(upto, 1))
    pad = np.zeros(bucket, dtype=np.int32)
    pad[:min(len(acts), bucket)] = acts[:bucket]
    r = [np.asarray(x)[:upto + 1] for x in _mob_scan(reset_key, env_keys[:bucket], jnp.asarray(pad), upto)]
    player = r[0]
    out = np.zeros((upto + 1, 63, 4), np.uint8)
    for ch, (pos, mask) in enumerate(((r[1], r[2]), (r[3], r[4]), (r[5], r[6]), (r[7], r[8]))):
        local = pos - player[:, None] + np.array([3, 4])
        on = mask & (local >= 0).all(-1) & (local < [7, 9]).all(-1)
        f, i = np.nonzero(on)
        out[f, local[f, i, 0] * 9 + local[f, i, 1], ch] = 1
    return out


def parse(eid):
    _, _, rnd, slot = eid.split(":")
    return int(rnd), int(slot)


def encoder():
    import teval as T
    from d4mj.checkpoint import read_lewm_bundle
    from d4mj.config import config_from_dict
    from d4mj.world_api import ModelBundle
    payload = read_lewm_bundle(T.ENCODERS["raw"])
    bundle = ModelBundle.create(config_from_dict(payload["config"]))
    bundle.encoder.load_state_dict(payload["modules"]["encoder"])
    enc = bundle.encoder.to("cpu").freeze()

    @torch.no_grad()
    def tok(frames):                                                       # uint8 [n,63,63,3] -> LN tokens [n,81,192] fp16
        return torch.cat([F.layer_norm(enc._hidden(frames[i:i + 64, None])[2].float(), (192,)).half()
                          for i in range(0, len(frames), 64)])
    return tok


def cmd_layouts():
    rows = [json.loads(l) for l in open(LEDGER)]
    n = len(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    lay = np.lib.format.open_memmap(OUT / "layouts.npy", mode="w+", dtype=np.uint8, shape=(n, 64, 63, 4))
    valid = np.zeros(n, bool)
    by_ep = {}
    for i, r in enumerate(rows):
        if r["episode_id"].startswith("support-v2"):
            by_ep.setdefault(r["episode_id"], []).append(i)
    checked = 0
    for k, (eid, idx) in enumerate(by_ep.items()):
        shard, slot = parse(eid)
        fields = RP.episode_fields(shard, slot)
        assert fields["episode_id"] == eid, (fields["episode_id"], eid)
        upto = max(rows[i]["start"] for i in idx) + 63
        L = mob_layouts(shard, slot, min(upto, len(fields["actions_taken"])))
        for i in idx:
            st = rows[i]["start"]
            seg = L[st:st + 64]
            lay[i, :len(seg)] = seg
            valid[i] = len(seg) == 64
        if k % 500 == 0:
            print(f"layouts episodes {k}/{len(by_ep)}", flush=True)
    lay.flush()
    np.save(OUT / "layouts_valid.npy", valid)
    # checks: replayed render == stored frame; pool tokens == encoder(stored frames)
    _, _, _, _, frame = RP.env_and_render()
    tok = encoder()
    ptok = np.memmap(POOL / "tokens.f16", dtype=np.float16, mode="r", shape=(n, 64, 81, 192))
    rng = np.random.default_rng(SEED)
    worst_px, worst_tok = 0, 0.0
    for i in rng.choice(np.flatnonzero(valid), 6, replace=False).tolist():
        shard, slot = parse(rows[i]["episode_id"])
        st = rows[i]["start"]
        obs = RP.episode_fields(shard, slot)["observations"]
        for j in (0, 31, 63):
            s = RP.advance_to(shard, slot, st + j)
            worst_px = max(worst_px, int(np.abs(np.asarray(frame(s)).astype(int) - obs[st + j].numpy().astype(int)).max()))
        t = tok(obs[st:st + 64])
        worst_tok = max(worst_tok, float((t.float() - torch.from_numpy(np.array(ptok[i])).float()).abs().max()))
    res = {"rows": n, "support_rows_valid": int(valid.sum()), "episodes": len(by_ep),
           "check_render_maxdiff": worst_px, "check_pool_token_maxabs": round(worst_tok, 5),
           "mob_cells_per_frame": [round(float(lay[valid][:, :, :, c].sum(2).mean()), 4) for c in range(4)]}
    (OUT / "layouts.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res), flush=True)


def cmd_panel(n=200):
    rng = np.random.default_rng(SEED + 1)
    shards = rng.permutation(len(RP.manifest()["shards"]))
    roots = []
    for sh in shards.tolist():
        eps = RP._shard(sh)["episodes"]
        dev = [i for i, e in enumerate(eps) if e["split"] == "dev" and len(e["actions_taken"]) >= 64]
        if not dev:
            continue
        slot = int(rng.choice(dev))
        n_act = len(eps[slot]["actions_taken"])
        roots.append({"shard": sh, "slot": slot, "t": int(rng.integers(15, n_act - 17 + 1)), "episode_id": eps[slot]["episode_id"]})
        if len(roots) == n:
            break
    tok = encoder()
    toks, lays, acts = [], [], []
    for r in roots:
        f = RP.episode_fields(r["shard"], r["slot"])
        a, b = r["t"] - 15, r["t"] + 16                                    # frames a..b (32)
        toks.append(tok(f["observations"][a:b + 1]))
        lays.append(torch.from_numpy(mob_layouts(r["shard"], r["slot"], b)[a:b + 1]))
        acts.append(f["actions_taken"][a:b].long())
    OUT.mkdir(parents=True, exist_ok=True)
    torch.save({"roots": roots, "tokens": torch.stack(toks), "layouts": torch.stack(lays), "actions": torch.stack(acts)},
               OUT / "panel.pt")
    L = torch.stack(lays).float()
    print(json.dumps({"roots": len(roots), "tokens": list(torch.stack(toks).shape),
                      "mob_cells_per_frame": [round(float(L[..., c].sum(-1).mean()), 4) for c in range(4)]}), flush=True)


if __name__ == "__main__":
    RP.env_and_render()
    {"layouts": cmd_layouts, "panel": cmd_panel}[sys.argv[1]]()
