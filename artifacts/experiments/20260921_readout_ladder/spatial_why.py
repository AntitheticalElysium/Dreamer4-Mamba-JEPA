"""Post hoc to `spatial.py` (declared reading spatial_health_fails): the head, or the generated state?

Every arm's trained continuation head chose below the action prior on the sealed block, and every
arm's Brier score was the same 0.541 -- what predicting almost no death on any branch scores. The
corpus terminal frames render health 0 exactly as the fork fatal successors do (checked: health-tile
distance 0 on 400 terminal episodes), so the heads were trained on the right look of death. Two
places remain, and they are separable because the fork rows hold every action's REAL successor:

  the head   reads death on the real successor too poorly     -> it never learned what the forks show
  the world  generates a successor the head cannot read as dead, though the real one it can

Per arm, on the same 800 sealed opportunity roots and the same context, with the same history
(`world(s, a)` at frame 3), the trained heads read (i) the generated successor and (ii) the real
successor encoded by the frozen encoder. Nothing is trained or refit.

  hindsight_auc   within-root AUC of the trained P(dead) on REAL successors, fatal vs surviving
  generated_auc   the same on GENERATED successors
  p_dead          mean trained P(dead) on real and generated fatal / surviving branches
  health          the health head's accuracy and cross-entropy on real (s_3, s_4) and generated pairs

Reading, per arm, committed before the run (post hoc; it does not replace the declared reading):
  hindsight_auc < 0.90                              -> head_fails_hindsight
  hindsight_auc >= 0.90 and generated_auc < 0.60    -> generation_fails
  otherwise                                         -> mixed
"""

import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path("/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA")
HERE = Path(__file__).parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from d4mj.data import _sha256

from boundary import judge_store  # noqa: E402
from diagnose import within_auc  # noqa: E402
from spatial import ARMS, HEALTH, N, SEALED, TOKENS, WORLDS, World, bridge, encode  # noqa: E402


@torch.no_grad()
def readings(world, heads, encoder, config, frames, actions, successors, device, state, health, batch=16):
    from d4mj.train import autocast_context
    out = {k: [] for k in ("gen", "real", "dh_gen", "dh_real")}
    for i in range(0, len(frames), batch):
        z, tokens = encode(encoder, frames[i:i + batch, -4:], device)
        rz, rt = encode(encoder, successors[i:i + batch], device)
        n = len(z)
        s = (z[:, :, None] if state == "z" else tokens.float()).to(device).repeat_interleave(N, 0)
        real = (rz[:, :, None] if state == "z" else rt.float()).to(device).flatten(0, 1)
        past = actions[i:i + batch, -3:].argmax(-1)
        a = torch.cat([past.repeat_interleave(N, 0), torch.arange(N).repeat(n)[:, None]], 1).to(device)
        with autocast_context(config):
            predicted, history = world(s, a)
            for key, succ in (("gen", predicted[:, 3]), ("real", real)):
                agent = world.agent(succ, history[:, 3])[:, None]
                out[key].append((1 - torch.sigmoid(heads(agent)["continuation"][:, 0, 0].float())).view(n, N).cpu())
                if health:
                    out[f"dh_{key}"].append(world.health(s[:, 3], succ).float().view(n, N, HEALTH).cpu())
    return {k: torch.cat(v) for k, v in out.items() if v}


def main():
    started = time.time()
    log = lambda **kw: print(json.dumps({**kw, "seconds": round(time.time() - started, 1)}), flush=True)
    device = torch.device("cuda")
    import os
    os.chdir(ROOT)
    from d4mj.agent import Heads
    judge, manifest, _ = judge_store(SEALED)
    recorded = json.loads((HERE / "evidence/spatial.json").read_text())
    if recorded["judge_manifest"] != manifest:
        raise SystemExit("not the block spatial.py scored")
    rows = [r for f in sorted(SEALED.glob("seed-*.pt")) for r in torch.load(f, weights_only=False)]
    successors = torch.stack([r["successors"] for r in rows])
    target = torch.stack([r["health_delta"] for r in rows]).round().long().clamp(-9, 1) + 9
    del rows
    p = judge["p_death1"]
    opp = p.amax(1) > p.amin(1)
    fatal = p > 0.5
    encoder, config = bridge()
    result = {}
    for arm, (state, health) in ARMS.items():
        stored = torch.load(WORLDS / f"{arm}.pt", map_location="cpu", weights_only=False)
        world, heads = World(1 if state == "z" else TOKENS, health).to(device), Heads(config).to(device)
        world.load_state_dict(stored["world"])
        heads.load_state_dict(stored["heads"])
        world.eval(), heads.eval()
        r = readings(world, heads, encoder, config, judge["frames"][opp], judge["actions"][opp], successors[opp],
                     device, state, health)
        f = fatal[opp]
        block = {}
        for key in ("real", "gen"):
            block[f"{key}_auc"] = within_auc(list(r[key]), list(f))
            block[f"{key}_p_dead_fatal"] = float(r[key][f].mean())
            block[f"{key}_p_dead_surviving"] = float(r[key][~f].mean())
            if health:
                logits, y = r[f"dh_{key}"].flatten(0, 1), target[opp].flatten()
                block[f"{key}_health_ce"] = float(F.cross_entropy(logits, y))
                block[f"{key}_health_accuracy"] = float((logits.argmax(-1) == y).float().mean())
                damaged = y < 9
                block[f"{key}_health_accuracy_on_damage"] = float((logits.argmax(-1)[damaged] == y[damaged]).float().mean())
        block["reading"] = ("head_fails_hindsight" if block["real_auc"] < 0.90 else
                            "generation_fails" if block["gen_auc"] < 0.60 else "mixed")
        result[arm] = block
        log(arm=arm, **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in block.items()})
    evidence = {"schema": "d4mj_spatial_why_v1", "status": "POST HOC on the sealed block spatial.py scored",
                "script_sha256": _sha256(Path(__file__)), "judge_manifest": manifest,
                "opportunity_roots": int(opp.sum()), "result": result}
    (HERE / "evidence/spatial_why.json").write_text(json.dumps(evidence, indent=2) + "\n")
    log(status="spatial_why_complete")


if __name__ == "__main__":
    main()
