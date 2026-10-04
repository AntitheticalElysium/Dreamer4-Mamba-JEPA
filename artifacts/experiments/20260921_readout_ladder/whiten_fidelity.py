"""Post hoc (56k, read before): WHY does the whitened world's trained head choose better?

Its own head's within-root death AUC on REAL vs IMAGINED successors, mean P(dead) on fatal vs surviving
branches, and interface.dev_fidelity on 400 DEV terminal episodes (death vs alive-10 / pre-death at the
evaluator position), for W and U seed 1. Descriptive; no fitting.
"""
import json, os, sys, torch
from pathlib import Path
ROOT = Path("/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA"); HERE = Path(__file__).parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(HERE))
from d4mj.data import _sha256


def main():
    os.chdir(ROOT)
    device = torch.device("cuda")
    import interface as I
    from d4mj.agent import Heads
    from d4mj.data import load_joint_corpus
    from boundary import judge_store
    from diagnose import within_auc
    from frozen_ladder import strata
    from whiten import whitened_pool
    pool, _ = whitened_pool()
    store = ROOT / "artifacts/eda/observe_fresh_v7"
    judge, manifest, _ = judge_store(store)
    judge["successors"] = torch.stack([r["successors"] for f in sorted(store.glob("seed-*.pt")) for r in torch.load(f, weights_only=False)])
    encoder, config = I.load_bridge()
    record = json.loads(I.DATASET.read_text())
    episodes, _ = load_joint_corpus(record["paths"], config)
    term = [e for e in episodes if e.split == "dev" and bool(e.terminated[-1]) and len(e) >= 15]
    term = [term[i] for i in torch.randperm(len(term), generator=torch.Generator().manual_seed(20261008))[:400].tolist()]
    pj = judge["p_death1"]; fatal = pj > 0.5
    opp = fatal.any(1) & (~fatal).any(1)
    zombie = opp & strata(judge["visible"])["zombie_adjacent"]
    out = {}
    for name, (arm, path) in {"W": ("W", ROOT / "artifacts/eda/interface_worlds_white/W.pt"),
                              "U_s1": ("U", ROOT / "artifacts/eda/interface_worlds_v1/U.pt")}.items():
        stored = torch.load(path, map_location="cpu", weights_only=False)
        b = I.world_bundle(config, encoder, device); b.world.load_state_dict(stored["world"]); b.world.eval()
        h = Heads(config).to(device); h.load_state_dict(stored["heads"]); h.eval()
        r = I.branches(b, h, pool["pca"], arm, encoder, judge["frames"], judge["actions"], device, judge["successors"])
        block = {}
        for sname, m in (("all", opp), ("zombie", zombie)):
            block[sname] = {"auc_real": within_auc(list(r["p_dead_real"][m]), list(fatal[m])),
                            "auc_generated": within_auc(list(r["p_dead"][m]), list(fatal[m])),
                            "p_dead_generated_fatal": float(r["p_dead"][m][fatal[m]].mean()),
                            "p_dead_generated_surviving": float(r["p_dead"][m][~fatal[m]].mean()),
                            "p_dead_real_fatal": float(r["p_dead_real"][m][fatal[m]].mean())}
        block["dev_fidelity"] = I.dev_fidelity(b, h, pool["pca"], arm, encoder, device, term)
        out[name] = block
        print(json.dumps({"world": name, **{k: {kk: round(vv, 4) for kk, vv in v.items()} for k, v in block.items()}}), flush=True)
        del b, h
    (HERE / "evidence/whiten_fidelity.json").write_text(json.dumps({"schema": "d4mj_whiten_fidelity_v1", "status": "POST HOC 56k",
        "script_sha256": _sha256(Path(__file__)), "judge_manifest": manifest, "result": out}, indent=2) + "\n")


if __name__ == "__main__":
    main()
