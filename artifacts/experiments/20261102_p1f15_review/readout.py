"""Read-only reanalysis of saved P1-F1.5 outputs, no new model intervention."""
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import torch

from saccade import model as M

HERE = Path(__file__).resolve().parent
RUN = Path("artifacts/experiments/20261101_p1f15")


def mean_se(x, ep):
    """Episode-cluster bootstrap over the observed rows; trained-world seed is fixed."""
    ep, ids = np.unique(ep, return_inverse=True)
    sums, counts = np.bincount(ids, weights=x), np.bincount(ids)
    rng = np.random.default_rng(1102)
    draws = rng.integers(0, len(ep), (2000, len(ep)))
    vals = sums[draws].sum(1) / counts[draws].sum(1)
    return {"mean": float(np.mean(x)), "se": float(vals.std(ddof=1)), "rows": len(x), "episodes": len(ep)}


def main():
    sources = [RUN / n for n in ("RESULTS.md", "PLAN.md", "consistency.py", "uptake.py", "readings_codex.py")]
    sources += [Path("artifacts/experiments/20261031_learned_fixation_sweep") / n for n in ("TEACHING.md", "locus.py")]
    out = {"scope": "completed-run descriptive reanalysis; no new training or intervention",
           "head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
           "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}}
    ckfile = RUN / "run/ckpt_10.pt"
    ck = torch.load(ckfile, map_location="cpu", weights_only=False, mmap=True)
    s1 = json.loads((RUN / "run/s1/results.json").read_text())
    sha = hashlib.sha256(ckfile.read_bytes()).hexdigest()
    rows = [json.loads(l) for l in (RUN / "run/log.jsonl").read_text().splitlines()]
    mons = [r["monitor"]["g1"] for r in rows if "checkpoint" in r]
    out["identity"] = {"checkpoint_sha256": sha, "s1_checkpoint_matches": s1["checkpoint_sha256"] == sha,
                       "frames": ck["state"]["frames"], "updates": ck["state"]["update"],
                       "monitor_g1": mons, "monitor_strictly_falls": all(a>b for a,b in zip(mons,mons[1:])),
                       "factorised": ck["cfg"]["denoiser"]["factorised"],
                       "faithful": ck["cfg"]["heads"]["pixel_decoder"],
                       "guided_key_absent": "guided" not in ck["cfg"]["p1"],
                       "training_sources_current": {p:hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
                                                    for p,h in ck["provenance"]["source_sha256"].items()}}
    assert out["identity"]["s1_checkpoint_matches"] and out["identity"]["monitor_strictly_falls"]
    del ck
    cen = torch.stack(torch.meshgrid(torch.arange(1,8),torch.arange(1,8),indexing="ij"),-1).reshape(-1,2)
    win = M.window(cen).numpy()
    hud_correct = (win >= 63).any(1)
    hud_old = cen[:,0].numpy() == 7
    assert hud_correct.sum()==14 and hud_old.sum()==7
    z = np.load(RUN / "readings/locus/p1f15.npz")
    random = z["cls"]=="random"
    out["hud_definition"] = {"all_legal_windows":49,"old_definition_windows":int(hud_old.sum()),
                              "any_hud_windows":int(hud_correct.sum()),
                              "old_uniform_reference":float(hud_old.mean()),"correct_uniform_reference":float(hud_correct.mean())}
    out["locus_random"] = {}
    for sig in ("contingency","surprise","predicted_change","spread","realized_change"):
        scores=z[sig][random]; choices=scores.argmax(1); ep=z["ep"][random]
        out["locus_random"][sig] = {"old_hud_share":mean_se(hud_old[choices],ep),
                                     "any_hud_share":mean_se(hud_correct[choices],ep),
                                     "flat_score_rows":int((np.ptp(scores,axis=1)==0).sum()),
                                     "mean_score_by_light":{name:float(scores[m].mean()) for name,m in
                                         (("bright",z["light"][random]>=.7),("dark",z["light"][random]<=.3))}}
    out["locus_attempt_coverage"]={}
    for kind in np.unique(z["cls"]):
        if kind=="random":continue
        m=z["cls"]==kind; ep=z["ep"][m]; cons=[z["counter"][m]]
        if kind in ("coal","iron","stone"):cons.append(z["faced"][m])
        out["locus_attempt_coverage"][str(kind)]={}
        for sig in ("contingency","surprise"):
            c=z[sig][m].argmax(1)
            covered=np.stack([(win[c]==x[:,None]).any(1) for x in cons]).any(0)
            out["locus_attempt_coverage"][str(kind)][sig]=mean_se(covered,ep)
    q=np.load(RUN/"readings/consistency/consistency.npz")
    er=q["er"]; mu=er.mean(1); sd=er.std(1,ddof=1); ranges=er.max(1)-er.min(1); active=mu>.05
    out["query_consistency"]={"contexts":len(mu),"active_mean_reward_above05":int(active.sum()),
                              "active_std_min_median_max":np.quantile(sd[active],[0,.5,1]).tolist(),
                              "active_range_min_median_max":np.quantile(ranges[active],[0,.5,1]).tolist(),
                              "worst_contexts":[]}
    for i in np.argsort(ranges)[-5:][::-1]:
        out["query_consistency"]["worst_contexts"].append({"episode":int(q["ep"][i]),"t":int(q["t"][i]),
            "population":str(q["kind"][i]),"mean_reward":float(mu[i]),"min_per_query_mean":float(er[i].min()),
            "max_per_query_mean":float(er[i].max()),"range":float(ranges[i]),"std":float(sd[i]),
            "largest_reported_per_query_se":float(q["er_se"][i].max())})
    u=json.loads((RUN/"readings/uptake/uptake.json").read_text())
    out["uptake"]={}
    for group in ("random","night"):
        m=u["summary"][group]["map_rows"]
        e=m["missed_mob_minus_none"]
        out["uptake"][group]={"frames":u["summary"][group]["frames"],"qualified_missed_mob_frames":e["n_cond"],
            "difference":e["est"],"se":e["se"],"mean_loss":m["loss_mean"]["est"],
            "relative_difference_pct":100*e["est"]/m["loss_mean"]["est"],
            "relative_approx95_interval_pct":[100*(e["est"]+k*1.96*e["se"])/m["loss_mean"]["est"] for k in (-1,1)]}
    (HERE/"readout.json").write_text(json.dumps(out,indent=2))
    print(json.dumps(out,indent=2))


if __name__=="__main__":main()
