"""CPU convex empirical D-loss minimization on conditional_stat's groups.

D(x; Y) = mean((x-Y)^2 + .1|x-Y|) + .01 mean_i max_j (x_j-Y_ij)^2.
Introduce radii r_i bounding each future's worst absolute coordinate error.
For fixed feasible radii, each coordinate minimizes the first term by clipping
its unconstrained scalar optimum to intersect_i [Y_ij-r_i, Y_ij+r_i].
The radii need at most five variables; feasibility is equivalent to
r_i+r_k >= max_j |Y_ij-Y_kj| for every pair. This is a convex reduction.
Two starts, explicit loss equality, primal feasibility and a tiny full-variable
epigraph comparison check the numerical solution. No model/GPU operations.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import LinearConstraint, minimize

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts/eda/oracle_v1"
MAPC = [c for c in range(63) if c != 31]
NAMES = ("zombie", "skeleton", "cow", "arrow")


def free_optimum(Y):
    n = len(Y)
    mu, ys = Y.mean(0), np.sort(Y, axis=0)
    x, assigned = np.empty_like(mu), np.zeros(len(mu), dtype=bool)
    for k in range(n + 1):
        cand = mu - .05 * (2 * k / n - 1)
        lo = ys[k - 1] if k else np.full_like(mu, -np.inf)
        hi = ys[k] if k < n else np.full_like(mu, np.inf)
        ok = (cand >= lo) & (cand <= hi)
        x[ok], assigned[ok] = cand[ok], True
    for k in range(n):
        left = 2 * (ys[k] - mu) + .1 * (2 * k / n - 1)
        right = 2 * (ys[k] - mu) + .1 * (2 * (k + 1) / n - 1)
        ok = (left <= 0) & (right >= 0)
        x[ok], assigned[ok] = ys[k, ok], True
    assert assigned.all()
    return x


def solve(Y, two_starts=True):
    Y = np.asarray(Y, dtype=np.float64)
    n, d = Y.shape
    x0 = free_optimum(Y)
    if np.max(np.ptp(Y, axis=0)) == 0:
        return x0, {"success": True, "loss": 0., "iterations": 0, "primal_violation": 0., "two_start_gap": 0.}
    pairs = [(i, k) for i in range(n) for k in range(i + 1, n)]
    A = np.zeros((len(pairs), n))
    lb = np.zeros(len(pairs))
    for q, (i, k) in enumerate(pairs):
        A[q, i] = A[q, k] = 1
        lb[q] = np.max(np.abs(Y[i] - Y[k]))
    constraints = LinearConstraint(A, lb, np.full(len(pairs), np.inf))

    def reconstruct(r):
        lo, hi = Y - r[:, None], Y + r[:, None]
        ilo, ihi = lo.argmax(0), hi.argmin(0)
        L, U = lo[ilo, np.arange(d)], hi[ihi, np.arange(d)]
        x = np.clip(x0, L, U)
        return x, ilo, ihi, L, U

    def fun(r):
        x, ilo, ihi, L, U = reconstruct(r)
        e = x - Y
        value = np.mean(e * e + .1 * np.abs(e)) + .01 * np.mean(r * r)
        gx = (2 * e + .1 * np.sign(e)).mean(0) / d
        grad = .02 * r / n
        below, above = x0 < L, x0 > U
        for i in range(n):
            grad[i] -= gx[below & (ilo == i)].sum()
            grad[i] += gx[above & (ihi == i)].sum()
        return value, grad

    start = np.abs(Y - x0).max(1)
    sol = [minimize(fun, start, jac=True, constraints=constraints, bounds=[(0, None)] * n,
                    method="SLSQP", options={"ftol": 1e-11, "maxiter": 400})]
    if two_starts:
        alt = np.full(n, np.max(np.ptp(Y, axis=0)))
        sol.append(minimize(fun, alt, jac=True, constraints=constraints, bounds=[(0, None)] * n,
                            method="SLSQP", options={"ftol": 1e-11, "maxiter": 400}))
    best = min(sol, key=lambda z: z.fun)
    x = reconstruct(best.x)[0]
    actual = np.abs(Y - x).max(1)
    explicit = np.mean((Y-x)**2 + .1*np.abs(Y-x)) + .01*np.mean(actual**2)
    violation = max(float(np.max(lb - A @ best.x)), float(np.max(actual - best.x)), 0.)
    gap = max(float(s.fun) for s in sol) - min(float(s.fun) for s in sol)
    valid = all(s.success for s in sol) and violation < 1e-6 and abs(explicit-best.fun) < 1e-7 and gap < 1e-6
    return x, {"success": bool(valid), "solver_success": [bool(s.success) for s in sol],
               "messages": [str(s.message) for s in sol], "loss": float(explicit),
               "iterations": [int(s.nit) for s in sol], "primal_violation": violation,
               "explicit_loss_gap": abs(float(explicit-best.fun)), "two_start_gap": gap}


def verify():
    rng = np.random.default_rng(20261012)
    maximum_gap = 0.
    for n in (2, 3, 5):
        for trial in range(4):
            Y = rng.normal(size=(n, 6))
            x, diagnostic = solve(Y)
            assert diagnostic["success"], diagnostic
            # Full epigraph has x, scalar absolute-loss slacks, worst radii.
            d = Y.shape[1]
            def loss(v):
                xx, r, u = v[:d], v[d:d+n], v[d+n:].reshape(n,d)
                return np.mean((xx-Y)**2+.1*u)+.01*np.mean(r*r)
            def con(v):
                xx, r, u = v[:d], v[d:d+n], v[d+n:].reshape(n,d)
                e=xx-Y
                return np.concatenate(((u-e).ravel(),(u+e).ravel(),(r[:,None]-e).ravel(),(r[:,None]+e).ravel()))
            v0=np.concatenate((Y.mean(0),np.abs(Y-Y.mean(0)).max(1),np.abs(Y-Y.mean(0)).ravel()))
            rr=minimize(loss,v0,constraints={"type":"ineq","fun":con},method="SLSQP",
                        options={"ftol":1e-10,"maxiter":1000})
            assert rr.success,rr.message
            gap=abs(rr.fun-diagnostic["loss"])
            maximum_gap=max(maximum_gap,gap)
            assert gap<2e-7,(n,trial,gap)
    return {"tiny_full_epigraph_checks":12,"maximum_loss_gap":maximum_gap}


def main():
    torch.set_num_threads(4)
    checks=verify()
    selection=json.loads((OUT/"conditional_stat.json").read_text())
    saved=torch.load(OUT/"snap_bank_readers.pt",weights_only=False)
    meta=torch.load(ROOT/"artifacts/eda/diagnosis_rollouts_v1/meta.pt",weights_only=False)
    R=len(meta["seed"])
    fut=np.memmap(ROOT/"artifacts/eda/stochdiag_v1/fut5.f16",mode="r",dtype=np.float16,shape=(R,5,16,81,192))
    h=saved["heads"]["mob"]
    threshold=torch.tensor(saved["thresholds"])
    rows=torch.zeros(R,4,4,dtype=torch.float64)
    diagnostics=[]
    for number,g in enumerate(selection["groups"]):
        r,ss=g["root"],g["members"]
        Y=np.array(fut[r,ss,0]).astype(np.float64)
        x,diagnostic=solve(Y.reshape(len(ss),-1))
        x=x.reshape(81,192)
        diagnostics.append({"root":r,"members":ss,**diagnostic})
        if not diagnostic["success"]:
            print(json.dumps({"failure":number,"diagnostic":diagnostic}),flush=True)
            continue
        tok=torch.from_numpy(x[MAPC])
        z=(tok-h["mu"].double())/h["sd"].double()
        z=torch.cat((z,torch.ones(len(z),1,dtype=z.dtype)),1)
        dec=(z@h["w"].double())>threshold
        vis=meta["future_visible"][r,ss[0],0].float()
        fl=vis[1071:1512].reshape(63,7)>0
        tr=torch.stack((fl[:,0],fl[:,2],fl[:,1],fl[:,3:].any(-1)),-1)[MAPC]
        n=len(ss)
        rows[r,:,0]+=tr.sum(0)*n
        rows[r,:,1]+=dec.sum(0)*n
        rows[r,:,2]+=(tr&dec).sum(0)*n
        rows[r,:,3]+=(~tr&dec).sum(0)*n
        if (number+1)%25==0:print(json.dumps({"groups_done":number+1,"groups_total":len(selection["groups"])}),flush=True)
    tot=rows.sum(0)
    metrics={name:{"true":int(tot[j,0]),"predicted":int(tot[j,1]),"true_positive":int(tot[j,2]),
                   "false_positive":int(tot[j,3]),"census_ratio":float(tot[j,1]/tot[j,0].clamp_min(1)),
                   "recall":float(tot[j,2]/tot[j,0].clamp_min(1)),"fp_per_true":float(tot[j,3]/tot[j,0].clamp_min(1))}
             for j,name in enumerate(NAMES)}
    report={"schema":"oracle_conditional_dopt_v1","checks":checks,"groups":diagnostics,
            "successful_groups":sum(d["success"] for d in diagnostics),"total_groups":len(diagnostics),
            "metrics":metrics,"script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "interpretation_scope":"Finite-sample empirical D optimum conditional on identical full next-mob layout, one-step only."}
    (OUT/"conditional_dopt.json").write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!="groups"},indent=2),flush=True)


if __name__=="__main__":main()
