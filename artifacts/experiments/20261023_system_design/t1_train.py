"""T1 training (T1_PROTOCOL.md). Usage: t1_train.py --data DIR --out DIR --arch {G,S,A,O,E,Estar} --seed S [--steps 6000]
Windows of 65 frames inside one episode: frames 0..63 are inputs, frames 1..64 are the targets (next symbolic frame)."""
import argparse, json, os, sys, time
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from t1_models import make, NB, NM, NV

T = 64


def load(path):
    z = np.load(path)
    return {k: z[k] for k in z.files}


def windows(d, stride=None):
    out = []
    for s, L in zip(d["starts"], d["lengths"]):
        if L < T + 1:
            continue
        if stride is None:
            out.append(np.arange(s, s + L - T))
        else:
            out.append(np.arange(s, s + L - T, stride))
    return np.concatenate(out)


def true_shift(pos):                                          # pos [B, T+1, 2] -> index of the move into frame t
    d = np.zeros(pos.shape[:2] + (2,), np.int64)
    d[:, 1:] = np.clip(pos[:, 1:].astype(np.int64) - pos[:, :-1].astype(np.int64), -1, 1)
    return (d[..., 0] + 1) * 3 + (d[..., 1] + 1)


def batch_at(d, starts, device):
    idx = starts[:, None] + np.arange(T + 1)[None]
    g = lambda k: torch.from_numpy(d[k][idx]).to(device)
    view = g("view").reshape(len(starts), T + 1, 63); mob = g("mob").reshape(len(starts), T + 1, 63)
    vit, inv = g("vit"), g("inv"); sleep, light, act = g("sleep"), g("light"), g("action")
    inp = (view[:, :T], mob[:, :T], vit[:, :T], inv[:, :T], sleep[:, :T], light[:, :T], act[:, :T])
    tgt = (view[:, 1:], mob[:, 1:], vit[:, 1:], inv[:, 1:].clamp(0, 9))
    sh = torch.from_numpy(true_shift(d["pos"][idx])[:, :T]).to(device)
    return inp, tgt, sh


def losses(out, tgt):
    lb, lm, lv, li = out
    vb, vm, vv, vi = tgt
    return (F.cross_entropy(lb.float().reshape(-1, NB), vb.reshape(-1).long()),
            F.cross_entropy(lm.float().reshape(-1, NM), vm.reshape(-1).long()),
            F.cross_entropy(lv.float().reshape(-1, NV), vv.reshape(-1).long().clamp(0, 9)),
            F.cross_entropy(li.float().reshape(-1, NV), vi.reshape(-1).long()))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True); p.add_argument("--out", required=True)
    p.add_argument("--arch", required=True); p.add_argument("--seed", type=int, required=True)
    p.add_argument("--steps", type=int, default=6000); p.add_argument("--batch", type=int, default=16)
    a = p.parse_args(); os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed + 100)
    d = load(os.path.join(a.data, "train_sym.npz")); W = windows(d)
    model = make(a.arch).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 300))
    log = open(os.path.join(a.out, "train_log.jsonl"), "a"); t0 = time.time()
    n_par = sum(x.numel() for x in model.parameters())
    for step in range(1, a.steps + 1):
        inp, tgt, sh = batch_at(d, rng.choice(W, a.batch), "cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(inp, true_shift=sh if a.arch == "Estar" else None)
        lb, lm, lv, li = losses(out, tgt)
        loss = lb + lm + lv + li
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
        if step % 100 == 0:
            rec = {"step": step, "loss": loss.item(), "block": lb.item(), "mob": lm.item(), "vit": lv.item(), "inv": li.item(),
                   "sec": round(time.time() - t0, 1)}
            log.write(json.dumps(rec) + "\n"); log.flush()
            if step % 1000 == 0:
                print(rec, flush=True)
    torch.save({"model": model.state_dict(), "args": vars(a), "params": n_par}, os.path.join(a.out, "final.pt"))
    json.dump({"done": True, "seconds": time.time() - t0, "params": n_par}, open(os.path.join(a.out, "done.json"), "w"))


if __name__ == "__main__":
    main()
