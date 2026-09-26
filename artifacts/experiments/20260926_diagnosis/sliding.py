"""D24. Would LeWM's own deployment rule -- condition on the last `history_size` frames only -- repair canonical H2?

le-wm jepa.rollout: `emb_trunc = emb[:, -HS:]` with HS = history_size = 3 at every rollout step, i.e. the
predictor never sees more context than it was trained on (config history_size 3 + num_preds 1 = 4 frames).
Our Mamba worlds instead carry an unbounded recurrent state. On the diagnostic futures (1,002 roots, the
policy's 16-step factual trajectory, eval-mode raw z), for the canonical JOINT world (pre-bridge) and the
BRIDGE world:
  recurrent  teacher on the 4 context frames, then 16 `advance` steps (the current protocol)
  sliding_k  at every step: a fresh teacher from a zero state over the last k latents (true context first,
             then imagined) and their k-1 actions, one advance. k = 3 is LeWM's truncation (HS = 3: predict
             the 4th frame, the last position the 4-frame joint windows train); k = 4 is our evaluator's context
             (predicts position 4, one past the trained range)
Teacher-forced one-step error under both rules too. Errors / V (the true states' variance), as D11.
"""
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).parent
ROOT = Path("/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(HERE))
import bnmode  # noqa: E402
from rollouts import load_roots  # noqa: E402

H = 16


@torch.no_grad()
def main():
    device = torch.device("cuda")
    worlds = {"joint": bnmode.load(ROOT / "artifacts/lewm_m4_canonical/raw/joint/step-010000.pt", "joint"),
              "bridge": bnmode.load(ROOT / "artifacts/lewm_m4_canonical/raw/bridge/step-002000.pt", "bridge")}
    enc = worlds["joint"].encoder.to(device).eval()
    roots = load_roots()
    meta = torch.load(ROOT / "artifacts/eda/diagnosis_rollouts_v1/meta.pt")
    alive = ~meta["future_dead"][:, 0].cumsum(1).bool()
    enc_z = lambda f: torch.cat([enc(f[i:i + 64][:, None].to(device)).float()[:, 0, 0] for i in range(0, len(f), 64)])
    ctx = enc_z(torch.stack([r["context"] for r in roots]).flatten(0, 1)).view(len(roots), 4, -1)
    true = enc_z(torch.stack([r["future_frames"][0] for r in roots]).flatten(0, 1)).view(len(roots), H, -1)
    ctx_a = torch.stack([r["context_actions"] for r in roots]).to(device)
    fut_a = torch.stack([r["future_actions"] for r in roots]).to(device)
    V = float(torch.cov(true.flatten(0, 1).T.double()).trace())
    result = {}
    for name, b in worlds.items():
        w = b.world.to(device).eval()
        out = {k: [] for k in ("recurrent_gen", "recurrent_tf", "sliding3_gen", "sliding3_tf", "sliding4_gen", "sliding4_tf")}
        for i in range(0, len(roots), 64):
            c, t, ca, fa = ctx[i:i + 64], true[i:i + 64], ctx_a[i:i + 64], fut_a[i:i + 64]
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                st = w.teacher(c[:, :, None], ca).state
                gen = []
                for k in range(H):
                    st, _ = w.advance(st, fa[:, k:k + 1])
                    gen.append(st.latent[:, 0, 0].float())
                out["recurrent_gen"].append(torch.stack(gen, 1))
                seq = torch.cat([c, t], 1)[:, :, None]
                out["recurrent_tf"].append(w.teacher(seq, torch.cat([ca, fa], 1)).predicted[:, 3:3 + H, 0].float())
                for hs in (3, 4):
                    lat, real = [c[:, j] for j in range(4)], [c[:, j] for j in range(4)]
                    acts, gen, tf = [ca[:, j] for j in range(3)], [], []
                    for k in range(H):
                        a = torch.stack(acts[-(hs - 1):] + [fa[:, k]], 1)
                        s = w.teacher(torch.stack(lat[-hs:], 1)[:, :, None], a[:, :hs - 1]).state
                        nxt, _ = w.advance(s, a[:, hs - 1:hs])
                        s2 = w.teacher(torch.stack(real[-hs:], 1)[:, :, None], a[:, :hs - 1]).state
                        nt, _ = w.advance(s2, a[:, hs - 1:hs])
                        g = nxt.latent[:, 0, 0].float()
                        gen.append(g); tf.append(nt.latent[:, 0, 0].float())
                        lat.append(g.to(c.dtype)); real.append(t[:, k]); acts.append(fa[:, k])
                    out[f"sliding{hs}_gen"].append(torch.stack(gen, 1)); out[f"sliding{hs}_tf"].append(torch.stack(tf, 1))
        rows = {}
        for k, v in out.items():
            e = ((torch.cat(v).cpu() - true.cpu()) ** 2).sum(-1)
            rows[k] = [float(e[alive[:, d], d].mean() / V) for d in range(H)]
        rows["persist"] = [float(((ctx[:, -1].cpu() - true[:, d].cpu()) ** 2).sum(-1)[alive[:, d]].mean() / V) for d in range(H)]
        result[name] = rows
        pick = lambda xs: [round(xs[d], 3) for d in (0, 1, 3, 7, 15)]
        print(name, json.dumps({k: pick(v) for k, v in rows.items()}), flush=True)
    (HERE / "sliding.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
