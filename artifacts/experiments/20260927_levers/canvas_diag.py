"""Post-hoc fcanvas diagnosis on TEST-seed factual futures.

Quantifies how often the canvas's token-based view-scroll estimate disagrees
with the same estimator on real encoded frames, where one-step validation
against simulator movement was 99.35% accurate. This is an approximate
reference, not access to hidden player coordinates. Also tests whether the
masked Mamba scan holds the *whole* temporal mixer state through absent frames.
"""
import hashlib
import json
from pathlib import Path

import torch

import teval as T
import tworld as TW
from scroll import estimate

HERE = Path(__file__).parent
WORLD = Path("artifacts/eda/levers_tworlds_v1/corrt_raw_suffix_s7_fcanvas.pt")


def scroll_diagnosis():
    from d4mj.config import config_from_dict
    import spatial as S

    device = torch.device("cuda")
    meta, train_roots, _ = T.split()
    ix = torch.where(~train_roots)[0]
    cache = T.build_cache("raw", device)
    world, payload = T.load_world(WORLD, device)
    assert payload["script_sha256"] == hashlib.sha256(Path(TW.__file__).read_bytes()).hexdigest()
    config = config_from_dict(torch.load(S.CHECKPOINT, map_location="cpu", weights_only=False)["config"])
    true_frames = torch.cat([cache["ctx"][ix, -1:], cache["fut"][ix]], 1).float()
    true_shift = estimate(true_frames[:, :-1], true_frames[:, 1:])
    predicted = torch.empty_like(true_shift)
    for i in range(0, len(ix), 8):
        sl = ix[i:i + 8]
        ctx = cache["ctx"][sl].float()
        ca, fa = cache["ctx_a"][sl], cache["fut_a"][sl]
        frames = [ctx[:, j] for j in range(4)]
        actions = [ca[:, j] for j in range(3)]
        for k in range(T.H):
            window = 4 if k == 0 else 5
            a = torch.stack(actions[-(window - 1):] + [fa[:, k]], 1)
            nxt = T.step(world, torch.stack(frames[-window:], 1), a, device, config)
            predicted[i:i + len(sl), k] = estimate(frames[-1], nxt)
            frames.append(nxt)
            actions.append(fa[:, k])
    alive = ~meta["future_dead"][ix, 0].cumsum(1).bool()
    out = {"checkpoint": str(WORLD),
           "checkpoint_sha256": hashlib.sha256(WORLD.read_bytes()).hexdigest(),
           "model_source_sha256": payload["script_sha256"],
           "roots": len(ix), "walk_seeds": int(meta["seed"][ix].unique().numel()),
           "true_frame_estimator_one_step_accuracy_against_simulator": 0.9935423135757446,
           "by_depth": {}}
    for lo, hi in ((0, 1), (1, 4), (4, 8), (8, 16), (0, 16)):
        m = alive[:, lo:hi]
        truth, pred = true_shift[:, lo:hi], predicted[:, lo:hi]
        no_scroll, scroll = (truth == 0) & m, (truth != 0) & m
        out["by_depth"][f"{lo+1}-{hi}"] = {
            "n": int(m.sum()), "true_scroll_rate": float(scroll.sum() / m.sum()),
            "agreement": float((pred[m] == truth[m]).float().mean()),
            "false_scroll_rate": float((pred[no_scroll] != 0).float().mean()),
            "missed_scroll_rate": float((pred[scroll] == 0).float().mean()),
            "wrong_direction_rate_on_scroll": float(((pred[scroll] != truth[scroll]) & (pred[scroll] != 0)).float().mean()),
        }
    return out


@torch.no_grad()
def convolution_hold_check():
    from d4mj.config import config_from_dict
    import spatial as S

    cfg = config_from_dict(torch.load(S.CHECKPOINT, map_location="cpu", weights_only=False)["config"]).dynamics
    from dataclasses import replace
    from d4mj.mamba_recurrence import FunctionalMamba2

    device = torch.device("cuda")
    torch.manual_seed(17)
    mixer = FunctionalMamba2(replace(cfg, chunk_size=64)).to(device).eval()
    x = torch.randn(8, 6, 256, device=device)
    patterns = torch.tensor([[1, 0, 0, 1, 1, 1], [1, 1, 0, 1, 0, 1],
                             [1, 0, 1, 0, 1, 1], [1, 1, 1, 1, 1, 1]], dtype=torch.bool, device=device)
    keep = patterns.repeat(2, 1)
    masked = TW.masked_scan(mixer, x * keep[..., None], keep)
    diffs, refs = [], []
    for i in range(len(x)):
        reference = mixer.scan(x[i:i+1, keep[i]])[0][0]
        differences = masked[i, keep[i]] - reference
        diffs.append(differences.flatten())
        refs.append(reference.flatten())
    diff, ref = torch.cat(diffs), torch.cat(refs)
    return {"maxabs": float(diff.abs().max()), "rmse": float(diff.square().mean().sqrt()),
            "reference_rms": float(ref.square().mean().sqrt()),
            "relative_rmse": float(diff.square().mean().sqrt() / ref.square().mean().sqrt()),
            "all_kept_maxabs": float((masked[3, keep[3]] - mixer.scan(x[3:4])[0][0]).abs().max())}


def main():
    out = {"scroll": scroll_diagnosis(), "convolution_hold": convolution_hold_check()}
    (HERE / "canvas_diag.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2), flush=True)


if __name__ == "__main__":
    main()
