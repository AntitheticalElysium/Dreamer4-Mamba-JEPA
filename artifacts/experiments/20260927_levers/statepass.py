"""E6c. Buitrago Ruiz & Gu 2025 (arXiv 2507.02782, read in the PDF 2026-09-28): recurrent models fail to length-
generalize because training visits only the states reachable within the training context ("unexplored states
hypothesis"); post-training ~100 steps with State Passing or TBTT fixes Mamba-2 from 2k to 128k without hurting
in-context performance. Our E1: the L4 world collapses past its 4-frame window (full history 1.7-11x copy while its
3-frame window reads 0.52-0.72). Does their fix give (a)'s short-context accuracy AND a usable recurrence?

Start: E1e's L4b512 world (10,000 updates, 512 windows of 4 frames). Post-training, every arm identical except the
initial recurrent state: 500 updates, 512 windows of 4 frames per update (sampler seed 13), fresh AdamW with the joint
settings, peak lr = joint lr / 10 = 5e-6 ("a peak learning rate that is ten times smaller"), 50-update linear
warmup then constant (their schedule shape is not stated), next-latent MSE, clip 1. Evaluated at 100 and 500 updates
exactly as E1 (`context_length.evaluate`).
  pt0    zero initial state: the post-training alone (control)
  sp     State Passing (their s4.4): each layer's SSM state initialized to a final SSM state of the PREVIOUS batch
         (random windows, so the pairing is random), zeroed per window with p = 0.1; convolution state zero (their
         pseudocode, App. I, covers the SSM state only); no gradient through it
  tbtt   TBTT (their s4.5): 512 segments of 49 frames, consumed as 16 consecutive 4-frame chunks (frames 3j..3j+3)
         over 16 updates, each chunk starting from the previous chunk's full final state (conv + SSM), detached;
         fresh segments every 16 updates
"""
import json
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import context_length as C  # noqa: E402

OUT = HERE / "statepass.json"
POST, EVAL_AT, CHUNKS = 500, (100, 500), 16


def post_train(arm, world, config, episodes, device, log):
    from d4mj.mamba_recurrence import MambaCarry
    from d4mj.train import autocast_context, optimizer, optimizer_step
    opt = optimizer([world], config.joint, exclude_vectors=True)
    params = [p for g in opt.param_groups for p in g["params"]]
    train = [e for e in episodes if e.split == "train"]
    windows = C.Windows(train, 4, seed=13)
    segments = C.Windows(train, 3 * CHUNKS + 1, seed=13)
    gen = torch.Generator(device=device).manual_seed(17)
    carry, segment, snapshots = None, None, {}
    for update in range(POST):
        if arm == "tbtt":
            j = update % CHUNKS
            if j == 0:
                segment, carry = segments.sample(512), None
            z, a = segment[0][:, 3 * j:3 * j + 4], segment[1][:, 3 * j:3 * j + 3]
        else:
            z, a = windows.sample(512)
        z, a = z.to(device), a.to(device)
        memory = None
        if carry is not None and arm == "sp":
            keep = (torch.rand(z.shape[0], device=device, generator=gen) > 0.1).float()[:, None, None, None]
            memory = tuple(MambaCarry(torch.zeros_like(c.conv), c.ssm * keep) for c in carry)
        elif carry is not None and arm == "tbtt":
            memory = carry
        with autocast_context(config):
            pred, _, final = world.scan_pairs(z[:, :-1], a, memory)
        loss = (pred.float() - z[:, 1:].float()).square().mean()
        lr = config.joint.learning_rate / 10 * min(1.0, (update + 1) / 50)
        norm = optimizer_step(opt, loss, params, learning_rate=lr, grad_clip=config.joint.grad_clip, strict=True,
                              zero_grad=True)
        carry = tuple(MambaCarry(c.conv.detach(), c.ssm.detach()) for c in final) if arm != "pt0" else None
        if (update + 1) % 100 == 0:
            log(stage="train", arm=arm, update=update + 1, loss=float(loss), gradient_norm=float(norm))
        if update + 1 in EVAL_AT:
            snapshots[update + 1] = {k: v.detach().clone().cpu() for k, v in world.state_dict().items()}
    return snapshots


def main():
    device = torch.device("cuda")
    started = time.time()
    log = lambda **kw: print(json.dumps({**kw, "seconds": round(time.time() - started, 1)}), flush=True)
    config, episodes = C.load(device)
    result = json.loads(OUT.read_text()) if OUT.exists() else {}
    base = torch.load(C.OUT / "L4b512.pt", map_location="cpu", weights_only=False)["world"]
    for arm in ("sp", "pt0", "tbtt"):
        if all(f"{arm}@{n}" in result for n in EVAL_AT):
            continue
        world = C.new_world(config, device)
        world.load_state_dict(base)
        world.train()
        snapshots = post_train(arm, world, config, episodes, device, log)
        for n, state in snapshots.items():
            world.load_state_dict(state)
            world.eval()
            torch.save({"world": state, "arm": arm, "post_updates": n, "parent": "L4b512"}, C.OUT / f"L4b512_{arm}{n}.pt")
            result[f"{arm}@{n}"] = C.evaluate(world, config, episodes, device)
            log(stage="eval", arm=f"{arm}@{n}", **{k: v for k, v in result[f"{arm}@{n}"].items() if k == "position"})
            OUT.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
