"""Matched corrected fork; world loss copied from the previous MultiStep trainer."""
import numpy as np
import torch
import torch.nn.functional as F
from saccade import model as M,train as TR,data as D

HORIZONS = (2, 4, 8, 16)
LAMBDA_MULTI = 0.1


def multistep_losses(model: M.M0, chunk: dict, carry, cache, targets, cfg, bootstrap=None, update=0, gen=None, mode="endpoint"):
    """Teacher pass over the chunk + blind student multi-step rollout from sampled prefix t over horizon H."""
    dev = next(model.parameters()).device
    P = cfg["data"]["context_prefix"]
    t = TR.tensors(chunk, dev)
    first, fix = t["first"], t["fixation"]
    L = first.shape[1]
    T_chunk = L - P

    # Sample horizon H and starting frame t
    # Horizons from {2, 4, 8, 16}
    h_idx = int(torch.randint(0, len(HORIZONS), (), generator=gen))
    H = HORIZONS[h_idx]
    assert T_chunk > H + 1, f"Chunk length {T_chunk} too short for horizon {H}"
    start_t = int(torch.randint(0, T_chunk - H, (), generator=gen))

    with TR.autocast(dev, cfg):
        Pc = model.perception(t["frames"][:, P:], fix[:, P:])
        with torch.no_grad():
            Pp = model.perception(t["frames"][:, :P], fix[:, :P])
        emb = model.belief.percept_embed(Pc)
        x = model.belief.tokens(emb, t["reward"][:, P:], t["terminated"][:, P:], t["action"][:, P:])
        seq = first[:, P:].long().cumsum(1).repeat_interleave(2, 1)

        # Split forward pass at start_t immediately after percept t (split_idx = 2 * start_t + 1)
        split_idx = 2 * start_t + 1
        y_hist, carry_t = model.belief(x[:, :split_idx], carry, seq[:, :split_idx])
        suffix = seq[:, split_idx:] - seq[:, split_idx-1:split_idx]
        y_rest, c_new = model.belief(x[:, split_idx:], carry_t, suffix)
        y = torch.cat([y_hist, y_rest], dim=1)

        # Student rollout from carry_t (un-detached: gradients flow into history!)
        # Process action t first, then interleave mask_token and actions up to t + H
        act_span = t["action"][:, P + start_t : P + start_t + H]          # [B, H]
        student_tokens = torch.stack([model.belief.action(act_span),
                                      model.belief.mask_token.expand(*act_span.shape,-1)],dim=2).flatten(1,2)            # [B, 2*H, W]
        y_student, _ = model.belief(student_tokens, carry_t)
        # Both arms run identical scans and compute both diagnostics. Only the time reduction affects gradients.
        y_obs = y[:, 2*(start_t+1):2*(start_t+H+1):2].detach()
        err = (y_student[:, 1::2].float()-y_obs.float()).square().mean(-1)
        assert err.shape == (len(act_span), H)
        valid = (~first[:, P+start_t:P+start_t+H+1].any(1)
                 & t["valid"][:, P+start_t:P+start_t+H+1].all(1))
        endpoint = err[valid,-1].mean() if valid.any() else err.sum()*0
        dense = err[valid].mean() if valid.any() else err.sum()*0
        multi_loss = endpoint if mode=="endpoint" else dense

        # Standard F4 / HUD-change losses (G1, G2, G3)
        nxt = y[:, 1::2]                                                 # predicted belief of frame t + 1
        b_pred = torch.cat([cache.to(nxt.dtype), nxt[:, :-1]], 1) * (~first)[..., None]
        new_cache = torch.cat([b_pred[:, -P:], nxt[:, -1:]], 1).detach().float()
        x0 = tuple(torch.cat([a, b.detach()], 1) for a, b in zip(Pp.parts(), Pc.parts()))
        sc = cfg["edm"]["shortcut"]
        g1 = M.shortcut_loss(model.denoiser, x0, t["prev_action"], model.belief.memory(b_pred), fix, t["episode"],
                             t["valid"], sc, torch.as_tensor(bootstrap, device=dev), update >= sc["bootstrap_start"])[:, P:].mean()
        bp, cells = b_pred[:, P:], M.window(fix[:, P:])
        pix = M.cell_pixels(M.crop(t["frames"][:, P:], fix[:, P:]))
        assert cfg["heads"]["pixel_decoder"]["mean_loss"] == "faithful"
        vs = Pc.v.detach()
        mt, st = model.pixels(model.tsp.predict(bp, Pc.p, Pc.v, cells), vs, cells)
        mr, sr = model.pixels(model.tsp.reconstruct(Pc.g, Pc.v, cells), vs, cells)
        y_pix = pix.float() / 127.5 - 1
        tsp, rec = (mt.float() - y_pix).square().mean(), (mr.float() - y_pix).square().mean()
        noise = M.pixel_nll(mt.detach(), st, pix).mean() + M.pixel_nll(mr.detach(), sr, pix).mean()
        logit = model.change(torch.cat([Pp.p, Pc.p], 1)[:, P - 1:L - 1], Pc.p, bp)
        idx = chunk["target"][:, P:]
        tgt = np.full((*idx.shape, D.GRID, D.GRID, len(D.CHANGE_MODES)), np.nan, np.float32)
        tgt[idx >= 0] = targets[idx[idx >= 0]] if targets is not None else np.nan
        tgt = torch.as_tensor(tgt, device=dev)
        ok = torch.isfinite(tgt)
        change = F.binary_cross_entropy_with_logits(logit.float()[ok], tgt[ok]) if ok.any() else logit.sum() * 0
        nf = ~first[:, P:]
        h = model.outcome.features(bp, emb)[nf]
        r = t["reward"][:, P:][nf]
        reward = -model.outcome.reward_logp(h, r).mean()
        cont = F.binary_cross_entropy_with_logits(model.outcome.cont_logit(h, r).float(),
                                                  (~t["terminated"][:, P:][nf]).float())

    terms = {"g1": g1, "change": change, "tsp": tsp, "rec": rec, "reward": reward, "cont": cont,
             "noise": noise, "multi": multi_loss, "endpoint_diag":endpoint.detach(),
             "dense_diag":dense.detach(), "valid_rollout":valid.float().mean(),
             "horizon":g1.new_tensor(H)}
    return {k: v.float() for k, v in terms.items()}, c_new, new_cache

