"""E14a. Where and why: can the OUTPUT HEAD alone learn interaction consequences from the frozen backbone, and what does the gradient
look like? (E13: every world copies DO / place consequences even on training windows, yet its backbone state encodes them, linear
change AUC 0.92-0.95. Kang et al., ICLR 2020, "Decoupling representation and classifier": representations learned under natural
sampling are good; the classifier is biased to frequent classes; re-training only the classifier with re-balanced sampling (cRT)
recovers the tail. An independent review asked for exactly this same-head comparison before any end-to-end intervention.)

Data: spatial_pool_v1 (the worlds' training windows). Labels precomputed once (CPU, teval's ridge probes):
  faced     the faced cell per transition (facing probe on the player token of frame t)
  attempt   action t is DO or place (5, 7-10) and frame t+1 alive: the faced tile of ANY attempt, successful or not
  cons      strict consequence: attempt AND tile class before != after AND squared token change > 120 (consfit's definition)
  copyres   per token, min over {self, up, down, left, right} of the squared distance from the next token to that candidate in
            frame t (zero pad off-grid): a camera-compensated change, defined from data only, not from Craftax semantics
  same      per token, the squared change at the same screen position
Part G -- gradient anatomy at the TRAINED checkpoint (20 seeded training batches of 40 windows, teacher-forced, all 5 positions).
  For the world's training objective (L1, or CE for the categorical world) and for L2: per-token losses split into consequence
  tokens and all others, each divided by the total token count (their contribution to the mean). Gradients w.r.t. head params
  (corrt also: proj / choose / move gate separately) and backbone params. Reported per batch then averaged: ||g_cons||, ||g_rest||,
  ||g_total||, cos(g_cons, g_rest), projection share <g_cons, g_total> / ||g_total||^2; and over all 20 batches summed (full-batch
  view). Every world is probed with the teacher-forced objective (suffix worlds also trained a 2-step suffix term). Corr heads
  also: the mean mixture weights (self, up, down, left, right, generate) at consequence tokens and at all others (a saturated
  "self" weight scales the generate candidate's gradient by its near-zero weight).
Part H -- head-only re-training (cRT). Backbone frozen. Head re-initialized with the module's own init (seed 0), except
  `continue` (the trained head). 3,000 updates, batches of 40 seeded windows (identical across arms), AdamW lr 1e-3, wd 0.01,
  100 warmup, clip 1. Arms (weights act per token on the per-token loss; "unit mean" = divided by the batch-mean weight):
    continue      trained head, uniform training loss (does more head training alone learn it?)
    uniform       re-init, uniform training loss (cRT with natural sampling: does a fresh head copy too?)
    l2            re-init, uniform mean squared error (continuous heads only)
    eawm_same     re-init, EAWM eq. 8/12 event-aware weights (ICLR 2026): events = same > 25, frames with >= 50% events are
                  boundaries (uniform); non-event tokens weight 1 - w, w = 0.5 (EAWM's value); unit mean
    eawm_copy     the same with events = copyres > 25 (camera-compensated), w = 0.9; unit mean
    eawm_copy99   events = copyres > 25, w = 0.99; unit mean
    mask10        CGSReg form (2607.15142 eq. 3): uniform mean + lambda * mean over the faced tiles of all attempts, lambda = 10
    resample      the literal cRT (Kang et al.): uniform loss, half of every batch drawn from windows holding >= 1 strict
                  consequence (re-balanced sampling instead of re-weighting; uses the Craftax labels, diagnostic only)
  The categorical world uses CE in place of L1 and has no l2 arm.
Evaluation (train subset: 4,096 seeded training windows as consfit; held: tworld's 2,048 held-out windows): strict consequence
  caught (predicted faced-tile class == true next class), copied, hallucinated (predicted class != the unchanged class, on
  attempts whose probe class did not change),
  mean per-token L1 on held over all tokens and over non-attempt tokens, and the HUD on consequence transitions (L1 of predicted vs
  true HUD tokens / L1 of copying the HUD: < 1 means the inventory change is predicted).
Readings, declared before running (per world):
  G_starved     under the training objective, ||g_cons|| / ||g_total|| < 0.05 for the head (batch average)
  G_cancel      cos(g_cons, g_rest) < -0.2 for the head (batch average)
  G_l1_sign     (||g_cons|| / ||g_rest|| under L2) / (the same under L1) >= 5 (continuous heads)
  H_uniform_copies   uniform: held caught < 0.1
  H_head_fixable     some arm reaches held caught >= 0.5 with all-token held L1 no more than 10% above `uniform` and held
                     hallucinated no more than 0.05 above `uniform` (a noisy head catches by chance: the 20-update smoke
                     test's random categorical head caught 0.12 at 0.71 hallucinated)
  H_geometry         l2 reaches held caught >= 0.5 while uniform < 0.1
  H_generic          an arm without Craftax semantics (l2, eawm_same, eawm_copy, eawm_copy99) reaches held caught >= 0.5
  H_converged        uniform's held all-token L1 <= 1.05 x the trained world's (else "copies" may mean "under-trained")
  If no arm reaches 0.5: not fixable at the head with these doses -> the fix must reach the representation (end-to-end).
Addendum (declared 2026-10-02 after the first world's arms, before any addendum arm ran; `--arms` adds arms to an existing
result and keeps its anatomy):
    clip50 / clip75 / clip90   SimPLe's clipped loss (Kaiser et al., ICLR 2020, sec. 4; tensor2tensor modalities.video_l1_internal_
                  loss: relu(|pred - target| - cutoff), per element): every element whose error is inside the cutoff gives no
                  gradient, so well-predicted background stops competing. Cutoff = the 50 / 75 / 90% quantile of the TRAINED
                  world's per-element |error| on 20 seeded training batches (no Craftax semantics). Categorical: per-token CE,
                  cutoffs = quantiles of its per-token CE
    ce_clip03     (categorical) SimPLe's softmax value: relu(CE - 0.03) per token ("no gradient once confidence exceeds 97%")
    mask0.1 / mask1 / mask3   the dose response of mask10 (per-token dose on the faced tile of an attempt: x31 / x304 / x911;
                  mask10 = x3,034 caught 0.58 at +59% all-token L1 on teacher s7; EAWM's x1.7-x8.8 and resample's x3.7 changed
                  nothing)
    mlp_uniform / mlp_mask10 / mlp_mask1   capacity or representation? (both seeds: a fresh LINEAR head re-learns the trained
                  head's catch rate exactly, and mask10 catches 0.57-0.58 only at +38-59% all-token L1). The head's linear
                  maps (proj, choose; logits for categorical) become 2-layer MLPs on the same frozen h (hidden 512, GELU; last
                  layer initialized as the linear module: choose zero weight, bias (3,0,0,0,0,0)); frame / target_gate unchanged
  H_clip          a clip arm passes H_head_fixable's three conditions
  mask_dose       the smallest mask dose reaching held caught >= 0.5, and its all-token L1 / uniform (reported)
  H_capacity      mlp_mask10 reaches held caught >= 0.5 with all-token L1 <= 1.1 x mlp_uniform's: the linear readout is the
                  bottleneck (an expressive head is a fix); otherwise (caught but L1 > 1.1x) H_representation: the frozen
                  representation cannot carry both, the fix must reach the backbone (end-to-end)
Addendum 2 (declared 2026-10-02 after check_allprobe, before any skip arm ran; `--tag skip` writes headfit_<world>_skip.json):
  check_allprobe: among ALL tokens at the natural rate, h singles out consequence tokens with AP 0.16 / 0.34 (s7 / s8, linear;
  MLP 0.28 / 0.44), while an MLP on the RAW local input (token, 4 neighbours, action) reaches AP 0.667, recall 0.917 at precision
  0.5: the backbone dilutes a conjunction the input carries. Substitution: the head reads it directly.
    skip_uniform / skip_mask1 / skip_mask10   proj and choose (logits for categorical) become 2-layer MLPs (512, GELU) on
                  [h, the token, its 4 grid neighbours (zero off-grid), the action one-hot] of the same forward pass (no Craftax
                  semantics: a locality prior, as Delta-IRIS's decoder conditioned on the previous frame and action)
  H_local_uniform  skip_uniform reaches held caught >= 0.5 with all-token L1 <= 1.1 x uniform's: the diluted representation
                   was the cause; a local-input readout learns the consequences with no re-weighting
  H_local_dose     skip_mask1 or skip_mask10 passes H_head_fixable's three conditions: with local input the dose no longer costs
Addendum 3 (declared 2026-10-02 after check_toperr, before any hard arm ran; `--tag hard`): the dose from the model's OWN
error, no Craftax labels. check_toperr: consequence tokens sit in the per-token error tail (median rank 0.32% / 0.53% of tokens,
s7 / s8 18k; top-1% recall 0.998 / 0.72), the top-1% set being 52% entering cells, 25% static, 12% HUD, 7% consequences.
    hardQxL       uniform mean + L x the mean over the top-Q% tokens of the batch by their current (detached) per-token loss
                  (online hard example mining, Shrivastava et al. 2016): dose ~L / Q% on what sits in the tail.
                  hard1x1 (x101), hard1x3 (x301), hard0.3x1 (x334); mlp_hard1x3 and skip_hard1x3 with those readouts
  H_hard          a hard arm (any readout) passes H_head_fixable's three conditions: a generic dose catches the consequences
Addendum 4 (declared 2026-10-02 after check_rho and the calm-event count, before any event arm ran; `--tag event`): a generic,
SELECTIVE dose. Label-free selection by loss (hard arms: 93% of the tail is entering / static / HUD) and by reducible loss
(check_rho: the IL model copies the consequences too) both fail. Counted from the labels: in calm frames (fewer than 10% of the
frame's tokens with copy residual > 60, i.e. no scroll), tokens with copy residual > 60 are 0.64% of all tokens, consequences
are 6.7% of them, and 75% of consequences are among them (entering cells are excluded by construction). EAWM's event definition
(camera-compensated change; boundary frames revert to uniform) at a CGSReg-form dose:
    eventCxL      uniform mean + L x the mean over calm-frame events (copy residual > C in frames whose share of such tokens
                  is < 10%); event60x2 (dose ~x300), skip_event60x1, skip_event60x2
  H_event         an event arm passes H_head_fixable's three conditions: a label-free selective dose works at the head
  Every addendum arm's re-trained head is saved (artifacts/eda/headfit_heads_v1/<world>_<arm>.pt) for the cost analysis.
  Every arm also logs its training objective every 500 updates (plateau check).
Usage: headfit.py <world.pt> ... -> evals/headfit_<name>.json
"""
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
import teval as T  # noqa: E402

FACED = torch.tensor([30, 32, 22, 40])           # facing one-hot index (left, right, up, down) -> cell
ACT = torch.tensor([5, 7, 8, 9, 10])
HEAD = ("proj.", "choose.", "frame.", "target_gate.", "gate.", "logits.")
LABELS = ROOT / "artifacts/eda/headfit_labels_v1.pt"
HEADS = ROOT / "artifacts/eda/headfit_heads_v1"
LOCAL = 5 * 192 + 17                                    # token + 4 grid neighbours + action one-hot (check_allprobe's input)


def local_features(s, a):
    """[B,T,81,192], [B,T] -> [B,T,81,977]: each token, its 4 grid neighbours (zeros off-grid) and the action."""
    g = s.float().view(*s.shape[:2], 9, 9, 192); p = F.pad(g, (0, 0, 1, 1, 1, 1))
    nb = [g] + [p[:, :, 1 + dr:10 + dr, 1 + dc:10 + dc] for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))]
    act = F.one_hot(a, 17).float()[:, :, None, None].expand(-1, -1, 9, 9, -1)
    return torch.cat(nb + [act], -1).view(*s.shape[:2], 81, -1)


class Skip(torch.nn.Module):
    """A head module reading [h, the raw local input of the current forward pass]."""

    def __init__(self, net, holder):
        super().__init__()
        self.net, self.holder = net, holder

    def forward(self, h):
        return self.net(torch.cat([h, self.holder["x"].to(h.dtype)], -1))
UPDATES, BATCH = 3000, 40


def copy_residual(x0, x1):
    g0 = x0.view(-1, 9, 9, 192); p = F.pad(g0, (0, 0, 1, 1, 1, 1))
    cands = [g0] + [p[:, 1 + dr:10 + dr, 1 + dc:10 + dc] for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))]
    return torch.stack([((x1.view(-1, 9, 9, 192) - c) ** 2).sum(-1) for c in cands]).amin(0).view(-1, 81)


def labels(pool, probes, log):
    if LABELS.exists():
        return torch.load(LABELS)
    n = len(pool["actions"])
    out = {"faced": torch.zeros(n, 5, dtype=torch.long), "attempt": torch.zeros(n, 5, dtype=torch.bool),
           "cons": torch.zeros(n, 5, dtype=torch.bool), "before": torch.zeros(n, 5, dtype=torch.long),
           "after": torch.zeros(n, 5, dtype=torch.long), "copyres": torch.zeros(n, 5, 81, dtype=torch.float16),
           "same": torch.zeros(n, 5, 81, dtype=torch.float16)}
    for i in range(0, n, 256):
        s = pool["tokens"][i:i + 256].float(); a = pool["actions"][i:i + 256]; b = len(s)
        face = FACED[probes.facing(s[:, :5, 31].flatten(0, 1)).argmax(-1)].view(b, 5)
        rows = torch.arange(b)
        for t in range(5):
            x0, x1, cell = s[:, t], s[:, t + 1], face[:, t]
            c0 = probes.tile(x0[rows, cell]).argmax(-1); c1 = probes.tile(x1[rows, cell]).argmax(-1)
            d = ((x1[rows, cell] - x0[rows, cell]) ** 2).sum(-1)
            att = torch.isin(a[:, t], ACT) & pool["alive"][i:i + b, t + 1]
            out["faced"][i:i + b, t] = cell; out["attempt"][i:i + b, t] = att
            out["cons"][i:i + b, t] = att & (c0 != c1) & (d > 120)
            out["before"][i:i + b, t] = c0; out["after"][i:i + b, t] = c1
            out["copyres"][i:i + b, t] = copy_residual(x0, x1).half(); out["same"][i:i + b, t] = ((x1 - x0) ** 2).sum(-1).half()
        if i % 5120 == 0:
            log(stage="labels", done=i)
    torch.save(out, LABELS)
    return out


def head_params(world):
    return [p for n, p in world.named_parameters() if n.startswith(HEAD)]


def per_token(world, s, a, kind, device, cut=None):
    """[B,5,81] per-token loss of the 5 teacher-forced next-frame predictions (cut: SimPLe's per-element dead zone)."""
    from tworld import quantize
    a = F.pad(a, (0, 1))
    if world.head == "categorical":
        idx = quantize(s, world.codes)
        _, _, logits = world(world.codes[idx], a)
        ce = F.cross_entropy(logits[:, :5].flatten(0, 2).float(), idx[:, 1:].flatten(), reduction="none").view(len(s), 5, 81)
        return ce if cut is None else (ce - cut).clamp(min=0)
    out = world(s, a)[0][:, :5].float()
    err = out - s[:, 1:].float()
    if cut is not None:
        return (err.abs() - cut).clamp(min=0).mean(-1)
    return err.abs().mean(-1) if kind == "l1" else (err ** 2).mean(-1)


def weights(arm, lab, rows, device):
    """[B,5,81] token weights for an arm (None = uniform)."""
    if arm in ("continue", "uniform", "l2"):
        return None
    if arm.startswith("eawm"):
        ev = (lab["same" if arm == "eawm_same" else "copyres"][rows].float() > 25).to(device)
        w_ = {"eawm_same": 0.5, "eawm_copy": 0.9, "eawm_copy99": 0.99}[arm]
        boundary = ev.float().mean(-1, keepdim=True) >= 0.5
        w = 1 + w_ * (~boundary).float() * (ev.float() - 1)
        return w / w.mean()
    raise ValueError(arm)


def objective(world, s, a, arm, lab, rows, device, cuts=None):
    if arm in (cuts or {}):
        return per_token(world, s, a, "l1", device, cuts[arm]).mean()
    if arm.startswith("event"):                # addendum 4: dose calm-frame camera-compensated change events
        c, lam = (float(v) for v in arm[5:].split("x"))
        tok = per_token(world, s, a, "l1", device)
        ev = lab["copyres"][rows].float().to(device) > c
        m = ev & (ev.float().mean(-1, keepdim=True) < 0.1)
        return tok.mean() + lam * (tok * m).sum() / m.sum().clamp(min=1)
    if arm.startswith("hard"):                 # addendum 3: dose the batch's top-Q% tokens by current loss
        q, lam = (float(v) for v in arm[4:].split("x"))
        tok = per_token(world, s, a, "l1", device)
        thr = tok.detach().flatten().topk(max(1, int(q / 100 * tok.numel()))).values[-1]
        m = tok.detach() >= thr
        return tok.mean() + lam * (tok * m).sum() / m.sum()
    kind = "l2" if arm == "l2" else "l1"
    tok = per_token(world, s, a, kind, device)
    if arm.startswith("mask"):
        m = torch.zeros_like(tok, dtype=torch.bool)
        att = lab["attempt"][rows].to(device); cell = lab["faced"][rows].to(device)
        m.scatter_(2, cell[..., None], att[..., None])
        return tok.mean() + float(arm[4:]) * (tok * m).sum() / m.sum().clamp(min=1)
    w = weights(arm, lab, rows, device)
    return tok.mean() if w is None else (tok * w).mean()


@torch.no_grad()
def cutoffs(world, pool, train_rows, device, config):
    """The trained world's per-element |error| (per-token CE for categorical) quantiles on 20 seeded training batches."""
    from d4mj.train import autocast_context
    from tworld import quantize
    gen = torch.Generator().manual_seed(123); keep = torch.Generator().manual_seed(5); vals = []
    for _ in range(20):
        rows = train_rows[torch.randint(len(train_rows), (BATCH,), generator=gen)]
        s = pool["tokens"][rows].float().to(device); a = F.pad(pool["actions"][rows], (0, 1)).to(device)
        with autocast_context(config):
            if world.head == "categorical":
                idx = quantize(s, world.codes)
                e = F.cross_entropy(world(world.codes[idx], a)[2][:, :5].flatten(0, 2).float(), idx[:, 1:].flatten(), reduction="none")
            else:
                e = (world(s, a)[0][:, :5].float() - s[:, 1:]).abs().flatten()
        vals.append(e.cpu()[torch.randint(len(e), (50000,), generator=keep)])
    v = torch.cat(vals)
    out = {f"clip{q}": float(torch.quantile(v, q / 100)) for q in (50, 75, 90)}
    if world.head == "categorical":
        out["ce_clip03"] = 0.03
    return out


def anatomy(world, pool, lab, train_rows, device, config, kinds):
    from d4mj.train import autocast_context
    groups = {"head": head_params(world)}
    if world.head == "corrt":
        groups.update({"proj": list(world.proj.parameters()), "choose": list(world.choose.parameters()),
                       "move_gate": list(world.frame.parameters()) + list(world.target_gate.parameters())})
    groups["backbone"] = [p for n, p in world.named_parameters() if not n.startswith(HEAD)]
    params = [p for p in world.parameters()]
    where = {id(p): i for i, p in enumerate(params)}
    gen = torch.Generator().manual_seed(123)
    res = {}
    for kind in kinds:
        stats = {g: {"cons": [], "rest": [], "total": [], "cos": [], "share": []} for g in groups}
        mix = {"cons": torch.zeros(6, device=device), "rest": torch.zeros(6, device=device), "n_cons": 0, "n_rest": 0}
        full = {g: [None, None] for g in groups}
        for _ in range(20):
            rows = train_rows[torch.randint(len(train_rows), (BATCH,), generator=gen)]
            s = pool["tokens"][rows].float().to(device); a = pool["actions"][rows].to(device)
            cons = torch.zeros(BATCH, 5, 81, dtype=torch.bool, device=device)
            cons.scatter_(2, lab["faced"][rows].to(device)[..., None], lab["cons"][rows].to(device)[..., None])
            with autocast_context(config):
                tok = per_token(world, s, a, kind, device)
            N = tok.numel()
            if hasattr(world, "last_weights"):
                w6 = world.last_weights[:, :5].float()
                mix["cons"] += w6[cons].sum(0); mix["rest"] += w6[~cons].sum(0)
                mix["n_cons"] += int(cons.sum()); mix["n_rest"] += int((~cons).sum())
            lc, lr = (tok * cons).sum() / N, (tok * ~cons).sum() / N
            gc = torch.autograd.grad(lc, params, retain_graph=True, allow_unused=True)
            gr = torch.autograd.grad(lr, params, allow_unused=True)
            flat = lambda gs, ps: torch.cat([(gs[where[id(p)]] if gs[where[id(p)]] is not None else torch.zeros_like(p)).flatten()
                                             for p in ps]).float()
            for g, ps in groups.items():
                vc, vr = flat(gc, ps), flat(gr, ps)
                vt = vc + vr
                stats[g]["cons"].append(float(vc.norm())); stats[g]["rest"].append(float(vr.norm())); stats[g]["total"].append(float(vt.norm()))
                stats[g]["cos"].append(float(F.cosine_similarity(vc, vr, dim=0)) if vc.norm() > 0 else 0.0)
                stats[g]["share"].append(float((vc @ vt) / (vt @ vt).clamp(min=1e-30)))
                full[g][0] = vc if full[g][0] is None else full[g][0] + vc
                full[g][1] = vr if full[g][1] is None else full[g][1] + vr
            del tok, gc, gr
        res[kind] = {}
        if mix["n_cons"]:
            res[kind]["mixture_weights"] = {k: [round(float(v), 5) for v in mix[k] / mix["n_" + k]] for k in ("cons", "rest")}
        for g in groups:
            m = lambda k: sum(stats[g][k]) / len(stats[g][k])
            vc, vr = full[g]; vt = vc + vr
            res[kind][g] = {"norm_cons": m("cons"), "norm_rest": m("rest"), "norm_total": m("total"),
                            "ratio": m("cons") / max(m("rest"), 1e-30), "cons_over_total": m("cons") / max(m("total"), 1e-30),
                            "cos": m("cos"), "share": m("share"),
                            "full_batch": {"ratio": float(vc.norm() / vr.norm().clamp(min=1e-30)),
                                           "cos": float(F.cosine_similarity(vc, vr, dim=0)), "share": float((vc @ vt) / (vt @ vt))}}
    return res


@torch.no_grad()
def evaluate(world, pool, lab, rows_eval, probes, device, config):
    from d4mj.train import autocast_context
    from tworld import quantize
    c = {"cons": 0, "caught": 0, "copied": 0, "att_nochange": 0, "halluc": 0}
    l1_all = l1_nonatt = 0.0; n_all = n_nonatt = 0; hud_pred = hud_copy = 0.0
    for i in range(0, len(rows_eval), 64):
        r = rows_eval[i:i + 64]
        s = pool["tokens"][r].float().to(device); a = F.pad(pool["actions"][r], (0, 1)).to(device)
        s_in = world.codes[quantize(s, world.codes)] if world.head == "categorical" else s
        with autocast_context(config):
            pred = world(s_in, a)[0][:, :5].float()
        err = (pred - s[:, 1:]).abs().mean(-1)                                       # [b,5,81]
        cell = lab["faced"][r].to(device); att = lab["attempt"][r].to(device); cons = lab["cons"][r].to(device)
        attm = torch.zeros_like(err, dtype=torch.bool); attm.scatter_(2, cell[..., None], att[..., None])
        l1_all += float(err.sum()); n_all += err.numel(); l1_nonatt += float(err[~attm].sum()); n_nonatt += int((~attm).sum())
        pf = pred.gather(2, cell[..., None, None].expand(-1, -1, 1, 192))[:, :, 0].cpu()   # [b,5,192]
        pc = probes.tile(pf.flatten(0, 1)).argmax(-1).view(len(r), 5)
        before, after = lab["before"][r], lab["after"][r]
        consc, attc = cons.cpu(), att.cpu()
        c["cons"] += int(consc.sum()); c["caught"] += int(((pc == after) & consc).sum()); c["copied"] += int(((pc == before) & consc).sum())
        nc = attc & ~consc
        nc = nc & (before == after)
        c["att_nochange"] += int(nc.sum()); c["halluc"] += int(((pc != before) & nc).sum())
        if consc.any():
            hp = (pred[:, :, 63:] - s[:, 1:, 63:]).abs().sum((-1, -2)); hc = (s[:, :5, 63:] - s[:, 1:, 63:]).abs().sum((-1, -2))
            hud_pred += float(hp[cons].sum()); hud_copy += float(hc[cons].sum())
    return {"caught": c["caught"] / max(c["cons"], 1), "copied": c["copied"] / max(c["cons"], 1),
            "hallucinated": c["halluc"] / max(c["att_nochange"], 1), "n_cons": c["cons"],
            "l1_all": l1_all / max(n_all, 1), "l1_nonattempt": l1_nonatt / max(n_nonatt, 1),
            "hud_pred_over_copy_on_cons": hud_pred / max(hud_copy, 1e-9)}


def main():
    from d4mj.config import config_from_dict
    from d4mj.train import autocast_context
    import spatial as Sp
    from tworld import POOLS, TWorld
    started = time.time()
    log = lambda **kw: print(json.dumps({**kw, "seconds": round(time.time() - started, 1)}), flush=True)
    device = torch.device("cuda")
    config = config_from_dict(torch.load(Sp.CHECKPOINT, map_location="cpu", weights_only=False)["config"])
    meta, train_roots, train_seeds = T.split()
    probes = T.Probes(T.build_cache("raw", torch.device("cpu")), meta, train_roots, train_seeds)
    pool = torch.load(POOLS["raw"] / "pool.pt", weights_only=False, mmap=True)
    lab = labels(pool, probes, log)
    main_rows = torch.where(~pool["terminal"])[0]
    held = main_rows[torch.randperm(len(main_rows), generator=torch.Generator().manual_seed(1))[:2048]]
    train_rows = torch.cat([main_rows[~torch.isin(main_rows, held)], torch.where(pool["terminal"])[0]])
    train_eval = train_rows[torch.randperm(len(train_rows), generator=torch.Generator().manual_seed(20261001))[:4096]]
    rich = train_rows[lab["cons"][train_rows].any(-1)]
    log(stage="labels", cons_rate=float(lab["cons"].float().mean()), attempt_rate=float(lab["attempt"].float().mean()),
        rich_windows=len(rich), train_windows=len(train_rows))
    argv = sys.argv[1:]
    only = argv[argv.index("--arms") + 1].split(",") if "--arms" in argv else None
    tag = argv[argv.index("--tag") + 1] if "--tag" in argv else None
    paths = [Path(p) for p in argv if p.endswith(".pt")]
    for path in paths:
        trained, st = T.load_world(path, device)
        name = st["name"]
        out = HERE / "evals" / f"headfit_{name}.json"
        if only:
            res = json.loads(out.read_text())
            trained.eval()
            res["cutoffs"] = cutoffs(trained, pool, train_rows, device, config)
            arms = only
            log(world=name, cutoffs=res["cutoffs"])
        else:
            res = {"world": name, "head": trained.head}
            trained.train()
            kinds = ["l1"] if trained.head == "categorical" else ["l1", "l2"]
            res["anatomy"] = anatomy(trained, pool, lab, train_rows, device, config, kinds)
            log(world=name, anatomy=res["anatomy"])
            trained.eval()
            res["trained"] = {"train": evaluate(trained, pool, lab, train_eval, probes, device, config),
                              "held": evaluate(trained, pool, lab, held, probes, device, config)}
            arms = ["continue", "uniform", "l2", "eawm_same", "eawm_copy", "eawm_copy99", "mask10", "resample"]
            if trained.head == "categorical":
                arms.remove("l2")
            res["arms"] = {}
        for arm in arms:
            world, _ = T.load_world(path, device)
            if arm != "continue":
                torch.manual_seed(0)
                fresh = TWorld(world.head, world.codes.cpu() if world.head == "categorical" else None,
                               st["args"].get("backbone", "full"), st["args"].get("regions", "all"), 0)
                sd = world.state_dict()
                sd.update({k: v.to(device) for k, v in fresh.state_dict().items() if k.startswith(HEAD)})
                world.load_state_dict(sd)
            if arm.startswith("skip_"):          # the head also reads the raw local neighbourhood (addendum 2)
                torch.manual_seed(0)
                holder = {}
                for nm in ("proj", "choose", "logits"):
                    lin = getattr(world, nm, None)
                    if isinstance(lin, torch.nn.Linear):
                        last = torch.nn.Linear(512, lin.out_features)
                        if nm == "choose":
                            torch.nn.init.zeros_(last.weight); last.bias.data.copy_(lin.bias.data)
                        net = torch.nn.Sequential(torch.nn.Linear(lin.in_features + LOCAL, 512), torch.nn.GELU(), last)
                        setattr(world, nm, Skip(net, holder).to(device))

                def fwd(s, a, forward=world.forward, holder=holder):
                    holder["x"] = local_features(s, a)
                    return forward(s, a)
                world.forward = fwd
            if arm.startswith("mlp_"):           # the same frozen h, an expressive readout
                torch.manual_seed(0)
                for nm in ("proj", "choose", "logits"):
                    lin = getattr(world, nm, None)
                    if isinstance(lin, torch.nn.Linear):
                        last = torch.nn.Linear(512, lin.out_features)
                        if nm == "choose":
                            torch.nn.init.zeros_(last.weight); last.bias.data.copy_(lin.bias.data)
                        setattr(world, nm, torch.nn.Sequential(torch.nn.Linear(lin.in_features, 512), torch.nn.GELU(), last).to(device))
            for n_, p in world.named_parameters():
                p.requires_grad_(n_.startswith(HEAD))
            params = head_params(world)
            opt = torch.optim.AdamW(params, lr=1e-3, weight_decay=0.01)
            gen = torch.Generator().manual_seed(11)
            world.train(); curve = []
            for u in range(UPDATES):
                for g in opt.param_groups:
                    g["lr"] = 1e-3 * min(1.0, (u + 1) / 100)
                rows = train_rows[torch.randint(len(train_rows), (BATCH,), generator=gen)]
                if arm == "resample":
                    rows = torch.cat([rows[:BATCH // 2], rich[torch.randint(len(rich), (BATCH // 2,), generator=gen)]])
                s = pool["tokens"][rows].float().to(device); a = pool["actions"][rows].to(device)
                with autocast_context(config):
                    base = "uniform" if arm == "resample" else arm.split("_", 1)[1] if arm.startswith(("mlp_", "skip_")) else arm
                    loss = objective(world, s, a, base, lab, rows, device, res.get("cutoffs"))
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                if (u + 1) % 500 == 0:
                    curve.append(round(float(loss), 5))
            world.eval()
            res["arms"][arm] = {"train": evaluate(world, pool, lab, train_eval, probes, device, config),
                                "held": evaluate(world, pool, lab, held, probes, device, config), "objective_every_500": curve}
            if only:
                HEADS.mkdir(parents=True, exist_ok=True)
                torch.save({k: v.cpu() for k, v in world.state_dict().items() if k.startswith(HEAD)}, HEADS / f"{name}_{arm}.pt")
            log(world=name, arm=arm, held={k: round(v, 4) if isinstance(v, float) else v for k, v in res["arms"][arm]["held"].items()},
                train_caught=round(res["arms"][arm]["train"]["caught"], 4))
            del world, opt
            torch.cuda.empty_cache()
        an, A = res["anatomy"], res["arms"]
        rd = {"G_starved": an["l1"]["head"]["cons_over_total"] < 0.05,
              "G_cancel": an["l1"]["head"]["cos"] < -0.2}
        if "l2" in an:
            rd["G_l1_sign_factor"] = an["l2"]["head"]["ratio"] / max(an["l1"]["head"]["ratio"], 1e-30)
            rd["G_l1_sign"] = rd["G_l1_sign_factor"] >= 5
        u = A["uniform"]["held"]
        rd["H_uniform_copies"] = u["caught"] < 0.1
        rd["H_converged"] = u["l1_all"] <= 1.05 * res["trained"]["held"]["l1_all"]
        ok = {k: v["held"]["caught"] >= 0.5 and v["held"]["l1_all"] <= 1.1 * u["l1_all"] and v["held"]["hallucinated"] <= u["hallucinated"] + 0.05
              for k, v in A.items() if k not in ("continue", "uniform")}
        rd["H_head_fixable"] = any(ok.values()); rd["H_fixing_arms"] = [k for k, v in ok.items() if v]
        if "l2" in A:
            rd["H_geometry"] = A["l2"]["held"]["caught"] >= 0.5 and u["caught"] < 0.1
        rd["H_generic"] = any(A[k]["held"]["caught"] >= 0.5 for k in ("l2", "eawm_same", "eawm_copy", "eawm_copy99") if k in A)
        clip = [k for k in ok if k.startswith(("clip", "ce_clip"))]
        if clip:
            rd["H_clip"] = any(ok[k] for k in clip)
        if "mlp_mask10" in A and "mlp_uniform" in A:
            m10, mu_ = A["mlp_mask10"]["held"], A["mlp_uniform"]["held"]
            rd["H_capacity"] = m10["caught"] >= 0.5 and m10["l1_all"] <= 1.1 * mu_["l1_all"]
            rd["H_representation"] = m10["caught"] >= 0.5 and m10["l1_all"] > 1.1 * mu_["l1_all"]
        event = [k for k in ok if k.split("_")[-1].startswith("event")]
        if event:
            rd["H_event"] = any(ok[k] for k in event); rd["H_event_arms"] = [k for k in event if ok[k]]
        hard = [k for k in ok if k.split("_")[-1].startswith("hard")]
        if hard:
            rd["H_hard"] = any(ok[k] for k in hard); rd["H_hard_arms"] = [k for k in hard if ok[k]]
        if "skip_uniform" in A:
            rd["H_local_uniform"] = A["skip_uniform"]["held"]["caught"] >= 0.5 and A["skip_uniform"]["held"]["l1_all"] <= 1.1 * u["l1_all"]
            rd["H_local_dose"] = any(ok.get(k, False) for k in ("skip_mask1", "skip_mask10"))
        dose = sorted((float(k[4:]), k) for k in A if k.startswith("mask") and A[k]["held"]["caught"] >= 0.5)
        if dose:
            rd["mask_dose"] = {"arm": dose[0][1], "l1_over_uniform": A[dose[0][1]]["held"]["l1_all"] / u["l1_all"]}
        res["readings"] = rd
        (out if tag is None else out.with_name(f"headfit_{name}_{tag}.json")).write_text(json.dumps(res, indent=2) + "\n")
        log(world=name, readings=rd)
        del trained
        torch.cuda.empty_cache()


if __name__ == "__main__":
    sys.path.insert(0, str(HERE.parent / "20260926_diagnosis"))
    sys.path.insert(0, str(HERE.parent / "20260921_readout_ladder"))
    main()
