"""Revised 2x2: world state (z vs per-tile tokens) x added health-change supervision, on a fresh block.

Why (PROPOSAL.md, revised after review): canonical LeWM's world receives only z = projector(CLS), which
reads the zombie hazard at the action prior while the same encoder's per-tile tokens carry it
(BOUNDARY.md, sealed). A world on a patch-derived state still lost the consequence under next-state
MSE (ABC.md, sealed). H2 ALREADY trains reward and continuation heads on generated readouts with
gradients into the world (`d4mj/train.py:_bridge_head_losses`), and those heads did not transfer
between real and generated states; so that supervision is held CONSTANT here, in every arm. What
is new, and what this crosses, is (i) the per-tile state and (ii) a health-change head that reads
ONLY the current and successor states -- H2's heads also read the predictor's history, which lets
them bypass the generated latent; this head's gradient can reach the predictor only through the
generated state. One-step decides this experiment; two-step and H16 come later.

ARMS -- identical in everything but the two factors:
  Z    z state (1 token per frame),        H2-style reward + continuation heads
  ZH   z state,                            + health-change head
  T    81 per-tile tokens (one per 7x7 tile, including the 18 HUD tiles), H2-style heads
  TH   81 per-tile tokens,                 + health-change head

HELD FIXED:
  encoder   the frozen canonical Raw H2 encoder (`raw/bridge/step-002000.pt`, sha pinned to
            confirm.json). State = its projected z (Z arms) or its 81 output patch tokens (T arms),
            each layer-normalized over its 192 features: V-JEPA 2-AC's `normalize_reps`.
  data      the M4 corpus (dataset.json contract), TRAIN split, uniform-eligible. A fixed pool of
            6-frame windows: 24,576 main windows, uniform over transitions (episode drawn by
            start count, start uniform), plus every TRAIN terminal episode's final window.
            Pool seed 20260926. Nothing from any evaluation block.
  predictor V-JEPA 2-AC's layout at H2's scale: per frame [action token, state tokens]; frame
            block-causal attention (tokens see their own frame and every earlier one); 6 pre-norm
            transformer layers, width 256 (H2 `dynamics.depth/width`), 4 heads, MLP 4x, GELU;
            learned space and time positions (the encoder's own scheme; V-JEPA uses RoPE);
            output projected to 192 and layer-normed. Output at frame t predicts frame t+1.
  losses    dynamics: V-JEPA 2-AC's L1 on normalized targets, over the teacher-forced
            predictions plus a depth-2 recursively generated suffix (H2 `recursive_depth` 2;
            `_bridge_dynamics_loss` shape, anchor = frame 3).
            heads: H2's own `Heads`, `_bridge_head_losses` (0.5 observed prefix / 0.25 observed
            suffix / 0.25 generated suffix) and 0.8/0.2 `paired_terminal_loss` on the terminal
            batch, reading per token H2's agent readout Linear(state + history) -> LN -> GELU,
            mean-pooled by `Heads`. The policy (behaviour-cloning) head is masked out in every
            arm: it supervises the logged action, which is the shortcut this test guards against.
            health (ZH, TH only): 11-way cross-entropy on dh in [-9, +1] (exposure.health_change:
            reward = achievements + 0.1 dh, verified exact on 7,021 fork transitions), from a head
            reading only (s_t, s_{t+1}): per token Linear -> LN -> GELU, mean-pooled, SwiGLU,
            Linear. Same 0.5/0.25/0.25 observed/generated weighting; main and terminal batches
            combined 0.8/0.2 as continuation.
            Every group RMS-balanced by H2's `_phase_balance` (decay 0.99).
  optimizer H2's phase optimizer and schedule: AdamW lr 1e-4, 1,000-step warmup, wd 0.01, clip 1.0,
            bf16 autocast. Init seed 7, batch seed 11 -> identical batch order across arms.
  budget    20,000 updates of 32 main + 8 terminal windows (H2's 4:1 ratio, twice its batch), every
            arm, fixed from the throughput smoke before this commit: at 64+16 the per-tile arms
            exceed the 6 GB GPU; at 32+8 they run 0.235 s/update in 2.4 GB (z arms 0.06 s, 0.3 GB).
            PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True in every arm (allocation only).
            Pool: 32,647 windows (24,576 main incl. 84 natural terminals, 8,071 terminal),
            pool_sha256 7662de45ae04...; 91.6% of transitions have dh = 0.

JUDGEMENT -- `observe.py collect --seed-start 53000 --target-opportunity 800 --max-seeds 1500 --out
artifacts/eda/observe_fresh_v4`, collected only after this commit; scored once. FIT roots for the
controls and diagnostic probes: the observability FIT set, as every earlier rung.
Context = the root's last 4 frames and the 3 actions between them; each of the 17 actions advances
once. The TRAINED head's risk for an action is 1 - sigmoid(continuation) on the generated
successor's readout; expected safe = 1 - P(death1 | chosen), on one-step opportunity roots; paired
episode-seed-clustered 95% intervals, 1,000 draws.

Controls (frozen_ladder heads, three probe seeds, inner-FIT selection): action marginal (the FIT
prior action); actions_only (the last 4 actions); root-plus-action (tokens_attn on the root's patch
tokens, the strongest root readout).

DECLARED RULES (one-step death):
  P0  boundary.py run unchanged on this block (`--store observe_fresh_v4 --name boundary_v4`):
      R1 tokens_attn - cls > 0, resolved.  Else -> void_no_root_advantage; nothing else is read.
  For TH's trained head:
  S1  TH - ZH > 0, resolved, on zombie-adjacent opportunity roots
  S2  TH - action marginal > 0 and TH - actions_only > 0, both resolved (all opportunity roots)
  S3  retention: TH - root-plus-action not resolved below zero. (A successor generated from the
      root and the action cannot carry more about one-step death than they do; the successor
      must keep it, not beat it.)
  S4  TH's generated-state errors below ZH's, both resolved: Brier of the trained continuation
      against 32-key P(death1) over all 17 branches; cross-entropy of the health head against
      the true health change
  S5  movement kept: TH's within-root normalized action-effect error <= 2 x T's
  All five -> spatial_health_passes. Otherwise -> spatial_health_fails, listing what failed.
Reported, not ruled: every arm's trained-head safe choice overall / zombie / lava / night; the 2x2
contrasts TH-ZH, TH-T, ZH-Z, T-Z and the interaction; roots where staying put kills and a move
survives, separately; SLEEP-vs-NOOP ranking against the truth; chosen-action histograms; fresh
generated-feature probes fitted on FIT generated states (diagnostic only).
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

ROOT = Path("/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA")
HERE = Path(__file__).parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from d4mj.data import _sha256

N, TOKENS, WIDTH, D = 17, 81, 192, 256
W, DEPTH = 6, 2
ANCHOR = W - 1 - DEPTH                                  # frames 0..3 observed, 4..5 generated
HEALTH = 11                                             # dh in [-9, +1]
CHECKPOINT = ROOT / "artifacts/lewm_m4_canonical/raw/bridge/step-002000.pt"
DATASET = ROOT / "artifacts/lewm_m4_canonical/raw/dataset.json"
POOL = ROOT / "artifacts/eda/spatial_pool_v1"
WORLDS = ROOT / "artifacts/eda/spatial_worlds_v1"
SEALED = ROOT / "artifacts/eda/observe_fresh_v4"
MAIN_WINDOWS, POOL_SEED = 24_576, 20260926
UPDATES, MAIN, TERMINAL = 20_000, 32, 8                 # fixed from the smoke before commit
ARMS = {"Z": ("z", False), "ZH": ("z", True), "T": ("tokens", False), "TH": ("tokens", True)}


def bridge():
    from d4mj.experiments import _load_bridge_parent
    if _sha256(CHECKPOINT) != json.loads((HERE / "evidence/confirm.json").read_text())["checkpoint_sha256"]:
        raise SystemExit("checkpoint differs from the one every earlier rung used")
    bundle, heads, _ = _load_bridge_parent(CHECKPOINT)
    bundle.encoder.freeze()
    return bundle.encoder, bundle.config


@torch.no_grad()
def encode(encoder, frames, device, batch=32):
    """uint8 [n, t, 63, 63, 3] -> layer-normed z [n, t, 192] and tokens [n, t, 81, 192] (fp16)."""
    zs, ts = [], []
    for i in range(0, len(frames), batch):
        f = torch.as_tensor(frames[i:i + batch]).to(device)
        n, t = f.shape[:2]
        z, _, tokens, _, _ = encoder._hidden(f)
        zs.append(F.layer_norm(z.float(), (WIDTH,)).reshape(n, t, WIDTH).cpu())
        ts.append(F.layer_norm(tokens.float(), (WIDTH,)).reshape(n, t, TOKENS, WIDTH).half().cpu())
    return torch.cat(zs), torch.cat(ts)


# ------------------------------------------------------------------------------------------ pool
def build_pool(device, log):
    from d4mj.data import load_joint_corpus
    from exposure import health_change
    encoder, config = bridge()
    record = json.loads(DATASET.read_text())
    episodes, contract = load_joint_corpus(record["paths"], config)     # relative, as recorded (cwd = ROOT)
    if contract != record["contract"]:
        raise SystemExit("the M4 corpus differs from its dataset contract")
    eligible = [e for e in episodes if e.split == "train" and e.uniform_eligible and len(e) + 1 >= W]
    terminal = [e for e in eligible if bool(e.terminated[-1])]
    gen = torch.Generator().manual_seed(POOL_SEED)
    counts = torch.tensor([len(e) + 2 - W for e in eligible], dtype=torch.float64)
    windows = [(eligible[i], int(torch.randint(int(counts[i]), (), generator=gen)))
               for i in torch.multinomial(counts, MAIN_WINDOWS, replacement=True, generator=gen).tolist()]
    windows += [(e, len(e) + 1 - W) for e in terminal]
    log(stage="windows", main=MAIN_WINDOWS, terminal=len(terminal))
    out = {k: [] for k in ("actions", "reward_led", "reward_valid", "alive", "dh", "ids")}
    out["z"] = torch.empty(len(windows), W, WIDTH)                                   # preallocated: ~6 GB of
    out["tokens"] = torch.empty(len(windows), W, TOKENS, WIDTH, dtype=torch.float16)  # tokens, never copied
    for b in range(0, len(windows), 256):
        chunk = windows[b:b + 256]
        z, tokens = encode(encoder, np.stack([np.asarray(e.observations[s:s + W]) for e, s in chunk]), device)
        out["z"][b:b + len(chunk)], out["tokens"][b:b + len(chunk)] = z, tokens
        for e, s in chunk:
            rewards = torch.as_tensor(np.asarray(e.rewards[max(s - 1, 0):s + W - 1]), dtype=torch.float64)
            led = rewards if s > 0 else torch.cat([rewards.new_zeros(1), rewards])       # reward that ARRIVED at each frame
            done = torch.as_tensor(np.asarray(e.terminated[s:s + W - 1]), dtype=torch.bool)
            out["actions"].append(torch.as_tensor(np.asarray(e.actions_taken[s:s + W - 1])).long())
            out["reward_led"].append(led.float())
            out["reward_valid"].append(torch.tensor([s > 0] + [True] * (W - 1)))
            out["alive"].append(torch.cat([torch.ones(1, dtype=torch.bool), ~done]))
            out["dh"].append(health_change(led[1:]))
            out["ids"].append((e.episode_id, s))
        if b % 4096 == 0:
            log(stage="encoding", done=b + len(chunk), of=len(windows))
    pool = {k: (v if k in ("z", "tokens", "ids") else torch.stack(v)) for k, v in out.items()}
    if not bool(((pool["dh"] >= -9) & (pool["dh"] <= 1)).all()):
        raise SystemExit("health change outside [-9, +1]")
    if bool((~pool["alive"][:, :-1]).any()):
        raise SystemExit("a window continues past a terminal")
    pool["terminal"] = torch.arange(len(windows)) >= MAIN_WINDOWS
    if not bool((~pool["alive"][pool["terminal"], -1]).all()):
        raise SystemExit("a terminal window does not end in a death")
    POOL.mkdir(parents=True, exist_ok=True)
    meta = {"windows": len(windows), "main": MAIN_WINDOWS, "terminal": len(terminal),
            "natural_terminal_in_main": int((~pool["alive"][:MAIN_WINDOWS, -1]).sum()),
            "dh_histogram": {int(k): int((pool["dh"] == k).sum()) for k in range(-9, 2)},
            "contract_sha256": record["contract"]["sha256"], "checkpoint_sha256": _sha256(CHECKPOINT)}
    torch.save({**pool, "meta": meta}, POOL / "pool.pt")
    (POOL / "pool.json").write_text(json.dumps({**meta, "pool_sha256": _sha256(POOL / "pool.pt")}, indent=2) + "\n")
    return meta


# ----------------------------------------------------------------------------------------- world
class World(nn.Module):
    """V-JEPA 2-AC's frame layout at H2's scale; one class for both states (n = 1 or 81 tokens)."""

    def __init__(self, n, health):
        super().__init__()
        self.n = n
        self.embed, self.action = nn.Linear(WIDTH, D), nn.Embedding(N, D)
        self.space, self.time = nn.Parameter(torch.zeros(n + 1, D)), nn.Parameter(torch.zeros(W, D))
        nn.init.trunc_normal_(self.space, std=0.02)
        nn.init.trunc_normal_(self.time, std=0.02)
        layer = nn.TransformerEncoderLayer(D, 4, 4 * D, dropout=0.0, activation="gelu", batch_first=True,
                                           norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, 6, enable_nested_tensor=False)
        self.norm, self.proj = nn.LayerNorm(D), nn.Linear(D, WIDTH)
        # H2's agent readout (`LeWMWorld.agent_readout`), applied per token.
        self.readout = nn.Sequential(nn.Linear(WIDTH + D, D), nn.LayerNorm(D), nn.GELU())
        if health:
            from d4mj.backbone import SwiGLU
            self.pair = nn.Sequential(nn.Linear(2 * WIDTH, D), nn.LayerNorm(D), nn.GELU())
            self.change = nn.Sequential(SwiGLU(D, 2.0), nn.Linear(D, HEALTH))
        t = torch.arange(W).repeat_interleave(n + 1)
        self.register_buffer("blocked", t[None, :] > t[:, None], persistent=False)   # True = may not attend

    def forward(self, s, a):
        """s [B, T, n, 192] normalized states, a [B, T] outgoing actions -> prediction of frame t+1 and
        the history at t, both [B, T, n, .]."""
        b, t = s.shape[:2]
        x = torch.cat([self.action(a)[:, :, None], self.embed(s)], 2) + self.space + self.time[:t, None]
        k = t * (self.n + 1)
        x = self.blocks(x.flatten(1, 2), mask=self.blocked[:k, :k]).view(b, t, self.n + 1, D)[:, :, 1:]
        history = self.norm(x)
        return F.layer_norm(self.proj(history).float(), (WIDTH,)), history

    def agent(self, state, history):
        return self.readout(torch.cat([state.to(history.dtype), history], -1))

    def health(self, now, then):
        return self.change(self.pair(torch.cat([now, then], -1)).mean(-2))


def rollout(world, s, a):
    """Teacher predictions for frames 1..W-1 and a depth-2 generated suffix from anchor frame 3."""
    a = F.pad(a, (0, 1))                                  # the last frame's action is never used
    predicted, history = world(s, a)
    first, first_history = predicted[:, ANCHOR], history[:, ANCHOR]
    second_all, second_history = world(torch.cat([s[:, :ANCHOR + 1], first[:, None]], 1), a[:, :ANCHOR + 2])
    generated = torch.stack([first, second_all[:, ANCHOR + 1]], 1)
    generated_history = torch.stack([first_history, second_history[:, ANCHOR + 1]], 1)
    observed_history = torch.cat([torch.zeros_like(history[:, :1]), history[:, :W - 1]], 1)
    return predicted, generated, observed_history, generated_history


def head_targets(batch, config):
    """H2's `head_targets` fields for these windows; the policy head is masked out."""
    from d4mj.agent import _leads
    leads, b = config.mtp_leads, len(batch["alive"])
    zeros = torch.zeros(b, W, leads, device=batch["alive"].device)
    return {"action": zeros, "action_valid": zeros, "reward": _leads(batch["reward_led"], leads, fill=0.0),
            "valid": _leads(batch["reward_valid"].float(), leads, fill=0.0),
            "continuation": batch["alive"].float()[..., None], "continuation_valid": torch.ones_like(zeros[..., :1]),
            "policy_rows": zeros[:, :1, :1], "reward_rows": torch.ones_like(zeros[:, :1, :1])}


def losses(world, heads, main, terminal, config, health):
    from d4mj.agent import paired_terminal_loss
    from d4mj.train import _bridge_head_losses
    out, parts = {}, {}
    for name, batch in (("main", main), ("terminal", terminal)):
        s = batch["s"]
        predicted, generated, observed_history, generated_history = rollout(world, s, batch["actions"])
        observed = world.agent(s, observed_history)
        recursive = world.agent(generated, generated_history)
        parts[name] = (s, predicted, generated, observed, recursive)
    s, predicted, generated, observed, recursive = parts["main"]
    target = s.float()
    out["dynamics"] = (predicted[:, :W - 1] - target[:, 1:]).abs().mean() + (generated - target[:, ANCHOR + 1:]).abs().mean()
    head = _bridge_head_losses(heads, observed, recursive, head_targets(main, config), config, ANCHOR)
    out["reward"] = head["reward"]
    s_t, _, generated_t, observed_t, recursive_t = parts["terminal"]
    t_observed = heads(observed_t) | {"centers": heads.centers}
    t_recursive = heads(torch.cat([observed_t[:, :ANCHOR + 1], recursive_t], 1)) | {"centers": heads.centers}
    out["continuation"] = 0.8 * head["continuation"] + 0.2 * paired_terminal_loss(
        t_recursive, t_observed, head_targets(terminal, config))
    if health:
        def change(s, generated, dh):
            ce = lambda logits, y: F.cross_entropy(logits.float().flatten(0, 1), (y + 9).flatten(), reduction="none").view(y.shape)
            observed_ce = ce(world.health(s[:, :-1], s[:, 1:]), dh)
            generated_ce = ce(world.health(torch.cat([s[:, ANCHOR:ANCHOR + 1], generated[:, :1]], 1), generated),
                              dh[:, ANCHOR:ANCHOR + 2])
            return 0.5 * observed_ce[:, :ANCHOR].mean() + 0.25 * observed_ce[:, ANCHOR:].mean() + 0.25 * generated_ce.mean()
        out["health"] = 0.8 * change(s, generated, main["dh"]) + 0.2 * change(s_t, generated_t, terminal["dh"])
    return out


def batch_of(pool, rows, state, device):
    s = pool["z"][rows][:, :, None] if state == "z" else pool["tokens"][rows]
    return {"s": s.to(device, non_blocking=True).float(), "actions": pool["actions"][rows].to(device),
            "reward_led": pool["reward_led"][rows].to(device), "reward_valid": pool["reward_valid"][rows].to(device),
            "alive": pool["alive"][rows].to(device), "dh": pool["dh"][rows].to(device)}


def train(arm, pool, device, log, *, updates, main_batch, terminal_batch, every=500):
    from d4mj.agent import Heads
    from d4mj.config import config_from_dict
    from d4mj.train import _phase_balance, _phase_lr, autocast_context, optimizer_step, phase_optimizer
    state, health = ARMS[arm]
    config = config_from_dict(torch.load(CHECKPOINT, map_location="cpu", weights_only=False)["config"])
    with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
        torch.manual_seed(7)
        torch.cuda.manual_seed_all(7)
        world = World(1 if state == "z" else TOKENS, health).to(device)
        heads = Heads(config).to(device)
    optimiser = phase_optimizer([world, heads], config)
    parameters = [p for group in optimiser.param_groups for p in group["params"]]
    main_rows, terminal_rows = torch.where(~pool["terminal"])[0], torch.where(pool["terminal"])[0]
    order, balance, history = torch.Generator().manual_seed(11), {}, []
    started = time.time()
    for update in range(updates):
        main = batch_of(pool, main_rows[torch.randint(len(main_rows), (main_batch,), generator=order)], state, device)
        terminal = batch_of(pool, terminal_rows[torch.randint(len(terminal_rows), (terminal_batch,), generator=order)],
                            state, device)
        with autocast_context(config):
            parts = losses(world, heads, main, terminal, config, health)
            objective = _phase_balance(parts, balance, config.agent.rms_decay)
        if not bool(torch.isfinite(objective)):
            raise RuntimeError(f"nonfinite objective at update {update}")
        optimiser.zero_grad(set_to_none=True)
        norm = optimizer_step(optimiser, objective, parameters, learning_rate=_phase_lr(config, update),
                              grad_clip=config.agent.grad_clip, strict=True, zero_grad=False)
        if (update + 1) % every == 0 or update + 1 == updates:
            row = {"update": update + 1, "gradient_norm": float(norm), "seconds": round(time.time() - started, 1),
                   **{k: round(float(v.detach()), 5) for k, v in parts.items()}}
            history.append(row)
            log(stage="train", arm=arm, **row)
    return world.eval(), heads.eval(), history


# ----------------------------------------------------------------------------------------- main
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("pool", "smoke", "train", "score"))
    parser.add_argument("--arm", choices=tuple(ARMS))
    parser.add_argument("--updates", type=int, default=60, help="smoke only")
    parser.add_argument("--batches", default="64:16,32:8", help="smoke only: main:terminal pairs")
    args = parser.parse_args(argv)
    started = time.time()
    log = lambda **kw: print(json.dumps({**kw, "seconds": round(time.time() - started, 1)}), flush=True)
    device = torch.device("cuda")
    import os
    os.chdir(ROOT)                                        # the corpus contract records relative paths

    if args.command == "pool":
        log(status="pool_complete", **build_pool(device, log))
        return 0
    pool = torch.load(POOL / "pool.pt", weights_only=False, mmap=True)
    recorded = json.loads((POOL / "pool.json").read_text())
    if args.command == "smoke":
        for arm in ((args.arm,) if args.arm else ARMS):
            torch.cuda.reset_peak_memory_stats()
            for main_batch, terminal_batch in (tuple(map(int, b.split(":"))) for b in args.batches.split(",")):
                begin = time.time()
                train(arm, pool, device, log, updates=args.updates, main_batch=main_batch,
                      terminal_batch=terminal_batch, every=args.updates)
                log(stage="smoke", arm=arm, main=main_batch, terminal=terminal_batch,
                    seconds_per_update=round((time.time() - begin) / args.updates, 4),
                    peak_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2))
        return 0
    if args.command == "train":
        world, heads, history = train(args.arm, pool, device, log, updates=UPDATES, main_batch=MAIN,
                                      terminal_batch=TERMINAL)
        WORLDS.mkdir(parents=True, exist_ok=True)
        torch.save({"arm": args.arm, "world": world.state_dict(), "heads": heads.state_dict(), "history": history,
                    "script_sha256": _sha256(Path(__file__)), "pool_sha256": recorded["pool_sha256"]},
                   WORLDS / f"{args.arm}.pt")
        log(status="train_complete", arm=args.arm)
        return 0
    return score(device, log, recorded)


# ----------------------------------------------------------------------------------------- score
@torch.no_grad()
def branches(world, heads, encoder, config, frames, actions, device, state, health, successors=None, batch=16):
    """Every root's 17 one-step branches: the trained head's P(dead), the health head's logits, the
    pooled readout, and (with real successors) the within-root action-effect error and energy."""
    from d4mj.train import autocast_context
    out = {k: [] for k in ("p_dead", "dh_logits", "features", "error", "energy")}
    for i in range(0, len(frames), batch):
        z, tokens = encode(encoder, frames[i:i + batch, -4:], device)
        n = len(z)
        s = (z[:, :, None] if state == "z" else tokens.float()).to(device).repeat_interleave(N, 0)
        past = actions[i:i + batch, -3:].argmax(-1)
        if bool((past >= N).any()):
            raise SystemExit("a context action is BOS: fewer than four real frames")
        a = torch.cat([past.repeat_interleave(N, 0), torch.arange(N).repeat(n)[:, None]], 1).to(device)
        with autocast_context(config):
            predicted, history = world(s, a)
            generated = predicted[:, 3]
            agent = world.agent(generated, history[:, 3])[:, None]
            out["p_dead"].append((1 - torch.sigmoid(heads(agent)["continuation"][:, 0, 0].float())).view(n, N).cpu())
            out["features"].append(agent[:, 0].mean(1).float().view(n, N, D).half().cpu())
            if health:
                out["dh_logits"].append(world.health(s[:, 3], generated).float().view(n, N, HEALTH).cpu())
        if successors is not None:
            rz, rt = encode(encoder, successors[i:i + batch], device)
            real = (rz[:, :, None] if state == "z" else rt.float()).to(device).flatten(2)
            g = generated.float().view(n, N, -1)
            gc, rc = g - g.mean(1, keepdim=True), real - real.mean(1, keepdim=True)
            out["error"].append(((gc - rc) ** 2).sum((1, 2)).cpu())
            out["energy"].append((rc ** 2).sum((1, 2)).cpu())
    return {k: torch.cat(v) for k, v in out.items() if v}


def score(device, log, recorded):
    from d4mj.agent import Heads
    from d4mj.lewm_diagnostics import FORK_STORE
    from boundary import judge_store
    from confirm import seeds_for
    from frozen_ladder import arm_input, scores, standardize, strata, train as probe_train
    from ladder import paired
    from observability import expected_safe, load
    draws, seed = 1000, 20260924

    fit_seeds, _ = seeds_for(json.loads((HERE / "evidence/root_partition.json").read_text()), FORK_STORE)
    fit = load(fit_seeds)["fit"]
    judge, manifest, files = judge_store(SEALED)
    rows = [r for f in sorted(SEALED.glob("seed-*.pt")) for r in torch.load(f, weights_only=False)]
    judge["successors"] = torch.stack([r["successors"] for r in rows])
    judge["health_delta"] = torch.stack([r["health_delta"] for r in rows]).round().long()
    del rows
    new = set(judge["seed"].unique().tolist())
    used = {int(f.stem.split("-")[1]) for store in ("observe_fresh_v1", "observe_fresh_v2", "observe_fresh_v3")
            for f in (ROOT / "artifacts/eda" / store).glob("seed-*.pt")}
    if min(new) < 53_000 or new & used or new & set(fit_seeds):
        raise SystemExit("judgement seeds are not a new, untouched block")
    boundary = json.loads((HERE / "evidence/boundary_v4.json").read_text())
    if boundary["judge_manifest"] != manifest:
        raise SystemExit("boundary_v4 was not run on this block")

    encoder, config = bridge()
    worlds = {}
    for arm, (state, health) in ARMS.items():
        stored = torch.load(WORLDS / f"{arm}.pt", map_location="cpu", weights_only=False)
        if stored["pool_sha256"] != recorded["pool_sha256"] or stored["script_sha256"] != _sha256(Path(__file__)):
            raise SystemExit(f"{arm} was not trained by this script on this pool")
        world, heads = World(1 if state == "z" else TOKENS, health).to(device), Heads(config).to(device)
        world.load_state_dict(stored["world"])
        heads.load_state_dict(stored["heads"])
        world.eval(), heads.eval()
        worlds[arm] = {"judge": branches(world, heads, encoder, config, judge["frames"], judge["actions"], device,
                                         state, health, judge["successors"]),
                       "fit": branches(world, heads, encoder, config, fit["frames"], fit["actions"], device, state,
                                       health),
                       "history": stored["history"]}
        log(stage="branches", arm=arm)
    with torch.no_grad():
        for data in (fit, judge):
            data["tokens1"] = torch.cat([encoder._hidden(data["frames"][i:i + 64, -1:].to(device))[2].cpu()
                                         for i in range(0, len(data["frames"]), 64)])
    del encoder
    torch.cuda.empty_cache()

    groups = fit["seed"].unique()
    held = set(groups[torch.randperm(len(groups), generator=torch.Generator().manual_seed(seed))
                      [:max(1, round(0.15 * len(groups)))]].tolist())
    inner = torch.tensor([int(s) in held for s in fit["seed"]])
    train_rows, hold_rows = torch.where(~inner)[0], torch.where(inner)[0]
    judge_rows = torch.arange(len(judge["seed"]))
    pf, pj, seeds = fit["p_death1"], judge["p_death1"], judge["seed"]
    opp_fit = (pf.amax(1) > pf.amin(1)) & ~inner
    prior = int(pf[opp_fit].mean(0).argmin())
    safe = {"prior": expected_safe(-F.one_hot(torch.full((len(pj),), prior), N).float(), pj)[0]}
    _, opp = expected_safe(pj, pj)

    def probe(kind, xf, xj):
        mean, scale = standardize(xf, train_rows)
        xf, xj = (xf.float() - mean) / scale, (xj.float() - mean) / scale
        runs = []
        for probe_seed in range(3):
            model, _ = probe_train(kind, xf.shape[1:], xf, pf, train_rows, hold_rows, seed=probe_seed,
                                   device=device, steps=3000)
            runs.append(expected_safe(scores(model, xj, judge_rows, device), pj)[0])
        return torch.stack(runs).mean(0)

    _, actions_fit, _ = arm_input(fit, "actions_only", None)
    _, actions_judge, _ = arm_input(judge, "actions_only", None)
    safe["actions_only"] = probe("vector", actions_fit, actions_judge)
    safe["root_tokens"] = probe("tokens_attn", fit["tokens1"], judge["tokens1"])
    risk = {}
    for arm, w in worlds.items():
        risk[arm] = w["judge"]["p_dead"]
        safe[arm] = expected_safe(risk[arm], pj)[0]
        safe[f"{arm}_fresh_probe"] = probe("branch", w["fit"]["features"], w["judge"]["features"])
        log(stage="scored", arm=arm, trained=round(float(safe[arm][opp].mean()), 4),
            fresh_probe=round(float(safe[f"{arm}_fresh_probe"][opp].mean()), 4))

    strat = strata(judge["visible"])
    test = lambda a, b, mask: paired(safe[a][mask], safe[b][mask], seeds[mask], draws=draws, seed=seed + 13)
    resolved = lambda r: r["difference"] is not None and r["difference"] > 0 and r["excludes_zero"]
    zombie = opp & strat["zombie_adjacent"]
    brier = {arm: ((w["judge"]["p_dead"] - pj) ** 2).mean(1) for arm, w in worlds.items()}
    target = judge["health_delta"].clamp(-9, 1) + 9
    ce = {arm: F.cross_entropy(w["judge"]["dh_logits"].flatten(0, 1), target.flatten(), reduction="none")
          .view(-1, N).mean(1) for arm, w in worlds.items() if "dh_logits" in w["judge"]}
    effect = {arm: float(w["judge"]["error"][opp].sum() / w["judge"]["energy"][opp].sum()) for arm, w in worlds.items()}
    lower = lambda values, a, b: paired(values[b][opp], values[a][opp], seeds[opp], draws=draws, seed=seed + 17)

    p0 = bool(boundary["holds"]["R1_tokens_attn_vs_cls"])
    rules = {"S1_TH_vs_ZH_zombie": test("TH", "ZH", zombie),
             "S2_TH_vs_prior": test("TH", "prior", opp), "S2_TH_vs_actions_only": test("TH", "actions_only", opp),
             "S3_TH_vs_root_tokens": test("TH", "root_tokens", opp),
             "S4_brier_ZH_minus_TH": lower(brier, "TH", "ZH"), "S4_health_ce_ZH_minus_TH": lower(ce, "TH", "ZH"),
             "S5_effect_error": {"TH": effect["TH"], "T": effect["T"]}}
    s3 = rules["S3_TH_vs_root_tokens"]
    holds = {"S1": resolved(rules["S1_TH_vs_ZH_zombie"]),
             "S2": resolved(rules["S2_TH_vs_prior"]) and resolved(rules["S2_TH_vs_actions_only"]),
             "S3": not (s3["difference"] < 0 and s3["excludes_zero"]),
             "S4": resolved(rules["S4_brier_ZH_minus_TH"]) and resolved(rules["S4_health_ce_ZH_minus_TH"]),
             "S5": effect["TH"] <= 2 * effect["T"]}
    reading = ("void_no_root_advantage" if not p0 else "spatial_health_passes" if all(holds.values())
               else "spatial_health_fails")

    stay = opp & (pj[:, 0] > 0.5) & (pj[:, 1:5].amin(1) < 0.5)            # NOOP kills, a move survives
    names = ("prior", "actions_only", "root_tokens", *ARMS, *(f"{a}_fresh_probe" for a in ARMS))
    reported = {
        "expected_safe": {k: {"overall": float(safe[k][opp].mean()), "stay_kills_move_survives": float(safe[k][stay].mean()),
                              **{s: float(safe[k][m & opp].mean()) for s, m in strat.items() if (m & opp).any()}}
                          for k in names},
        "contrasts": {f"{a}_vs_{b}": {"overall": test(a, b, opp), "zombie": test(a, b, zombie),
                                      "stay_kills_move_survives": test(a, b, stay)}
                      for a, b in (("TH", "ZH"), ("TH", "T"), ("ZH", "Z"), ("T", "Z"), ("TH", "root_tokens"),
                                   ("ZH", "prior"), ("Z", "prior"), ("T", "prior"))},
        "interaction": float(((safe["TH"] - safe["T"]) - (safe["ZH"] - safe["Z"]))[opp].mean()),
        "brier": {a: float(v[opp].mean()) for a, v in brier.items()},
        "health_ce": {a: float(v[opp].mean()) for a, v in ce.items()},
        "effect_error": effect,
        "sleep_rated_safer_than_noop": {**{a: float((risk[a][opp, 6] < risk[a][opp, 0]).float().mean()) for a in ARMS},
                                        "truth": float((pj[opp, 6] < pj[opp, 0]).float().mean())},
        "chosen_histogram": {a: torch.bincount(risk[a][opp].argmin(1), minlength=N).tolist() for a in ARMS},
        "training_history": {a: w["history"] for a, w in worlds.items()}}
    evidence = {"schema": "d4mj_spatial_2x2_v1", "status": "SEALED: new seed block, rules committed before collection",
                "script_sha256": _sha256(Path(__file__)), "pool": recorded, "judge_store": str(SEALED),
                "judge_manifest": manifest, "judge_seed_files": files, "judge_seeds": [min(new), max(new), len(new)],
                "roots": {"fit": len(fit["seed"]), "judge": len(pj), "opportunity": int(opp.sum()),
                          "zombie_opportunity": int(zombie.sum()), "stay_kills_move_survives": int(stay.sum())},
                "prior_action": prior,
                "P0_root_patch_advantage": boundary["outcomes"]["death1"]["rules"]["R1_tokens_attn_vs_cls"],
                "rules": rules, "holds": holds, "reading": reading, "reported": reported}
    (HERE / "evidence/spatial.json").write_text(json.dumps(evidence, indent=2) + "\n")
    log(status="spatial_complete", reading=reading, holds=holds)
    return 0



if __name__ == "__main__":
    raise SystemExit(main())
