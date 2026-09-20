"""Stage 4: fresh Raw and TC from update zero, on the merged corpus, with Direct's fork mix.

This reproduces `experiments.run_joint_pair`'s paired protocol -- saved identical initialization,
paired 2k, G1, then the accepted budget -- rather than calling it, because the campaign changes two
things the sealed M0-M3 protocol should not learn about: the corpus has two sources, and the world
receives a separately declared counterfactual term. Keeping that here is the point. `joint_loss`
is untouched, so "LeWM" still means what it meant, and a run trained with `--fork-mass 0` is the
paper-minimal arm.

The fork term matches `train_terminal_arms.py`'s counterfactual contract, which is what Direct
received:

    loss = (1 - fork_mass) * joint + fork_mass * (branch + second_weight * second)

with `fork_roots` roots per update, all 17 successors off one encoder pass per root, and the second
NOOP successor scored only where `second_valid`. Mass 0.2, roots 4, second weight 0.5 -- Direct's
declared values, not retuned.

One deliberate choice worth naming: the fork targets are encoded by the LIVE encoder and their
gradient is NOT stopped. That is exactly how `joint_loss` treats `pairs[:, 1:]`, so the factual and
counterfactual terms are consistent, and SIGReg is what prevents collapse in both. Stopping the
gradient on one and not the other would make the two terms mean different things.
"""

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path("/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA")
HERE = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from d4mj.config import load_recipe, recipe_dict, recipe_digest
from d4mj.data import atomic_manifest, load_joint_corpus, screen_windows
from d4mj.gates import ComponentGateError, contract_digest, preflight, require_joint_gates
from d4mj.lewm_config import LeWMConfig, ScreenConfig, pair_axis as declared_axis
from d4mj.train import initialize_joint, train_joint

CORPUS = [ROOT / "artifacts/craftax_expert_store_v1", ROOT / "artifacts/craftax_support_v2"]
FORKS = HERE / "cache/forks"


class Forks:
    """The fork pool, mmap-backed, sampled a few roots at a time.

    Shards are mmap-loaded and indexed per batch so only the sampled roots' pages are touched;
    the pool is 6.8 GB of raw frames and joint training has no room to hold it resident.
    """

    def __init__(self, path: Path, generator: torch.Generator, train_only: bool = True):
        manifest = json.loads((path / "manifest.json").read_text())
        self.shards = [torch.load(path / s["file"], map_location="cpu", weights_only=False,
                                  mmap=True) for s in manifest["shards"]]
        self.rows = [torch.where(~s["held_out"])[0] if train_only
                     else torch.arange(len(s["frames"])) for s in self.shards]
        counts = torch.tensor([len(r) for r in self.rows], dtype=torch.float64)
        if not counts.sum():
            raise SystemExit("fork pool has no training roots")
        self.weights, self.generator = counts, generator
        self.roots = int(counts.sum())

    def sample(self, count: int):
        shard = int(torch.multinomial(self.weights, 1, generator=self.generator))
        rows = self.rows[shard]
        pick = rows[torch.randint(len(rows), (count,), generator=self.generator)]
        data = self.shards[shard]
        return {k: data[k][pick] for k in ("frames", "past_actions", "first", "second",
                                           "second_valid")}


def branch_term(bundle, batch, second_weight: float):
    """All-17 branch MSE from one prefilled root, plus the surviving second NOOP step.

    One encoder pass covers the window, every first successor and every second successor, so the
    cost is one forward rather than 35. Averaged per root before across roots, so a root cannot
    dominate by carrying more valid branches -- the normalization Direct uses.

    Every BatchNorm runs on its RUNNING statistics for this term, in the encoder's projector as
    well as the world's. `advance` refuses to stream otherwise, and its reason is the right one: a
    branch fan is 17 highly correlated samples of one root, and letting those define batch
    statistics would move the normalization that the factual stream -- a real B=128 draw -- is
    supposed to establish. Suppressing the update is honouring that guard, not evading it.
    Gradients still flow through the affine weights, and the term is then normalized exactly the
    way evaluation will normalize it.
    """
    encoder, world = bundle.encoder, bundle.world
    device = bundle.device
    frames, past = batch["frames"].to(device), batch["past_actions"].to(device)
    first, second = batch["first"].to(device), batch["second"].to(device)
    keep = batch["second_valid"].to(device)
    n, actions = first.shape[0], first.shape[1]

    frozen = [m for m in (*encoder.modules(), *world.modules())
              if isinstance(m, torch.nn.BatchNorm1d) and m.training]
    for module in frozen:
        module.eval()
    try:
        return _branch(bundle, frames, past, first, second, keep, n, actions, second_weight)
    finally:
        for module in frozen:
            module.train()


def _branch(bundle, frames, past, first, second, keep, n, actions, second_weight):
    encoder, world, device = bundle.encoder, bundle.world, bundle.device
    z_context = encoder(frames)
    z_first = encoder(first.flatten(0, 1).unsqueeze(1))[:, 0]
    z_second = encoder(second.flatten(0, 1).unsqueeze(1))[:, 0]

    state = world.start(z_context[:, :1])
    if z_context.shape[1] > 1:
        state = world.teacher(z_context, past, state=state).state
    flat = torch.arange(actions, device=device).repeat(n)[:, None]
    branches = bundle.repeat_state(state, actions)
    advanced, _ = bundle.advance(branches, flat)
    error = (advanced.latent[:, 0].float() - z_first.float()).square().mean(-1)
    loss = error.reshape(n, actions).mean(1).mean()

    if bool(keep.any()):
        noop = torch.zeros_like(flat)
        onward, _ = bundle.advance(advanced, noop)
        second_error = (onward.latent[:, 0].float() - z_second.float()).square().mean(-1)
        mask = keep.reshape(-1).float()
        loss = loss + second_weight * (second_error * mask).sum() / mask.sum().clamp(min=1.0)
    return loss


def supplement(bundle_forks, mass: float, second_weight: float, roots: int):
    """Build the `train_joint` callback that blends the declared fork term into the objective."""
    def extra(bundle, update, total):
        if mass <= 0:
            return total, {}
        term = branch_term(bundle, bundle_forks.sample(roots), second_weight)
        return (1 - mass) * total + mass * term, {"fork": float(term.detach())}
    return extra


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts/lewm_m4_paired")
    parser.add_argument("--raw-recipe", type=Path, default=HERE / "recipes/lewm_mamba_raw_m4.json")
    parser.add_argument("--tc-recipe", type=Path, default=HERE / "recipes/lewm_mamba_tc_m4.json")
    parser.add_argument("--screen-recipe", type=Path, default=ROOT / "d4mj/recipes/joint_screen.json")
    parser.add_argument("--forks", type=Path, default=FORKS)
    parser.add_argument("--corpus", type=Path, nargs="+", default=CORPUS)
    parser.add_argument("--stop-at", type=int, default=None)
    parser.add_argument("--screen-only", action="store_true")
    parser.add_argument("--smoke", type=int, default=0, help="updates for a plumbing check only")
    args = parser.parse_args(argv)

    configs = {"raw": load_recipe(args.raw_recipe), "tc": load_recipe(args.tc_recipe)}
    settings = load_recipe(args.screen_recipe)
    if not all(isinstance(c, LeWMConfig) for c in configs.values()) or not isinstance(settings, ScreenConfig):
        raise SystemExit("a paired run needs two LeWM recipes and a screen recipe")
    axis = declared_axis(*(recipe_dict(c) for c in configs.values()))
    if axis != "variant":
        raise SystemExit(f"the M4 pair isolates the regularizer target; found axis {axis}")
    for name, c in configs.items():
        if c.agent is None:
            raise SystemExit(f"{name}: an M4 recipe must declare `agent`")
        if c.variant != name:
            raise SystemExit("a variant pair names its arms raw and tc")

    args.out.mkdir(parents=True, exist_ok=True)
    def status(stage, **detail):
        row = {"stage": stage, "updated_unix": time.time(), **detail}
        atomic_manifest(args.out / "status.json", row)
        print(json.dumps(row), flush=True)

    status("dataset_validation", sources=[str(p) for p in args.corpus])
    episodes, contract = load_joint_corpus(args.corpus, configs["raw"])
    atomic_manifest(args.out / "dataset.json",
                    {"sources": [str(Path(p).resolve()) for p in args.corpus], "contract": contract})
    for split in ("train", "dev"):
        screen_windows(episodes, configs["raw"], settings, split)

    agent = configs["raw"].agent
    atomic_manifest(args.out / "campaign.json", {
        "schema": "d4mj_m4_campaign_v1",
        "corpus": "expert archive + support-v2, splits and eligibility preserved per source",
        "bc_eligible_note": "only the archive is BC-eligible; the world sees both, BC sees the archive",
        "fork_contract": {"mass": agent.fork_mass, "roots": agent.fork_roots,
                          "second_weight": agent.fork_second_weight,
                          "source": "artifacts/eda/broad_forks_v2 via fork_pool.py",
                          "matched_to": "train_terminal_arms.py --arm counterfactual --roots v2"},
        "objective_note": "joint_loss is unchanged; the fork term is blended by train_joint's "
                          "`extra` hook and declared here, so this is our LeWM-Mamba candidate "
                          "under the corrected Direct data contract, not paper-minimal LeWM",
        "dataset_id": contract_digest(contract),
        "recipes": {v: recipe_digest(c) for v, c in configs.items()}})

    runs = {v: args.out / v for v in configs}
    reports, initial = {}, {}
    for variant, c in configs.items():
        status("preflight", variant=variant)
        runs[variant].mkdir(exist_ok=True)
        atomic_manifest(runs[variant] / "resolved_recipe.json", recipe_dict(c))
        report = preflight(c, episodes, contract)
        atomic_manifest(runs[variant] / "gates.json", report)
        require_joint_gates(report, c, contract)
        reports[variant] = report
        _, initial[variant] = initialize_joint(episodes, c, runs[variant] / "joint",
                                               dataset_contract=contract, gate_report=report)
        gc.collect(); torch.cuda.empty_cache()
    if initial["raw"] != initial["tc"]:
        raise ComponentGateError("initial_identity", "paired initial weights differ")
    atomic_manifest(args.out / "pair.json", {"initial_identity": initial, "axis": axis})
    status("paired_initialization", identity=initial["raw"])

    screen_step = configs["raw"].joint.screen_step
    target = args.smoke or args.stop_at or screen_step
    for variant, c in configs.items():
        # Each arm draws its fork roots from its OWN stream, seeded identically, so the two arms
        # see the same roots in the same order and differ only on the declared axis.
        forks = Forks(args.forks, torch.Generator().manual_seed(c.seed + 991))
        status("joint_to_screen", variant=variant, target_update=target, fork_roots=forks.roots)
        bundle, _ = train_joint(episodes, c, runs[variant] / "joint", dataset_contract=contract,
                                gate_report=reports[variant], stop_at=target,
                                resume=runs[variant] / "joint/step-000000.pt",
                                extra=supplement(forks, c.agent.fork_mass,
                                                 c.agent.fork_second_weight, c.agent.fork_roots))
        del bundle, forks
        gc.collect(); torch.cuda.empty_cache()
    status("screen_budget_complete", target=target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
