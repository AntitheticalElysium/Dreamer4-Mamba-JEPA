"""Recipe preflight and joint-training orchestration for the shared D4MJ package."""

import argparse
from dataclasses import replace
import json
from pathlib import Path

from .data import atomic_manifest, _sha256
from .data import load_joint_corpus
from .cache import cache_latents_to_store
from .cache import load_latent_cache
from .config import load_recipe, recipe_dict, recipe_digest, config_from_dict
from .lewm_config import LeWMConfig, ScreenConfig
from .gates import preflight, require_joint_gates, ComponentGateError
from .train import train_joint, initialize_joint, train_bridge, train_actor_lewm
from .world_api import load_bundle
from .world_api import ModelBundle
from .checkpoint import read_lewm_bundle, read_lewm_bridge
from .agent import Heads


def record_joint_preflight(config, episodes, contract, dataset, out):
    """One artifact writer and gate runner for standalone and paired launches."""
    atomic_manifest(out / "resolved_recipe.json", recipe_dict(config))
    paths = [dataset] if isinstance(dataset, (str, Path)) else list(dataset)
    atomic_manifest(out / "dataset.json", {
        "paths": [str(Path(path).resolve()) for path in paths], "contract": contract})
    atomic_manifest(out / "dataset_audit.json", contract["audit"])
    atomic_manifest(out / "baseline_manifest.json", {
        "scope": "M0-M4 recipe and mechanics; empirical phase gates still control continuation",
        "legacy": {"family": "MAE64x16 Flow|Direct x Attention|Mamba", "status": "retained_in_shared_runtime"},
        "new_comparison": "raw versus temporally centered SIGReg; paired initial weights and sampler seeds",
        "external_dreamerv3": {"status": "unresolved", "claim_authorized": False,
            "required": ["same Craftax variant", "metric", "training access", "compute", "evaluation protocol"]},
        "collector_training_access": contract["collector_training_access"],
    })
    report = preflight(config, episodes, contract)
    atomic_manifest(out / "gates.json", report)
    if "sources" in report:
        atomic_manifest(out / "source_manifest.json", report["sources"])
    require_joint_gates(report, config, contract)
    return report


def run_joint_pair(configs, settings, dataset, output, *, screen_only=False, resume=False):
    """G0 -> saved initialization -> paired 2k -> G1 -> accepted finite budget."""
    import gc
    import os
    import time
    import torch
    from .gates import contract_digest
    from .lewm_diagnostics import screen_joint_pair
    from .data import screen_windows

    if not isinstance(settings,ScreenConfig) or not all(isinstance(c,LeWMConfig) for c in configs.values()):
        raise ValueError("paired-run requires two model recipes and a screen recipe")
    if set(configs) != {"raw","tc"}:
        raise ComponentGateError("pair_recipe", "paired recipes are keyed raw and tc")
    from .lewm_config import pair_axis as declared_axis
    try:
        pair_axis = declared_axis(*(recipe_dict(c) for c in configs.values()))
    except ValueError as error:
        raise ComponentGateError("pair_recipe", str(error)) from error
    if pair_axis == "variant" and any(c.variant != v for v,c in configs.items()):
        raise ComponentGateError("pair_recipe", "a variant pair must name its arms raw and tc")
    output = Path(output)
    existing = output.exists() and any(output.iterdir())
    if existing and not resume:
        raise ValueError("run_output: paired run requires --resume for an existing directory")
    output.mkdir(parents=True, exist_ok=True)
    def status(stage, **detail):
        row = {"stage":stage,"pid":os.getpid(),"updated_unix":time.time(),"m4_authorized":False,**detail}
        atomic_manifest(output/"status.json",row)
        print(json.dumps(row),flush=True)
    if existing:
        if json.loads((output/"screen_recipe.json").read_text()) != recipe_dict(settings):
            raise ComponentGateError("pair_recipe", "resume changed the screen recipe")
    else:
        atomic_manifest(output/"screen_recipe.json",recipe_dict(settings))
    # The arm directories stay raw/tc for tooling; this records what they hold.
    families = {c.family for c in configs.values()}
    backend_scope = {"families": sorted(families), "comparison_only": True,
                     "thesis_architecture": "lewm_mamba",
                     "note": "a source-exact predictor package is a control; its ~4.9x parameter "
                             "count means a win implicates capacity, conditioning and mixer together",
                     "m4_authorized": False}
    if not existing:
        atomic_manifest(output/"backend_scope.json", backend_scope)
        atomic_manifest(output/"pair_axis.json",
                    {"axis":pair_axis, "arms":{k:{"variant":c.variant,
                                                  "centering":c.joint.centering,
                                                  "centering_stride":c.joint.centering_stride}
                                               for k,c in configs.items()}})
        for variant,c in configs.items():
            atomic_manifest(output/f"{variant}_recipe.json",recipe_dict(c))
    else:
        for variant,c in configs.items():
            if json.loads((output/f"{variant}_recipe.json").read_text()) != recipe_dict(c):
                raise ComponentGateError("pair_recipe", f"resume changed the {variant} recipe")
    status("dataset_validation")
    episodes, contract = load_joint_corpus(dataset,configs["raw"])
    paths = [dataset] if isinstance(dataset, (str, Path)) else list(dataset)
    dataset_record = {"paths":[str(Path(path).resolve()) for path in paths],"contract":contract}
    if existing:
        if json.loads((output/"dataset.json").read_text()) != dataset_record:
            raise ComponentGateError("dataset_identity", "resume changed the paired dataset")
    else:
        atomic_manifest(output/"dataset.json", dataset_record)
    # Fail coverage before spending the joint budget; no model features are inspected.
    for split in ("train","dev"):
        screen_windows(episodes,configs["raw"],settings,split)
    runs = {v:output/v for v in configs}
    reports, initial = {}, {}
    for variant,c in configs.items():
        if existing:
            reports[variant] = json.loads((runs[variant]/"gates.json").read_text())
            require_joint_gates(reports[variant], c, contract)
            initial[variant] = read_lewm_bundle(runs[variant]/"joint/step-000000.pt")["initial_identity"]
        else:
            status("preflight",variant=variant)
            runs[variant].mkdir()
            reports[variant] = record_joint_preflight(c,episodes,contract,paths,runs[variant])
            _,initial[variant] = initialize_joint(episodes,c,runs[variant]/"joint",dataset_contract=contract,gate_report=reports[variant])
        gc.collect(); torch.cuda.empty_cache()
    if initial["raw"] != initial["tc"]:
        raise ComponentGateError("initial_identity", "paired initial weights differ")
    pair = {"schema":"d4mj_joint_pair_v1","dataset_id":contract_digest(contract),
            "initial_identity":initial,"screen_settings_id":recipe_digest(settings),
            "recipes":{v:recipe_digest(c) for v,c in configs.items()},"m4_authorized":False}
    if existing:
        if json.loads((output/"pair.json").read_text()) != pair:
            raise ComponentGateError("pair_identity", "resume changed the pre-training pair seal")
    else:
        atomic_manifest(output/"pair.json", pair)

    def latest(variant):
        path = runs[variant]/"joint/latest.pt"
        if not path.exists():
            path = runs[variant]/"joint/step-000000.pt"
        return path.resolve()

    for variant,c in configs.items():
        checkpoint = latest(variant)
        step = read_lewm_bundle(checkpoint)["step"]
        if step < c.joint.screen_step:
            status("joint_to_screen",variant=variant,target_update=c.joint.screen_step,resume_step=step)
            bundle,_ = train_joint(episodes,c,runs[variant]/"joint",dataset_contract=contract,
                                   gate_report=reports[variant], stop_at=c.joint.screen_step,
                                   resume=checkpoint)
            del bundle; gc.collect(); torch.cuda.empty_cache()
    report_path = output/"G1/screen.json"
    if report_path.exists():
        report = json.loads(report_path.read_text())
    else:
        status("G1")
        report = screen_joint_pair(runs,episodes,contract,settings,output/"G1")
    status("G1_complete",decision=report["decision"],component=report.get("blocked_component"))
    if report["decision"] != "continue_joint_budget":
        return 1
    if screen_only:
        return 0
    for variant,c in configs.items():
        checkpoint = latest(variant)
        step = read_lewm_bundle(checkpoint)["step"]
        if step == c.joint.steps:
            continue
        if step < c.joint.screen_step:
            raise ComponentGateError("joint_screen_parent", "arm still precedes G1 after resume")
        status("joint_to_budget",variant=variant,target_update=c.joint.steps,resume_step=step)
        bundle,_ = train_joint(episodes,c,runs[variant]/"joint",dataset_contract=contract,gate_report=reports[variant],
                               stop_at=c.joint.steps,resume=checkpoint,
                               screen_report=report)
        del bundle; gc.collect(); torch.cuda.empty_cache()
    status("joint_budget_complete",next_component="M4",next_component_status="blocked")
    return 0


def _read_json(path: Path | None):
    return None if path is None else json.loads(Path(path).read_text())


def _cache_for_checkpoint(cache_path: Path, bundle: ModelBundle, checkpoint_path: Path,
                          parent: dict):
    """Load the frozen cache only after binding its bytes, encoder and joint parent."""
    manifest_path = cache_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    cache = manifest.get("cache", {})
    parent_hash = _sha256(checkpoint_path)
    if cache.get("parent_checkpoint") != parent_hash:
        raise ComponentGateError("cache_parent", "latent cache belongs to another joint checkpoint")
    if cache.get("dataset_contract") != parent["dataset"]:
        raise ComponentGateError("cache_dataset", "latent cache belongs to another dataset contract")
    episodes = load_latent_cache(cache_path, bundle.encoder, bundle.config)
    contract = {
        "schema": "d4mj_lewm_cache_identity_v1", "path": str(cache_path.resolve()),
        "manifest_sha256": _sha256(manifest_path), "parent_checkpoint": parent_hash,
        "latent_digest": cache.get("latent_digest"), "dataset": parent["dataset"],
        "episodes": manifest.get("episodes"), "transitions": manifest.get("transitions"),
    }
    return episodes, contract


def _load_joint_parent(checkpoint: Path):
    payload = read_lewm_bundle(checkpoint)
    config = config_from_dict(payload["config"])
    bundle = ModelBundle.create(config)
    bundle.encoder.load_state_dict(payload["modules"]["encoder"], strict=True)
    bundle.world.load_state_dict(payload["modules"]["world"], strict=True)
    return bundle, payload


def _load_bridge_parent(checkpoint: Path):
    payload = read_lewm_bridge(checkpoint)
    config = config_from_dict(payload["config"])
    bundle = ModelBundle.create(config)
    bundle.encoder.load_state_dict(payload["modules"]["encoder"], strict=True)
    bundle.world.load_state_dict(payload["modules"]["world"], strict=True)
    bundle.capabilities = dict(payload["capabilities"])
    heads = Heads(config).to(config.runtime.device)
    heads.load_state_dict(payload["modules"]["heads"], strict=True)
    return bundle, heads, payload


def _load_actor_parent(checkpoint: Path):
    """The actor screen snapshot: frozen world, trained heads, and the immutable BC prior."""
    from .checkpoint import read_lewm_actor
    payload = read_lewm_actor(checkpoint)
    config = config_from_dict(payload["config"])
    bundle = ModelBundle.create(config)
    bundle.encoder.load_state_dict(payload["modules"]["encoder"], strict=True)
    bundle.world.load_state_dict(payload["modules"]["world"], strict=True)
    bundle.capabilities = dict(payload["capabilities"])
    bundle.encoder.freeze()
    bundle.world.eval()
    heads = Heads(config).to(config.runtime.device)
    heads.load_state_dict(payload["modules"]["heads"], strict=True)
    prior = Heads(config).to(config.runtime.device)
    prior.load_state_dict(payload["modules"]["prior"], strict=True)
    return bundle, heads.eval(), prior.eval(), payload


def _cache_from_contract(cache_path: Path, bundle: ModelBundle, expected: dict):
    manifest_path = cache_path / "manifest.json"
    if (str(cache_path.resolve()) != expected.get("path")
            or _sha256(manifest_path) != expected.get("manifest_sha256")):
        raise ComponentGateError("cache_identity", "bridge cache path or manifest bytes changed")
    return load_latent_cache(cache_path, bundle.encoder, bundle.config), expected


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("preflight")
    p.add_argument("--recipe", type=Path, required=True)
    p.add_argument("--dataset", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", choices=("cpu", "cuda"))
    p.add_argument("--backend", choices=("reference", "triton", "sdpa"))
    p.add_argument("--verification", action="store_true", help="non-research contract, does not certify GPU fit")
    p = sub.add_parser("joint")
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--stop-at", type=int)
    p.add_argument("--resume", type=Path)
    p.add_argument("--screen-report", type=Path)
    p = sub.add_parser("export")
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--diagnostic", action="store_true", help="allow an explicitly partial joint checkpoint")
    p = sub.add_parser("paired-run", help="paired joint run with a sealed G1 stop")
    p.add_argument("--dataset", type=Path, nargs="+", required=True)
    p.add_argument("--out", type=Path, required=True)
    recipes = Path(__file__).with_name("recipes")
    p.add_argument("--raw-recipe", type=Path, default=recipes/"lewm_mamba_raw_m4.json")
    p.add_argument("--tc-recipe", type=Path, default=recipes/"lewm_mamba_tc_m4.json")
    p.add_argument("--screen-recipe", type=Path, default=recipes/"joint_screen.json")
    p.add_argument("--screen-only", action="store_true", help="pause after G1 even if it passes")
    p.add_argument("--resume", action="store_true", help="continue the exact sealed pair in place")
    p = sub.add_parser("bridge", help="canonical H2/H16 recursive bridge")
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--cache", type=Path)
    p.add_argument("--out", type=Path)
    p.add_argument("--stop-after", choices=("h2", "h16"), default="h2")
    p.add_argument("--resume", type=Path)
    p.add_argument("--gate", type=Path, help="identity-bound H2 report required for H16")
    p = sub.add_parser("actor", help="canonical frozen-world imagination actor")
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--bridge", type=Path)
    p.add_argument("--cache", type=Path)
    p.add_argument("--out", type=Path)
    p.add_argument("--bridge-gate", type=Path, required=True)
    p.add_argument("--stop-after", choices=("screen", "budget"), default="screen")
    p.add_argument("--resume", type=Path)
    p.add_argument("--actor-gate", type=Path, help="identity-bound screen report for full budget")
    p = sub.add_parser("gate", help="measure a sealed G2/G3/G4 phase report")
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--stage", choices=("h2", "h16", "actor"), required=True)
    p.add_argument("--dataset", type=Path, nargs="+",
                   help="raw corpus; semantic retention compares projected z against CLS and "
                        "cannot read that from the projected-only latent cache")
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--cache", type=Path)
    p.add_argument("--out", type=Path)
    p.add_argument("--screen-recipe", type=Path, default=recipes/"joint_screen.json")
    p = sub.add_parser("evaluate", help="real Craftax actor versus its immutable own BC")
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--actor", type=Path)
    p.add_argument("--out", type=Path)
    p.add_argument("--episodes", type=int)
    p.add_argument("--limit", type=int)
    for stage in ("render-fit", "play"):
        sub.add_parser(stage, help="not enabled before M6")
    args = parser.parse_args(argv)
    if args.command in ("render-fit", "play"):
        parser.error(f"phase_gate: {args.command} requires M6 or later")
    destination = None
    try:
        if args.command == "paired-run":
            configs = {v:load_recipe(getattr(args, v+"_recipe")) for v in ("raw","tc")}
            settings = load_recipe(args.screen_recipe)
            destination = args.out
            dataset = args.dataset[0] if len(args.dataset) == 1 else args.dataset
            return run_joint_pair(configs,settings,dataset,args.out,screen_only=args.screen_only,
                                  resume=args.resume)
        if args.command == "preflight":
            if args.out.exists() and any(args.out.iterdir()):
                raise ValueError("run_output: preflight requires a fresh directory")
            config = load_recipe(args.recipe)
            if not isinstance(config, LeWMConfig):
                if args.backend or args.verification or args.dataset:
                    raise ValueError("legacy preflight takes its mixer from the recipe and uses synthetic gate inputs")
                config = replace(config, device=args.device or config.device)
                args.out.mkdir(parents=True, exist_ok=True)
                destination = args.out
                atomic_manifest(args.out / "resolved_recipe.json", recipe_dict(config))
                report = preflight(config)
                atomic_manifest(args.out / "gates.json", report)
                failed = [name for name, result in report["components"].items() if result["status"] != "pass"]
                if failed:
                    raise ComponentGateError(failed[0], report["components"][failed[0]]["reason"])
                print(json.dumps({"stage": "Stage-A preflight", "status": "pass", "recipe_id": recipe_digest(config)}))
                return 0
            if args.dataset is None:
                raise ValueError("joint preflight requires --dataset")
            if args.backend:
                # A backend belongs to one family; selecting the other family's kernel
                # would build a recipe that cannot run.
                allowed = ("sdpa",) if config.family == "lewm_transformer" else ("reference", "triton")
                if args.backend not in allowed:
                    raise ValueError(f"{config.family} supports backends {allowed}")
            config = replace(config,
                             runtime=replace(config.runtime, device=args.device or config.runtime.device,
                                             purpose="verification" if args.verification else config.runtime.purpose),
                             dynamics=replace(config.dynamics, backend=args.backend or config.dynamics.backend))
            episodes, contract = load_joint_corpus(args.dataset, config)
            args.out.mkdir(parents=True, exist_ok=True)
            destination = args.out
            record_joint_preflight(config, episodes, contract, args.dataset, args.out)
            print(json.dumps({"stage": "M0-M3 preflight", "status": "pass", "recipe_id": recipe_digest(config),
                              "m4_authorized": False}))
            return 0
        destination = args.run
        if args.command == "bridge":
            arm_recipe = config_from_dict(json.loads((args.run / "resolved_recipe.json").read_text()))
            checkpoint = args.checkpoint or args.run / "joint" / f"step-{arm_recipe.joint.steps:06d}.pt"
            cache_path = args.cache or args.run / "cache"
            output = args.out or args.run / "bridge"
            destination = output
            bundle, parent = _load_joint_parent(checkpoint)
            if bundle.config.agent is None:
                raise ComponentGateError("phase_gate", "joint parent uses an M0-M3 recipe without M4 settings")
            episodes, cache_contract = _cache_for_checkpoint(cache_path, bundle, checkpoint, parent)
            train_bridge(
                episodes, bundle, parent, output, parent_path=checkpoint,
                cache_contract=cache_contract, stop_after=args.stop_after, resume=args.resume,
                gate_report=_read_json(args.gate),
            )
            return 0
        if args.command == "actor":
            arm_recipe = config_from_dict(json.loads((args.run / "resolved_recipe.json").read_text()))
            settings = arm_recipe.agent
            if settings is None:
                raise ComponentGateError("phase_gate", "run recipe has no M4 settings")
            bridge_path = args.bridge or args.run / "bridge" / f"step-{settings.h2_steps + settings.h16_steps:06d}.pt"
            cache_path = args.cache or args.run / "cache"
            output = args.out or args.run / "actor"
            destination = output
            bundle, heads, bridge_payload = _load_bridge_parent(bridge_path)
            episodes, cache_contract = _cache_from_contract(cache_path, bundle, bridge_payload["cache"])
            train_actor_lewm(
                episodes, bundle, heads, bridge_payload, output, bridge_path=bridge_path,
                cache_contract=cache_contract, bridge_gate=_read_json(args.bridge_gate),
                stop_after=args.stop_after, resume=args.resume,
                actor_gate=_read_json(args.actor_gate),
            )
            return 0
        if args.command == "gate":
            from .lewm_diagnostics import actor_gate, bridge_gate
            arm_recipe = config_from_dict(json.loads((args.run / "resolved_recipe.json").read_text()))
            settings = arm_recipe.agent
            if settings is None:
                raise ComponentGateError("phase_gate", "run recipe has no M4 settings")
            screen = load_recipe(args.screen_recipe)
            cache_path = args.cache or args.run / "cache"
            output = args.out or args.run / "gates" / args.stage
            destination = output
            raw = None
            if args.dataset:
                raw, _ = load_joint_corpus(args.dataset, arm_recipe)
            if args.stage == "actor":
                checkpoint = args.checkpoint or args.run / "actor" / f"step-{settings.actor_screen_steps:06d}.pt"
                bundle, heads, prior, payload = _load_actor_parent(checkpoint)
                episodes, cache_contract = _cache_from_contract(cache_path, bundle, payload["cache"])
                report = actor_gate(bundle, heads, prior, payload, episodes, cache_contract,
                                    screen, output, checkpoint=checkpoint)
            else:
                steps = settings.h2_steps if args.stage == "h2" else settings.h2_steps + settings.h16_steps
                checkpoint = args.checkpoint or args.run / "bridge" / f"step-{steps:06d}.pt"
                bundle, heads, payload = _load_bridge_parent(checkpoint)
                episodes, cache_contract = _cache_from_contract(cache_path, bundle, payload["cache"])
                report = bridge_gate(bundle, heads, payload, episodes, cache_contract, screen,
                                     output, stage=args.stage, checkpoint=checkpoint,
                                     raw_episodes=raw)
            failed = [n for n, c in report["components"].items() if c["status"] != "pass"]
            print(json.dumps({"stage": args.stage, "decision": report["decision"],
                              "validated_recursive_depth": report["validated_recursive_depth"],
                              "failed_components": sorted(failed),
                              "report": str(output / (f"bridge_gate_{args.stage}.json"
                                                      if args.stage != "actor" else "actor_gate.json"))}))
            return 0 if not failed else 1
        if args.command == "evaluate":
            from .execution import evaluate_lewm_actor
            arm_recipe = config_from_dict(json.loads((args.run / "resolved_recipe.json").read_text()))
            settings = arm_recipe.agent
            if settings is None:
                raise ComponentGateError("phase_gate", "run recipe has no M4 settings")
            actor_path = args.actor or args.run / "actor" / f"step-{settings.actor_steps:06d}.pt"
            output = args.out or args.run / "evaluation"
            destination = output
            report = evaluate_lewm_actor(actor_path, output, episodes=args.episodes, limit=args.limit)
            print(json.dumps({"status": "evaluation_complete", **report["primary"]}), flush=True)
            return 0
        config = config_from_dict(json.loads((args.run / "resolved_recipe.json").read_text()))
        if not isinstance(config, LeWMConfig):
            raise ComponentGateError("recipe_family", "joint/export require a LeWM recipe; legacy phase trainers remain in d4mj.train")
        recorded = json.loads((args.run / "dataset.json").read_text())
        sources = recorded.get("paths", [recorded.get("path")])
        episodes, contract = load_joint_corpus(sources[0] if len(sources) == 1 else sources, config)
        if contract != recorded["contract"]:
            raise ComponentGateError("dataset_identity", "dataset changed after M0")
        gates = json.loads((args.run / "gates.json").read_text())
        require_joint_gates(gates, config, contract)
        if args.command == "joint":
            train_joint(episodes, config, args.run / "joint", dataset_contract=contract,
                        gate_report=gates, stop_at=args.stop_at, resume=args.resume,
                        screen_report=None if args.screen_report is None else json.loads(args.screen_report.read_text()))
        else:
            bundle, checkpoint = load_bundle(args.checkpoint)
            if checkpoint["recipe_id"] != recipe_digest(config) or checkpoint["dataset"] != contract:
                raise ComponentGateError("export_parent", "checkpoint does not belong to this run")
            if not checkpoint["capabilities"]["joint_complete"] and not args.diagnostic:
                raise ComponentGateError("joint_completion", "use --diagnostic for a partial export")
            cache_latents_to_store(bundle.encoder, episodes, config, args.out,
                                   source_contract=contract, parent_checkpoint=_sha256(args.checkpoint))
            manifest = json.loads((args.out / "manifest.json").read_text())
            manifest["export"] = {"joint_step": checkpoint["step"],
                                  "parent_checkpoint_path": str(args.checkpoint.resolve()),
                                  "diagnostic_only": not checkpoint["capabilities"]["joint_complete"]}
            atomic_manifest(args.out / "manifest.json", manifest)
        return 0
    except Exception as error:
        failure = {"status": "stopped", "component": getattr(error, "component", "lewm_execution"),
                   "reason": str(error), "architecture_verdict": "not_evaluated", "m4_authorized": False}
        if destination is not None and destination.exists():
            atomic_manifest(destination / "failure.json", failure)
        print(json.dumps(failure))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
