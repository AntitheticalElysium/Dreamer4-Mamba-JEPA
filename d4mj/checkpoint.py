from dataclasses import asdict
from pathlib import Path
import os
import tempfile

import torch

from .config import Config
from .sources import source_digests, verify_sources

FORMAT = "d4mj_checkpoint_v1"


def save(path: Path, config: Config, **objects) -> None:
    """Atomic, and carrying enough to prove what produced it and to resume it: the
    config, the digests of every pinned source a decision rests on, and any
    plain-dict state -- the running-RMS normalisers and, via
    `train.generator_state`, the sampler and model generators that actually drive
    training. The global stream is stored too, but nothing here draws from it."""
    payload = {
        "format": FORMAT,
        "config": asdict(config),
        "sources": source_digests(config),
        "rng": torch.get_rng_state(),
        "modules": {
            name: value.state_dict() if hasattr(value, "state_dict") else value
            for name, value in objects.items()
        },
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.rename(path)


def load(path: Path, config: Config, **objects) -> dict:
    """Restores modules and plain dicts in place."""
    payload = torch.load(path, weights_only=False)
    if payload["format"] != FORMAT:
        raise ValueError(f"expected {FORMAT}, found {payload['format']}")
    stored, requested = payload["config"], asdict(config)
    # A field added after a checkpoint was written is absent from its stored config, and
    # a whole-dict comparison then rejects every older checkpoint. Fields listed here
    # take their default when missing -- and only when the caller is asking for that
    # default, so a checkpoint that never trained with alignment cannot be loaded as
    # though it had.
    for field, default in (("align_weight", 0.0),):
        if field not in stored:
            if requested.get(field, default) != default:
                raise ValueError(
                    f"checkpoint predates `{field}` and cannot be loaded with "
                    f"{field}={requested[field]}"
                )
            stored = stored | {field: default}
    if stored != requested:
        raise ValueError("checkpoint config differs from the one requested")
    verify_sources(payload["sources"], config)
    torch.set_rng_state(payload["rng"])
    for name, target in objects.items():
        stored = payload["modules"][name]
        if hasattr(target, "load_state_dict"):
            target.load_state_dict(stored)
        else:
            target.clear()
            target.update(stored)
    return payload


LEWM_FORMAT = "d4mj_lewm_bundle_v2"
LEWM_BRIDGE_FORMAT = "d4mj_lewm_bridge_v1"
LEWM_ACTOR_FORMAT = "d4mj_lewm_actor_v1"


def _save_immutable(path: Path, payload: dict) -> None:
    """Publish numbered research state once; movable aliases are managed separately."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        torch.save(payload, temporary)
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def save_lewm_bundle(path, bundle, *, step: int, dataset_contract: dict, initial_identity: dict,
                     optimizer, sampler, projection_rng, gate_report: dict) -> dict:
    """Resumable joint encoder AND predictor; no new phase is implied by a save."""
    from .config import recipe_dict, recipe_digest
    from .sources import lewm_source_manifest

    if not 0 <= step <= bundle.config.joint.steps:
        raise ValueError("checkpoint_schedule: step outside the declared joint schedule")
    modules = {"encoder": bundle.encoder, "world": bundle.world}
    payload = {
        "format": LEWM_FORMAT, "phase": "joint", "step": step,
        "config": recipe_dict(bundle.config), "recipe_id": recipe_digest(bundle.config),
        "sources": lewm_source_manifest(bundle.config), "dataset": dataset_contract,
        "initial_identity": initial_identity, "gates": gate_report,
        "modules": {k: v.state_dict() for k, v in modules.items()},
        "modes": {k: {n: m.training for n, m in v.named_modules()} for k, v in modules.items()},
        "requires_grad": {k: {n: p.requires_grad for n, p in v.named_parameters()} for k, v in modules.items()},
        "encoder_frozen": bundle.encoder._frozen,
        "optimizer": optimizer.state_dict(), "sampler": sampler.state_dict(),
        "projection_rng": projection_rng.get_state(), "cpu_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        # Scheduler is a pure function of completed optimizer updates + sealed config.
        "scheduler": {"type": "linear_warmup_cosine_v1", "completed_updates": step},
        "capabilities": {"joint_complete": step == bundle.config.joint.steps,
                         "trained_recursive_depth": 0, "validated_recursive_depth": 0,
                         "readout_trained": False, "m4_authorized": False},
    }
    _save_immutable(Path(path), payload)
    return payload


def publish_lewm_latest(path) -> None:
    """Index a preserved snapshot by hash and atomically move the convenience link."""
    import json
    from .data import atomic_manifest, _sha256

    path = Path(path)
    index_path = path.parent / "checkpoints.json"
    index = json.loads(index_path.read_text()) if index_path.exists() else {
        "schema": "d4mj_lewm_checkpoints_v1", "snapshots": {}}
    digest = _sha256(path)
    if path.name in index["snapshots"] and index["snapshots"][path.name] != digest:
        raise ValueError("checkpoint_identity: immutable snapshot changed")
    index["snapshots"][path.name] = digest
    index["latest"] = path.name
    atomic_manifest(index_path, index)
    temporary = path.parent / "latest.pt.tmp"
    temporary.unlink(missing_ok=True)
    temporary.symlink_to(path.name)
    temporary.replace(path.parent / "latest.pt")


def read_lewm_bundle(path) -> dict:
    from .config import config_from_dict, recipe_digest
    from .sources import verify_lewm_sources

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != LEWM_FORMAT or payload.get("phase") != "joint":
        raise ValueError("checkpoint_family: expected a v2 LeWM joint bundle")
    config = config_from_dict(payload["config"])
    if recipe_digest(config) != payload["recipe_id"]:
        raise ValueError("checkpoint_recipe: payload recipe digest mismatch")
    if (type(payload["step"]) is not int or not 0 <= payload["step"] <= config.joint.steps
            or payload["scheduler"] != {"type": "linear_warmup_cosine_v1", "completed_updates": payload["step"]}):
        raise ValueError("checkpoint_schedule: inconsistent completed updates")
    capabilities = {"joint_complete": payload["step"] == config.joint.steps,
                    "trained_recursive_depth": 0, "validated_recursive_depth": 0,
                    "readout_trained": False, "m4_authorized": False}
    if payload.get("capabilities") != capabilities:
        raise ValueError("checkpoint_phase: unsupported capabilities in an M0-M3 bundle")
    verify_lewm_sources(payload["sources"], config)
    return payload


def restore_lewm_bundle(path, bundle, *, dataset_contract, optimizer, sampler, projection_rng) -> dict:
    from .config import recipe_dict

    payload = read_lewm_bundle(path)
    if payload["config"] != recipe_dict(bundle.config):
        raise ValueError("checkpoint_recipe: resume cannot change family, normalization, backend or schedule")
    if payload["dataset"] != dataset_contract:
        raise ValueError("checkpoint_dataset: data bytes, split or collector lineage changed")
    if payload["encoder_frozen"]:
        raise ValueError("checkpoint_phase: an exported frozen encoder is not a joint-training resume")
    if payload["cuda_rng"] and (
        not torch.cuda.is_available() or len(payload["cuda_rng"]) != torch.cuda.device_count()
    ):
        raise ValueError("checkpoint_rng: CUDA devices changed during resume")
    for name, module in (("encoder", bundle.encoder), ("world", bundle.world)):
        module.load_state_dict(payload["modules"][name], strict=True)
        for key, child in module.named_modules():
            child.training = payload["modes"][name][key]
        for key, parameter in module.named_parameters():
            parameter.requires_grad_(payload["requires_grad"][name][key])
    bundle.encoder._frozen = payload["encoder_frozen"]
    optimizer.load_state_dict(payload["optimizer"])
    sampler.load_state_dict(payload["sampler"])
    projection_rng.set_state(payload["projection_rng"].cpu())
    torch.set_rng_state(payload["cpu_rng"])
    if payload["cuda_rng"]:
        torch.cuda.set_rng_state_all(payload["cuda_rng"])
    return payload


def _phase_common(bundle, *, phase: str, step: int, cache_contract: dict) -> dict:
    from .config import recipe_dict, recipe_digest
    from .sources import lewm_source_manifest

    return {
        "format": LEWM_BRIDGE_FORMAT if phase == "bridge" else LEWM_ACTOR_FORMAT,
        "phase": phase,
        "step": step,
        "config": recipe_dict(bundle.config),
        "recipe_id": recipe_digest(bundle.config),
        "sources": lewm_source_manifest(bundle.config),
        "cache": cache_contract,
        "cpu_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def save_lewm_bridge(path, bundle, heads, *, step: int, parent_path, cache_contract: dict,
                     optimizer, sampler, balance: dict, bn_identity: str,
                     gate_report: dict | None, gate_identity: dict | None,
                     head_initial_identity: str | None = None) -> dict:
    """Immutable Phase-2 state, including every stream and the exact H2 authorization."""
    from .data import _sha256

    settings = bundle.config.agent
    if settings is None or not 0 < step <= settings.h2_steps + settings.h16_steps:
        raise ValueError("checkpoint_schedule: bridge step is outside the declared budget")
    depth = settings.recursive_depth if step <= settings.h2_steps else settings.recursive_depth_final
    validated = settings.recursive_depth if gate_identity is not None else 0
    payload = _phase_common(bundle, phase="bridge", step=step, cache_contract=cache_contract) | {
        "parent": {"path": str(Path(parent_path).resolve()), "sha256": _sha256(Path(parent_path))},
        "modules": {
            "encoder": bundle.encoder.state_dict(), "world": bundle.world.state_dict(),
            "heads": heads.state_dict(),
        },
        "modes": {
            "encoder": {name: module.training for name, module in bundle.encoder.named_modules()},
            "world": {name: module.training for name, module in bundle.world.named_modules()},
            "heads": {name: module.training for name, module in heads.named_modules()},
        },
        "requires_grad": {
            "encoder": {name: parameter.requires_grad for name, parameter in bundle.encoder.named_parameters()},
            "world": {name: parameter.requires_grad for name, parameter in bundle.world.named_parameters()},
            "heads": {name: parameter.requires_grad for name, parameter in heads.named_parameters()},
        },
        "optimizer": optimizer.state_dict(), "sampler_rng": sampler.get_state(),
        "balance": dict(balance), "bn_identity": bn_identity,
        "head_initial_identity": head_initial_identity,
        "scheduler": {"type": "linear_warmup_constant_v1", "completed_updates": step},
        "gate": gate_report, "gate_identity": gate_identity,
        "capabilities": {
            "joint_complete": True, "readout_trained": True,
            "trained_recursive_depth": depth, "validated_recursive_depth": validated,
            "actor_trained": False, "m4_authorized": False,
        },
    }
    _save_immutable(Path(path), payload)
    return payload


def read_lewm_bridge(path) -> dict:
    from .config import config_from_dict, recipe_digest
    from .sources import verify_lewm_sources

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != LEWM_BRIDGE_FORMAT or payload.get("phase") != "bridge":
        raise ValueError("checkpoint_family: expected a LeWM bridge checkpoint")
    config = config_from_dict(payload["config"])
    settings = config.agent
    if settings is None or recipe_digest(config) != payload.get("recipe_id"):
        raise ValueError("checkpoint_recipe: bridge recipe is absent or changed")
    step = payload.get("step")
    if (type(step) is not int or not 0 < step <= settings.h2_steps + settings.h16_steps
            or payload.get("scheduler") != {
                "type": "linear_warmup_constant_v1", "completed_updates": step}):
        raise ValueError("checkpoint_schedule: inconsistent bridge update")
    expected_depth = settings.recursive_depth if step <= settings.h2_steps else settings.recursive_depth_final
    capabilities = payload.get("capabilities", {})
    expected_validated = settings.recursive_depth if step > settings.h2_steps else 0
    if (capabilities.get("trained_recursive_depth") != expected_depth
            or capabilities.get("validated_recursive_depth") != expected_validated
            or capabilities.get("readout_trained") is not True
            or capabilities.get("actor_trained") is not False
            or capabilities.get("m4_authorized") is not False):
        raise ValueError("checkpoint_phase: bridge capabilities contradict its stage")
    if step > settings.h2_steps and not isinstance(payload.get("gate_identity"), dict):
        raise ValueError("checkpoint_gate: H16 bridge state lacks its accepted H2 report")
    if step > settings.h2_steps:
        gate, identity = payload.get("gate"), payload["gate_identity"]
        if (not isinstance(gate, dict) or gate.get("report_id") != identity.get("report_id")
                or identity.get("stage") != "h2"):
            raise ValueError("checkpoint_gate: H16 bridge gate lineage is inconsistent")
    from .sources import tensor_state_digest
    projector = {
        name.removeprefix("predictor_projector."): tensor
        for name, tensor in payload["modules"]["world"].items()
        if name.startswith("predictor_projector.")
        and ("running_" in name or "num_batches_tracked" in name)
    }
    if tensor_state_digest(projector) != payload.get("bn_identity"):
        raise ValueError("checkpoint_normalization: bridge BN identity contradicts its tensors")
    verify_lewm_sources(payload["sources"], config)
    return payload


def _restore_module(module, state, modes, gradients) -> None:
    module.load_state_dict(state, strict=True)
    for name, child in module.named_modules():
        child.training = modes[name]
    for name, parameter in module.named_parameters():
        parameter.requires_grad_(gradients[name])


def _restore_rng(payload) -> None:
    torch.set_rng_state(payload["cpu_rng"])
    recorded = payload.get("cuda_rng", [])
    if recorded:
        if not torch.cuda.is_available() or len(recorded) != torch.cuda.device_count():
            raise ValueError("checkpoint_rng: CUDA devices changed during resume")
        torch.cuda.set_rng_state_all(recorded)


def restore_lewm_bridge(path, bundle, heads, *, optimizer, sampler, parent_path,
                        cache_contract: dict, balance: dict) -> dict:
    from .data import _sha256
    from .config import recipe_dict

    payload = read_lewm_bridge(path)
    if payload["config"] != recipe_dict(bundle.config):
        raise ValueError("checkpoint_recipe: bridge resume changed its recipe")
    if payload["parent"]["sha256"] != _sha256(Path(parent_path)):
        raise ValueError("checkpoint_parent: bridge resume changed its joint parent")
    if payload["cache"] != cache_contract:
        raise ValueError("checkpoint_dataset: bridge resume changed its latent cache")
    for name, module in (("encoder", bundle.encoder), ("world", bundle.world), ("heads", heads)):
        _restore_module(module, payload["modules"][name], payload["modes"][name],
                        payload["requires_grad"][name])
    optimizer.load_state_dict(payload["optimizer"])
    sampler.set_state(payload["sampler_rng"].cpu())
    balance.clear(); balance.update(payload["balance"])
    _restore_rng(payload)
    return payload


def save_lewm_actor(path, bundle, heads, prior, *, step: int, bridge_path,
                    cache_contract: dict, optimizer, sampler, policy_rng, balance: dict,
                    frozen_identity: dict, bridge_gate: dict, actor_gate: dict | None,
                    actor_gate_identity: dict | None) -> dict:
    """Immutable Phase-3 state; the encoder/world are included so execution is one-file."""
    from .data import _sha256

    settings = bundle.config.agent
    if settings is None or not 0 < step <= settings.actor_steps:
        raise ValueError("checkpoint_schedule: actor step is outside the declared budget")
    payload = _phase_common(bundle, phase="actor", step=step, cache_contract=cache_contract) | {
        "bridge": {"path": str(Path(bridge_path).resolve()), "sha256": _sha256(Path(bridge_path))},
        "modules": {
            "encoder": bundle.encoder.state_dict(), "world": bundle.world.state_dict(),
            "heads": heads.state_dict(), "prior": prior.state_dict(),
        },
        "modes": {
            name: {key: module.training for key, module in value.named_modules()}
            for name, value in (("encoder", bundle.encoder), ("world", bundle.world),
                                ("heads", heads), ("prior", prior))
        },
        "requires_grad": {
            name: {key: parameter.requires_grad for key, parameter in value.named_parameters()}
            for name, value in (("encoder", bundle.encoder), ("world", bundle.world),
                                ("heads", heads), ("prior", prior))
        },
        "optimizer": optimizer.state_dict(), "sampler_rng": sampler.get_state(),
        "policy_rng": policy_rng.get_state(), "balance": dict(balance),
        "frozen_identity": frozen_identity,
        "scheduler": {"type": "linear_warmup_constant_v1", "completed_updates": step},
        "bridge_gate": bridge_gate, "actor_gate": actor_gate,
        "actor_gate_identity": actor_gate_identity,
        "capabilities": {
            "joint_complete": True, "readout_trained": True,
            "trained_recursive_depth": settings.recursive_depth_final,
            "validated_recursive_depth": settings.recursive_depth_final,
            "actor_trained": step == settings.actor_steps, "m4_authorized": False,
        },
    }
    _save_immutable(Path(path), payload)
    return payload


def read_lewm_actor(path) -> dict:
    from .config import config_from_dict, recipe_digest
    from .sources import verify_lewm_sources

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != LEWM_ACTOR_FORMAT or payload.get("phase") != "actor":
        raise ValueError("checkpoint_family: expected a LeWM actor checkpoint")
    config = config_from_dict(payload["config"])
    settings = config.agent
    if settings is None or recipe_digest(config) != payload.get("recipe_id"):
        raise ValueError("checkpoint_recipe: actor recipe is absent or changed")
    step = payload.get("step")
    if (type(step) is not int or not 0 < step <= settings.actor_steps
            or payload.get("scheduler") != {
                "type": "linear_warmup_constant_v1", "completed_updates": step}):
        raise ValueError("checkpoint_schedule: inconsistent actor update")
    capabilities = payload.get("capabilities", {})
    if (capabilities.get("actor_trained") != (step == settings.actor_steps)
            or capabilities.get("validated_recursive_depth") != settings.recursive_depth_final
            or capabilities.get("m4_authorized") is not False):
        raise ValueError("checkpoint_phase: actor capabilities contradict its stage")
    if not isinstance(payload.get("bridge_gate"), dict):
        raise ValueError("checkpoint_gate: actor lacks its accepted H16 report")
    if step > settings.actor_screen_steps and not isinstance(payload.get("actor_gate_identity"), dict):
        raise ValueError("checkpoint_gate: post-screen actor lacks its accepted screen report")
    if step > settings.actor_screen_steps:
        gate, identity = payload.get("actor_gate"), payload["actor_gate_identity"]
        if (not isinstance(gate, dict) or gate.get("report_id") != identity.get("report_id")
                or identity.get("stage") != "actor_screen"):
            raise ValueError("checkpoint_gate: actor screen-gate lineage is inconsistent")
    from .sources import tensor_state_digest
    heads = payload["modules"]["heads"]
    measured = {
        "encoder": tensor_state_digest(payload["modules"]["encoder"]),
        "world": tensor_state_digest(payload["modules"]["world"]),
        "prior": tensor_state_digest(payload["modules"]["prior"]),
        "model_heads": tensor_state_digest({
            name: tensor for name, tensor in heads.items()
            if name.startswith(("model_body.", "reward.", "continuation."))
        }),
    }
    if measured != payload.get("frozen_identity"):
        raise ValueError("checkpoint_freeze: actor frozen identity contradicts its tensors")
    verify_lewm_sources(payload["sources"], config)
    return payload


def restore_lewm_actor(path, bundle, heads, prior, *, optimizer, sampler, policy_rng,
                       bridge_path, cache_contract: dict, balance: dict) -> dict:
    from .data import _sha256
    from .config import recipe_dict

    payload = read_lewm_actor(path)
    if payload["config"] != recipe_dict(bundle.config):
        raise ValueError("checkpoint_recipe: actor resume changed its recipe")
    if payload["bridge"]["sha256"] != _sha256(Path(bridge_path)):
        raise ValueError("checkpoint_parent: actor resume changed its bridge")
    if payload["cache"] != cache_contract:
        raise ValueError("checkpoint_dataset: actor resume changed its latent cache")
    for name, module in (("encoder", bundle.encoder), ("world", bundle.world),
                         ("heads", heads), ("prior", prior)):
        _restore_module(module, payload["modules"][name], payload["modes"][name],
                        payload["requires_grad"][name])
    optimizer.load_state_dict(payload["optimizer"])
    sampler.set_state(payload["sampler_rng"].cpu())
    policy_rng.set_state(payload["policy_rng"].to(policy_rng.get_state().device))
    balance.clear(); balance.update(payload["balance"])
    _restore_rng(payload)
    return payload
