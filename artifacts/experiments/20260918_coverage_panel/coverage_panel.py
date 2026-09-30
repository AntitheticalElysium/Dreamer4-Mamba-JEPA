"""A declared coverage panel over the audited addresses, scored both ways.

The completed gate reports `insufficient_coverage` on 116 entries, which are twelve
distinct binary labels repeated across arms, probe families and conditions.  The
coverage audit showed that is a *panel-selection* limit, not corpus scarcity: its
129 verified roots carry at least ten positive and ten negative independent episodes
per split for every one of those twelve, with exact root and factual-successor pixels.

This scores those addresses.  It is a stress distribution, deliberately selected for
the rare labels, and it does not replace the broad population panel -- the gate's own
rows stand unchanged.  Selection was made model-free by the audit, before any arm was
scored here, and is not re-derived from anything this script measures.

Every label is reported under both readout protocols, because the generated-fit
rescore showed the two disagree by up to 0.29 AUC on outcomes:

  observed-fit  -> generated   shared-decoder transfer, the gate's protocol
  generated-fit -> generated   native recoverability, fitted where it is evaluated

with action-only and root+action controls alongside, so a score is never read as
world-model skill when a control reaches it.  Probes, metrics, bootstrap and the
episode clustering are the gate's own functions, not reimplementations.

Read-only: loads published checkpoints, writes only under this experiment directory,
changes no gate output and authorizes nothing.
"""

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/m03-coverage-mpl")

import numpy as np
import torch

ROOT = Path("/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA")
sys.path.insert(0, str(ROOT))

from d4mj.data import _sha256, atomic_manifest
from d4mj.m03.gate import (STATIC_BINARY, M03Settings, _binary_metrics, _encode_lewm,
                           _fit_probe_many, _one_hot_actions, _recorded_step_key,
                           _state_binary_labels, load_m03_bundle)

INDEX = ROOT / "artifacts/experiments/20260918_m03_probe_coverage_audit/evidence/verified_root_index.json"
DATASET = ROOT / "artifacts/craftax_support_v2/manifest.json"
# The twelve the completed gate could not measure.
DECLARED = ("front_lava", "front_water", "front_tree", "front_ripe_plant", "near_table", "place_plant",
            "make_wood_pickaxe", "make_stone_pickaxe", "make_iron_pickaxe",
            "make_wood_sword", "make_stone_sword", "make_iron_sword")


def replay_rows(index, settings, destination):
    """Render each audited root's context, its 17 factual successors, and both label sets.

    Re-verifies exact pixels here rather than trusting the recorded audit fields: the
    panel must not inherit an integrity claim it did not make itself.
    """
    if destination.exists():
        payload = torch.load(destination, map_location="cpu", weights_only=False)
        if payload.get("index_sha256") == _sha256(INDEX):
            return payload
        raise ValueError("coverage_panel: cached replay does not match the audited index")
    from artifacts.eda import replay

    _, _, _, step, frame = replay.env_and_render()
    width = settings.lewm_context
    context, past, successors, root_labels, successor_labels, episodes, splits = [], [], [], [], [], [], []
    worst = 0
    for position, row in enumerate(index):
        fields = replay.episode_fields(row["shard"], row["slot"])
        t = row["t"]
        observations = fields["observations"][t - width + 1:t + 1].numpy()
        actions = fields["actions_taken"][t - width + 1:t].numpy()
        state = replay.advance_to(row["shard"], row["slot"], t)
        key = _recorded_step_key(replay, row)
        rendered = np.asarray(frame(state)).astype(np.int16)
        worst = max(worst, int(np.abs(rendered - observations[-1].astype(np.int16)).max()))
        branch_frames, branch_labels = [], []
        for action in range(17):
            _, nxt, _, _, _ = step(key, state, action)
            branch_frames.append(np.asarray(frame(nxt)).astype(np.uint8))
            branch_labels.append(_state_binary_labels(nxt))
        factual = branch_frames[int(fields["actions_taken"][t])]
        worst = max(worst, int(np.abs(factual.astype(np.int16)
                                      - fields["observations"][t + 1].numpy().astype(np.int16)).max()))
        context.append(torch.from_numpy(observations.astype(np.uint8)))
        past.append(torch.from_numpy(actions.astype(np.int64)))
        successors.append(torch.from_numpy(np.stack(branch_frames)))
        root_labels.append(torch.tensor(np.asarray(_state_binary_labels(state)), dtype=torch.bool))
        successor_labels.append(torch.tensor(np.stack(branch_labels), dtype=torch.bool))
        episodes.append(row["episode_id"])
        splits.append(row["split"])
        if position % 20 == 0:
            print(json.dumps({"stage": "replay", "root": position, "of": len(index),
                              "worst_pixel_abs": worst}), flush=True)
    unique = sorted(set(episodes))
    payload = {"context": torch.stack(context), "past_actions": torch.stack(past),
               "successors": torch.stack(successors), "root_labels": torch.stack(root_labels),
               "successor_labels": torch.stack(successor_labels),
               "episode": torch.tensor([unique.index(e) for e in episodes], dtype=torch.long),
               "split": splits, "episode_ids": episodes, "worst_pixel_abs": worst,
               "index_sha256": _sha256(INDEX)}
    if worst != 0:
        raise ValueError(f"coverage_panel: replay is not pixel exact (max abs {worst})")
    torch.save(payload, destination)
    return payload


def coverage_census(rows, names):
    """Independent episodes carrying each label, per split, for roots and successors."""
    out = {}
    for name in names:
        index = STATIC_BINARY.index(name)
        entry = {}
        for split in ("train", "dev"):
            keep = [i for i, s in enumerate(rows["split"]) if s == split]
            for field, tag in (("root_labels", "root"), ("successor_labels", "successor")):
                truth = rows[field][keep]
                truth = truth[:, index] if tag == "root" else truth[:, :, index].any(1)
                episodes = np.array([rows["episode_ids"][i] for i in keep])
                entry[f"{split}_{tag}"] = {"positive_episodes": int(len(set(episodes[truth.numpy()]))),
                                           "negative_episodes": int(len(set(episodes[~truth.numpy()])))}
        out[name] = entry
    return out


def score_arm(bundle, rows, settings, device):
    """Both readout protocols plus their controls, on identical DEV rows and labels."""
    features = _encode_lewm(bundle, {"context": rows["context"], "past_actions": rows["past_actions"],
                                     "successors": rows["successors"]}, settings)
    keep = {s: [i for i, v in enumerate(rows["split"]) if v == s] for s in ("train", "dev")}
    report = {}
    truth = {s: rows["successor_labels"][keep[s]].reshape(-1, len(STATIC_BINARY)).float().to(device)
             for s in ("train", "dev")}
    roots = {s: rows["episode"][keep[s]].repeat_interleave(17) for s in ("train", "dev")}
    action = {s: _one_hot_actions(len(keep[s]), device=device) for s in ("train", "dev")}

    def joined(split, field):
        value = features[field][keep[split]].flatten(0, 1).float().to(device)
        return torch.cat((value, action[split]), 1)

    def root_action(split):
        base = features["projected"][keep[split]].float().to(device)
        return torch.cat((base.repeat_interleave(17, 0), action[split]), 1)

    for hidden in (False, True):
        head = "mlp" if hidden else "linear"
        conditions = {}
        observed = _fit_probe_many(joined("train", "observed_successor"), truth["train"],
                                   {"observed": joined("dev", "observed_successor"),
                                    "generated": joined("dev", "generated_successor")},
                                   settings, hidden=hidden, binary=True)
        conditions["observed_fit_on_observed"] = observed["observed"]
        conditions["observed_fit_on_generated"] = observed["generated"]
        generated = _fit_probe_many(joined("train", "generated_successor"), truth["train"],
                                    {"generated": joined("dev", "generated_successor"),
                                     "observed": joined("dev", "observed_successor")},
                                    settings, hidden=hidden, binary=True)
        conditions["generated_fit_on_generated"] = generated["generated"]
        conditions["generated_fit_on_observed"] = generated["observed"]
        conditions["root_action"] = _fit_probe_many(root_action("train"), truth["train"],
                                                    {"dev": root_action("dev")},
                                                    settings, hidden=hidden, binary=True)["dev"]
        conditions["action_only"] = _fit_probe_many(action["train"], truth["train"],
                                                    {"dev": action["dev"]},
                                                    settings, hidden=hidden, binary=True)["dev"]
        # `_binary_metrics` indexes roots by the truth mask and then hands them to numpy,
        # so logits, truth and roots must all be on the host, as the gate itself passes them.
        report[head] = {name: _binary_metrics(logits.cpu(), truth["dev"].cpu(), roots["dev"],
                                              STATIC_BINARY, settings)
                        for name, logits in conditions.items()}
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", type=Path, required=True, help="JSON mapping arm name to checkpoint path")
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "evidence")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    destination = args.out / "coverage_panel.json"
    if destination.exists():
        raise FileExistsError(f"coverage_panel: refusing to replace {destination}")

    settings = M03Settings()
    index = json.loads(INDEX.read_text())
    rows = replay_rows(index, settings, args.out / "coverage_rows.pt")
    print(json.dumps({"stage": "replay_complete", "roots": len(index),
                      "worst_pixel_abs": rows["worst_pixel_abs"]}), flush=True)
    census = coverage_census(rows, DECLARED)
    report = {"schema": "d4mj_m03_coverage_panel_v1",
              "question": "are the twelve labels the completed gate could not measure decodable on the "
                          "audited addresses, under both readout protocols?",
              "scope": "declared stress panel over audited addresses; does NOT replace the broad population "
                       "panel and changes no gate output",
              "selection": {"source": str(INDEX), "sha256": _sha256(INDEX), "roots": len(index),
                            "made": "model-free by the coverage audit, before any arm was scored here"},
              "replay_integrity": {"root_and_factual_pixel_max_abs": rows["worst_pixel_abs"],
                                   "re_verified_here": True},
              "declared_labels": list(DECLARED), "episode_census": census,
              "arms": {}, "m03_capability": "not_a_gate_result", "m4_authorized": False}
    for name, path in json.loads(args.arms.read_text()).items():
        bundle, _, _ = load_m03_bundle(Path(path), device=args.device, dataset_sha256=_sha256(DATASET))
        report["arms"][name] = {"checkpoint": str(path), "checkpoint_sha256": _sha256(Path(path)),
                                "scores": score_arm(bundle, rows, settings, args.device)}
        del bundle
        torch.cuda.empty_cache()
        print(json.dumps({"stage": "arm_complete", "arm": name}), flush=True)
    atomic_manifest(destination, report)
    print(json.dumps({"status": "complete", "report": str(destination)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
