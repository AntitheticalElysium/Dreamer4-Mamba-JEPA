"""Post-smoke artifact identities and full-dev unlock-event check; no final-split examples."""
import json
from pathlib import Path

import numpy as np
import torch

from saccade import train as TR
from saccade.eval import s1

OUT = Path(__file__).resolve().parent

if __name__ == "__main__":
    L = np.load(TR.ROOT / "artifacts/eda/s0/s0_dev_labels.npz")
    offsets = np.r_[0, np.cumsum(L["ep_T"] + 1)]
    dh = np.concatenate([np.diff(L["s_body"][a:b, 0].astype(float)) for a, b in zip(offsets[:-1], offsets[1:])])
    reward = L["t_reward"].astype(float)
    unlocked = np.round(reward - .1 * dh) == 1
    values, inverse = np.unique(reward, return_inverse=True)
    probability = np.zeros(len(reward), np.float32)
    for a in range(0, len(reward), 10_000):
        b = min(len(reward), a + 10_000)
        logp = torch.full((b - a, len(values)), float("-inf"))
        logp[torch.arange(b - a), torch.as_tensor(inverse[a:b])] = 0
        probability[a:b] = s1.unlock_probability(logp, torch.as_tensor(values), torch.as_tensor(dh[a:b])).numpy()
    event = {"transitions": len(reward), "unlocks": int(unlocked.sum()),
             "old_threshold_misses": int((unlocked & ~(reward > .45)).sum()),
             "corrected_event_mismatches": int((probability != unlocked).sum()),
             "perfect_reward_unlock_brier": float(np.mean((probability - unlocked)**2))}
    assert event["corrected_event_mismatches"] == 0
    (OUT / "unlock_full_dev.json").write_text(json.dumps(event, indent=2) + "\n")

    checkpoint = OUT / "smoke/latest.pt"
    ck = torch.load(checkpoint, weights_only=False, map_location="cpu")
    TR.validate_resume(ck, ck["cfg"], 1898, TR.run_provenance(ck["cfg"]))
    assert ck["state"]["update"] == 20 and ck["state"]["frames"] == 20_480
    metrics = json.loads((OUT / "s1_neural_smoke.json").read_text())
    assert all(k in metrics for k in ("s1_1", "s1_2", "s1_3", "s1_4", "reader", "s3_perception_blocker"))
    assert metrics["checkpoint_sha256"] == TR.file_sha256(checkpoint)
    reader = s1.load_reader(OUT / "s1_neural_smoke.reader.pt", checkpoint, "cpu")
    assert all(not p.requires_grad for p in reader.parameters())
    assert metrics["reader"]["sha256"] == TR.file_sha256(OUT / "s1_neural_smoke.reader.pt")
    mon = [json.loads(line) for line in (OUT / "smoke/log.jsonl").read_text().splitlines()
           if "checkpoint" in json.loads(line)]
    assert len(mon) == 2 and all(len(m["encoder_grad_norms"]) == 6 for m in mon)
    assert all(m["monitor"]["raw_token_statistics"].keys() == {"periphery", "fovea", "lighting"} for m in mon)
    paths = ["saccade/model.py", "saccade/data.py", "saccade/train.py", "saccade/eval/s1.py", "saccade/m0.yaml"]
    result = {"checkpoint_identity": "matches current sources/config/data", "updates": 20,
              "s1_sections_written": 4, "reader_reloaded_frozen": True,
              "monitor_checkpoints": len(mon), "source_sha256": {p: TR.file_sha256(TR.ROOT / p) for p in paths},
              "data_sha256": {p: TR.file_sha256(TR.ROOT / "artifacts/saccade_m0/targets" / p)
                              for p in ("change_teacher.txt", "change_targets.npy")}, "unlock": event}
    lens = L["ep_T"][:15]
    ep = np.random.default_rng(0).choice(len(lens), min(30000, int(lens.sum())), p=lens / lens.sum())
    test = s1.G.episode_split(ep, ep, np.random.default_rng(1))[:len(ep)]
    result["classifier_test_episodes"] = len(np.unique(ep[test]))
    (OUT / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
