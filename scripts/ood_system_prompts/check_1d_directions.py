"""Check 1d competing: is topicpos consistently more positive than shellpos?"""

import json
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).parent.parent.parent
RESULTS = REPO_ROOT / "experiments" / "ood_system_prompts" / "analysis_results_full.json"

data = json.load(open(RESULTS))
res = data["exp1d"]["L31"]["on_target"]

beh = np.array(res["behavioral_deltas"])
probe = np.array(res["probe_deltas"])
labels = np.array(res["condition_labels"])

# Parse into pairs
pairs: dict[str, dict] = {}
for i, cid in enumerate(labels):
    parts = cid.split("_")
    direction = parts[-1]  # topicpos or shellpos
    pair_id = "_".join(parts[:-1])
    if pair_id not in pairs:
        pairs[pair_id] = {}
    pairs[pair_id][direction] = {"beh": beh[i], "probe": probe[i]}

print(f"{'Pair':<45} {'Beh(topic)':>10} {'Beh(shell)':>10} {'topic>shell?':>12} {'Probe(topic)':>12} {'Probe(shell)':>12} {'topic>shell?':>12}")
print("-" * 130)

beh_correct = 0
probe_correct = 0
n = 0

for pair_id, dirs in sorted(pairs.items()):
    if "topicpos" not in dirs or "shellpos" not in dirs:
        continue
    n += 1
    bt = dirs["topicpos"]["beh"]
    bs = dirs["shellpos"]["beh"]
    pt = dirs["topicpos"]["probe"]
    ps = dirs["shellpos"]["probe"]

    beh_ok = bt > bs
    probe_ok = pt > ps
    beh_correct += beh_ok
    probe_correct += probe_ok

    print(f"{pair_id:<45} {bt:>+10.3f} {bs:>+10.3f} {'YES' if beh_ok else 'no':>12} {pt:>+12.3f} {ps:>+12.3f} {'YES' if probe_ok else 'no':>12}")

print(f"\nPairs: {n}")
print(f"Behavioral: topicpos > shellpos in {beh_correct}/{n} = {beh_correct/n:.0%}")
print(f"Probe:      topicpos > shellpos in {probe_correct}/{n} = {probe_correct/n:.0%}")

# Ground truth: topicpos should be +1 (love topic), shellpos should be -1 (hate topic)
# So topicpos delta should be MORE POSITIVE than shellpos delta
# Which means: ground_truth(topicpos) = +1, ground_truth(shellpos) = -1
# And we check: does delta correlate with this?
gt = np.array([1 if "topicpos" in c else -1 for c in labels])
from scipy import stats
r_beh, p_beh = stats.pearsonr(gt, beh)
r_probe, p_probe = stats.pearsonr(gt, probe)
print(f"\nGround truth correlation:")
print(f"  Beh ↔ GT:   r={r_beh:.3f} (p={p_beh:.4f})")
print(f"  Probe ↔ GT: r={r_probe:.3f} (p={p_probe:.4f})")

# Sign agreement
beh_sign = np.mean(np.sign(beh) == gt)
probe_sign = np.mean(np.sign(probe) == gt)
print(f"\nSign agreement with ground truth:")
print(f"  Beh:   {beh_sign:.0%}")
print(f"  Probe: {probe_sign:.0%}")
