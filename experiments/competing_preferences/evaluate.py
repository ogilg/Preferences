"""Evaluate competing preferences: probe deltas between competing conditions.

Key analyses:
1. For each pair, compute probe delta under each competing condition
2. Compare: does the probe delta flip between topic_positive and shell_positive?
3. Correlate behavioral competing delta with probe competing delta
4. Dissociation test: same content mentions → different probe scores?
"""

import json
from pathlib import Path

import numpy as np
from scipy import stats

EXP_DIR = Path("experiments/competing_preferences")
CROSSED_DIR = Path("experiments/crossed_preferences")
ACT_DIR = EXP_DIR / "activations"
PROBE_DIR = Path("results/probes/gemma3_3k_completion_preference/probes")
RESULTS_DIR = EXP_DIR / "results"
LAYERS = [31, 43, 55]


def load_probe(layer: int) -> tuple[np.ndarray, float]:
    weights = np.load(PROBE_DIR / f"probe_ridge_L{layer}.npy")
    return weights[:-1], weights[-1]


def score_activations(activations: np.ndarray, coef: np.ndarray, intercept: float) -> np.ndarray:
    return activations @ coef + intercept


def load_activations(filename: str) -> dict[int, np.ndarray]:
    data = np.load(ACT_DIR / filename, allow_pickle=True)
    return {layer: data[f"layer_{layer}"] for layer in LAYERS}


def load_task_index() -> dict[str, int]:
    data = np.load(ACT_DIR / "baseline.npz", allow_pickle=True)
    task_ids = list(data["task_ids"])
    return {tid: i for i, tid in enumerate(task_ids)}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--behavioral", default="behavioral_competing.json")
    args = parser.parse_args()

    # Load behavioral results
    beh_path = RESULTS_DIR / args.behavioral
    with open(beh_path) as f:
        behavioral = json.load(f)

    task_idx = load_task_index()

    # Load probes and baseline
    probes = {layer: load_probe(layer) for layer in LAYERS}
    baseline_acts = load_activations("baseline.npz")
    baseline_scores = {}
    for layer in LAYERS:
        coef, intercept = probes[layer]
        baseline_scores[layer] = score_activations(baseline_acts[layer], coef, intercept)

    # Compute probe deltas for each condition
    results = []
    for beh in behavioral:
        prompt_id = beh["prompt_id"]
        target_task_id = beh["target_task_id"]

        if target_task_id not in task_idx:
            print(f"  WARNING: {target_task_id} not in activation index")
            continue

        npz_path = ACT_DIR / f"{prompt_id}.npz"
        if not npz_path.exists():
            print(f"  WARNING: missing activations for {prompt_id}")
            continue

        tidx = task_idx[target_task_id]
        manip_acts = load_activations(f"{prompt_id}.npz")

        probe_deltas = {}
        for layer in LAYERS:
            coef, intercept = probes[layer]
            manip_score = score_activations(manip_acts[layer], coef, intercept)
            probe_deltas[layer] = float(manip_score[tidx] - baseline_scores[layer][tidx])

        results.append({
            "pair_id": beh["pair_id"],
            "target_topic": beh["target_topic"],
            "category_shell": beh["category_shell"],
            "target_task_id": target_task_id,
            "prompt_id": prompt_id,
            "favored_dim": beh["favored_dim"],
            "behavioral_delta": beh["delta"],
            "probe_delta_L31": probe_deltas[31],
            "probe_delta_L43": probe_deltas[43],
            "probe_delta_L55": probe_deltas[55],
        })

    if len(results) < 2:
        print(f"Only {len(results)} results, not enough for analysis")
        return

    # === Analysis 1: Per-pair competing deltas ===
    print("\n" + "=" * 70)
    print("COMPETING PREFERENCE ANALYSIS")
    print("=" * 70)

    pair_ids = sorted(set(r["pair_id"] for r in results))
    pair_comparisons = []

    print(f"\n{'Pair':<25} {'Dim':>6} {'BehΔ':>8} {'ProbeΔ L31':>12}")
    print("-" * 55)
    for pid in pair_ids:
        topic_r = [r for r in results if r["pair_id"] == pid and r["favored_dim"] == "topic"]
        shell_r = [r for r in results if r["pair_id"] == pid and r["favored_dim"] == "shell"]

        if not topic_r or not shell_r:
            continue

        tr = topic_r[0]
        sr = shell_r[0]

        print(f"{pid:<25} {'topic':>6} {tr['behavioral_delta']:>+8.3f} {tr['probe_delta_L31']:>+12.1f}")
        print(f"{'':<25} {'shell':>6} {sr['behavioral_delta']:>+8.3f} {sr['probe_delta_L31']:>+12.1f}")

        beh_competing = tr["behavioral_delta"] - sr["behavioral_delta"]
        probe_competing = tr["probe_delta_L31"] - sr["probe_delta_L31"]
        print(f"{'':<25} {'diff':>6} {beh_competing:>+8.3f} {probe_competing:>+12.1f}")
        print()

        pair_comparisons.append({
            "pair_id": pid,
            "topic": tr["target_topic"],
            "shell": tr["category_shell"],
            "beh_topic_delta": tr["behavioral_delta"],
            "beh_shell_delta": sr["behavioral_delta"],
            "beh_competing_delta": beh_competing,
            "probe_topic_delta_L31": tr["probe_delta_L31"],
            "probe_shell_delta_L31": sr["probe_delta_L31"],
            "probe_competing_delta_L31": probe_competing,
            "probe_sign_flips": bool(np.sign(tr["probe_delta_L31"]) != np.sign(sr["probe_delta_L31"])),
        })

    # === Analysis 2: Does probe delta flip between conditions? ===
    print("\n" + "=" * 70)
    print("PROBE SIGN FLIP ANALYSIS")
    print("=" * 70)
    n_flips = sum(1 for p in pair_comparisons if p["probe_sign_flips"])
    print(f"Pairs where probe delta flips sign: {n_flips}/{len(pair_comparisons)}")
    for p in pair_comparisons:
        flip_str = "FLIP" if p["probe_sign_flips"] else "same"
        print(f"  {p['pair_id']:<25}: topic_probe={p['probe_topic_delta_L31']:+.1f}, "
              f"shell_probe={p['probe_shell_delta_L31']:+.1f} → {flip_str}")

    # === Analysis 3: Correlation of competing deltas ===
    if len(pair_comparisons) >= 3:
        beh_comp = np.array([p["beh_competing_delta"] for p in pair_comparisons])
        probe_comp = np.array([p["probe_competing_delta_L31"] for p in pair_comparisons])

        print(f"\n{'='*70}")
        print("COMPETING DELTA CORRELATION")
        print(f"{'='*70}")
        pr, pp = stats.pearsonr(beh_comp, probe_comp)
        sr, sp = stats.spearmanr(beh_comp, probe_comp)
        print(f"  N pairs: {len(pair_comparisons)}")
        print(f"  Pearson r: {pr:.3f} (p={pp:.3e})")
        print(f"  Spearman r: {sr:.3f} (p={sp:.3e})")

    # === Analysis 4: Effect sizes ===
    topic_probe_deltas = [p["probe_topic_delta_L31"] for p in pair_comparisons]
    shell_probe_deltas = [p["probe_shell_delta_L31"] for p in pair_comparisons]

    print(f"\n{'='*70}")
    print("EFFECT SIZES")
    print(f"{'='*70}")
    print(f"  Mean probe delta (topic_positive): {np.mean(topic_probe_deltas):+.1f}")
    print(f"  Mean probe delta (shell_positive): {np.mean(shell_probe_deltas):+.1f}")
    if len(pair_comparisons) >= 2:
        t_stat, t_p = stats.ttest_rel(topic_probe_deltas, shell_probe_deltas)
        print(f"  Paired t-test: t={t_stat:.2f}, p={t_p:.3e}")

    topic_beh_deltas = [p["beh_topic_delta"] for p in pair_comparisons]
    shell_beh_deltas = [p["beh_shell_delta"] for p in pair_comparisons]
    print(f"\n  Mean behavioral delta (topic_positive): {np.mean(topic_beh_deltas):+.3f}")
    print(f"  Mean behavioral delta (shell_positive): {np.mean(shell_beh_deltas):+.3f}")
    if len(pair_comparisons) >= 2:
        t_stat, t_p = stats.ttest_rel(topic_beh_deltas, shell_beh_deltas)
        print(f"  Paired t-test: t={t_stat:.2f}, p={t_p:.3e}")

    # === Analysis 5: All individual conditions (for debugging) ===
    print(f"\n{'='*70}")
    print("ALL CONDITIONS — PROBE-BEHAVIORAL CORRELATION")
    print(f"{'='*70}")
    beh_all = np.array([r["behavioral_delta"] for r in results])
    for layer in LAYERS:
        key = f"probe_delta_L{layer}"
        probe_all = np.array([r[key] for r in results])
        pr, pp = stats.pearsonr(beh_all, probe_all)
        print(f"  L{layer}: r={pr:.3f} (p={pp:.3e}), n={len(results)}")

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "probe_competing.json", "w") as f:
        json.dump(results, f, indent=2)
    with open(RESULTS_DIR / "pair_comparisons.json", "w") as f:
        json.dump(pair_comparisons, f, indent=2)
    print(f"\nSaved results to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
