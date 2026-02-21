"""Exp 3C analysis: anti-prompt selectivity and A vs C comparison.

Analyzes C (anti) conditions alongside A (pro) and B (neutral) for minimal pairs:
- Probe delta correlation with behavioral delta for C conditions
- Selectivity: target task rank under C (should be near 50/50)
- A vs C specificity comparison

The target task for each (role, target) pair is defined as the task with the
largest behavioral delta under version A. This target is then looked up in B and C.

Usage: python scripts/exp3c_anti/analyze_exp3c.py [--layer LAYER]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

ACTS_DIR = REPO_ROOT / "activations" / "ood" / "exp3_minimal_pairs"
PROBE_DIR = REPO_ROOT / "results" / "probes" / "gemma3_10k_heldout_std_demean" / "probes"
BEH_PATH = REPO_ROOT / "results" / "ood" / "minimal_pairs_v7" / "behavioral.json"
CFG_PATH = REPO_ROOT / "configs" / "ood" / "prompts" / "minimal_pairs_v7.json"
OUT_DIR = REPO_ROOT / "experiments" / "ood_system_prompts" / "exp3c_anti"

SELECTED_ROLES = {"midwest", "brooklyn"}
LAYERS = [31, 43, 55]


def load_probe(layer):
    probe = np.load(PROBE_DIR / f"probe_ridge_L{layer}.npy")
    return probe[:-1], float(probe[-1])


def score_npz(npz_path, layer, weights, bias):
    data = np.load(npz_path, allow_pickle=True)
    acts = data[f"layer_{layer}"]
    scores = (acts @ weights + bias).tolist()
    return dict(zip(data["task_ids"].tolist(), scores))


def permutation_r(beh, probe, n=1000, seed=42):
    rng = np.random.RandomState(seed)
    r_obs, _ = stats.pearsonr(beh, probe)
    perm_rs = [stats.pearsonr(beh[rng.permutation(len(beh))], probe)[0] for _ in range(n)]
    return float((np.array(perm_rs) >= r_obs).mean())


def sign_agreement(beh, probe, threshold=0.02):
    mask = np.abs(beh) >= threshold
    if mask.sum() == 0:
        return float("nan"), 0
    return float((np.sign(beh[mask]) == np.sign(probe[mask])).mean()), int(mask.sum())


def get_deltas(cid, beh_data, baseline_scores, baseline_rates, layer, weights, bias):
    npz = ACTS_DIR / cid / "activations_prompt_last.npz"
    if not npz.exists():
        return {}, {}
    cond_scores = score_npz(npz, layer, weights, bias)
    cond_data = beh_data["conditions"].get(cid, {})
    if not cond_data:
        return {}, {}
    cond_rates = {tid: v["p_choose"] for tid, v in cond_data["task_rates"].items()}
    pd, bd = {}, {}
    for tid in baseline_scores:
        if tid in cond_scores and tid in cond_rates and tid in baseline_rates:
            pd[tid] = cond_scores[tid] - baseline_scores[tid]
            bd[tid] = cond_rates[tid] - baseline_rates[tid]
    return pd, bd


def rank_in_sorted(probe_deltas, target_task, ascending=False):
    if target_task not in probe_deltas:
        return None
    sorted_vals = sorted(probe_deltas.values(), reverse=not ascending)
    return sorted_vals.index(probe_deltas[target_task]) + 1


def analyze_layer(beh_data, cfg, baseline_scores, baseline_rates, layer, weights, bias):
    all_conds = {c["condition_id"]: c for c in cfg["conditions"] if c["base_role"] in SELECTED_ROLES}

    # Group by (role, target)
    pairs_by_key = {}
    for cid, cond in all_conds.items():
        key = (cond["base_role"], cond["target"])
        if key not in pairs_by_key:
            pairs_by_key[key] = {"base_role": cond["base_role"], "target": cond["target"]}
        pairs_by_key[key][cond["version"]] = cid

    pair_records = []
    version_pts = {"A": [], "B": [], "C": []}

    for key, pair in pairs_by_key.items():
        # Need A to define target task
        if "A" not in pair:
            continue
        cid_a = pair["A"]
        a_pd, a_bd = get_deltas(cid_a, beh_data, baseline_scores, baseline_rates, layer, weights, bias)
        if not a_pd:
            continue

        # Target task = max behavioral delta under A
        target_task = max(a_bd, key=lambda t: a_bd[t])

        record = {
            "base_role": pair["base_role"],
            "target": pair["target"],
            "target_task": target_task,
        }

        for version in ["A", "B", "C"]:
            if version not in pair:
                continue
            cid = pair[version]
            if version == "A":
                pd, bd = a_pd, a_bd
            else:
                pd, bd = get_deltas(cid, beh_data, baseline_scores, baseline_rates, layer, weights, bias)
            if not pd:
                continue

            version_pts[version].extend(zip(bd.values(), pd.values()))

            record[version] = {
                "condition_id": cid,
                "n_tasks": len(pd),
                "target_probe_delta": pd.get(target_task),
                "target_beh_delta": bd.get(target_task),
                "probe_rank_desc": rank_in_sorted(pd, target_task, ascending=False),
                "probe_rank_asc": rank_in_sorted(pd, target_task, ascending=True),
            }

        pair_records.append(record)

    # Overall correlation stats per version
    version_stats = {}
    for version, pts in version_pts.items():
        if not pts:
            continue
        beh_arr = np.array([p[0] for p in pts])
        probe_arr = np.array([p[1] for p in pts])
        r, p = stats.pearsonr(beh_arr, probe_arr)
        sign, sign_n = sign_agreement(beh_arr, probe_arr)
        perm_p = permutation_r(beh_arr, probe_arr)
        version_stats[version] = {
            "n": len(beh_arr),
            "pearson_r": float(r),
            "pearson_p": float(p),
            "permutation_p": perm_p,
            "sign_agreement": sign,
            "sign_n": sign_n,
        }

    return {"pairs": pair_records, "version_stats": version_stats}


def print_summary(results, layer):
    print(f"\n=== Layer {layer} ===")
    for version, s in results["version_stats"].items():
        print(f"  {version}: r={s['pearson_r']:.3f} (perm_p={s['permutation_p']:.3f}), "
              f"sign={s['sign_agreement']:.1%} ({s['sign_n']}), n={s['n']}")

    print(f"\n  Probe rank of target task ({len(results['pairs'])} pairs):")
    for version in ["A", "B", "C"]:
        for rank_key, label in [("probe_rank_desc", "1=highest"), ("probe_rank_asc", "1=most_neg")]:
            ranks = [r[version][rank_key] for r in results["pairs"]
                     if version in r and r[version].get(rank_key) is not None]
            if not ranks:
                continue
            ranks_arr = np.array(ranks)
            n5 = (ranks_arr <= 5).sum()
            print(f"    {version} [{label}]: mean={ranks_arr.mean():.1f}, top5={n5}/{len(ranks_arr)}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, default=None)
    args = parser.parse_args()
    layers = [args.layer] if args.layer else LAYERS

    beh_data = json.load(open(BEH_PATH))
    cfg = json.load(open(CFG_PATH))
    baseline_rates = {tid: v["p_choose"] for tid, v in beh_data["conditions"]["baseline"]["task_rates"].items()}

    all_results = {}
    for layer in layers:
        weights, bias = load_probe(layer)
        baseline_scores = score_npz(ACTS_DIR / "baseline" / "activations_prompt_last.npz", layer, weights, bias)
        results = analyze_layer(beh_data, cfg, baseline_scores, baseline_rates, layer, weights, bias)
        all_results[f"L{layer}"] = results
        print_summary(results, layer)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_file = OUT_DIR / "exp3c_results.json"
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {out_file}")


if __name__ == "__main__":
    main()


def compute_specificity_analysis(results, layer_key, beh_data, cfg, baseline_scores, baseline_rates, layer, weights, bias):
    """Compute A-B and A-C probe specificity for target tasks.

    Specificity = target delta / mean(|all other deltas|)
    """
    all_conds = {c["condition_id"]: c for c in cfg["conditions"] if c["base_role"] in SELECTED_ROLES}
    pairs_by_key = {}
    for cid, cond in all_conds.items():
        key = (cond["base_role"], cond["target"])
        if key not in pairs_by_key:
            pairs_by_key[key] = {}
        pairs_by_key[key][cond["version"]] = cid

    ab_specificities = []
    ac_specificities = []
    ab_ranks = []
    ac_ranks = []

    for key, pair in pairs_by_key.items():
        if "A" not in pair or ("B" not in pair and "C" not in pair):
            continue
        a_pd, a_bd = get_deltas(pair["A"], beh_data, baseline_scores, baseline_rates, layer, weights, bias)
        if not a_pd:
            continue
        target_task = max(a_bd, key=lambda t: a_bd[t])

        for version, spec_list, rank_list in [("B", ab_specificities, ab_ranks), ("C", ac_specificities, ac_ranks)]:
            if version not in pair:
                continue
            v_pd, v_bd = get_deltas(pair[version], beh_data, baseline_scores, baseline_rates, layer, weights, bias)
            if not v_pd:
                continue

            # Probe delta = A probe delta - version probe delta (for each task)
            diff_probe = {}
            for tid in a_pd:
                if tid in v_pd:
                    diff_probe[tid] = a_pd[tid] - v_pd[tid]

            if target_task not in diff_probe:
                continue

            target_diff = diff_probe[target_task]
            others = [abs(diff_probe[t]) for t in diff_probe if t != target_task]
            if not others:
                continue
            mean_other = np.mean(others)
            specificity = abs(target_diff) / mean_other if mean_other > 0 else float("inf")

            # Rank by A-version diff (descending)
            sorted_tids = sorted(diff_probe.keys(), key=lambda t: diff_probe[t], reverse=True)
            rank = sorted_tids.index(target_task) + 1 if target_task in sorted_tids else None

            spec_list.append(specificity)
            if rank is not None:
                rank_list.append(rank)

    return {
        "ab_probe_specificity": {
            "mean": float(np.mean(ab_specificities)) if ab_specificities else None,
            "median": float(np.median(ab_specificities)) if ab_specificities else None,
            "values": ab_specificities,
        },
        "ac_probe_specificity": {
            "mean": float(np.mean(ac_specificities)) if ac_specificities else None,
            "median": float(np.median(ac_specificities)) if ac_specificities else None,
            "values": ac_specificities,
        },
        "ab_probe_ranks": ab_ranks,
        "ac_probe_ranks": ac_ranks,
    }
