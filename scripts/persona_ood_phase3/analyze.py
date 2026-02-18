"""Phase 3 analysis: correlate behavioral and probe deltas.

Computes:
1. Pooled behavioral-probe correlation (original, enriched, combined)
2. Per-persona correlations
3. Sign agreement
4. Attenuation analysis (split-half reliability of behavioral deltas)
5. Controls: shuffled labels, cross-persona
"""

import json
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy import stats

BASE = Path("/workspace/repo")
PHASE3 = BASE / "experiments/probe_generalization/persona_ood/phase3"
RESULTS_PATH = PHASE3 / "results.json"
CORE_TASKS_PATH = PHASE3 / "core_tasks.json"
ACTIVATIONS_DIR = BASE / "activations/persona_ood_phase3"
ORIGINAL_CONFIG = BASE / "experiments/probe_generalization/persona_ood/v2_config.json"
ENRICHED_CONFIG = BASE / "experiments/probe_generalization/persona_ood/prompt_enrichment/prompts.json"

PROBE_CONFIGS = {
    "demean/ridge_L31": BASE / "results/probes/gemma3_3k_std_demean/gemma3_3k_std_demean/probes/probe_ridge_L31.npy",
    "raw/ridge_L31": BASE / "results/probes/gemma3_3k_std_raw/gemma3_3k_std_raw/probes/probe_ridge_L31.npy",
    "demean/ridge_L43": BASE / "results/probes/gemma3_3k_std_demean/gemma3_3k_std_demean/probes/probe_ridge_L43.npy",
    "raw/ridge_L43": BASE / "results/probes/gemma3_3k_std_raw/gemma3_3k_std_raw/probes/probe_ridge_L43.npy",
    "demean/ridge_L55": BASE / "results/probes/gemma3_3k_std_demean/gemma3_3k_std_demean/probes/probe_ridge_L55.npy",
    "raw/ridge_L55": BASE / "results/probes/gemma3_3k_std_raw/gemma3_3k_std_raw/probes/probe_ridge_L55.npy",
}

LAYER_MAP = {"L31": 31, "L43": 43, "L55": 55}


def load_probe(probe_path):
    probe = np.load(probe_path)
    return probe[:-1], probe[-1]  # weights, bias


def compute_probe_scores(activations, layer, weights, bias):
    acts = activations[f"layer_{layer}"]
    return acts @ weights + bias


def main():
    with open(CORE_TASKS_PATH) as f:
        task_ids = json.load(f)["task_ids"]
    with open(RESULTS_PATH) as f:
        results = json.load(f)
    with open(ORIGINAL_CONFIG) as f:
        original_names = [p["name"] for p in json.load(f)["personas"] if p["part"] == "A"]
    with open(ENRICHED_CONFIG) as f:
        enriched_names = list(json.load(f).keys())

    all_personas = original_names + enriched_names
    print(f"Tasks: {len(task_ids)}, Personas: {len(all_personas)} ({len(original_names)} original + {len(enriched_names)} enriched)")

    # Load neutral activations
    neutral_data = np.load(ACTIVATIONS_DIR / "neutral" / "activations_prompt_last.npz", allow_pickle=True)
    neutral_task_ids = list(neutral_data["task_ids"])
    neutral_id_to_idx = {tid: i for i, tid in enumerate(neutral_task_ids)}

    # Baseline behavioral rates
    baseline_rates = results["baseline"]["task_rates"]

    # == Analysis for each probe ==
    output = {}
    for probe_name, probe_path in PROBE_CONFIGS.items():
        layer_str = probe_name.split("_")[-1]
        layer = LAYER_MAP[layer_str]
        weights, bias = load_probe(probe_path)

        neutral_scores = compute_probe_scores(neutral_data, layer, weights, bias)

        # Collect all (persona, task) deltas
        all_behavioral = []
        all_probe = []
        all_persona_labels = []
        all_task_labels = []

        per_persona_results = {}

        for persona_name in all_personas:
            if persona_name not in results:
                print(f"  WARNING: {persona_name} not in results, skipping")
                continue

            persona_rates = results[persona_name]["task_rates"]
            persona_data = np.load(
                ACTIVATIONS_DIR / persona_name / "activations_prompt_last.npz",
                allow_pickle=True,
            )
            persona_task_ids = list(persona_data["task_ids"])
            persona_id_to_idx = {tid: i for i, tid in enumerate(persona_task_ids)}
            persona_scores = compute_probe_scores(persona_data, layer, weights, bias)

            b_deltas = []
            p_deltas = []
            for tid in task_ids:
                if tid not in baseline_rates or tid not in persona_rates:
                    continue
                if tid not in neutral_id_to_idx or tid not in persona_id_to_idx:
                    continue
                b_delta = persona_rates[tid]["p_choose"] - baseline_rates[tid]["p_choose"]
                p_delta = float(persona_scores[persona_id_to_idx[tid]] - neutral_scores[neutral_id_to_idx[tid]])
                b_deltas.append(b_delta)
                p_deltas.append(p_delta)
                all_behavioral.append(b_delta)
                all_probe.append(p_delta)
                all_persona_labels.append(persona_name)
                all_task_labels.append(tid)

            if len(b_deltas) >= 5:
                r, p = stats.pearsonr(b_deltas, p_deltas)
                per_persona_results[persona_name] = {"r": r, "p": p, "n": len(b_deltas)}

        all_behavioral = np.array(all_behavioral)
        all_probe = np.array(all_probe)
        all_persona_labels = np.array(all_persona_labels)

        # Pooled correlation
        pooled_r, pooled_p = stats.pearsonr(all_behavioral, all_probe)

        # By group
        orig_mask = np.isin(all_persona_labels, original_names)
        enrich_mask = np.isin(all_persona_labels, enriched_names)
        orig_r, orig_p = stats.pearsonr(all_behavioral[orig_mask], all_probe[orig_mask])
        enrich_r, enrich_p = stats.pearsonr(all_behavioral[enrich_mask], all_probe[enrich_mask])

        # Sign agreement
        def sign_agreement(beh, prb, threshold=0.02):
            mask = np.abs(beh) >= threshold
            if mask.sum() == 0:
                return float("nan"), 0
            agree = np.sign(beh[mask]) == np.sign(prb[mask])
            return agree.mean(), int(mask.sum())

        sign_all, sign_n_all = sign_agreement(all_behavioral, all_probe)
        sign_orig, sign_n_orig = sign_agreement(all_behavioral[orig_mask], all_probe[orig_mask])
        sign_enrich, sign_n_enrich = sign_agreement(all_behavioral[enrich_mask], all_probe[enrich_mask])

        output[probe_name] = {
            "pooled_r": pooled_r,
            "pooled_p": pooled_p,
            "n_total": len(all_behavioral),
            "original_r": orig_r,
            "original_p": orig_p,
            "enriched_r": enrich_r,
            "enriched_p": enrich_p,
            "sign_agreement_all": sign_all,
            "sign_agreement_n_all": sign_n_all,
            "sign_agreement_original": sign_orig,
            "sign_agreement_enriched": sign_enrich,
            "per_persona": per_persona_results,
        }

        print(f"\n=== {probe_name} ===")
        print(f"  Pooled: r={pooled_r:.3f} (p={pooled_p:.2e}, n={len(all_behavioral)})")
        print(f"  Original: r={orig_r:.3f} (p={orig_p:.2e})")
        print(f"  Enriched: r={enrich_r:.3f} (p={enrich_p:.2e})")
        print(f"  Sign agreement: all={sign_all:.3f} ({sign_n_all}), orig={sign_orig:.3f}, enrich={sign_enrich:.3f}")

        # Per-persona table
        print(f"\n  Per-persona (sorted by r):")
        sorted_personas = sorted(per_persona_results.items(), key=lambda x: -x[1]["r"])
        for pname, pr in sorted_personas:
            group = "orig" if pname in original_names else "enrich"
            print(f"    {pname:30s} r={pr['r']:.3f}  p={pr['p']:.3e}  n={pr['n']}  [{group}]")

    # == Controls (using primary probe: demean/ridge_L31) ==
    primary = "demean/ridge_L31"
    layer = 31
    weights, bias = load_probe(PROBE_CONFIGS[primary])
    neutral_scores = compute_probe_scores(neutral_data, layer, weights, bias)

    # Rebuild deltas for primary probe
    persona_behavioral = {}  # persona -> {task_id: b_delta}
    persona_probe = {}  # persona -> {task_id: p_delta}

    for persona_name in all_personas:
        if persona_name not in results:
            continue
        persona_rates = results[persona_name]["task_rates"]
        persona_data = np.load(
            ACTIVATIONS_DIR / persona_name / "activations_prompt_last.npz",
            allow_pickle=True,
        )
        persona_task_ids = list(persona_data["task_ids"])
        persona_id_to_idx = {tid: i for i, tid in enumerate(persona_task_ids)}
        persona_scores = compute_probe_scores(persona_data, layer, weights, bias)

        b_map = {}
        p_map = {}
        for tid in task_ids:
            if tid not in baseline_rates or tid not in persona_rates:
                continue
            if tid not in neutral_id_to_idx or tid not in persona_id_to_idx:
                continue
            b_map[tid] = persona_rates[tid]["p_choose"] - baseline_rates[tid]["p_choose"]
            p_map[tid] = float(persona_scores[persona_id_to_idx[tid]] - neutral_scores[neutral_id_to_idx[tid]])
        persona_behavioral[persona_name] = b_map
        persona_probe[persona_name] = p_map

    # Shuffled labels control
    print("\n\n=== CONTROLS ===")
    rng = np.random.RandomState(42)
    observed_r = output[primary]["pooled_r"]

    # Pool all deltas
    all_b = []
    all_p = []
    for pn in all_personas:
        if pn not in persona_behavioral:
            continue
        common_tasks = set(persona_behavioral[pn].keys()) & set(persona_probe[pn].keys())
        for tid in sorted(common_tasks):
            all_b.append(persona_behavioral[pn][tid])
            all_p.append(persona_probe[pn][tid])
    all_b = np.array(all_b)
    all_p = np.array(all_p)

    n_shuffles = 1000
    shuffled_rs = []
    for _ in range(n_shuffles):
        perm = rng.permutation(len(all_b))
        r_shuf, _ = stats.pearsonr(all_b[perm], all_p)
        shuffled_rs.append(r_shuf)
    shuffled_rs = np.array(shuffled_rs)
    perm_p = (shuffled_rs >= observed_r).mean()
    print(f"Shuffled labels: mean r = {shuffled_rs.mean():.4f} ± {shuffled_rs.std():.4f}")
    print(f"  Observed r = {observed_r:.4f}, permutation p = {perm_p:.4f}")

    # Cross-persona control
    matched_rs = []
    cross_rs = []
    active_personas = [pn for pn in all_personas if pn in persona_behavioral]

    for pn in active_personas:
        common = sorted(set(persona_behavioral[pn].keys()) & set(persona_probe[pn].keys()))
        if len(common) < 5:
            continue
        b = np.array([persona_behavioral[pn][tid] for tid in common])
        p = np.array([persona_probe[pn][tid] for tid in common])
        r, _ = stats.pearsonr(b, p)
        matched_rs.append(r)

    for pn_a, pn_b in combinations(active_personas, 2):
        common = sorted(
            set(persona_behavioral[pn_a].keys()) & set(persona_probe[pn_b].keys())
        )
        if len(common) < 5:
            continue
        b = np.array([persona_behavioral[pn_a][tid] for tid in common])
        p = np.array([persona_probe[pn_b][tid] for tid in common])
        r, _ = stats.pearsonr(b, p)
        cross_rs.append(r)

    print(f"\nMatched persona: mean r = {np.mean(matched_rs):.3f} (n={len(matched_rs)})")
    print(f"Cross-persona: mean r = {np.mean(cross_rs):.3f} (n={len(cross_rs)})")
    print(f"Gap: {np.mean(matched_rs) - np.mean(cross_rs):.3f}")

    # Attenuation analysis: split-half reliability of behavioral deltas
    print("\n\n=== ATTENUATION ANALYSIS ===")
    reliabilities = []
    for pn in active_personas:
        if pn not in results:
            continue
        raw = results[pn].get("raw_results", [])
        if not raw:
            continue

        # Split by resample: even/odd pairs
        task_wins_even: dict[str, list[bool]] = {}
        task_wins_odd: dict[str, list[bool]] = {}
        for idx_r, r in enumerate(raw):
            if r["is_refusal"]:
                continue
            ti, tj = r["task_i"], r["task_j"]
            target = task_wins_even if idx_r % 2 == 0 else task_wins_odd
            target.setdefault(ti, [])
            target.setdefault(tj, [])
            if r["chose_i"]:
                target[ti].append(True)
                target[tj].append(False)
            else:
                target[ti].append(False)
                target[tj].append(True)

        # Compute p_choose for each half
        common_tasks = sorted(set(task_wins_even.keys()) & set(task_wins_odd.keys()))
        if len(common_tasks) < 10:
            continue
        p_even = np.array([np.mean(task_wins_even[t]) for t in common_tasks])
        p_odd = np.array([np.mean(task_wins_odd[t]) for t in common_tasks])
        r_half, _ = stats.pearsonr(p_even, p_odd)
        # Spearman-Brown prophecy for full reliability
        reliability = 2 * r_half / (1 + r_half)
        reliabilities.append(reliability)

    mean_reliability = np.mean(reliabilities)
    print(f"Split-half reliability of p_choose: {mean_reliability:.3f} (n={len(reliabilities)} personas)")
    print(f"  Range: [{min(reliabilities):.3f}, {max(reliabilities):.3f}]")

    # Disattenuated correlation
    disattenuated_r = observed_r / np.sqrt(mean_reliability)
    print(f"Disattenuated r: {disattenuated_r:.3f} (observed {observed_r:.3f} / sqrt({mean_reliability:.3f}))")

    # Also compute delta reliability
    delta_reliabilities = []
    if "baseline" in results and results["baseline"].get("raw_results"):
        base_raw = results["baseline"]["raw_results"]
        base_wins_even: dict[str, list[bool]] = {}
        base_wins_odd: dict[str, list[bool]] = {}
        for idx_r, r in enumerate(base_raw):
            if r["is_refusal"]:
                continue
            ti, tj = r["task_i"], r["task_j"]
            target = base_wins_even if idx_r % 2 == 0 else base_wins_odd
            target.setdefault(ti, [])
            target.setdefault(tj, [])
            if r["chose_i"]:
                target[ti].append(True)
                target[tj].append(False)
            else:
                target[ti].append(False)
                target[tj].append(True)

        for pn in active_personas:
            if pn not in results or not results[pn].get("raw_results"):
                continue
            raw = results[pn]["raw_results"]
            tw_even: dict[str, list[bool]] = {}
            tw_odd: dict[str, list[bool]] = {}
            for idx_r, r in enumerate(raw):
                if r["is_refusal"]:
                    continue
                ti, tj = r["task_i"], r["task_j"]
                target = tw_even if idx_r % 2 == 0 else tw_odd
                target.setdefault(ti, [])
                target.setdefault(tj, [])
                if r["chose_i"]:
                    target[ti].append(True)
                    target[tj].append(False)
                else:
                    target[ti].append(False)
                    target[tj].append(True)

            common = sorted(
                set(tw_even.keys()) & set(tw_odd.keys()) &
                set(base_wins_even.keys()) & set(base_wins_odd.keys())
            )
            if len(common) < 10:
                continue

            delta_even = np.array([np.mean(tw_even[t]) - np.mean(base_wins_even[t]) for t in common])
            delta_odd = np.array([np.mean(tw_odd[t]) - np.mean(base_wins_odd[t]) for t in common])
            r_half, _ = stats.pearsonr(delta_even, delta_odd)
            reliability = 2 * r_half / (1 + r_half)
            delta_reliabilities.append(reliability)

        if delta_reliabilities:
            mean_delta_rel = np.mean(delta_reliabilities)
            print(f"\nDelta reliability (persona - baseline): {mean_delta_rel:.3f} (n={len(delta_reliabilities)})")
            print(f"  Range: [{min(delta_reliabilities):.3f}, {max(delta_reliabilities):.3f}]")
            disattenuated_r_delta = observed_r / np.sqrt(mean_delta_rel)
            print(f"Disattenuated r (delta-based): {disattenuated_r_delta:.3f}")

    # Save results
    output["controls"] = {
        "shuffled_labels": {
            "mean_r": float(shuffled_rs.mean()),
            "std_r": float(shuffled_rs.std()),
            "perm_p": float(perm_p),
            "observed_r": float(observed_r),
        },
        "cross_persona": {
            "matched_mean_r": float(np.mean(matched_rs)),
            "cross_mean_r": float(np.mean(cross_rs)),
            "gap": float(np.mean(matched_rs) - np.mean(cross_rs)),
        },
    }
    output["attenuation"] = {
        "p_choose_reliability": float(mean_reliability),
        "delta_reliability": float(np.mean(delta_reliabilities)) if delta_reliabilities else None,
        "disattenuated_r": float(disattenuated_r),
        "disattenuated_r_delta": float(disattenuated_r_delta) if delta_reliabilities else None,
    }

    output_path = PHASE3 / "analysis_results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else x)
    print(f"\nSaved analysis to {output_path}")


if __name__ == "__main__":
    main()
