"""Midway bias analysis: do multi-persona probes predict persona means more accurately?

For probes trained on default (noprompt) only vs multi-persona training,
check whether predictions are pulled toward the noprompt mean ("midway bias").

Training: split A (1000 tasks). Alpha sweep: split B (500). Eval: split C (1000).
Noprompt always included in training. N=1..8 personas, all combinations enumerated.

Metric: midway ratio per topic = (pred_topic_mean - noprompt_topic_mean) /
                                  (actual_topic_mean - noprompt_topic_mean)
  ratio=1.0 → correct persona mean, ratio=0.0 → stuck at noprompt mean.

Usage: python -m scripts.multi_role_ablation.midway_bias [--selector turn_boundary:-2]
"""

from __future__ import annotations

import argparse
import itertools
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from src.probes.core.activations import load_activations
from src.probes.data_loading import load_thurstonian_scores

TOPICS_PATH = Path("data/topics/topics.json")
ASSETS = Path("experiments/probe_generalization/multi_role_ablation/assets")
OUTPUT_DIR = Path("results/experiments/mra_exp3/midway_bias")

ALL_PERSONAS = [
    "noprompt", "villain", "aesthete", "midwest",
    "provocateur", "trickster", "autocrat", "sadist",
]
NON_DEFAULT = [p for p in ALL_PERSONAS if p != "noprompt"]

ACTIVATION_DIRS = {
    "noprompt": Path("activations/gemma_3_27b_turn_boundary_sweep"),
    "villain": Path("activations/gemma_3_27b_villain_tb"),
    "aesthete": Path("activations/gemma_3_27b_aesthete_tb"),
    "midwest": Path("activations/gemma_3_27b_midwest_tb"),
    "provocateur": Path("activations/gemma_3_27b_provocateur_tb"),
    "trickster": Path("activations/gemma_3_27b_trickster_tb"),
    "autocrat": Path("activations/gemma_3_27b_autocrat_tb"),
    "sadist": Path("activations/gemma_3_27b_sadist_tb"),
}

PERSONA_RUNS = {
    "noprompt": (Path("results/experiments/mra_exp2/pre_task_active_learning"), ""),
    "villain": (Path("results/experiments/mra_exp2/pre_task_active_learning"), "syse8f24ac6"),
    "aesthete": (Path("results/experiments/mra_exp2/pre_task_active_learning"), "sys021d8ca1"),
    "midwest": (Path("results/experiments/mra_exp2/pre_task_active_learning"), "sys5d504504"),
    "provocateur": (Path("results/experiments/mra_exp3/pre_task_active_learning"), "sysf4d93514"),
    "trickster": (Path("results/experiments/mra_exp3/pre_task_active_learning"), "sys09a42edc"),
    "autocrat": (Path("results/experiments/mra_exp3/pre_task_active_learning"), "sys1c18219a"),
    "sadist": (Path("results/experiments/mra_exp3/pre_task_active_learning"), "sys39e01d59"),
}

SPLIT_TASK_ID_FILES = {
    "a": Path("configs/measurement/active_learning/mra_exp2_split_a_1000_task_ids.txt"),
    "b": Path("configs/measurement/active_learning/mra_exp2_split_b_500_task_ids.txt"),
    "c": Path("configs/measurement/active_learning/mra_exp2_split_c_1000_task_ids.txt"),
}

LAYERS = [25, 32, 39, 46, 53]
SELECTORS = ["turn_boundary:-2", "turn_boundary:-5"]
ALPHAS = np.logspace(-1, 5, 10)


# --- Data loading ---

def load_split_task_ids(split: str) -> set[str]:
    with open(SPLIT_TASK_ID_FILES[split]) as f:
        return {line.strip() for line in f if line.strip()}


def get_run_dir(persona: str, split: str) -> Path:
    results_dir, sys_hash = PERSONA_RUNS[persona]
    n = {"a": 1000, "b": 500, "c": 1000}[split]
    prefix = "completion_preference_gemma-3-27b_completion_canonical_seed0"
    suffix = f"mra_exp2_split_{split}_{n}_task_ids"
    dirname = f"{prefix}_{sys_hash}_{suffix}" if sys_hash else f"{prefix}_{suffix}"
    return results_dir / dirname


def activation_file(persona: str, selector: str) -> Path:
    safe_name = selector.replace(":", "")
    return ACTIVATION_DIRS[persona] / f"activations_{safe_name}.npz"


def load_persona_split_data(
    persona: str, split: str, layer: int, selector: str,
    cached_activations: dict,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    run_dir = get_run_dir(persona, split)
    scores = load_thurstonian_scores(run_dir)
    split_ids = load_split_task_ids(split)
    task_ids = sorted(split_ids & set(scores.keys()))

    cache_key = (persona, selector)
    if cache_key not in cached_activations:
        path = activation_file(persona, selector)
        act_task_ids, act_dict = load_activations(path, layers=LAYERS)
        cached_activations[cache_key] = (act_task_ids, act_dict)

    act_task_ids, act_dict = cached_activations[cache_key]
    act_matrix = act_dict[layer]
    act_id_to_idx = {tid: i for i, tid in enumerate(act_task_ids)}

    matched_ids, matched_indices, matched_scores = [], [], []
    for tid in task_ids:
        if tid in act_id_to_idx and tid in scores:
            matched_ids.append(tid)
            matched_indices.append(act_id_to_idx[tid])
            matched_scores.append(scores[tid])

    X = act_matrix[matched_indices]
    y = np.array(matched_scores)
    return X, y, matched_ids


def load_topic_map() -> dict[str, str]:
    with open(TOPICS_PATH) as f:
        raw = json.load(f)
    topic_map = {}
    for tid, models in raw.items():
        for _model_name, cats in models.items():
            topic_map[tid] = cats["primary"]
            break
    return topic_map


# --- Training ---

def train_probe(X_train, y_train, X_val, y_val):
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)

    best_alpha, best_r2 = None, -np.inf
    for alpha in ALPHAS:
        probe = Ridge(alpha=alpha)
        probe.fit(X_train_s, y_train)
        y_pred = probe.predict(X_val_s)
        r2 = 1 - np.sum((y_val - y_pred) ** 2) / np.sum((y_val - np.mean(y_val)) ** 2)
        if r2 > best_r2:
            best_r2 = r2
            best_alpha = alpha

    probe = Ridge(alpha=best_alpha)
    probe.fit(X_train_s, y_train)
    return probe, scaler


def predict_raw(probe, scaler, X):
    return probe.predict(scaler.transform(X))


# --- Main analysis ---

def run_for_selector_layer(
    selector: str,
    layer: int,
    topic_map: dict[str, str],
    rng: np.random.RandomState,
) -> list[dict]:
    print(f"\n{'='*70}")
    print(f"Selector: {selector}, Layer: {layer}")
    print(f"{'='*70}")

    cached_activations: dict = {}

    # Load all data for this selector/layer
    print("Loading data...")
    train_data = {}
    sweep_data = {}
    eval_data = {}

    for persona in ALL_PERSONAS:
        X_a, y_a, ids_a = load_persona_split_data(persona, "a", layer, selector, cached_activations)
        X_b, y_b, ids_b = load_persona_split_data(persona, "b", layer, selector, cached_activations)
        X_c, y_c, ids_c = load_persona_split_data(persona, "c", layer, selector, cached_activations)
        train_data[persona] = (X_a, y_a, ids_a)
        sweep_data[persona] = (X_b, y_b, ids_b)
        eval_data[persona] = (X_c, y_c, ids_c)
        print(f"  {persona}: train={len(ids_a)}, sweep={len(ids_b)}, eval={len(ids_c)}")

    # Build topic arrays for eval data
    eval_topics = {}
    for persona in ALL_PERSONAS:
        _, _, ids_c = eval_data[persona]
        eval_topics[persona] = [topic_map.get(tid, "unknown") for tid in ids_c]

    # Get noprompt eval means per topic (the "anchor")
    _, y_noprompt_eval, _ = eval_data["noprompt"]
    noprompt_topics = eval_topics["noprompt"]
    noprompt_topic_means = {}
    for topic in set(noprompt_topics):
        mask = [t == topic for t in noprompt_topics]
        noprompt_topic_means[topic] = float(np.mean(y_noprompt_eval[mask]))

    results = []
    total_train = 1000

    for n_personas in range(1, len(ALL_PERSONAS) + 1):
        if n_personas == 1:
            combos = [("noprompt",)]
        else:
            others_combos = list(itertools.combinations(NON_DEFAULT, n_personas - 1))
            combos = [("noprompt",) + others for others in others_combos]

        tasks_per_persona = total_train // n_personas
        print(f"\nN={n_personas}: {len(combos)} combos, {tasks_per_persona} tasks/persona")

        for combo in combos:
            combo_set = set(combo)

            # Build training data
            X_parts, y_parts = [], []
            for p in combo:
                X_full, y_full, _ = train_data[p]
                if tasks_per_persona < len(y_full):
                    idx = rng.choice(len(y_full), size=tasks_per_persona, replace=False)
                    X_parts.append(X_full[idx])
                    y_parts.append(y_full[idx])
                else:
                    X_parts.append(X_full)
                    y_parts.append(y_full)

            X_train = np.concatenate(X_parts)
            y_train = np.concatenate(y_parts)

            # Build sweep data (from training personas only)
            X_val = np.concatenate([sweep_data[p][0] for p in combo])
            y_val = np.concatenate([sweep_data[p][1] for p in combo])

            probe, scaler = train_probe(X_train, y_train, X_val, y_val)

            # Evaluate on ALL personas
            for eval_persona in ALL_PERSONAS:
                X_eval, y_eval, eval_ids = eval_data[eval_persona]
                y_pred = predict_raw(probe, scaler, X_eval)
                topics = eval_topics[eval_persona]
                is_in_dist = eval_persona in combo_set

                r, _ = pearsonr(y_eval, y_pred)

                topic_results = {}
                for topic in set(topics):
                    mask = np.array([t == topic for t in topics])
                    n_topic = int(mask.sum())
                    if n_topic < 5:
                        continue

                    pred_mean = float(np.mean(y_pred[mask]))
                    actual_mean = float(np.mean(y_eval[mask]))
                    anchor_mean = noprompt_topic_means.get(topic)
                    if anchor_mean is None:
                        continue

                    denom = actual_mean - anchor_mean
                    if abs(denom) < 0.1:
                        continue

                    midway_ratio = (pred_mean - anchor_mean) / denom

                    topic_results[topic] = {
                        "n": n_topic,
                        "pred_mean": pred_mean,
                        "actual_mean": actual_mean,
                        "anchor_mean": anchor_mean,
                        "midway_ratio": midway_ratio,
                    }

                results.append({
                    "selector": selector,
                    "layer": layer,
                    "n_personas": n_personas,
                    "train_personas": sorted(combo),
                    "eval_persona": eval_persona,
                    "is_in_dist": is_in_dist,
                    "pearson_r": float(r),
                    "topics": topic_results,
                })

            combo_label = "+".join(sorted(combo))
            print(f"  {combo_label}: done")

    return results


def print_summary(all_results: list[dict]):
    print("\n" + "=" * 80)
    print("SUMMARY: Mean midway ratio by N, selector, layer")
    print("=" * 80)

    for selector in SELECTORS:
        for layer in LAYERS:
            subset = [r for r in all_results if r["selector"] == selector and r["layer"] == layer]
            if not subset:
                continue

            print(f"\n--- {selector} / Layer {layer} ---")

            agg = defaultdict(list)
            for entry in subset:
                n = entry["n_personas"]
                ep = entry["eval_persona"]
                dist = "in" if entry["is_in_dist"] else "ood"
                if ep == "noprompt":
                    continue
                ratios = [t["midway_ratio"] for t in entry["topics"].values()]
                if ratios:
                    mean_ratio = float(np.mean(ratios))
                    agg[(n, dist)].append(mean_ratio)
                    agg[(n, dist, ep)].append(mean_ratio)

            print(f"  {'N':>3}  {'In-dist':>10}  {'OOD':>10}")
            for n in range(1, 9):
                in_vals = agg.get((n, "in"), [])
                ood_vals = agg.get((n, "ood"), [])
                in_str = f"{np.mean(in_vals):.3f}" if in_vals else "n/a"
                ood_str = f"{np.mean(ood_vals):.3f}" if ood_vals else "n/a"
                print(f"  {n:>3}  {in_str:>10}  {ood_str:>10}")

            print(f"\n  Per persona (OOD):")
            header = f"  {'N':>3}"
            for ep in NON_DEFAULT:
                header += f"  {ep:>12}"
            print(header)
            for n in range(1, 9):
                row = f"  {n:>3}"
                for ep in NON_DEFAULT:
                    vals = agg.get((n, "ood", ep), [])
                    row += f"  {np.mean(vals):>12.3f}" if vals else f"  {'n/a':>12}"
                print(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--selector", type=str, help="Run only this selector")
    parser.add_argument("--layer", type=int, help="Run only this layer")
    args = parser.parse_args()

    ASSETS.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    topic_map = load_topic_map()

    selectors = [args.selector] if args.selector else SELECTORS
    layers = [args.layer] if args.layer else LAYERS

    all_results = []
    for selector in selectors:
        for layer in layers:
            rng = np.random.RandomState(42)
            results = run_for_selector_layer(selector, layer, topic_map, rng)
            all_results.extend(results)

    suffix = ""
    if args.selector:
        suffix += f"_{args.selector.replace(':', '').replace('-', '')}"
    if args.layer:
        suffix += f"_L{args.layer}"

    output_path = OUTPUT_DIR / f"midway_bias_results{suffix}.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    print_summary(all_results)


if __name__ == "__main__":
    main()
