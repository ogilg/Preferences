"""Full 8-persona MRA: cross-evaluation + diversity ablation.

Phase 1: 8x8 cross-eval matrix (train on each persona, eval on all).
Phase 2: Diversity ablation — vary N training personas at fixed 2000 total,
         track generalization to each held-out persona separately.

Usage: python -m scripts.multi_role_ablation.run_mra_8persona
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from src.probes.core.activations import load_probe_data
from src.probes.core.linear_probe import get_default_alphas
from src.probes.data_loading import load_thurstonian_scores

DATE_PREFIX = "030326"
ASSETS = Path("experiments/probe_generalization/multi_role_ablation/assets")
OUTPUT_DIR = Path("results/experiments/mra_exp3/probes")

ALL_PERSONAS = [
    "noprompt", "villain", "aesthete", "midwest",
    "provocateur", "trickster", "autocrat", "sadist",
]

PERSONA_LABELS = {
    "noprompt": "No prompt",
    "villain": "Villain",
    "aesthete": "Aesthete",
    "midwest": "Midwest",
    "provocateur": "Provocateur",
    "trickster": "Trickster",
    "autocrat": "Autocrat",
    "sadist": "Sadist",
}

ACTIVATION_PATHS = {
    "noprompt": Path("activations/gemma_3_27b/activations_prompt_last.npz"),
    "villain": Path("activations/gemma_3_27b_villain/activations_prompt_last.npz"),
    "midwest": Path("activations/gemma_3_27b_midwest/activations_prompt_last.npz"),
    "aesthete": Path("activations/gemma_3_27b_aesthete/activations_prompt_last.npz"),
    "provocateur": Path("activations/gemma_3_27b_provocateur/activations_prompt_last.npz"),
    "trickster": Path("activations/gemma_3_27b_trickster/activations_prompt_last.npz"),
    "autocrat": Path("activations/gemma_3_27b_autocrat/activations_prompt_last.npz"),
    "sadist": Path("activations/gemma_3_27b_sadist/activations_prompt_last.npz"),
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

LAYERS = [31, 43, 55]
ALPHAS = get_default_alphas(10)


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


def load_persona_split_data(persona: str, split: str, layer: int):
    run_dir = get_run_dir(persona, split)
    scores = load_thurstonian_scores(run_dir)
    task_ids = sorted(load_split_task_ids(split) & set(scores.keys()))
    return load_probe_data(ACTIVATION_PATHS[persona], scores, task_ids, layer)


def load_persona_train_data(persona: str, layer: int):
    X_a, y_a, ids_a = load_persona_split_data(persona, "a", layer)
    X_c, y_c, ids_c = load_persona_split_data(persona, "c", layer)
    return np.concatenate([X_a, X_c]), np.concatenate([y_a, y_c]), list(ids_a) + list(ids_c)


# --- Training/eval ---

def train_probe_with_alpha_selection(X_train, y_train, X_val, y_val):
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)

    best_alpha, best_r2 = None, -np.inf
    for alpha in ALPHAS:
        probe = Ridge(alpha=alpha)
        probe.fit(X_train_s, y_train)
        y_pred = probe.predict(X_val_s)
        r2 = 1 - np.sum((y_val - y_pred)**2) / np.sum((y_val - np.mean(y_val))**2)
        if r2 > best_r2:
            best_r2 = r2
            best_alpha = alpha

    probe = Ridge(alpha=best_alpha)
    probe.fit(X_train_s, y_train)
    return probe, scaler, best_alpha, best_r2


def evaluate_probe(probe, scaler, X_eval, y_eval):
    X_s = scaler.transform(X_eval)
    y_pred = probe.predict(X_s)
    r, _ = pearsonr(y_eval, y_pred)
    r2 = 1 - np.sum((y_eval - y_pred)**2) / np.sum((y_eval - np.mean(y_eval))**2)
    y_pred_adj = y_pred - np.mean(y_pred) + np.mean(y_eval)
    r2_adj = 1 - np.sum((y_eval - y_pred_adj)**2) / np.sum((y_eval - np.mean(y_eval))**2)
    return {"r2": float(r2), "r2_adjusted": float(r2_adj), "pearson_r": float(r), "n_samples": len(y_eval)}


# --- Phase 1: Cross-evaluation ---

def run_phase1():
    print("=" * 70)
    print("PHASE 1: 8x8 cross-evaluation")
    print("  Train on split_a+c (2000), sweep alpha on half split_b, eval on other half")
    print("=" * 70)

    results = {}
    rng = np.random.RandomState(42)

    for layer in LAYERS:
        print(f"\n--- Layer {layer} ---")
        layer_key = f"L{layer}"
        results[layer_key] = {"cross_eval": {}, "probes": {}}

        # Load split_b for all personas, split into sweep/eval halves
        eval_data = {}
        sweep_data = {}
        for persona in ALL_PERSONAS:
            X_b, y_b, ids_b = load_persona_split_data(persona, "b", layer)
            n = len(y_b)
            idx = rng.permutation(n)
            half = n // 2
            sweep_data[persona] = (X_b[idx[:half]], y_b[idx[:half]])
            eval_data[persona] = (X_b[idx[half:]], y_b[idx[half:]])

        # Train one probe per persona
        probes = {}
        for train_persona in ALL_PERSONAS:
            X_train, y_train, train_ids = load_persona_train_data(train_persona, layer)
            X_val, y_val = sweep_data[train_persona]

            probe, scaler, best_alpha, val_r2 = train_probe_with_alpha_selection(
                X_train, y_train, X_val, y_val
            )
            probes[train_persona] = (probe, scaler)
            results[layer_key]["probes"][train_persona] = {
                "best_alpha": float(best_alpha), "val_r2": float(val_r2), "n_train": len(train_ids),
            }
            print(f"  Trained {train_persona}: alpha={best_alpha:.1f}, sweep R²={val_r2:.4f}")

        # Cross-evaluate
        cross_eval = {}
        for train_persona in ALL_PERSONAS:
            cross_eval[train_persona] = {}
            probe, scaler = probes[train_persona]
            for eval_persona in ALL_PERSONAS:
                X_eval, y_eval = eval_data[eval_persona]
                metrics = evaluate_probe(probe, scaler, X_eval, y_eval)
                cross_eval[train_persona][eval_persona] = metrics
                marker = " <--" if train_persona == eval_persona else ""
                print(f"    {train_persona:>12} -> {eval_persona:<12}: "
                      f"r={metrics['pearson_r']:.3f}{marker}")

        results[layer_key]["cross_eval"] = cross_eval

    return results


# --- Phase 2: Diversity ablation ---

def run_phase2():
    print("\n" + "=" * 70)
    print("PHASE 2: Diversity ablation (8-persona pool)")
    print("  Fixed 2000 training tasks, vary number of training personas")
    print("=" * 70)

    results = {}
    rng = np.random.RandomState(42)

    for layer in LAYERS:
        print(f"\n--- Layer {layer} ---")
        layer_key = f"L{layer}"
        results[layer_key] = {"conditions": []}

        # Preload all training data
        train_pool = {}
        for persona in ALL_PERSONAS:
            X_train, y_train, ids_train = load_persona_train_data(persona, layer)
            train_pool[persona] = (X_train, y_train, ids_train)

        # Split_b into sweep/eval halves
        rng_split = np.random.RandomState(42)
        sweep_data = {}
        eval_data = {}
        for persona in ALL_PERSONAS:
            X_b, y_b, ids_b = load_persona_split_data(persona, "b", layer)
            n = len(y_b)
            idx = rng_split.permutation(n)
            half = n // 2
            sweep_data[persona] = (X_b[idx[:half]], y_b[idx[:half]])
            eval_data[persona] = (X_b[idx[half:]], y_b[idx[half:]])

        def subsample(X, y, n_samples, rng):
            idx = rng.choice(len(X), size=min(n_samples, len(X)), replace=False)
            return X[idx], y[idx]

        def run_condition(cond_name, train_personas, tasks_per_persona, eval_persona):
            X_parts, y_parts = [], []
            for p in train_personas:
                X_full, y_full, _ = train_pool[p]
                X_sub, y_sub = subsample(X_full, y_full, tasks_per_persona, rng)
                X_parts.append(X_sub)
                y_parts.append(y_sub)

            X_train = np.concatenate(X_parts)
            y_train = np.concatenate(y_parts)

            # Sweep alpha on training personas' sweep data
            X_val = np.concatenate([sweep_data[p][0] for p in train_personas])
            y_val = np.concatenate([sweep_data[p][1] for p in train_personas])

            probe, scaler, best_alpha, val_r2 = train_probe_with_alpha_selection(
                X_train, y_train, X_val, y_val
            )

            X_eval, y_eval = eval_data[eval_persona]
            metrics = evaluate_probe(probe, scaler, X_eval, y_eval)
            return {
                "condition": cond_name,
                "n_train_personas": len(train_personas),
                "train_personas": sorted(train_personas),
                "eval_persona": eval_persona,
                "tasks_per_persona": tasks_per_persona,
                "n_total_train": len(y_train),
                "best_alpha": float(best_alpha),
                **metrics,
            }

        # For each held-out eval persona, train on subsets of the other 7
        for eval_persona in ALL_PERSONAS:
            available = [p for p in ALL_PERSONAS if p != eval_persona]
            print(f"\n  Eval: {eval_persona}")

            # N=1: each single persona x 2000
            for train_p in available:
                entry = run_condition("1x2000", [train_p], 2000, eval_persona)
                results[layer_key]["conditions"].append(entry)

            # N=2: all pairs x 1000 each
            for p1, p2 in itertools.combinations(available, 2):
                entry = run_condition("2x1000", [p1, p2], 1000, eval_persona)
                results[layer_key]["conditions"].append(entry)

            # N=3: all triples x 667 each
            for triple in itertools.combinations(available, 3):
                entry = run_condition("3x667", list(triple), 667, eval_persona)
                results[layer_key]["conditions"].append(entry)

            # N=4: all quads x 500 each
            for quad in itertools.combinations(available, 4):
                entry = run_condition("4x500", list(quad), 500, eval_persona)
                results[layer_key]["conditions"].append(entry)

            # N=5: all quintuples x 400 each
            for quint in itertools.combinations(available, 5):
                entry = run_condition("5x400", list(quint), 400, eval_persona)
                results[layer_key]["conditions"].append(entry)

            # N=6: all sextuples x 333 each
            for sext in itertools.combinations(available, 6):
                entry = run_condition("6x333", list(sext), 333, eval_persona)
                results[layer_key]["conditions"].append(entry)

            # N=7: all 7 x 286 each (matched 2000 total)
            entry = run_condition("7x286", available, 286, eval_persona)
            results[layer_key]["conditions"].append(entry)

            # Ceiling: N=7 x 2000 each (14000 total)
            entry = run_condition("7x2000", available, 2000, eval_persona)
            results[layer_key]["conditions"].append(entry)

            # Report summary for this eval persona
            for n_p in [1, 2, 3, 4, 5, 6, 7]:
                cond_entries = [e for e in results[layer_key]["conditions"]
                                if e["eval_persona"] == eval_persona
                                and e["n_train_personas"] == n_p
                                and e["n_total_train"] <= 2100]
                if cond_entries:
                    rs = [e["pearson_r"] for e in cond_entries]
                    print(f"    N={n_p}: mean r={np.mean(rs):.3f} "
                          f"(min={np.min(rs):.3f}, max={np.max(rs):.3f}, n_combos={len(rs)})")

            ceil_entries = [e for e in results[layer_key]["conditions"]
                           if e["eval_persona"] == eval_persona and e["condition"] == "7x2000"]
            if ceil_entries:
                print(f"    Ceiling (7x2000): r={ceil_entries[0]['pearson_r']:.3f}")

    return results


# --- Plotting ---

def plot_phase1_heatmap(phase1_results: dict):
    for layer in LAYERS:
        layer_key = f"L{layer}"
        cross_eval = phase1_results[layer_key]["cross_eval"]

        matrix = np.zeros((len(ALL_PERSONAS), len(ALL_PERSONAS)))
        for i, train_p in enumerate(ALL_PERSONAS):
            for j, eval_p in enumerate(ALL_PERSONAS):
                matrix[i, j] = cross_eval[train_p][eval_p]["pearson_r"]

        fig, ax = plt.subplots(figsize=(9, 8))
        im = ax.imshow(matrix, cmap="RdYlGn", vmin=-0.2, vmax=1.0)
        ax.set_xticks(range(len(ALL_PERSONAS)))
        ax.set_yticks(range(len(ALL_PERSONAS)))
        ax.set_xticklabels([PERSONA_LABELS[p] for p in ALL_PERSONAS], rotation=45, ha="right", fontsize=9)
        ax.set_yticklabels([PERSONA_LABELS[p] for p in ALL_PERSONAS], fontsize=9)
        ax.set_xlabel("Eval persona", fontsize=11)
        ax.set_ylabel("Train persona", fontsize=11)
        ax.set_title(f"Cross-persona probe transfer (Layer {layer}, Pearson r)", fontsize=12)

        for i in range(len(ALL_PERSONAS)):
            for j in range(len(ALL_PERSONAS)):
                color = "white" if matrix[i, j] < 0.3 else "black"
                ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=8, color=color)

        fig.colorbar(im, ax=ax, shrink=0.8)
        fig.tight_layout()
        path = ASSETS / f"plot_{DATE_PREFIX}_8persona_cross_eval_L{layer}.png"
        fig.savefig(path, dpi=150)
        plt.close()
        print(f"Saved {path}")


def plot_phase2_diversity(phase2_results: dict):
    for layer in LAYERS:
        layer_key = f"L{layer}"
        conditions = phase2_results[layer_key]["conditions"]

        fig, ax = plt.subplots(figsize=(12, 7))
        persona_colors = {
            "noprompt": "#2c3e50", "villain": "#e74c3c", "aesthete": "#9b59b6",
            "midwest": "#2ecc71", "provocateur": "#e67e22", "trickster": "#3498db",
            "autocrat": "#1abc9c", "sadist": "#c0392b",
        }

        for eval_persona in ALL_PERSONAS:
            means = []
            stds = []
            ns = list(range(1, 8))
            for n_p in ns:
                entries = [e for e in conditions
                           if e["eval_persona"] == eval_persona
                           and e["n_train_personas"] == n_p
                           and e["n_total_train"] <= 2100]
                if entries:
                    rs = [e["pearson_r"] for e in entries]
                    means.append(np.mean(rs))
                    stds.append(np.std(rs))
                else:
                    means.append(np.nan)
                    stds.append(0)

            means = np.array(means)
            stds = np.array(stds)
            color = persona_colors[eval_persona]
            lw = 2.5 if eval_persona in ("noprompt", "sadist") else 1.2
            alpha = 1.0 if eval_persona in ("noprompt", "sadist") else 0.6
            ax.plot(ns, means, "o-", color=color, label=PERSONA_LABELS[eval_persona],
                    linewidth=lw, alpha=alpha, markersize=4)
            ax.fill_between(ns, means - stds, means + stds, color=color, alpha=0.1)

        ax.set_xlabel("Number of training personas (fixed 2000 total tasks)", fontsize=11)
        ax.set_ylabel("Pearson r on held-out persona", fontsize=11)
        ax.set_title(f"Diversity ablation: generalization vs training persona count (Layer {layer})", fontsize=12)
        ax.set_xticks(range(1, 8))
        ax.set_ylim(0, 1.0)
        ax.legend(fontsize=9, loc="lower right", ncol=2)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        path = ASSETS / f"plot_{DATE_PREFIX}_8persona_diversity_L{layer}.png"
        fig.savefig(path, dpi=150)
        plt.close()
        print(f"Saved {path}")


def main():
    ASSETS.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    phase1 = run_phase1()
    plot_phase1_heatmap(phase1)

    phase2 = run_phase2()
    plot_phase2_diversity(phase2)

    results = {"phase1": phase1, "phase2": phase2}
    output_path = OUTPUT_DIR / "mra_8persona_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
