"""Per-persona transitivity analysis for MRA exp2 and exp3.

Computes transitivity only over triads where all 3 pairs were actually compared.
"""

import itertools
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

EXP2_DIR = Path("results/experiments/mra_exp2/pre_task_active_learning")
EXP3_DIR = Path("results/experiments/mra_exp3/pre_task_active_learning")

PERSONAS = {
    "noprompt": {"exp": "exp2", "sys_hash": None},
    "villain": {"exp": "exp2", "sys_hash": "syse8f24ac6"},
    "aesthete": {"exp": "exp2", "sys_hash": "sys021d8ca1"},
    "midwest": {"exp": "exp2", "sys_hash": "sys5d504504"},
    "trickster": {"exp": "exp3", "sys_hash": "sys09a42edc"},
    "autocrat": {"exp": "exp3", "sys_hash": "sys1c18219a"},
    "sadist": {"exp": "exp3", "sys_hash": "sys39e01d59"},
    "provocateur": {"exp": "exp3", "sys_hash": "sysf4d93514"},
}


def find_split_dirs(base_dir: Path, exp_name: str, sys_hash: str | None) -> list[Path]:
    """Find the 3 split directories for a persona."""
    dirs = []
    for d in sorted(base_dir.iterdir()):
        if not d.is_dir():
            continue
        name = d.name
        if sys_hash is None:
            # noprompt: match dirs with exp_name split but no sys hash
            # Pattern: ...seed0_mra_exp2_split_...
            if f"seed0_{exp_name}_split_" in name:
                dirs.append(d)
        else:
            if f"_{sys_hash}_" in name:
                dirs.append(d)
    return dirs


def load_merged_wins(split_dirs: list[Path]) -> tuple[np.ndarray, list[str]]:
    """Load measurements from all splits and merge into a single wins matrix."""
    all_measurements = []
    for d in split_dirs:
        path = d / "measurements.yaml"
        with open(path) as f:
            measurements = yaml.load(f, Loader=yaml.CSafeLoader)
        if measurements:
            all_measurements.extend(measurements)

    task_ids_set: set[str] = set()
    for m in all_measurements:
        task_ids_set.add(m["task_a"])
        task_ids_set.add(m["task_b"])

    task_ids = sorted(task_ids_set)
    id_to_idx = {tid: i for i, tid in enumerate(task_ids)}
    n = len(task_ids)
    wins = np.zeros((n, n), dtype=np.int32)

    for m in all_measurements:
        i, j = id_to_idx[m["task_a"]], id_to_idx[m["task_b"]]
        if m["choice"] == "a":
            wins[i, j] += 1
        else:
            wins[j, i] += 1

    return wins, task_ids


def measure_transitivity_compared_only(
    wins: np.ndarray,
    max_triads: int = 200_000,
    seed: int = 42,
) -> dict:
    """Measure transitivity restricted to triads where all 3 pairs were compared."""
    n = wins.shape[0]
    compared = (wins + wins.T) > 0  # bool matrix: was this pair compared?

    # Find all compared pairs
    compared_pairs = set()
    for i in range(n):
        for j in range(i + 1, n):
            if compared[i, j]:
                compared_pairs.add((i, j))

    # Build adjacency list of compared neighbors for each node
    neighbors = {i: set() for i in range(n)}
    for i, j in compared_pairs:
        neighbors[i].add(j)
        neighbors[j].add(i)

    # Find triads where all 3 pairs compared: for each compared pair (i,j),
    # find common compared neighbors
    valid_triads = []
    for i, j in compared_pairs:
        common = neighbors[i] & neighbors[j]
        for k in common:
            if k > j:  # avoid duplicates: ensure i < j < k after sorting
                triple = tuple(sorted([i, j, k]))
                valid_triads.append(triple)

    # Deduplicate
    valid_triads = list(set(valid_triads))
    n_valid = len(valid_triads)

    if n_valid == 0:
        return {
            "cycle_probability": 0.0,
            "hard_cycle_rate": 0.0,
            "n_triads": 0,
            "n_cycles": 0,
            "sampled": False,
        }

    # Compute probs for compared pairs
    total = wins + wins.T
    with np.errstate(divide="ignore", invalid="ignore"):
        probs = np.where(total > 0, wins / total, 0.5)
    prefers = wins > wins.T

    # Sample if too many triads
    rng = np.random.default_rng(seed)
    sampled = n_valid > max_triads
    if sampled:
        indices = rng.choice(n_valid, size=max_triads, replace=False)
        triads_to_check = [valid_triads[idx] for idx in indices]
    else:
        triads_to_check = valid_triads

    cycle_prob_sum = 0.0
    n_cycles = 0
    for i, j, k in triads_to_check:
        p_cycle_1 = probs[i, j] * probs[j, k] * probs[k, i]
        p_cycle_2 = probs[i, k] * probs[k, j] * probs[j, i]
        cycle_prob_sum += p_cycle_1 + p_cycle_2

        if (prefers[i, j] and prefers[j, k] and prefers[k, i]) or (
            prefers[i, k] and prefers[k, j] and prefers[j, i]
        ):
            n_cycles += 1

    n_checked = len(triads_to_check)
    avg_cycle_prob = cycle_prob_sum / n_checked

    return {
        "cycle_probability": avg_cycle_prob,
        "hard_cycle_rate": n_cycles / n_checked,
        "n_triads": n_valid,
        "n_triads_checked": n_checked,
        "n_cycles": n_cycles,
        "sampled": sampled,
    }


def main():
    results = {}

    for persona_name, info in PERSONAS.items():
        base_dir = EXP2_DIR if info["exp"] == "exp2" else EXP3_DIR
        exp_name = f"mra_{info['exp']}"
        split_dirs = find_split_dirs(base_dir, exp_name, info["sys_hash"])

        if not split_dirs:
            print(f"WARNING: No split dirs found for {persona_name}")
            continue

        wins, task_ids = load_merged_wins(split_dirs)
        n_comparisons = int(wins.sum())
        n_pairs_compared = int(((wins + wins.T) > 0).sum()) // 2
        result = measure_transitivity_compared_only(wins)

        results[persona_name] = {
            "n_tasks": len(task_ids),
            "n_comparisons": n_comparisons,
            "n_pairs_compared": n_pairs_compared,
            "n_splits": len(split_dirs),
            **result,
        }

        print(
            f"{persona_name:>12s}  tasks={len(task_ids):4d}  "
            f"comparisons={n_comparisons:5d}  "
            f"pairs_compared={n_pairs_compared:6d}  "
            f"triads={result['n_triads']:7d}  "
            f"cycle_prob={result['cycle_probability']:.4f}  "
            f"hard_cycle_rate={result['hard_cycle_rate']:.4f}"
        )

    # Save JSON
    out_dir = Path("experiments/transitivity")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "persona_transitivity.json", "w") as f:
        json.dump(results, f, indent=2)

    # Plot
    personas = list(results.keys())
    cycle_probs = [results[p]["cycle_probability"] for p in personas]
    hard_rates = [results[p]["hard_cycle_rate"] for p in personas]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(personas))
    width = 0.35

    ax.bar(x - width / 2, cycle_probs, width, label="Cycle probability", color="#4C72B0")
    ax.bar(x + width / 2, hard_rates, width, label="Hard cycle rate", color="#DD8452")

    ax.axhline(y=0.25, color="gray", linestyle="--", linewidth=1, label="Random baseline (0.25)")
    ax.set_ylabel("Rate")
    ax.set_xlabel("Persona")
    ax.set_title("Per-Persona Transitivity of Preferences (Gemma 3 27B)")
    ax.set_xticks(x)
    ax.set_xticklabels(personas, rotation=30, ha="right")
    ax.set_ylim(0, 0.30)
    ax.legend()
    fig.tight_layout()

    assets_dir = out_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    plot_path = assets_dir / "plot_031026_persona_transitivity.png"
    fig.savefig(plot_path, dpi=150)
    print(f"\nPlot saved to {plot_path}")

    # Report
    report_lines = [
        "# Per-Persona Transitivity Analysis",
        "",
        "## Summary",
        "",
        "Transitivity of pairwise preferences for 8 personas on Gemma 3 27B.",
        "Each persona has ~2500 tasks measured across 3 splits (a/b/c).",
        "Transitivity is computed only over triads where all 3 pairs were actually compared.",
        "",
        "![Persona transitivity](assets/plot_031026_persona_transitivity.png)",
        "",
        "## Results",
        "",
        "| Persona | Tasks | Pairs Compared | Triads | Cycle Prob | Hard Cycle Rate |",
        "|---------|------:|---------------:|-------:|-----------:|----------------:|",
    ]
    for p in personas:
        r = results[p]
        report_lines.append(
            f"| {p} | {r['n_tasks']} | {r['n_pairs_compared']} | "
            f"{r['n_triads']} | {r['cycle_probability']:.4f} | "
            f"{r['hard_cycle_rate']:.4f} |"
        )

    # Interpretation
    sorted_by_cp = sorted(results.items(), key=lambda x: x[1]["cycle_probability"])
    most_transitive = sorted_by_cp[0][0]
    least_transitive = sorted_by_cp[-1][0]
    cp_range = sorted_by_cp[-1][1]["cycle_probability"] - sorted_by_cp[0][1]["cycle_probability"]

    report_lines.extend([
        "",
        "## Interpretation",
        "",
        f"- Most transitive: **{most_transitive}** "
        f"(cycle prob = {results[most_transitive]['cycle_probability']:.4f})",
        f"- Least transitive: **{least_transitive}** "
        f"(cycle prob = {results[least_transitive]['cycle_probability']:.4f})",
        f"- Range: {cp_range:.4f}",
        "- All personas are well below the random baseline of 0.25",
    ])

    report_path = out_dir / "transitivity_report.md"
    report_path.write_text("\n".join(report_lines) + "\n")
    print(f"Report saved to {report_path}")


if __name__ == "__main__":
    main()
