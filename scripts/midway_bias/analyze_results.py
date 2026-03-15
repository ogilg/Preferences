"""Analyze midway bias results: summary tables, per-topic diagnostics, and plots."""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

RESULTS_PATH = Path("results/experiments/mra_exp3/midway_bias/midway_bias_results.json")
ASSETS = Path("experiments/probe_generalization/multi_role_ablation/assets")

NON_DEFAULT = ["villain", "aesthete", "midwest", "provocateur", "trickster", "autocrat", "sadist"]
SELECTORS = ["turn_boundary:-2", "turn_boundary:-5"]
LAYERS = [25, 32, 39, 46, 53]
FOCUS_TOPICS = {"harmful_request", "math", "knowledge_qa", "fiction", "coding", "content_generation"}


def load_results():
    with open(RESULTS_PATH) as f:
        return json.load(f)


def analyze_topic_stability(results):
    """Check which topics have stable midway ratios vs extreme outliers."""
    print("=" * 80)
    print("TOPIC DIAGNOSTICS: denominator sizes and midway ratio distributions")
    print("=" * 80)

    # Gather all topic-level data points
    topic_data = defaultdict(list)
    topic_denoms = defaultdict(list)
    for entry in results:
        if entry["eval_persona"] == "noprompt":
            continue
        for topic, tdata in entry["topics"].items():
            ratio = tdata["midway_ratio"]
            denom = tdata["actual_mean"] - tdata["anchor_mean"]
            topic_data[topic].append(ratio)
            topic_denoms[topic].append(abs(denom))

    print(f"\n{'Topic':<25} {'N pts':>6} {'Med |denom|':>12} {'Mean ratio':>12} {'Med ratio':>12} {'Std ratio':>12} {'|ratio|>2':>10}")
    for topic in sorted(topic_data.keys()):
        ratios = np.array(topic_data[topic])
        denoms = np.array(topic_denoms[topic])
        extreme = np.sum(np.abs(ratios) > 2)
        print(f"{topic:<25} {len(ratios):>6} {np.median(denoms):>12.3f} {np.mean(ratios):>12.3f} {np.median(ratios):>12.3f} {np.std(ratios):>12.3f} {extreme:>10}")


def compute_summary_tables(results, use_median=False, clip_ratio=None):
    """Compute midway ratio summary by N, selector, layer."""
    print("\n" + "=" * 80)
    agg_label = "Median" if use_median else "Mean"
    clip_label = f" (clipped to [{-clip_ratio},{clip_ratio}])" if clip_ratio else ""
    print(f"SUMMARY: {agg_label} midway ratio by N, selector, layer{clip_label}")
    print("=" * 80)

    for selector in SELECTORS:
        for layer in LAYERS:
            subset = [r for r in results if r["selector"] == selector and r["layer"] == layer]
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
                # Only use focus topics
                ratios = []
                for topic, tdata in entry["topics"].items():
                    if topic not in FOCUS_TOPICS:
                        continue
                    r = tdata["midway_ratio"]
                    if clip_ratio is not None:
                        r = np.clip(r, -clip_ratio, clip_ratio)
                    ratios.append(r)
                if ratios:
                    val = float(np.median(ratios) if use_median else np.mean(ratios))
                    agg[(n, dist)].append(val)
                    agg[(n, dist, ep)].append(val)

            central = np.median if use_median else np.mean
            print(f"  {'N':>3}  {'In-dist':>10}  {'OOD':>10}")
            for n in range(1, 9):
                in_vals = agg.get((n, "in"), [])
                ood_vals = agg.get((n, "ood"), [])
                in_str = f"{central(in_vals):.3f}" if in_vals else "n/a"
                ood_str = f"{central(ood_vals):.3f}" if ood_vals else "n/a"
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
                    row += f"  {central(vals):>12.3f}" if vals else f"  {'n/a':>12}"
                print(row)


def compute_pearson_summary(results):
    """Also show Pearson r by N for comparison."""
    print("\n" + "=" * 80)
    print("PEARSON r by N, selector, layer (mean across combos)")
    print("=" * 80)

    for selector in SELECTORS:
        for layer in LAYERS:
            subset = [r for r in results if r["selector"] == selector and r["layer"] == layer]
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
                agg[(n, dist)].append(entry["pearson_r"])

            print(f"  {'N':>3}  {'In-dist':>10}  {'OOD':>10}")
            for n in range(1, 9):
                in_vals = agg.get((n, "in"), [])
                ood_vals = agg.get((n, "ood"), [])
                in_str = f"{np.mean(in_vals):.3f}" if in_vals else "n/a"
                ood_str = f"{np.mean(ood_vals):.3f}" if ood_vals else "n/a"
                print(f"  {n:>3}  {in_str:>10}  {ood_str:>10}")


def compute_grand_summary(results, use_median=False):
    """Average across layers for each selector, focus on high-n topics, median midway ratio."""
    print("\n" + "=" * 80)
    agg_label = "Median" if use_median else "Mean"
    print(f"GRAND SUMMARY: {agg_label} midway ratio across layers, by N and selector")
    print("  (focus topics only, median per entry)")
    print("=" * 80)

    for selector in SELECTORS:
        print(f"\n--- {selector} ---")
        agg_in = defaultdict(list)
        agg_ood = defaultdict(list)

        for entry in results:
            if entry["selector"] != selector:
                continue
            if entry["eval_persona"] == "noprompt":
                continue

            ratios = [tdata["midway_ratio"] for topic, tdata in entry["topics"].items()
                      if topic in FOCUS_TOPICS]
            if not ratios:
                continue

            val = float(np.median(ratios))
            n = entry["n_personas"]
            if entry["is_in_dist"]:
                agg_in[n].append(val)
            else:
                agg_ood[n].append(val)

        central = np.median if use_median else np.mean
        print(f"  {'N':>3}  {'In-dist':>10} {'(n)':>5}  {'OOD':>10} {'(n)':>5}")
        for n in range(1, 9):
            in_vals = agg_in.get(n, [])
            ood_vals = agg_ood.get(n, [])
            in_str = f"{central(in_vals):.3f}" if in_vals else "n/a"
            ood_str = f"{central(ood_vals):.3f}" if ood_vals else "n/a"
            print(f"  {n:>3}  {in_str:>10} {len(in_vals):>5}  {ood_str:>10} {len(ood_vals):>5}")


def main():
    results = load_results()
    print(f"Loaded {len(results)} result entries")

    analyze_topic_stability(results)
    compute_summary_tables(results, use_median=True)
    compute_summary_tables(results, use_median=False)
    compute_pearson_summary(results)
    compute_grand_summary(results, use_median=True)
    compute_grand_summary(results, use_median=False)


if __name__ == "__main__":
    main()
