"""Load all probe results and print summary tables for the turn boundary sweep."""

import json
from pathlib import Path

LAYERS = [25, 32, 39, 46, 53]

# Token labels for each selector (position in turn boundary suffix)
SELECTORS = {
    "tb-5": "<end_of_turn>",
    "tb-4": "\\n",
    "tb-3": "<start_of_turn>",
    "tb-2": "model",
    "tb-1": "\\n (final)",
    "task_mean": "task_mean",
}

def load_heldout_results() -> dict[str, dict[int, float]]:
    """Load final_r per layer from each heldout manifest."""
    results = {}
    for key in SELECTORS:
        dir_name = f"heldout_eval_gemma3_{key}"
        manifest_path = Path(f"results/probes/{dir_name}/manifest.json")
        with open(manifest_path) as f:
            manifest = json.load(f)
        layer_r = {}
        for probe in manifest["probes"]:
            layer_r[probe["layer"]] = probe["final_r"]
        results[key] = layer_r
    return results


def load_hoo_results() -> dict[str, dict[int, float]]:
    """Load mean cross-topic r per layer from each HOO summary."""
    results = {}
    for key in SELECTORS:
        dir_name = f"gemma3_10k_hoo_topic_{key}"
        summary_path = Path(f"results/probes/{dir_name}/hoo_summary.json")
        with open(summary_path) as f:
            summary = json.load(f)
        layer_r = {}
        for layer_str, methods in summary["layer_summary"].items():
            layer_r[int(layer_str)] = methods["ridge"]["mean_hoo_r"]
        results[key] = layer_r
    return results


def print_table(title: str, results: dict[str, dict[int, float]]) -> None:
    print(f"\n{title}")
    print(f"{'Selector':<20} {'Token':<18}", end="")
    for layer in LAYERS:
        print(f"  L{layer:>2}", end="")
    print(f"  {'Best':>6}")
    print("-" * 80)
    for key, label in SELECTORS.items():
        layer_r = results[key]
        best_r = max(layer_r.values())
        print(f"{key:<20} {label:<18}", end="")
        for layer in LAYERS:
            r = layer_r.get(layer, float('nan'))
            marker = " *" if r == best_r else "  "
            print(f" {r:.3f}{marker}", end="")
        print(f"  {best_r:.3f}")


def save_results_json(heldout: dict, hoo: dict) -> None:
    """Save combined results for plotting."""
    output = {
        "layers": LAYERS,
        "selectors": {k: v for k, v in SELECTORS.items()},
        "heldout": {k: {str(layer): r for layer, r in v.items()} for k, v in heldout.items()},
        "hoo": {k: {str(layer): r for layer, r in v.items()} for k, v in hoo.items()},
    }
    out_path = Path("experiments/eot_probes/turn_boundary_sweep/results_summary.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved combined results to {out_path}")


if __name__ == "__main__":
    heldout = load_heldout_results()
    hoo = load_hoo_results()
    print_table("Heldout Eval — Pearson r", heldout)
    print_table("Hold-One-Out by Topic — Mean Cross-Topic r", hoo)
    save_results_json(heldout, hoo)
