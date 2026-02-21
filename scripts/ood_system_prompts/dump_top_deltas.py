"""Dump per-condition top/bottom delta examples for all OOD experiments to JSON.

For each experiment, condition, and task: behavioral delta, probe delta, task prompt.
Sorted by probe delta within each condition. Output: one JSON per experiment.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.ood.analysis import (
    compute_p_choose_from_pairwise,
)
from src.task_data import load_tasks, OriginDataset

REPO_ROOT = Path(__file__).parent.parent.parent
ACTS_DIR = REPO_ROOT / "activations" / "ood"
PROBE_DIR = REPO_ROOT / "results" / "probes" / "gemma3_10k_heldout_std_demean" / "probes"
RESULTS_OOD = REPO_ROOT / "results" / "ood"
OUTPUT_DIR = REPO_ROOT / "experiments" / "ood_system_prompts" / "task_examples"
LAYER = 31


def load_probe():
    probe = np.load(PROBE_DIR / f"probe_ridge_L{LAYER}.npy")
    return probe[:-1], float(probe[-1])


def score_activations(npz_path: Path, weights: np.ndarray, bias: float):
    data = np.load(npz_path, allow_pickle=True)
    acts = data[f"layer_{LAYER}"]
    scores = acts @ weights + bias
    task_ids = list(data["task_ids"])
    return dict(zip(task_ids, scores.tolist()))


def load_prompt_lookup() -> dict[str, str]:
    all_tasks = load_tasks(
        n=100000,
        origins=[
            OriginDataset.WILDCHAT, OriginDataset.ALPACA, OriginDataset.MATH,
            OriginDataset.BAILBENCH, OriginDataset.STRESS_TEST,
        ],
    )
    lookup = {t.id: t.prompt for t in all_tasks}
    for tasks_file in [
        REPO_ROOT / "configs/ood/tasks/target_tasks.json",
        REPO_ROOT / "configs/ood/tasks/crossed_tasks.json",
    ]:
        if tasks_file.exists():
            custom = json.load(open(tasks_file))
            for t in custom:
                lookup[t["task_id"]] = t["prompt"]
    return lookup


def compute_records(
    rates: dict[str, dict[str, float]],
    acts_dir: Path,
    weights: np.ndarray,
    bias: float,
    baseline_key: str = "baseline",
) -> list[dict]:
    baseline_rates = rates[baseline_key]
    baseline_npz = acts_dir / baseline_key / "activations_prompt_last.npz"
    baseline_scores = score_activations(baseline_npz, weights, bias)

    records = []
    for cid in rates:
        if cid == baseline_key:
            continue
        cond_npz = acts_dir / cid / "activations_prompt_last.npz"
        if not cond_npz.exists():
            continue
        cond_scores = score_activations(cond_npz, weights, bias)

        for tid, cond_rate in rates[cid].items():
            if tid not in baseline_rates or tid not in baseline_scores or tid not in cond_scores:
                continue
            records.append({
                "condition": cid,
                "task_id": tid,
                "beh_delta": round(cond_rate - baseline_rates[tid], 4),
                "probe_delta": round(cond_scores[tid] - baseline_scores[tid], 4),
            })
    return records


def build_output(
    records: list[dict],
    prompts: dict[str, str],
    system_prompts: dict[str, str],
    top_n: int = 3,
) -> dict:
    """Structure: {condition: {system_prompt, top_probe, bottom_probe}} with task prompts."""
    from collections import defaultdict
    by_cond: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_cond[r["condition"]].append(r)

    output = {}
    for cid in sorted(by_cond):
        cond_records = sorted(by_cond[cid], key=lambda r: r["probe_delta"], reverse=True)

        def enrich(r: dict) -> dict:
            prompt = prompts.get(r["task_id"], "?")
            return {
                "task_id": r["task_id"],
                "task_prompt": prompt[:200],
                "beh_delta": r["beh_delta"],
                "probe_delta": r["probe_delta"],
            }

        output[cid] = {
            "system_prompt": system_prompts.get(cid, ""),
            "n_tasks": len(cond_records),
            "top_probe": [enrich(r) for r in cond_records[:top_n]],
            "bottom_probe": [enrich(r) for r in cond_records[-top_n:]],
        }

    return output


def load_system_prompts(cfg_path: Path) -> dict[str, str]:
    cfg = json.load(open(cfg_path))
    return {c["condition_id"]: c["system_prompt"] for c in cfg["conditions"]}


def main():
    weights, bias = load_probe()
    print("Loading task prompts...")
    prompts = load_prompt_lookup()
    print(f"Loaded {len(prompts)} task prompts")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Exp 1a ---
    print("\n=== Exp 1a ===")
    pairwise = json.load(open(RESULTS_OOD / "category_preference" / "pairwise.json"))
    rates = compute_p_choose_from_pairwise(pairwise["results"])
    records = compute_records(rates, ACTS_DIR / "exp1_category", weights, bias)
    sys_prompts = load_system_prompts(REPO_ROOT / "configs/ood/prompts/category_preference.json")
    out = build_output(records, prompts, sys_prompts)
    path = OUTPUT_DIR / "exp1a_category.json"
    json.dump(out, open(path, "w"), indent=2)
    print(f"Wrote {path} ({len(out)} conditions)")

    # --- Exp 1b ---
    print("\n=== Exp 1b ===")
    pairwise = json.load(open(RESULTS_OOD / "hidden_preference" / "pairwise.json"))
    rates = compute_p_choose_from_pairwise(pairwise["results"])
    targeted_rates = {k: v for k, v in rates.items() if not k.startswith("compete_")}
    targeted_rates = {
        k: {tid: v for tid, v in rd.items() if tid.startswith("hidden_")}
        for k, rd in targeted_rates.items()
    }
    records = compute_records(targeted_rates, ACTS_DIR / "exp1_prompts", weights, bias)
    sys_prompts = load_system_prompts(REPO_ROOT / "configs/ood/prompts/targeted_preference.json")
    out = build_output(records, prompts, sys_prompts)
    path = OUTPUT_DIR / "exp1b_hidden.json"
    json.dump(out, open(path, "w"), indent=2)
    print(f"Wrote {path} ({len(out)} conditions)")

    # --- Exp 1c ---
    print("\n=== Exp 1c ===")
    pairwise = json.load(open(RESULTS_OOD / "crossed_preference" / "pairwise.json"))
    rates = compute_p_choose_from_pairwise(pairwise["results"])
    targeted_rates = {k: v for k, v in rates.items() if not k.startswith("compete_")}
    crossed_tids = set(
        t for rd in targeted_rates.values() for t in rd if t.startswith("crossed_")
    )
    if crossed_tids:
        targeted_rates = {
            k: {tid: v for tid, v in rd.items() if tid.startswith("crossed_")}
            for k, rd in targeted_rates.items()
        }
    records = compute_records(targeted_rates, ACTS_DIR / "exp1_prompts", weights, bias)
    out = build_output(records, prompts, sys_prompts)
    path = OUTPUT_DIR / "exp1c_crossed.json"
    json.dump(out, open(path, "w"), indent=2)
    print(f"Wrote {path} ({len(out)} conditions)")

    # --- Exp 1d ---
    print("\n=== Exp 1d ===")
    pairwise = json.load(open(RESULTS_OOD / "crossed_preference" / "pairwise.json"))
    rates = compute_p_choose_from_pairwise(pairwise["results"])
    competing_rates = {
        k: {tid: v for tid, v in rd.items() if tid.startswith("crossed_")}
        for k, rd in rates.items()
        if k.startswith("compete_") or k == "baseline"
    }
    records = compute_records(competing_rates, ACTS_DIR / "exp1_prompts", weights, bias)
    compete_sys = load_system_prompts(REPO_ROOT / "configs/ood/prompts/competing_preference.json")
    out = build_output(records, prompts, {**sys_prompts, **compete_sys})
    path = OUTPUT_DIR / "exp1d_competing.json"
    json.dump(out, open(path, "w"), indent=2)
    print(f"Wrote {path} ({len(out)} conditions)")

    # --- Exp 2 ---
    print("\n=== Exp 2 ===")
    rp_beh = json.load(open(RESULTS_OOD / "role_playing" / "behavioral.json"))
    rp_rates = {cid: {tid: v["p_choose"] for tid, v in cd["task_rates"].items()}
                for cid, cd in rp_beh["conditions"].items()}
    np_beh = json.load(open(RESULTS_OOD / "narrow_preference" / "behavioral.json"))
    np_rates = {cid: {tid: v["p_choose"] for tid, v in cd["task_rates"].items()}
                for cid, cd in np_beh["conditions"].items()}
    exp2_rates = {**rp_rates, **np_rates}
    exp2_rates["baseline"] = rp_rates["baseline"]
    records = compute_records(exp2_rates, ACTS_DIR / "exp2_roles", weights, bias)
    rp_sys = load_system_prompts(REPO_ROOT / "configs/ood/prompts/role_playing.json")
    np_sys = load_system_prompts(REPO_ROOT / "configs/ood/prompts/narrow_preference.json")
    out = build_output(records, prompts, {**rp_sys, **np_sys})
    path = OUTPUT_DIR / "exp2_roles.json"
    json.dump(out, open(path, "w"), indent=2)
    print(f"Wrote {path} ({len(out)} conditions)")

    # --- Exp 3 ---
    print("\n=== Exp 3 ===")
    mp_beh = json.load(open(RESULTS_OOD / "minimal_pairs_v7" / "behavioral.json"))
    mp_rates = {cid: {tid: v["p_choose"] for tid, v in cd["task_rates"].items()}
                for cid, cd in mp_beh["conditions"].items()}
    mp_cfg = json.load(open(REPO_ROOT / "configs/ood/prompts/minimal_pairs_v7.json"))
    selected_cids = {
        c["condition_id"]
        for c in mp_cfg["conditions"]
        if c["base_role"] in {"midwest", "brooklyn"} and c["version"] in {"A", "B"}
    }
    selected_cids.add("baseline")
    mp_rates = {k: v for k, v in mp_rates.items() if k in selected_cids}
    records = compute_records(mp_rates, ACTS_DIR / "exp3_minimal_pairs", weights, bias)
    mp_sys = {c["condition_id"]: c["system_prompt"] for c in mp_cfg["conditions"]}
    out = build_output(records, prompts, mp_sys)
    path = OUTPUT_DIR / "exp3_minimal_pairs.json"
    json.dump(out, open(path, "w"), indent=2)
    print(f"Wrote {path} ({len(out)} conditions)")

    print(f"\nAll files written to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
