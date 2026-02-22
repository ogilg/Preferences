"""Ground truth analysis: how well do behavioral and probe deltas track expected direction?

For each (condition, task) pair, assigns a ground truth label:
  +1 = condition should increase preference for this task
  -1 = condition should decrease preference
   0 = no clear prediction (off-target)

Then fits OLS: ground_truth -> delta, reports R² for behavioral and probe deltas.

Usage: python -m scripts.ood_system_prompts.analyze_ground_truth [--layer LAYER]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats

REPO_ROOT = Path(__file__).parent.parent.parent
ACTS_DIR = REPO_ROOT / "activations" / "ood"
PROBE_DIR = REPO_ROOT / "results" / "probes" / "gemma3_10k_heldout_std_demean" / "probes"
RESULTS_OOD = REPO_ROOT / "results" / "ood"
CONFIGS = REPO_ROOT / "configs" / "ood"

# Exp 3: task_id -> set of target interests it's relevant to (high bar, manually tagged)
EXP3_TASK_TARGETS: dict[str, set[str]] = {
    "alpaca_14631": {"shakespeare"},       # describe Romeo and Juliet plot
    "stresstest_73_1202_value1": {"lotr"},  # Fellowship military strategy
    "stresstest_54_530_neutral": {"chess"}, # live chess tournament
    "alpaca_13003": {"convexhull"},         # implement convex hull in Python
    "alpaca_3808": {"detective"},           # create a detective riddle
    # stresstest_92_2_value2 removed: adversarial stress-test prompt (demands model reveal
    # internal weights/activations), baseline p_choose=0.041 creates floor effect
    "alpaca_13255": {"haiku"},             # make a poetic haiku
    "stresstest_89_193_value1": {"haiku"}, # CRISPR through haiku
    "alpaca_5529": {"pyramids"},           # how Ancient Egyptians built pyramids
    "wildchat_35599": {"simpsons"},        # write a Simpsons episode
    "stresstest_43_948_value2": {"wwii"},  # causes of WWII essay
}


def probe_path(layer: int) -> Path:
    return PROBE_DIR / f"probe_ridge_L{layer}.npy"


def _load_condition_metadata(path: Path) -> dict[str, dict]:
    """Load {condition_id: {category, direction, ...}} from prompt config."""
    data = json.load(open(path))
    return {c["condition_id"]: c for c in data["conditions"]}


def _load_full_results(layer: int) -> dict:
    path = REPO_ROOT / "experiments" / "ood_system_prompts" / "analysis_results_full.json"
    data = json.load(open(path))
    return data


def _ground_truth_exp1a(condition_labels: np.ndarray, task_ids: list[str]) -> np.ndarray:
    """Exp 1a: +1 if pos condition matches task category, -1 if neg matches, 0 otherwise."""
    cond_meta = _load_condition_metadata(CONFIGS / "prompts" / "category_preference.json")
    cat_tasks: dict[str, set[str]] = {}
    for cat, tids in json.load(open(CONFIGS / "tasks" / "category_tasks.json")).items():
        cat_tasks[cat] = set(tids)

    # Build task_id -> category lookup
    task_to_cat: dict[str, str] = {}
    for cat, tids in cat_tasks.items():
        for tid in tids:
            task_to_cat[tid] = cat

    gt = np.zeros(len(condition_labels))
    for i, (cid, tid) in enumerate(zip(condition_labels, task_ids)):
        meta = cond_meta[cid]
        task_cat = task_to_cat.get(tid)
        if task_cat is None or meta["category"] != task_cat:
            gt[i] = 0
        elif meta["direction"] == "pos":
            gt[i] = 1
        else:
            gt[i] = -1
    return gt


def _ground_truth_exp1b(condition_labels: np.ndarray, task_ids: list[str]) -> np.ndarray:
    """Exp 1b: +1 if pos condition matches task topic, -1 if neg matches, 0 otherwise."""
    cond_meta = _load_condition_metadata(CONFIGS / "prompts" / "targeted_preference.json")
    tasks = json.load(open(CONFIGS / "tasks" / "target_tasks.json"))
    task_to_topic = {t["task_id"]: t["topic"] for t in tasks}

    gt = np.zeros(len(condition_labels))
    for i, (cid, tid) in enumerate(zip(condition_labels, task_ids)):
        meta = cond_meta[cid]
        task_topic = task_to_topic.get(tid)
        if task_topic is None or meta["category"] != task_topic:
            gt[i] = 0
        elif meta["direction"] == "pos":
            gt[i] = 1
        else:
            gt[i] = -1
    return gt


def _ground_truth_exp1c(condition_labels: np.ndarray, task_ids: list[str]) -> np.ndarray:
    """Exp 1c: same as 1b but matching on crossed task's topic field."""
    cond_meta = _load_condition_metadata(CONFIGS / "prompts" / "targeted_preference.json")
    tasks = json.load(open(CONFIGS / "tasks" / "crossed_tasks.json"))
    task_to_topic = {t["task_id"]: t["topic"] for t in tasks}

    gt = np.zeros(len(condition_labels))
    for i, (cid, tid) in enumerate(zip(condition_labels, task_ids)):
        meta = cond_meta[cid]
        task_topic = task_to_topic.get(tid)
        if task_topic is None or meta["category"] != task_topic:
            gt[i] = 0
        elif meta["direction"] == "pos":
            gt[i] = 1
        else:
            gt[i] = -1
    return gt


def _ground_truth_exp1d(condition_labels: np.ndarray, task_ids: list[str]) -> np.ndarray:
    """Exp 1d: per-(condition, task) ground truth based on topic/shell matching.

    Each competing condition has a loved and hated dimension.
    love_subject: loves subject (topic), hates task_type (shell)
    love_task_type: loves task_type (shell), hates subject (topic)

    For each crossed task (which has topic + category_shell):
      - task topic matches loved dimension → +1
      - task category_shell matches hated dimension → -1
      - task matches BOTH loved and hated → 0 (conflicted)
      - task matches neither → 0
    """
    cond_meta = _load_condition_metadata(CONFIGS / "prompts" / "competing_preference.json")
    tasks = json.load(open(CONFIGS / "tasks" / "crossed_tasks.json"))
    task_lookup = {t["task_id"]: t for t in tasks}

    gt = np.zeros(len(condition_labels))
    for i, (cid, tid) in enumerate(zip(condition_labels, task_ids)):
        meta = cond_meta[cid]
        task = task_lookup.get(tid)
        if task is None:
            continue

        subject = meta["subject"]
        task_type = meta["task_type"]
        direction = meta["direction"]

        # Determine which dimension is loved vs hated
        if direction == "love_subject":
            loved_topic, hated_shell = subject, task_type
        else:  # love_task_type
            loved_topic, hated_shell = None, None
            # love_task_type: loves the shell activity (math), hates the subject (cheese)
            # So tasks about math → +1, tasks about cheese → -1
            # But tasks are crossed: they have topic (content) and category_shell (activity type)
            # A task with category_shell matching task_type → loved activity → +1
            # A task with topic matching subject → hated content → -1

        topic_matches_subject = task["topic"] == subject
        shell_matches_task_type = task["category_shell"] == task_type

        if topic_matches_subject and shell_matches_task_type:
            # Conflicted: the crossed task has both dimensions
            gt[i] = 0
        elif direction == "love_subject":
            if topic_matches_subject:
                gt[i] = 1  # loved topic content
            elif shell_matches_task_type:
                gt[i] = -1  # hated activity type
        else:  # love_task_type
            if shell_matches_task_type:
                gt[i] = 1  # loved activity type
            elif topic_matches_subject:
                gt[i] = -1  # hated topic content

    return gt


def _ground_truth_exp3(condition_labels: np.ndarray, task_ids: list[str]) -> np.ndarray:
    """Exp 3: +1 for version A on target-relevant task, -1 for version C, 0 otherwise."""
    cond_meta = _load_condition_metadata(CONFIGS / "prompts" / "minimal_pairs_v7.json")

    gt = np.zeros(len(condition_labels))
    for i, (cid, tid) in enumerate(zip(condition_labels, task_ids)):
        meta = cond_meta[cid]
        target = meta["target"]
        version = meta["version"]
        task_targets = EXP3_TASK_TARGETS.get(tid, set())
        if target not in task_targets:
            gt[i] = 0
        elif version == "A":
            gt[i] = 1
        elif version == "C":
            gt[i] = -1
        else:
            gt[i] = 0  # version B = neutral
    return gt


def _ols_r_squared(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """OLS regression of x -> y. Returns (R², slope, intercept)."""
    slope, intercept, r, p, se = stats.linregress(x, y)
    return r ** 2, slope, intercept


def analyze_experiment(
    name: str,
    behavioral: np.ndarray,
    probe: np.ndarray,
    ground_truth: np.ndarray,
    layer: int,
) -> dict:
    n = len(behavioral)
    n_on_target = int(np.sum(ground_truth != 0))
    n_pos = int(np.sum(ground_truth > 0))
    n_neg = int(np.sum(ground_truth < 0))

    beh_r2, beh_slope, _ = _ols_r_squared(ground_truth, behavioral)
    probe_r2, probe_slope, _ = _ols_r_squared(ground_truth, probe)

    # Also compute on on-target only (exclude gt=0)
    on_mask = ground_truth != 0
    gt_on = ground_truth[on_mask]
    has_variance = on_mask.sum() >= 3 and np.unique(gt_on).size > 1
    if has_variance:
        beh_r2_on, _, _ = _ols_r_squared(gt_on, behavioral[on_mask])
        probe_r2_on, _, _ = _ols_r_squared(gt_on, probe[on_mask])
    else:
        beh_r2_on = float("nan")
        probe_r2_on = float("nan")

    # Sign agreement on on-target pairs
    if on_mask.sum() > 0:
        beh_sign = float(np.mean(np.sign(behavioral[on_mask]) == ground_truth[on_mask]))
        probe_sign = float(np.mean(np.sign(probe[on_mask]) == ground_truth[on_mask]))
    else:
        beh_sign = float("nan")
        probe_sign = float("nan")

    # Behavioral vs probe correlation (all and on-target)
    beh_probe_r_all = float(stats.pearsonr(behavioral, probe)[0])
    if on_mask.sum() >= 3:
        beh_probe_r_on = float(stats.pearsonr(behavioral[on_mask], probe[on_mask])[0])
    else:
        beh_probe_r_on = float("nan")

    # Beh↔probe sign agreement (all, with threshold, and on-target)
    threshold = 0.02
    beh_thresh_mask = np.abs(behavioral) >= threshold
    if beh_thresh_mask.sum() > 0:
        beh_probe_sign_all = float(np.mean(
            np.sign(behavioral[beh_thresh_mask]) == np.sign(probe[beh_thresh_mask])
        ))
        beh_probe_sign_all_n = int(beh_thresh_mask.sum())
    else:
        beh_probe_sign_all = float("nan")
        beh_probe_sign_all_n = 0

    on_thresh_mask = on_mask & beh_thresh_mask
    if on_thresh_mask.sum() > 0:
        beh_probe_sign_on = float(np.mean(
            np.sign(behavioral[on_thresh_mask]) == np.sign(probe[on_thresh_mask])
        ))
        beh_probe_sign_on_n = int(on_thresh_mask.sum())
    else:
        beh_probe_sign_on = float("nan")
        beh_probe_sign_on_n = 0

    # Ground truth correlation (r, not just R²)
    beh_gt_r_all = float(stats.pearsonr(ground_truth, behavioral)[0])
    probe_gt_r_all = float(stats.pearsonr(ground_truth, probe)[0])
    if has_variance:
        beh_gt_r_on = float(stats.pearsonr(gt_on, behavioral[on_mask])[0])
        probe_gt_r_on = float(stats.pearsonr(gt_on, probe[on_mask])[0])
    else:
        beh_gt_r_on = float("nan")
        probe_gt_r_on = float("nan")

    print(f"\n  {name} L{layer}: n={n}, on-target={n_on_target} (+{n_pos}/-{n_neg})")
    print(f"    Beh↔Probe:  r={beh_probe_r_all:.3f} (all), r={beh_probe_r_on:.3f} (on-target)")
    print(f"                sign={beh_probe_sign_all:.1%} (all, n={beh_probe_sign_all_n}), sign={beh_probe_sign_on:.1%} (on-target, n={beh_probe_sign_on_n})")
    print(f"    Beh→GT:     r={beh_gt_r_all:.3f} (all), r={beh_gt_r_on:.3f} (on-target), sign={beh_sign:.1%}")
    print(f"    Probe→GT:   r={probe_gt_r_all:.3f} (all), r={probe_gt_r_on:.3f} (on-target), sign={probe_sign:.1%}")

    return {
        "layer": layer,
        "n": n,
        "n_on_target": n_on_target,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "behavioral_r2": beh_r2,
        "behavioral_r2_on_target": beh_r2_on,
        "behavioral_slope": beh_slope,
        "behavioral_sign_agreement": beh_sign,
        "probe_r2": probe_r2,
        "probe_r2_on_target": probe_r2_on,
        "probe_slope": probe_slope,
        "probe_sign_agreement": probe_sign,
        "beh_probe_r_all": beh_probe_r_all,
        "beh_probe_r_on_target": beh_probe_r_on,
        "beh_probe_sign_all": beh_probe_sign_all,
        "beh_probe_sign_all_n": beh_probe_sign_all_n,
        "beh_probe_sign_on_target": beh_probe_sign_on,
        "beh_probe_sign_on_target_n": beh_probe_sign_on_n,
        "beh_gt_r_all": beh_gt_r_all,
        "beh_gt_r_on_target": beh_gt_r_on,
        "probe_gt_r_all": probe_gt_r_all,
        "probe_gt_r_on_target": probe_gt_r_on,
    }


def run_experiment(exp_name: str, full_results: dict, layer: int) -> dict:
    layer_key = f"L{layer}"

    gt_fn = {
        "exp1a": _recompute_with_ground_truth_1a,
        "exp1b": _recompute_with_ground_truth_1b,
        "exp1c": _recompute_with_ground_truth_1c,
        "exp1d": _recompute_with_ground_truth_1d,
        "exp3": _recompute_with_ground_truth_3,
    }[exp_name]

    return gt_fn(layer)


def _recompute_with_task_ids(
    rates: dict[str, dict[str, float]],
    acts_dir: Path,
    layer: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Like compute_deltas but also returns task_ids."""
    from src.ood.analysis import _split_probe, _score_activations

    weights, bias = _split_probe(probe_path(layer))
    baseline_rates = rates["baseline"]
    baseline_npz = acts_dir / "baseline" / "activations_prompt_last.npz"
    baseline_scores = _score_activations(baseline_npz, layer, weights, bias)

    all_behavioral, all_probe, all_labels, all_task_ids = [], [], [], []

    for cid in rates:
        if cid == "baseline":
            continue
        cond_npz = acts_dir / cid / "activations_prompt_last.npz"
        if not cond_npz.exists():
            continue
        cond_scores = _score_activations(cond_npz, layer, weights, bias)

        for tid, cond_rate in rates[cid].items():
            if tid not in baseline_rates or tid not in baseline_scores or tid not in cond_scores:
                continue
            all_behavioral.append(cond_rate - baseline_rates[tid])
            all_probe.append(cond_scores[tid] - baseline_scores[tid])
            all_labels.append(cid)
            all_task_ids.append(tid)

    return (
        np.array(all_behavioral),
        np.array(all_probe),
        np.array(all_labels),
        all_task_ids,
    )


def _recompute_with_ground_truth_1a(layer: int) -> dict:
    from src.ood.analysis import compute_p_choose_from_pairwise

    pairwise = json.load(open(RESULTS_OOD / "category_preference" / "pairwise.json"))
    rates = compute_p_choose_from_pairwise(pairwise["results"])
    acts_dir = ACTS_DIR / "exp1_category"

    beh, probe, labels, task_ids = _recompute_with_task_ids(rates, acts_dir, layer)
    gt = _ground_truth_exp1a(labels, task_ids)
    return analyze_experiment("exp1a", beh, probe, gt, layer)


def _recompute_with_ground_truth_1b(layer: int) -> dict:
    from src.ood.analysis import compute_p_choose_from_pairwise

    pairwise = json.load(open(RESULTS_OOD / "hidden_preference" / "pairwise.json"))
    rates = compute_p_choose_from_pairwise(pairwise["results"])
    targeted_rates = {k: v for k, v in rates.items() if not k.startswith("compete_")}
    targeted_rates = {
        k: {tid: v for tid, v in rd.items() if tid.startswith("hidden_")}
        for k, rd in targeted_rates.items()
    }
    acts_dir = ACTS_DIR / "exp1_prompts"

    beh, probe, labels, task_ids = _recompute_with_task_ids(targeted_rates, acts_dir, layer)
    gt = _ground_truth_exp1b(labels, task_ids)
    return analyze_experiment("exp1b", beh, probe, gt, layer)


def _recompute_with_ground_truth_1c(layer: int) -> dict:
    from src.ood.analysis import compute_p_choose_from_pairwise

    pairwise = json.load(open(RESULTS_OOD / "crossed_preference" / "pairwise.json"))
    rates = compute_p_choose_from_pairwise(pairwise["results"])
    targeted_rates = {k: v for k, v in rates.items() if not k.startswith("compete_")}
    targeted_rates = {
        k: {tid: v for tid, v in rd.items() if tid.startswith("crossed_")}
        for k, rd in targeted_rates.items()
    }
    acts_dir = ACTS_DIR / "exp1_prompts"

    beh, probe, labels, task_ids = _recompute_with_task_ids(targeted_rates, acts_dir, layer)
    gt = _ground_truth_exp1c(labels, task_ids)
    return analyze_experiment("exp1c", beh, probe, gt, layer)


def _recompute_with_ground_truth_1d(layer: int) -> dict:
    from src.ood.analysis import compute_p_choose_from_pairwise

    pairwise = json.load(open(RESULTS_OOD / "crossed_preference" / "pairwise.json"))
    rates = compute_p_choose_from_pairwise(pairwise["results"])

    # Use all crossed tasks × competing conditions (full grid)
    competing_rates = {k: v for k, v in rates.items()
                       if k.startswith("compete_") or k == "baseline"}
    competing_rates = {
        k: {tid: v for tid, v in rate_dict.items() if tid.startswith("crossed_")}
        for k, rate_dict in competing_rates.items()
    }
    acts_dir = ACTS_DIR / "exp1_prompts"

    beh, probe, labels, task_ids = _recompute_with_task_ids(competing_rates, acts_dir, layer)
    gt = _ground_truth_exp1d(labels, task_ids)
    return analyze_experiment("exp1d", beh, probe, gt, layer)


def _recompute_with_ground_truth_3(layer: int) -> dict:
    mp_cfg = json.load(open(CONFIGS / "prompts" / "minimal_pairs_v7.json"))
    selected_roles = {"midwest", "brooklyn"}
    selected_versions = {"A", "B", "C"}
    selected_cids = {
        c["condition_id"]
        for c in mp_cfg["conditions"]
        if c["base_role"] in selected_roles and c["version"] in selected_versions
    }
    selected_cids.add("baseline")

    beh_data = json.load(open(RESULTS_OOD / "minimal_pairs_v7" / "behavioral.json"))
    rates: dict[str, dict[str, float]] = {}
    for cid, cond_data in beh_data["conditions"].items():
        if cid not in selected_cids:
            continue
        rates[cid] = {tid: v["p_choose"] for tid, v in cond_data["task_rates"].items()}

    acts_dir = ACTS_DIR / "exp3_minimal_pairs"
    beh, probe, labels, task_ids = _recompute_with_task_ids(rates, acts_dir, layer)
    gt = _ground_truth_exp3(labels, task_ids)
    return analyze_experiment("exp3", beh, probe, gt, layer)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, default=31)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    layer = args.layer
    results = {}

    print(f"=== Ground Truth Analysis (L{layer}) ===")

    for exp_name, fn in [
        ("exp1a", _recompute_with_ground_truth_1a),
        ("exp1b", _recompute_with_ground_truth_1b),
        ("exp1c", _recompute_with_ground_truth_1c),
        ("exp1d", _recompute_with_ground_truth_1d),
        ("exp3", _recompute_with_ground_truth_3),
    ]:
        try:
            results[exp_name] = fn(layer)
        except Exception as e:
            print(f"  {exp_name}: ERROR — {e}")
            import traceback
            traceback.print_exc()

    output_path = args.output or (
        REPO_ROOT / "experiments" / "ood_system_prompts" / "ground_truth_results.json"
    )
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
