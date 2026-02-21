"""Extract activations for all OOD system prompt experiments.

Usage: python scripts/ood_system_prompts/extract_ood_activations.py [--exp EXP]

EXP options: all, exp1a, exp1b_1c_1d, exp2, exp3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from dotenv import load_dotenv

load_dotenv()

REPO_ROOT = Path(__file__).parent.parent.parent
ACTIVATIONS_DIR = REPO_ROOT / "activations" / "ood"
MAIN_ACTIVATIONS = REPO_ROOT / "activations" / "gemma_3_27b" / "activations_prompt_last.npz"
LAYERS = [31, 43, 55]
SELECTOR = "prompt_last"
BATCH_SIZE = 16


def load_main_activations() -> tuple[np.ndarray, np.ndarray]:
    """Load main baseline activations. Returns (task_ids, layer_data)."""
    data = np.load(MAIN_ACTIVATIONS, allow_pickle=True)
    return data["task_ids"], data


def slice_and_save_baseline(
    task_ids_to_slice: list[str],
    main_data,
    output_path: Path,
) -> None:
    """Slice activations for specific task_ids from main activations and save."""
    output_path.mkdir(parents=True, exist_ok=True)
    out_file = output_path / "activations_prompt_last.npz"
    if out_file.exists():
        print(f"  Baseline already exists: {out_file}")
        return

    main_ids = [str(t) for t in main_data["task_ids"]]
    id_to_idx = {tid: i for i, tid in enumerate(main_ids)}
    missing = [t for t in task_ids_to_slice if t not in id_to_idx]
    if missing:
        print(f"  WARNING: {len(missing)} tasks missing from main activations: {missing[:5]}")

    indices = [id_to_idx[t] for t in task_ids_to_slice if t in id_to_idx]
    sliced_ids = np.array([task_ids_to_slice[i] for i, t in enumerate(task_ids_to_slice) if t in id_to_idx])

    save_kwargs = {"task_ids": sliced_ids}
    for layer in LAYERS:
        key = f"layer_{layer}"
        if key in main_data:
            save_kwargs[key] = main_data[key][indices]

    np.savez(out_file, **save_kwargs)
    print(f"  Saved baseline: {len(sliced_ids)} tasks to {out_file}")


def extract_condition_activations(
    model,
    tasks: list,
    system_prompt: str,
    output_path: Path,
) -> None:
    """Extract activations for a single condition (system prompt) and save."""
    from src.probes.extraction.simple import extract_activations

    out_file = output_path / "activations_prompt_last.npz"
    if out_file.exists():
        print(f"  Already exists: {out_file}")
        return

    output_path.mkdir(parents=True, exist_ok=True)
    print(f"  Extracting {len(tasks)} tasks -> {output_path.name}")
    extract_activations(
        model=model,
        tasks=tasks,
        layers=LAYERS,
        selectors=[SELECTOR],
        batch_size=BATCH_SIZE,
        save_path=output_path,
        system_prompt=system_prompt,
    )


def load_tasks_for_ids(task_ids: list[str], prompts_lookup: dict[str, str]) -> list:
    """Build Task objects from task_ids using a prompt lookup dict."""
    from src.task_data.task import Task, OriginDataset

    tasks = []
    for tid in task_ids:
        if tid not in prompts_lookup:
            print(f"  WARNING: No prompt for {tid}")
            continue
        tasks.append(Task(
            id=tid,
            prompt=prompts_lookup[tid],
            origin=OriginDataset.SYNTHETIC,
            metadata={},
        ))
    return tasks


def load_standard_task_prompts(task_ids: list[str]) -> dict[str, str]:
    """Load prompts for standard dataset task IDs from the full dataset."""
    from src.task_data import load_tasks, OriginDataset

    all_tasks = load_tasks(
        n=100000,
        origins=[
            OriginDataset.WILDCHAT, OriginDataset.ALPACA, OriginDataset.MATH,
            OriginDataset.BAILBENCH, OriginDataset.STRESS_TEST,
        ],
    )
    lookup = {t.id: t.prompt for t in all_tasks}
    missing = [tid for tid in task_ids if tid not in lookup]
    if missing:
        print(f"  WARNING: {len(missing)} standard tasks not found in dataset: {missing[:5]}")
    return lookup


def exp1a_category(model=None) -> None:
    """Exp 1a: Category preference — 38 conditions × 30 tasks."""
    print("\n=== Exp 1a: Category preference ===")

    # Load config
    prompts_cfg = json.load(open(REPO_ROOT / "configs/ood/prompts/category_preference.json"))
    tasks_cfg = json.load(open(REPO_ROOT / "configs/ood/tasks/category_tasks.json"))

    # Flatten all category task IDs
    all_task_ids = []
    for category, tids in tasks_cfg.items():
        all_task_ids.extend(tids)
    all_task_ids = sorted(set(all_task_ids))
    print(f"Total category tasks: {len(all_task_ids)}")

    # Baseline: slice from main activations
    main_data = np.load(MAIN_ACTIVATIONS, allow_pickle=True)
    out_dir = ACTIVATIONS_DIR / "exp1_category"
    slice_and_save_baseline(all_task_ids, main_data, out_dir / "baseline")

    if model is None:
        print("No model loaded, skipping condition extraction")
        return

    # Load task prompts for standard tasks
    prompts_lookup = load_standard_task_prompts(all_task_ids)

    from src.task_data.task import Task, OriginDataset
    tasks_list = [
        Task(id=tid, prompt=prompts_lookup[tid], origin=OriginDataset.MATH if 'math' in tid or 'competition' in tid else OriginDataset.WILDCHAT, metadata={})
        for tid in all_task_ids if tid in prompts_lookup
    ]

    # Extract per condition
    conditions = prompts_cfg["conditions"]
    print(f"Conditions: {len(conditions)}")
    for cond in conditions:
        cid = cond["condition_id"]
        sysprompt = cond["system_prompt"]
        cond_dir = out_dir / cid
        extract_condition_activations(model, tasks_list, sysprompt, cond_dir)


def exp1b_1c_1d(model=None) -> None:
    """Exp 1b/1c/1d: Hidden, Crossed, Competing — shared extraction."""
    print("\n=== Exp 1b/1c/1d: Hidden/Crossed/Competing ===")

    # Load configs
    targeted_cfg = json.load(open(REPO_ROOT / "configs/ood/prompts/targeted_preference.json"))
    competing_cfg = json.load(open(REPO_ROOT / "configs/ood/prompts/competing_preference.json"))
    target_tasks = json.load(open(REPO_ROOT / "configs/ood/tasks/target_tasks.json"))
    crossed_tasks = json.load(open(REPO_ROOT / "configs/ood/tasks/crossed_tasks.json"))
    anchor_tasks_cfg = json.load(open(REPO_ROOT / "configs/ood/tasks/anchor_tasks.json"))

    # Custom tasks (need baseline extraction)
    custom_task_ids = [t["task_id"] for t in target_tasks + crossed_tasks]
    custom_prompts = {t["task_id"]: t["prompt"] for t in target_tasks + crossed_tasks}
    print(f"Custom tasks (need baseline): {len(custom_task_ids)}")

    # Anchor tasks (standard tasks in main activations)
    anchor_task_ids = anchor_tasks_cfg["anchor_ids"]  # list of task IDs
    print(f"Anchor tasks (from main): {len(anchor_task_ids)}")

    # All tasks for this experiment set
    all_task_ids = custom_task_ids + anchor_task_ids
    print(f"Total tasks: {len(all_task_ids)}")

    out_dir = ACTIVATIONS_DIR / "exp1_prompts"

    # 1) Baseline for custom tasks — extract with no system prompt
    if model is not None:
        baseline_file = out_dir / "baseline" / "activations_prompt_last.npz"
        if not baseline_file.exists():
            print("Extracting baseline for custom tasks (no system prompt)...")
            from src.task_data.task import Task, OriginDataset
            custom_tasks_list = [
                Task(id=tid, prompt=custom_prompts[tid], origin=OriginDataset.SYNTHETIC, metadata={})
                for tid in custom_task_ids
            ]
            extract_condition_activations(model, custom_tasks_list, None, out_dir / "baseline")
        else:
            print("  Custom task baseline already exists")

        # Add anchor task activations to baseline
        _merge_anchor_baseline(anchor_task_ids, out_dir / "baseline")
    else:
        print("No model loaded, skipping baseline extraction for custom tasks")

    # All conditions
    all_conditions = targeted_cfg["conditions"] + competing_cfg["conditions"]
    print(f"Total conditions: {len(all_conditions)}")

    if model is None:
        print("No model loaded, skipping condition extraction")
        return

    # Build full task list for extraction
    from src.task_data.task import Task, OriginDataset
    standard_prompts = load_standard_task_prompts(anchor_task_ids)
    prompts_lookup = {**custom_prompts, **standard_prompts}

    tasks_list = [
        Task(id=tid, prompt=prompts_lookup[tid], origin=OriginDataset.SYNTHETIC, metadata={})
        for tid in all_task_ids if tid in prompts_lookup
    ]
    print(f"Total tasks to extract per condition: {len(tasks_list)}")

    # Extract per condition
    for cond in all_conditions:
        cid = cond["condition_id"]
        sysprompt = cond["system_prompt"]
        cond_dir = out_dir / cid
        extract_condition_activations(model, tasks_list, sysprompt, cond_dir)


def _merge_anchor_baseline(anchor_task_ids: list[str], baseline_dir: Path) -> None:
    """Merge anchor tasks (from main activations) into the custom-task baseline npz."""
    out_file = baseline_dir / "activations_prompt_last.npz"
    if not out_file.exists():
        print("  Baseline file doesn't exist yet, can't merge anchors")
        return

    # Check if anchors already merged
    existing = np.load(out_file, allow_pickle=True)
    existing_ids = set(str(t) for t in existing["task_ids"])
    new_anchors = [t for t in anchor_task_ids if t not in existing_ids]
    if not new_anchors:
        print(f"  Anchors already in baseline ({len(existing_ids)} tasks total)")
        return

    print(f"  Merging {len(new_anchors)} anchor tasks into baseline...")
    main_data = np.load(MAIN_ACTIVATIONS, allow_pickle=True)
    main_ids = [str(t) for t in main_data["task_ids"]]
    id_to_idx = {tid: i for i, tid in enumerate(main_ids)}

    missing = [t for t in new_anchors if t not in id_to_idx]
    if missing:
        print(f"  WARNING: {len(missing)} anchors missing from main: {missing[:5]}")

    valid_anchors = [t for t in new_anchors if t in id_to_idx]
    anchor_indices = [id_to_idx[t] for t in valid_anchors]
    anchor_ids = np.array(valid_anchors)

    # Merge
    combined_ids = np.concatenate([existing["task_ids"], anchor_ids])
    save_kwargs = {"task_ids": combined_ids}
    for layer in LAYERS:
        key = f"layer_{layer}"
        if key in existing:
            anchor_acts = main_data[key][anchor_indices]
            save_kwargs[key] = np.concatenate([existing[key], anchor_acts], axis=0)

    np.savez(out_file, **save_kwargs)
    print(f"  Baseline now has {len(combined_ids)} tasks")


def exp2_roles(model=None) -> None:
    """Exp 2: Role-induced preferences — 20 conditions × 50 tasks."""
    print("\n=== Exp 2: Role-induced preferences ===")

    rp_cfg = json.load(open(REPO_ROOT / "configs/ood/prompts/role_playing.json"))
    np_cfg = json.load(open(REPO_ROOT / "configs/ood/prompts/narrow_preference.json"))
    behavioral = json.load(open(REPO_ROOT / "results/ood/role_playing/behavioral.json"))

    task_ids = list(behavioral["conditions"]["baseline"]["task_rates"].keys())
    print(f"Tasks: {len(task_ids)}")

    out_dir = ACTIVATIONS_DIR / "exp2_roles"
    main_data = np.load(MAIN_ACTIVATIONS, allow_pickle=True)
    slice_and_save_baseline(task_ids, main_data, out_dir / "baseline")

    if model is None:
        print("No model loaded, skipping condition extraction")
        return

    standard_prompts = load_standard_task_prompts(task_ids)
    from src.task_data.task import Task, OriginDataset
    tasks_list = [
        Task(id=tid, prompt=standard_prompts[tid], origin=OriginDataset.WILDCHAT, metadata={})
        for tid in task_ids if tid in standard_prompts
    ]

    all_conditions = rp_cfg["conditions"] + np_cfg["conditions"]
    print(f"Conditions: {len(all_conditions)}")
    for cond in all_conditions:
        cid = cond["condition_id"]
        sysprompt = cond["system_prompt"]
        extract_condition_activations(model, tasks_list, sysprompt, out_dir / cid)


def exp3_minimal_pairs(model=None) -> None:
    """Exp 3: Minimal pairs — 40 conditions × 50 tasks."""
    print("\n=== Exp 3: Minimal pairs ===")

    mp_cfg = json.load(open(REPO_ROOT / "configs/ood/prompts/minimal_pairs_v7.json"))
    behavioral = json.load(open(REPO_ROOT / "results/ood/minimal_pairs_v7/behavioral.json"))

    task_ids = list(behavioral["conditions"]["baseline"]["task_rates"].keys())
    print(f"Tasks: {len(task_ids)}")

    out_dir = ACTIVATIONS_DIR / "exp3_minimal_pairs"
    main_data = np.load(MAIN_ACTIVATIONS, allow_pickle=True)
    slice_and_save_baseline(task_ids, main_data, out_dir / "baseline")

    if model is None:
        print("No model loaded, skipping condition extraction")
        return

    standard_prompts = load_standard_task_prompts(task_ids)
    from src.task_data.task import Task, OriginDataset
    tasks_list = [
        Task(id=tid, prompt=standard_prompts[tid], origin=OriginDataset.WILDCHAT, metadata={})
        for tid in task_ids if tid in standard_prompts
    ]

    # Subsample: 2 base roles × 10 targets × versions A+B
    selected_roles = {"midwest", "brooklyn"}
    selected_versions = {"A", "B"}
    all_conditions = [
        c for c in mp_cfg["conditions"]
        if c["base_role"] in selected_roles and c["version"] in selected_versions
    ]
    print(f"Conditions (subsampled A+B): {len(all_conditions)}")

    for cond in all_conditions:
        cid = cond["condition_id"]
        sysprompt = cond["system_prompt"]
        extract_condition_activations(model, tasks_list, sysprompt, out_dir / cid)


def exp3c_minimal_pairs_anti(model=None) -> None:
    """Exp 3 extension: Anti (version C) conditions — 20 conditions × 50 tasks."""
    print("\n=== Exp 3C: Minimal pairs (anti conditions) ===")

    mp_cfg = json.load(open(REPO_ROOT / "configs/ood/prompts/minimal_pairs_v7.json"))
    behavioral = json.load(open(REPO_ROOT / "results/ood/minimal_pairs_v7/behavioral.json"))

    task_ids = list(behavioral["conditions"]["baseline"]["task_rates"].keys())
    print(f"Tasks: {len(task_ids)}")

    out_dir = ACTIVATIONS_DIR / "exp3_minimal_pairs"
    # Baseline already exists from exp3

    if model is None:
        print("No model loaded, skipping condition extraction")
        return

    standard_prompts = load_standard_task_prompts(task_ids)
    from src.task_data.task import Task, OriginDataset
    tasks_list = [
        Task(id=tid, prompt=standard_prompts[tid], origin=OriginDataset.WILDCHAT, metadata={})
        for tid in task_ids if tid in standard_prompts
    ]

    selected_roles = {"midwest", "brooklyn"}
    all_conditions = [
        c for c in mp_cfg["conditions"]
        if c["base_role"] in selected_roles and c["version"] == "C"
    ]
    print(f"Conditions (C only): {len(all_conditions)}")

    for cond in all_conditions:
        cid = cond["condition_id"]
        sysprompt = cond["system_prompt"]
        extract_condition_activations(model, tasks_list, sysprompt, out_dir / cid)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", default="all",
                        choices=["all", "exp1a", "exp1b_1c_1d", "exp2", "exp3", "exp3c", "exp3_all"])
    parser.add_argument("--no-model", action="store_true", help="Slice baselines only, no GPU needed")
    args = parser.parse_args()

    model = None
    if not args.no_model:
        from src.models.huggingface_model import HuggingFaceModel
        print("Loading model gemma-3-27b...")
        model = HuggingFaceModel("gemma-3-27b")
        print(f"Model loaded. Layers: {model.n_layers}, Hidden: {model.hidden_dim}")

    if args.exp in ("all", "exp1a"):
        exp1a_category(model)

    if args.exp in ("all", "exp1b_1c_1d"):
        exp1b_1c_1d(model)

    if args.exp in ("all", "exp2"):
        exp2_roles(model)

    if args.exp in ("all", "exp3", "exp3_all"):
        exp3_minimal_pairs(model)

    if args.exp in ("exp3c", "exp3_all"):
        exp3c_minimal_pairs_anti(model)

    print("\nDone!")


if __name__ == "__main__":
    main()
