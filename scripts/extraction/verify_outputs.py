"""Verify extraction outputs: 30 .npz files per model, correct shapes."""
import numpy as np
from pathlib import Path

ACTIVATIONS_DIR = Path("activations/character_probes")
EXPECTED_SELECTORS = [
    "turn_boundary:-1", "turn_boundary:-2", "turn_boundary:-3",
    "turn_boundary:-4", "turn_boundary:-5", "task_mean"
]
EXPECTED_LAYERS = [8, 12, 16, 20, 24]
EXPECTED_TASKS = 2500
EXPECTED_DIM = 4096

models = sorted(ACTIVATIONS_DIR.iterdir()) if ACTIVATIONS_DIR.exists() else []
print(f"Found {len(models)} model directories in {ACTIVATIONS_DIR}\n")

all_ok = True
for model_dir in models:
    if not model_dir.is_dir():
        continue
    npz_files = sorted(model_dir.glob("*.npz"))
    n_npz = len(npz_files)
    issues = []

    for sel in EXPECTED_SELECTORS:
        fname = f"activations_{sel}.npz"
        fpath = model_dir / fname
        if not fpath.exists():
            issues.append(f"Missing {fname}")
            continue
        data = np.load(fpath)
        task_ids = data["task_ids"]
        if len(task_ids) != EXPECTED_TASKS:
            issues.append(f"{fname}: {len(task_ids)} tasks (expected {EXPECTED_TASKS})")
        for layer in EXPECTED_LAYERS:
            key = f"layer_{layer}"
            if key not in data:
                issues.append(f"{fname}: missing {key}")
            else:
                shape = data[key].shape
                if shape != (EXPECTED_TASKS, EXPECTED_DIM):
                    issues.append(f"{fname}/{key}: shape {shape} (expected ({EXPECTED_TASKS}, {EXPECTED_DIM}))")

    status = "OK" if not issues else "ISSUES"
    if issues:
        all_ok = False
    print(f"{model_dir.name}: {n_npz} files — {status}")
    for issue in issues:
        print(f"  ! {issue}")

print(f"\n{'All models verified OK' if all_ok else 'Some models have issues'}")
