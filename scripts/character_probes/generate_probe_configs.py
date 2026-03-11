"""Generate probe training configs for all selectors on the base Llama 3.1 8B model."""

from pathlib import Path
import yaml

SELECTORS = [
    "task_mean",
    "turn_boundary:-1",
    "turn_boundary:-2",
    "turn_boundary:-3",
    "turn_boundary:-4",
    "turn_boundary:-5",
]
LAYERS = [8, 12, 16, 20, 24]

BASE_RUN_DIR = (
    "results/experiments/character_probes/pre_task_active_learning/"
    "completion_preference_llama-3.1-8b_completion_canonical_seed0_mra_exp2_split_a_1000_task_ids"
)
EVAL_RUN_DIR = (
    "results/experiments/character_probes/pre_task_active_learning/"
    "completion_preference_llama-3.1-8b_completion_canonical_seed0_mra_exp2_split_b_500_task_ids"
)

CONFIG_DIR = Path("configs/probes/character_probes")
CONFIG_DIR.mkdir(parents=True, exist_ok=True)

for selector in SELECTORS:
    safe_name = selector.replace(":", "_").replace("-", "m")
    config = {
        "experiment_name": f"llama8b_base_{safe_name}",
        "run_dir": BASE_RUN_DIR,
        "eval_run_dir": EVAL_RUN_DIR,
        "activations_path": f"activations/character_probes/llama_3_1_8b_base/activations_{selector}.npz",
        "output_dir": f"results/probes/character_probes/llama8b_base_{safe_name}",
        "layers": LAYERS,
        "modes": ["ridge"],
        "standardize": True,
        "alpha_sweep_size": 20,
        "eval_split_seed": 42,
    }
    path = CONFIG_DIR / f"llama8b_base_{safe_name}.yaml"
    with open(path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    print(f"Wrote {path}")
