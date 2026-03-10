"""Run all 11 remaining probe training configs for the turn boundary sweep."""

import subprocess
import sys

configs = [
    # Heldout eval (tb-1 already done)
    "configs/probes/heldout_eval_gemma3_tb-2.yaml",
    "configs/probes/heldout_eval_gemma3_tb-3.yaml",
    "configs/probes/heldout_eval_gemma3_tb-4.yaml",
    "configs/probes/heldout_eval_gemma3_tb-5.yaml",
    "configs/probes/heldout_eval_gemma3_task_mean.yaml",
    # HOO by topic
    "configs/probes/gemma3_10k_hoo_topic_tb-1.yaml",
    "configs/probes/gemma3_10k_hoo_topic_tb-2.yaml",
    "configs/probes/gemma3_10k_hoo_topic_tb-3.yaml",
    "configs/probes/gemma3_10k_hoo_topic_tb-4.yaml",
    "configs/probes/gemma3_10k_hoo_topic_tb-5.yaml",
    "configs/probes/gemma3_10k_hoo_topic_task_mean.yaml",
]

for i, config in enumerate(configs):
    print(f"\n{'='*60}")
    print(f"[{i+1}/{len(configs)}] {config}")
    print(f"{'='*60}")
    result = subprocess.run(
        [sys.executable, "-m", "src.probes.experiments.run_dir_probes", "--config", config],
        capture_output=False,
    )
    if result.returncode != 0:
        print(f"FAILED: {config}")
        sys.exit(1)

print("\nAll probe configs completed successfully!")
