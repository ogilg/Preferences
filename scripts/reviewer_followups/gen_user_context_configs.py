"""Generate the user-context persona steering configs.

Copies the L23 settings recovered from the comparator checkpoints and replaces
`system_prompt` with the two-turn `context_messages` construction. The persona
text is loaded from the authoritative config rather than retyped, so it stays
byte-identical to the published run.
"""

import sys
from pathlib import Path

import yaml

PERSONA_SRC = Path("configs/steering/cross_persona_differential/sadist.yaml")
OUT_DIR = Path("configs/steering/reviewer_followup_user_context")
CKPT_DIR = "experiments/reviewer_followups/user_context_persona/checkpoints"

# Settings taken from configs/steering/layer_sweep/harm_breakdown/contrastive_L23_150.yaml,
# the surviving L23 config on the same 150-pair set (the later finegrain config, which only
# widened the multiplier grid, was never committed). Cross-checked against the finegrain
# comparator checkpoints: layer 23, norm_at_layer 29381.541015625, sample_idx {0,1,2},
# ordering {0,1}, 150 pairs.
L23_NORM = {23: 29381.541015625}
PROBE_MANIFEST = "results/probes/layer_sweep/eot/"
MULTIPLIERS = [-0.06, -0.02, 0.0, 0.02, 0.06]
ACK = "Understood."


def base(probe_manifest: str, checkpoint: str, conditions: list[dict], persona: str) -> dict:
    return {
        "model": "gemma-3-27b",
        "max_new_tokens": 64,
        "pairs_path": "experiments/layer_sweep/harm_breakdown/steering_pairs_150.json",
        "probe_manifest": probe_manifest,
        "checkpoint_path": f"{CKPT_DIR}/{checkpoint}",
        "mean_norm": L23_NORM,
        "n_trials": 3,
        "temperature": 1.0,
        "seed": 42,
        "n_pairs": None,
        "template_path": "src/measurement/elicitation/prompt_templates/data/completion_preference.yaml",
        "context_messages": [
            {"role": "user", "content": persona},
            {"role": "assistant", "content": ACK},
        ],
        "conditions": conditions,
    }


def main(probe_manifest: str = PROBE_MANIFEST) -> None:
    persona = yaml.safe_load(PERSONA_SRC.read_text())["system_prompt"]
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    contrastive = base(
        probe_manifest,
        "sadist_user_context_contrastive.jsonl",
        [{
            "name": "contrastive_L23",
            "cache_injection": "differential",
            "probe": "ridge_L23",
            "layers": [23],
            "multipliers": MULTIPLIERS,
            "spans": {"first": 1, "second": -1},
        }],
        persona,
    )

    single_task = base(
        probe_manifest,
        "sadist_user_context_single_task.jsonl",
        [
            {
                "name": "unilateral_first",
                "cache_injection": "differential",
                "probe": "ridge_L23",
                "layers": [23],
                "multipliers": MULTIPLIERS,
                "spans": {"first": 1},
            },
            {
                "name": "unilateral_second",
                "cache_injection": "differential",
                "probe": "ridge_L23",
                "layers": [23],
                "multipliers": MULTIPLIERS,
                "spans": {"second": 1},
            },
        ],
        persona,
    )

    for fname, cfg in [
        ("sadist_user_context_contrastive.yaml", contrastive),
        ("sadist_user_context_single_task.yaml", single_task),
    ]:
        path = OUT_DIR / fname
        path.write_text(yaml.safe_dump(cfg, sort_keys=False, width=100, allow_unicode=True))
        print(f"wrote {path}")

    # Round-trip: persona survives yaml dump/load byte-identically.
    for fname in ["sadist_user_context_contrastive.yaml", "sadist_user_context_single_task.yaml"]:
        loaded = yaml.safe_load((OUT_DIR / fname).read_text())
        assert loaded["context_messages"][0]["content"] == persona, f"persona corrupted in {fname}"
        assert "system_prompt" not in loaded, f"system_prompt still present in {fname}"
    print("persona round-trip byte-exact; no system_prompt present")


if __name__ == "__main__":
    main(*sys.argv[1:])
