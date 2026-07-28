"""CPU validation of the user-context configs: config parsing + probe resolution."""

import json
from pathlib import Path

import numpy as np
import yaml
from dotenv import load_dotenv

from src.probes.core.storage import load_probe_direction
from src.steering.runner import load_config

load_dotenv()

CONFIG_DIR = Path("configs/steering/reviewer_followup_user_context")
PERSONA_SRC = Path("configs/steering/cross_persona_differential/sadist.yaml")
REFERENCE = Path("configs/steering/layer_sweep/harm_breakdown/contrastive_L23_150.yaml")

persona = yaml.safe_load(PERSONA_SRC.read_text())["system_prompt"]
ref = yaml.safe_load(REFERENCE.read_text())
failures = []

for path in sorted(CONFIG_DIR.glob("*.yaml")):
    cfg = load_config(path)
    print(f"=== {path.name} ===")
    print(f"  model={cfg.model} layer_norm={cfg.mean_norm} n_trials={cfg.n_trials} "
          f"temp={cfg.temperature} seed={cfg.seed} max_new_tokens={cfg.max_new_tokens}")
    print(f"  system_prompt={cfg.system_prompt!r}")
    print(f"  context roles={[m['role'] for m in cfg.context_messages]}")
    print(f"  conditions={[(c.name, c.probe, c.layers, c.spans) for c in cfg.conditions]}")

    if cfg.system_prompt is not None:
        failures.append(f"{path.name}: system_prompt should be absent, got {cfg.system_prompt!r}")
    if cfg.context_messages is None:
        failures.append(f"{path.name}: context_messages did not survive load_config")
    else:
        if cfg.context_messages[0]["content"] != persona:
            failures.append(f"{path.name}: persona text not byte-exact after load_config")
        if [m["role"] for m in cfg.context_messages] != ["user", "assistant"]:
            failures.append(f"{path.name}: context roles must be user,assistant")

    # Matched against the surviving L23 reference config.
    for field in ["model", "max_new_tokens", "pairs_path", "probe_manifest", "mean_norm",
                  "n_trials", "temperature", "seed", "template_path"]:
        got = yaml.safe_load(path.read_text())[field]
        if got != ref[field]:
            failures.append(f"{path.name}: {field}={got!r} != reference {ref[field]!r}")

    if not Path(cfg.pairs_path).exists():
        failures.append(f"{path.name}: pairs_path missing: {cfg.pairs_path}")
    else:
        n = len(json.loads(Path(cfg.pairs_path).read_text()))
        print(f"  pairs={n}")
        if n != 150:
            failures.append(f"{path.name}: expected 150 pairs, got {n}")

    for cond in cfg.conditions:
        layer, direction = load_probe_direction(cfg.probe_manifest, cond.probe)
        print(f"  probe {cond.probe}: layer={layer} dim={direction.shape[0]} "
              f"unit_norm={np.linalg.norm(direction):.6f}")
        if layer != 23:
            failures.append(f"{path.name}/{cond.name}: probe layer {layer} != 23")
        if cond.multipliers != [-0.06, -0.02, 0.0, 0.02, 0.06]:
            failures.append(f"{path.name}/{cond.name}: multipliers {cond.multipliers}")
    print()

if failures:
    print(f"FAILED ({len(failures)}):")
    for f in failures:
        print(f"  - {f}")
    raise SystemExit(1)
print("Configs parse, persona is byte-exact, settings match the L23 reference, probe resolves.")
