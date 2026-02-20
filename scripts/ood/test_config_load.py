from pathlib import Path
from src.ood.config import load_ood_config
from src.ood.measurement import _make_template

for p in sorted(Path("configs/ood").glob("*.yaml")):
    c = load_ood_config(p)
    t = _make_template(c)
    print(f"{p.name}: template={t.name}, custom_tasks='{c.custom_tasks}'")
