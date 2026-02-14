"""Step 3: Per-prompt dose-response analysis for D_valence and F_affect.

Check if aggregate null hides prompt-specific effects.
Also deeper dive into the question mark signal from Step 2.
"""
import json
from pathlib import Path

import numpy as np
from scipy import stats

data_path = Path("experiments/steering/program/coefficient_calibration/generation_results.json")
with open(data_path) as f:
    data = json.load(f)

prompt_meta = {}
for p in data["prompts"]:
    prompt_meta[p["prompt_id"]] = p

COHERENT_COEFS = [-5000, -3000, -2000, -1000, -500, 0, 500, 1000, 2000, 3000, 5000]

# ============================
# Per-prompt length dose-response
# ============================
print("=" * 60)
print("Per-prompt length dose-response (D_valence + F_affect)")
print("=" * 60)

for cat in ["D_valence", "F_affect"]:
    print(f"\n--- {cat} ---")
    cat_prompts = [p["prompt_id"] for p in data["prompts"] if p["category"] == cat]

    for pid in sorted(cat_prompts):
        coefs, lengths = [], []
        for r in data["results"]:
            if r["prompt_id"] == pid and r["coefficient"] in COHERENT_COEFS:
                coefs.append(r["coefficient"])
                lengths.append(r["response_length"])

        rho, p = stats.spearmanr(coefs, lengths)
        prompt_text = prompt_meta[pid]["metadata"].get("prompt_text", "N/A")[:50]
        sig = " ***" if p < 0.001 else " **" if p < 0.01 else " *" if p < 0.05 else ""
        print(f"  {pid}: rho={rho:>7.4f}, p={p:.4f}{sig}  \"{prompt_text}\"")

# ============================
# Per-prompt question mark dose-response (investigating the signal)
# ============================
print("\n" + "=" * 60)
print("Per-prompt question mark dose-response (C_completion)")
print("=" * 60)

for pid in ["C_00", "C_01", "C_02", "C_03", "C_04", "C_05", "C_06", "C_07", "C_08"]:
    coefs, qmarks = [], []
    for r in data["results"]:
        if r["prompt_id"] == pid and r["coefficient"] in COHERENT_COEFS:
            coefs.append(r["coefficient"])
            qmarks.append(r["response"].count("?"))

    rho, p = stats.spearmanr(coefs, qmarks)
    mu = prompt_meta[pid]["metadata"]["mu"]
    sig = " ***" if p < 0.001 else " **" if p < 0.01 else " *" if p < 0.05 else ""
    print(f"  {pid} (mu={mu:>6.2f}): rho={rho:>7.4f}, p={p:.4f}{sig}")

    # Show mean question marks at key coefficients
    for coef in [-3000, 0, 3000]:
        qm = [r["response"].count("?") for r in data["results"]
              if r["prompt_id"] == pid and r["coefficient"] == coef]
        print(f"    coef={coef:>6}: mean_qmarks={np.mean(qm):.1f} (n={len(qm)})")

# ============================
# Per-prompt exclamation mark dose-response (C_completion)
# ============================
print("\n" + "=" * 60)
print("Per-prompt exclamation mark dose-response (C_completion)")
print("=" * 60)

for pid in ["C_00", "C_01", "C_02", "C_03", "C_04", "C_05", "C_06", "C_07", "C_08"]:
    coefs, emarks = [], []
    for r in data["results"]:
        if r["prompt_id"] == pid and r["coefficient"] in COHERENT_COEFS:
            coefs.append(r["coefficient"])
            emarks.append(r["response"].count("!"))

    rho, p = stats.spearmanr(coefs, emarks)
    mu = prompt_meta[pid]["metadata"]["mu"]
    sig = " ***" if p < 0.001 else " **" if p < 0.01 else " *" if p < 0.05 else ""
    print(f"  {pid} (mu={mu:>6.2f}): rho={rho:>7.4f}, p={p:.4f}{sig}")

# ============================
# B_rating: Investigate the HIGH mu drop at coef=-3000
# ============================
print("\n" + "=" * 60)
print("B_rating: Per-task breakdown at coef=-3000 (HIGH mu)")
print("=" * 60)

for pid in ["B_05", "B_06", "B_07", "B_08"]:
    mu = prompt_meta[pid]["metadata"]["mu"]
    print(f"\n{pid} (mu={mu:.2f}):")
    for coef in [-3000, -1000, 0, 1000, 3000]:
        responses = [r["response"].strip().lower()[:50] for r in data["results"]
                     if r["prompt_id"] == pid and r["coefficient"] == coef]
        good = sum(1 for r in responses if "good" in r[:20])
        print(f"  coef={coef:>6}: {responses} -> good={good}/{len(responses)}")

# ============================
# Per-prompt response length for ALL categories
# ============================
print("\n" + "=" * 60)
print("Per-prompt length dose-response (ALL categories)")
print("=" * 60)

for cat in ["A_pairwise", "B_rating", "C_completion", "D_valence", "E_neutral", "F_affect"]:
    print(f"\n--- {cat} ---")
    cat_prompts = [p["prompt_id"] for p in data["prompts"] if p["category"] == cat]

    for pid in sorted(cat_prompts):
        coefs, lengths = [], []
        for r in data["results"]:
            if r["prompt_id"] == pid and r["coefficient"] in COHERENT_COEFS:
                coefs.append(r["coefficient"])
                lengths.append(r["response_length"])

        rho, p = stats.spearmanr(coefs, lengths)
        sig = " ***" if p < 0.001 else " **" if p < 0.01 else " *" if p < 0.05 else ""

        # Get mu if available
        meta = prompt_meta[pid]["metadata"]
        mu_str = f"mu={meta['mu']:.1f}" if "mu" in meta else ""
        print(f"  {pid} {mu_str:>10}: rho={rho:>7.4f}, p={p:.4f}{sig}")
