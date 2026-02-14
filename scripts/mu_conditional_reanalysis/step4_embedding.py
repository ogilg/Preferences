"""Step 4: Embedding distance analysis.

Embed all completions with MiniLM, compute cosine distance from unsteered mean.
Test: does distance increase with |coefficient|? Condition on mu.
"""
import json
from pathlib import Path

import numpy as np
from scipy import stats
from sentence_transformers import SentenceTransformer

data_path = Path("experiments/steering/program/coefficient_calibration/generation_results.json")
with open(data_path) as f:
    data = json.load(f)

prompt_meta = {}
for p in data["prompts"]:
    prompt_meta[p["prompt_id"]] = p

COHERENT_COEFS = [-5000, -3000, -2000, -1000, -500, 0, 500, 1000, 2000, 3000, 5000]

# Load sentence transformer
print("Loading sentence transformer...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# Group results by prompt_id
results_by_prompt = {}
for r in data["results"]:
    pid = r["prompt_id"]
    if pid not in results_by_prompt:
        results_by_prompt[pid] = []
    results_by_prompt[pid].append(r)

print("Computing embeddings for all prompts...")

# For each prompt, embed all responses and compute distance from unsteered mean
all_results = []
for pid in sorted(results_by_prompt.keys()):
    results = results_by_prompt[pid]

    # Get all responses in coherent range
    coherent = [r for r in results if r["coefficient"] in COHERENT_COEFS]
    if not coherent:
        continue

    # Embed all coherent responses (truncate to first 512 chars for efficiency)
    texts = [r["response"][:2000] for r in coherent]
    embeddings = model.encode(texts, show_progress_bar=False)

    # Get unsteered (coef=0) embeddings
    zero_indices = [i for i, r in enumerate(coherent) if r["coefficient"] == 0]
    if not zero_indices:
        continue
    zero_mean = np.mean(embeddings[zero_indices], axis=0)
    zero_mean_norm = zero_mean / np.linalg.norm(zero_mean)

    # Compute cosine distance from unsteered mean for each response
    for i, r in enumerate(coherent):
        emb = embeddings[i]
        emb_norm = emb / np.linalg.norm(emb)
        cosine_sim = np.dot(emb_norm, zero_mean_norm)
        cosine_dist = 1 - cosine_sim

        meta = prompt_meta[pid]["metadata"]
        mu = meta.get("mu", None)
        category = prompt_meta[pid]["category"]

        all_results.append({
            "prompt_id": pid,
            "category": category,
            "mu": mu,
            "coefficient": r["coefficient"],
            "seed": r["seed"],
            "cosine_dist": float(cosine_dist),
            "response_length": r["response_length"],
        })

print(f"Computed {len(all_results)} embedding distances.")

# ============================
# Analysis: Overall distance vs |coefficient|
# ============================
print("\n" + "=" * 60)
print("Embedding distance vs |coefficient| (all prompts)")
print("=" * 60)

abs_coefs = [abs(r["coefficient"]) for r in all_results]
dists = [r["cosine_dist"] for r in all_results]
rho, p = stats.spearmanr(abs_coefs, dists)
print(f"Spearman rho(|coef|, distance): {rho:.4f}, p={p:.6f}")

# Per |coefficient| mean distance
for ac in sorted(set(abs_coefs)):
    subset = [r["cosine_dist"] for r in all_results if abs(r["coefficient"]) == ac]
    print(f"  |coef|={ac:>5}: mean_dist={np.mean(subset):.6f} (n={len(subset)})")

# ============================
# Per-category analysis
# ============================
print("\n" + "=" * 60)
print("Embedding distance vs |coefficient| by category")
print("=" * 60)

categories = sorted(set(r["category"] for r in all_results))
for cat in categories:
    cat_results = [r for r in all_results if r["category"] == cat]
    abs_c = [abs(r["coefficient"]) for r in cat_results]
    d = [r["cosine_dist"] for r in cat_results]
    rho, p = stats.spearmanr(abs_c, d)
    sig = " ***" if p < 0.001 else " **" if p < 0.01 else " *" if p < 0.05 else ""
    print(f"  {cat:>15}: rho={rho:.4f}, p={p:.6f}{sig}")

# ============================
# C_completion: distance by mu group
# ============================
print("\n" + "=" * 60)
print("C_completion: embedding distance by mu group")
print("=" * 60)

MU_GROUPS = {
    "LOW": ["C_00", "C_01", "C_02"],
    "MID": ["C_03", "C_04"],
    "HIGH": ["C_05", "C_06", "C_07", "C_08"],
}

for group, pids in MU_GROUPS.items():
    grp = [r for r in all_results if r["prompt_id"] in pids]
    abs_c = [abs(r["coefficient"]) for r in grp]
    d = [r["cosine_dist"] for r in grp]
    rho, p = stats.spearmanr(abs_c, d)

    # Also test signed coefficient (directional)
    signed_c = [r["coefficient"] for r in grp]
    rho_signed, p_signed = stats.spearmanr(signed_c, d)

    sig = " ***" if p < 0.001 else " **" if p < 0.01 else " *" if p < 0.05 else ""
    sig_s = " ***" if p_signed < 0.001 else " **" if p_signed < 0.01 else " *" if p_signed < 0.05 else ""
    print(f"  {group}: rho(|coef|, dist)={rho:.4f}, p={p:.6f}{sig}")
    print(f"       rho(coef, dist)={rho_signed:.4f}, p={p_signed:.6f}{sig_s}")

    # Mean distance at key coefficients
    for coef in [-3000, 0, 3000]:
        subset = [r["cosine_dist"] for r in grp if r["coefficient"] == coef]
        if subset:
            print(f"       coef={coef:>6}: mean_dist={np.mean(subset):.6f}")

# ============================
# Per-prompt analysis (C_completion)
# ============================
print("\n" + "=" * 60)
print("C_completion: per-prompt embedding distance vs |coefficient|")
print("=" * 60)

for pid in ["C_00", "C_01", "C_02", "C_03", "C_04", "C_05", "C_06", "C_07", "C_08"]:
    grp = [r for r in all_results if r["prompt_id"] == pid]
    abs_c = [abs(r["coefficient"]) for r in grp]
    d = [r["cosine_dist"] for r in grp]
    rho, p = stats.spearmanr(abs_c, d)
    mu = prompt_meta[pid]["metadata"]["mu"]
    sig = " ***" if p < 0.001 else " **" if p < 0.01 else " *" if p < 0.05 else ""
    print(f"  {pid} (mu={mu:>6.2f}): rho={rho:.4f}, p={p:.6f}{sig}")

# Save results for plotting
output_path = Path("scripts/mu_conditional_reanalysis/embedding_results.json")
with open(output_path, "w") as f:
    json.dump(all_results, f, indent=2)
print(f"\nResults saved to {output_path}")
