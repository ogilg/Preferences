"""Step 2: Quantitative mu-conditional analyses on existing generation data.

Analyzes: response length, word-level features, all conditioned on mu group and coefficient.
"""
import json
import re
from pathlib import Path

import numpy as np
from scipy import stats

data_path = Path("experiments/steering/program/coefficient_calibration/generation_results.json")
with open(data_path) as f:
    data = json.load(f)

# Build prompt metadata lookup
prompt_meta = {}
for p in data["prompts"]:
    prompt_meta[p["prompt_id"]] = p

# Focus on coherent range
COHERENT_COEFS = [-5000, -3000, -2000, -1000, -500, 0, 500, 1000, 2000, 3000, 5000]

# Define mu groups for C_completion and B_rating
MU_GROUPS = {
    "LOW": ["C_00", "C_01", "C_02"],   # mu < -2
    "MID": ["C_03", "C_04"],           # mu 0-2
    "HIGH": ["C_05", "C_06", "C_07", "C_08"],  # mu > 4
}

HEDGING_PHRASES = ["however", "but ", "although", "that said", "to be fair", "on the other hand",
                   "nonetheless", "nevertheless", "it's worth noting", "it should be noted",
                   "i should mention", "i must", "i cannot", "i can't", "i'm unable"]

def extract_word_features(text: str) -> dict:
    text_lower = text.lower()
    return {
        "length": len(text),
        "exclamation_marks": text.count("!"),
        "question_marks": text.count("?"),
        "bullet_points": text.count("* ") + text.count("- ") + len(re.findall(r'^\d+\.', text, re.MULTILINE)),
        "headers": len(re.findall(r'^#{1,3}\s', text, re.MULTILINE)) + len(re.findall(r'^\*\*[^*]+\*\*$', text, re.MULTILINE)),
        "hedging_count": sum(1 for phrase in HEDGING_PHRASES if phrase in text_lower),
        "bold_count": len(re.findall(r'\*\*[^*]+\*\*', text)),
        "word_count": len(text.split()),
    }

# ============================
# Analysis 1: Response length x mu x coefficient
# ============================
print("=" * 60)
print("ANALYSIS 1: Response Length × Mu × Coefficient")
print("=" * 60)

for group_name, prompt_ids in MU_GROUPS.items():
    mu_values = [prompt_meta[pid]["metadata"]["mu"] for pid in prompt_ids]
    print(f"\n--- {group_name} (mu: {[f'{m:.1f}' for m in mu_values]}) ---")

    # Collect (coefficient, length) pairs for this group
    coef_length_pairs = []
    for r in data["results"]:
        if r["prompt_id"] in prompt_ids and r["coefficient"] in COHERENT_COEFS:
            coef_length_pairs.append((r["coefficient"], r["response_length"]))

    coefs = [p[0] for p in coef_length_pairs]
    lengths = [p[1] for p in coef_length_pairs]

    rho, p = stats.spearmanr(coefs, lengths)
    print(f"  Spearman rho(coef, length): {rho:.4f}, p={p:.4f}, n={len(coefs)}")

    # Mean length per coefficient
    for coef in [-3000, -1000, 0, 1000, 3000]:
        subset = [l for c, l in coef_length_pairs if c == coef]
        if subset:
            print(f"  coef={coef:>6}: mean_length={np.mean(subset):.0f} (n={len(subset)})")

# Test interaction: is the rho different for LOW vs HIGH?
print("\n--- Interaction test ---")
low_pairs = [(r["coefficient"], r["response_length"]) for r in data["results"]
             if r["prompt_id"] in MU_GROUPS["LOW"] and r["coefficient"] in COHERENT_COEFS]
high_pairs = [(r["coefficient"], r["response_length"]) for r in data["results"]
              if r["prompt_id"] in MU_GROUPS["HIGH"] and r["coefficient"] in COHERENT_COEFS]

rho_low, _ = stats.spearmanr([p[0] for p in low_pairs], [p[1] for p in low_pairs])
rho_high, _ = stats.spearmanr([p[0] for p in high_pairs], [p[1] for p in high_pairs])
print(f"  rho_LOW  = {rho_low:.4f}")
print(f"  rho_HIGH = {rho_high:.4f}")
print(f"  Difference: {rho_high - rho_low:.4f}")

# Fisher z-test for difference in correlations
def fisher_z_test(r1, n1, r2, n2):
    z1 = np.arctanh(r1)
    z2 = np.arctanh(r2)
    se = np.sqrt(1/(n1-3) + 1/(n2-3))
    z = (z1 - z2) / se
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return z, p

z, p = fisher_z_test(rho_low, len(low_pairs), rho_high, len(high_pairs))
print(f"  Fisher z-test: z={z:.3f}, p={p:.4f}")

# ============================
# Analysis 2: Word-level features x mu x coefficient
# ============================
print("\n" + "=" * 60)
print("ANALYSIS 2: Word Features × Mu × Coefficient")
print("=" * 60)

feature_names = ["length", "exclamation_marks", "question_marks", "bullet_points",
                 "headers", "hedging_count", "bold_count", "word_count"]

for group_name, prompt_ids in MU_GROUPS.items():
    print(f"\n--- {group_name} ---")

    features_by_coef = {}
    for r in data["results"]:
        if r["prompt_id"] in prompt_ids and r["coefficient"] in COHERENT_COEFS:
            feats = extract_word_features(r["response"])
            coef = r["coefficient"]
            if coef not in features_by_coef:
                features_by_coef[coef] = {fn: [] for fn in feature_names}
            for fn in feature_names:
                features_by_coef[coef][fn].append(feats[fn])

    # Spearman correlation of each feature with coefficient
    for fn in feature_names:
        all_coefs = []
        all_vals = []
        for coef in COHERENT_COEFS:
            if coef in features_by_coef:
                for val in features_by_coef[coef][fn]:
                    all_coefs.append(coef)
                    all_vals.append(val)

        rho, p = stats.spearmanr(all_coefs, all_vals)
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"  {fn:>20}: rho={rho:>7.4f}, p={p:.4f} {sig}")

# ============================
# Analysis 3: Interaction test for each feature
# ============================
print("\n" + "=" * 60)
print("ANALYSIS 3: Feature × Mu Interaction (LOW vs HIGH)")
print("=" * 60)

for fn in feature_names:
    low_coefs, low_vals = [], []
    high_coefs, high_vals = [], []

    for r in data["results"]:
        if r["coefficient"] not in COHERENT_COEFS:
            continue
        feats = extract_word_features(r["response"])
        if r["prompt_id"] in MU_GROUPS["LOW"]:
            low_coefs.append(r["coefficient"])
            low_vals.append(feats[fn])
        elif r["prompt_id"] in MU_GROUPS["HIGH"]:
            high_coefs.append(r["coefficient"])
            high_vals.append(feats[fn])

    rho_low, p_low = stats.spearmanr(low_coefs, low_vals)
    rho_high, p_high = stats.spearmanr(high_coefs, high_vals)

    # Clip to avoid arctanh(1) = inf
    rho_low_c = np.clip(rho_low, -0.999, 0.999)
    rho_high_c = np.clip(rho_high, -0.999, 0.999)
    z, p_diff = fisher_z_test(rho_low_c, len(low_coefs), rho_high_c, len(high_coefs))

    sig = " ***" if p_diff < 0.001 else " **" if p_diff < 0.01 else " *" if p_diff < 0.05 else ""
    print(f"  {fn:>20}: rho_LOW={rho_low:>7.4f} (p={p_low:.3f}), rho_HIGH={rho_high:>7.4f} (p={p_high:.3f}), diff_p={p_diff:.4f}{sig}")

# ============================
# Also do B_rating: does the binary good/bad rating shift differently by mu?
# ============================
print("\n" + "=" * 60)
print("ANALYSIS 4: B_rating good/bad by mu group")
print("=" * 60)

B_MU_GROUPS = {
    "LOW": ["B_00", "B_01", "B_02"],
    "MID": ["B_03", "B_04"],
    "HIGH": ["B_05", "B_06", "B_07", "B_08"],
}

for group_name, prompt_ids in B_MU_GROUPS.items():
    print(f"\n--- {group_name} ---")
    for coef in [-3000, -1000, 0, 1000, 3000]:
        responses = [r["response"].strip().lower() for r in data["results"]
                     if r["prompt_id"] in prompt_ids and r["coefficient"] == coef]
        good = sum(1 for r in responses if "good" in r[:20])
        bad = sum(1 for r in responses if "bad" in r[:20])
        total = len(responses)
        pct_good = good / total * 100 if total > 0 else 0
        print(f"  coef={coef:>6}: good={good}, bad={bad}, other={total-good-bad}, %good={pct_good:.0f}% (n={total})")
