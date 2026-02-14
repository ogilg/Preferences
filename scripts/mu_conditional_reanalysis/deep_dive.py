"""Deep dive into the length dose-response patterns and B_05 rating flip."""
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
# 1. Verify B_05 rating flip with full coefficient sweep
# ============================
print("=" * 60)
print("B_05 rating flip: full coefficient sweep")
print("=" * 60)

for coef in COHERENT_COEFS:
    responses = []
    for r in data["results"]:
        if r["prompt_id"] == "B_05" and r["coefficient"] == coef:
            responses.append(r["response"].strip().lower()[:30])
    good = sum(1 for r in responses if "good" in r)
    bad = sum(1 for r in responses if "bad" in r)
    print(f"  coef={coef:>6}: responses={responses}, good={good}, bad={bad}")

# ============================
# 2. Length direction: categorize prompts by their rho sign
# ============================
print("\n" + "=" * 60)
print("Length dose-response: all prompts with significant rho")
print("=" * 60)

sig_results = []
for pid in sorted(prompt_meta.keys()):
    cat = prompt_meta[pid]["category"]
    coefs, lengths = [], []
    for r in data["results"]:
        if r["prompt_id"] == pid and r["coefficient"] in COHERENT_COEFS:
            coefs.append(r["coefficient"])
            lengths.append(r["response_length"])

    if not coefs or len(set(lengths)) <= 1:
        continue

    rho, p = stats.spearmanr(coefs, lengths)
    mu = prompt_meta[pid]["metadata"].get("mu", None)
    prompt_text = prompt_meta[pid]["metadata"].get("prompt_text", "")

    if p < 0.05:
        sig_results.append({
            "pid": pid, "category": cat, "mu": mu,
            "rho": rho, "p": p, "prompt_text": prompt_text
        })

# Sort by rho
sig_results.sort(key=lambda x: x["rho"])

print(f"\n{len(sig_results)} prompts with significant length dose-response (p<0.05):")
print(f"\nNEGATIVE rho (positive steering → shorter responses):")
for r in sig_results:
    if r["rho"] < 0:
        sig = "***" if r["p"] < 0.001 else "**" if r["p"] < 0.01 else "*"
        mu_str = f"mu={r['mu']:.1f}" if r["mu"] is not None else ""
        pt_str = f'"{r["prompt_text"][:50]}"' if r["prompt_text"] else ""
        print(f"  {r['pid']:>6} [{r['category']:>12}] rho={r['rho']:>7.3f} {sig:>3} {mu_str:>10}  {pt_str}")

print(f"\nPOSITIVE rho (positive steering → longer responses):")
for r in sig_results:
    if r["rho"] > 0:
        sig = "***" if r["p"] < 0.001 else "**" if r["p"] < 0.01 else "*"
        mu_str = f"mu={r['mu']:.1f}" if r["mu"] is not None else ""
        pt_str = f'"{r["prompt_text"][:50]}"' if r["prompt_text"] else ""
        print(f"  {r['pid']:>6} [{r['category']:>12}] rho={r['rho']:>7.3f} {sig:>3} {mu_str:>10}  {pt_str}")

# ============================
# 3. Multiple testing correction (Bonferroni)
# ============================
print("\n" + "=" * 60)
print("Multiple testing correction")
print("=" * 60)

all_ps = []
for pid in sorted(prompt_meta.keys()):
    coefs, lengths = [], []
    for r in data["results"]:
        if r["prompt_id"] == pid and r["coefficient"] in COHERENT_COEFS:
            coefs.append(r["coefficient"])
            lengths.append(r["response_length"])
    if coefs and len(set(lengths)) > 1:
        rho, p = stats.spearmanr(coefs, lengths)
        all_ps.append((pid, rho, p))

n_tests = len(all_ps)
bonferroni_threshold = 0.05 / n_tests
print(f"Number of tests: {n_tests}")
print(f"Bonferroni threshold: {bonferroni_threshold:.6f}")
print(f"\nPrompts surviving Bonferroni correction:")
for pid, rho, p in sorted(all_ps, key=lambda x: x[2]):
    if p < bonferroni_threshold:
        cat = prompt_meta[pid]["category"]
        mu = prompt_meta[pid]["metadata"].get("mu", None)
        mu_str = f"mu={mu:.1f}" if mu is not None else ""
        print(f"  {pid:>6} [{cat:>12}] rho={rho:>7.3f}, p={p:.6f} {mu_str}")

# ============================
# 4. Asymmetry test: is the length effect directional (neg vs pos steering)?
# ============================
print("\n" + "=" * 60)
print("Asymmetry test: negative vs positive steering")
print("=" * 60)

# For prompts with strong effects, compare mean length at -3000 vs +3000
for pid in ["D_08", "D_09", "D_04", "F_01", "F_00", "C_07", "C_08", "B_05", "E_08"]:
    neg3k = [r["response_length"] for r in data["results"]
             if r["prompt_id"] == pid and r["coefficient"] == -3000]
    zero = [r["response_length"] for r in data["results"]
            if r["prompt_id"] == pid and r["coefficient"] == 0]
    pos3k = [r["response_length"] for r in data["results"]
             if r["prompt_id"] == pid and r["coefficient"] == 3000]

    cat = prompt_meta[pid]["category"]
    mu = prompt_meta[pid]["metadata"].get("mu", None)
    mu_str = f"mu={mu:.1f}" if mu is not None else ""

    # Mann-Whitney U test: neg vs zero, pos vs zero
    if neg3k and zero:
        u_neg, p_neg = stats.mannwhitneyu(neg3k, zero, alternative="two-sided")
    else:
        p_neg = 1.0
    if pos3k and zero:
        u_pos, p_pos = stats.mannwhitneyu(pos3k, zero, alternative="two-sided")
    else:
        p_pos = 1.0

    print(f"  {pid} [{cat}] {mu_str}")
    print(f"    neg3k: mean={np.mean(neg3k):.0f} (n={len(neg3k)})")
    print(f"    zero:  mean={np.mean(zero):.0f} (n={len(zero)})")
    print(f"    pos3k: mean={np.mean(pos3k):.0f} (n={len(pos3k)})")
    print(f"    neg vs zero: p={p_neg:.4f}")
    print(f"    pos vs zero: p={p_pos:.4f}")

# ============================
# 5. Count how many D/F prompts show negative vs positive rho
# ============================
print("\n" + "=" * 60)
print("Direction consistency: D_valence and F_affect")
print("=" * 60)

for cat in ["D_valence", "F_affect"]:
    cat_pids = [p["prompt_id"] for p in data["prompts"] if p["category"] == cat]
    neg_rho, pos_rho, null_rho = 0, 0, 0
    for pid in cat_pids:
        coefs, lengths = [], []
        for r in data["results"]:
            if r["prompt_id"] == pid and r["coefficient"] in COHERENT_COEFS:
                coefs.append(r["coefficient"])
                lengths.append(r["response_length"])
        rho, p = stats.spearmanr(coefs, lengths)
        if rho < -0.1:
            neg_rho += 1
        elif rho > 0.1:
            pos_rho += 1
        else:
            null_rho += 1

    print(f"  {cat}: negative_rho={neg_rho}, positive_rho={pos_rho}, null={null_rho}")
    # Binomial test: is the proportion of negative significantly different from 50%?
    n = neg_rho + pos_rho
    if n > 0:
        binom_p = stats.binom_test(neg_rho, n, 0.5)
        print(f"    Binomial test (neg vs pos): p={binom_p:.4f}")
