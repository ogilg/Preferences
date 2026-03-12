"""Quick check: Cohen's d at first and last token across layers and probes."""
import json
import numpy as np

with open("experiments/truth_probes/error_prefill/per_token_scoring/scored_tokens.json") as f:
    results = json.load(f)

correct = [r for r in results if r["answer_condition"] == "correct"]
incorrect = [r for r in results if r["answer_condition"] == "incorrect"]

LAYERS = ["L25", "L32", "L39", "L46", "L53"]
PROBES = ["tb-2", "tb-5"]

def cohens_d(a, b):
    a, b = np.array(a), np.array(b)
    pooled_std = np.sqrt((np.std(a)**2 + np.std(b)**2) / 2)
    return (np.mean(a) - np.mean(b)) / pooled_std if pooled_std > 0 else 0

print("=== Cohen's d at LAST token (correct - incorrect) ===")
for probe in PROBES:
    print(f"\n{probe}:")
    for layer in LAYERS:
        c_last = [r["scores"][probe][layer][-1] for r in correct]
        i_last = [r["scores"][probe][layer][-1] for r in incorrect]
        d = cohens_d(c_last, i_last)
        print(f"  {layer}: d={d:.3f} (correct mean={np.mean(c_last):.3f}, incorrect mean={np.mean(i_last):.3f})")

print("\n=== Cohen's d at FIRST token ===")
for probe in PROBES:
    print(f"\n{probe}:")
    for layer in LAYERS:
        c_first = [r["scores"][probe][layer][0] for r in correct]
        i_first = [r["scores"][probe][layer][0] for r in incorrect]
        d = cohens_d(c_first, i_first)
        print(f"  {layer}: d={d:.3f} (correct mean={np.mean(c_first):.3f}, incorrect mean={np.mean(i_first):.3f})")

print("\n=== Mean score across ALL tokens ===")
for probe in PROBES:
    print(f"\n{probe}:")
    for layer in LAYERS:
        c_means = [np.mean(r["scores"][probe][layer]) for r in correct]
        i_means = [np.mean(r["scores"][probe][layer]) for r in incorrect]
        d = cohens_d(c_means, i_means)
        print(f"  {layer}: d={d:.3f}")

# Token count stats
c_lens = [r["n_tokens"] for r in correct]
i_lens = [r["n_tokens"] for r in incorrect]
print(f"\nToken counts: correct mean={np.mean(c_lens):.1f} (range {min(c_lens)}-{max(c_lens)}), incorrect mean={np.mean(i_lens):.1f} (range {min(i_lens)}-{max(i_lens)})")
