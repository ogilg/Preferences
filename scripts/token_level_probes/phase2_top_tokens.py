"""Phase 2: Analyze which non-critical tokens have highest/lowest probe scores.

Examines token type frequencies at extreme scores, semantic categories,
positional effects, and critical vs non-critical score comparisons.
"""

import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DATA_PATH = Path("experiments/token_level_probes/scoring_results.json")
NPZ_PATH = Path("experiments/token_level_probes/all_token_scores.npz")
ASSETS_DIR = Path("experiments/token_level_probes/assets")

PROBE = "task_mean_L39"

SPECIAL_TOKENS = {"<bos>", "<start_of_turn>", "<end_of_turn>", "\n"}

DOMAIN_CONDITIONS = {
    "truth": ["true", "false", "nonsense"],
    "harm": ["harmful", "benign", "nonsense"],
    "politics": ["left", "right", "nonsense"],
}

FUNCTION_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "shall",
    "should", "may", "might", "can", "could", "must", "to", "of", "in",
    "for", "on", "with", "at", "by", "from", "as", "into", "through",
    "during", "before", "after", "above", "below", "between", "under",
    "and", "but", "or", "nor", "not", "so", "yet", "both", "either",
    "neither", "each", "every", "all", "any", "few", "more", "most",
    "other", "some", "such", "no", "only", "own", "same", "than",
    "too", "very", "just", "about", "up", "out", "if", "then", "that",
    "this", "these", "those", "it", "its", "he", "she", "they", "them",
    "his", "her", "their", "my", "your", "our", "who", "which", "what",
    "where", "when", "how", "there", "here", "also", "while",
}


def load_data() -> tuple[list[dict], dict[str, np.ndarray]]:
    with open(DATA_PATH) as f:
        items = json.load(f)["items"]
    scores = dict(np.load(NPZ_PATH))
    return items, scores


def get_noncritical_ranked_tokens(
    item: dict,
    scores_arr: np.ndarray,
    top_k: int = 5,
) -> tuple[list[tuple[str, float, int]], list[tuple[str, float, int]]]:
    """Return top-k and bottom-k non-critical, non-special tokens by score.

    Returns lists of (token_text, score, position).
    """
    tokens = item["tokens"]
    critical_set = set(item["critical_token_indices"])
    n = min(len(tokens), len(scores_arr))

    candidates = []
    for i in range(n):
        tok = tokens[i]
        if i in critical_set:
            continue
        if tok in SPECIAL_TOKENS:
            continue
        candidates.append((tok, float(scores_arr[i]), i))

    candidates.sort(key=lambda x: x[1], reverse=True)
    top = candidates[:top_k]
    bottom = candidates[-top_k:]
    return top, bottom


def classify_token(tok: str) -> str:
    """Classify a token into a rough semantic category."""
    stripped = tok.strip()
    if not stripped:
        return "other"
    if all(c in ".,!?;:'\"-()[]{}/" for c in stripped):
        return "punctuation"
    word = stripped.lower()
    if word in FUNCTION_WORDS:
        return "function_word"
    if len(stripped) > 3:
        return "content_word"
    # Short words not in function word list
    if stripped.isalpha():
        return "function_word"
    return "other"


# ── Analysis 1: Token type frequency at extreme scores ──


def analysis_token_frequency(items: list[dict], scores: dict[str, np.ndarray]) -> None:
    print("=" * 90)
    print("ANALYSIS 1: Token type frequency at extreme scores (non-critical, non-special)")
    print("=" * 90)

    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for item in items:
        grouped[(item["domain"], item["condition"])].append(item)

    for domain in ["truth", "harm", "politics"]:
        for condition in DOMAIN_CONDITIONS[domain]:
            group_items = grouped[(domain, condition)]
            top_counter: Counter = Counter()
            bottom_counter: Counter = Counter()

            for item in group_items:
                key = f"{item['id']}__{PROBE}"
                if key not in scores:
                    continue
                top, bottom = get_noncritical_ranked_tokens(item, scores[key])
                for tok, _, _ in top:
                    top_counter[tok] += 1
                for tok, _, _ in bottom:
                    bottom_counter[tok] += 1

            print(f"\n--- {domain} / {condition} (n={len(group_items)}) ---")
            print(f"  TOP-5 scoring tokens (most frequent):")
            for tok, count in top_counter.most_common(20):
                print(f"    {repr(tok):>20s}  count={count}")
            print(f"  BOTTOM-5 scoring tokens (most frequent):")
            for tok, count in bottom_counter.most_common(20):
                print(f"    {repr(tok):>20s}  count={count}")


# ── Analysis 2: Semantic category analysis ──


def analysis_semantic_categories(items: list[dict], scores: dict[str, np.ndarray]) -> None:
    print("\n" + "=" * 90)
    print("ANALYSIS 2: Semantic category proportions in top-5 vs bottom-5")
    print("=" * 90)

    for domain in ["truth", "harm", "politics"]:
        top_categories: Counter = Counter()
        bottom_categories: Counter = Counter()
        top_total = 0
        bottom_total = 0

        for item in items:
            if item["domain"] != domain:
                continue
            key = f"{item['id']}__{PROBE}"
            if key not in scores:
                continue
            top, bottom = get_noncritical_ranked_tokens(item, scores[key])
            for tok, _, _ in top:
                cat = classify_token(tok)
                top_categories[cat] += 1
                top_total += 1
            for tok, _, _ in bottom:
                cat = classify_token(tok)
                bottom_categories[cat] += 1
                bottom_total += 1

        print(f"\n--- {domain} ---")
        print(f"  {'Category':<20s} {'Top-5 %':>10s} {'Bottom-5 %':>12s} {'Top-5 n':>10s} {'Bottom-5 n':>12s}")
        print(f"  {'-'*64}")
        for cat in ["content_word", "function_word", "punctuation", "other"]:
            top_pct = 100 * top_categories[cat] / top_total if top_total else 0
            bot_pct = 100 * bottom_categories[cat] / bottom_total if bottom_total else 0
            print(
                f"  {cat:<20s} {top_pct:>9.1f}% {bot_pct:>11.1f}% "
                f"{top_categories[cat]:>10d} {bottom_categories[cat]:>12d}"
            )


# ── Analysis 3: Position analysis ──


def analysis_position(items: list[dict], scores: dict[str, np.ndarray]) -> None:
    print("\n" + "=" * 90)
    print("ANALYSIS 3: Score vs relative position")
    print("=" * 90)

    n_bins = 50
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        f"Probe score vs relative position ({PROBE})",
        fontsize=14,
        fontweight="bold",
    )

    for ax, domain in zip(axes, ["truth", "harm", "politics"]):
        # Collect (relative_position, score) for all non-special tokens
        bin_scores: list[list[float]] = [[] for _ in range(n_bins)]

        for item in items:
            if item["domain"] != domain:
                continue
            key = f"{item['id']}__{PROBE}"
            if key not in scores:
                continue
            tokens = item["tokens"]
            arr = scores[key]
            n = min(len(tokens), len(arr))
            if n < 2:
                continue

            for i in range(n):
                if tokens[i] in SPECIAL_TOKENS:
                    continue
                rel_pos = i / (n - 1)
                bin_idx = min(int(rel_pos * n_bins), n_bins - 1)
                bin_scores[bin_idx].append(float(arr[i]))

        bin_centers = [(i + 0.5) / n_bins for i in range(n_bins)]
        bin_means = []
        bin_sems = []
        valid_centers = []
        for i in range(n_bins):
            if bin_scores[i]:
                m = np.mean(bin_scores[i])
                s = np.std(bin_scores[i]) / np.sqrt(len(bin_scores[i]))
                bin_means.append(m)
                bin_sems.append(s)
                valid_centers.append(bin_centers[i])

        ax.plot(valid_centers, bin_means, color="#1f77b4", linewidth=1.5)
        ax.fill_between(
            valid_centers,
            [m - s for m, s in zip(bin_means, bin_sems)],
            [m + s for m, s in zip(bin_means, bin_sems)],
            alpha=0.2,
            color="#1f77b4",
        )
        ax.axhline(0, color="black", linewidth=0.5, alpha=0.3)
        ax.set_xlabel("Relative position (0=start, 1=end)")
        ax.set_ylabel("Mean probe score")
        ax.set_title(f"{domain.title()}")

        # Print summary
        if valid_centers:
            early = [m for c, m in zip(valid_centers, bin_means) if c < 0.2]
            late = [m for c, m in zip(valid_centers, bin_means) if c > 0.8]
            print(f"  {domain}: early mean={np.mean(early):.3f}, late mean={np.mean(late):.3f}")

    plt.tight_layout()
    fname = "plot_031426_score_vs_position.png"
    fig.savefig(ASSETS_DIR / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {fname}")


# ── Analysis 4: Critical vs non-critical comparison ──


def analysis_critical_vs_noncritical(items: list[dict], scores: dict[str, np.ndarray]) -> None:
    print("\n" + "=" * 90)
    print("ANALYSIS 4: Critical vs non-critical token scores")
    print("=" * 90)

    FLANK = 5

    print(
        f"\n  {'Domain':<12s} {'Condition':<12s} {'n':>5s} "
        f"{'Critical':>10s} {'Non-crit':>10s} {'Pre-flank':>10s} {'Post-flank':>10s} "
        f"{'Crit-NonCr':>10s}"
    )
    print(f"  {'-'*80}")

    for domain in ["truth", "harm", "politics"]:
        for condition in DOMAIN_CONDITIONS[domain]:
            critical_means = []
            noncritical_means = []
            preflank_means = []
            postflank_means = []

            for item in items:
                if item["domain"] != domain or item["condition"] != condition:
                    continue
                key = f"{item['id']}__{PROBE}"
                if key not in scores:
                    continue

                tokens = item["tokens"]
                arr = scores[key]
                n = min(len(tokens), len(arr))
                crit_indices = set(item["critical_token_indices"])
                crit_indices_list = sorted(item["critical_token_indices"])

                if not crit_indices_list:
                    continue

                # Critical span scores
                crit_scores = [float(arr[i]) for i in crit_indices_list if i < n]
                if crit_scores:
                    critical_means.append(np.mean(crit_scores))

                # Non-critical scores (excluding special tokens)
                nc_scores = [
                    float(arr[i]) for i in range(n)
                    if i not in crit_indices and tokens[i] not in SPECIAL_TOKENS
                ]
                if nc_scores:
                    noncritical_means.append(np.mean(nc_scores))

                # Pre-flank: FLANK tokens before critical span
                crit_start = crit_indices_list[0]
                pre_start = max(0, crit_start - FLANK)
                pre_scores = [
                    float(arr[i]) for i in range(pre_start, crit_start)
                    if i < n and i not in crit_indices and tokens[i] not in SPECIAL_TOKENS
                ]
                if pre_scores:
                    preflank_means.append(np.mean(pre_scores))

                # Post-flank: FLANK tokens after critical span
                crit_end = crit_indices_list[-1] + 1
                post_end = min(n, crit_end + FLANK)
                post_scores = [
                    float(arr[i]) for i in range(crit_end, post_end)
                    if i < n and i not in crit_indices and tokens[i] not in SPECIAL_TOKENS
                ]
                if post_scores:
                    postflank_means.append(np.mean(post_scores))

            n_items = len(critical_means)
            if n_items == 0:
                continue

            cm = np.mean(critical_means)
            ncm = np.mean(noncritical_means) if noncritical_means else float("nan")
            pfm = np.mean(preflank_means) if preflank_means else float("nan")
            pofm = np.mean(postflank_means) if postflank_means else float("nan")
            diff = cm - ncm if not np.isnan(ncm) else float("nan")

            print(
                f"  {domain:<12s} {condition:<12s} {n_items:>5d} "
                f"{cm:>10.3f} {ncm:>10.3f} {pfm:>10.3f} {pofm:>10.3f} "
                f"{diff:>+10.3f}"
            )


def main() -> None:
    items, scores = load_data()
    print(f"Loaded {len(items)} items, {len(scores)} score arrays\n")

    analysis_token_frequency(items, scores)
    analysis_semantic_categories(items, scores)
    analysis_position(items, scores)
    analysis_critical_vs_noncritical(items, scores)

    print("\nDone.")


if __name__ == "__main__":
    main()
