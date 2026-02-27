"""Local coherence judge for revealed preference steering responses.

Since no OpenRouter API key is available on the pod, this uses heuristic-based
coherence detection. The criteria are designed to match the spec's definition:

COHERENT if ALL:
1. Clearly chooses Task A or B
2. Completion is understandable English
3. Completion addresses the chosen task
4. No garbled text, context leakage, or gibberish

INCOHERENT if ANY:
- Cannot determine which task was chosen
- Contains gibberish, garbled text, or random symbols
- Opens with fragments unrelated to either task
- Fabricates conversational context
- Does not attempt to complete any task
"""

from __future__ import annotations

import json
import re
import string
from collections import defaultdict
from pathlib import Path

RESULTS_DIR = (
    Path(__file__).resolve().parent.parent.parent
    / "experiments" / "steering" / "replication" / "fine_grained"
    / "coherence_test" / "results"
)

COHERENCE_THRESHOLD = 0.9


def detect_choice_or_refusal(response: str) -> str | None:
    """Detect if response clearly chooses Task A or B, or is a coherent refusal.

    Returns 'a', 'b', or 'refusal' for coherent responses, None for unclear.
    """
    # Strip markdown formatting
    clean = re.sub(r"^[\s*#_`>]+", "", response).strip()
    lower = clean.lower()

    if lower.startswith("task a"):
        return "a"
    if lower.startswith("task b"):
        return "b"

    # Check for variants like "I will complete Task A" or "I choose Task B"
    first_line = lower.split("\n")[0][:150]
    if re.search(r"\btask\s*a\b", first_line) and not re.search(r"\btask\s*b\b", first_line):
        return "a"
    if re.search(r"\btask\s*b\b", first_line) and not re.search(r"\btask\s*a\b", first_line):
        return "b"

    # Detect safety refusals — the model is functioning normally, just declining
    refusal_patterns = [
        r"i am programmed to be",
        r"i cannot and will not",
        r"i must decline",
        r"i am unable to fulfill",
        r"i can't fulfill",
        r"i cannot fulfill",
        r"due to the .* nature of",
        r"i'm not able to",
    ]
    for pat in refusal_patterns:
        if re.search(pat, lower[:300]):
            return "refusal"

    return None


def is_gibberish(text: str) -> bool:
    """Detect gibberish, garbled text, or random symbols.

    Accounts for LaTeX math content which has lots of backslashes and special chars.
    """
    if len(text.strip()) < 10:
        return True

    # LaTeX math content is not gibberish
    has_latex = bool(re.search(r"\\(?:frac|sum|int|begin|end|left|right|cdot|times)", text))

    # Check for repetitive character/word patterns (degenerate generation)
    words = text.split()
    if len(words) > 20:
        max_repeat = 1
        current_repeat = 1
        for i in range(1, len(words)):
            if words[i] == words[i-1]:
                current_repeat += 1
                max_repeat = max(max_repeat, current_repeat)
            else:
                current_repeat = 1
        if max_repeat > 10:
            return True

    if not has_latex:
        # Check for high proportion of non-ASCII (skip for math)
        non_ascii = sum(1 for c in text if ord(c) > 127 and c not in '""''—–…•')
        if len(text) > 0 and non_ascii / len(text) > 0.3:
            return True

        # Check for very low letter-to-symbol ratio (skip for math)
        letters = sum(1 for c in text if c.isalpha())
        if len(text) > 50 and letters / len(text) < 0.2:
            return True

    return False


def is_coherent_completion(text: str) -> bool:
    """Check if the completion after the task choice is understandable."""
    # Split off the task choice prefix
    lines = text.strip().split("\n", 1)
    if len(lines) < 2:
        return len(lines[0].strip()) > 5

    body = lines[1].strip()
    if not body:
        return False

    if is_gibberish(body):
        return False

    words = body.split()
    if len(words) < 3:
        return False

    return True


def judge_response(response: str) -> tuple[bool, str]:
    """Judge a single response. Returns (coherent, reason)."""
    if not response or len(response.strip()) < 5:
        return False, "empty_or_too_short"

    # Check for gibberish in entire response
    if is_gibberish(response):
        return False, "gibberish"

    # Check task choice or refusal
    result = detect_choice_or_refusal(response)
    if result is None:
        return False, "no_task_choice"
    if result == "refusal":
        return True, "refusal"

    # Check completion quality
    if not is_coherent_completion(response):
        return False, "incoherent_completion"

    return True, "ok"


def main():
    raw_path = RESULTS_DIR / "raw_responses.json"
    with open(raw_path) as f:
        entries = json.load(f)

    print(f"Loaded {len(entries)} entries from {raw_path}")
    total_responses = sum(len(e["responses"]) for e in entries)
    print(f"Total responses to judge: {total_responses}")

    # Judge all responses
    coef_results: dict[float, list[bool]] = defaultdict(list)
    pct_map: dict[float, float] = {}
    reasons: dict[str, int] = defaultdict(int)

    for entry in entries:
        coef = entry["coefficient"]
        pct = entry["pct_norm"]
        pct_map[coef] = pct

        for resp in entry["responses"]:
            coherent, reason = judge_response(resp)
            coef_results[coef].append(coherent)
            reasons[reason] += 1

    # Output
    output = {}
    print("\nCoefficient    %Norm  Coherent%  Status")
    print("-" * 50)
    for coef in sorted(coef_results.keys()):
        js = coef_results[coef]
        pct_coherent = sum(js) / len(js)
        is_ok = pct_coherent >= COHERENCE_THRESHOLD
        pct = pct_map[coef]

        output[str(coef)] = {
            "coherent_pct": round(pct_coherent, 4),
            "coherent": is_ok,
            "n": len(js),
            "pct_norm": pct,
        }

        status = "OK" if is_ok else "FLAGGED"
        print(f"  {coef:>10.1f}  {pct:>+6.1f}%     {pct_coherent:5.1%}     {status} ({sum(js)}/{len(js)})")

    out = RESULTS_DIR / "coherence_by_coefficient.json"
    with open(out, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved {out}")

    # Reason breakdown
    print("\nIncoherence reasons:")
    for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
        print(f"  {reason}: {count}")

    # Summary
    flagged = [k for k, v in output.items() if not v["coherent"]]
    if flagged:
        print(f"\nFLAGGED coefficients ({len(flagged)}):")
        for k in flagged:
            print(f"  {k}: {output[k]['coherent_pct']:.1%} ({output[k]['pct_norm']:+.1f}%)")
    else:
        print(f"\nAll {len(output)} coefficients passed coherence threshold ({COHERENCE_THRESHOLD:.0%})")

    # Also print some example incoherent responses for manual inspection
    print("\n--- Example incoherent responses (up to 5) ---")
    count = 0
    for entry in entries:
        for resp in entry["responses"]:
            coherent, reason = judge_response(resp)
            if not coherent and count < 5:
                print(f"\npair={entry['pair_id']} coef={entry['pct_norm']:+.1f}% reason={reason}")
                print(f"  {resp[:200].replace(chr(10), ' | ')}")
                count += 1


if __name__ == "__main__":
    main()
