"""Pre-flight render check for the user-context persona steering condition.

Implements the confirmations required by
experiments/reviewer_followups/system_vs_user_persona_steering_spec.md (lines 69-77).
CPU only: loads the Gemma tokenizer, never the model weights.
"""

import json
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv
from transformers import AutoTokenizer

from src.measurement.elicitation.prompt_templates.template import load_templates_from_yaml
from src.measurement.runners.runners import build_revealed_builder
from src.steering.runner import _pair_to_tasks
from src.steering.tokenization import find_pairwise_task_spans

load_dotenv()

MODEL = "google/gemma-3-27b-it"
TEMPLATE_PATH = "src/measurement/elicitation/prompt_templates/data/completion_preference.yaml"
PERSONA_CONFIG = "configs/steering/cross_persona_differential/sadist.yaml"
ACK = "Understood."


def render(tokenizer, messages: list[dict]) -> str:
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def main(pairs_path: str) -> int:
    persona = yaml.safe_load(Path(PERSONA_CONFIG).read_text())["system_prompt"]
    pairs = json.loads(Path(pairs_path).read_text())
    template = load_templates_from_yaml(TEMPLATE_PATH)[0]

    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)

    sys_builder = build_revealed_builder(template, "completion", system_prompt=persona)
    ctx_builder = build_revealed_builder(
        template,
        "completion",
        context_messages=[
            {"role": "user", "content": persona},
            {"role": "assistant", "content": ACK},
        ],
    )

    task_a, task_b = _pair_to_tasks(pairs[0])
    failures = []

    for order_name, (first, second) in [("AB", (task_a, task_b)), ("BA", (task_b, task_a))]:
        sys_msgs = sys_builder.build(first, second).messages
        ctx_msgs = ctx_builder.build(first, second).messages

        # 1. exactly three turns: user persona, assistant ack, user task choice
        roles = [m["role"] for m in ctx_msgs]
        if roles != ["user", "assistant", "user"]:
            failures.append(f"[{order_name}] expected 3 turns user/assistant/user, got {roles}")

        # 2. P and C(A,B) byte-for-byte the intended text
        if ctx_msgs[0]["content"] != persona:
            failures.append(f"[{order_name}] persona text not byte-exact in turn 0")
        if ctx_msgs[1]["content"] != ACK:
            failures.append(f"[{order_name}] ack is {ctx_msgs[1]['content']!r}, expected {ACK!r}")
        task_choice = ctx_msgs[-1]["content"]
        if task_choice != sys_msgs[-1]["content"]:
            failures.append(f"[{order_name}] task-choice message differs between conditions")

        sys_text = render(tokenizer, sys_msgs)
        ctx_text = render(tokenizer, ctx_msgs)
        sys_ids = tokenizer(sys_text, add_special_tokens=True).input_ids
        ctx_ids = tokenizer(ctx_text, add_special_tokens=True).input_ids

        # 3. the new rendered token sequence differs from the existing condition
        if sys_ids == ctx_ids:
            failures.append(
                f"[{order_name}] token sequences are IDENTICAL — the new condition "
                f"collapses to the existing one, so the run would measure only noise"
            )

        # 4. spans locate the same task text in both conditions
        sys_spans = find_pairwise_task_spans(tokenizer, sys_text, first.prompt, second.prompt)
        ctx_spans = find_pairwise_task_spans(tokenizer, ctx_text, first.prompt, second.prompt)
        for label, span, text in [("system", sys_spans, sys_text), ("context", ctx_spans, ctx_text)]:
            ids = tokenizer(text, add_special_tokens=True).input_ids
            for which, (s, e), expected in [
                ("first", span[0], first.prompt),
                ("second", span[1], second.prompt),
            ]:
                decoded = tokenizer.decode(ids[s:e]).strip()
                if decoded != expected.strip():
                    failures.append(
                        f"[{order_name}/{label}] {which} span decodes to {decoded!r}, "
                        f"expected {expected.strip()!r}"
                    )

        print(f"--- {order_name} ---")
        print(f"  turns (context cond): {roles}")
        print(f"  token len: system={len(sys_ids)} context={len(ctx_ids)} "
              f"delta={len(ctx_ids) - len(sys_ids)}")
        print(f"  token ids differ: {sys_ids != ctx_ids}")
        print(f"  spans system={sys_spans} context={ctx_spans}")

    print()
    if failures:
        print(f"FAILED ({len(failures)}):")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("All pre-flight checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
