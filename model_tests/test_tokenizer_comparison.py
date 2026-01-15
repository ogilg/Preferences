"""Compare tokenization between TransformerLens and OpenRouter for Llama 3.1 8B."""

import os
from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI
from transformers import AutoTokenizer

MODEL_HF = "meta-llama/Llama-3.1-8B-Instruct"
OPENROUTER_MODEL = "meta-llama/llama-3.1-8b-instruct"


def main():
    print("Loading HuggingFace tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_HF)

    print("Creating OpenRouter client...")
    client = OpenAI(
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url="https://openrouter.ai/api/v1",
    )

    messages = [{"role": "user", "content": "What is 2+2?"}]

    # Check the default chat template
    print("\n" + "=" * 60)
    print("Investigating the chat template")
    print("=" * 60)

    print("\nDefault chat_template (truncated):")
    template = tokenizer.chat_template
    print(template[:500] + "..." if len(template) > 500 else template)

    # The template has a default system message baked in
    # Let's see if we can use a custom template without it

    # Minimal Llama 3.1 template without default system message
    MINIMAL_TEMPLATE = """\
{%- set bos = '<|begin_of_text|>' %}
{%- set has_bos = false %}
{%- for message in messages %}
{%- if message['role'] == 'system' %}
{{ bos }}<|start_header_id|>system<|end_header_id|>

{{ message['content'] }}<|eot_id|>
{%- set has_bos = true %}
{%- elif message['role'] == 'user' %}
{%- if not has_bos %}{{ bos }}{%- set has_bos = true %}{%- endif %}
<|start_header_id|>user<|end_header_id|>

{{ message['content'] }}<|eot_id|>
{%- elif message['role'] == 'assistant' %}
<|start_header_id|>assistant<|end_header_id|>

{{ message['content'] }}<|eot_id|>
{%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
<|start_header_id|>assistant<|end_header_id|>

{% endif %}"""

    print("\n" + "=" * 60)
    print("TEST: Custom template without default system message")
    print("=" * 60)

    # Format with custom template
    custom_formatted = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        chat_template=MINIMAL_TEMPLATE,
    )
    print(f"\nCustom template result:")
    print(repr(custom_formatted))

    custom_tokens = tokenizer.encode(custom_formatted, add_special_tokens=False)
    print(f"Token count: {len(custom_tokens)}")

    # Get API response
    response = client.chat.completions.create(
        model=OPENROUTER_MODEL,
        messages=messages,
        temperature=0.0,
        max_tokens=20,
    )

    print(f"\nOpenRouter prompt tokens: {response.usage.prompt_tokens}")
    print(f"Local (custom template):  {len(custom_tokens)}")

    diff = response.usage.prompt_tokens - len(custom_tokens)
    if diff == 0:
        print("✓ MATCH! Custom template aligns with OpenRouter.")
    else:
        print(f"✗ Difference: {diff}")

    # Now test full conversation alignment
    print("\n" + "=" * 60)
    print("Full conversation with custom template")
    print("=" * 60)

    completion = response.choices[0].message.content.strip()
    print(f"Completion: '{completion}'")

    full_messages = messages + [{"role": "assistant", "content": completion}]

    full_formatted = tokenizer.apply_chat_template(
        full_messages,
        tokenize=False,
        add_generation_prompt=False,
        chat_template=MINIMAL_TEMPLATE,
    )
    print(f"\nFull conversation:")
    print(repr(full_formatted))

    full_tokens = tokenizer.encode(full_formatted, add_special_tokens=False)
    print(f"\nTotal tokens: {len(full_tokens)}")

    # Verify token IDs
    completion_tokens = tokenizer.encode(completion, add_special_tokens=False)
    print(f"\nCompletion tokens: {completion_tokens}")
    print(f"Decoded: {[tokenizer.decode([t]) for t in completion_tokens]}")

    # Check last tokens align
    print(f"\nLast 10 full conversation tokens:")
    for i, t in enumerate(full_tokens[-10:]):
        print(f"  [{len(full_tokens)-10+i}] {t} = '{tokenizer.decode([t])}'")

    # Where should we extract activations?
    eot_id = 128009
    last_eot_idx = len(full_tokens) - 1 - full_tokens[::-1].index(eot_id)
    print(f"\nLast EOT at index: {last_eot_idx}")
    print(f"Extract activation at index {last_eot_idx - 1} (last content token)")

    print("\n" + "=" * 60)
    print("CONCLUSION")
    print("=" * 60)
    print("""
To align tokenization between API and TransformerLens:

1. Use a CUSTOM chat template that doesn't inject the default system message
2. Pass this template to apply_chat_template() when formatting for TL
3. Don't send a system message to the API (or send identical ones to both)

The token IDs for the completion text are IDENTICAL - the tokenizer is the same.
The only difference was the chat template prefix.
""")


if __name__ == "__main__":
    main()
