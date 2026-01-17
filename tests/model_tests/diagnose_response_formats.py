"""Test different response formats with Qwen 2.5."""
from pathlib import Path
from datetime import datetime

from src.running_measurements.utils.experiment_utils import setup_experiment
from src.preference_measurement import RATING_FORMATS, StatedScoreMeasurer, measure_stated
from src.prompt_templates import PreTaskStatedPromptBuilder


def parse_scale_from_template(template):
    scale_str = template.tags_dict["scale"]
    min_str, max_str = scale_str.split("-")
    return int(min_str), int(max_str)


def test_response_format(model: str, response_format: str, n_tasks: int = 5):
    """Test a specific response format with a model."""

    output_dir = Path("results/qwen_diagnostic")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{model.replace('/', '_')}_{response_format}_test.txt"

    # Create a temporary config
    config_path = Path(f"/tmp/test_{response_format}.yaml")
    config_content = f"""preference_mode: stated
model: {model}
temperature: 1.0
n_tasks: {n_tasks}
task_origins:
  - wildchat
templates: src/preferences/templates/data/stated_v1.yaml
n_samples: 1
template_sampling: lhs
n_template_samples: 1
lhs_seed: 42
response_formats:
  - {response_format}
generation_seeds:
  - 0
"""
    config_path.write_text(config_content)

    ctx = setup_experiment(config_path, expected_mode="stated")
    config = ctx.config
    client = ctx.client

    template = ctx.templates[0]
    tasks = ctx.tasks[:n_tasks]

    with open(output_file, "w") as f:
        f.write(f"="*80 + "\n")
        f.write(f"RESPONSE FORMAT TEST: {response_format}\n")
        f.write(f"Model: {config.model}\n")
        f.write(f"Time: {datetime.now().isoformat()}\n")
        f.write(f"Template: {template.name}\n")
        f.write(f"Tasks: {len(tasks)}\n")
        f.write(f"="*80 + "\n\n")

        # Setup builder and response format
        scale_info = parse_scale_from_template(template)
        scale_min, scale_max = scale_info
        response_format_obj = RATING_FORMATS[response_format](scale_min, scale_max)

        builder = PreTaskStatedPromptBuilder(
            measurer=StatedScoreMeasurer(),
            response_format=response_format_obj,
            template=template,
        )

        print(f"\n{'='*80}")
        print(f"Testing {config.model} with {response_format} format")
        print(f"Running batch of {len(tasks)} tasks")
        print(f"{'='*80}\n")

        # Run the batch measurement
        batch = measure_stated(
            client=client,
            tasks=tasks,
            builder=builder,
            temperature=config.temperature,
            max_concurrent=ctx.max_concurrent,
            seed=0,
        )

        # Write results
        f.write(f"RESULTS:\n")
        f.write(f"  Successes: {len(batch.successes)}/{len(tasks)}\n")
        f.write(f"  Failures: {len(batch.failures)}/{len(tasks)}\n")
        f.write(f"  Success rate: {len(batch.successes)/len(tasks)*100:.1f}%\n\n")

        # Sample successful responses
        f.write(f"\n{'='*80}\n")
        f.write(f"SAMPLE OUTPUTS:\n")
        f.write(f"{'='*80}\n\n")

        for i, task in enumerate(tasks[:3]):
            f.write(f"\n--- Sample #{i+1} ---\n")

            prompt_obj = builder.build(task)
            messages = prompt_obj.messages
            prompt_text = messages[0]["content"]

            f.write(f"Task ID: {task.origin}/{task.id}\n\n")
            f.write(f"FULL PROMPT:\n")
            f.write(f"{'-'*40}\n")
            f.write(f"{prompt_text}\n")
            f.write(f"{'-'*40}\n\n")

            # Make request
            try:
                response = client.generate(messages, temperature=config.temperature, tools=prompt_obj.response_format.tools)
                f.write(f"RAW RESPONSE: {repr(response)}\n\n")
                f.write(f"DISPLAYED RESPONSE:\n")
                f.write(f"{'-'*40}\n")
                f.write(f"{response}\n")
                f.write(f"{'-'*40}\n\n")

                # Try to parse
                try:
                    parsed = prompt_obj.measurer.parse(response, prompt_obj)
                    f.write(f"✓ Parsed successfully: {parsed.result.score}\n")
                except Exception as e:
                    f.write(f"✗ Parse failed: {e}\n")

            except Exception as e:
                f.write(f"✗ Request failed: {e}\n")

            f.write("\n")

        # Write failures
        if batch.failures:
            f.write(f"\n{'='*80}\n")
            f.write(f"FAILURES ({len(batch.failures)}):\n")
            f.write(f"{'='*80}\n\n")
            for i, (prompt, error) in enumerate(batch.failures[:5]):
                f.write(f"--- Failure #{i+1} ---\n")
                f.write(f"Error: {error[:200]}\n\n")

        print(f"✓ Results written to {output_file}")
        print(f"  Successes: {len(batch.successes)}/{len(tasks)} ({len(batch.successes)/len(tasks)*100:.1f}%)")
        print(f"  Failures: {len(batch.failures)}/{len(tasks)}")


def main():
    """Test all response formats with Qwen 2.5."""
    model = "qwen/qwen-2.5-7b-instruct"
    formats = ["regex", "tool_use", "xml"]

    for fmt in formats:
        try:
            test_response_format(model, fmt, n_tasks=5)
        except Exception as e:
            print(f"\n✗ Error testing {fmt}: {e}\n")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
