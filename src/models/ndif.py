"""NDIF (National Deep Inference Fabric) model implementation using nnsight."""

from nnsight import LanguageModel

from src.types import Message

from dotenv import load_dotenv
load_dotenv()


class NDIFModel:
    """Model via NDIF using nnsight."""

    def __init__(
        self,
        model_name: str,
        max_new_tokens: int = 256,
    ):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens

        self.model = LanguageModel(model_name, device_map="auto")

    def _format_messages(self, messages: list[Message]) -> str:
        """Format messages using the tokenizer's chat template."""
        chat_messages = [
            {"role": msg["role"], "content": msg["content"]} for msg in messages
        ]
        return self.model.tokenizer.apply_chat_template(
            chat_messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    def generate(
        self,
        messages: list[Message],
        temperature: float = 1.0,
    ) -> str:
        """Generate a response for the given messages."""
        prompt = self._format_messages(messages)

        with self.model.generate(
            prompt,
            max_new_tokens=self.max_new_tokens,
            temperature=temperature,
            remote=True,
        ) as generator:
            output = generator.output

        generated_text = self.model.tokenizer.decode(
            output[0][len(self.model.tokenizer.encode(prompt)) :],
            skip_special_tokens=True,
        )
        return generated_text.strip()

    def get_logprobs(
        self,
        messages: list[Message],
        max_tokens: int = 1,
    ) -> dict[str, float]:
        """Get log probabilities for next tokens."""
        import torch

        prompt = self._format_messages(messages)

        with self.model.trace(prompt, remote=True) as _:
            logits = self.model.lm_head.output[0, -1, :].save()

        log_probs = torch.nn.functional.log_softmax(logits.value, dim=-1)
        top_values, top_indices = torch.topk(log_probs, k=max_tokens)

        result = {}
        for i in range(max_tokens):
            token = self.model.tokenizer.decode([top_indices[i].item()])
            result[token] = top_values[i].item()

        return result
