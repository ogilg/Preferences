from src.steering.tokenization import find_text_span, find_pairwise_task_spans
from src.steering.calibration import suggest_coefficient_range


def __getattr__(name: str):
    """Lazy import for steering.client to avoid circular import with models."""
    if name == "SteeredHFClient":
        from src.steering.client import SteeredHFClient
        return SteeredHFClient
    if name == "create_steered_client":
        from src.steering.client import create_steered_client
        return create_steered_client
    raise AttributeError(f"module 'src.steering' has no attribute {name!r}")


__all__ = [
    "SteeredHFClient",
    "create_steered_client",
    "find_text_span",
    "find_pairwise_task_spans",
    "suggest_coefficient_range",
]
