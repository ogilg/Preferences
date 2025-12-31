"""Visualization tools for Thurstonian model results."""

from .plots import plot_utility_ranking

__all__ = [
    "plot_utility_ranking",
    "RunBrowser",
]


def __getattr__(name: str):
    """Lazy import for optional dependencies."""
    if name == "RunBrowser":
        from .widgets import RunBrowser
        return RunBrowser
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
