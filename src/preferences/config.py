"""Configuration for dataset-wide preference measurement."""

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class PairingStrategy(str, Enum):
    """Strategy for generating task pairs for binary comparisons."""

    ALL_PAIRS = "all_pairs"  # n(n-1)/2 unique pairs
    RANDOM_PAIRS = "random_pairs"  # Random sample of pairs


class DatasetMeasurementConfig(BaseModel, frozen=True):
    """Configuration for dataset-wide preference measurement."""

    temperature: float = Field(
        default=1.0,
        ge=0.0,
        le=2.0,
        description="Sampling temperature for model generation",
    )
    num_samples: int = Field(
        default=1,
        ge=1,
        description="Number of times to repeat each measurement",
    )
    measurement_types: frozenset[Literal["rating", "binary"]] = Field(
        default_factory=lambda: frozenset({"rating", "binary"}),
        description="Which measurement types to run",
    )
    pairing_strategy: PairingStrategy = Field(
        default=PairingStrategy.ALL_PAIRS,
        description="How to generate task pairs for binary comparisons",
    )
    max_pairs: int | None = Field(
        default=None,
        description="Maximum number of pairs when using RANDOM_PAIRS strategy",
    )
    seed: int | None = Field(
        default=None,
        description="Random seed for reproducibility",
    )
